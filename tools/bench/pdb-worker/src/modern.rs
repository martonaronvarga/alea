//! Explicit v3 workflows. Version-two jobs retain their declared HMC warmup.
use super::*;
use alea_mcmc::adapt::{FisherNutsWarmup, NutsWarmup, NutsWarmupObservation, NutsWarmupReport};

/// Uncensored one-step energies with identical starting phase for both policies.
/// This isolates the energy cutoff; the end-to-end warmup ablation changes the
/// whole controller and must not be interpreted as this single-factor comparison.
pub(super) fn search_probes(job: &Job, t: &Target) -> Result<(), Any> {
    use alea_core::target::PointState;
    use alea_mcmc::hamiltonian::{PhaseWorkspace, SignedStep};
    let metric = IdentityMetric::new(t.dimension());
    let mut rows = Vec::new();
    for (chain, q) in job.initials.iter().enumerate() {
        let point = PointState::new(t, owned(q))?;
        let mut workspace = PhaseWorkspace::new(&point);
        for h in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1] {
            let before = t.counts();
            let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
            let result = (|| -> Result<Value, Any> {
                let mut phase = workspace.start(&point, &metric, &mut rng)?;
                let initial = phase.energy()?;
                let mut phase = phase.step(SignedStep::new(h).ok_or("invalid probe step")?)?;
                let delta = phase.energy()? - initial;
                if !delta.is_finite() {
                    return Err("nonfinite probe energy difference".into());
                }
                let probability = (-delta).min(0.).exp();
                Ok(
                    json!({"delta_energy":delta,"one_sided_probability":if delta>1000. {0.} else {probability},
                    "symmetric_probability":if delta.abs()>1000. {0.} else {probability}}),
                )
            })();
            rows.push(json!({"chain":chain,"seed":job.seeds[chain],"initial":q,"step":h,
                "outcome":result.as_ref().ok(),"failure":result.as_ref().err().map(ToString::to_string),
                "work_before":before,"work_after":t.counts()}));
        }
    }
    write_json(
        &job.directory.join("search-probes.json"),
        &json!({"cutoff":1000.,"rows":rows}),
    )
}

fn report(r: &NutsWarmupReport) -> Value {
    json!({"iterations":r.iterations,"moved":r.moved,"divergences":r.divergences,
        "depth_saturations":r.depth_saturations,"metric_updates":r.metric_updates,
        "search_probes":r.search_probes,"warmup_integration_steps":r.integration_attempts,
        "bounded_step_updates":r.bounded_step_updates,"step":r.step_size.value(),
        "initial_log_density":r.initial_log_density,"final_log_density":r.final_log_density,
        "minimum_log_density":r.minimum_log_density,"maximum_log_density":r.maximum_log_density})
}
fn retain_nuts<T: LogDensityGradient + ?Sized, M: EuclideanMetric, I: Integrator>(
    mut s: Nuts<'_, T, M, I>,
    rng: &mut SmallRng,
    n: usize,
) -> Result<Vec<(Vec<f64>, Info)>, Any> {
    collect(n, || {
        let i = s.step(rng)?;
        Ok((
            s.point().position().to_vec(),
            Info {
                steps: i.integration_steps,
                divergent: matches!(
                    i.termination,
                    alea_mcmc::nuts::NutsTermination::Divergence(_)
                ),
                saturated: i.termination == alea_mcmc::nuts::NutsTermination::MaxDepth,
                acceptance: Some(i.acceptance_probability),
                energy: i.initial_energy,
                error: i.max_energy_error,
            },
        ))
    })
}
pub(super) fn run_chain(job: &Job, t: &Target, chain: usize, out: &mut Output) -> Result<(), Any> {
    match job.method.as_str() {
        "atlas_full" => return atlas_chain(job, t, chain, out),
        "mams" | "mclmc" => return isokinetic_chain(job, t, chain, out),
        "nuts_transport" => return transport_chain(job, t, chain, out),
        "hmc_cov" | "hmc_leapfrog" | "ghmc" | "drghmc" => {
            return static_chain(job, t, chain, out, alea_mcmc::integrator::Leapfrog);
        }
        "hmc_omf" => return static_chain(job, t, chain, out, TwoStage::OMF),
        "hmc_bcss" => return static_chain(job, t, chain, out, TwoStage::BCSS),
        method if method.starts_with("walnuts") => return walnuts_chain(job, t, chain, out),
        _ => {}
    }
    let start = Instant::now();
    let mut before = t.counts();
    let mut warmup_start = start;
    let mut initialization_seconds = None;
    let mut initialization_work_before = json!(null);
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let nuts = alea_mcmc::NutsOptions::new(0.1, job.depth)?;
    let nuts = if matches!(
        job.method.as_str(),
        "nuts_grad_diag_jitter" | "nuts_grad_diag_position"
    ) {
        nuts.with_step_size_jitter(0.1)?
            .with_position_turning(job.method == "nuts_grad_diag_position")
    } else {
        nuts
    };
    let mut initial = job.initials[chain].clone();
    let mut initialization = json!(null);
    if matches!(
        job.method.as_str(),
        "nuts_pathfinder_raw" | "nuts_pathfinder_psis"
    ) {
        use alea_mcmc::adapt::ImportanceResampling;
        let initialization_start = Instant::now();
        initialization_work_before = before.clone();
        let starts: Vec<_> = (0..4)
            .flat_map(|_| {
                initial
                    .iter()
                    .map(|q| q + rng.random_range(-0.5..0.5))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mode = if job.method.ends_with("psis") {
            ImportanceResampling::PsisPathwise
        } else {
            ImportanceResampling::RawMixture
        };
        let initialized = multi_pathfinder(
            t,
            &starts,
            MultiPathfinderOptions::new(4, 40)?
                .with_elbo_draws(8)?
                .with_draws_per_path(64)?
                .with_resampling(mode),
            &mut rng,
        )?;
        let r = initialized.report;
        initialization = json!({"requested_paths":r.requested_paths,"successful_paths":r.paths,"optimization_steps":r.optimization_steps,
            "target_evaluations":r.target_evaluations,"proposal_draws":r.proposal_draws,"invalid_proposal_draws":r.invalid_proposal_draws,
            "importance_ess":r.importance_ess,"pareto_shape":r.pareto_shape,"psis_tail_length":r.psis_tail_length,
            "psis_status":format!("{:?}",r.psis_status),"resampling":format!("{:?}",mode),"paths":initialized.paths.iter().map(|p|
                json!({"path":p.path,"status":format!("{:?}",p.status),"optimization_steps":p.optimization_steps,"target_evaluations":p.target_evaluations})).collect::<Vec<_>>()});
        initial = initialized.position.to_vec();
        initialization_seconds = Some(initialization_start.elapsed().as_secs_f64());
        before = t.counts();
        warmup_start = Instant::now();
    }
    let mut trace = Vec::new();
    let stride = job.warmup.div_ceil(4096).max(1);
    let mut observe = |o: NutsWarmupObservation<'_>| {
        if o.completed.is_multiple_of(stride) || o.completed == job.warmup {
            trace.push(json!({"completed":o.completed,"stage":format!("{:?}",o.stage),
                "log_density":o.log_density,"step":o.step_size.value(),
                "acceptance":o.transition.acceptance_probability,"steps":o.transition.integration_steps,
                "tree_depth":o.transition.tree_depth,"selected_index":o.transition.selected_index,
                "termination":format!("{:?}",o.transition.termination)}));
        }
    };
    let warm;
    let work_before_retained;
    let retained_seconds;
    let rows;
    let summary;
    if job.method.starts_with("nuts_fisher") {
        let mut options =
            FisherOptions::new(job.warmup).with_max_rank(if job.method.ends_with("rank4") {
                4.min(t.dimension())
            } else {
                0
            });
        if job.method.ends_with("rm") {
            options = options.with_step_adaptation(StepAdaptation::robbins_monro(
                0.1,
                DiminishingSchedule::new(10., 0.6)?,
            )?);
        }
        if job.method.ends_with("adam") {
            options = options.with_step_adaptation(StepAdaptation::adam(
                0.05,
                DiminishingSchedule::new(10., 0.6)?,
                0.9,
                0.999,
                1e-8,
            )?);
        }
        let warmup = FisherNutsWarmup::new(t, owned(&initial), nuts, options)?;
        let (s, r) = if job.record_warmup_trace {
            warmup.run_observed(&mut rng, &mut observe)?
        } else {
            warmup.run(&mut rng)?
        };
        summary = report(&r);
        warm = warmup_start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain_nuts(s, &mut rng, job.draws)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else if job.method == "nuts_saia_joint" {
        use alea_mcmc::adapt::JointNutsWarmup;
        let (s, r) = JointNutsWarmup::new(
            t,
            owned(&initial),
            nuts,
            JointWarmupOptions::new(
                WarmupOptions::new(job.warmup),
                CurvatureProbeOptions::new(2, 3)?,
                2.,
            )?,
        )?
        .run(&mut rng)?;
        summary = json!({"step":r.step_size.value(),"coefficient":r.two_stage_coefficient,"hvp_calls":r.hvp_calls,
            "warmup_integration_steps":r.integration_attempts,"coefficient_updates":r.coefficient_updates,
            "metric_updates":r.metric_updates,"final_fast_iterations":r.final_fast_iterations,
            "curvature_policy_upper":r.curvature_upper_bound,"selection_dimensionless_limit":r.selection_dimensionless_limit,
            "bounded_step_updates":r.bounded_step_updates});
        warm = warmup_start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain_nuts(s, &mut rng, job.draws)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else {
        let options =
            WarmupOptions::new(job.warmup).with_metric(if job.method == "nuts_cov_dense" {
                MetricKind::Dense
            } else {
                MetricKind::Diagonal
            });
        let warmup = NutsWarmup::new(t, owned(&initial), nuts, options)?;
        let warmup = if job.method.starts_with("nuts_grad_diag") {
            warmup.with_gradient_diagonal()?
        } else {
            warmup
        };
        let warmup = if matches!(
            job.method.as_str(),
            "nuts_grad_diag_jitter" | "nuts_grad_diag_position"
        ) {
            warmup.with_gradient_final_restart(false)
        } else {
            warmup
        };
        let (s, r) = if job.record_warmup_trace {
            warmup.run_observed(&mut rng, &mut observe)?
        } else {
            warmup.run(&mut rng)?
        };
        summary = report(&r);
        warm = warmup_start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain_nuts(s, &mut rng, job.draws)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    }
    let mut phase = json!({"chain":chain,"workflow_version":3,"warmup_seconds":warm,
        "retained_seconds":retained_seconds,"work_before":before,"work_before_retained":work_before_retained,
        "work_after":t.counts(),"frozen":summary,"initialization":initialization,"warmup_observations":trace,
        "observation_scope":if job.record_warmup_trace { "bounded live-state trace collected during measured warmup; no retained tracing" }
            else { "warmup trace disabled by sealed protocol; summary and actual work counters retained" }});
    if let Some(seconds) = initialization_seconds {
        phase["initialization_seconds"] = json!(seconds);
        phase["initialization_work_before"] = initialization_work_before;
        phase["initialization_work_after"] = before;
        phase["initialization_scope"] =
            json!("Pathfinder initialization, disjoint from warmup_seconds");
    }
    out.phases.push(phase);
    out.save(job, t, chain, rows)
}

fn static_chain<I: Integrator + Clone>(
    job: &Job,
    t: &Target,
    chain: usize,
    out: &mut Output,
    integrator: I,
) -> Result<(), Any> {
    use alea_mcmc::adapt::{
        StaticWarmupOptions, warmup_drghmc_step_size, warmup_ghmc_step_size, warmup_hmc_step_size,
    };
    let start = Instant::now();
    let before = t.counts();
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let pilot_iterations = if job.method.starts_with("walnuts") {
        job.warmup
    } else {
        job.warmup * 3 / 4
    };
    let (pilot, pilot_report) = NutsWarmup::new_with_integrator(
        t,
        owned(&job.initials[chain]),
        alea_mcmc::NutsOptions::new(0.1, job.depth)?,
        WarmupOptions::new(pilot_iterations),
        integrator.clone(),
    )?
    .run(&mut rng)?;
    let tuning = StaticWarmupOptions::new(job.warmup - pilot_iterations);
    let step = pilot.options().step_size().value();
    let warm;
    let work_before_retained;
    let retained_seconds;
    let rows;
    let frozen;
    if job.method == "ghmc" {
        let s = Ghmc::new_with_integrator(
            t,
            owned(pilot.point().position()),
            pilot.metric().clone(),
            GhmcOptions::new(step, 8, 0.8)?,
            integrator,
            &mut rng,
        )?;
        let (mut s, r) = warmup_ghmc_step_size(s, tuning, &mut rng)?;
        frozen = json!({"step":r.step_size.value(),"kernel_warmup_integration_steps":r.integration_attempts,"kernel_warmup_divergences":r.divergences});
        warm = start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = collect(job.draws, || {
            let i = s.step(&mut rng)?;
            Ok((
                s.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    acceptance: Some(i.acceptance_probability),
                    divergent: i.divergence.is_some(),
                    energy: i.initial_energy,
                    error: i.energy_error,
                    ..Info::default()
                },
            ))
        })?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else if job.method == "drghmc" {
        let s = DrGhmc::new_with_integrator(
            t,
            owned(pilot.point().position()),
            pilot.metric().clone(),
            DrGhmcOptions::new(step, 2., 0.8)?.with_integration_steps(8)?,
            integrator,
            &mut rng,
        )?;
        let (mut s, r) = warmup_drghmc_step_size(s, tuning, &mut rng)?;
        let mut retained_partition = [0_usize; 3];
        frozen = json!({"step":r.step_size.value(),"kernel_warmup_integration_steps":r.integration_attempts,
            "kernel_warmup_divergences":r.divergences,"step_adaptation":"first-stage MH probability; work partitioned",
            "first_stage_integration_steps":r.first_stage_integration_steps,"delayed_integration_steps":r.delayed_integration_steps,"ghost_integration_steps":r.ghost_integration_steps});
        warm = start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = collect(job.draws, || {
            let i = s.step(&mut rng)?;
            retained_partition[0] += i.first.integration_steps;
            retained_partition[1] += i.second.integration_steps;
            retained_partition[2] += i.ghost_integration_steps;
            Ok((
                s.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    acceptance: None,
                    divergent: i.first.divergence.is_some() || i.second.divergence.is_some(),
                    energy: i.initial_energy,
                    error: i.first.energy_error,
                    ..Info::default()
                },
            ))
        })?;
        retained_seconds = retained.elapsed().as_secs_f64();
        out.phases.push(json!({"chain":chain,"retained_first_stage_steps":retained_partition[0],"retained_delayed_steps":retained_partition[1],"retained_ghost_steps":retained_partition[2]}));
    } else {
        let h = pilot.into_hmc(std::num::NonZeroUsize::new(8).expect("eight"));
        let (h, r) = warmup_hmc_step_size(h, tuning, &mut rng)?;
        frozen = json!({"step":r.step_size.value(),"kernel_warmup_integration_steps":r.integration_attempts,"kernel_warmup_divergences":r.divergences});
        warm = start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain(h, &job.method, &mut rng, job.draws, job.depth)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    }
    out.phases.push(json!({"chain":chain,"workflow_version":3,"warmup_seconds":warm,"retained_seconds":retained_seconds,
        "work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"pilot":report(&pilot_report),"frozen":frozen}));
    out.save(job, t, chain, rows)
}
fn walnuts_chain(job: &Job, t: &Target, chain: usize, out: &mut Output) -> Result<(), Any> {
    let start = Instant::now();
    let before = t.counts();
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let (s, r) = NutsWarmup::new(
        t,
        owned(&job.initials[chain]),
        alea_mcmc::NutsOptions::new(0.1, job.depth)?,
        WarmupOptions::new(job.warmup),
    )?
    .run(&mut rng)?;
    let h = s.into_hmc(std::num::NonZeroUsize::new(1).expect("one"));
    let warm = start.elapsed().as_secs_f64();
    let work_before_retained = t.counts();
    let retained = Instant::now();
    let rows = retain_walnuts(h, &job.method, &mut rng, job.draws, job.depth)?;
    out.phases.push(json!({"chain":chain,"workflow_version":3,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),
        "work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"pilot":report(&r)}));
    out.save(job, t, chain, rows)
}

fn transport_chain(job: &Job, t: &Target, chain: usize, out: &mut Output) -> Result<(), Any> {
    use alea_mcmc::adapt::{TransportLearningError, WarmupStage};
    let start = Instant::now();
    let before = t.counts();
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let iterations = job.warmup / 2;
    let mut learner = AffineTransportLearner::new(
        t.dimension(),
        TransportLearningOptions::new(iterations.max(2), 4.min(t.dimension()))?,
    )?;
    let mut failed: Option<TransportLearningError> = None;
    let (pilot, pilot_report) = NutsWarmup::new(
        t,
        owned(&job.initials[chain]),
        alea_mcmc::NutsOptions::new(0.1, job.depth)?,
        WarmupOptions::new(iterations),
    )?
    .run_observed(&mut rng, |o| {
        if matches!(o.stage, WarmupStage::Slow { .. }) && failed.is_none() {
            failed = learner.observe(o.position).err();
        }
    })?;
    if let Some(error) = failed {
        return Err(error.into());
    }
    let observations = learner.observations();
    let map = learner.finish()?;
    let mut latent = OwnedBuffer::new(t.dimension());
    map.inverse(pilot.point().position(), &mut latent)?;
    let transformed = FrozenTransportTarget::new(t, map)?;
    let (s, r) = NutsWarmup::new(
        &transformed,
        latent,
        alea_mcmc::NutsOptions::new(0.1, job.depth)?,
        WarmupOptions::new(job.warmup - iterations),
    )?
    .run(&mut rng)?;
    let warm = start.elapsed().as_secs_f64();
    let work_before_retained = t.counts();
    let retained = Instant::now();
    let mut rows = retain_nuts(s, &mut rng, job.draws)?;
    for (q, _) in &mut rows {
        let mut target = vec![0.; q.len()];
        transformed.to_target(q, &mut target)?;
        *q = target;
    }
    out.phases.push(json!({"chain":chain,"workflow_version":3,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),
        "work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"pilot":report(&pilot_report),
        "transport_observations":observations,"transport_observation_scope":"live slow-window NUTS states including repeats; one frozen map",
        "frozen":report(&r)}));
    out.save(job, t, chain, rows)
}

fn isokinetic_chain(job: &Job, t: &Target, chain: usize, out: &mut Output) -> Result<(), Any> {
    use alea_mcmc::adapt::{
        MclmcWarmupOptions, StaticWarmupOptions, WarmupStage, warmup_mams_step_size,
        warmup_mclmc_step_size,
    };
    let start = Instant::now();
    let before = t.counts();
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let pilot_iterations = job.warmup * 3 / 4;
    let mut learner = AffineTransportLearner::new(
        t.dimension(),
        TransportLearningOptions::new(pilot_iterations.max(2), 0)?,
    )?;
    let mut failure = None;
    let mut pilot_states = Vec::new();
    let (pilot, pilot_report) = NutsWarmup::new(
        t,
        owned(&job.initials[chain]),
        alea_mcmc::NutsOptions::new(0.1, job.depth)?,
        WarmupOptions::new(pilot_iterations),
    )?
    .run_observed(&mut rng, |o| {
        if matches!(o.stage, WarmupStage::Slow { .. }) && failure.is_none() {
            failure = learner.observe(o.position).err();
            pilot_states.push(o.position.to_vec());
        }
    })?;
    if let Some(error) = failure {
        return Err(error.into());
    }
    let map = learner.finish()?;
    let mut latent = OwnedBuffer::new(t.dimension());
    map.inverse(pilot.point().position(), &mut latent)?;
    let transformed = FrozenTransportTarget::new(t, map)?;
    // Estimate the radius from actual centered pilot variances in the shared
    // transformed coordinates. This retains ridge/data effects rather than
    // replacing the measured scales by an assumed unit variance.
    let mut mean = vec![0.; t.dimension()];
    let mut m2 = vec![0.; t.dimension()];
    let mut latent_state = vec![0.; t.dimension()];
    for (index, q) in pilot_states.iter().enumerate() {
        transformed.map().inverse(q, &mut latent_state)?;
        for k in 0..t.dimension() {
            let delta = latent_state[k] - mean[k];
            mean[k] += delta / (index + 1) as f64;
            m2[k] += delta * (latent_state[k] - mean[k]);
        }
    }
    let refresh_length =
        (m2.iter().sum::<f64>() / pilot_states.len().saturating_sub(1) as f64).sqrt();
    if !refresh_length.is_finite() || refresh_length <= 0. {
        return Err("invalid measured isokinetic pilot scale".into());
    }
    let step = 0.1 * refresh_length;
    let warm;
    let work_before_retained;
    let retained_seconds;
    let mut rows;
    let frozen;
    if job.method == "mams" {
        let s = Mams::new(
            &transformed,
            latent,
            MamsOptions::new(step, 1)?.with_trajectory_time(refresh_length, 1024)?,
        )?;
        let (mut s, r) = warmup_mams_step_size(
            s,
            StaticWarmupOptions::new(job.warmup - pilot_iterations),
            &mut rng,
        )?;
        frozen = json!({"step":r.step_size.value(),"target_acceptance":0.8,"trajectory_time":refresh_length,"integration_steps":s.options().integration_steps().get(),"maximum_steps":1024,"kernel_warmup_integration_steps":r.integration_attempts,"kernel_warmup_divergences":r.divergences});
        warm = start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = collect(job.draws, || {
            let i = s.step(&mut rng)?;
            Ok((
                s.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    acceptance: Some(i.acceptance_probability),
                    divergent: i.divergence.is_some(),
                    error: i.correction_error,
                    ..Info::default()
                },
            ))
        })?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else {
        let s = ApproximateMclmc::new(
            ApproximateTarget::acknowledge_bias(),
            &transformed,
            latent,
            ApproximateMclmcOptions::new(step, 8, Some(refresh_length))?,
            &mut rng,
        )?;
        let (mut s, r) = warmup_mclmc_step_size(
            s,
            MclmcWarmupOptions::new(job.warmup - pilot_iterations),
            &mut rng,
        )?;
        frozen = json!({"step":r.step_size.value(),"target_error_per_dimension":r.target_per_dimension,"error_power":r.error_power,
            "observed_error_per_dimension":r.observed_error_per_dimension,"kernel_warmup_integration_steps":r.integration_attempts,
            "refresh_length":refresh_length,"pilot_scale_observations":pilot_states.len(),"recovered_numerical_failures":r.recovered_numerical_failures,"error_statistic":"centered signed correction coefficient variance","target_guarantee":"approximate finite-step target; omitted MH correction"});
        warm = start.elapsed().as_secs_f64();
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = collect(job.draws, || {
            let i = s.step(&mut rng)?;
            Ok((
                s.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    error: Some(i.correction_error),
                    ..Info::default()
                },
            ))
        })?;
        retained_seconds = retained.elapsed().as_secs_f64();
    }
    for (q, _) in &mut rows {
        let mut original = vec![0.; q.len()];
        transformed.to_target(q, &mut original)?;
        *q = original;
    }
    out.phases.push(json!({"chain":chain,"workflow_version":3,"warmup_seconds":warm,"retained_seconds":retained_seconds,
        "work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"pilot":report(&pilot_report),"frozen":frozen,
        "preconditioning":"frozen affine map from pilot slow-window variances; repeated states included"}));
    out.save(job, t, chain, rows)
}

pub(super) fn run_chees(job: &Job, t: &Target, out: &mut Output) -> Result<(), Any> {
    use alea_mcmc::adapt::WarmupStage;
    let start = Instant::now();
    let before = t.counts();
    let pilot_iterations = job.warmup / 2;
    let mut learner = AffineTransportLearner::new(
        t.dimension(),
        TransportLearningOptions::new((pilot_iterations * job.seeds.len()).clamp(2, 4096), 0)?,
    )?;
    let mut pilots = Vec::new();
    let mut reports = Vec::new();
    let mut initial_steps = Vec::new();
    for (c, seed) in job.seeds.iter().enumerate() {
        let mut seed_bytes = [0; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = chain_rng(seed_bytes, ChainId(c as u64));
        let mut failure = None;
        let (pilot, r) = NutsWarmup::new(
            t,
            owned(&job.initials[c]),
            alea_mcmc::NutsOptions::new(0.1, job.depth)?,
            WarmupOptions::new(pilot_iterations),
        )?
        .run_observed(&mut rng, |o| {
            if matches!(o.stage, WarmupStage::Slow { .. }) && failure.is_none() {
                failure = learner.observe(o.position).err();
            }
        })?;
        if let Some(error) = failure {
            return Err(error.into());
        }
        initial_steps.push(pilot.options().step_size().value());
        reports.push(report(&r));
        pilots.push((pilot.point().position().to_vec(), rng));
    }
    let observations = learner.observations();
    let map = learner.finish()?;
    let transformed = FrozenTransportTarget::new(t, map)?;
    let step = (initial_steps.iter().map(|s| s.ln()).sum::<f64>() / initial_steps.len() as f64)
        .exp()
        .clamp(1e-8, 10.);
    let mut adapter = CheesEnsembleAdapter::new(
        t.dimension(),
        step.try_into()?,
        1.,
        0.01,
        8.,
        CheesCriterion::Global,
    )?
    .with_max_integration_steps(1024)?;
    let frozen = adapter.clone().finish()?;
    let mut chains = Vec::new();
    for (c, (q, rng)) in pilots.into_iter().enumerate() {
        let mut latent = OwnedBuffer::new(t.dimension());
        transformed.map().inverse(&q, &mut latent)?;
        chains.push(Chain {
            id: ChainId(c as u64),
            sampler: CheesHmc::new(
                &transformed,
                latent,
                IdentityMetric::new(t.dimension()),
                CheesHmcOptions::new(frozen),
            )?,
            rng,
            sink: (),
        });
    }
    // This serial benchmark deliberately supports chain-local RefCell scratch;
    // the production Rayon orchestrator correctly requires Sync targets.
    let mut warmup_steps = 0_usize;
    for _ in 0..job.warmup - pilot_iterations {
        let plan = adapter.next_plan()?;
        let transitions = chains
            .iter_mut()
            .map(|c| c.sampler.step_with_plan(plan, &mut c.rng))
            .collect::<Result<Vec<_>, _>>()?;
        let observations = chains
            .iter()
            .zip(&transitions)
            .map(|(c, t)| c.sampler.observation(t))
            .collect::<Vec<_>>();
        let _ = adapter.update(&observations)?;
        warmup_steps = warmup_steps
            .checked_add(
                plan.integration_steps
                    .checked_mul(chains.len())
                    .ok_or("ChEES work overflow")?,
            )
            .ok_or("ChEES work overflow")?;
    }
    let frozen = adapter.finish()?;
    for c in &mut chains {
        c.sampler.install_frozen(frozen);
    }
    out.phases.push(json!({"workflow_version":3,"warmup_seconds":start.elapsed().as_secs_f64(),"work_before":before,
        "work_after_warmup":t.counts(),"pilots":reports,"shared_affine_observations":observations,"warmup_integration_steps":warmup_steps,
        "maximum_integration_steps":1024,"seed_policy":"full u64 seed in ChaCha12 key; logical chain stream; pilot stream continues"}));
    for (c, chain) in chains.iter_mut().enumerate() {
        let before = t.counts();
        let retained = Instant::now();
        let rows = collect(job.draws, || {
            let i = chain.sampler.step(&mut chain.rng)?;
            let mut q = vec![0.; t.dimension()];
            transformed.to_target(chain.sampler.point().position(), &mut q)?;
            Ok((
                q,
                Info {
                    steps: i.plan.integration_steps,
                    acceptance: Some(i.acceptance_probability),
                    divergent: i.divergence.is_some(),
                    energy: i.initial_energy,
                    error: i.energy_error,
                    ..Info::default()
                },
            ))
        })?;
        out.phases.push(json!({"chain":c,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before_retained":before,"work_after":t.counts()}));
        out.save(job, t, c, rows)?;
    }
    Ok(())
}

fn atlas_chain(job: &Job, t: &Target, chain: usize, out: &mut Output) -> Result<(), Any> {
    use alea_mcmc::atlas_sampler::{AtlasOptions, AtlasStage, AtlasWarmup};
    let start = Instant::now();
    let before = t.counts();
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let pilot_iterations = job.warmup * 3 / 4;
    let (pilot, pilot_report) = NutsWarmup::new(
        t,
        owned(&job.initials[chain]),
        alea_mcmc::NutsOptions::new(0.1, job.depth)?,
        WarmupOptions::new(pilot_iterations),
    )?
    .run(&mut rng)?;
    let options = AtlasOptions::new(pilot.options().step_size().value(), 1024)?;
    let (mut s, r) = AtlasWarmup::from_hmc(
        pilot.into_hmc(std::num::NonZeroUsize::new(1).expect("one")),
        options,
        job.warmup - pilot_iterations,
    )?
    .run(&mut rng)?;
    let frozen_lengths = s.trajectory_lengths().to_vec();
    let warm = start.elapsed().as_secs_f64();
    let work_before_retained = t.counts();
    let retained = Instant::now();
    let mut stages = [0_usize; 3];
    let mut hvps = 0;
    let mut partitions = [0_usize; 3];
    let mut rejections = std::collections::BTreeMap::new();
    let rows = collect(job.draws, || {
        let i = s.step(&mut rng)?;
        stages[match i.stage {
            AtlasStage::Preliminary => 0,
            AtlasStage::Delayed => 1,
            AtlasStage::Failure => 2,
        }] += 1;
        hvps += i.hvp_calls;
        partitions[0] += i.forward_integration_steps;
        partitions[1] += i.reverse_integration_steps;
        partitions[2] += i.ghost_integration_steps;
        if let Some(reason) = i.rejection {
            *rejections.entry(format!("{reason:?}")).or_insert(0_usize) += 1;
        }
        Ok((
            s.point().position().to_vec(),
            Info {
                steps: i.integration_steps,
                acceptance: None,
                energy: i.initial_energy,
                error: i.energy_error,
                ..Info::default()
            },
        ))
    })?;
    out.phases.push(json!({"chain":chain,"workflow_version":3,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),
        "work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"pilot":report(&pilot_report),
        "atlas_warmup_integration_steps":r.integration_steps,"atlas_warmup_hvp_calls":r.hvp_calls,
        "atlas_warmup_forward_steps":r.forward_integration_steps,"atlas_warmup_reverse_steps":r.reverse_integration_steps,"atlas_warmup_ghost_steps":r.ghost_integration_steps,
        "retained_forward_steps":partitions[0],"retained_reverse_steps":partitions[1],"retained_ghost_steps":partitions[2],
        "frozen":{"baseline_step":options.baseline_step().value(),"trajectory_lengths":frozen_lengths,"maximum_steps_per_leg":1024,
            "constant_trajectory":2,"probabilistic":1,"delayed_stages":1,"curvature":"deterministic14-call HVP policy; no auxiliary random fit",
            "acceptance":"paper Appendix E.2; actual selection densities rather than the Python implementation's Hastings ratios"},
        "retained_stages":stages,"retained_hvp_calls":hvps,"retained_rejections":rejections}));
    out.save(job, t, chain, rows)
}
