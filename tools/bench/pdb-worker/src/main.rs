//! PosteriorDB experiment worker; intentionally outside the production workspace.
#![cfg_attr(feature = "native-enzyme", feature(autodiff))]
mod diagnostics;
mod modern;
mod target;
mod third_ad;
use alea_core::{
    capability::{HessianVector, PotentialHessianDerivative},
    target::LogDensityGradient,
    transport::FrozenTransportTarget,
};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric},
};
use alea_mcmc::{
    Hmc, HmcOptions, MarkovChain, Nuts,
    adapt::{
        AffineTransportLearner, CheesCriterion, CheesEnsembleAdapter, CurvatureProbeOptions,
        DiminishingSchedule, FisherHmcWarmup, FisherOptions, HmcWarmup, JointHmcWarmup,
        JointWarmupOptions, ManifoldBayesTuner, ManifoldTunerOptions, MetricKind,
        MultiPathfinderOptions, RmhmcManifoldWarmup, StepAdaptation, TransportLearningOptions,
        WarmupOptions, learn_affine_transport, multi_pathfinder,
    },
    chees::{CheesHmc, CheesHmcOptions},
    ghmc::{DrGhmc, DrGhmcOptions, Ghmc, GhmcOptions},
    integrator::{Integrator, TwoStage},
    mams::{ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget, Mams, MamsOptions},
    riemannian::{ExactnessPendingRmhmc, RmhmcOptions, SoftAbsMetric, SoftAbsOptions},
    walnuts::{Walnuts, WalnutsMicrostepDistribution, WalnutsOptions},
};
use alea_runtime::{
    chees,
    ensemble::{Chain, Execution, chain_rng},
    output::ChainId,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use target::Target;
type Any = Box<dyn Error>;
#[derive(Deserialize)]
struct Job {
    #[serde(default)]
    workflow_version: u8,
    #[serde(default = "default_warmup_trace")]
    record_warmup_trace: bool,
    method: String,
    directory: PathBuf,
    initials: Vec<Vec<f64>>,
    seeds: Vec<u64>,
    warmup: usize,
    draws: usize,
    depth: usize,
    include_tp: bool,
}
fn default_warmup_trace() -> bool {
    true
}
#[derive(Clone, Copy, Default)]
struct Info {
    divergent: bool,
    saturated: bool,
    steps: usize,
    acceptance: Option<f64>,
    energy: Option<f64>,
    error: Option<f64>,
}
struct Output {
    raw: BufWriter<fs::File>,
    constrained: BufWriter<fs::File>,
    phases: Vec<Value>,
}
fn new_file(path: &Path) -> Result<BufWriter<fs::File>, Any> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn write_json(path: &Path, v: &Value) -> Result<(), Any> {
    let mut f = new_file(path)?;
    serde_json::to_writer_pretty(&mut f, v)?;
    writeln!(f)?;
    f.flush()?;
    Ok(())
}
fn owned(q: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(q.len(), |i| q[i])
}
fn optional(x: Option<f64>) -> String {
    x.map_or_else(String::new, |x| format!("{x:0.17e}"))
}
impl Output {
    fn new(job: &Job, t: &Target) -> Result<Self, Any> {
        let mut raw = new_file(&job.directory.join("draws.csv"))?;
        write!(
            raw,
            "chain,draw,divergent,saturated,steps,acceptance,energy,energy_error"
        )?;
        for i in 0..t.dimension() {
            write!(raw, ",q{i}")?;
        }
        writeln!(raw)?;
        let mut constrained = new_file(&job.directory.join("constrained.csv"))?;
        writeln!(
            constrained,
            "chain,draw,{}",
            t.model.param_names(job.include_tp, false)
        )?;
        Ok(Self {
            raw,
            constrained,
            phases: vec![],
        })
    }
    fn save(
        &mut self,
        job: &Job,
        t: &Target,
        chain: usize,
        rows: Vec<(Vec<f64>, Info)>,
    ) -> Result<(), Any> {
        let start = Instant::now();
        for (draw, (q, i)) in rows.into_iter().enumerate() {
            write!(
                self.raw,
                "{chain},{draw},{},{},{},{},{},{}",
                u8::from(i.divergent),
                u8::from(i.saturated),
                i.steps,
                optional(i.acceptance),
                optional(i.energy),
                optional(i.error)
            )?;
            for v in &q {
                write!(self.raw, ",{v:0.17e}")?;
            }
            writeln!(self.raw)?;
            let p = t.constrain(&q, job.include_tp)?;
            write!(self.constrained, "{chain},{draw}")?;
            for v in p {
                write!(self.constrained, ",{v:0.17e}")?;
            }
            writeln!(self.constrained)?;
        }
        self.raw.flush()?;
        self.constrained.flush()?;
        self.phases
            .push(json!({"chain":chain,"serialization_seconds":start.elapsed().as_secs_f64()}));
        self.progress(job, t)
    }
    fn progress(&self, job: &Job, t: &Target) -> Result<(), Any> {
        fs::write(
            job.directory.join("progress-worker.json"),
            serde_json::to_vec_pretty(&json!({"phases":self.phases,"work":t.counts()}))?,
        )?;
        Ok(())
    }
}
fn collect<F>(n: usize, mut step: F) -> Result<Vec<(Vec<f64>, Info)>, Any>
where
    F: FnMut() -> Result<(Vec<f64>, Info), Any>,
{
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        rows.push(step()?);
    }
    Ok(rows)
}
fn retain<T, M, I>(
    chain: Hmc<'_, T, M, I>,
    method: &str,
    rng: &mut SmallRng,
    n: usize,
    depth: usize,
) -> Result<Vec<(Vec<f64>, Info)>, Any>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    if method.starts_with("hmc_") {
        let mut chain = chain;
        collect(n, || {
            let i = chain.step(rng)?;
            Ok((
                chain.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    divergent: i.divergence.is_some(),
                    acceptance: Some(i.acceptance_probability),
                    energy: i.initial_energy,
                    error: i.energy_error,
                    ..Info::default()
                },
            ))
        })
    } else {
        let mut sampler = Nuts::from_hmc(chain, depth.try_into()?)?;
        collect(n, || {
            let i = sampler.step(rng)?;
            Ok((
                sampler.point().position().to_vec(),
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
}

fn retain_walnuts<T: LogDensityGradient + ?Sized, M: EuclideanMetric>(
    chain: Hmc<'_, T, M>,
    method: &str,
    rng: &mut SmallRng,
    n: usize,
    depth: usize,
) -> Result<Vec<(Vec<f64>, Info)>, Any> {
    let mut opts = WalnutsOptions::new(chain.options().step_size().value() * 2., depth)?
        .with_local_refinement(6, 1, 0.21)?;
    if method.ends_with("r2p") {
        opts =
            opts.with_microstep_distribution(WalnutsMicrostepDistribution::RandomizedTwoPoint)?;
    }
    let mut sampler = Walnuts::from_hmc(chain, opts)?;
    collect(n, || {
        let i = sampler.step(rng)?;
        Ok((
            sampler.point().position().to_vec(),
            Info {
                steps: i.microsteps,
                acceptance: None,
                divergent: matches!(
                    i.termination,
                    alea_mcmc::walnuts::WalnutsTermination::Divergence(_)
                ),
                saturated: i.termination == alea_mcmc::walnuts::WalnutsTermination::MaxDepth,
                ..Info::default()
            },
        ))
    })
}

fn add_rmhmc_work(
    total: &mut Value,
    o: &alea_mcmc::riemannian::RmhmcTransition,
) -> Result<(), Any> {
    let c = o.costs;
    for (name, value) in [
        ("target_evaluations", c.target_evaluations),
        (
            "metric_derivative_evaluations",
            c.metric_derivative_evaluations,
        ),
        ("factorizations", c.factorizations),
        ("triangular_solves", c.triangular_solves),
        ("inverse_actions", c.inverse_actions),
        ("dynamics_steps", c.dynamics_steps),
        ("momentum_iterations", c.momentum_iterations),
        ("position_iterations", c.position_iterations),
        ("newton_jacobians", c.newton_jacobians),
        ("newton_linear_solves", c.newton_linear_solves),
        ("reverse_steps", c.reverse_steps),
    ] {
        total[name] = json!(
            total[name]
                .as_u64()
                .unwrap_or(0)
                .checked_add(value as u64)
                .ok_or("RMHMC accounting overflow")?
        );
    }
    for (name, event) in [
        ("reverse_passed", o.reverse_passed == Some(true)),
        ("reverse_failed", o.reverse_passed == Some(false)),
        ("terminated", o.termination.is_some()),
    ] {
        total[name] = json!(total[name].as_u64().unwrap_or(0) + u64::from(event));
    }
    if let Some(error) = o.maximum_reverse_error
        && error.is_finite()
    {
        total["maximum_reverse_error"] = json!(
            total["maximum_reverse_error"]
                .as_f64()
                .unwrap_or(0.)
                .max(error)
        );
    }
    Ok(())
}

fn run_chain(job: &Job, t: &Target, chain: usize, out: &mut Output) -> Result<(), Any> {
    // nuts_legacy_warmup deliberately follows the historical HmcWarmup path
    // below and converts its frozen state to the unchanged retained NUTS law.
    if job.workflow_version == 3
        && (job.method.starts_with("nuts_cov_")
            || job.method == "nuts_grad_diag"
            || job.method == "nuts_grad_diag_jitter"
            || job.method == "nuts_grad_diag_position"
            || job.method.starts_with("nuts_fisher")
            || matches!(
                job.method.as_str(),
                "nuts_transport"
                    | "nuts_pathfinder_raw"
                    | "nuts_pathfinder_psis"
                    | "nuts_saia_joint"
            )
            || job.method.starts_with("hmc_")
            || matches!(
                job.method.as_str(),
                "ghmc" | "drghmc" | "mams" | "mclmc" | "atlas_full"
            )
            || job.method.starts_with("walnuts"))
    {
        return modern::run_chain(job, t, chain, out);
    }
    let mut rng = SmallRng::seed_from_u64(job.seeds[chain]);
    let mut initial = job.initials[chain].clone();
    let method = job.method.as_str();
    let start = Instant::now();
    let before = t.counts();
    let mut extra = json!({});
    if method.starts_with("rmhmc") {
        let metric = SoftAbsMetric::new(t, SoftAbsOptions::new(1., 64, 1e-10)?)?;
        let options = if job.workflow_version == 3 {
            [0.01, 0.02, 0.04].map(|step| {
                RmhmcOptions::new(step, 2, 100, 1e-12)
                    .expect("fixed valid policy")
                    .with_newton_solver()
                    .with_reverse_check(1e-8)
                    .expect("fixed replay policy")
            })
        } else {
            [
                RmhmcOptions::new(0.01, 2, 4, 1e-7)?,
                RmhmcOptions::new(0.02, 2, 4, 1e-7)?,
                RmhmcOptions::new(0.04, 2, 4, 1e-7)?,
            ]
        };
        let mut warmup_work = json!({});
        let mut sampler = if method.ends_with("manifold") {
            let mut tuner = ManifoldBayesTuner::new(&options, ManifoldTunerOptions::default())?;
            let mut warm = RmhmcManifoldWarmup::new(t, &metric, owned(&initial), options[0])?;
            for _ in 0..job.warmup {
                let k = tuner.suggest()?;
                let q = warm.position().to_vec();
                let info = warm.step(options[k], &mut rng)?;
                add_rmhmc_work(&mut warmup_work, &info)?;
                // Positive declared dynamics cost also covers early implicit rejections.
                tuner.observe(k, &q, warm.position(), options[k].integration_steps().get())?;
            }
            let frozen = tuner.finish()?;
            extra = json!({"selected_candidate":frozen.candidate_index()});
            warm.finish(frozen)
        } else {
            let mut s = ExactnessPendingRmhmc::new(t, &metric, owned(&initial), options[1])?;
            for _ in 0..job.warmup {
                let info = s.step(&mut rng)?;
                add_rmhmc_work(&mut warmup_work, &info)?;
            }
            s
        };
        let warm = start.elapsed().as_secs_f64();
        let work_before_retained = t.counts();
        let retained = Instant::now();
        let mut factorizations = 0;
        let mut solves = 0;
        let mut retained_work = json!({});
        let rows = collect(job.draws, || {
            let i = sampler.step(&mut rng)?;
            add_rmhmc_work(&mut retained_work, &i)?;
            factorizations += i.costs.factorizations;
            solves += i.costs.triangular_solves;
            Ok((
                sampler.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    divergent: i.termination.is_some(),
                    acceptance: Some(i.acceptance_probability),
                    energy: i.initial_energy,
                    error: i.energy_error,
                    ..Info::default()
                },
            ))
        })?;
        out.phases.push(json!({"chain":chain,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"retained_factorizations":factorizations,"retained_triangular_solves":solves,"warmup_transition_work":warmup_work,"retained_transition_work":retained_work,"frozen":extra}));
        return out.save(job, t, chain, rows);
    }
    if method.starts_with("mams") || method.starts_with("mclmc") {
        // Explicit fixed identity-coordinate policy; corrected and unadjusted variants
        // remain separate. Warmup includes the exact same kernel as retained sampling.
        let step = 0.08 / (t.dimension() as f64).sqrt();
        if method == "mams" {
            let mut s = Mams::new(t, owned(&initial), MamsOptions::new(step, 8)?)?;
            for _ in 0..job.warmup {
                let _ = s.step(&mut rng)?;
            }
            let warm = start.elapsed().as_secs_f64();
            let work_before_retained = t.counts();
            let retained = Instant::now();
            let rows = collect(job.draws, || {
                let i = s.step(&mut rng)?;
                Ok((
                    s.point().position().to_vec(),
                    Info {
                        steps: i.integration_steps,
                        divergent: i.divergence.is_some(),
                        acceptance: Some(i.acceptance_probability),
                        error: i.correction_error,
                        ..Info::default()
                    },
                ))
            })?;
            out.phases.push(json!({"chain":chain,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"step":step}));
            return out.save(job, t, chain, rows);
        }
        let mut s = ApproximateMclmc::new(
            ApproximateTarget::acknowledge_bias(),
            t,
            owned(&initial),
            ApproximateMclmcOptions::new(step, 8, Some(1.))?,
            &mut rng,
        )?;
        for _ in 0..job.warmup {
            let _ = s.step(&mut rng)?;
        }
        let warm = start.elapsed().as_secs_f64();
        let work_before_retained = t.counts();
        let retained = Instant::now();
        let rows = collect(job.draws, || {
            let i = s.step(&mut rng)?;
            Ok((
                s.point().position().to_vec(),
                Info {
                    steps: i.integration_steps,
                    ..Info::default()
                },
            ))
        })?;
        out.phases.push(json!({"chain":chain,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"step":step}));
        return out.save(job, t, chain, rows);
    }
    if method == "nuts_pathfinder" {
        let starts: Vec<f64> = (0..4)
            .flat_map(|_| {
                initial
                    .iter()
                    .map(|q| q + rng.random_range(-0.5..0.5))
                    .collect::<Vec<_>>()
            })
            .collect();
        let initialized = multi_pathfinder(
            t,
            &starts,
            MultiPathfinderOptions::new(4, 40)?
                .with_elbo_draws(8)?
                .with_draws_per_path(16)?,
            &mut rng,
        )?;
        initial = initialized.position.to_vec();
        extra = json!({"pathfinder_seconds":start.elapsed().as_secs_f64(),"paths":initialized.report.paths,"optimization_steps":initialized.report.optimization_steps,"proposal_draws":initialized.report.proposal_draws,"raw_importance_resampling":true});
    }
    let settings = WarmupOptions::new(job.warmup)
        .with_target(0.8.try_into()?)
        .with_metric(if method == "nuts_cov_dense" {
            MetricKind::Dense
        } else {
            MetricKind::Diagonal
        });
    let hmc = HmcOptions::new(0.1, 8)?;
    if method == "nuts_transport" {
        let iterations = job.warmup / 2;
        let mut s = Hmc::new(
            t,
            owned(&initial),
            IdentityMetric::new(t.dimension()),
            HmcOptions::new(0.02, 8)?,
        )?;
        let learned = learn_affine_transport(
            &mut s,
            AffineTransportLearner::new(
                t.dimension(),
                TransportLearningOptions::new(iterations + 1, 4.min(t.dimension()))?,
            )?,
            iterations,
            &mut rng,
        )?;
        let transformed = FrozenTransportTarget::new(t, learned.map)?;
        let (h, report) = HmcWarmup::new(
            &transformed,
            learned.latent_position,
            hmc,
            WarmupOptions::new(job.warmup - iterations),
        )?
        .run(&mut rng)?;
        let warm = start.elapsed().as_secs_f64();
        let work_before_retained = t.counts();
        let retained = Instant::now();
        let mut rows = retain(h, "nuts", &mut rng, job.draws, job.depth)?;
        for (q, _) in &mut rows {
            let mut target_q = vec![0.; q.len()];
            transformed.to_target(q, &mut target_q)?;
            *q = target_q;
        }
        out.phases.push(json!({"chain":chain,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"warmup_integration_steps":iterations*8+report.integration_attempts}));
        return out.save(job, t, chain, rows);
    }
    let rows;
    let warm;
    let retained_seconds;
    let work_before_retained;
    if method.starts_with("nuts_fisher") {
        let mut options =
            FisherOptions::new(job.warmup).with_max_rank(if method.ends_with("rank4") {
                4.min(t.dimension())
            } else {
                0
            });
        if method.ends_with("rm") {
            options = options.with_step_adaptation(StepAdaptation::robbins_monro(
                0.1,
                DiminishingSchedule::new(10., 0.6)?,
            )?);
        }
        if method.ends_with("adam") {
            options = options.with_step_adaptation(StepAdaptation::adam(
                0.05,
                DiminishingSchedule::new(10., 0.6)?,
                0.9,
                0.999,
                1e-8,
            )?);
        }
        let (h, report) = FisherHmcWarmup::new(t, owned(&initial), hmc, options)?.run(&mut rng)?;
        warm = start.elapsed().as_secs_f64();
        extra["step"] = json!(h.options().step_size().value());
        extra["warmup_integration_steps"] = json!(report.integration_attempts);
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain(h, method, &mut rng, job.draws, job.depth)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else if method == "nuts_atlas_joint" {
        let (h, report) = JointHmcWarmup::new(
            t,
            owned(&initial),
            hmc,
            JointWarmupOptions::new(settings, CurvatureProbeOptions::new(2, 3)?, 2.)?,
        )?
        .run(&mut rng)?;
        warm = start.elapsed().as_secs_f64();
        extra = json!({"coefficient":report.two_stage_coefficient,"hvp_calls":report.hvp_calls,"step":report.step_size.value(),"warmup_integration_steps":report.integration_attempts});
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain(h, method, &mut rng, job.draws, job.depth)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else if method == "hmc_omf" || method == "hmc_bcss" {
        let integrator = if method == "hmc_omf" {
            TwoStage::OMF
        } else {
            TwoStage::BCSS
        };
        let (h, report) =
            HmcWarmup::new_with_integrator(t, owned(&initial), hmc, settings, integrator)?
                .run(&mut rng)?;
        warm = start.elapsed().as_secs_f64();
        extra = json!({"step":h.options().step_size().value(),"warmup_integration_steps":report.integration_attempts});
        work_before_retained = t.counts();
        let retained = Instant::now();
        rows = retain(h, method, &mut rng, job.draws, job.depth)?;
        retained_seconds = retained.elapsed().as_secs_f64();
    } else {
        let (h, report) = HmcWarmup::new(t, owned(&initial), hmc, settings)?.run(&mut rng)?;
        warm = start.elapsed().as_secs_f64();
        let step = h.options().step_size().value();
        extra["step"] = json!(step);
        extra["warmup_integration_steps"] = json!(report.integration_attempts);
        work_before_retained = t.counts();
        let retained = Instant::now();
        if method == "ghmc" {
            let mut s = Ghmc::new(
                t,
                owned(h.point().position()),
                h.metric().clone(),
                GhmcOptions::new(step, 8, 0.8)?,
                &mut rng,
            )?;
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
        } else if method == "drghmc" {
            let mut s = DrGhmc::new(
                t,
                owned(h.point().position()),
                h.metric().clone(),
                DrGhmcOptions::new(step, 2., 0.8)?.with_integration_steps(8)?,
                &mut rng,
            )?;
            rows = collect(job.draws, || {
                let i = s.step(&mut rng)?;
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
        } else if method.starts_with("walnuts") {
            rows = retain_walnuts(h, method, &mut rng, job.draws, job.depth)?;
        } else {
            rows = retain(h, method, &mut rng, job.draws, job.depth)?;
        }
        retained_seconds = retained.elapsed().as_secs_f64();
    }
    out.phases.push(json!({"chain":chain,"warmup_seconds":warm,"retained_seconds":retained_seconds,"work_before":before,"work_before_retained":work_before_retained,"work_after":t.counts(),"frozen":extra}));
    out.save(job, t, chain, rows)
}
fn run_chees(job: &Job, t: &Target, out: &mut Output) -> Result<(), Any> {
    let start = Instant::now();
    let adapter = CheesEnsembleAdapter::new(
        t.dimension(),
        0.05.try_into()?,
        1.,
        0.01,
        8.,
        CheesCriterion::Global,
    )?;
    let frozen = adapter.clone().finish()?;
    let mut chains = Vec::new();
    for (c, seed) in job.seeds.iter().enumerate() {
        chains.push(Chain {
            id: ChainId(c as u64),
            sampler: CheesHmc::new(
                t,
                owned(&job.initials[c]),
                IdentityMetric::new(t.dimension()),
                CheesHmcOptions::new(frozen),
            )?,
            rng: chain_rng([(*seed % 256) as u8; 32], ChainId(c as u64)),
            sink: (),
        });
    }
    let (_, report) = chees::warmup(&mut chains, adapter, job.warmup, Execution::Sequential)?;
    out.phases.push(json!({"warmup_seconds":start.elapsed().as_secs_f64(),"cohort_chains":job.seeds.len(),"warmup_integration_steps":report.integration_steps}));
    for (c, chain) in chains.iter_mut().enumerate() {
        let work_before_retained = t.counts();
        let retained = Instant::now();
        let rows = collect(job.draws, || {
            let i = chain.sampler.step(&mut chain.rng)?;
            Ok((
                chain.sampler.point().position().to_vec(),
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
        out.phases.push(json!({"chain":c,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before_retained":work_before_retained,"work_after":t.counts()}));
        out.save(job, t, c, rows)?;
    }
    Ok(())
}
fn metadata(t: &mut Target) -> Result<Value, Any> {
    let d = t.dimension();
    let unames = t.model.param_unc_names().to_owned();
    let tp = t.model.param_num(true, false) <= 256;
    let mut initials = vec![];
    for seed in [18001, 18002, 18003, 18004] {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut found = None;
        for _ in 0..100 {
            let q: Vec<f64> = (0..d).map(|_| rng.random_range(-2.0..2.0)).collect();
            let mut g = vec![0.; d];
            if t.logp_grad(&q, &mut g)?.is_finite() && g.iter().all(|v| v.is_finite()) {
                found = Some(q);
                break;
            }
        }
        initials.push(found.ok_or("no finite initialization in 100 declared attempts")?);
    }
    let validation = t.validate_derivatives(&initials[0])?;
    let constrained: Vec<_> = initials
        .iter()
        .map(|q| t.constrain(q, false))
        .collect::<Result<_, _>>()?;
    Ok(
        json!({"derivative_validation":validation,"dimension":d,"parameter_names":t.model.param_names(tp,false),"initial_parameter_names":t.model.param_names(false,false),"unconstrained_names":unames,"include_tp":tp,"initials":initials,"initial_constrained":constrained,"model_info":t.model.info().to_string_lossy()}),
    )
}
fn main() -> Result<(), Any> {
    let entry = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: alea-pdb-worker metadata|sample|diagnose CONFIG [OUTPUT]".into());
    }
    if args[0] == "diagnose" {
        return diagnostics::run(Path::new(&args[1]));
    }
    let bytes = fs::read(&args[1])?;
    if bytes.len() > 16_777_216 {
        return Err("job input too large".into());
    }
    let v: Value = serde_json::from_slice(&bytes)?;
    let mut t = Target::open(
        Path::new(v["library"].as_str().ok_or("missing library")?),
        Path::new(v["data"].as_str().ok_or("missing data")?),
    )?;
    if v["target_backend"].as_str() == Some("enzyme") {
        t.enable_native(v["case"].as_str().ok_or("missing native case")?)?;
    }
    if args[0] == "native-probe" {
        let points: Vec<Vec<f64>> = serde_json::from_value(v["points"].clone())?;
        let mut rows = Vec::new();
        for q in points {
            let mut gradient = vec![0.; t.dimension()];
            let value = t.logp_grad(&q, &mut gradient)?;
            let direction: Vec<_> = (0..q.len())
                .map(|i| if i % 2 == 0 { 1. } else { -1. })
                .collect();
            let mut hvp = vec![0.; q.len()];
            let admitted = value.is_finite() && gradient.iter().all(|x| x.is_finite());
            let hvp_error = if admitted {
                t.potential_hvp(&q, &direction, &mut hvp)
                    .err()
                    .map(|e| e.to_string())
            } else {
                Some("outside finite support".to_owned())
            };
            let mut hessian = vec![0.; q.len() * q.len()];
            let mut third = vec![0.; q.len() * q.len() * q.len()];
            let third_error = if admitted && q.len() <= 10 {
                t.potential_hessian_and_derivatives(&q, &mut hessian, &mut third)
                    .err()
                    .map(|e| e.to_string())
            } else {
                Some("outside support or dense derivative dimension budget".to_owned())
            };
            let mut third_relative_error = None;
            if third_error.is_none() && rows.is_empty() {
                let d = q.len();
                let mut error = 0_f64;
                let mut point = q.clone();
                let mut plus = vec![0.; d * d];
                let mut minus = vec![0.; d * d];
                for k in 0..d {
                    let delta = 1e-5 * (1. + q[k].abs());
                    point[k] = q[k] + delta;
                    t.hessian(&point, &mut plus)?;
                    point[k] = q[k] - delta;
                    t.hessian(&point, &mut minus)?;
                    point[k] = q[k];
                    for i in 0..d * d {
                        let expected = (plus[i] - minus[i]) / (2. * delta);
                        if !expected.is_finite() {
                            return Err("nonfinite third-derivative reference".into());
                        }
                        error = error
                            .max((third[k * d * d + i] - expected).abs() / (1. + expected.abs()));
                    }
                }
                third_relative_error = Some(error);
            }
            rows.push(json!({"q":q,"admitted":admitted,"logp":value,"gradient":gradient,"hvp":hvp,"hvp_error":hvp_error,"hessian":if third_error.is_none(){Some(hessian)}else{None},"third":if third_error.is_none(){Some(third)}else{None},"third_error":third_error,"third_relative_error":third_relative_error}));
        }
        return write_json(
            Path::new(args.get(2).ok_or("missing probe output")?),
            &json!({"backend":t.backend(),"rows":rows}),
        );
    }
    if v["workflow_version"].as_u64() == Some(3)
        && v["method"].as_str().is_some_and(|m| m.starts_with("rmhmc"))
        && v["target_backend"].as_str() != Some("enzyme")
    {
        t.enable_third_ad(
            Path::new(
                v["third_ad_library"]
                    .as_str()
                    .ok_or("v3 RMHMC requires a per-model third-AD helper")?,
            ),
            Path::new(v["data"].as_str().ok_or("missing data")?),
        )?;
    }
    if args[0] == "metadata" {
        return write_json(
            Path::new(args.get(2).ok_or("missing output")?),
            &metadata(&mut t)?,
        );
    }
    let job: Job = serde_json::from_value(v)?;
    if job.workflow_version == 3
        && !matches!(
            job.method.as_str(),
            "nuts_cov_diag"
                | "nuts_grad_diag"
                | "nuts_grad_diag_jitter"
                | "nuts_grad_diag_position"
                | "nuts_legacy_warmup"
                | "nuts_cov_dense"
                | "nuts_fisher_diag"
                | "nuts_fisher_rank4"
                | "nuts_fisher_rm"
                | "nuts_fisher_adam"
                | "nuts_saia_joint"
                | "nuts_pathfinder_raw"
                | "nuts_pathfinder_psis"
                | "nuts_transport"
                | "hmc_cov"
                | "hmc_omf"
                | "hmc_bcss"
                | "ghmc"
                | "drghmc"
                | "mams"
                | "mclmc"
                | "chees"
                | "walnuts_d"
                | "walnuts_r2p"
                | "atlas_full"
                | "rmhmc_softabs"
                | "rmhmc_manifold"
                | "nuts_rs"
        )
    {
        return Err("unknown version-three workflow".into());
    }
    if !matches!(job.workflow_version, 0 | 3)
        || job.seeds.is_empty()
        || job.seeds.len() != job.initials.len()
        || job.seeds.len() > 16
        || job.draws > 1_000_000
        || job.warmup > 1_000_000
        || t.dimension() > 128
        || job.initials.iter().any(|q| q.len() != t.dimension())
    {
        return Err("job shape/resource admission failed".into());
    }
    if args[0] == "search-probes" {
        return modern::search_probes(&job, &t);
    }
    let mut out = Output::new(&job, &t)?;
    let cold_seconds = entry.elapsed().as_secs_f64();
    let start = Instant::now();
    let result = if job.method == "chees" {
        if job.workflow_version == 3 {
            modern::run_chees(&job, &t, &mut out)
        } else {
            run_chees(&job, &t, &mut out)
        }
    } else if job.method == "nuts_rs" {
        upstream(&job, &t, &mut out)
    } else {
        (0..job.seeds.len()).try_for_each(|c| run_chain(&job, &t, c, &mut out))
    };
    let record = json!({"schema_version":2,"status":if result.is_ok(){"completed"}else{"failed"},"failure":result.as_ref().err().map(|e|e.to_string()),"cold_seconds":cold_seconds,"cold_scope":"Rust main entry through target loading and output setup; dynamic loader costs remain in process wall time","worker_seconds":start.elapsed().as_secs_f64(),"phases":out.phases,"work":t.counts(),"backend":t.backend(),"hessian":if uses_native_backend(&t) {"Enzyme reverse-over-forward HVP and reverse-over-forward-over-forward third AD; no finite differences"} else if t.has_third_ad() {"BridgeStan Hessian AD; per-model Stan grad_hessian third AD; no finite differences"} else {"BridgeStan forward-over-reverse AD; third derivatives central differences with h=1e-4*(1+abs(q))"}});
    write_json(&job.directory.join("worker.json"), &record)?;
    result
}
fn uses_native_backend(t: &Target) -> bool {
    t.backend().starts_with("native Rust")
}
fn upstream(job: &Job, t: &Target, out: &mut Output) -> Result<(), Any> {
    use nuts_rs::{
        Chain as _, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError,
        Settings,
    };
    use std::collections::HashMap;
    struct Func<'a>(&'a Target);
    impl HasDims for Func<'_> {
        fn dim_sizes(&self) -> HashMap<String, u64> {
            HashMap::new()
        }
    }
    #[derive(Debug)]
    struct E(String);
    impl std::fmt::Display for E {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl Error for E {}
    impl LogpError for E {
        fn is_recoverable(&self) -> bool {
            true
        }
    }
    impl CpuLogpFunc for Func<'_> {
        type LogpError = E;
        type ExpandedVector = Vec<f64>;
        type FlowParameters = ();
        fn dim(&self) -> usize {
            self.0.dimension()
        }
        fn logp(&mut self, q: &[f64], g: &mut [f64]) -> Result<f64, E> {
            self.0.logp_grad(q, g).map_err(|e| E(e.to_string()))
        }
        fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
            &mut self,
            _: &mut R,
            q: &[f64],
        ) -> Result<Vec<f64>, CpuMathError> {
            Ok(q.to_vec())
        }
    }
    for (c, seed) in job.seeds.iter().enumerate() {
        let start = Instant::now();
        let mut rng = nuts_rs::rand::rngs::SmallRng::seed_from_u64(*seed);
        let settings = DiagNutsSettings {
            num_tune: job.warmup as u64,
            num_draws: job.draws as u64,
            maxdepth: job.depth as u64,
            ..Default::default()
        };
        let mut sampler = settings.new_chain(c as u64, CpuMath::new(Func(t)), &mut rng);
        sampler.set_position(&job.initials[c])?;
        for _ in 0..job.warmup {
            sampler.draw()?;
        }
        let warm = start.elapsed().as_secs_f64();
        let work_before_retained = t.counts();
        let retained = Instant::now();
        let rows = collect(job.draws, || {
            let (q, _expanded, stats, info) = sampler.expanded_draw()?;
            if info.tuning {
                return Err("retained adaptation".into());
            }
            Ok((
                q.to_vec(),
                Info {
                    steps: info.num_steps as usize,
                    divergent: info.diverging,
                    saturated: stats.maxdepth_reached,
                    acceptance: Some(stats.adapt.step_size.mean_tree_accept),
                    error: Some(stats.adapt.step_size.max_energy_error),
                    ..Info::default()
                },
            ))
        })?;
        out.phases.push(json!({"chain":c,"warmup_seconds":warm,"retained_seconds":retained.elapsed().as_secs_f64(),"work_before_retained":work_before_retained,"work_after":t.counts()}));
        out.save(job, t, c, rows)?;
    }
    Ok(())
}
