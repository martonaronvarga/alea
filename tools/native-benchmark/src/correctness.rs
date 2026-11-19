//! Independent posterior equations and finite differences, never timed.
use super::*;

#[test]
fn timed_state_bank_validates_every_coordinate_before_evaluation() {
    assert_eq!(
        parse_state_bank("1 2\n3 4", 2).unwrap(),
        vec![1., 2., 3., 4.]
    );
    for (text, d) in [
        ("", 2),
        ("1 2 3", 2),
        ("1 NaN", 2),
        ("1 inf", 2),
        ("1 no", 2),
        ("1", 0),
    ] {
        assert!(parse_state_bank(text, d).is_err());
    }
}

#[test]
fn chain_parallelism_preserves_seeded_native_trajectories() {
    let input = Input {
        n: 32,
        k: 0,
        x: vec![],
        y: vec![0.25; 32],
    };
    let run = |execution| {
        let session = SessionBuilder::new("parallel benchmark control", 2, vec![vec![0.; 2]; 4])
            .seed([17; 32])
            .draws(32)
            .sampler(SamplerConfig::Nuts(NutsOptions::new(0.05, 5).unwrap()))
            .build()
            .unwrap();
        let outcome = session
            .run(
                |_| {
                    Ok::<_, alea::model::ModelError<AssemblyError>>(
                        build_model_with_execution("mean", &input, RegressionExecution::Auto)
                            .unwrap()
                            .into_target()
                            .unwrap(),
                    )
                },
                execution,
                &Discard,
            )
            .unwrap();
        outcome
            .chains
            .into_iter()
            .map(|chain| {
                assert!(chain.failure.is_none());
                assert_eq!(chain.completed, 32);
                chain
                    .draws
                    .into_iter()
                    .map(|draw| draw.position)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    assert_eq!(run(Execution::Sequential), run(Execution::Parallel(&pool)));
}

#[test]
fn scalar_and_auto_regression_use_equivalent_native_posteriors() {
    assert!(regression_execution(Some("invalid")).is_err());
    for name in ["mean", "linear", "logistic"] {
        let input = Input {
            n: 32,
            k: 2,
            x: (0..64).map(|i| (i % 7) as f64 / 8.).collect(),
            y: (0..32).map(|i| (i % 2) as f64).collect(),
        };
        let auto = build_model_with_execution(name, &input, RegressionExecution::Auto)
            .unwrap()
            .into_target()
            .unwrap();
        let scalar = build_model_with_execution(name, &input, RegressionExecution::Scalar)
            .unwrap()
            .into_target()
            .unwrap();
        for value in [-1., 0., 0.3] {
            let q = vec![value; auto.dimension()];
            let mut ga = vec![0.; q.len()];
            let mut gs = ga.clone();
            let a = auto.logp_grad(&q, &mut ga).unwrap();
            let s = scalar.logp_grad(&q, &mut gs).unwrap();
            assert!((a - s).abs() < 1e-10 * (1. + a.abs()));
            for (a, s) in ga.iter().zip(&gs) {
                assert!((a - s).abs() < 1e-10 * (1. + a.abs()));
            }
        }
    }
}

#[test]
fn acceptance_configuration_is_explicit_bounded_and_preserves_default() {
    assert_eq!(acceptance_option(None).unwrap(), 0.8);
    assert_eq!(acceptance_option(Some("0.95")).unwrap(), 0.95);
    for text in ["NaN", "inf", "0", "1", "-0.5", "2", "not-a-number"] {
        assert!(acceptance_option(Some(text)).is_err());
    }
}

#[test]
fn compound_models_have_correct_scores_charts_and_recover_after_failure() {
    for name in ["schools", "schools_nc", "covariance"] {
        let data = if name == "covariance" {
            let (x, y) = conformance::covariance_data();
            Input { n: 32, k: 2, x, y }
        } else {
            Input {
                n: 8,
                k: 1,
                x: vec![15., 10., 16., 11., 9., 11., 10., 18.],
                y: vec![28., 8., -3., 7., -1., 1., 18., 12.],
            }
        };
        let model = build_model(name, &data).unwrap();
        if name != "covariance" {
            assert!(matches!(
                model.rust_primal(),
                Err(AssemblyError::MissingPrimal { .. })
            ));
        }
        let target = model.into_target().unwrap();
        let d = target.dimension();
        for scale in [-1.5, 0., 0.3] {
            let mut q: Vec<_> = (0..d).map(|j| scale + 0.07 * j as f64).collect();
            let mut g = vec![0.; d];
            let value = target.logp_grad(&q, &mut g).unwrap();
            for j in 0..d {
                q[j] += 1e-5;
                let plus = target.logp_grad(&q, &mut vec![0.; d]).unwrap();
                q[j] -= 2e-5;
                let minus = target.logp_grad(&q, &mut vec![0.; d]).unwrap();
                q[j] += 1e-5;
                close(g[j], (plus - minus) / 2e-5, 2e-6);
            }
            let mut invalid = q.clone();
            invalid[if name == "covariance" { 2 } else { d - 1 }] = -1000.;
            assert!(alea::target::evaluate(&target, &invalid, &mut g).is_err());
            close(target.logp_grad(&q, &mut g).unwrap(), value, 2e-12);
        }
    }
}

#[test]
fn centered_and_noncentered_schools_agree_under_explicit_change_of_variables() {
    let data = Input {
        n: 8,
        k: 1,
        x: vec![15., 10., 16., 11., 9., 11., 10., 18.],
        y: vec![28., 8., -3., 7., -1., 1., 18., 12.],
    };
    let c = build_model("schools", &data)
        .unwrap()
        .into_target()
        .unwrap();
    let nc = build_model("schools_nc", &data)
        .unwrap()
        .into_target()
        .unwrap();
    for log_tau in [-2_f64, 0., 1.] {
        let mut q = vec![0.; 10];
        q[8] = 0.7;
        q[9] = log_tau;
        for (i, x) in q.iter_mut().take(8).enumerate() {
            *x = i as f64 / 4. - 1.;
        }
        let mut theta = q.clone();
        for i in 0..8 {
            theta[i] = q[8] + log_tau.exp() * q[i];
        }
        let mut gc = vec![0.; 10];
        let mut gn = vec![0.; 10];
        close(
            nc.logp_grad(&q, &mut gn).unwrap(),
            c.logp_grad(&theta, &mut gc).unwrap() + 8. * log_tau,
            2e-12,
        );
        for i in 0..8 {
            close(gn[i], log_tau.exp() * gc[i], 2e-12);
        }
        close(gn[8], gc[8] + gc[..8].iter().sum::<f64>(), 2e-12);
        close(
            gn[9],
            gc[9]
                + gc[..8]
                    .iter()
                    .zip(&q)
                    .map(|(g, z)| g * log_tau.exp() * z)
                    .sum::<f64>()
                + 8.,
            2e-12,
        );
    }
}

#[cfg(feature = "enzyme")]
#[test]
fn generated_covariance_matches_native_and_rejects_invalid_points() {
    let (x, y) = conformance::covariance_data();
    let native = conformance::build("covariance", 32, 2, &x, &y)
        .unwrap()
        .into_target()
        .unwrap();
    let generated = generated_covariance::target();
    for q in [[0.; 5], [0.3, -0.7, 0.2, -0.4, 0.6], [-1.5; 5]] {
        let mut a = [0.; 5];
        let mut b = [0.; 5];
        close(
            native.logp_grad(&q, &mut a).unwrap(),
            generated.logp_grad(&q, &mut b).unwrap(),
            2e-12,
        );
        for (a, b) in a.into_iter().zip(b) {
            close(a, b, 2e-11);
        }
    }
    assert!(alea::target::evaluate(&generated, &[0., 0., -1000., 0., 0.], &mut [0.; 5]).is_err());
    assert!(generated.logp_grad(&[0.; 5], &mut [0.; 5]).is_ok());
}

fn input(n: usize) -> Input {
    let k = 8;
    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<_> = (0..k)
            .map(|j| {
                (((i + 1) * (j + 1)) as f64 * 0.017).sin()
                    + 0.5 * ((((i + 3) * (j + 2)) as f64) * 0.019).cos()
            })
            .collect();
        let eta: f64 = row
            .iter()
            .enumerate()
            .map(|(j, x)| x * if j % 2 == 0 { 0.3 } else { -0.3 } / (1. + j as f64 / 4.))
            .sum();
        y.push(eta + 0.4 * (i as f64 * 0.71).sin());
        x.extend(row);
    }
    Input { n, k, x, y }
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1. + expected.abs()),
        "{actual} vs {expected}"
    );
}

// Stan's positive-real chart is sigma=exp(s). Its normal prior omits the
// half-normal ln(2) constant; Alea's positive normal factor does so as well.
fn reference(name: &str, data: &Input, q: &[f64]) -> (f64, Vec<f64>) {
    let p = if name == "mean" { 1 } else { data.k };
    let mut value = -(p as f64) * (2_f64.ln() + 0.5 * std::f64::consts::TAU.ln());
    let mut gradient = vec![0.; q.len()];
    for j in 0..p {
        value -= q[j] * q[j] / 8.;
        gradient[j] = -q[j] / 4.;
    }
    if name != "logistic" {
        let s = q[p];
        let variance = (2. * s).exp();
        value += -0.5 * variance - 0.5 * std::f64::consts::TAU.ln() + s;
        gradient[p] = 1. - variance - data.n as f64;
        for i in 0..data.n {
            let eta: f64 = if name == "mean" {
                q[0]
            } else {
                (0..p).map(|j| data.x[i * p + j] * q[j]).sum()
            };
            let r = data.y[i] - eta;
            value += -s - 0.5 * std::f64::consts::TAU.ln() - 0.5 * r * r / variance;
            for (j, score) in gradient.iter_mut().take(p).enumerate() {
                *score += r / variance
                    * if name == "mean" {
                        1.
                    } else {
                        data.x[i * p + j]
                    };
            }
            gradient[p] += r * r / variance;
        }
    } else {
        for i in 0..data.n {
            let eta: f64 = (0..p).map(|j| data.x[i * p + j] * q[j]).sum();
            let signed = if data.y[i] == 1. { -eta } else { eta };
            value -= signed.max(0.) + (-signed.abs()).exp().ln_1p();
            let probability = if eta >= 0. {
                1. / (1. + (-eta).exp())
            } else {
                eta.exp() / (1. + eta.exp())
            };
            for (j, score) in gradient.iter_mut().take(p).enumerate() {
                *score += data.x[i * p + j] * (data.y[i] - probability);
            }
        }
    }
    (value, gradient)
}

#[test]
fn posterior_density_scores_and_finite_differences_match_equations() {
    for n in [37, 1000, 2500, 5000, 7500, 10000] {
        for name in ["mean", "linear", "logistic"] {
            let mut data = input(n);
            if name == "logistic" {
                data.y
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, y)| *y = f64::from(i % 3 != 0));
            }
            let model = build_model(name, &data).unwrap();
            let target = model.select_posterior().unwrap().into_target().unwrap();
            for scale in [0., 0.3, -1.5] {
                let mut q = vec![scale; target.dimension()];
                if name != "logistic" {
                    *q.last_mut().unwrap() = scale;
                }
                let mut g = vec![0.; q.len()];
                let value = target.logp_grad(&q, &mut g).unwrap();
                let (expected, score) = reference(name, &data, &q);
                close(value, expected, 2e-12);
                for (a, b) in g.iter().zip(score) {
                    close(*a, b, 3e-10);
                }
                if n == 37 {
                    for j in 0..q.len() {
                        let h = 1e-5;
                        q[j] += h;
                        let plus = target.logp_grad(&q, &mut vec![0.; g.len()]).unwrap();
                        q[j] -= 2. * h;
                        let minus = target.logp_grad(&q, &mut vec![0.; g.len()]).unwrap();
                        q[j] += h;
                        close(g[j], (plus - minus) / (2. * h), 2e-7);
                    }
                }
            }
        }
    }
}

#[test]
fn extreme_coordinates_preserve_finite_values_or_explicit_errors() {
    let data = input(37);
    let target = build_model("linear", &data).unwrap().into_target().unwrap();
    for log_sigma in [-1000., -745.2, -400., -300., 400., 710., 1000.] {
        let mut q = vec![0.; 9];
        q[8] = log_sigma;
        let mut g = vec![42.; 9];
        let error = target.logp_grad(&q, &mut g).unwrap_err();
        assert!(target.is_numerical_error(&error), "{error:?}");
        assert_eq!(g, vec![42.; 9]);
    }
    let mut q = vec![0.; 9];
    q[0] = 1e200;
    let error = target.logp_grad(&q, &mut [0.; 9]).unwrap_err();
    assert!(target.is_numerical_error(&error));
    let mut data = input(37);
    data.y
        .iter_mut()
        .enumerate()
        .for_each(|(i, y)| *y = f64::from(i % 2 == 0));
    let target = build_model("logistic", &data)
        .unwrap()
        .into_target()
        .unwrap();
    for q in [vec![1000.; 8], vec![-1000.; 8]] {
        let mut g = vec![0.; 8];
        let value = target.logp_grad(&q, &mut g).unwrap();
        let (expected, score) = reference("logistic", &data, &q);
        close(value, expected, 1e-12);
        for (a, b) in g.iter().zip(score) {
            close(*a, b, 1e-12);
        }
    }
}

#[test]
fn original_seed_large_regression_recovers_from_second_warmup_transition() {
    use alea::{
        math::buffer::OwnedBuffer,
        output::ChainId,
        sampling::adapt::{NutsWarmup, WarmupOptions},
    };
    let metrics = Arc::new(Mutex::new(Metrics::default()));
    let target = Observed {
        target: build_model("linear", &input(10000))
            .unwrap()
            .select_posterior()
            .unwrap()
            .into_target()
            .unwrap(),
        metrics: Arc::clone(&metrics),
        timed: false,
    };
    let warmup = NutsWarmup::new(
        &target,
        OwnedBuffer::new(9),
        NutsOptions::new(0.2, 10).unwrap(),
        WarmupOptions::new(2).with_search(None),
    )
    .unwrap();
    let mut rng = alea::ensemble::chain_rng([17; 32], ChainId(0));
    let (chain, report) = warmup.run(&mut rng).unwrap();
    assert_eq!(report.iterations, 2);
    assert_eq!(report.divergences, 2);
    assert_eq!(chain.point().position(), &[0.; 9]);
    let m = metrics.lock().unwrap();
    assert_eq!(m.calls, 3);
    assert_eq!(m.errors, 1);
    assert!(
        m.first_error
            .as_ref()
            .unwrap()
            .contains("PrecisionLoss { index: 0 }")
    );
}

#[test]
fn block_models_match_retained_scalar_references_and_keep_source_limits_explicit() {
    for name in ["schools", "schools_nc", "covariance"] {
        let (n, k, x, y) = if name == "covariance" {
            let (x, y) = conformance::covariance_data();
            (32, 2, x, y)
        } else {
            (
                8,
                1,
                vec![15., 10., 16., 11., 9., 11., 10., 18.],
                vec![28., 8., -3., 7., -1., 1., 18., 12.],
            )
        };
        let native = conformance::native(name, n, k, &x, &y).unwrap();
        assert!(matches!(
            native.rust_primal(),
            Err(AssemblyError::MissingPrimal { .. })
        ));
        let reference = conformance::build(name, n, k, &x, &y).unwrap();
        assert_eq!(reference.rust_primal().is_ok(), name == "covariance");
        let native = native.into_target().unwrap();
        let reference = reference.into_target().unwrap();
        let d = native.dimension();
        let mut points: Vec<Vec<_>> = [-3., 0., 2.]
            .into_iter()
            .map(|level| (0..d).map(|i| level + (i % 5) as f64 / 8.).collect())
            .collect();
        if name == "covariance" {
            // A zero predictor slope still needs its loading score. Opposite
            // mean/scale directions also check the grouped region's input order.
            points.push(vec![0.25, -0.5, -0.3, 0.4, 0.]);
            points.push(vec![-0.5, 0.25, 0.4, -0.3, -1.25]);
        }
        for q in points {
            let mut a = vec![0.; d];
            let mut b = a.clone();
            close(
                native.logp_grad(&q, &mut a).unwrap(),
                reference.logp_grad(&q, &mut b).unwrap(),
                2e-11,
            );
            for (a, b) in a.into_iter().zip(b) {
                close(a, b, 2e-10);
            }
        }
    }
}

#[test]
fn radon_native_port_has_checked_indices_and_explicit_source_gap() {
    let model = conformance::radon(2, &[0, 1, 0], &[0., 1., 1.], &[0.5, 1., -0.25]).unwrap();
    assert!(matches!(
        model.rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    assert!(conformance::radon(2, &[2], &[0.], &[1.]).is_err());
    assert!(
        model
            .into_target()
            .unwrap()
            .logp_grad(&[0.; 6], &mut [0.; 6])
            .unwrap()
            .is_finite()
    );
}

#[test]
fn block_predictor_graph_size_does_not_scalarize_observation_rows() {
    let nodes: Vec<_> = [8, 257]
        .into_iter()
        .map(|n| {
            conformance::native("covariance", n, 2, &vec![0.25; 2 * n], &vec![0.; n])
                .unwrap()
                .expression_count()
        })
        .collect();
    assert_eq!(nodes[0], nodes[1]);
    let nodes: Vec<_> = [8, 257]
        .into_iter()
        .map(|n| {
            conformance::radon(
                2,
                &(0..n).map(|i| i % 2).collect::<Vec<_>>(),
                &vec![0.25; n],
                &vec![0.; n],
            )
            .unwrap()
            .expression_count()
        })
        .collect();
    assert_eq!(nodes[0], nodes[1]);
    assert!(conformance::native("schools", 0, 1, &[], &[]).is_err());
    assert!(conformance::native("covariance", 1, 1, &[0.], &[0.]).is_err());
    for build in [conformance::build, conformance::native] {
        assert!(matches!(
            build("schools", 1, 1, &[1.], &[f64::NAN]),
            Err(AssemblyError::Numerical)
        ));
        assert!(matches!(
            build("covariance", 1, 2, &[0., f64::NAN], &[0.]),
            Err(AssemblyError::Numerical)
        ));
    }
    assert!(matches!(
        conformance::radon(1, &[0], &[0.], &[f64::NAN]),
        Err(AssemblyError::Numerical)
    ));
}

#[test]
fn radon_json_admission_never_rounds_or_wraps_county_indices() {
    let mut data = serde_json::json!({"J":2,"N":3,"county_idx":[1,2,1],"floor_measure":[0.,1.,1.],"log_radon":[0.5,1.,-0.25]});
    assert!(radon_input(&data).is_ok());
    for bad in [
        serde_json::json!(0),
        serde_json::json!(3),
        serde_json::json!(1.5),
        serde_json::json!(u64::MAX),
    ] {
        data["county_idx"][1] = bad;
        assert!(radon_input(&data).is_err());
    }
    data["county_idx"][1] = serde_json::json!(2);
    data["J"] = serde_json::json!(u64::MAX);
    assert!(radon_input(&data).is_err());
    data["J"] = serde_json::json!(2);
    data["N"] = serde_json::json!(2);
    assert!(radon_input(&data).is_err());
}

#[test]
fn composition_representation_is_explicit_and_defaults_to_the_existing_block_path() {
    assert!(composition_blocks(None).unwrap());
    assert!(composition_blocks(Some("block")).unwrap());
    assert!(!composition_blocks(Some("scalar")).unwrap());
    for invalid in ["auto", "source", "", "v2"] {
        assert!(composition_blocks(Some(invalid)).is_err());
    }
}

#[test]
fn scientific_quantities_recover_original_noncentered_effects_and_scale_moments() {
    for (name, d, groups) in [("schools_nc", 10, 8), ("radon", 7, 3)] {
        let chains: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|c| {
                (0..d)
                    .map(|j| {
                        (0..16)
                            .map(|i| 0.1 + (i + c) as f64 * 0.03 + j as f64 * 0.02)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let result = scientific_summaries(name, d, &chains);
        assert_eq!(result.len(), 2 * d);
        for j in 0..groups {
            let values: Vec<Vec<f64>> = chains
                .iter()
                .map(|c| {
                    (0..16)
                        .map(|i| {
                            let (mu, tau) = if name == "radon" {
                                (d - 3, d - 2)
                            } else {
                                (d - 2, d - 1)
                            };
                            c[mu][i] + c[tau][i].exp() * c[j][i]
                        })
                        .collect()
                })
                .collect();
            let expected = values.iter().flatten().sum::<f64>() / 64.;
            let second = values.iter().flatten().map(|v| v * v).sum::<f64>() / 64.;
            assert!((result[2 * j]["mean"].as_f64().unwrap() - expected).abs() < 1e-12);
            assert!((result[2 * j + 1]["mean"].as_f64().unwrap() - second).abs() < 1e-12);
            assert!(result[2 * j]["mcse_mean"].as_f64().unwrap() > 0.);
        }
    }
}

#[test]
fn output_modes_validate_paths_and_discard_preserves_the_actual_trajectory() {
    assert_eq!(sampling_output(None, false), Ok("memory"));
    assert_eq!(sampling_output(None, true), Ok("csv"));
    assert_eq!(sampling_output(Some("discard"), false), Ok("discard"));
    for (mode, path) in [
        (Some("csv"), false),
        (Some("discard"), true),
        (Some("memory"), true),
        (Some("other"), false),
    ] {
        assert!(sampling_output(mode, path).is_err());
    }
    let run = |collect| {
        let input = Input {
            n: 32,
            k: 0,
            x: vec![],
            y: vec![0.25; 32],
        };
        let session = SessionBuilder::new("output control", 2, vec![vec![0.; 2]; 4])
            .seed([17; 32])
            .draws(16)
            .collect(collect)
            .sampler(SamplerConfig::Nuts(NutsOptions::new(0.05, 5).unwrap()))
            .build()
            .unwrap();
        let evidence: Vec<_> = (0..4)
            .map(|_| Mutex::new(RetainedEvidence::default()))
            .collect();
        let observer = alea::session::storage::Observer(|draw: &alea::session::OwnedDraw| {
            evidence[draw.chain.0 as usize].lock().unwrap().record(draw);
            Ok(())
        });
        let result = session
            .run(
                |_| {
                    Ok::<_, alea::model::ModelError<AssemblyError>>(
                        build_model("mean", &input).unwrap().into_target().unwrap(),
                    )
                },
                Execution::Sequential,
                &observer,
            )
            .unwrap();
        assert!(
            result
                .chains
                .iter()
                .all(|c| c.failure.is_none() && c.completed == 16)
        );
        assert!(
            result
                .chains
                .iter()
                .all(|c| c.draws.len() == if collect { 16 } else { 0 })
        );
        evidence
            .into_iter()
            .map(|e| {
                let e = e.into_inner().unwrap();
                (
                    e.hash.finish(),
                    e.steps,
                    e.divergences,
                    e.saturated,
                    e.energies,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(true), run(false));
}
