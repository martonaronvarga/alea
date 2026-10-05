#[path = "../benches/support/warmup.rs"]
mod workload;

use alea_core::target::LogDensityGradient;
use std::cell::Cell;
use workload::*;

#[test]
fn benchmark_targets_have_the_declared_score_and_precision() {
    for shape in SHAPES {
        for dimension in DIMENSIONS {
            let target = Target::new(dimension, shape);
            let mut q: Vec<_> = (0..dimension)
                .map(|i| (i as f64 - 3.0) / dimension as f64)
                .collect();
            let mut g = vec![0.0; dimension];
            let logp = target.logp_grad(&q, &mut g).unwrap();
            let quadratic = match shape {
                Shape::Isotropic => q.iter().map(|x| x * x).sum::<f64>(),
                Shape::Correlated => {
                    q.iter()
                        .enumerate()
                        .map(|(i, x)| [0.25, 1.0, 4.0][i % 3] * x * x)
                        .sum::<f64>()
                        + 9.0 * q.iter().sum::<f64>().powi(2) / dimension as f64
                }
            };
            assert!((logp + 0.5 * quadratic).abs() < 1e-12 * (1.0 + quadratic));
            let mut scratch = vec![0.0; dimension];
            for i in 0..dimension {
                let saved = q[i];
                q[i] = saved + 1e-5;
                let plus = target.logp_grad(&q, &mut scratch).unwrap();
                q[i] = saved - 1e-5;
                let minus = target.logp_grad(&q, &mut scratch).unwrap();
                q[i] = saved;
                assert!(((plus - minus) / 2e-5 - g[i]).abs() < 1e-7 * (1.0 + g[i].abs()));
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "native benchmark grid with exact deterministic replay")]
fn matched_warmup_workloads_complete_with_bounded_real_evaluation_counts() {
    for shape in SHAPES {
        for dimension in DIMENSIONS {
            for &method in METHODS {
                for seed in SEEDS {
                    let target = Counted {
                        target: Target::new(dimension, shape),
                        calls: Cell::new(0),
                    };
                    let result = run(&target, method, seed);
                    let calls = target.calls.get();
                    let r = result.report;
                    let case = (shape.name(), dimension, method.name(), seed);
                    assert_eq!(r.iterations, ITERATIONS, "{case:?}");
                    assert!(result.logp.is_finite(), "{case:?}");
                    assert!(r.accepted + r.divergences <= ITERATIONS, "{case:?}");
                    assert!(r.search_probes > 0 && r.metric_updates > 0, "{case:?}");
                    // Leapfrog attempts can fail before a backend entry; do not
                    // substitute the report's attempted steps for measured calls.
                    assert!(calls > ITERATIONS, "{case:?}");
                    assert!(calls <= 1 + r.integration_attempts, "{case:?}: {calls}");
                    assert!(r.integration_attempts <= r.search_probes + ITERATIONS * STEPS);
                    assert!(result.rank <= 4);
                    target.calls.set(0);
                    let replay = run(&target, method, seed);
                    assert_eq!(target.calls.get(), calls, "{case:?}");
                    assert_eq!(replay.logp.to_bits(), result.logp.to_bits(), "{case:?}");
                    assert_eq!(replay.rank, result.rank, "{case:?}");
                    assert_eq!(replay.report.step_size, r.step_size, "{case:?}");
                    assert_eq!(replay.report.accepted, r.accepted, "{case:?}");
                    assert_eq!(replay.report.divergences, r.divergences, "{case:?}");
                    // Instrumentation must not alter the target or RNG stream.
                    let uncounted = run(&target.target, method, seed);
                    assert_eq!(uncounted.logp.to_bits(), result.logp.to_bits(), "{case:?}");
                    assert_eq!(uncounted.rank, result.rank, "{case:?}");
                    assert_eq!(uncounted.report.step_size, r.step_size, "{case:?}");
                }
            }
        }
    }
}
