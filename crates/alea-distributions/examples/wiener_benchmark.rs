//! Operation-matched worker for tools/stan-reference/benchmark_wiener.py.
use alea_distributions::wiener::*;
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let family: usize = args[1].parse().expect("family");
    let operation = &args[2];
    let repetitions: usize = args[3].parse().expect("repetitions");
    let size: usize = args[4].parse().expect("batch size");
    assert!(matches!(family, 4 | 5 | 7) && repetitions > 0 && size > 0);
    let batch = operation.starts_with("batch");
    let observations: Vec<_> = (0..if batch { size } else { 1 })
        .map(|i| WienerObservation {
            rt: if batch {
                0.35 + 1.5 * ((i * 37) % 1000) as f64 / 1000.0
            } else {
                1.2
            },
            boundary: if batch && i % 3 == 0 {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        })
        .collect();
    let strategy = if operation == "batch_buckets" {
        BatchStrategy::BranchBuckets
    } else {
        BatchStrategy::Direct
    };
    let p4 = Wiener4Params::with_params(1.6, 0.2, 0.45, 0.7).unwrap();
    let p5 = Wiener5Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1).unwrap();
    let p7 = Wiener7Params::with_params(1.6, 0.2, 0.45, 0.7, 0.05, 0.1, 0.1).unwrap();
    let m4 = WienerModel::new(
        Wiener4,
        WienerObservations::from(observations.clone()).with_batch_strategy(strategy),
    );
    let m5 = WienerModel::new(
        Wiener5,
        WienerObservations::from(observations.clone()).with_batch_strategy(strategy),
    );
    let m7 = WienerModel::new(Wiener7, WienerObservations::from(observations.clone()));
    let mut workspace = WienerBatchWorkspace::new(size);
    let mut evaluate = || {
        let mut g = [0.0; 7];
        let lp = if batch {
            match family {
                4 => m4.batch_value_gradient_with_workspace(
                    black_box(&p4),
                    (&mut g[..4]).try_into().unwrap(),
                    &mut workspace,
                ),
                5 => m5.batch_value_gradient_with_workspace(
                    black_box(&p5),
                    (&mut g[..5]).try_into().unwrap(),
                    &mut workspace,
                ),
                _ => m7.batch_value_gradient(black_box(&p7), &mut g),
            }
        } else if operation == "logp" {
            match family {
                4 => {
                    Wiener4
                        .log_prob(black_box(&observations[0]), black_box(&p4), 1e-12)
                        .log_prob
                }
                5 => {
                    Wiener5
                        .log_prob(black_box(&observations[0]), black_box(&p5), 1e-12)
                        .log_prob
                }
                _ => {
                    Wiener7
                        .log_prob(black_box(&observations[0]), black_box(&p7), 1e-4)
                        .log_prob
                }
            }
        } else {
            assert_eq!(operation, "fused");
            match family {
                4 => {
                    let e = Wiener4.fused(black_box(&observations[0]), black_box(&p4), 1e-12);
                    g[..4].copy_from_slice(&e.grad_array());
                    e.log_prob
                }
                5 => {
                    let e = Wiener5.fused(black_box(&observations[0]), black_box(&p5), 1e-12);
                    g[..5].copy_from_slice(&e.grad_array());
                    e.log_prob
                }
                _ => {
                    let e = Wiener7.fused(black_box(&observations[0]), black_box(&p7), 1e-4);
                    g = e.grad_array();
                    e.log_prob
                }
            }
        };
        black_box((lp, g))
    };
    let (logp, scores) = evaluate();
    assert!(logp.is_finite() && scores.iter().all(|v| v.is_finite()));
    for _ in 0..5 {
        black_box(evaluate());
    }
    let start = Instant::now();
    for _ in 0..repetitions {
        black_box(evaluate());
    }
    let elapsed_ns = start.elapsed().as_nanos();
    println!(
        "{{\"family\":{family},\"operation\":{operation:?},\"repetitions\":{repetitions},\"observations\":{},\"elapsed_ns\":{elapsed_ns},\"logp\":{logp},\"scores\":{scores:?}}}",
        observations.len()
    );
}
