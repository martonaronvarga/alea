use alea::distributions::wiener::*;
use std::{
    hint::black_box,
    io::{self, BufRead},
    time::Instant,
};
fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let x: Vec<_> = line.split(',').collect();
        if x.len() != 14 {
            continue;
        }
        let id = x[0];
        let family: usize = x[1].parse().unwrap();
        let n = |i: usize| x[i].parse::<f64>().unwrap();
        let rt = n(2);
        let a = n(3);
        let tau = n(4);
        let beta = n(5);
        let v = n(6);
        let sd = n(7);
        let sw = n(8);
        let st = n(9);
        let boundary = if x[10] == "1" {
            Boundary::Lower
        } else {
            Boundary::Upper
        };
        let eps = n(11);
        let op = x[12];
        let samples: usize = x[13].parse().unwrap();
        let obs = WienerObservation { rt, boundary };
        let p4 = Wiener4Params::from_array([a, tau, beta, v]);
        let p5 = Wiener5Params::from_array([a, tau, beta, v, sd]);
        let p7 = Wiener7Params::with_params(a, tau, beta, v, sw, st, sd).unwrap();
        let batch = op.starts_with("batch");
        let size = if batch { 32 } else { 1 };
        let data: Vec<_> = (0..size)
            .map(|i| WienerObservation {
                rt: rt * (1. + 0.01 * ((i % 5) as f64)),
                boundary: if i % 2 == 0 {
                    boundary
                } else if boundary == Boundary::Upper {
                    Boundary::Lower
                } else {
                    Boundary::Upper
                },
            })
            .collect();
        let strategy = if op == "batch_bucket" {
            BatchStrategy::BranchBuckets
        } else {
            BatchStrategy::Direct
        };
        let m4 = WienerModel::new(
            Wiener4,
            WienerObservations::from(data.clone()).with_batch_strategy(strategy),
        );
        let m5 = WienerModel::new(
            Wiener5,
            WienerObservations::from(data).with_batch_strategy(strategy),
        );
        let mut workspace = WienerBatchWorkspace::new(size);
        let mut call = || -> Option<(f64, [f64; 7])> {
            let mut g = [0.; 7];
            let lp = if batch {
                match family {
                    4 => m4.batch_value_gradient_with_workspace(
                        &p4,
                        (&mut g[..4]).try_into().unwrap(),
                        &mut workspace,
                    ),
                    5 => m5.batch_value_gradient_with_workspace(
                        &p5,
                        (&mut g[..5]).try_into().unwrap(),
                        &mut workspace,
                    ),
                    _ => return None,
                }
            } else {
                match (family, op) {
                    #[cfg(feature = "cdf")]
                    (4, "cdf") => {
                        let e = Wiener4
                            .try_fused_cdf(&obs, &p4, WienerOptions::new(eps))
                            .ok()?;
                        g[..4].copy_from_slice(&e.grad_array());
                        e.log_prob
                    }
                    #[cfg(feature = "cdf")]
                    (4, "tail") => {
                        let e = Wiener4
                            .try_fused_survival(&obs, &p4, WienerOptions::new(eps))
                            .ok()?;
                        g[..4].copy_from_slice(&e.grad_array());
                        e.log_prob
                    }
                    (4, "density") => {
                        Wiener4
                            .try_log_prob(black_box(&obs), black_box(&p4), WienerOptions::new(eps))
                            .ok()?
                            .log_prob
                    }
                    (5, "density") => {
                        Wiener5
                            .try_log_prob(black_box(&obs), black_box(&p5), WienerOptions::new(eps))
                            .ok()?
                            .log_prob
                    }
                    (7, "density") => {
                        Wiener7
                            .try_log_prob(black_box(&obs), black_box(&p7), WienerOptions::new(eps))
                            .ok()?
                            .log_prob
                    }
                    (4, _) => {
                        let e = Wiener4
                            .try_fused(black_box(&obs), black_box(&p4), WienerOptions::new(eps))
                            .ok()?;
                        g[..4].copy_from_slice(&e.grad_array());
                        e.log_prob
                    }
                    (5, _) => {
                        let e = Wiener5
                            .try_fused(black_box(&obs), black_box(&p5), WienerOptions::new(eps))
                            .ok()?;
                        g[..5].copy_from_slice(&e.grad_array());
                        e.log_prob
                    }
                    (7, _) => {
                        let e = Wiener7
                            .try_fused(black_box(&obs), black_box(&p7), WienerOptions::new(eps))
                            .ok()?;
                        g = e.grad_array();
                        e.log_prob
                    }
                    _ => return None,
                }
            };
            if lp.is_finite() && g.iter().all(|v| v.is_finite()) {
                Some((lp, g))
            } else {
                None
            }
        };
        let Some((lp, g)) = call() else {
            println!("{id},{op},error");
            continue;
        };
        for _ in 0..3 {
            black_box(call());
        }
        let start = Instant::now();
        black_box(call());
        let one = start.elapsed().as_nanos().max(1);
        let repetitions = (200_000 / one).clamp(1, 512) as usize;
        let mut times = Vec::new();
        let mut ok = true;
        for _ in 0..samples {
            let start = Instant::now();
            for _ in 0..repetitions {
                if black_box(call()).is_none() {
                    ok = false;
                    break;
                }
            }
            times.push(start.elapsed().as_nanos() as f64 / repetitions as f64 / size as f64);
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        let tail = times[((times.len() - 1) as f64 * 0.95).ceil() as usize];
        print!(
            "{id},{op},{},{median},{tail},{repetitions},{lp}",
            if ok { "ok" } else { "error" }
        );
        for s in g {
            print!(",{s}");
        }
        println!();
    }
}
