//! Independent downstream model formulas, retained inference and calibration.
use alea_core::{
    distribution::Sample,
    model::prediction::{NamedOutput, posterior_predict},
    target::LogDensityGradient,
};
use alea_distributions::{
    modeling::ddm_upper_probability,
    scalar::{Bernoulli, Normal},
    wiener::{Boundary, WienerObservation},
};
use alea_downstream_conformance::modeling::{self, Observation};
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    HmcOptions, Nuts,
    adapt::{HmcWarmup, WarmupOptions},
    config::TreeDepth,
};
use alea_runtime::diagnostics::modern;
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};
const LOG_NORM: f64 = 0.9189385332046727;
fn normal(x: f64, m: f64, s: f64) -> f64 {
    -0.5 * ((x - m) / s).powi(2) - s.ln() - LOG_NORM
}
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
/// Independent long-time spectral Wiener density (no Alea density/gradient calls).
fn wiener_logp(event: WienerObservation, drift: f64) -> f64 {
    let t = event.rt - 0.15;
    let alpha = 1.2;
    let beta = 0.5;
    let mut sum = 0.0;
    for index in 1usize..=4096 {
        let n = index as f64;
        let sign = if matches!(event.boundary, Boundary::Upper) {
            if index.is_multiple_of(2) { -1.0 } else { 1.0 }
        } else {
            1.0
        };
        let exponent = -n * n * std::f64::consts::PI.powi(2) * t / (2.0 * alpha * alpha);
        sum += sign * n * (n * std::f64::consts::PI * beta).sin() * exponent.exp();
        if n > 16.0 && exponent < -45.0 {
            break;
        }
    }
    let displacement = if matches!(event.boundary, Boundary::Upper) {
        alpha * (1.0 - beta)
    } else {
        -alpha * beta
    };
    (std::f64::consts::PI / (alpha * alpha)).ln() + drift * displacement - 0.5 * drift * drift * t
        + sum.ln()
}
fn oracle(q: &[f64], data: &[Observation]) -> f64 {
    let mut value = normal(q[0], 0.0, 0.7)
        + normal(q[1], -0.7, 0.3)
        + normal(q[2], 0.0, 1.0)
        + normal(q[3], 0.0, 1.0);
    for row in data {
        let group = match row {
            Observation::Logistic { group, .. } | Observation::Ddm { group, .. } => *group,
        };
        let effect = q[0] + q[1].exp() * q[2 + group];
        value += match row {
            Observation::Logistic { event, .. } => f64::from(*event) * effect - softplus(effect),
            Observation::Ddm { event, .. } => wiener_logp(*event, effect),
        };
    }
    value
}
#[test]
fn three_downstream_models_match_independent_formulas_and_derivatives() {
    let regression_data = [(-1.0, -0.2), (0.0, 0.4), (1.0, 1.3)];
    let target = modeling::regression(&regression_data)
        .into_target()
        .unwrap();
    let q = [0.2, 0.7];
    let mut g = [0.0; 2];
    let value = target.logp_grad(&q, &mut g).unwrap();
    let expected = normal(q[0], 0.0, 1.0)
        + normal(q[1], 0.0, 1.0)
        + regression_data
            .iter()
            .map(|(x, y)| normal(*y, q[0] + q[1] * x, 0.7))
            .sum::<f64>();
    assert!((value - expected).abs() < 1e-12);
    let expected_gradient = [
        -q[0]
            + regression_data
                .iter()
                .map(|(x, y)| (y - q[0] - q[1] * x) / 0.49)
                .sum::<f64>(),
        -q[1]
            + regression_data
                .iter()
                .map(|(x, y)| x * (y - q[0] - q[1] * x) / 0.49)
                .sum::<f64>(),
    ];
    for (a, b) in g.iter().zip(expected_gradient) {
        assert!((a - b).abs() < 1e-12);
    }
    for data in [
        vec![
            Observation::Logistic {
                group: 0,
                event: true,
            },
            Observation::Logistic {
                group: 1,
                event: false,
            },
        ],
        vec![
            Observation::Ddm {
                group: 0,
                event: WienerObservation {
                    rt: 0.6,
                    boundary: Boundary::Upper,
                },
            },
            Observation::Ddm {
                group: 1,
                event: WienerObservation {
                    rt: 0.9,
                    boundary: Boundary::Lower,
                },
            },
        ],
    ] {
        let target = modeling::hierarchy(&data).into_target().unwrap();
        for q in [
            [0.2, -0.5, 0.3, -0.4],
            [-0.4, -1.0, 1.0, 0.2],
            [0.0, -0.7, 0.0, 0.0],
        ] {
            let mut g = [0.0; 4];
            let value = target.logp_grad(&q, &mut g).unwrap();
            assert!(
                (value - oracle(&q, &data)).abs() < 2e-8,
                "{value} vs {}",
                oracle(&q, &data)
            );
            for i in 0..4 {
                let mut plus = q;
                let mut minus = q;
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let fd = (oracle(&plus, &data) - oracle(&minus, &data)) / 2e-5;
                assert!(
                    (g[i] - fd).abs() < 2e-5 * (1.0 + fd.abs()),
                    "coordinate {i}: {} vs {fd}",
                    g[i]
                );
            }
        }
    }
}
fn retained(data: &[Observation], seed: u64) -> (Vec<Vec<[f64; 4]>>, usize) {
    let mut result = Vec::new();
    let mut divergences = 0;
    for chain in 0..4 {
        let target = modeling::hierarchy(data).into_target().unwrap();
        let mut rng = SmallRng::seed_from_u64(seed + chain * 10_000);
        let (hmc, _) = HmcWarmup::new(
            &target,
            OwnedBuffer::from_fn(4, |i| [0.0, -0.7, 0.0, 0.0][i]),
            HmcOptions::new(0.08, 1).unwrap(),
            WarmupOptions::new(512),
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        let mut nuts = Nuts::from_hmc(hmc, TreeDepth::try_from(7).unwrap()).unwrap();
        let mut draws = Vec::new();
        for _ in 0..512 {
            let event = nuts.step(&mut rng).unwrap();
            divergences += usize::from(matches!(
                event.termination,
                alea_mcmc::nuts::NutsTermination::Divergence(_)
            ));
            draws.push(nuts.point().position().try_into().unwrap());
        }
        result.push(draws);
    }
    (result, divergences)
}
#[test]
fn hierarchical_inference_diagnostics_and_prediction() {
    for (seed, data) in [
        (
            14020,
            vec![
                Observation::Logistic {
                    group: 0,
                    event: true,
                },
                Observation::Logistic {
                    group: 1,
                    event: false,
                },
            ],
        ),
        (
            14021,
            vec![
                Observation::Ddm {
                    group: 0,
                    event: WienerObservation {
                        rt: 0.6,
                        boundary: Boundary::Upper,
                    },
                },
                Observation::Ddm {
                    group: 1,
                    event: WienerObservation {
                        rt: 0.9,
                        boundary: Boundary::Lower,
                    },
                },
            ],
        ),
    ] {
        let (draws, divergences) = retained(&data, seed);
        for i in 0..4 {
            let chains: Vec<Vec<f64>> = draws
                .iter()
                .map(|c| c.iter().map(|q| q[i]).collect())
                .collect();
            let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
            let d = modern::summarize(&refs).unwrap();
            assert!(d.rhat < 1.05 && d.bulk_ess > 100.0, "{d:?}");
            println!(
                "model={seed} coordinate={i} rhat={} bulk={} tail={} divergences={divergences}",
                d.rhat, d.bulk_ess, d.tail_ess
            );
        }
        let model = modeling::hierarchy(&data);
        let coordinates = model.coordinates();
        let mut rng = SmallRng::seed_from_u64(seed + 100);
        let prediction = posterior_predict(
            model.layout(),
            draws.iter().flatten().map(|q| q.as_slice()),
            &mut rng,
            |p, rng| {
                let drift = p[0] + p[1] * p[2];
                let probability = if matches!(data[0], Observation::Ddm { .. }) {
                    ddm_upper_probability(1.2, 0.5, drift)?
                } else {
                    1.0 / (1.0 + (-drift).exp())
                };
                Ok::<_, alea_core::distribution::DistributionError>((
                    probability,
                    Bernoulli::new(probability)?.sample(rng)?,
                ))
            },
        )
        .unwrap();
        let expected = prediction.generated.iter().map(|x| x.0).sum::<f64>()
            / prediction.generated.len() as f64;
        let actual = prediction.generated.iter().filter(|x| x.1).count() as f64
            / prediction.generated.len() as f64;
        assert!((actual - expected).abs() < 0.05);
        let named = NamedOutput::new(coordinates, model.layout().clone(), prediction).unwrap();
        assert_eq!(named.blocks().len(), 2);
    }
}
/// Independently integrated spectral tail of joint first-passage time/boundary.
fn ddm_tail(t: f64, drift: f64, upper: bool) -> f64 {
    let a = 1.2;
    let mut sum = 0.0;
    for index in 1usize..=4096 {
        let n = index as f64;
        let rate = 0.5 * (drift * drift + (n * std::f64::consts::PI / a).powi(2));
        let sign = if upper && index.is_multiple_of(2) {
            -1.0
        } else {
            1.0
        };
        sum += sign * n * (n * std::f64::consts::PI * 0.5).sin() * (-rate * t).exp() / rate;
        if n > 16.0 && rate * t > 45.0 {
            break;
        }
    }
    std::f64::consts::PI / (a * a)
        * (if upper {
            drift * a * 0.5
        } else {
            -drift * a * 0.5
        })
        .exp()
        * sum
}
fn ddm_sample<R: Rng + ?Sized>(drift: f64, rng: &mut R) -> WienerObservation {
    let p = ddm_upper_probability(1.2, 0.5, drift).unwrap();
    let upper = rng.random::<f64>() < p;
    let mass = if upper { p } else { 1.0 - p };
    let tail = mass * (1.0 - rng.random::<f64>());
    let mut hi = 1.0;
    for _ in 0..32 {
        if ddm_tail(hi, drift, upper) < tail {
            break;
        }
        hi *= 2.0;
    }
    let mut lo = 1e-5;
    assert!(
        ddm_tail(hi, drift, upper) < tail,
        "reference RT upper bracket exhausted"
    );
    assert!(
        ddm_tail(lo, drift, upper) > tail,
        "reference RT below declared lower bracket"
    );
    for _ in 0..64 {
        let mid = (lo + hi) * 0.5;
        if ddm_tail(mid, drift, upper) > tail {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    WienerObservation {
        rt: 0.15 + 0.5 * (hi + lo),
        boundary: if upper {
            Boundary::Upper
        } else {
            Boundary::Lower
        },
    }
}
#[test]
fn posterior_sbc_for_both_hierarchical_verticals() {
    for (ddm, seed) in [(false, 14100), (true, 14200)] {
        let mut bins = [[0usize; 4]; 4];
        let mut divergences = 0;
        for dataset in 0..32 {
            let mut rng = SmallRng::seed_from_u64(seed + dataset);
            let truth = [
                Normal::new(0.0, 0.7).unwrap().sample(&mut rng).unwrap(),
                Normal::new(-0.7, 0.3).unwrap().sample(&mut rng).unwrap(),
                Normal::new(0.0, 1.0).unwrap().sample(&mut rng).unwrap(),
                Normal::new(0.0, 1.0).unwrap().sample(&mut rng).unwrap(),
            ];
            let mut data = Vec::new();
            for group in 0..2 {
                let drift = truth[0] + truth[1].exp() * truth[2 + group];
                for _ in 0..2 {
                    data.push(if ddm {
                        Observation::Ddm {
                            group,
                            event: ddm_sample(drift, &mut rng),
                        }
                    } else {
                        Observation::Logistic {
                            group,
                            event: Bernoulli::new(1.0 / (1.0 + (-drift).exp()))
                                .unwrap()
                                .sample(&mut rng)
                                .unwrap(),
                        }
                    });
                }
            }
            let (draws, d) = retained(&data, seed + dataset + 100_000);
            divergences += d;
            for i in 0..4 {
                let below = draws.iter().flatten().filter(|q| q[i] < truth[i]).count();
                let bin = (below * 4 / 2049).min(3);
                bins[i][bin] += 1;
            }
        }
        for (i, counts) in bins.iter().enumerate() {
            let chi = counts
                .iter()
                .map(|n| (*n as f64 - 8.0).powi(2) / 8.0)
                .sum::<f64>();
            println!(
                "SBC ddm={ddm} coordinate={i} bins={counts:?} chi={chi} divergences={divergences}"
            );
            assert!(chi < 16.3, "ddm={ddm} coordinate={i} counts={counts:?}");
        }
    }
}
#[test]
fn conjugate_normal_sbc_and_predictive_check() {
    let mut bins = [0usize; 4];
    for dataset in 0..64 {
        let mut rng = SmallRng::seed_from_u64(14050 + dataset);
        let truth = Normal::new(0.0, 1.0).unwrap().sample(&mut rng).unwrap();
        let ys = (0..4)
            .map(|_| Normal::new(truth, 1.0).unwrap().sample(&mut rng).unwrap())
            .collect::<Vec<_>>();
        let mean = ys.iter().sum::<f64>() / 5.0;
        let posterior = Normal::new(mean, 5.0f64.sqrt().recip()).unwrap();
        let below = (0..512)
            .filter(|_| posterior.sample(&mut rng).unwrap() < truth)
            .count();
        bins[(below * 4 / 513).min(3)] += 1;
    }
    let chi = bins
        .iter()
        .map(|n| (*n as f64 - 16.0).powi(2) / 16.0)
        .sum::<f64>();
    println!("normal SBC bins={bins:?} chi={chi}");
    assert!(chi < 16.3);
}
