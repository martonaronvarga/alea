//! Distributional checks use independent hitting probabilities and spectral tails.
use alea_distributions::wiener::*;
use rand::{SeedableRng, rngs::SmallRng};
use std::f64::consts::PI;

fn upper_probability(a: f64, beta: f64, drift: f64) -> f64 {
    let r = 2. * a * drift;
    if r == 0. {
        beta
    } else if r > 0. {
        (-(-r * beta).exp_m1()) / (-(-r).exp_m1())
    } else {
        (r * (1. - beta)).exp() * (r * beta).exp_m1() / r.exp_m1()
    }
}

// Integral from time t to infinity of the joint time/boundary density.
// No production likelihood, sampler, or acceptance-series code is called.
fn tail(a: f64, beta: f64, drift: f64, t: f64, upper: bool) -> f64 {
    let (q, v) = if upper {
        (1. - beta, drift)
    } else {
        (beta, -drift)
    };
    let mut sum = 0.;
    for j in 1..=128 {
        let k = f64::from(j);
        let rate = 0.5 * (v * v + (k * PI / a).powi(2));
        sum += k * (k * PI * q).sin() * (-rate * t).exp() / rate;
        if j > 16 && rate * t > 80. {
            break;
        }
    }
    PI / (a * a) * (v * a * q).exp() * sum
}

fn binomial(actual: usize, n: usize, p: f64, label: &str) {
    let expected = n as f64 * p;
    let tolerance = 6. * (n as f64 * p * (1. - p)).max(0.).sqrt() + 3.;
    assert!(
        (actual as f64 - expected).abs() < tolerance,
        "{label}: observed={actual}/{n}, expected probability={p}"
    );
}

#[test]
fn joint_samples_match_choice_time_tails_and_mean_across_regimes() {
    let mut case = 0;
    for a in [0.2, 1., 3.] {
        for beta in [0.01, 0.5, 0.99] {
            for drift in [-5., 0., 5.] {
                let p = Wiener4Params::with_params(a, 0.15, beta, drift).unwrap();
                let mut rng = SmallRng::seed_from_u64(51000 + case);
                case += 1;
                let n = 8000;
                let (mut upper, mut sum, mut square) = (0, 0., 0.);
                let mut counts = [[0; 2]; 3];
                for _ in 0..n {
                    let obs = Wiener4.sample(&p, &mut rng).unwrap();
                    let t = obs.rt - p.tau;
                    assert!(t > 0. && t.is_finite());
                    upper += usize::from(obs.boundary == Boundary::Upper);
                    sum += t;
                    square += t * t;
                    for (i, s) in [0.05, 0.3, 2.].into_iter().enumerate() {
                        if t > s * a * a {
                            counts[i][usize::from(obs.boundary == Boundary::Upper)] += 1;
                        }
                    }
                }
                let probability = upper_probability(a, beta, drift);
                let label = format!("a={a} beta={beta} drift={drift}");
                binomial(upper, n, probability, &label);
                for (i, s) in [0.05, 0.3, 2.].into_iter().enumerate() {
                    for (side, is_upper) in [false, true].into_iter().enumerate() {
                        binomial(
                            counts[i][side],
                            n,
                            tail(a, beta, drift, s * a * a, is_upper),
                            &label,
                        );
                    }
                }
                let expected = if drift == 0. {
                    a * a * beta * (1. - beta)
                } else {
                    a * (probability - beta) / drift
                };
                let mean = sum / n as f64;
                let variance = (square / n as f64 - mean * mean).max(0.);
                assert!(
                    (mean - expected).abs() < 6. * (variance / n as f64).sqrt() + 1e-5 * a * a,
                    "{label}: time mean={mean}, expected={expected}"
                );
            }
        }
    }
}

#[test]
fn near_boundary_and_strong_drift_samples_stay_checked() {
    for beta in [1e-12, 1. - 1e-12] {
        for drift in [-30., 30.] {
            let p = Wiener4Params::with_params(1., 0., beta, drift).unwrap();
            let mut rng = SmallRng::seed_from_u64(64001);
            let mut upper = 0;
            for _ in 0..2000 {
                let obs = Wiener4.sample(&p, &mut rng).unwrap();
                assert!(obs.rt > 0. && obs.rt.is_finite());
                upper += usize::from(obs.boundary == Boundary::Upper);
            }
            binomial(
                upper,
                2000,
                upper_probability(1., beta, drift),
                "near boundary",
            );
        }
    }
}

#[test]
fn zero_variability_delegates_without_consuming_extra_randomness() {
    let p4 = Wiener4Params::with_params(1.2, 0.2, 0.4, 0.7).unwrap();
    let p5 = Wiener5Params::with_params(1.2, 0.2, 0.4, 0.7, 0.).unwrap();
    let p7 = Wiener7Params::with_params(1.2, 0.2, 0.4, 0.7, 0., 0., 0.).unwrap();
    let mut a = SmallRng::seed_from_u64(65001);
    let (mut b, mut c) = (a.clone(), a.clone());
    for _ in 0..1000 {
        let x = Wiener4.sample(&p4, &mut a).unwrap();
        let y = Wiener5.sample(&p5, &mut b).unwrap();
        let z = Wiener7.sample(&p7, &mut c).unwrap();
        assert_eq!(x.rt.to_bits(), y.rt.to_bits());
        assert_eq!(x.rt.to_bits(), z.rt.to_bits());
        assert_eq!(x.boundary, y.boundary);
        assert_eq!(x.boundary, z.boundary);
    }
}

#[test]
fn full_model_choices_match_independent_normal_uniform_mixture() {
    // Integrate independent drift and start distributions. Midpoint quadrature
    // on [-9,9] normal SDs, and 80 uniform nodes; refine below independently.
    fn reference(order: usize) -> f64 {
        let mut total = 0.;
        for i in 0..order {
            let z = -9. + 18. * (i as f64 + 0.5) / order as f64;
            let weight = (-0.5 * z * z).exp() / (2. * PI).sqrt() * 18. / order as f64;
            for j in 0..80 {
                let beta = 0.4 + 0.3 * ((f64::from(j) + 0.5) / 80. - 0.5);
                total += weight * upper_probability(1.2, beta, -0.7 + 1.1 * z) / 80.;
            }
        }
        total
    }
    let expected = reference(800);
    assert!((expected - reference(1600)).abs() < 1e-10);
    let p = Wiener7Params::with_params(1.2, 0.15, 0.4, -0.7, 0.3, 0.1, 1.1).unwrap();
    let mut rng = SmallRng::seed_from_u64(66001);
    fn mixture_tail(order: usize) -> [f64; 2] {
        let m = order / 10;
        let mut result = [0.; 2];
        for i in 0..order {
            let z = -9. + 18. * (i as f64 + 0.5) / order as f64;
            let weight = (-0.5 * z * z).exp() / (2. * PI).sqrt() * 18. / order as f64;
            let drift = -0.7 + 1.1 * z;
            for j in 0..m {
                let beta = 0.4 + 0.3 * ((j as f64 + 0.5) / m as f64 - 0.5);
                for k in 0..m {
                    let tau = 0.15 + 0.1 * (k as f64 + 0.5) / m as f64;
                    for (side, upper) in [false, true].into_iter().enumerate() {
                        result[side] +=
                            weight * tail(1.2, beta, drift, 0.8 - tau, upper) / (m * m) as f64;
                    }
                }
            }
        }
        result
    }
    let tails = mixture_tail(240);
    for (coarse, fine) in mixture_tail(120).into_iter().zip(tails) {
        assert!(
            (coarse - fine).abs() < 2e-5,
            "mixture quadrature unresolved"
        );
    }
    let mut upper = 0;
    let mut observed_tails = [0; 2];
    for _ in 0..20000 {
        let obs = Wiener7.sample(&p, &mut rng).unwrap();
        assert!(obs.rt > 0.15);
        upper += usize::from(obs.boundary == Boundary::Upper);
        if obs.rt > 0.8 {
            observed_tails[usize::from(obs.boundary == Boundary::Upper)] += 1;
        }
    }
    binomial(upper, 20000, expected, "full-model mixture");
    for (count, probability) in observed_tails.into_iter().zip(tails) {
        binomial(count, 20000, probability, "full-model joint RT tail");
    }
}

#[test]
fn generative_drift_calibration_matches_an_independent_conjugate_posterior() {
    // With fixed alpha/beta/tau, the drift likelihood is exp(v*D-v²*T/2).
    // A Normal(0,1) prior gives Normal(D/(1+T), 1/(1+T)). No inference engine
    // or likelihood evaluation is used to construct this posterior reference.
    let mut rng = SmallRng::seed_from_u64(67001);
    let mut ranks = [0usize; 8];
    let (mut z_sum, mut z_square, mut covered) = (0., 0., 0);
    let mut datasets = 0;
    for a in [0.5, 2.] {
        for beta in [0.2, 0.8] {
            for trials in [8, 64] {
                for _ in 0..64 {
                    use alea_core::distribution::Sample;
                    let truth = alea_distributions::scalar::Normal::new(0., 1.)
                        .unwrap()
                        .sample(&mut rng)
                        .unwrap();
                    let params = Wiener4Params::with_params(a, 0.15, beta, truth).unwrap();
                    let mut data = Vec::new();
                    let (mut displacement, mut time) = (0., 0.);
                    for _ in 0..trials {
                        let obs = Wiener4.sample(&params, &mut rng).unwrap();
                        displacement += if obs.boundary == Boundary::Upper {
                            a * (1. - beta)
                        } else {
                            -a * beta
                        };
                        time += obs.rt - params.tau;
                        data.push(obs);
                    }
                    let mean = displacement / (1. + time);
                    let sd = (1. + time).sqrt().recip();
                    let z = (truth - mean) / sd;
                    z_sum += z;
                    z_square += z * z;
                    covered += usize::from(z.abs() < 1.6448536269514722);
                    let pit = 0.5 * statrs::function::erf::erfc(-z / std::f64::consts::SQRT_2);
                    ranks[((8. * pit) as usize).min(7)] += 1;
                    datasets += 1;
                    // Bind the existing density to this independent posterior law.
                    let reference = Wiener4Params::with_params(a, 0.15, beta, 0.).unwrap();
                    let zero: f64 = data
                        .iter()
                        .map(|obs| {
                            Wiener4
                                .try_fused(obs, &reference, WienerOptions::default())
                                .unwrap()
                                .log_prob
                        })
                        .sum();
                    for v in [-2., -0.3, 0.7, 2.] {
                        let candidate = Wiener4Params::with_params(a, 0.15, beta, v).unwrap();
                        let value: f64 = data
                            .iter()
                            .map(|obs| {
                                Wiener4
                                    .try_fused(obs, &candidate, WienerOptions::default())
                                    .unwrap()
                                    .log_prob
                            })
                            .sum();
                        let expected = v * displacement - 0.5 * v * v * time;
                        assert!((value - zero - expected).abs() < 1e-8 * (1. + expected.abs()));
                    }
                }
            }
        }
    }
    for count in ranks {
        binomial(count, datasets, 0.125, "drift posterior PIT bin");
    }
    binomial(covered, datasets, 0.9, "drift posterior 90% coverage");
    assert!((z_sum / datasets as f64).abs() < 6. / (datasets as f64).sqrt());
    assert!((z_square / datasets as f64 - 1.).abs() < 6. * (2. / datasets as f64).sqrt());
}

#[test]
fn invalid_simulation_parameters_are_errors() {
    let mut rng = SmallRng::seed_from_u64(68001);
    let invalid = Wiener4Params {
        alpha: 1.,
        tau: 0.,
        beta: 0.,
        delta: 0.,
    };
    assert!(matches!(
        Wiener4.sample(&invalid, &mut rng),
        Err(ProbError::InvalidParameters(_))
    ));
    let invalid = Wiener5Params::with_params_unchecked(1., 0., 0.5, 0., -1.);
    assert!(matches!(
        Wiener5.sample(&invalid, &mut rng),
        Err(ProbError::InvalidParameters(_))
    ));
    let invalid = Wiener7Params::with_params_unchecked(1., 0., 0.1, 0., 0.3, 0.1, 0.);
    assert!(matches!(
        Wiener7.sample(&invalid, &mut rng),
        Err(ProbError::InvalidParameters(_))
    ));
}
