//! High precision value/score lookups at extreme but valid support points.
use alea_distributions::wiener::*;

#[test]
fn extreme_support_matches_independent_high_precision_table() {
    let mut failures = Vec::new();
    for (row, line) in include_str!("support/wiener_extremes.csv")
        .lines()
        .skip(1)
        .enumerate()
    {
        let cells: Vec<_> = line.split(',').collect();
        let x: Vec<f64> = cells[..12].iter().map(|s| s.parse().unwrap()).collect();
        let boundary = if cells[12] == "lower" {
            Boundary::Lower
        } else {
            Boundary::Upper
        };
        let p = Wiener5Params::with_params(x[1], x[2], x[3], x[4], x[5]).unwrap();
        let obs = WienerObservation { rt: x[0], boundary };
        let e = Wiener5
            .try_fused(&obs, &p, WienerOptions::default())
            .unwrap();
        for (i, (actual, expected)) in std::iter::once(e.log_prob)
            .chain(e.grad_array())
            .zip(x[6..].iter().copied())
            .enumerate()
        {
            let tolerance = if i == 0 {
                1e-8 + 64. * f64::EPSILON * expected.abs()
            } else {
                2e-9 * expected.abs().max(1.)
            };
            if !actual.is_finite() || (actual - expected).abs() > tolerance {
                failures.push(format!(
                    "row={row} component={i}: {actual} vs {expected} inputs={:?}",
                    &x[..6]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn reflection_preserves_values_and_score_signs_at_both_endpoints() {
    for beta in [1e-12, 1e-8, 0.5, 1.0 - 1e-8, 1.0 - 1e-12] {
        for rt in [1e-8, 0.01, 0.1, 0.25, 1., 100.] {
            let p = Wiener5Params::with_params(1., 0., beta, -3., 0.7).unwrap();
            let lower = Wiener5
                .try_fused(
                    &WienerObservation {
                        rt,
                        boundary: Boundary::Lower,
                    },
                    &p,
                    WienerOptions::default(),
                )
                .unwrap();
            // Reflect only when the coordinate survives the round trip exactly.
            // Otherwise comparing 1-(1-beta) changes the requested model.
            let reflected_beta = 1.0 - beta;
            if 1.0 - reflected_beta != beta {
                continue;
            }
            let reflected = Wiener5Params::with_params(1., 0., reflected_beta, 3., 0.7).unwrap();
            let upper = Wiener5
                .try_fused(
                    &WienerObservation {
                        rt,
                        boundary: Boundary::Upper,
                    },
                    &reflected,
                    WienerOptions::default(),
                )
                .unwrap();
            assert!((lower.log_prob - upper.log_prob).abs() < 1e-10 * (1. + upper.log_prob.abs()));
            for (i, (a, b)) in lower
                .grad_array()
                .into_iter()
                .zip(upper.grad_array())
                .enumerate()
            {
                let sign = if i == 2 || i == 3 { -1. } else { 1. };
                assert!((a - sign * b).abs() < 1e-9 * (1. + b.abs()));
            }
        }
    }
}

#[test]
fn enormous_scaled_time_does_not_overflow_the_branch_cost_comparison() {
    let observation = WienerObservation {
        rt: 1e300,
        boundary: Boundary::Upper,
    };
    let p = Wiener4Params::with_params(1., 0., 0.5, 0.).unwrap();
    let e = Wiener4
        .try_fused(&observation, &p, WienerOptions::default())
        .unwrap();
    assert!(e.grad_array().iter().all(|x| x.is_finite()));
    let expected = std::f64::consts::PI.ln() - 0.5 * std::f64::consts::PI.powi(2) * observation.rt;
    assert!((e.log_prob - expected).abs() < 1e-14 * expected.abs());
}

#[test]
fn checked_evaluators_handle_tiny_scales_or_report_numerical_failure() {
    for sv in [0., 0.1] {
        let p = Wiener5Params::with_params(1e-150, 0., 0.5, 0., sv).unwrap();
        let obs = WienerObservation {
            rt: 1e-300,
            boundary: Boundary::Upper,
        };
        let e = Wiener5
            .try_fused(&obs, &p, WienerOptions::default())
            .unwrap();
        assert!(e.grad_array().iter().all(|x| x.is_finite()));
        let unit = Wiener5Params::with_params(1., 0., 0.5, 0., 0.).unwrap();
        let reference = Wiener5
            .try_fused(
                &WienerObservation {
                    rt: 1.,
                    boundary: Boundary::Upper,
                },
                &unit,
                WienerOptions::default(),
            )
            .unwrap();
        assert!((e.log_prob - reference.log_prob + 2. * p.to_array()[0].ln()).abs() < 1e-10);
        assert!((e.grad.alpha * p.to_array()[0] - reference.grad.alpha).abs() < 1e-10);
        assert!((e.grad.tau * obs.rt - reference.grad.tau).abs() < 1e-10);
        // Here the true time score is larger than binary64 can represent.
        let unrepresentable = Wiener5Params::with_params(1e-154, 0., 0.5, 0., sv).unwrap();
        assert!(matches!(
            Wiener5.try_fused(
                &WienerObservation {
                    rt: 1e-308,
                    boundary: Boundary::Upper
                },
                &unrepresentable,
                WienerOptions::default()
            ),
            Err(ProbError::NumericalError(_))
        ));
    }
}

#[test]
fn reflected_distance_rounded_to_one_does_not_change_support() {
    let beta = 1e-20;
    assert_eq!(1. - beta, 1.);
    let p = Wiener4Params::with_params(1., 0., beta, 0.).unwrap();
    let e = Wiener4
        .try_fused(
            &WienerObservation {
                rt: 100.,
                boundary: Boundary::Lower,
            },
            &p,
            WienerOptions::default(),
        )
        .unwrap();
    // Remaining spectral terms are suppressed by exp(-150*pi²).
    let expected = 2. * std::f64::consts::PI.ln() + beta.ln() - 50. * std::f64::consts::PI.powi(2);
    assert!((e.log_prob - expected).abs() < 1e-12);
    assert!((e.grad.beta * beta - 1.).abs() < 1e-12);
}
