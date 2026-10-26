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
            if !actual.is_finite() || (actual - expected).abs() > 2e-8 * expected.abs().max(1.) {
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
