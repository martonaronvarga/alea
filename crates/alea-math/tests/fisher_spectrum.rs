//! Scalar spectral oracles for fitted (not manually constructed) Fisher metrics.
#![cfg(feature = "faer")]

use alea_math::{fisher::fit_low_rank, metric::EuclideanMetric};

#[test]
fn fitted_unsupported_spectrum_returns_the_conditioning_error() {
    use alea_math::{fisher::FisherFitError, metric::MetricError};
    // The untruncated solution has an eigenvalue near 1e8, beyond the
    // correction representation's conservative rounding budget in two axes.
    let result = fit_low_rank(
        &[1e8, 0.0, -1e8, 0.0, 0.0, 1.0, 0.0, -1.0],
        &[-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        &[1.0; 2],
        0.125,
        2.0,
        2,
    );
    assert!(matches!(
        result,
        Err(FisherFitError::Metric(
            MetricError::IllConditionedCorrection
        ))
    ));
}

#[test]
fn fitted_spectrum_grid_preserves_actions_rank_and_momentum_energy() {
    let mut cases = 0;
    for dim in [4_usize, 16] {
        let norm = (dim as f64).sqrt().recip();
        let directions: Vec<Vec<_>> = (0..3_usize)
            .map(|j| {
                (0..dim)
                    .map(|i| {
                        if (i & j).count_ones() % 2 == 0 {
                            norm
                        } else {
                            -norm
                        }
                    })
                    .collect()
            })
            .collect();
        let scales: Vec<_> = (0..dim).map(|i| [1e-8, 1.0, 1e8][i % 3]).collect();
        for exponent in [-3, -1, 0, 1, 3] {
            let amplitudes = [
                10_f64.powi(exponent),
                1.0,
                10_f64.powf(-0.5 * exponent as f64),
            ];
            // Independent commuting-scatter eigenvalues, before common scaling.
            let values = amplitudes.map(|a| ((2.0 * a * a + 0.125) / 2.125).sqrt());
            for common in [1e-100, 1.0, 1e100] {
                let mut q = Vec::new();
                let mut s = Vec::new();
                for j in 0..3 {
                    for sign in [-1.0, 1.0] {
                        for (i, &scale) in scales.iter().enumerate() {
                            q.push(sign * common * amplitudes[j] * directions[j][i] * scale);
                            s.push(-sign * common * directions[j][i] / scale);
                        }
                    }
                }
                for cutoff in [1.1_f64, 2.0] {
                    for cap in 0..=3 {
                        let mut retained: Vec<_> = (0..3)
                            .filter(|&j| values[j] < cutoff.recip() || values[j] > cutoff)
                            .collect();
                        retained.sort_by(|&a, &b| {
                            values[b].ln().abs().total_cmp(&values[a].ln().abs())
                        });
                        retained.truncate(cap);
                        let metric = fit_low_rank(&q, &s, &scales, 0.125 * common * common, cutoff, cap)
                            .unwrap_or_else(|error| panic!("d={dim}, e={exponent}, common={common}, cap={cap}, cutoff={cutoff}: {error}"));
                        assert_eq!(metric.rank(), retained.len());
                        // Coordinate basis vectors check both fitted and residual subspaces.
                        for j in 0..dim {
                            let mut p = vec![0.0; dim];
                            p[j] = scales[j].recip();
                            let mut actual = vec![0.0; dim];
                            metric.velocity(&p, &mut actual).unwrap();
                            for i in 0..dim {
                                let expected = f64::from(i == j)
                                    + retained
                                        .iter()
                                        .map(|&k| {
                                            (values[k] - 1.0) * directions[k][i] * directions[k][j]
                                        })
                                        .sum::<f64>();
                                let diagonal = |i: usize| {
                                    1.0 + retained
                                        .iter()
                                        .map(|&k| (values[k] - 1.0) * directions[k][i].powi(2))
                                        .sum::<f64>()
                                };
                                assert!(
                                    (actual[i] / scales[i] - expected).abs()
                                        <= 1e-8 * (diagonal(i) * diagonal(j)).sqrt(),
                                    "d={dim}, e={exponent}, common={common}, cap={cap}, cutoff={cutoff}, ({i},{j})"
                                );
                            }
                        }
                        let expected_logdet = -2.0 * scales.iter().map(|s| s.ln()).sum::<f64>()
                            - retained.iter().map(|&j| values[j].ln()).sum::<f64>();
                        assert!(
                            (metric.log_det() - expected_logdet).abs()
                                < 1e-8 * (1.0 + expected_logdet.abs())
                        );
                        let z: Vec<_> = (0..dim).map(|i| (i as f64 + 1.0) / dim as f64).collect();
                        let mut p = vec![0.0; dim];
                        let mut work = vec![0.0; dim];
                        metric.sample_momentum(&z, &mut p).unwrap();
                        let expected_energy = 0.5 * z.iter().map(|v| v * v).sum::<f64>();
                        assert!(
                            (metric.kinetic_energy(&p, &mut work).unwrap() - expected_energy).abs()
                                < 1e-8 * expected_energy
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 240);
}
