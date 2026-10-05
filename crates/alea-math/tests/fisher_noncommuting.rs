//! Constructive G F G = C oracles: no eigensolver or matrix square-root oracle.
#![cfg(feature = "faer")]

use alea_math::{fisher::fit_low_rank, metric::EuclideanMetric};

const RIDGE: f64 = 1e-6;

// Two-dimensional active subspace, embedded in non-coordinate directions.
// Return paired +/- rows with exactly the requested scatter before rounding.
fn rows(scatter: [[f64; 2]; 2], directions: &[Vec<f64>; 2]) -> Vec<f64> {
    let l00 = scatter[0][0].sqrt();
    let l10 = scatter[1][0] / l00;
    let l11 = (scatter[1][1] - l10 * l10).sqrt();
    assert!(l00.is_finite() && l11.is_finite() && l11 > 0.0);
    let columns = [[l00, l10], [0.0, l11]];
    let mut data = Vec::new();
    for column in columns {
        for sign in [-1.0, 1.0] {
            for (u, v) in directions[0].iter().zip(&directions[1]) {
                data.push(sign * std::f64::consts::FRAC_1_SQRT_2 * (column[0] * u + column[1] * v));
            }
        }
    }
    // Validate the generated input, rather than trusting the Cholesky helper.
    for i in 0..2 {
        for j in 0..2 {
            let actual = data
                .chunks_exact(directions[0].len())
                .map(|row| {
                    let dot = |axis: usize| {
                        row.iter()
                            .zip(&directions[axis])
                            .map(|(x, u)| x * u)
                            .sum::<f64>()
                    };
                    dot(i) * dot(j)
                })
                .sum::<f64>();
            assert!(
                (actual - scatter[i][j]).abs() < 1e-12 * (scatter[i][i] * scatter[j][j]).sqrt()
            );
        }
    }
    data
}

#[test]
fn noncommuting_scatter_grid_recovers_planted_metrics_and_truncation() {
    let mut cases = 0;
    for dim in [4_usize, 16, 64] {
        let norm = (dim as f64).sqrt().recip();
        let directions = [
            vec![norm; dim],
            (0..dim)
                .map(|i| if i % 2 == 0 { norm } else { -norm })
                .collect(),
        ];
        let scales: Vec<_> = (0..dim).map(|i| [1e-8, 1.0, 1e8][i % 3]).collect();
        for values in [[0.25_f64, 3.0], [0.05, 5.0]] {
            for condition in [4.0_f64, 64.0, 1024.0] {
                // F is rotated by 45 degrees relative to G=diag(values).
                let f = [
                    [(condition + 1.0) / 2.0, (condition - 1.0) / 2.0],
                    [(condition - 1.0) / 2.0, (condition + 1.0) / 2.0],
                ];
                let c: [[f64; 2]; 2] = std::array::from_fn(|i| {
                    std::array::from_fn(|j| values[i] * f[i][j] * values[j])
                });
                assert!(
                    (f[0][1] * (c[0][0] - c[1][1])).abs() > 1.0,
                    "must not commute"
                );
                // Subtract the very ridge which the fitter adds back. Off the
                // active subspace C=F=ridge*I, hence the exact solution is I.
                let raw = |matrix: [[f64; 2]; 2]| {
                    std::array::from_fn(|i| {
                        std::array::from_fn(|j| matrix[i][j] - if i == j { RIDGE } else { 0.0 })
                    })
                };
                let position_rows = rows(raw(c), &directions);
                let score_rows = rows(raw(f), &directions);
                for common in [2.0_f64.powi(-510), 1e-100, 1.0, 1e100, 2.0_f64.powi(510)] {
                    let positions: Vec<_> = position_rows
                        .iter()
                        .enumerate()
                        .map(|(i, x)| common * x * scales[i % dim])
                        .collect();
                    let scores: Vec<_> = score_rows
                        .iter()
                        .enumerate()
                        .map(|(i, x)| -common * x / scales[i % dim])
                        .collect();
                    for cutoff in [1.1_f64, 10.0] {
                        for cap in 0..=2 {
                            let mut retained: Vec<_> = (0..2)
                                .filter(|&i| values[i] < cutoff.recip() || values[i] > cutoff)
                                .collect();
                            retained.sort_by(|&i, &j| {
                                values[j].ln().abs().total_cmp(&values[i].ln().abs())
                            });
                            retained.truncate(cap);
                            let context = format!(
                                "d={dim}, G={values:?}, cond(F)={condition}, common={common}, cutoff={cutoff}, cap={cap}"
                            );
                            let metric = fit_low_rank(
                                &positions,
                                &scores,
                                &scales,
                                RIDGE * common * common,
                                cutoff,
                                cap,
                            )
                            .unwrap_or_else(|error| panic!("{context}: {error}"));
                            assert_eq!(metric.rank(), retained.len(), "{context}");
                            let diagonal = |i: usize| {
                                1.0 + retained
                                    .iter()
                                    .map(|&k| (values[k] - 1.0) * directions[k][i].powi(2))
                                    .sum::<f64>()
                            };
                            // Ambient entries approach identity as d grows.
                            // Check active directions too, so their error cannot
                            // be hidden by dilution across 64 coordinates.
                            let eigenvalue = |i: usize| {
                                if retained.contains(&i) {
                                    values[i]
                                } else {
                                    1.0
                                }
                            };
                            for j in 0..2 {
                                let input: Vec<_> = directions[j]
                                    .iter()
                                    .zip(&scales)
                                    .map(|(u, s)| u / s)
                                    .collect();
                                let mut actual = vec![0.0; dim];
                                metric.velocity(&input, &mut actual).unwrap();
                                for (i, direction) in directions.iter().enumerate() {
                                    let projection: f64 = actual
                                        .iter()
                                        .zip(&scales)
                                        .zip(direction)
                                        .map(|((v, s), u)| v / s * u)
                                        .sum();
                                    let expected = if i == j { eigenvalue(i) } else { 0.0 };
                                    assert!(
                                        (projection - expected).abs()
                                            < 1e-8 * (eigenvalue(i) * eigenvalue(j)).sqrt(),
                                        "{context}, projected ({i},{j}): {projection} != {expected}"
                                    );
                                }
                            }
                            for j in 0..dim {
                                let mut input = vec![0.0; dim];
                                input[j] = scales[j].recip();
                                let mut actual = vec![0.0; dim];
                                metric.velocity(&input, &mut actual).unwrap();
                                for i in 0..dim {
                                    let expected = f64::from(i == j)
                                        + retained
                                            .iter()
                                            .map(|&k| {
                                                (values[k] - 1.0)
                                                    * directions[k][i]
                                                    * directions[k][j]
                                            })
                                            .sum::<f64>();
                                    assert!(
                                        (actual[i] / scales[i] - expected).abs()
                                            < 1e-8 * (diagonal(i) * diagonal(j)).sqrt(),
                                        "{context}, ({i},{j}): {} != {expected}",
                                        actual[i] / scales[i]
                                    );
                                }
                            }
                            let expected_logdet = -2.0 * scales.iter().map(|x| x.ln()).sum::<f64>()
                                - retained.iter().map(|&i| values[i].ln()).sum::<f64>();
                            assert!(
                                (metric.log_det() - expected_logdet).abs()
                                    < 1e-8 * (1.0 + expected_logdet.abs()),
                                "{context}: determinant"
                            );
                            let noise: Vec<_> =
                                (0..dim).map(|i| (i as f64 + 1.0) / dim as f64).collect();
                            let mut momentum = vec![0.0; dim];
                            let mut work = vec![0.0; dim];
                            metric.sample_momentum(&noise, &mut momentum).unwrap();
                            let expected_energy = 0.5 * noise.iter().map(|x| x * x).sum::<f64>();
                            assert!(
                                (metric.kinetic_energy(&momentum, &mut work).unwrap()
                                    - expected_energy)
                                    .abs()
                                    < 1e-8 * expected_energy,
                                "{context}: momentum energy"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 540);
}

// A diagonal scatter plus a dense outer product couples all four active axes.
// Construct its samples directly, independently of any decomposition routine.
fn coupled_rows(values: [f64; 4], coupling: f64, directions: &[Vec<f64>]) -> Vec<f64> {
    let mut result = Vec::with_capacity(10 * directions[0].len());
    for column in 0..5 {
        let coefficients: [f64; 4] = std::array::from_fn(|i| {
            if column == 4 {
                values[i] * coupling.sqrt() * [1.0, -2.0, 0.5, 3.0][i]
            } else if column == i {
                (values[i].powi(2) * (i + 1) as f64 - RIDGE).sqrt()
            } else {
                0.0
            }
        });
        for sign in [-1.0, 1.0] {
            for coordinate in 0..directions[0].len() {
                result.push(
                    sign * std::f64::consts::FRAC_1_SQRT_2
                        * coefficients
                            .iter()
                            .zip(directions)
                            .map(|(c, u)| c * u[coordinate])
                            .sum::<f64>(),
                );
            }
        }
    }
    // Check every active scatter entry before passing samples to the fitter.
    for i in 0..4 {
        for j in 0..4 {
            let actual = result
                .chunks_exact(directions[0].len())
                .map(|row| {
                    let dot = |axis: usize| {
                        row.iter()
                            .zip(&directions[axis])
                            .map(|(x, u)| x * u)
                            .sum::<f64>()
                    };
                    dot(i) * dot(j)
                })
                .sum::<f64>();
            let expected = values[i]
                * values[j]
                * coupling
                * [1.0, -2.0, 0.5, 3.0][i]
                * [1.0, -2.0, 0.5, 3.0][j]
                + if i == j {
                    values[i].powi(2) * (i + 1) as f64 - RIDGE
                } else {
                    0.0
                };
            assert!((actual - expected).abs() < 1e-12 * (1.0 + expected.abs()));
        }
    }
    result
}

#[test]
fn four_coupled_directions_preserve_active_and_complement_actions_after_reordering() {
    let values = [0.1_f64, 0.4, 2.0, 6.0];
    let mut cases = 0;
    for dim in [8_usize, 32] {
        let norm = (dim as f64).sqrt().recip();
        // Complete Walsh basis: the remaining axes probe the untouched subspace.
        let directions: Vec<Vec<f64>> = (0..dim)
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
        for coupling in [0.5, 8.0, 128.0] {
            // F=D+coupling*v*v^T and C=GFG, with G=diag(values).
            // Both are SPD and do not commute; all off-diagonal entries couple.
            let position_rows = coupled_rows(values, coupling, &directions[..4]);
            let score_rows = coupled_rows([1.0; 4], coupling, &directions[..4]);
            for common in [2.0_f64.powi(-510), 1e-100, 1.0, 1e100, 2.0_f64.powi(510)] {
                for reversed in [false, true] {
                    let scaled = |data: &[f64], score: bool| {
                        (0..10)
                            .flat_map(|row| {
                                let row = if reversed { 9 - row } else { row };
                                data[row * dim..(row + 1) * dim].iter().zip(&scales).map(
                                    move |(x, s)| {
                                        if score {
                                            -common * x / s
                                        } else {
                                            common * x * s
                                        }
                                    },
                                )
                            })
                            .collect::<Vec<_>>()
                    };
                    let positions = scaled(&position_rows, false);
                    let scores = scaled(&score_rows, true);
                    for threshold in [1.1_f64, 3.0, 20.0] {
                        for cap in 0..=4 {
                            let mut retained: Vec<_> = (0..4)
                                .filter(|&i| values[i] < threshold.recip() || values[i] > threshold)
                                .collect();
                            retained.sort_by(|&i, &j| {
                                values[j].ln().abs().total_cmp(&values[i].ln().abs())
                            });
                            retained.truncate(cap);
                            let context = format!(
                                "d={dim}, coupling={coupling}, common={common}, reversed={reversed}, threshold={threshold}, cap={cap}"
                            );
                            let metric = fit_low_rank(
                                &positions,
                                &scores,
                                &scales,
                                RIDGE * common * common,
                                threshold,
                                cap,
                            )
                            .unwrap_or_else(|error| panic!("{context}: {error}"));
                            assert_eq!(metric.rank(), retained.len(), "{context}");
                            let eigenvalue = |i: usize| {
                                if retained.contains(&i) {
                                    values[i]
                                } else {
                                    1.0
                                }
                            };
                            for (j, direction) in directions.iter().enumerate() {
                                let input: Vec<_> =
                                    direction.iter().zip(&scales).map(|(u, s)| u / s).collect();
                                let mut actual = vec![0.0; dim];
                                metric.velocity(&input, &mut actual).unwrap();
                                for (i, axis) in directions.iter().enumerate() {
                                    let projection: f64 = actual
                                        .iter()
                                        .zip(&scales)
                                        .zip(axis)
                                        .map(|((v, s), u)| v / s * u)
                                        .sum();
                                    let expected = if i == j { eigenvalue(i) } else { 0.0 };
                                    assert!(
                                        (projection - expected).abs()
                                            < 1e-8 * (eigenvalue(i) * eigenvalue(j)).sqrt(),
                                        "{context}, ({i},{j}): {projection} != {expected}"
                                    );
                                }
                                // Unit noise along each eigenvector has energy 1/2,
                                // including every untouched complementary axis.
                                let mut momentum = vec![0.0; dim];
                                metric.sample_momentum(direction, &mut momentum).unwrap();
                                assert!(
                                    (metric.kinetic_energy(&momentum, &mut actual).unwrap() - 0.5)
                                        .abs()
                                        < 5e-9,
                                    "{context}, axis={j}: momentum energy"
                                );
                            }
                            let expected_logdet = -2.0 * scales.iter().map(|s| s.ln()).sum::<f64>()
                                - retained.iter().map(|&i| values[i].ln()).sum::<f64>();
                            assert!(
                                (metric.log_det() - expected_logdet).abs()
                                    < 1e-8 * (1.0 + expected_logdet.abs()),
                                "{context}: determinant"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 900);
}
