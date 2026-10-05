use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, LowRankDiagonalMetric},
};
fn buf(v: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(v.len(), |i| v[i])
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-10 * (1.0 + b.abs()), "{a} != {b}");
}
fn metric() -> LowRankDiagonalMetric {
    let u = 0.5_f64.sqrt();
    LowRankDiagonalMetric::new(
        buf(&[2.0, 0.5, 3.0]),
        buf(&[u, u, 0.0, u, -u, 0.0]),
        buf(&[0.25, 4.0]),
    )
    .unwrap()
}
#[test]
#[allow(clippy::needless_range_loop)] // Matrix-index oracle intentionally mirrors A A^T G.
fn contracting_and_expanding_directions_match_dense_oracle() {
    let m = metric();
    let g = [[8.5, -1.875, 0.0], [-1.875, 0.53125, 0.0], [0.0, 0.0, 9.0]];
    let mut out = [0.0; 3];
    for p in [[0.2, -0.7, 1.2], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        m.velocity(&p, &mut out).unwrap();
        for i in 0..3 {
            close(out[i], g[i].iter().zip(p).map(|(a, b)| a * b).sum());
        }
        let mut sampled = [0.0; 3];
        m.sample_momentum(&p, &mut sampled).unwrap();
        close(
            m.kinetic_energy(&sampled, &mut out).unwrap(),
            p.iter().map(|v| v * v).sum::<f64>() / 2.0,
        );
    }
    close(m.log_det(), -2.0 * 3.0_f64.ln());
    // Reconstruct A A^T from the momentum map and check (A A^T) G = I.
    let mut a = [[0.0; 3]; 3];
    for j in 0..3 {
        let mut e = [0.0; 3];
        e[j] = 1.0;
        m.sample_momentum(&e, &mut out).unwrap();
        for i in 0..3 {
            a[i][j] = out[i];
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let product: f64 = (0..3)
                .map(|k| (0..3).map(|l| a[i][l] * a[k][l]).sum::<f64>() * g[k][j])
                .sum();
            close(product, if i == j { 1.0 } else { 0.0 });
        }
    }
}
#[test]
fn rank_zero_validation_and_failed_updates_are_atomic() {
    let mut m = LowRankDiagonalMetric::new(buf(&[2.0, 0.5]), buf(&[]), buf(&[])).unwrap();
    let mut out = [11.0; 2];
    m.velocity(&[1.0, 1.0], &mut out).unwrap();
    assert_eq!(out, [4.0, 0.25]);
    for scales in [&[0.0, 1.0][..], &[f64::NAN, 1.0], &[1.0], &[1e300, 1.0]] {
        assert!(m.set_scales(scales).is_err());
        assert_eq!(m.scales(), &[2.0, 0.5]);
    }
    out.fill(11.0);
    assert!(m.velocity(&[1.0], &mut out).is_err());
    assert_eq!(out, [11.0; 2]);
    assert!(LowRankDiagonalMetric::new(buf(&[1.0, 1.0]), buf(&[1.0, 1.0]), buf(&[1.0])).is_err());
    assert!(LowRankDiagonalMetric::new(buf(&[1.0]), buf(&[1.0]), buf(&[-1.0])).is_err());
    assert!(LowRankDiagonalMetric::new(buf(&[1.0]), buf(&[f64::NAN]), buf(&[1.0])).is_err());
}

#[test]
fn correction_rejects_spectra_that_destroy_positive_definiteness_in_f64() {
    // Before the guard, lambda-1 rounded to -1 and velocity([1]) became [0].
    assert!(LowRankDiagonalMetric::new(buf(&[1.0]), buf(&[1.0]), buf(&[1e-20])).is_err());
    // The inverse square-root correction also rounds to -1 for huge lambda,
    // turning a standard-normal momentum into an identically zero vector.
    assert!(LowRankDiagonalMetric::new(buf(&[1.0]), buf(&[1.0]), buf(&[1e40])).is_err());
    // A basis accepted by the old absolute 1e-10 check made this G negative.
    assert!(LowRankDiagonalMetric::new(buf(&[1.0]), buf(&[1.0 + 4e-11]), buf(&[1e-12])).is_err());
}

#[test]
fn spectral_grid_preserves_momentum_energy_and_log_determinant() {
    // A fixed orthonormal basis and an exponent grid give reproducible coverage
    // of contractions, expansions, truncation and full rank without tuning seeds.
    let basis = [0.6, 0.8, 0.0, -0.8, 0.6, 0.0, 0.0, 0.0, 1.0];
    for rank in 0..=3 {
        for exponent in -3..=3 {
            let values: Vec<_> = (0..rank)
                .map(|i| 10.0_f64.powi(if i % 2 == 0 { exponent } else { -exponent }))
                .collect();
            let m = LowRankDiagonalMetric::new(
                buf(&[0.125, 3.0, 16.0]),
                buf(&basis[..3 * rank]),
                buf(&values),
            )
            .unwrap();
            for z in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, -0.7, 1.2]] {
                let mut p = [0.0; 3];
                let mut velocity = [0.0; 3];
                m.sample_momentum(&z, &mut p).unwrap();
                let energy = m.kinetic_energy(&p, &mut velocity).unwrap();
                let expected = 0.5 * z.iter().map(|v| v * v).sum::<f64>();
                assert!(energy > 0.0);
                assert!(
                    (energy - expected).abs() < 1e-8 * expected,
                    "rank {rank}, exponent {exponent}: {energy} != {expected}"
                );
            }
            close(
                m.log_det(),
                -2.0 * 6.0_f64.ln() - values.iter().map(|v| v.ln()).sum::<f64>(),
            );
        }
    }
}

#[test]
fn all_low_rank_operations_reject_shapes_before_writing() {
    let m = metric();
    for input_len in 0..=4 {
        for output_len in 0..=4 {
            if input_len == 3 && output_len == 3 {
                continue;
            }
            let input = vec![1.0; input_len];
            let mut output = vec![123.0; output_len];
            assert!(m.velocity(&input, &mut output).is_err());
            assert!(m.sample_momentum(&input, &mut output).is_err());
            assert!(m.kinetic_energy(&input, &mut output).is_err());
            assert!(output.iter().all(|&v| v == 123.0)); // Exact sentinel, not a numerical approximation.
        }
    }
}
#[cfg(not(miri))]
#[test]
fn low_rank_hot_operations_allocate_nothing() {
    let m = metric();
    let mut out = [0.0; 3];
    let measured = allocation_counter::measure(|| {
        for _ in 0..100 {
            m.velocity(&[1.0; 3], &mut out).unwrap();
            m.sample_momentum(&[1.0; 3], &mut out).unwrap();
            std::hint::black_box(m.kinetic_energy(&[1.0; 3], &mut out).unwrap());
        }
    });
    assert_eq!(measured.count_total, 0);
}
#[cfg(feature = "faer")]
#[test]
fn fisher_fit_satisfies_dense_riccati_equation_and_handles_degenerate_data() {
    use alea_math::fisher::fit_low_rank;
    // Full rank paired data, C and F independently formed as raw centered scatters.
    let q = [-2.0, 1.0, 1.0, 3.0, 2.0, -1.0, -1.0, -3.0];
    let s = [1.0, -2.0, -3.0, -1.0, -1.0, 2.0, 3.0, 1.0];
    let ridge = 1e-5;
    let m = fit_low_rank(&q, &s, &[1.0, 1.0], ridge, 1.0, 2).unwrap();
    let mut g = [[0.0; 2]; 2];
    let mut v = [0.0; 2];
    for j in 0..2 {
        let mut e = [0.0; 2];
        e[j] = 1.0;
        m.velocity(&e, &mut v).unwrap();
        for i in 0..2 {
            g[i][j] = v[i];
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let c = (0..4).map(|n| q[2 * n + i] * q[2 * n + j]).sum::<f64>()
                + if i == j { ridge } else { 0.0 };
            let gfg: f64 = (0..2)
                .flat_map(|k| {
                    (0..2).map(move |l| {
                        g[i][k]
                            * ((0..4).map(|n| s[2 * n + k] * s[2 * n + l]).sum::<f64>()
                                + if k == l { ridge } else { 0.0 })
                            * g[l][j]
                    })
                })
                .sum();
            close(gfg, c);
        }
    }
    let empty = fit_low_rank(&[1.0; 8], &[0.0; 8], &[2.0, 3.0], ridge, 2.0, 2).unwrap();
    assert_eq!(empty.rank(), 0);
    let capped = fit_low_rank(&q, &s, &[1.0, 1.0], ridge, 1.0, 1).unwrap();
    assert_eq!(capped.rank(), 1);
    assert!(fit_low_rank(&[f64::NAN; 8], &s, &[1.0, 1.0], ridge, 1.0, 2).is_err());
}

#[cfg(feature = "faer")]
#[test]
fn fisher_fit_matches_independent_high_precision_dense_fixtures() {
    let csv = include_str!("fixtures/fisher-dense.csv");
    let mut cases = 0;
    for line in csv.lines().filter(|line| !line.starts_with('#')).skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 13);
        let vector = |text: &str| {
            text.split(';')
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let number = |index: usize| fields[index].parse::<f64>().unwrap();
        let m = alea_math::fisher::fit_low_rank(
            &vector(fields[1]),
            &vector(fields[2]),
            &[number(3), number(4)],
            number(5),
            number(6),
            fields[7].parse().unwrap(),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", fields[0]));
        assert_eq!(
            m.rank(),
            fields[12].parse::<usize>().unwrap(),
            "{}",
            fields[0]
        );
        let expected = [[number(8), number(9)], [number(9), number(10)]];
        for j in 0..2 {
            let mut unit = [0.0; 2];
            unit[j] = 1.0;
            let mut actual = [0.0; 2];
            m.velocity(&unit, &mut actual).unwrap();
            for i in 0..2 {
                let scale = (expected[i][i] * expected[j][j]).sqrt();
                assert!(
                    (actual[i] - expected[i][j]).abs() < 1e-8 * scale,
                    "{} ({i},{j}): {} != {}",
                    fields[0],
                    actual[i],
                    expected[i][j]
                );
            }
        }
        assert!(
            (m.log_det() - number(11)).abs() < 1e-8 * (1.0 + number(11).abs()),
            "{}",
            fields[0]
        );
        cases += 1;
    }
    assert_eq!(cases, 7);
}

#[cfg(feature = "faer")]
#[test]
fn unrepresentable_fisher_scatter_is_a_typed_error_not_silent_identity() {
    use alea_math::fisher::{FisherFitError, fit_low_rank};
    let values = [-1e200, 0.0, 1e200, 0.0];
    assert!(matches!(
        fit_low_rank(&values, &values, &[1.0; 2], 1e-5, 2.0, 2),
        Err(FisherFitError::Numerical)
    ));
}

#[cfg(feature = "faer")]
#[test]
fn fisher_common_scaling_preserves_subnormal_and_overflowing_scatters() {
    use alea_math::fisher::fit_low_rank;

    // In direction u=(1,1,0)/sqrt(2), C=4*a*a+1 and F=4+1.
    // The orthogonal complement has C=F=1. This oracle uses only scalar
    // arithmetic, not a fitted metric at a less extreme scale.
    for (a, exponent) in [4.0_f64, 3.25]
        .into_iter()
        .flat_map(|a| [-537, -535, -530, -510, 0, 500, 510].map(|exponent| (a, exponent)))
    {
        let lambda = ((4.0 * a * a + 1.0) / 5.0).sqrt();
        let amplitude = 2.0_f64.powi(exponent);
        let ridge = amplitude * amplitude;
        assert!(ridge.is_finite() && ridge > 0.0);
        let positions = [-a, -a, 0.0, a, a, 0.0].map(|x| x * amplitude);
        let scores = [-1.0, -1.0, 0.0, 1.0, 1.0, 0.0].map(|x| x * amplitude);
        let metric = fit_low_rank(&positions, &scores, &[1.0; 3], ridge, 1.1, 3)
            .unwrap_or_else(|error| panic!("a={a}, exponent {exponent}: {error:?}"));
        assert_eq!(metric.rank(), 1);
        let mut out = [0.0; 3];
        for (input, expected) in [
            ([1.0, 1.0, 0.0], [lambda, lambda, 0.0]),
            ([1.0, -1.0, 0.0], [1.0, -1.0, 0.0]),
            ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ] {
            metric.velocity(&input, &mut out).unwrap();
            for (actual, expected) in out.into_iter().zip(expected) {
                close(actual, expected);
            }
        }
        close(metric.log_det(), -lambda.ln());
    }
}

#[cfg(feature = "faer")]
#[test]
fn high_dimensional_fisher_preserves_known_subspace_and_rank_budget() {
    // Two orthonormal, non-axis-aligned directions embedded in a larger space.
    // Their independent scalar scatter ratios determine the exact eigenvalues;
    // every orthogonal direction must remain identity in standardized coordinates.
    let ridge = 1e-5;
    let lambda = [
        ((18.0_f64 + ridge) / (0.5 + ridge)).sqrt(),
        ((1.125_f64 + ridge) / (4.5 + ridge)).sqrt(),
    ];
    for dim in [8, 32, 128] {
        let norm = 1.0 / (dim as f64).sqrt();
        let directions = [
            vec![norm; dim],
            (0..dim)
                .map(|i| if i % 2 == 0 { norm } else { -norm })
                .collect(),
        ];
        let scales: Vec<_> = (0..dim).map(|i| [0.5, 2.0, 4.0][i % 3]).collect();
        let mut positions = Vec::with_capacity(4 * dim);
        let mut scores = Vec::with_capacity(4 * dim);
        for direction in 0..2 {
            for sign in [-1.0, 1.0] {
                for (i, scale) in scales.iter().enumerate() {
                    positions
                        .push(sign * [3.0, 0.75][direction] * directions[direction][i] * scale);
                    scores.push(-sign * [0.5, 1.5][direction] * directions[direction][i] / scale);
                }
            }
        }
        for (cap, threshold, retained) in [
            (0, 1.1, 0),
            (1, 1.1, 1),
            (2, 1.1, 2),
            (2, 2.0, 1),
            (2, 8.0, 0),
        ] {
            let fitted = alea_math::fisher::fit_low_rank(
                &positions, &scores, &scales, ridge, threshold, cap,
            )
            .unwrap();
            assert_eq!(
                fitted.rank(),
                retained,
                "dimension {dim}, cap {cap}, threshold {threshold}"
            );
            let expected_logdet = -2.0 * scales.iter().map(|s| s.ln()).sum::<f64>()
                - lambda[..retained].iter().map(|v| v.ln()).sum::<f64>();
            assert!((fitted.log_det() - expected_logdet).abs() < 1e-8);
            // Changing observation order changes the numerical basis, not G.
            let reversed_q: Vec<_> = positions
                .chunks_exact(dim)
                .rev()
                .flatten()
                .copied()
                .collect();
            let reversed_s: Vec<_> = scores.chunks_exact(dim).rev().flatten().copied().collect();
            let reordered = alea_math::fisher::fit_low_rank(
                &reversed_q,
                &reversed_s,
                &scales,
                ridge,
                threshold,
                cap,
            )
            .unwrap();
            assert_eq!(reordered.rank(), retained);
            assert!((reordered.log_det() - expected_logdet).abs() < 1e-8);
            for vector in 0..3 {
                let p: Vec<_> = (0..dim)
                    .map(|i| match vector {
                        0 => directions[0][i] / scales[i],
                        1 => directions[1][i] / scales[i],
                        // Orthogonal to both fitted directions, after standardization.
                        _ => {
                            if i % 4 < 2 {
                                norm / scales[i]
                            } else {
                                -norm / scales[i]
                            }
                        }
                    })
                    .collect();
                let mut expected: Vec<_> = p.iter().zip(&scales).map(|(p, s)| p * s).collect();
                for j in 0..retained {
                    let projection: f64 = p
                        .iter()
                        .zip(&scales)
                        .zip(&directions[j])
                        .map(|((p, s), u)| p * s * u)
                        .sum();
                    for (v, u) in expected.iter_mut().zip(&directions[j]) {
                        *v += (lambda[j] - 1.0) * projection * u;
                    }
                }
                for (v, s) in expected.iter_mut().zip(&scales) {
                    *v *= s;
                }
                let mut actual = vec![0.0; dim];
                fitted.velocity(&p, &mut actual).unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-8 * (1.0 + b.abs()),
                        "dimension {dim}, vector {vector}: {a} != {b}"
                    );
                }
                reordered.velocity(&p, &mut actual).unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()));
                }
            }
        }
    }
}
