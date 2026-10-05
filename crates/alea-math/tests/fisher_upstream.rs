//! Executed upstream fits, not a reimplementation of the reference equations.
#![cfg(feature = "faer")]

use alea_math::{fisher::fit_low_rank, metric::EuclideanMetric};

#[test]
fn fisher_prefix_fits_meet_decimal_accuracy_and_characterize_upstream_limits() {
    let mut oracles = include_str!("fixtures/fisher-nuts-rs-decimal.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1);
    let mut cases = 0;
    for line in include_str!("fixtures/fisher-nuts-rs.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
    {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 11);
        let oracle: Vec<_> = oracles.next().unwrap().split(',').collect();
        assert_eq!(oracle.len(), 4);
        assert_eq!(fields[0], oracle[0]);
        let vector = |index: usize| {
            fields[index]
                .split(';')
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let dim = fields[1].parse::<usize>().unwrap();
        let n = fields[2].parse::<usize>().unwrap();
        let positions = vector(3);
        let scores = vector(4);
        let scales = vector(5);
        assert_eq!(positions.len(), dim * n);
        assert_eq!(scores.len(), dim * n);
        assert_eq!(scales.len(), dim);
        // Upstream computes these scales without Alea's diagonal ridge/clamps.
        // Supply the emitted scales to isolate the common low-rank estimator.
        let metric = fit_low_rank(
            &positions,
            &scores,
            &scales,
            fields[6].parse().unwrap(),
            fields[7].parse().unwrap(),
            dim, // Upstream has threshold filtering, but no Alea rank cap.
        )
        .unwrap_or_else(|error| panic!("{}: {error}", fields[0]));
        assert_eq!(
            metric.rank(),
            fields[8].parse::<usize>().unwrap(),
            "{}",
            fields[0]
        );
        assert_eq!(fields[8], oracle[1], "{}: reference rank", fields[0]);
        let upstream = vector(9);
        assert!(upstream.iter().all(|v| v.is_finite()));
        assert!(fields[10].parse::<f64>().unwrap().is_finite());
        let expected: Vec<f64> = oracle[2].split(';').map(|v| v.parse().unwrap()).collect();
        assert_eq!(expected.len(), dim * dim);
        assert_eq!(upstream.len(), dim * dim);
        let mut unit = vec![0.0; dim];
        let mut actual = vec![0.0; dim];
        let mut reference_error = 0.0_f64;
        for j in 0..dim {
            unit.fill(0.0);
            unit[j] = 1.0;
            metric.velocity(&unit, &mut actual).unwrap();
            for i in 0..dim {
                let scale = (expected[i * dim + i] * expected[j * dim + j]).sqrt();
                assert!(
                    (actual[i] - expected[i * dim + j]).abs() <= 1e-8 * scale,
                    "{} ({i},{j}): Alea {} != Decimal {}",
                    fields[0],
                    actual[i],
                    expected[i * dim + j]
                );
                reference_error = reference_error
                    .max((upstream[i * dim + j] - expected[i * dim + j]).abs() / scale);
            }
        }
        let logdet = oracle[3].parse::<f64>().unwrap();
        assert!(
            (metric.log_det() - logdet).abs() <= 1e-8 * (1.0 + logdet.abs()),
            "{}: Alea log determinant",
            fields[0]
        );
        // These two executed upstream results miss the original tolerance.
        // Keep the evidence explicit; never widen Alea's accuracy requirement
        // to imitate the reference's nested-square-root roundoff.
        if matches!(fields[0], "d16_n8_t1.01" | "d16_n8_t2") {
            assert!(
                reference_error > 1e-8,
                "{}: reassess known reference limitation",
                fields[0]
            );
        } else {
            assert!(
                reference_error <= 1e-8,
                "{}: upstream matrix error {reference_error:e}",
                fields[0]
            );
            let upstream_logdet = fields[10].parse::<f64>().unwrap();
            assert!((upstream_logdet - logdet).abs() <= 1e-8 * (1.0 + logdet.abs()));
        }
        cases += 1;
    }
    assert_eq!(cases, 18);
    assert!(oracles.next().is_none());
}
