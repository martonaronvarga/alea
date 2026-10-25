//! Independent algebra and admission contracts for inverse-mass geometry.
use alea_math::{
    buffer::OwnedBuffer,
    metric::{
        CholeskyFactor, EuclideanMetric, InverseDenseMetric, InverseDiagonalMetric, MetricError,
    },
};

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-14 * expected.abs().max(1e-300),
        "{actual} != {expected}"
    );
}

#[test]
fn dense_matches_independent_matrix_algebra_and_shared_energy_velocity() {
    // L=[[2,0],[1,3]], G=[[4,2],[2,10]], det G=36.
    let metric =
        InverseDenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[2., 1., 0., 3.])).unwrap());
    let mut velocity = [0.; 2];
    metric.velocity(&[3., -2.], &mut velocity).unwrap();
    assert_eq!(velocity, [8., -14.]);
    let energy = metric
        .kinetic_energy_with_velocity(&[3., -2.], &mut velocity)
        .unwrap();
    assert_eq!(energy, 26.);
    assert_eq!(velocity, [8., -14.]);
    close(metric.log_det(), -36_f64.ln());
    let z = [1.2, -0.7];
    let mut p = [0.; 2];
    metric.sample_momentum(&z, &mut p).unwrap();
    close(p[1], z[1] / 3.);
    close(p[0], (z[0] - p[1]) / 2.);
    close(
        metric.kinetic_energy(&p, &mut velocity).unwrap(),
        0.5 * (z[0] * z[0] + z[1] * z[1]),
    );
}

#[test]
fn extreme_diagonal_scales_do_not_require_a_representable_mass_inverse() {
    let g = [1e-310, 1e300];
    let diagonal = InverseDiagonalMetric::new(buffer(&g)).unwrap();
    let dense = InverseDenseMetric::new(
        CholeskyFactor::new_lower(2, buffer(&[g[0].sqrt(), 0., 0., g[1].sqrt()])).unwrap(),
    );
    for metric in [&diagonal as &dyn EuclideanMetric, &dense] {
        let mut p = [0.; 2];
        metric.sample_momentum(&[1., -2.], &mut p).unwrap();
        assert!(p.iter().all(|v| v.is_finite()));
        let mut v = [0.; 2];
        close(
            metric.kinetic_energy_with_velocity(&p, &mut v).unwrap(),
            2.5,
        );
        for i in 0..2 {
            close(v[i], g[i] * p[i]);
        }
        close(metric.log_det(), -(g[0].ln() + g[1].ln()));
    }
}

#[test]
fn diagonal_refit_preserves_all_cached_actions_on_failure() {
    let mut metric = InverseDiagonalMetric::new(buffer(&[2.0, 3.0])).unwrap();
    let before = metric.clone();
    for bad in [
        vec![1.0],
        vec![1.0, 0.0],
        vec![1.0, f64::NAN],
        vec![1.0, f64::INFINITY],
    ] {
        assert!(metric.set_inverse_masses(&bad).is_err());
        assert_eq!(metric, before);
    }
    metric.set_inverse_masses(&[4.0, 9.0]).unwrap();
    let mut p = [0.0; 2];
    metric.sample_momentum(&[2.0, 3.0], &mut p).unwrap();
    assert_eq!(p, [1.0, 1.0]);
    let mut v = [0.0; 2];
    close(
        metric.kinetic_energy_with_velocity(&p, &mut v).unwrap(),
        6.5,
    );
    assert_eq!(v, [4.0, 9.0]);
}

#[test]
fn factored_energy_stays_finite_when_unhalved_squared_norm_overflows() {
    let diagonal = InverseDiagonalMetric::new(buffer(&[1., 1.])).unwrap();
    let dense =
        InverseDenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[1., 0., 0., 1.])).unwrap());
    for metric in [&diagonal as &dyn EuclideanMetric, &dense] {
        let mut v = [0.; 2];
        let energy = metric
            .kinetic_energy_with_velocity(&[1e154, 1e154], &mut v)
            .unwrap();
        close(energy, 1e308);
        assert_eq!(v, [1e154, 1e154]);
    }
}

#[test]
fn shapes_are_transactional_and_invalid_inverse_diagonals_are_rejected() {
    for value in [0., -0., -1., f64::NAN, f64::INFINITY] {
        assert_eq!(
            InverseDiagonalMetric::new(buffer(&[1., value])).unwrap_err(),
            MetricError::InvalidDiagonal { index: 1 }
        );
    }
    let diagonal = InverseDiagonalMetric::new(buffer(&[2., 3.])).unwrap();
    let dense =
        InverseDenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[2., 1., 0., 3.])).unwrap());
    for metric in [&diagonal as &dyn EuclideanMetric, &dense] {
        for src_len in [0, 1, 2, 3] {
            for dst_len in [0, 1, 2, 3] {
                if src_len == 2 && dst_len == 2 {
                    continue;
                }
                let input = vec![1.; src_len];
                for action in 0..3 {
                    let mut out = vec![17.; dst_len];
                    let error = match action {
                        0 => metric.velocity(&input, &mut out),
                        1 => metric.sample_momentum(&input, &mut out),
                        _ => metric
                            .kinetic_energy_with_velocity(&input, &mut out)
                            .map(|_| ()),
                    };
                    assert!(matches!(error, Err(MetricError::VectorLength { .. })));
                    assert_eq!(out, vec![17.; dst_len]);
                }
            }
        }
    }
    let empty = InverseDenseMetric::new(CholeskyFactor::new_lower(0, buffer(&[])).unwrap());
    assert_eq!(
        empty.kinetic_energy_with_velocity(&[], &mut []).unwrap(),
        0.
    );
    assert!(InverseDiagonalMetric::new(buffer(&[])).is_ok());
}

#[test]
#[cfg(not(miri))]
fn inverse_metric_hot_actions_allocate_nothing() {
    let diagonal = InverseDiagonalMetric::new(buffer(&[2., 3.])).unwrap();
    let dense =
        InverseDenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[2., 1., 0., 3.])).unwrap());
    let mut out = [0.; 2];
    let count = allocation_counter::measure(|| {
        for metric in [&diagonal as &dyn EuclideanMetric, &dense] {
            for _ in 0..8 {
                metric.velocity(&[1., 2.], &mut out).unwrap();
                metric.sample_momentum(&[1., 2.], &mut out).unwrap();
                metric
                    .kinetic_energy_with_velocity(&[1., 2.], &mut out)
                    .unwrap();
            }
        }
    });
    assert_eq!(count.count_total, 0);
}

#[test]
fn inverse_actions_agree_with_independent_mass_solves_across_dimensions() {
    use alea_math::metric::DenseMetric;
    for n in [1, 2, 7, 16, 33] {
        let factor = CholeskyFactor::new_lower(
            n,
            OwnedBuffer::from_fn(n * n, |k| {
                let (row, col) = (k % n, k / n);
                if row == col {
                    1. + row as f64 / 10.
                } else if row > col {
                    0.03 * ((row * 7 + col * 13) as f64).sin()
                } else {
                    0.
                }
            }),
        )
        .unwrap();
        let independent = DenseMetric::new(factor.clone());
        let metric = InverseDenseMetric::new(factor);
        let p: Vec<_> = (0..n).map(|i| (i as f64).cos()).collect();
        let mut storage = OwnedBuffer::new(n + 1);
        let v = &mut storage[1..]; // Deliberately offset the aligned output.
        metric.velocity(&p, v).unwrap();
        let mut recovered = vec![0.; n];
        independent.velocity(v, &mut recovered).unwrap();
        for (actual, expected) in recovered.iter().zip(&p) {
            assert!((actual - expected).abs() < 2e-13);
        }
        metric.sample_momentum(&p, v).unwrap();
        let l = metric.factor().as_slice();
        for col in 0..n {
            let actual: f64 = (0..n).map(|row| l[row + col * n] * v[row]).sum();
            assert!((actual - p[col]).abs() < 2e-13);
        }
        let energy = metric.kinetic_energy(v, &mut recovered).unwrap();
        close(energy, p.iter().map(|x| 0.5 * x * x).sum());
    }
}
#[test]
fn vectorized_energy_matches_factored_oracle_and_avoids_unhalved_overflow() {
    use alea_math::metric::IdentityMetric;
    for n in 16..=81 {
        let p: Vec<_> = (0..n).map(|i| (i as f64 - 20.0) * 0.125).collect();
        let g: Vec<_> = (0..n).map(|i| 2.0_f64.powi(i as i32 % 9 - 4)).collect();
        let metric = InverseDiagonalMetric::new(OwnedBuffer::from_fn(n, |i| g[i])).unwrap();
        let mut v = vec![123.0; n];
        let actual = metric.kinetic_energy_with_velocity(&p, &mut v).unwrap();
        let expected = p
            .iter()
            .zip(&g)
            .map(|(p, g)| {
                let u = p * g.sqrt();
                (0.5 * u) * u
            })
            .sum::<f64>();
        assert!((actual - expected).abs() <= 8e-15 * expected);
        for i in 0..n {
            assert_eq!(v[i].to_bits(), (p[i] * g[i]).to_bits());
        }
        let identity = IdentityMetric::new(n);
        let actual = identity.kinetic_energy_with_velocity(&p, &mut v).unwrap();
        let expected = p.iter().map(|p| (0.5 * p) * p).sum::<f64>();
        assert!((actual - expected).abs() <= 8e-15 * expected);
        assert_eq!(v, p);
    }
    let mut p = vec![0.0; 17];
    p[0] = 1e154;
    p[16] = 1e154;
    let mut v = vec![0.0; 17];
    let energy = IdentityMetric::new(17)
        .kinetic_energy_with_velocity(&p, &mut v)
        .unwrap();
    assert!(energy.is_finite());
    assert!((energy / 1e308 - 1.0).abs() < 2e-15);
    // This is an analytically known sum; ordered accumulation loses all 63
    // small squared terms after the leading 2^54 term.
    let mut p = vec![1.0; 64];
    p[0] = 2.0_f64.powi(27);
    let energy = IdentityMetric::new(64)
        .kinetic_energy_with_velocity(&p, &mut vec![0.0; 64])
        .unwrap();
    let exact_rounded = 9_007_199_254_741_024.0;
    assert!((energy - exact_rounded).abs() <= 4.0);
}
