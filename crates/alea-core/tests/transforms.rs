use alea_core::transform::{ParameterLayout, Transform, TransformError};
use rand::{RngExt, SeedableRng, rngs::StdRng};

// Shrinking complements the deterministic numerical-Jacobian matrix below.
// Native only: OS randomness/persistence is not part of the Miri safety gate.
#[cfg(not(miri))]
mod properties {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]
        #[test]
        fn vector_roundtrip(kind in 0usize..9, values in prop::collection::vec(-5.0f64..5.0, 1..16)) {
            let n = values.len();
            let t = match kind {
                0 => Transform::identity(n), 1 => Transform::positive(n),
                2 => Transform::lower(n, -3.0), 3 => Transform::upper(n, 5.0),
                4 => Transform::interval(n, -2.0, 3.0), 5 => Transform::ordered(n),
                6 => Transform::positive_ordered(n), 7 => Transform::simplex(n + 1),
                _ => Transform::unit_vector(n + 1),
            }.unwrap();
            let mut x = vec![0.0; t.constrained_dimension()];
            let mut q = vec![0.0; n];
            let mut scratch = vec![0.0; t.scratch_dimension()];
            t.constrain(&values, &mut x, &mut scratch).unwrap();
            t.unconstrain(&x, &mut q, &mut scratch).unwrap();
            for (a, b) in values.iter().zip(&q) { prop_assert!((a-b).abs() < 1e-8, "{a} != {b}"); }
        }
    }
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())),
        "{a} != {b}"
    );
}

fn cases(n: usize) -> Vec<(Transform, Vec<usize>, bool)> {
    let diagonal = (0..n).collect::<Vec<_>>();
    let lower = (0..n)
        .flat_map(|i| (0..=i).map(move |j| i * n + j))
        .collect::<Vec<_>>();
    let strict = (0..n)
        .flat_map(|i| (0..i).map(move |j| i * n + j))
        .collect::<Vec<_>>();
    vec![
        (Transform::identity(n).unwrap(), diagonal.clone(), false),
        (Transform::positive(n).unwrap(), diagonal.clone(), false),
        (Transform::lower(n, -1.3).unwrap(), diagonal.clone(), false),
        (Transform::upper(n, 4.2).unwrap(), diagonal.clone(), false),
        (
            Transform::interval(n, -2.3, 5.1).unwrap(),
            diagonal.clone(),
            false,
        ),
        (Transform::ordered(n).unwrap(), diagonal.clone(), false),
        (
            Transform::positive_ordered(n).unwrap(),
            diagonal.clone(),
            false,
        ),
        (Transform::simplex(n + 1).unwrap(), diagonal.clone(), false),
        (
            Transform::unit_vector(n + 1).unwrap(),
            (0..=n).collect(),
            true,
        ),
        (
            Transform::cholesky_covariance(n).unwrap(),
            lower.clone(),
            false,
        ),
        (Transform::covariance(n).unwrap(), lower, false),
        (
            Transform::cholesky_correlation(n).unwrap(),
            strict.clone(),
            false,
        ),
        (Transform::correlation(n).unwrap(), strict, false),
    ]
}

fn log_determinant(mut a: Vec<f64>, n: usize) -> f64 {
    let mut log = 0.0;
    for j in 0..n {
        let pivot = (j..n)
            .max_by(|&i, &k| a[i * n + j].abs().total_cmp(&a[k * n + j].abs()))
            .unwrap();
        for k in 0..n {
            a.swap(j * n + k, pivot * n + k);
        }
        log += a[j * n + j].abs().ln();
        for i in j + 1..n {
            let factor = a[i * n + j] / a[j * n + j];
            for k in j + 1..n {
                a[i * n + k] -= factor * a[j * n + k];
            }
        }
    }
    log
}

#[test]
fn randomized_roundtrips_jacobians_and_pullbacks() {
    let mut rng = StdRng::seed_from_u64(0x414c4541);
    let iterations = if cfg!(miri) { 1 } else { 32 };
    for n in 1..=4 {
        for (transform, independent, surface) in cases(n) {
            let d = transform.unconstrained_dimension();
            let m = transform.constrained_dimension();
            let mut scratch = vec![0.0; transform.scratch_dimension()];
            for _ in 0..iterations {
                let q = (0..d)
                    .map(|_| rng.random_range(-0.7..0.7))
                    .collect::<Vec<_>>();
                let mut x = vec![0.0; m];
                let jac = transform.constrain(&q, &mut x, &mut scratch).unwrap();
                let mut roundtrip = vec![0.0; d];
                transform
                    .unconstrain(&x, &mut roundtrip, &mut scratch)
                    .unwrap();
                for (&a, &b) in q.iter().zip(&roundtrip) {
                    close(a, b, 1e-12);
                }
                let gx = (0..m)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect::<Vec<_>>();
                let mut gq = vec![0.0; d];
                transform.pullback(&q, &gx, &mut gq, &mut scratch).unwrap();
                let mut derivatives = vec![0.0; independent.len() * d];
                for j in 0..d {
                    let mut qp = q.clone();
                    let mut qm = q.clone();
                    qp[j] += 1e-5;
                    qm[j] -= 1e-5;
                    let mut xp = vec![0.0; m];
                    let mut xm = vec![0.0; m];
                    let jp = transform.constrain(&qp, &mut xp, &mut scratch).unwrap();
                    let jm = transform.constrain(&qm, &mut xm, &mut scratch).unwrap();
                    let delta: f64 = xp
                        .iter()
                        .zip(&xm)
                        .zip(&gx)
                        .map(|((p, m), g)| (p - m) * g)
                        .sum();
                    close(gq[j], (delta + jp - jm) / 2e-5, 2e-8);
                    for (i, &coordinate) in independent.iter().enumerate() {
                        derivatives[i * d + j] = (xp[coordinate] - xm[coordinate]) / 2e-5;
                    }
                }
                let numerical_jac = if surface {
                    let mut gram = vec![0.0; d * d];
                    for i in 0..d {
                        for j in 0..d {
                            gram[i * d + j] = (0..m)
                                .map(|k| derivatives[k * d + i] * derivatives[k * d + j])
                                .sum();
                        }
                    }
                    0.5 * log_determinant(gram, d)
                } else {
                    log_determinant(derivatives, d)
                };
                close(jac, numerical_jac, 2e-8);
            }
        }
    }
}

#[test]
fn layout_counts_shapes_and_zero_dimension_blocks() {
    let layout = ParameterLayout::new([
        Transform::identity(0).unwrap(),
        Transform::simplex(3).unwrap(),
        Transform::covariance(2).unwrap(),
    ])
    .unwrap();
    assert_eq!(layout.unconstrained_dimension(), 5);
    assert_eq!(layout.constrained_dimension(), 7);
    assert_eq!(layout.blocks()[1].unconstrained_range(), 0..2);
    assert_eq!(layout.blocks()[2].constrained_range(), 3..7);
    let mut scratch = vec![0.0; layout.scratch_dimension()];
    let mut x = [0.0; 7];
    let q = [0.1, -0.3, 0.5, -0.1, 0.2];
    layout.constrain(&q, &mut x, &mut scratch).unwrap();
    let mut inverse = [0.0; 5];
    layout.unconstrain(&x, &mut inverse, &mut scratch).unwrap();
    for (&a, &b) in q.iter().zip(&inverse) {
        close(a, b, 1e-12);
    }
    assert!(layout.constrain(&q[..4], &mut x, &mut scratch).is_err());
    let before = x;
    assert!(layout.constrain(&q, &mut x, &mut []).is_err());
    assert_eq!(x, before);
    for transform in [
        Transform::identity(0).unwrap(),
        Transform::simplex(1).unwrap(),
        Transform::correlation(1).unwrap(),
        Transform::cholesky_correlation(1).unwrap(),
    ] {
        let mut scratch = vec![0.0; transform.scratch_dimension()];
        let mut x = vec![0.0; transform.constrained_dimension()];
        assert_eq!(transform.constrain(&[], &mut x, &mut scratch).unwrap(), 0.0);
        transform.unconstrain(&x, &mut [], &mut scratch).unwrap();
        transform.pullback(&[], &x, &mut [], &mut scratch).unwrap();
    }
}

#[test]
fn rejects_invalid_shapes_domains_nonfinite_and_precision_collapse() {
    assert!(Transform::identity(usize::MAX).is_err());
    assert!(Transform::covariance(usize::MAX).is_err());
    assert!(Transform::simplex(0).is_err());
    assert!(Transform::unit_vector(1).is_err());
    for (a, b) in [
        (0.0, 0.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (-f64::MAX, f64::MAX),
        (1.0, 1.0f64.next_up()),
    ] {
        assert!(Transform::interval(1, a, b).is_err());
    }
    for (transform, q) in [
        (Transform::positive(1).unwrap(), vec![-1000.0]),
        (Transform::lower(1, 1.0).unwrap(), vec![-100.0]),
        (Transform::interval(1, 0.0, 1.0).unwrap(), vec![100.0]),
        (Transform::ordered(2).unwrap(), vec![1e20, 0.0]),
        (Transform::simplex(2).unwrap(), vec![1000.0]),
        (Transform::unit_vector(2).unwrap(), vec![1e300]),
        (Transform::correlation(2).unwrap(), vec![100.0]),
    ] {
        let mut x = vec![0.0; transform.constrained_dimension()];
        let mut s = vec![0.0; transform.scratch_dimension()];
        assert!(transform.constrain(&q, &mut x, &mut s).is_err());
    }
    for (t, _, _) in cases(2) {
        let mut x = vec![123.0; t.constrained_dimension()];
        let mut s = vec![0.0; t.scratch_dimension()];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let q = vec![bad; t.unconstrained_dimension()];
            assert!(matches!(
                t.constrain(&q, &mut x, &mut s),
                Err(TransformError::NonFiniteInput { .. })
            ));
            assert!(x.iter().all(|v| *v == 123.0));
        }
    }
    let c = Transform::covariance(2).unwrap();
    assert!(
        c.unconstrain(&[1.0, 2.0, 2.0, 1.0], &mut [0.0; 3], &mut [0.0; 8])
            .is_err()
    );
    assert!(
        c.unconstrain(&[1.0, 0.0, 0.1, 1.0], &mut [0.0; 3], &mut [0.0; 8])
            .is_err()
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "allocation counter uses native thread-local instrumentation"
)]
fn every_transform_reuses_caller_storage() {
    for (t, _, _) in cases(3) {
        let q = vec![0.1; t.unconstrained_dimension()];
        let mut x = vec![0.0; t.constrained_dimension()];
        let gx = vec![0.2; x.len()];
        let mut gq = q.clone();
        let mut s = vec![0.0; t.scratch_dimension()];
        let allocations = allocation_counter::measure(|| {
            for _ in 0..32 {
                t.constrain(&q, &mut x, &mut s).unwrap();
                t.pullback(&q, &gx, &mut gq, &mut s).unwrap();
                t.unconstrain(&x, &mut gq, &mut s).unwrap();
            }
        });
        assert_eq!(allocations.count_total, 0);
    }
}

#[test]
fn inverse_domains_and_pullback_shape_failures() {
    for (t, x) in [
        (Transform::positive(1).unwrap(), vec![0.0]),
        (Transform::upper(1, 2.0).unwrap(), vec![2.0]),
        (Transform::interval(1, 0.0, 1.0).unwrap(), vec![1.0]),
        (Transform::ordered(2).unwrap(), vec![1.0, 1.0]),
        (Transform::positive_ordered(2).unwrap(), vec![-1.0, 1.0]),
        (Transform::simplex(2).unwrap(), vec![0.2, 0.2]),
        (Transform::unit_vector(2).unwrap(), vec![0.0, 1.0]),
        (Transform::unit_vector(2).unwrap(), vec![0.0, 0.0]),
        (
            Transform::cholesky_covariance(2).unwrap(),
            vec![1.0, 0.1, 0.0, 1.0],
        ),
        (
            Transform::cholesky_correlation(2).unwrap(),
            vec![1.0, 0.0, 0.8, 0.8],
        ),
        (Transform::correlation(2).unwrap(), vec![2.0, 0.0, 0.0, 1.0]),
    ] {
        let mut q = vec![0.0; t.unconstrained_dimension()];
        let mut scratch = vec![0.0; t.scratch_dimension()];
        assert!(t.unconstrain(&x, &mut q, &mut scratch).is_err(), "{t:?}");
        let mut g = vec![23.0; q.len()];
        assert!(t.pullback(&q, &[], &mut g, &mut scratch).is_err());
        assert!(g.iter().all(|v| *v == 23.0));
        assert!(
            t.pullback(&q, &vec![f64::NAN; x.len()], &mut g, &mut scratch)
                .is_err()
        );
        assert!(g.iter().all(|v| *v == 23.0));
    }
}
