#[path = "../models.rs"]
mod models;
use alea::{
    capability::{HessianVector, PotentialHessianDerivative},
    target::LogDensityGradient,
};
use alea_enzyme_conformance as generated;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn nonzero_third_derivatives_match_independent_polynomial() {
    let t = generated::polynomial::target();
    for [x, y] in [[0.3, -0.7], [0.0, 0.0], [-1.2, 0.5]] {
        let q = [x, y];
        let mut g = [41.0; 2];
        let lp = t.logp_grad(&q, &mut g).unwrap();
        close(
            lp,
            -0.5 * (x * x + y * y) - 0.1 * x * x * y * y - 0.01 * x * x * x * x,
            1e-13,
        );
        close(g[0], -x - 0.2 * x * y * y - 0.04 * x * x * x, 1e-13);
        close(g[1], -y - 0.2 * x * x * y, 1e-13);
        let mut h = [41.0; 4];
        let mut d = [41.0; 8];
        t.potential_hessian_and_derivatives(&q, &mut h, &mut d)
            .unwrap();
        let expected_h = [
            1.0 + 0.2 * y * y + 0.12 * x * x,
            0.4 * x * y,
            0.4 * x * y,
            1.0 + 0.2 * x * x,
        ];
        let expected_d = [
            0.24 * x,
            0.4 * y,
            0.4 * y,
            0.4 * x,
            0.4 * y,
            0.4 * x,
            0.4 * x,
            0.0,
        ];
        for (actual, expected) in h.into_iter().zip(expected_h) {
            close(actual, expected, 1e-13);
        }
        for (actual, expected) in d.into_iter().zip(expected_d) {
            close(actual, expected, 1e-13);
        }
        let v = [0.4, -0.2];
        let mut hv = [41.0; 2];
        t.potential_hvp(&q, &v, &mut hv).unwrap();
        for i in 0..2 {
            close(
                hv[i],
                expected_h[2 * i] * v[0] + expected_h[2 * i + 1] * v[1],
                1e-13,
            );
        }
    }
}

fn check_model<T: PotentialHessianDerivative>(name: &str, target: T) {
    let model = models::all()
        .unwrap()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1
        .into_target()
        .unwrap();
    let n = target.dimension();
    for scale in [-0.7, 0.0, 0.4] {
        let q: Vec<_> = (0..n).map(|i| scale + i as f64 * 0.13).collect();
        let mut g = vec![0.0; n];
        let mut expected = vec![0.0; n];
        close(
            target.logp_grad(&q, &mut g).unwrap(),
            model.logp_grad(&q, &mut expected).unwrap(),
            2e-12,
        );
        for (&a, &b) in g.iter().zip(&expected) {
            close(a, b, 2e-11);
        }
        let mut h = vec![0.0; n * n];
        let mut third = vec![0.0; n * n * n];
        target
            .potential_hessian_and_derivatives(&q, &mut h, &mut third)
            .unwrap();
        let mut plus = q.clone();
        let mut minus = q.clone();
        let mut gp = g.clone();
        let mut gm = g.clone();
        let mut hp = h.clone();
        let mut hm = h.clone();
        let mut scratch = third.clone();
        let mut v = vec![0.0; n];
        let mut product = g.clone();
        for k in 0..n {
            plus[k] += 1e-5;
            minus[k] -= 1e-5;
            model.logp_grad(&plus, &mut gp).unwrap();
            model.logp_grad(&minus, &mut gm).unwrap();
            v[k] = 1.0;
            target.potential_hvp(&q, &v, &mut product).unwrap();
            for i in 0..n {
                close(h[i * n + k], -(gp[i] - gm[i]) / 2e-5, 2e-7);
                close(product[i], h[i * n + k], 2e-12);
            }
            target
                .potential_hessian_and_derivatives(&plus, &mut hp, &mut scratch)
                .unwrap();
            target
                .potential_hessian_and_derivatives(&minus, &mut hm, &mut scratch)
                .unwrap();
            for i in 0..n * n {
                close(third[k * n * n + i], (hp[i] - hm[i]) / 2e-5, 2e-6);
            }
            plus[k] = q[k];
            minus[k] = q[k];
            v[k] = 0.0;
        }
    }
}

#[test]
fn typed_models_and_every_transform_include_one_jacobian_through_third_order() {
    macro_rules! check { ($($name:ident),* $(,)?) => {$(check_model(stringify!($name), generated::$name::target());)*}; }
    check!(
        regression,
        logistic,
        interval,
        lower,
        upper,
        ordered,
        positive_ordered,
        simplex,
        unit_vector,
        cholesky_covariance,
        cholesky_correlation,
        covariance,
        correlation,
        covariance3,
        correlation3
    );
}

#[test]
fn repeated_calls_overwrite_outputs_allocate_nothing_and_reject_bad_shapes() {
    let t = generated::polynomial::target();
    let mut g = [1.0; 2];
    let mut h = [1.0; 4];
    let mut d = [1.0; 8];
    let allocations = allocation_counter::measure(|| {
        for _ in 0..32 {
            t.logp_grad(&[0.3, -0.7], &mut g).unwrap();
            t.potential_hvp(&[0.3, -0.7], &[1.0, 0.0], &mut g).unwrap();
            t.potential_hessian_and_derivatives(&[0.3, -0.7], &mut h, &mut d)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
    let saved = g;
    assert!(t.logp_grad(&[0.0], &mut g).is_err());
    assert_eq!(g, saved);
    assert!(t.potential_hvp(&[0.0; 2], &[0.0], &mut g).is_err());
    assert_eq!(g, saved);
    assert!(
        t.potential_hessian_and_derivatives(&[0.0; 2], &mut h[..3], &mut d)
            .is_err()
    );
    assert!(t.logp_grad(&[f64::NAN, 0.0], &mut g).is_err());
    let transformed = generated::regression::target();
    assert!(transformed.logp_grad(&[0.0, -1000.0], &mut g).is_err());
    assert!(
        transformed
            .potential_hvp(&[0.0, -1000.0], &[1.0, 0.0], &mut g)
            .is_err()
    );
}

#[test]
fn larger_generated_target_has_correct_hessian_and_zero_third_tensor() {
    let target = generated::gaussian16::target();
    let q = [0.3; 16];
    let mut g = [41.0; 16];
    close(
        target.logp_grad(&q, &mut g).unwrap(),
        -0.5 * 16.0 * 0.3 * 0.3,
        1e-13,
    );
    for v in g {
        close(v, -0.3, 1e-13);
    }
    let mut h = [41.0; 256];
    let mut third = [41.0; 4096];
    target
        .potential_hessian_and_derivatives(&q, &mut h, &mut third)
        .unwrap();
    for i in 0..16 {
        for j in 0..16 {
            close(h[i * 16 + j], f64::from(i == j), 1e-13);
        }
    }
    assert!(third.iter().all(|&v| v == 0.0));
}

#[test]
fn rounded_singular_covariances_and_collapsed_intervals_fail_all_derivative_paths() {
    fn reject<T: PotentialHessianDerivative>(target: T, q: &[f64]) {
        let n = q.len();
        let mut g = vec![41.0; n];
        let mut h = vec![41.0; n * n];
        let mut third = vec![41.0; n * n * n];
        assert!(target.logp_grad(q, &mut g).is_err());
        assert!(target.potential_hvp(q, &vec![1.0; n], &mut g).is_err());
        assert!(
            target
                .potential_hessian_and_derivatives(q, &mut h, &mut third)
                .is_err()
        );
        target.logp_grad(&vec![0.0; n], &mut g).unwrap();
    }
    reject(generated::covariance::target(), &[0.0, 1e150, 0.0]);
    reject(generated::interval::target(), &[1000.0]);
    reject(generated::correlation::target(), &[1000.0]);
}
