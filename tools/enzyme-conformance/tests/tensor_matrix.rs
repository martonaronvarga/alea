//! Regression for nested AD of mutable tensor intermediates (not scalarized).
#![cfg_attr(alea_probe_reverse, feature(autodiff))]
use alea::{
    autodiff::autodiff::AutodiffError,
    capability::{PotentialHessianDerivative, PotentialHessianVector},
    target::LogDensityGradient,
};
use alea_enzyme_conformance::tensor_matrix;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn matrix_density_score_curvature_and_third_match_independent_quadratic() {
    let target = tensor_matrix::target();
    let checkpoint = alea_enzyme_conformance::tensor_checkpoint_matrix::target();
    for [a, b] in [[0.3, -0.4], [0., 0.], [-10., 1.], [2., 0.5]] {
        let mut g = [41.; 2];
        let mut h = [41.; 4];
        let mut t = [41.; 8];
        let q = [a, b];
        close(
            target.logp_grad(&q, &mut g).unwrap(),
            -0.5 * ((a + 0.5 * b).powi(2) + (-0.5 * a + 1.5 * b).powi(2)),
        );
        close(g[0], -1.25 * a + 0.25 * b);
        close(g[1], 0.25 * a - 2.5 * b);
        let mut cg = [41.; 2];
        close(
            checkpoint.logp_grad(&q, &mut cg).unwrap(),
            tensor_matrix::log_density(&q),
        );
        close(cg[0], g[0]);
        close(cg[1], g[1]);
        target
            .potential_hessian_and_derivatives(&q, &mut h, &mut t)
            .unwrap();
        for (actual, expected) in h.into_iter().zip([1.25, -0.25, -0.25, 2.5]) {
            close(actual, expected);
        }
        for value in t {
            close(value, 0.);
        }
        target.potential_hvp(&q, &[0.4, -0.2], &mut g).unwrap();
        close(g[0], 0.55);
        close(g[1], -0.6);
    }
}

#[test]
fn invalid_tensor_states_remain_typed_errors_and_do_not_poison_scratch() {
    let target = tensor_matrix::target();
    let mut g = [41.; 2];
    let mut h = [41.; 4];
    let mut t = [41.; 8];
    for q in [[f64::NAN, 0.], [f64::MAX, f64::MAX]] {
        assert!(matches!(
            target.logp_grad(&q, &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hvp(&q, &[1., 0.], &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hessian_and_derivatives(&q, &mut h, &mut t),
            Err(AutodiffError::NonFinite)
        ));
        target.logp_grad(&[0.3, -0.4], &mut g).unwrap();
        target
            .potential_hessian_and_derivatives(&[0.3, -0.4], &mut h, &mut t)
            .unwrap();
        close(g[0], -0.475);
        close(h[0], 1.25);
    }
    let allocations = allocation_counter::measure(|| {
        for _ in 0..8 {
            target.logp_grad(&[0.3, -0.4], &mut g).unwrap();
            target
                .potential_hvp(&[0.3, -0.4], &[1., 0.], &mut g)
                .unwrap();
            target
                .potential_hessian_and_derivatives(&[0.3, -0.4], &mut h, &mut t)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}

// Bounded direct-provider probes. Keep deliberate compiler-failure cases out of
// ordinary conformance builds; cargo rustc can opt in for this existing target.
#[cfg(any(feature = "faer", alea_probe_blas))]
mod provider_probe {
    use super::close;
    use alea::math::gemm::{MatrixKernel, try_gemm};

    fn product(kernel: MatrixKernel, x: &[f64; 9]) -> f64 {
        let mut out = [0.; 2];
        assert!(try_gemm(
            kernel,
            &x[..6],
            &x[6..],
            &mut out,
            (2, 3, 1),
            (false, false),
            false,
        ));
        out[0] + 2. * out[1]
    }

    #[cfg(feature = "faer")]
    #[cfg_attr(
        alea_probe_reverse,
        std::autodiff::autodiff_reverse(faer_reverse, Duplicated, Active)
    )]
    fn faer_primal(x: &[f64; 9]) -> f64 {
        product(MatrixKernel::Faer, x)
    }

    #[cfg(alea_probe_blas)]
    #[cfg_attr(
        alea_probe_reverse,
        std::autodiff::autodiff_reverse(blas_reverse, Duplicated, Active)
    )]
    fn blas_primal(x: &[f64; 9]) -> f64 {
        product(MatrixKernel::Blas, x)
    }

    fn primal_answers(primal: fn(&[f64; 9]) -> f64) {
        // A=[[1,2,-1],[.5,-2,3]], b=[2,-1,.5], Ab=[-.5,4.5].
        let q = [1., 2., -1., 0.5, -2., 3., 2., -1., 0.5];
        for scale in [0., -1., 1., 2.] {
            close(primal(&q.map(|v| v * scale)), 8.5 * scale * scale);
        }
    }

    #[cfg(alea_probe_reverse)]
    fn reverse_answers(reverse: fn(&[f64; 9], &mut [f64; 9], f64) -> f64) {
        let q = [1., 2., -1., 0.5, -2., 3., 2., -1., 0.5];
        // Independent derivatives of (a00+2*a10)*b0 +
        // (a01+2*a11)*b1 + (a02+2*a12)*b2, both operands active.
        let score = [2., -1., 0.5, 4., -2., 1., 2., -2., 5.];
        for scale in [0., -1., 1., 2.] {
            for seed in [0., -0.7, 1.] {
                let mut g = [0.; 9];
                close(
                    reverse(&q.map(|v| v * scale), &mut g, seed),
                    8.5 * scale * scale,
                );
                for (actual, expected) in g.into_iter().zip(score) {
                    close(actual, seed * scale * expected);
                }
            }
        }
    }

    #[cfg(feature = "faer")]
    #[test]
    fn faer_primal_has_independent_rectangular_answers() {
        primal_answers(faer_primal);
    }
    #[cfg(all(feature = "faer", alea_probe_reverse))]
    #[test]
    fn faer_compiler_reverse_has_independent_operand_scores() {
        reverse_answers(faer_reverse);
    }
    #[cfg(alea_probe_blas)]
    #[test]
    fn blas_primal_has_independent_rectangular_answers() {
        primal_answers(blas_primal);
    }
    #[cfg(all(alea_probe_blas, alea_probe_reverse))]
    #[test]
    fn blas_compiler_reverse_has_independent_operand_scores() {
        reverse_answers(blas_reverse);
    }
}
