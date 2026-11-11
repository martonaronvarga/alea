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

#[cfg(any(feature = "faer", alea_probe_blas))]
#[test]
fn generated_provider_first_order_matches_independent_operand_answers() {
    fn check<T: LogDensityGradient<Error = AutodiffError>>(
        target: T,
        value: fn(&[f64; 9]) -> f64,
        variant: usize,
    ) {
        if variant == 0 || (8..=10).contains(&variant) {
            // Finite cancelling primal, overflowing right-operand VJP.
            let h = f64::MAX / 2.;
            let q = [h, -h, 0., h, 0., -h, 1e-308, 1e-308, 1e-308];
            assert!(value(&q).is_finite());
            assert!(matches!(
                target.logp_grad(&q, &mut [41.; 9]),
                Err(AutodiffError::NonFinite)
            ));
        }
        for scale in [0., -1., 1., 2.] {
            if variant == 4 && scale == 0. {
                continue;
            }
            let q = [1., 2., -1., 0.5, -2., 3., 2., -1., 0.5].map(|v| v * scale);
            let (expected, score) = if variant == 0 || (8..=10).contains(&variant) {
                (
                    8.5 * scale * scale + 3.5 * scale,
                    [2., -1., 0.5, 4., -2., 1., 2., -2., 5.].map(|x| x * scale),
                )
            } else if variant == 1 {
                // Independent polynomial for the sum of a 2x2 matrix square.
                let [a, b, c, d, ..] = q;
                (
                    (a + b) * (a + c) + (c + d) * (b + d),
                    [
                        2. * a + b + c,
                        a + 2. * c + d,
                        a + 2. * b + d,
                        b + c + 2. * d,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                    ],
                )
            } else if variant == 7 {
                // 2x3 * 3x2 with overlapping gathers; includes existing A adjoints.
                (
                    -3.25 * scale * scale + 3.5 * scale,
                    [
                        1. - 1.5 * scale,
                        1. + 5. * scale,
                        1. - 0.5 * scale,
                        1.,
                        1. + 6.5 * scale,
                        1. - 0.5 * scale,
                        0.,
                        2. * scale,
                        2. * scale,
                    ],
                )
            } else if variant == 6 {
                (8.5 * scale + 3.5, [0., 0., 0., 0., 0., 0., 2., -2., 5.])
            } else if variant == 4 {
                let y0 = q[0] * q[6] + q[1] * q[7] + q[2] * q[8];
                let y1 = q[3] * q[6] + q[4] * q[7] + q[5] * q[8];
                let (u, v) = (-1. / (y0 * y0), -1. / (y1 * y1));
                (
                    1. / y0 + 1. / y1,
                    [
                        u * q[6],
                        u * q[7],
                        u * q[8],
                        v * q[6],
                        v * q[7],
                        v * q[8],
                        u * q[0] + v * q[3],
                        u * q[1] + v * q[4],
                        u * q[2] + v * q[5],
                    ],
                )
            } else {
                (0., [0.; 9])
            };
            let mut score = score;
            if variant == 0 || (8..=10).contains(&variant) {
                for s in &mut score[..6] {
                    *s += 1.;
                }
            }
            let mut g = [41.; 9];
            close(value(&q), expected);
            close(target.logp_grad(&q, &mut g).unwrap(), expected);
            for (x, y) in g.into_iter().zip(score) {
                close(x, y);
            }
            assert!(matches!(
                target.logp_grad(&[f64::NAN; 9], &mut g),
                Err(AutodiffError::NonFinite)
            ));
            if !matches!(variant, 2 | 11 | 12) {
                assert!(matches!(
                    target.logp_grad(&[f64::MAX; 9], &mut g),
                    Err(AutodiffError::NonFinite)
                ));
            }
            close(target.logp_grad(&q, &mut g).unwrap(), expected);
            for (x, y) in g.into_iter().zip(score) {
                close(x, y);
            }
        }
    }
    macro_rules! case {
        ($name:ident,$variant:expr) => {{
            use alea_enzyme_conformance::$name as model;
            const {
                assert!(model::SOURCE_EXPLICIT_FIRST_ORDER);
            }
            check(model::target(), model::value::log_density, $variant);
            check(
                model::composition::objective(),
                model::value::log_density,
                $variant,
            );
        }};
    }
    #[cfg(feature = "faer")]
    {
        case!(provider_faer_rect, 0);
        case!(provider_faer_alias, 1);
        case!(provider_faer_empty, 2);
        case!(provider_faer_zero, 3);
        case!(provider_faer_inverse, 4);
        case!(provider_faer_inactive, 6);
        case!(provider_faer_gemm, 7);
        case!(provider_faer_branch, 8);
        case!(provider_faer_scan, 9);
        case!(provider_faer_checkpoint, 10);
        case!(provider_faer_empty_inner, 11);
        case!(provider_faer_empty_columns, 12);
    }
    #[cfg(alea_probe_blas)]
    {
        case!(provider_blas_rect, 0);
        case!(provider_blas_alias, 1);
        case!(provider_blas_empty, 2);
        case!(provider_blas_zero, 3);
        case!(provider_blas_inverse, 4);
        case!(provider_blas_inactive, 6);
        case!(provider_blas_gemm, 7);
        case!(provider_blas_branch, 8);
        case!(provider_blas_scan, 9);
        case!(provider_blas_checkpoint, 10);
        case!(provider_blas_empty_inner, 11);
        case!(provider_blas_empty_columns, 12);
    }
}

#[cfg(any(feature = "faer", alea_probe_blas))]
#[test]
fn provider_first_order_preserves_ordinary_factor_compiler_route_and_admission() {
    fn check<T: LogDensityGradient<Error = AutodiffError>>(target: T) {
        let mut q = [1., 2., -1., 0.5, -2., 3., 2., -1., 0.5, 0.5];
        let mut g = [41.; 10];
        for sign in [-1., 1.] {
            q[9] = 0.5 * sign;
            close(target.logp_grad(&q, &mut g).unwrap(), 12.5);
            for (x, y) in g
                .into_iter()
                .zip([3., 0., 1.5, 5., -1., 2., 2., -2., 5., sign])
            {
                close(x, y);
            }
            q[9] = 0.;
            assert!(matches!(
                target.logp_grad(&q, &mut g),
                Err(AutodiffError::NonFinite)
            ));
        }
        q[9] = 0.5;
        close(target.logp_grad(&q, &mut g).unwrap(), 12.5);
        close(g[9], 1.);
    }
    #[cfg(feature = "faer")]
    {
        let source = include_str!(concat!(env!("OUT_DIR"), "/provider_faer_mixed_first.rs"));
        assert_eq!(
            source.matches("autodiff_reverse(__factor_reverse").count(),
            1
        );
        check(alea_enzyme_conformance::provider_faer_mixed::target());
        check(alea_enzyme_conformance::provider_faer_mixed::composition::objective());
    }
    #[cfg(alea_probe_blas)]
    {
        let source = include_str!(concat!(env!("OUT_DIR"), "/provider_blas_mixed_first.rs"));
        assert_eq!(
            source.matches("autodiff_reverse(__factor_reverse").count(),
            1
        );
        check(alea_enzyme_conformance::provider_blas_mixed::target());
        check(alea_enzyme_conformance::provider_blas_mixed::composition::objective());
    }
}
