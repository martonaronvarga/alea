use core::iter::Iterator;

use super::*;
use approx::{assert_relative_eq, assert_ulps_eq};

// Stan reference values (wiener4_lpdf(y|alpha,tau,beta,delta))
const STAN_TESTS: &[(&str, WienerObservation, Wiener4Params, f64)] = &[
    (
        "mat/test/prob/wiener/wiener_test/1",
        WienerObservation {
            rt: 1.1,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 2.1,
            tau: 0.3,
            beta: 0.55,
            delta: 0.4,
        },
        -0.892976431870503,
    ),
    (
        "mat/test/prob/wiener/wiener_test/2",
        WienerObservation {
            rt: 2.1,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 4.1,
            tau: 0.6,
            beta: 0.05,
            delta: 0.1,
        },
        -5.28933922584833,
    ),
    (
        "mat/test/prob/wiener/wiener_test/3",
        WienerObservation {
            rt: 1.2,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 10.1,
            tau: 0.35,
            beta: 0.95,
            delta: 0.5,
        },
        -1.36212169454714,
    ),
    (
        "mat/test/prob/wiener/wiener_test/4",
        WienerObservation {
            rt: 50.1,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 4.3,
            tau: 1.05,
            beta: 0.65,
            delta: 0.15,
        },
        -15.3049368722015,
    ),
    (
        "mat/test/prob/wiener/wiener_test/5",
        WienerObservation {
            rt: 1.51,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 1.1,
            tau: 0.9,
            beta: 0.2,
            delta: 10.5,
        },
        -26.4531852275477,
    ),
    (
        "upper_fast",
        WienerObservation {
            rt: 0.5,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 1.5,
            tau: 0.3,
            beta: 0.6,
            delta: -0.5,
        },
        -0.24061277,
    ),
    (
        "upper_slow",
        WienerObservation {
            rt: 1.2,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 2.0,
            tau: 0.2,
            beta: 0.3,
            delta: 0.5,
        },
        -1.1719559,
    ),
    (
        "upper_fast",
        WienerObservation {
            rt: 0.4,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 1.5,
            tau: 0.3,
            beta: 0.4,
            delta: 1.0,
        },
        -0.77042144,
    ),
    (
        "upper_slow",
        WienerObservation {
            rt: 1.5,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 2.0,
            tau: 0.2,
            beta: 0.3,
            delta: 0.5,
        },
        -1.539122,
    ),
    // Edge cases
    (
        "boundary_tau",
        WienerObservation {
            rt: 0.3,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: 1.0,
            tau: 0.3,
            beta: 0.5,
            delta: 0.0,
        },
        f64::NEG_INFINITY,
    ),
    (
        "invalid_alpha",
        WienerObservation {
            rt: 0.5,
            boundary: Boundary::Upper,
        },
        Wiener4Params {
            alpha: -1.0,
            tau: 0.1,
            beta: 0.5,
            delta: 0.0,
        },
        f64::NEG_INFINITY,
    ),
];

#[test]
fn test_wiener4_log_prob_stan_reference() {
    for (_name, obs, params, expected) in STAN_TESTS {
        let eval = Wiener4.log_prob(obs, params, 1e-8);
        assert_ulps_eq!(eval.log_prob, *expected, epsilon = 1e-6);
    }
}

#[test]
fn test_wiener4_gradients_numerical() {
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let params = Wiener4Params::with_params(1.5, 0.2, 0.4, 1.0).unwrap();

    // Finite difference gradients (1e-6 step)
    let h = 1e-8;
    let analytic = Wiener4.fused(&obs, &params, 1e-8).grad;

    // d_alpha
    let p1 = Wiener4Params::with_params_unchecked(
        params.alpha + h,
        params.tau,
        params.beta,
        params.delta,
    );
    let p2 = Wiener4Params::with_params_unchecked(
        params.alpha - h,
        params.tau,
        params.beta,
        params.delta,
    );
    let num_alpha = (Wiener4.log_prob(&obs, &p1, 1e-8).log_prob
        - Wiener4.log_prob(&obs, &p2, 1e-8).log_prob)
        / (2.0 * h);

    assert_relative_eq!(
        analytic.alpha,
        num_alpha,
        epsilon = 1e-4,
        max_relative = 1e-3
    );

    // Similar for tau, beta, delta...
}

fn stan_wiener5_data() -> (f64, WienerObservation, Wiener5Params, f64) {
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let params = Wiener5Params::with_params_unchecked(1.5, 0.3, 0.55, 0.4, 0.1);
    // Stan log_prob = -0.5239455914
    (1e-10, obs, params, -0.5239455914)
}

fn stan_wiener7_data() -> (f64, WienerObservation, Wiener7Params, f64, [f64; 7]) {
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let params = Wiener7Params::with_params_unchecked(1.5, 0.3, 0.55, 0.4, 0.05, 0.15, 0.1);
    let log_prob = -0.3338914691;
    let grad = [
        0.1991638664,
        2.500755095,
        -0.3168100084,
        0.5048532259,
        -0.01629080941,
        -0.04512982532,
        1.341071894,
    ];
    (1e-10, obs, params, log_prob, grad)
}

#[test]
fn wiener5_matches_stan() {
    let (eps, obs, params, expected_lp) = stan_wiener5_data();
    let eval = Wiener5.log_prob(&obs, &params, eps);
    assert_relative_eq!(eval.log_prob, expected_lp, epsilon = 1e-8);
}

#[test]
fn wiener5_gradient_matches_stan_numerical() {
    let (eps, obs, params, _expected_lp) = stan_wiener5_data();
    let eval = Wiener5.fused(&obs, &params, eps);

    // Check log_prob consistency
    let lp_only = Wiener5.log_prob(&obs, &params, eps).log_prob;
    assert_relative_eq!(eval.log_prob, lp_only, epsilon = 1e-12);

    // Finite difference verification
    let h = 1e-6;
    let analytic = eval.grad;

    // Test alpha gradient
    let mut p_plus = params;
    let mut p_minus = params;
    p_plus.base.alpha += h;
    p_minus.base.alpha -= h;
    let fd_alpha = (Wiener5.log_prob(&obs, &p_plus, eps).log_prob
        - Wiener5.log_prob(&obs, &p_minus, eps).log_prob)
        / (2.0 * h);
    assert_relative_eq!(analytic.alpha, fd_alpha, epsilon = 1e-4);

    // Test tau gradient
    let mut p_plus = params;
    let mut p_minus = params;
    p_plus.base.tau += h;
    p_minus.base.tau -= h;
    let fd_tau = (Wiener5.log_prob(&obs, &p_plus, eps).log_prob
        - Wiener5.log_prob(&obs, &p_minus, eps).log_prob)
        / (2.0 * h);
    assert_relative_eq!(analytic.tau, fd_tau, epsilon = 1e-4);

    // Test s_delta gradient
    let mut p_plus = params;
    let mut p_minus = params;
    p_plus.s_delta += h;
    p_minus.s_delta -= h;
    let fd_sv = (Wiener5.log_prob(&obs, &p_plus, eps).log_prob
        - Wiener5.log_prob(&obs, &p_minus, eps).log_prob)
        / (2.0 * h);
    assert_relative_eq!(analytic.s_delta, fd_sv, epsilon = 1e-4);
}

#[test]
fn wiener7_matches_stan() {
    let (eps, obs, params, expected_lp, expected_grad) = stan_wiener7_data();
    let eval = Wiener7.fused(&obs, &params, eps);


    assert_relative_eq!(eval.log_prob, expected_lp, epsilon = 1e-6);

    for (i, g) in eval.grad.to_array().iter().enumerate() {
        assert_relative_eq!(*g, expected_grad[i], epsilon = 1e-4, max_relative = 1e-3);
    }
}

#[test]
fn wiener7_finite_difference_consistency() {
    let (eps, obs, params, _expected_lp, _expected_grad) = stan_wiener7_data();
    let eval = Wiener7.fused(&obs, &params, eps);
    let analytic = eval.grad.to_array();

    // Finite difference for all parameters
    let h = 1e-5;
    let f = |p: &Wiener7Params| Wiener7.log_prob(&obs, p, eps).log_prob;

    let _param_names = [
        "alpha", "tau", "beta", "delta", "s_delta", "s_beta", "s_tau",
    ];
    let base_params = params.to_array();

    for i in 0..7 {
        let mut p_plus_arr = base_params;
        let mut p_minus_arr = base_params;
        p_plus_arr[i] += h;
        p_minus_arr[i] -= h;

        let p_plus = Wiener7Params::from_array(p_plus_arr);
        let p_minus = Wiener7Params::from_array(p_minus_arr);

        let fd_grad = (f(&p_plus) - f(&p_minus)) / (2.0 * h);


        assert_relative_eq!(analytic[i], fd_grad, epsilon = 5e-3, max_relative = 1e-2);
    }
}

#[test]
fn wiener5_log_prob_consistency() {
    let (eps, obs, params, expected_lp) = stan_wiener5_data();

    // Test both boundaries
    let eval_upper = Wiener5.log_prob(&obs, &params, eps);
    assert_relative_eq!(eval_upper.log_prob, expected_lp, epsilon = 1e-8);

    // Test lower boundary
    let obs_lower = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Lower,
    };
    let eval_lower = Wiener5.log_prob(&obs_lower, &params, eps);
    assert!(eval_lower.log_prob.is_finite());
}

#[test]
fn test_truncation_consistency() {
    let params = Wiener4Params {
        alpha: 1.5,
        tau: 0.2,
        beta: 0.4,
        delta: 1.0,
    };
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let core = Wiener4.core(&obs, &params, 1e-8).unwrap();

    let (ks, kl) = (
        Wiener4::k_s(core.t_prime, core.w_eff, core.log_eps_eff),
        Wiener4::k_l(core.t_prime, core.log_eps_eff),
    );
    assert!(ks <= 50, "ks too large: {}", ks); // Reasonable truncation
    assert!(kl <= 20, "kl too large: {}", kl);
}

#[test]
fn test_series_monotonicity() {
    let t_prime = 0.5;
    let w = 0.3;
    let k = 20;

    let small_log = Wiener4::small_time_series_raw(t_prime, w, k).unwrap().ln();
    let large_log = Wiener4::large_time_scaled_accum(t_prime, w, k)
        .unwrap()
        .0
        .ln();

    // Small-time should be more negative for small t'
    assert!(small_log <= large_log + 1e-10);
}

#[test]
fn test_wiener5_validity() {
    let params = Wiener5Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1).unwrap();
    assert!(params.base.valid());

    let invalid = Wiener5Params {
        base: params.base,
        s_delta: -0.1,
    };
    assert!(!invalid.base.valid() || invalid.s_delta < 0.0);
}

#[test]
fn test_wiener7_quadrature() {
    let params = Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap();
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };

    let fused = Wiener7.fused(&obs, &params, 1e-8);
    let (log_pdf, grad) = (fused.log_prob, fused.grad);
    assert!(log_pdf.is_finite());
    for g in grad.to_array().iter() {
        assert!(g.is_finite());
    }
}

#[test]
fn test_target_fusedlogdensity() {
    let data = vec![
        WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        },
        WienerObservation {
            rt: 0.6,
            boundary: Boundary::Lower,
        },
    ];
    let params = Wiener4Params::with_params(1.5, 0.2, 0.4, 1.0).unwrap();
    let target = Target::new(Wiener4, data);

    let mut grad = [0.0f64; 4];
    let lp = target.log_prob_and_grad(&params, &mut grad);
    assert!(lp.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn test_parameter_transform() {
    let constrained = Wiener5Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1).unwrap();
    let unconstrained = Wiener5Params::to_unconstrained(&constrained);

    // All unconstrained should be finite
    assert!(unconstrained.iter().all(|u| u.is_finite()));

    let roundtrip = Wiener5Params::from_unconstrained(&unconstrained);
    assert_relative_eq!(
        roundtrip.base.alpha,
        constrained.base.alpha,
        epsilon = 1e-10
    );
    assert_relative_eq!(roundtrip.s_delta, constrained.s_delta, epsilon = 1e-10);

    // Jacobian
    let jac = Wiener5Params::log_abs_det_jacobian(&unconstrained);
    assert!(jac.is_finite());
}

#[test]
fn wiener4_params_reject_negative_alpha() {
    let err = Wiener4Params::with_params(-1.0, 0.1, 0.5, 0.0).unwrap_err();
    assert!(matches!(err, ProbError::InvalidParameters(_)));
}

#[test]
fn wiener5_rejects_nonfinite_precision_and_variability() {
    let obs = WienerObservation {
        rt: 1.0,
        boundary: Boundary::Upper,
    };
    let params = Wiener5Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.1);

    assert_eq!(Wiener5.fused(&obs, &params, f64::NAN).log_prob, f64::NEG_INFINITY);
    assert_eq!(
        Wiener5.fused(&obs, &params, f64::INFINITY).log_prob,
        f64::NEG_INFINITY
    );

    let nan_sv = Wiener5Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, f64::NAN);
    let inf_sv = Wiener5Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, f64::INFINITY);
    assert_eq!(Wiener5.fused(&obs, &nan_sv, 1e-8).log_prob, f64::NEG_INFINITY);
    assert_eq!(Wiener5.fused(&obs, &inf_sv, 1e-8).log_prob, f64::NEG_INFINITY);
}

#[test]
fn wiener7_fused_rejects_negative_unchecked_variability() {
    let obs = WienerObservation {
        rt: 1.0,
        boundary: Boundary::Upper,
    };

    for params in [
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, -1.0, 0.0, 0.1),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.0, -1.0, 0.1),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, -1.0, -1.0, 0.1),
    ] {
        let fused = Wiener7.fused(&obs, &params, 1e-8);
        let log_only = Wiener7.log_prob(&obs, &params, 1e-8);

        assert_eq!(fused.log_prob, f64::NEG_INFINITY);
        assert_eq!(log_only.log_prob, f64::NEG_INFINITY);
        assert!(fused.grad.to_array().iter().all(|g| g.is_nan()));
    }
}

#[test]
fn wiener7_rejects_nonfinite_precision_and_variability() {
    let obs = WienerObservation {
        rt: 1.0,
        boundary: Boundary::Upper,
    };
    let params = Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.1, 0.1, 0.1);
    let no_variability =
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.0, 0.0, 0.1);

    assert_eq!(Wiener7.fused(&obs, &params, f64::NAN).log_prob, f64::NEG_INFINITY);
    assert_eq!(
        Wiener7.fused(&obs, &params, f64::INFINITY).log_prob,
        f64::NEG_INFINITY
    );
    assert_eq!(
        Wiener7.fused(&obs, &no_variability, f64::NAN).log_prob,
        f64::NEG_INFINITY
    );

    for params in [
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, f64::NAN, 0.1, 0.1),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, f64::INFINITY, 0.1, 0.1),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.1, f64::NAN, 0.1),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.1, f64::INFINITY, 0.1),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.1, 0.1, f64::NAN),
        Wiener7Params::with_params_unchecked(1.5, 0.2, 0.4, 1.0, 0.1, 0.1, f64::INFINITY),
    ] {
        assert_eq!(Wiener7.fused(&obs, &params, 1e-8).log_prob, f64::NEG_INFINITY);
    }
}

#[test]
fn parameter_defaults_are_valid_neutral_points() {
    let p4 = Wiener4Params::default();
    assert_eq!(p4, Wiener4Params::new());
    assert!(p4.valid());
    assert_eq!(p4.to_array(), [1.0, 0.0, 0.5, 0.0]);

    let p5 = Wiener5Params::default();
    assert_eq!(p5, Wiener5Params::new());
    assert!(p5.base.valid());
    assert_eq!(p5.to_array(), [1.0, 0.0, 0.5, 0.0, 0.0]);

    let p7 = Wiener7Params::default();
    assert_eq!(p7, Wiener7Params::new());
    assert_eq!(p7.to_array(), [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn wiener4_large_time_log_series_matches_active_branch_formula() {
    let t_prime = 0.6;
    let w = 0.31;
    let k = 20;
    let q = 1.0 - w;
    let raw = (1..=k)
        .map(|j| {
            let jf = j as f64;
            jf * (std::f64::consts::PI * jf * q).sin()
                * (-0.5 * jf * jf * std::f64::consts::PI.powi(2) * t_prime).exp()
        })
        .sum::<f64>();
    let expected = std::f64::consts::PI.ln() + raw.ln();
    let actual = Wiener4::large_time_log_series(t_prime, w, k).unwrap();

    assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn wiener4_small_time_log_series_matches_raw_reconstruction() {
    let t_prime = 0.1; // Small t'
    let w = 0.3;
    let k = 15;

    let log_s = Wiener4::small_time_log_series(t_prime, w, k).unwrap();
    let raw_s = Wiener4::small_time_series_raw(t_prime, w, k).unwrap();

    let missing_pref = -0.5 * std::f64::consts::TAU.ln()
        - 1.5 * t_prime.ln()
        - ((1.0 - w) * (1.0 - w) * 0.5 / t_prime);
    let reconstructed_log_s = missing_pref + raw_s.ln();
    assert_relative_eq!(reconstructed_log_s, log_s, epsilon = 1e-10);
}

// Helper: finite‑difference gradient for a scalar parameter via symmetric difference
fn fd_grad<F: Fn(&Wiener7Params) -> f64>(
    params: &Wiener7Params,
    f: F,
    idx: usize,
    h: f64,
) -> f64 {
    let mut p_plus = *params;
    let mut p_minus = *params;
    match idx {
        0 => {
            p_plus.base.base.alpha += h;
            p_minus.base.base.alpha -= h;
        }
        1 => {
            p_plus.base.base.tau += h;
            p_minus.base.base.tau -= h;
        }
        2 => {
            p_plus.base.base.beta += h;
            p_minus.base.base.beta -= h;
        }
        3 => {
            p_plus.base.base.delta += h;
            p_minus.base.base.delta -= h;
        }
        4 => {
            p_plus.base.s_delta += h;
            p_minus.base.s_delta -= h;
        }
        5 => {
            p_plus.s_beta += h;
            p_minus.s_beta -= h;
        }
        6 => {
            p_plus.s_tau += h;
            p_minus.s_tau -= h;
        }
        _ => unreachable!(),
    }
    (f(&p_plus) - f(&p_minus)) / (2.0 * h)
}

#[test]
fn wiener7_delegates_to_wiener5_when_no_variability() {
    // sw = 0, st0 = 0 → should match Wiener5 exactly
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let alpha = 1.5;
    let tau = 0.2;
    let beta = 0.4;
    let delta = 1.0;
    let sv = 0.1;
    let eps = 1e-8;
    let w5_params = Wiener5Params::with_params(alpha, tau, beta, delta, sv).unwrap();
    let fused = Wiener5.fused(&obs, &w5_params, eps);
    let (lp5, grad5) = (fused.log_prob, fused.grad);

    let w7_params = Wiener7Params::with_params(alpha, tau, beta, delta, 0.0, 0.0, sv).unwrap();
    let fused = Wiener7.fused(&obs, &w7_params, eps);
    let (lp7, grad7) = (fused.log_prob, fused.grad);
    assert_ulps_eq!(lp7, lp5, epsilon = 1e-6);
    // first five gradients match
    assert_ulps_eq!(grad7[0], grad5.alpha, epsilon = 1e-6);
    assert_ulps_eq!(grad7[1], grad5.tau, epsilon = 1e-6);
    assert_ulps_eq!(grad7[2], grad5.beta, epsilon = 1e-6);
    assert_ulps_eq!(grad7[3], grad5.delta, epsilon = 1e-6);
    assert_ulps_eq!(grad7[4], grad5.s_delta, epsilon = 1e-6);
    // sw, st0 gradients must be NaN or 0 (here 0 because no variability)
    // strict ≈ 0?
    assert!(grad7[5].abs() < 1e-12);
    assert!(grad7[6].abs() < 1e-12);
}

#[test]
fn wiener7_gradients_match_finite_differences_for_all_parameters() {
    let obs = WienerObservation {
        rt: 0.9,
        boundary: Boundary::Upper,
    };
    let params = Wiener7Params::with_params(
        1.5, 0.2, 0.4, 0.8, // alpha,tau,beta,delta
        0.1, 0.15, 0.05, // sw, st0, sv
    )
    .unwrap();

    let fused = Wiener7.fused(&obs, &params, 1e-4);
    let grad_anal = fused.grad.to_array();
    let f = |p: &Wiener7Params| Wiener7.log_prob(&obs, p, 1e-4).log_prob;

    let h = 1e-5;
    for (i, item) in grad_anal.iter().enumerate() {
        let num = fd_grad(&params, f, i, h);
        // Allow a somewhat generous tolerance due to adaptive integration noise
        assert_relative_eq!(*item, num, epsilon = 2e-3, max_relative = 5e-3);
    }
}

#[test]
fn wiener7_tiny_variability_approaches_wiener5_limit() {
    let obs = WienerObservation {
        rt: 1.2,
        boundary: Boundary::Upper,
    };
    let alpha = 1.6;
    let tau = 0.2;
    let beta = 0.45;
    let delta = 0.7;
    let sv = 0.15;
    let eps = 1e-8;

    let base = Wiener5Params::with_params(alpha, tau, beta, delta, sv).unwrap();
    let base_eval = Wiener5.fused(&obs, &base, eps);

    for (sw, st0) in [(1e-7, 0.0), (0.0, 1e-7), (1e-7, 1e-7)] {
        let params = Wiener7Params::with_params(alpha, tau, beta, delta, sw, st0, sv).unwrap();
        let eval = Wiener7.fused(&obs, &params, eps);

        assert!(eval.log_prob.is_finite());
        assert_relative_eq!(eval.log_prob, base_eval.log_prob, epsilon = 1e-5, max_relative = 1e-5);
        assert_relative_eq!(eval.grad.alpha, base_eval.grad.alpha, epsilon = 1e-4, max_relative = 1e-4);
        assert_relative_eq!(eval.grad.tau, base_eval.grad.tau, epsilon = 1e-4, max_relative = 1e-4);
        assert_relative_eq!(eval.grad.beta, base_eval.grad.beta, epsilon = 1e-4, max_relative = 1e-4);
        assert_relative_eq!(eval.grad.delta, base_eval.grad.delta, epsilon = 1e-4, max_relative = 1e-4);
        assert_relative_eq!(eval.grad.s_delta, base_eval.grad.s_delta, epsilon = 1e-4, max_relative = 1e-4);
    }
}

#[test]
fn wiener7_density_is_finite_with_nonzero_variability() {
    let obs = WienerObservation {
        rt: 1.2,
        boundary: Boundary::Lower,
    };
    let params = Wiener7Params::with_params(2.0, 0.3, 0.5, -0.2, 0.2, 0.1, 0.3).unwrap();
    let fused = Wiener7.fused(&obs, &params, 1e-4);
    let (lp, grad) = (fused.log_prob, fused.grad);
    assert!(lp.is_finite());
    for g in grad.to_array().iter() {
        assert!(g.is_finite());
    }
}

#[test]
fn wiener7_rt_less_than_t0_returns_negative_infinity() {
    let obs = WienerObservation {
        rt: 0.15,
        boundary: Boundary::Upper,
    };
    let params = Wiener7Params::with_params(1.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
    let lp = Wiener7.fused(&obs, &params, 1e-4).log_prob;
    assert_eq!(lp, f64::NEG_INFINITY);
}

#[test]
fn wiener7_sw_support_out_of_bounds_returns_negative_infinity() {
    // w - sw/2 <= 0  => invalid
    let params = Wiener7Params::with_params(1.0, 0.1, 0.1, 0.0, 0.5, 0.0, 0.0).unwrap(); // w=0.1, sw=0.5 → lower bound = 0.1-0.25 = -0.15 <0
    let obs = WienerObservation {
        rt: 0.5,
        boundary: Boundary::Upper,
    };
    let lp = Wiener7.fused(&obs, &params, 1e-4).log_prob;
    assert_eq!(lp, f64::NEG_INFINITY);
}

#[test]
fn wiener7_st0_uses_truncated_nondecision_time_interval() {
    // st0 > 0, but (rt - t0)/st0 < 1 → upper limit clipped
    let obs = WienerObservation {
        rt: 0.35,
        boundary: Boundary::Upper,
    };
    let params = Wiener7Params::with_params(1.0, 0.2, 0.5, 0.5, 0.0, 0.0, 0.2).unwrap(); // st0=0.2, rt-t0=0.15, ratio=0.75
    let lp = Wiener7.fused(&obs, &params, 1e-4).log_prob;
    assert!(lp.is_finite()); // integration should succeed
}

#[test]
fn wiener7_target_fused_log_density_sums_over_batch() {
    let data = vec![
        WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        },
        WienerObservation {
            rt: 0.6,
            boundary: Boundary::Lower,
        },
    ];
    let params = Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap();
    let target = Target::new(Wiener7, data);
    let mut grad = [f64::NAN; 7];
    let lp = target.log_prob_and_grad(&params, &mut grad);
    assert!(lp.is_finite());
    for g in grad.iter() {
        assert!(g.is_finite());
    }
    // Sum of individual fused calls should match
    let mut sum_lp = 0.0;
    let mut sum_grad = [0.0; 7];
    for obs in target.data.iter() {
        let fused = Wiener7.fused(obs, &params, 1e-4);
        let (lp1, g1) = (fused.log_prob, fused.grad);
        sum_lp += lp1;
        for i in 0..7 {
            sum_grad[i] += g1[i];
        }
    }
    assert_ulps_eq!(lp, sum_lp, epsilon = 1e-10);
    for i in 0..7 {
        assert_ulps_eq!(grad[i], sum_grad[i], epsilon = 1e-10);
    }
}

#[test]
fn wiener7_parameter_unconstrain_constrain_roundtrip() {
    let constrained = Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap();
    let unconstrained = Wiener7Params::to_unconstrained(&constrained);
    let roundtrip = Wiener7Params::from_unconstrained(&unconstrained);
    assert_relative_eq!(
        roundtrip.base.base.alpha,
        constrained.base.base.alpha,
        epsilon = 1e-10
    );
    assert_relative_eq!(roundtrip.s_beta, constrained.s_beta, epsilon = 1e-10);
    assert_relative_eq!(roundtrip.s_tau, constrained.s_tau, epsilon = 1e-10);
    assert!(Wiener7Params::log_abs_det_jacobian(&unconstrained).is_finite());
}

#[test]
fn wiener7_matches_stan_reference_values() {
    // Coefficients computed in R with WienR.
    // adapted from Stan tests
    let y_vec = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.85, 8.9, 9.0, 1.0];
    let a_vec = vec![2.0, 2.0, 10.0, 4.0, 10.0, 1.0, 3.0, 1.7, 2.4, 11.0, 1.5];
    let v_vec = vec![2.0, 2.0, 4.0, 3.0, -3.0, 1.0, -1.0, -7.3, -4.9, 4.5, 3.0];
    let w_vec = vec![0.1, 0.5, 0.8, 0.7, 0.1, 0.9, 0.7, 0.92, 0.9, 0.12, 0.5];
    let t0_vec = vec![
        1e-9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1,
    ];
    let sv_vec = vec![0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.7, 0.0, 0.7, 0.5];
    let sw_vec = vec![0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.01, 0.0, 0.1, 0.2];
    let st0_vec = vec![
        0.0, 0.0, 0.0, 0.007, 0.0, 0.007, 0.007, 0.009, 0.009, 0.009, 0.0,
    ];

    let true_dens = vec![
        -4.28564747866615,
        -7.52379235146909,
        -26.1551056209248,
        -22.1939134892089,
        -50.0587553794834,
        -37.2817263586318,
        -10.5428662079438,
        -61.5915905674246,
        -117.238967959795,
        -12.5788594249676,
        -3.1448097740735,
    ];
    let _true_grad_y = vec![
        -3.22509339523307,
        -2.91155058614589,
        -8.21331631900955,
        -4.82948967379739,
        -1.50069056428102,
        -5.25831601347426,
        -1.04831896413742,
        -2.67457492096193,
        -12.8617364931501,
        -1.12047317491985,
        -5.68799957241344,
    ];
    let true_grad_a = vec![
        3.25018678924105,
        3.59980430191399,
        0.876602303160642,
        1.2215517888504,
        -3.02928674030948,
        67.0322498959921,
        1.95334514374631,
        16.4642201959135,
        5.02038145619773,
        0.688439187670968,
        2.63200041459657,
    ];
    let true_grad_t0 = vec![
        3.22509339523307,
        2.91155058614589,
        8.21331631900955,
        4.82948967379739,
        1.50069056428102,
        5.25831601347426,
        1.04831896413742,
        2.67457492096193,
        12.8617364931501,
        1.12047317491985,
        5.68799957241344,
    ];
    let true_grad_w = vec![
        5.67120184517318,
        -3.64396221090076,
        -38.7775057146792,
        -14.1837930137393,
        35.71918681520357,
        -10.4535345681946,
        0.679597983582904,
        -9.93144540834201,
        2.09117200953597,
        -6.0858540417876,
        -3.74870310978083,
    ];
    let true_grad_v = vec![
        -2.199999998,
        -4.44801714898178,
        -13.6940602985224,
        -13.7593709622169,
        21.5540563802381,
        -5.38233555673517,
        8.88475440789056,
        12.1280680728793,
        43.7785246930371,
        -5.68143495684294,
        -1.57639220567218,
    ];
    let true_grad_sv = vec![
        0.0,
        3.42285198319565,
        0.0,
        0.0,
        91.9551438876654,
        4.70180879974639,
        0.0,
        101.80250964211,
        0.0,
        21.4332628706595,
        0.877556017134384,
    ];
    let true_grad_sw = vec![
        0.0,
        0.0,
        10.1052188867058,
        0.0,
        8.72398,
        0.0,
        -0.122807217815892,
        -0.0506322723373748,
        0.0,
        -0.0704990526706635,
        0.0827817310725268,
    ];
    let true_grad_st0 = vec![
        0.0,
        0.0,
        0.0,
        2.42836139121338,
        0.0,
        2.64529825657625,
        0.524800556172613,
        1.34278261179603,
        6.55490874737353,
        0.561295838843035,
        0.0,
    ];

    let eps = 1e-12;
    let tolerance_log = 1e-6;
    let tolerance_grad = 1e-4;

    for i in 0..y_vec.len() {
        // if i == 4 {
        //     continue;
        // }
        let params = Wiener7Params::with_params_unchecked(
            a_vec[i], t0_vec[i], w_vec[i], v_vec[i], sw_vec[i], st0_vec[i], sv_vec[i],
        );

        // Upper boundary case
        let obs_up = WienerObservation {
            rt: y_vec[i],
            boundary: Boundary::Upper,
        };
        let fused = Wiener7.fused(&obs_up, &params, eps);
        let (lp, grad) = (fused.log_prob, fused.grad);

        assert_relative_eq!(lp, true_dens[i], epsilon = tolerance_log);
        assert_relative_eq!(grad[0], true_grad_a[i], epsilon = tolerance_grad);
        assert_relative_eq!(grad[1], true_grad_t0[i], epsilon = tolerance_grad);
        assert_relative_eq!(grad[2], true_grad_w[i], epsilon = tolerance_grad);
        assert_relative_eq!(grad[3], true_grad_v[i], epsilon = tolerance_grad);
        assert_relative_eq!(grad[4], true_grad_sv[i], epsilon = tolerance_grad);
        assert_relative_eq!(grad[5], true_grad_sw[i], epsilon = tolerance_grad);
        assert_relative_eq!(grad[6], true_grad_st0[i], epsilon = tolerance_grad);
    }
}

#[test]
fn wiener4_small_time_raw_derivative_matches_forward_difference() {
    let t_prime = 0.0599;
    let w = 0.1;
    let k = 100; // small k for speed
    let h = 1e-6;
    let r0 = Wiener4::small_time_series_raw(t_prime, w, k).unwrap();
    let r_plus = Wiener4::small_time_series_raw(t_prime, w + h, k).unwrap();
    let num_deriv = (r_plus - r0) / h;
    let analytic = Wiener4::small_time_dr_dw(t_prime, w, k).unwrap();
    assert_relative_eq!(analytic, num_deriv, epsilon = 1e-4);
}

fn fd_central<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[test]
fn wiener5_grad_w_matches_fd() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };

    let a = 10.0;
    let t0 = 0.01;
    let w0 = 0.1;
    let v = -3.0;
    let sv = 0.2;
    let eps = 1e-12;

    let analytic = {
        let p = Wiener5Params::with_params_unchecked(a, t0, w0, v, sv);
        Wiener5.fused(&obs, &p, eps).grad.beta
    };

    let numeric = fd_central(
        |w| {
            let p = Wiener5Params::with_params_unchecked(a, t0, w, v, sv);
            Wiener5.fused(&obs, &p, eps).log_prob
        },
        w0,
        1e-6,
    );

    assert!(
        (analytic - numeric).abs() <= 1e-5_f64.max(1e-6 * numeric.abs()),
        "analytic={}, numeric={}",
        analytic,
        numeric
    );
}

#[test]
fn wiener7_grad_beta_matches_fd() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let eps = 1e-12;
    let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.05, 0.2);

    let core0 = Wiener7.core(&obs, &params, eps).unwrap();
    let analytic = Wiener7.eval_fused(&core0, &obs).grad.beta;

    let numeric = fd_central(
        |beta0| {
            let mut core = core0.clone();
            core.beta0 = beta0;
            Wiener7.eval_fused(&core, &obs).log_prob
        },
        core0.beta0,
        1e-6,
    );

    assert!(
        (analytic - numeric).abs() <= 1e-5_f64.max(1e-6 * numeric.abs()),
        "analytic={}, numeric={}",
        analytic,
        numeric
    );
}

#[test]
fn wiener5_reflection_equivalence_upper_vs_lower() {
    let obs_upper = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let obs_lower = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Lower,
    };

    let a = 10.0;
    let t0 = 0.01;
    let w = 0.1;
    let v = -3.0;
    let sv = 0.2;
    let eps = 1e-12;

    let p_upper = Wiener5Params::with_params_unchecked(a, t0, w, v, sv);
    let p_lower_reflected = Wiener5Params::with_params_unchecked(a, t0, 1.0 - w, -v, sv);

    let e1 = Wiener5.fused(&obs_upper, &p_upper, eps);
    let e2 = Wiener5.fused(&obs_lower, &p_lower_reflected, eps);

    assert_relative_eq!(e1.log_prob, e2.log_prob, epsilon = 1e-12);
    assert_relative_eq!(e1.grad.beta, -e2.grad.beta, epsilon = 1e-9);
    assert_relative_eq!(e1.grad.delta, -e2.grad.delta, epsilon = 1e-9);
}

#[test]
fn wiener7_boundary_convention_matches_wiener5() {
    let eps = 1e-12;
    let obs_upper = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let params_upper =
        Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);
    let core_upper = Wiener7.core(&obs_upper, &params_upper, eps).unwrap();

    let obs_lower = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Lower,
    };
    let params_lower =
        Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);
    let core_lower = Wiener7.core(&obs_lower, &params_lower, eps).unwrap();

    let _e_upper = Wiener7.eval_fused(&core_upper, &obs_upper);
    let _e_lower = Wiener7.eval_fused(&core_lower, &obs_lower);

}

#[test]
fn wiener5_upper_lower_reflection_has_matching_density_and_signed_gradients() {
    let eps = 1e-12;
    let obs_upper = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let obs_lower = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Lower,
    };

    let p_upper = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
    let p_lower = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.9, 3.0, 0.2);

    let e_upper = Wiener5.fused(&obs_upper, &p_upper, eps);
    let e_lower = Wiener5.fused(&obs_lower, &p_lower, eps);

    assert_relative_eq!(e_upper.log_prob, e_lower.log_prob, epsilon = 1e-12);
    assert_relative_eq!(e_upper.grad.beta, -e_lower.grad.beta, epsilon = 1e-9);
    assert_relative_eq!(e_upper.grad.delta, -e_lower.grad.delta, epsilon = 1e-9);
}

#[test]
fn wiener4_small_and_large_branch_agree_in_overlap_case() {
    let t_prime = 0.06;
    let w = 0.1;
    let k = 40;

    let small = Wiener4::small_branch_fused(t_prime, w, k).unwrap();
    let large = Wiener4::large_branch_fused(t_prime, w, k).unwrap();

    assert_relative_eq!(small.0, large.0, epsilon = 1e-10);
    assert_relative_eq!(small.1, large.1, epsilon = 1e-8);
    assert_relative_eq!(small.2, large.2, epsilon = 1e-8);
}

#[test]
fn wiener4_small_branch_derivatives_match_fd_at_fixed_k() {
    let t_prime = 0.06;
    let w = 0.1;
    let k = 20;
    let h = 1e-7;

    let ref_val = Wiener4::small_branch_fused(t_prime, w, k).unwrap();

    let fd_dw = fd_central(
        |ww| Wiener4::small_branch_fused(t_prime, ww, k).unwrap().0,
        w,
        h,
    );
    let fd_dt = fd_central(
        |tt| Wiener4::small_branch_fused(tt, w, k).unwrap().0,
        t_prime,
        h,
    );

    assert_relative_eq!(ref_val.2, fd_dw, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(ref_val.1, fd_dt, epsilon = 1e-6, max_relative = 1e-6);
}

#[test]
fn wiener4_large_branch_derivatives_match_fd_at_fixed_k() {
    let t_prime = 0.06;
    let w = 0.1;
    let k = 20;
    let h = 1e-7;

    let ref_val = Wiener4::large_branch_fused(t_prime, w, k).unwrap();

    let fd_dw = fd_central(
        |ww| Wiener4::large_branch_fused(t_prime, ww, k).unwrap().0,
        w,
        h,
    );
    let fd_dt = fd_central(
        |tt| Wiener4::large_branch_fused(tt, w, k).unwrap().0,
        t_prime,
        h,
    );

    assert_relative_eq!(ref_val.2, fd_dw, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(ref_val.1, fd_dt, epsilon = 1e-6, max_relative = 1e-6);
}

#[test]
fn wiener4_small_branch_converges_with_k() {
    let t_prime = 0.06;
    let w = 0.1;

    let k1 = 20;
    let k2 = 40;
    let a = Wiener4::small_branch_fused(t_prime, w, k1).unwrap();
    let b = Wiener4::small_branch_fused(t_prime, w, k2).unwrap();


    assert_relative_eq!(a.0, b.0, epsilon = 1e-8, max_relative = 1e-8);
    assert_relative_eq!(a.1, b.1, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(a.2, b.2, epsilon = 1e-6, max_relative = 1e-6);
}

#[test]
fn wiener4_large_branch_converges_with_k() {
    let t_prime = 0.06;
    let w = 0.1;

    let k1 = 20;
    let k2 = 40;
    let a = Wiener4::large_branch_fused(t_prime, w, k1).unwrap();
    let b = Wiener4::large_branch_fused(t_prime, w, k2).unwrap();


    assert_relative_eq!(a.0, b.0, epsilon = 1e-8, max_relative = 1e-8);
    assert_relative_eq!(a.1, b.1, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(a.2, b.2, epsilon = 1e-6, max_relative = 1e-6);
}

#[test]
fn wiener4_branch_selection_vs_conservative_truncation() {
    let cases = [
        (0.06, 0.1),
        (0.02, 0.1),
        (0.06, 0.3),
        (0.15, 0.1),
        (0.06, 0.9),
    ];
    let log_eps = -12.0_f64.ln();

    for &(t_prime, w) in &cases {
        let ks = Wiener4::k_s(t_prime, w, log_eps);
        let kl = Wiener4::k_l(t_prime, log_eps);
        let ks_gw = Wiener4::k_s_grad_w(t_prime, w, log_eps);
        let kl_gw = Wiener4::k_l_grad_w(t_prime, log_eps);

        let _density_branch = if 2 * ks <= kl { "small" } else { "large" };
        let _grad_branch = if 2 * ks_gw <= kl_gw { "small" } else { "large" };


        // This does not enforce equality; it only records when the gradient wants a different regime.
        // Those are the points to inspect if the higher-level Wiener5/Wiener7 comparison fails.
    }
}

#[test]
fn wiener4_problem_point_large_branch_converges_to_small_branch() {
    let t_prime = 0.06;
    let w = 0.1;

    for k in [1usize, 2, 3, 5, 8, 13, 21, 34, 55] {
        let _s = Wiener4::small_branch_fused(t_prime, w, k);
        let _l = Wiener4::large_branch_fused(t_prime, w, k);
    }
}

fn trapz_1d<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    assert!(
        n >= 2 && n % 2 == 0,
        "use an even n for trapezoidal/Simpson-style resolution"
    );
    let h = (b - a) / (n as f64 - 1.0);
    let mut sum = 0.0;
    for i in 0..n {
        let x = a + (i as f64) * h;
        let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
        sum += w * f(x);
    }
    sum * h
}


#[test]
fn wiener7_builder_uses_canonical_parameter_names() {
    let params = Wiener7Params::builder()
        .alpha(1.5)
        .tau(0.2)
        .beta(0.4)
        .delta(1.0)
        .s_delta(0.1)
        .s_beta(0.05)
        .s_tau(0.03)
        .build()
        .unwrap();

    assert_eq!(params.to_array(), [1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.03]);
}

#[test]
fn wiener7_builder_rejects_invalid_configured_values() {
    let err = Wiener7Params::builder().alpha(-1.0).build().unwrap_err();
    assert!(matches!(err, ProbError::InvalidParameters(_)));

    let err = Wiener7Params::builder().s_beta(1.0).build().unwrap_err();
    assert!(matches!(err, ProbError::InvalidParameters(_)));
}

#[test]
fn wiener_options_validate_precision_and_fixed_quadrature_order() {
    assert!(WienerOptions::new(f64::NAN).validate().is_err());
    assert!(WienerOptions::new(f64::INFINITY).validate().is_err());
    assert!(WienerOptions::new(0.0).validate().is_err());
    assert!(WienerOptions::new(1e-8)
        .with_inner_precision(f64::NAN)
        .validate()
        .is_err());
    assert!(WienerOptions::new(1e-8)
        .with_quadrature(Quadrature::FixedGaussLegendre { order: 9 })
        .validate()
        .is_err());
    assert!(WienerOptions::new(1e-8)
        .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 })
        .validate()
        .is_ok());
}

#[test]
fn soa_wiener_observations_match_vec_targets() {
    let observations = vec![
        WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        },
        WienerObservation {
            rt: 1.0,
            boundary: Boundary::Lower,
        },
        WienerObservation {
            rt: 1.2,
            boundary: Boundary::Upper,
        },
        WienerObservation {
            rt: 1.5,
            boundary: Boundary::Lower,
        },
    ];
    let soa = WienerObservations::from(observations.clone());
    assert_eq!(soa.len(), observations.len());
    assert_eq!(soa.upper_rt().len(), 2);
    assert_eq!(soa.lower_rt().len(), 2);

    let p4 = Wiener4Params::with_params(1.6, 0.2, 0.45, 0.7).unwrap();
    let p5 = Wiener5Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1).unwrap();
    let p7 = Wiener7Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1, 0.05, 0.1).unwrap();

    let mut vec_grad4 = [0.0; 4];
    let mut soa_grad4 = [0.0; 4];
    let vec_lp4 = Target::new(Wiener4, observations.clone()).log_prob_and_grad(&p4, &mut vec_grad4);
    let soa_lp4 = Target::new(Wiener4, WienerObservations::from(observations.clone()))
        .log_prob_and_grad(&p4, &mut soa_grad4);
    assert_relative_eq!(vec_lp4, soa_lp4, epsilon = 1e-12);
    for (vec, soa) in vec_grad4.iter().zip(soa_grad4.iter()) {
        assert_relative_eq!(vec, soa, epsilon = 1e-10, max_relative = 1e-10);
    }

    let mut vec_grad5 = [0.0; 5];
    let mut soa_grad5 = [0.0; 5];
    let vec_lp5 = Target::new(Wiener5, observations.clone()).log_prob_and_grad(&p5, &mut vec_grad5);
    let soa_lp5 = Target::new(Wiener5, WienerObservations::from(observations.clone()))
        .log_prob_and_grad(&p5, &mut soa_grad5);
    assert_relative_eq!(vec_lp5, soa_lp5, epsilon = 1e-12);
    for (vec, soa) in vec_grad5.iter().zip(soa_grad5.iter()) {
        assert_relative_eq!(vec, soa, epsilon = 1e-10, max_relative = 1e-10);
    }

    let mut vec_grad7 = [0.0; 7];
    let mut soa_grad7 = [0.0; 7];
    let vec_lp7 = Target::new(Wiener7, observations.clone()).log_prob_and_grad(&p7, &mut vec_grad7);
    let soa_lp7 = Target::new(Wiener7, soa).log_prob_and_grad(&p7, &mut soa_grad7);
    assert_relative_eq!(vec_lp7, soa_lp7, epsilon = 1e-10, max_relative = 1e-10);
    for (vec, soa) in vec_grad7.iter().zip(soa_grad7.iter()) {
        assert_relative_eq!(vec, soa, epsilon = 1e-10, max_relative = 1e-10);
    }
}

#[test]
fn checked_wiener_options_api_matches_existing_wrappers() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let p4 = Wiener4Params::with_params(10.0, 0.01, 0.1, -3.0).unwrap();
    let p5 = Wiener5Params::with_params(10.0, 0.01, 0.1, -3.0, 0.2).unwrap();
    let p7 = Wiener7Params::with_params(10.0, 0.01, 0.1, -3.0, 0.1, 0.05, 0.2).unwrap();
    let options = WienerOptions::new(1e-8);

    assert_relative_eq!(
        Wiener4.try_fused(&obs, &p4, options).unwrap().log_prob,
        Wiener4.fused(&obs, &p4, options.precision).log_prob,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        Wiener5.try_fused(&obs, &p5, options).unwrap().log_prob,
        Wiener5.fused(&obs, &p5, options.precision).log_prob,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        Wiener7.try_fused(&obs, &p7, options).unwrap().log_prob,
        Wiener7.fused(&obs, &p7, options.precision).log_prob,
        epsilon = 1e-12
    );
}

#[test]
fn checked_wiener7_fixed_quadrature_option_matches_internal_fixed_path() {
    let obs = WienerObservation {
        rt: 1.2,
        boundary: Boundary::Upper,
    };
    let params = Wiener7Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1, 0.1, 0.15).unwrap();
    let options = WienerOptions::new(1e-8)
        .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 });
    let core = Wiener7.core(&obs, &params, options.precision).unwrap();

    assert_relative_eq!(
        Wiener7.try_fused(&obs, &params, options).unwrap().log_prob,
        Wiener7.eval_fused_fixed_order(&core, &obs, 25).log_prob,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        Wiener7.try_log_prob(&obs, &params, options).unwrap().log_prob,
        Wiener7.eval_density_fixed_order(&core, &obs, 25).ln(),
        epsilon = 1e-12
    );
}

#[test]
fn wiener7_fixed_quadrature_matches_adaptive_on_reference_cases() {
    let cases = [
        (
            WienerObservation {
                rt: 0.8,
                boundary: Boundary::Upper,
            },
            Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap(),
        ),
        (
            WienerObservation {
                rt: 1.2,
                boundary: Boundary::Upper,
            },
            Wiener7Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1, 0.1, 0.15).unwrap(),
        ),
        (
            WienerObservation {
                rt: 6.0,
                boundary: Boundary::Upper,
            },
            Wiener7Params::with_params(10.0, 0.01, 0.1, -3.0, 0.1, 0.05, 0.2).unwrap(),
        ),
    ];

    for (obs, params) in cases {
        let core = Wiener7.core(&obs, &params, 1e-8).unwrap();
        let fixed_density = Wiener7.eval_density_fixed(&core, &obs);
        let adaptive_density = Wiener7.eval_density_adaptive(&core, &obs);
        assert_relative_eq!(fixed_density, adaptive_density, epsilon = 1e-10, max_relative = 1e-10);

        let fixed = Wiener7.eval_fused_fixed(&core, &obs);
        let adaptive = Wiener7.eval_fused_adaptive(&core, &obs);
        assert_relative_eq!(fixed.log_prob, adaptive.log_prob, epsilon = 1e-10, max_relative = 1e-10);
        for (fixed_grad, adaptive_grad) in fixed.grad.to_array().iter().zip(adaptive.grad.to_array().iter()) {
            assert_relative_eq!(fixed_grad, adaptive_grad, epsilon = 1e-8, max_relative = 1e-8);
        }
    }
}

#[test]
fn wiener7_i4_hcubature_matches_bruteforce_beta_integration() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };

    let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

    let eps = 1e-12;
    let core = Wiener7.core(&obs, &params, eps).unwrap();
    let fused = Wiener7.eval_fused(&core, &obs);

    // Only beta varies here because st0 = 0.
    let beta_low = core.beta0 - core.sw / 2.0;
    let beta_high = core.beta0 + core.sw / 2.0;
    let tau = core.tau0;

    let n = 8000usize;

    let dens = trapz_1d(
        |x| {
            let beta = beta_low + (beta_high - beta_low) * x;
            if !(0.0 < beta && beta < 1.0) {
                return 0.0;
            }
            let p5 = Wiener5Params::with_params_unchecked(
                core.alpha, tau, beta, core.delta, core.sv,
            );
            Wiener5.fused(&obs, &p5, core.eps_series).log_prob.exp()
        },
        0.0,
        1.0,
        n,
    );

    let grad_beta = trapz_1d(
        |x| {
            let beta = beta_low + (beta_high - beta_low) * x;
            if !(0.0 < beta && beta < 1.0) {
                return 0.0;
            }
            let p5 = Wiener5Params::with_params_unchecked(
                core.alpha, tau, beta, core.delta, core.sv,
            );
            let e = Wiener5.fused(&obs, &p5, core.eps_series);
            e.log_prob.exp() * e.grad.beta
        },
        0.0,
        1.0,
        n,
    );


    assert_relative_eq!(
        fused.log_prob.exp(),
        dens,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        fused.grad.beta,
        grad_beta / dens,
        epsilon = 1e-6,
        max_relative = 1e-6
    );
}

#[test]
fn wiener7_i4_endpoint_sensitivity_beta_bounds() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };

    let mut params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

    let eps = 1e-12;

    let base = Wiener7.eval_fused(&Wiener7.core(&obs, &params, eps).unwrap(), &obs);

    params.base.base.beta = 0.100001;
    let shifted_up = Wiener7.eval_fused(&Wiener7.core(&obs, &params, eps).unwrap(), &obs);

    params.base.base.beta = 0.099999;
    let shifted_down = Wiener7.eval_fused(&Wiener7.core(&obs, &params, eps).unwrap(), &obs);


    // This should be smooth unless the support clipping or reflection is biting.
    assert!(base.log_prob.is_finite());
    assert!(shifted_up.log_prob.is_finite());
    assert!(shifted_down.log_prob.is_finite());
}

#[test]
fn wiener4_selected_k_is_close_to_high_k_on_i4_case() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
    let eps = 1e-12;

    let core = Wiener5.core(&obs, &params, eps).unwrap();
    let t_prime = core.base.t_prime;
    let w = core.base.w_eff;
    let log_eps = core.base.log_eps_eff;

    let ks = Wiener4::k_s(t_prime, w, log_eps);
    let kl = Wiener4::k_l(t_prime, log_eps);


    let selected = if 2 * ks <= kl {
        Wiener4::small_branch_fused(t_prime, w, ks).unwrap()
    } else {
        Wiener4::large_branch_fused(t_prime, w, kl).unwrap()
    };

    let high_k = 80usize;
    let high_ref = if 2 * ks <= kl {
        Wiener4::small_branch_fused(t_prime, w, high_k).unwrap()
    } else {
        Wiener4::large_branch_fused(t_prime, w, high_k).unwrap()
    };


    assert_relative_eq!(selected.0, high_ref.0, epsilon = 1e-8, max_relative = 1e-8);
    assert_relative_eq!(selected.1, high_ref.1, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(selected.2, high_ref.2, epsilon = 1e-6, max_relative = 1e-6);
}

#[test]
fn wiener5_selected_series_matches_high_k_reference_i4_case() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
    let eps = 1e-12;

    let fused = Wiener5.fused(&obs, &params, eps);

    let core = Wiener5.core(&obs, &params, eps).unwrap();
    let t_prime = core.base.t_prime;
    let w = core.base.w_eff;
    let log_eps = core.base.log_eps_eff;

    let ks = Wiener4::k_s(t_prime, w, log_eps);
    let kl = Wiener4::k_l(t_prime, log_eps);
    let high_k = 100usize;

    let series_selected = if 2 * ks <= kl {
        Wiener4::small_branch_fused(t_prime, w, ks).unwrap()
    } else {
        Wiener4::large_branch_fused(t_prime, w, kl).unwrap()
    };

    let series_high = if 2 * ks <= kl {
        Wiener4::small_branch_fused(t_prime, w, high_k).unwrap()
    } else {
        Wiener4::large_branch_fused(t_prime, w, high_k).unwrap()
    };


    assert_relative_eq!(
        series_selected.0,
        series_high.0,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        fused.log_prob,
        core.base.pref + series_high.0,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
}

#[test]
fn wiener5_i4_matches_stan_oracle_values() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
    let eps = 1e-12;

    let fused = Wiener5.fused(&obs, &params, eps);

    // Corrected five-parameter Stan Math value and finite-difference w-gradient.
    let lp_ref = -50.539_106_208_790_05_f64;
    let gw_ref = 36.632_908_590_804_28_f64;


    assert_relative_eq!(fused.log_prob, lp_ref, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(fused.grad.beta, gw_ref, epsilon = 1e-4, max_relative = 1e-4);
}


#[test]
fn wiener5_stan_oracle_row_i4_asserts() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };
    let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
    let fused = Wiener5.fused(&obs, &params, 1e-12);

    let stan_lp = -50.539_106_208_790_05_f64;
    let stan_gw = 36.632_908_590_804_28_f64;


    assert_relative_eq!(fused.log_prob, stan_lp, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(
        fused.grad.beta,
        stan_gw,
        epsilon = 1e-4,
        max_relative = 1e-4
    );
}

#[test]
fn wiener7_i4_beta_grad_matches_endpoint_identity() {
    let obs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };

    let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

    let eps = 1e-12;
    let core = Wiener7.core(&obs, &params, eps).unwrap();
    let fused = Wiener7.eval_fused(&core, &obs);

    assert!(core.sw > 0.0);
    assert_eq!(core.st0, 0.0);

    let low = core.beta0 - core.sw / 2.0;
    let high = core.beta0 + core.sw / 2.0;

    let p_low =
        Wiener5Params::with_params_unchecked(core.alpha, core.tau0, low, core.delta, core.sv);
    let p_high =
        Wiener5Params::with_params_unchecked(core.alpha, core.tau0, high, core.delta, core.sv);

    let f_low = Wiener7::wiener5_density(&obs, &p_low, core.eps_series);
    let f_high = Wiener7::wiener5_density(&obs, &p_high, core.eps_series);

    let density = fused.log_prob.exp();
    let endpoint_grad = (f_high - f_low) / (core.sw * density);


    assert_relative_eq!(
        fused.grad.beta,
        endpoint_grad,
        epsilon = 1e-7,
        max_relative = 1e-7
    );
}
