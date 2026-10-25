use alea_distributions::wiener::{
    Boundary, Wiener5, Wiener5Params, Wiener7, Wiener7Params, WienerObservation, WienerOptions,
};

#[test]
fn stan_3329_tail_regressions_have_finite_values_and_scores() {
    // Independent regression values from Stan's underflow-fix commit c8b33c4.
    for (sw, st, expected) in [
        (1e-8, 0.0, -4_982.643_752_563_856),
        (0.0, 0.001, -4_998.068_538_408_878),
        (1e-8, 0.001, -4_998.068_538_408_461),
    ] {
        let p = Wiener7Params::with_params(0.1, 0.0, 0.0001, -10.0, sw, st, 0.1).unwrap();
        let obs = WienerObservation {
            rt: 1e-6,
            boundary: Boundary::Upper,
        };
        let options = WienerOptions::new(1e-8).with_inner_precision(1e-12);
        let e = Wiener7.try_fused(&obs, &p, options).unwrap();
        let value = Wiener7.try_log_prob(&obs, &p, options).unwrap();
        assert!(
            (e.log_prob - expected).abs() < 1e-8,
            "sw={sw} st={st}: {} vs {expected}",
            e.log_prob
        );
        assert!((value.log_prob - e.log_prob).abs() < 1e-8);
        assert!(e.grad_array().iter().all(|x| x.is_finite()));
        for i in 0..7 {
            if (i == 5 && sw == 0.0) || (i == 6 && st == 0.0) {
                continue;
            }
            let mut plus = p.to_array();
            let mut minus = plus;
            let h = [1e-7, 1e-12, 1e-8, 1e-5, 1e-6, 2.5e-9, 1e-7][i];
            plus[i] += h;
            minus[i] -= h;
            let hi = Wiener7
                .try_log_prob(&obs, &Wiener7Params::from_array(plus), options)
                .unwrap()
                .log_prob;
            let fd = if i == 1 {
                plus[i] += h;
                let twice = Wiener7
                    .try_log_prob(&obs, &Wiener7Params::from_array(plus), options)
                    .unwrap()
                    .log_prob;
                (-3.0 * e.log_prob + 4.0 * hi - twice) / (2.0 * h)
            } else {
                let lo = Wiener7
                    .try_log_prob(&obs, &Wiener7Params::from_array(minus), options)
                    .unwrap()
                    .log_prob;
                (hi - lo) / (2.0 * h)
            };
            let tolerance = if i == 5 {
                5e-3
            } else {
                5e-5 * fd.abs().max(1.0)
            };
            assert!(
                (fd - e.grad_array()[i]).abs() < tolerance,
                "coordinate{i}: fd={fd} analytic={}",
                e.grad_array()[i]
            );
        }
    }
}

#[test]
fn zero_variability_preserves_wiener5_log_scale_in_extreme_tails() {
    let obs = WienerObservation {
        rt: 1e-6,
        boundary: Boundary::Upper,
    };
    let p5 = Wiener5Params::with_params(0.2, 0.0, 0.1, -10.0, 0.1).unwrap();
    let p7 = Wiener7Params::with_params(0.2, 0.0, 0.1, -10.0, 0.0, 0.0, 0.1).unwrap();
    let expected = Wiener5
        .try_log_prob(&obs, &p5, WienerOptions::default())
        .unwrap()
        .log_prob;
    assert!(expected < -745.0);
    assert_eq!(
        Wiener7
            .try_log_prob(&obs, &p7, WienerOptions::default())
            .unwrap()
            .log_prob,
        expected
    );
    assert_eq!(Wiener7.log_prob(&obs, &p7, 1e-12).log_prob, expected);
}

#[test]
fn unachievable_tail_precision_is_a_numerical_error() {
    let p = Wiener7Params::with_params(0.1, 0.0, 0.0001, -10.0, 1e-8, 0.001, 0.1).unwrap();
    let obs = WienerObservation {
        rt: 1e-6,
        boundary: Boundary::Upper,
    };
    assert!(matches!(
        Wiener7.try_fused(&obs, &p, WienerOptions::new(1e-30)),
        Err(alea_distributions::wiener::ProbError::NumericalError(_))
    ));
}

fn support_grid(large: bool) {
    let times: &[f64] = if large {
        &[1e-6, 1e-5, 1e-4, 0.01, 0.2, 2.0, 10.0]
    } else {
        &[1e-6, 0.2]
    };
    for alpha in [0.1, 0.2] {
        for beta in [0.1, 0.9] {
            for delta in [-10.0, 10.0] {
                for &(sw, st) in &[(1e-4, 0.0), (0.0, 0.001), (1e-4, 0.001)] {
                    for &rt in times {
                        for boundary in [Boundary::Upper, Boundary::Lower] {
                            let p =
                                Wiener7Params::with_params(alpha, 0.0, beta, delta, sw, st, 0.1)
                                    .unwrap();
                            let e = Wiener7
                                .try_fused(
                                    &WienerObservation { rt, boundary },
                                    &p,
                                    WienerOptions::new(1e-7),
                                )
                                .unwrap();
                            assert!(
                                e.log_prob.is_finite()
                                    && e.grad_array().iter().all(|g| g.is_finite()),
                                "{p:?} rt={rt}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn finite_support_tail_grid() {
    support_grid(false);
}

#[test]
#[ignore = "larger finite-support sweep for local numerical reports"]
fn finite_support_extended_grid() {
    support_grid(true);
}
