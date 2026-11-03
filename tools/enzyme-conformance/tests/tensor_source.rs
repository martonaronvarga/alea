//! Actual compiled source parity, deliberately independent of nested Enzyme AD.
#[path = "../models.rs"]
mod models;
mod matrix {
    include!(concat!(env!("OUT_DIR"), "/tensor_matrix_primal.rs"));
}
mod control {
    include!(concat!(env!("OUT_DIR"), "/tensor_control_primal.rs"));
}
mod empty {
    include!(concat!(env!("OUT_DIR"), "/tensor_empty_primal.rs"));
}
use alea::target::LogDensityGradient;

#[test]
fn compiled_tensor_regions_match_native_in_both_branches_and_after_loop_execution() {
    for (name, source) in [
        ("tensor_matrix", matrix::log_density as fn(&[f64; 2]) -> f64),
        ("tensor_control", control::log_density),
    ] {
        let target = models::all()
            .unwrap()
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1
            .into_target()
            .unwrap();
        for q in [[0.3, -0.4], [-10., 1.], [2., 0.5], [0., 0.]] {
            let expected = target.logp_grad(&q, &mut [0.; 2]).unwrap();
            assert!((source(&q) - expected).abs() < 1e-12 * (1. + expected.abs()));
        }
        assert!(!source(&[f64::NAN, 0.]).is_finite());
        assert!(!source(&[f64::MAX, f64::MAX]).is_finite());
    }
}

#[test]
fn empty_region_result_does_not_erase_selected_branch_domain_errors() {
    let target = models::tensor_empty().unwrap().into_target().unwrap();
    assert!(target.logp_grad(&[-1.], &mut [0.]).is_err());
    assert!(!empty::log_density(&[-1.]).is_finite());
    for value in [0., 1., 10.] {
        assert_eq!(target.logp_grad(&[value], &mut [0.]).unwrap(), 0.);
        assert_eq!(empty::log_density(&[value]), 0.);
    }
}
