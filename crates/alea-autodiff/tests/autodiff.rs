#![cfg(feature = "std-autodiff")]
#![feature(autodiff)]
#[path = "../examples/support/models.rs"]
mod models;
use alea_autodiff::autodiff::EnzymeModel;
use alea_autodiff::gradient_check::{GradientCheckOptions, check_gradient};
use alea_core::model::{ConstrainedModel, SumModel, TransformedTarget};
use alea_core::transform::{ParameterLayout, Transform};
use std::autodiff::autodiff_reverse;

#[test]
fn enzyme_matches_analytic_and_finite_difference() {
    let derivatives = [
        models::d_gaussian,
        models::d_correlated,
        models::d_logistic,
        models::d_banana,
        models::d_funnel,
    ];
    for (kind, derivative) in derivatives.into_iter().enumerate() {
        let model = EnzymeModel::new(2, derivative);
        for q in [[0.3, -0.7], [-1.2, 2.3], [0.0, 0.0]] {
            let mut actual = [42.0; 2];
            let value = model.logp_grad(&q, &mut actual).unwrap();
            let mut expected = [0.0; 2];
            let lp = models::analytic(kind, &q, &mut expected).unwrap();
            assert!((value - lp).abs() < 1e-12);
            for i in 0..2 {
                assert!(
                    (actual[i] - expected[i]).abs() < 1e-10,
                    "{}: {actual:?} != {expected:?}",
                    models::NAMES[kind]
                );
            }
            assert_eq!(value, model.logp_grad(&q, &mut actual).unwrap());
        }
        let target = TransformedTarget::new(
            model,
            ParameterLayout::new([
                Transform::positive(1).unwrap(),
                Transform::interval(1, -2.0, 3.0).unwrap(),
            ])
            .unwrap(),
        )
        .unwrap();
        let report =
            check_gradient(&target, &[0.3, -0.7], GradientCheckOptions::default()).unwrap();
        assert!(report.passed(), "{report:?}");
    }
}

#[autodiff_reverse(d_prior, Duplicated, Active)]
fn prior(q: &[f64]) -> f64 {
    -0.5 * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
}

#[test]
fn opaque_wiener_composes_with_enzyme_prior() {
    let model = SumModel::new(EnzymeModel::new(4, d_prior), models::wiener_primitive()).unwrap();
    let target = TransformedTarget::new(model, models::wiener_layout()).unwrap();
    for q in [
        [0.2, -0.5, 0.3, -0.4],
        [-0.7, 0.4, -0.5, 1.2],
        [0.8, -0.1, 0.7, -1.0],
    ] {
        let report = check_gradient(&target, &q, GradientCheckOptions::default()).unwrap();
        assert!(report.passed(), "{report:?}");
    }
}

#[test]
fn repeated_enzyme_and_hybrid_evaluations_allocate_nothing() {
    use alea_core::target::evaluate;
    let model = SumModel::new(EnzymeModel::new(4, d_prior), models::wiener_primitive()).unwrap();
    let target = TransformedTarget::new(model, models::wiener_layout()).unwrap();
    let mut gradient = [0.0; 4];
    let q = [0.2, -0.5, 0.3, -0.4];
    evaluate(&target, &q, &mut gradient).unwrap();
    let allocations = allocation_counter::measure(|| {
        for _ in 0..128 {
            evaluate(&target, &q, &mut gradient).unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[autodiff_reverse(d_with_data, Const, Duplicated, Active)]
fn with_data(data: &[f64; 2], q: &[f64]) -> f64 {
    let a = q[0] - data[0];
    let b = q[1] - data[1];
    -0.5 * (a * a + b * b)
}

#[test]
fn constant_data_and_adapter_failures() {
    let data = [1.5, -2.0];
    let model = EnzymeModel::new(2, |q: &[f64], g: &mut [f64], seed| {
        d_with_data(&data, q, g, seed)
    });
    let mut g = [42.0; 2];
    assert_eq!(model.logp_grad(&[2.0, -1.0], &mut g).unwrap(), -0.625);
    assert_eq!(g, [-0.5, -1.0]);
    assert!(model.logp_grad(&[0.0], &mut g).is_err());
    assert!(model.logp_grad(&[f64::NAN, 0.0], &mut g).is_err());
    assert_eq!(g, [-0.5, -1.0]);
    let target = TransformedTarget::new(
        model,
        ParameterLayout::new([Transform::identity(2).unwrap()]).unwrap(),
    )
    .unwrap();
    assert!(
        check_gradient(&target, &[0.3, -0.7], GradientCheckOptions::default())
            .unwrap()
            .passed()
    );
}
