use alea_core::{
    model::ConstrainedModel,
    target::{DimensionError, LogDensityGradient},
};
use alea_distributions::{Gaussian, wiener::*};

#[test]
fn gaussian_is_dimension_bound_and_rejects_shapes_before_mutation() {
    let target = Gaussian::new(2);
    let mut gradient = [123.0; 2];
    assert_eq!(
        target.logp_grad(&[1.0], &mut gradient),
        Err(DimensionError {
            expected: 2,
            position: 1,
            gradient: 2
        })
    );
    assert_eq!(gradient, [123.0; 2]);
    assert_eq!(target.logp_grad(&[1.0, 2.0], &mut gradient).unwrap(), -2.5);
    assert_eq!(gradient, [-1.0, -2.0]);
}

#[test]
fn wiener_batches_implement_the_constrained_model_contract() {
    fn check<M: ConstrainedModel<Error = WienerModelError>>(
        model: M,
        q: &[f64],
    ) -> (f64, Vec<f64>) {
        let mut gradient = vec![123.0; q.len()];
        assert!(matches!(
            model.logp_grad(&q[..1], &mut gradient),
            Err(WienerModelError::Dimension)
        ));
        assert_eq!(gradient, vec![123.0; q.len()]);
        let value = model.logp_grad(q, &mut gradient).unwrap();
        assert!(value.is_finite() && gradient.iter().all(|x| x.is_finite()));
        let mut invalid = q.to_vec();
        invalid[0] = -1.0;
        assert!(matches!(
            model.logp_grad(&invalid, &mut vec![0.0; q.len()]),
            Err(WienerModelError::NonFinite)
        ));
        (value, gradient)
    }
    let data = vec![WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    }];
    let q4 = Wiener4Params::with_params(1.5, 0.2, 0.4, 0.3)
        .unwrap()
        .to_array();
    let q5 = Wiener5Params::with_params(1.5, 0.2, 0.4, 0.3, 0.1)
        .unwrap()
        .to_array();
    let q7 = Wiener7Params::with_params(1.5, 0.2, 0.4, 0.3, 0.1, 0.05, 0.1)
        .unwrap()
        .to_array();
    for (a, b) in [
        (
            check(WienerModel::new(Wiener4, data.clone()), &q4),
            check(
                WienerModel::new(Wiener4, WienerObservations::from(data.clone())),
                &q4,
            ),
        ),
        (
            check(WienerModel::new(Wiener5, data.clone()), &q5),
            check(
                WienerModel::new(Wiener5, WienerObservations::from(data.clone())),
                &q5,
            ),
        ),
        (
            check(WienerModel::new(Wiener7, data.clone()), &q7),
            check(
                WienerModel::new(Wiener7, WienerObservations::from(data)),
                &q7,
            ),
        ),
    ] {
        assert!((a.0 - b.0).abs() < 1e-5);
        for (a, b) in a.1.iter().zip(b.1) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
