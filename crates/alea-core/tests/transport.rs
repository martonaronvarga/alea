use alea_core::{
    target::{LogDensityGradient, evaluate},
    transport::FrozenTransportTarget,
};
use alea_math::{affine::LowRankAffineMap, buffer::OwnedBuffer};

#[derive(Debug, thiserror::Error)]
#[error("infallible")]
struct Never;

struct StandardNormalTarget;
impl alea_core::target::TargetSpace for StandardNormalTarget {
    type Error = Never;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for StandardNormalTarget {
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Never> {
        gradient[0] = -q[0] / 4.0;
        gradient[1] = -q[1];
        Ok(-0.5 * (q[0] * q[0] / 4.0 + q[1] * q[1]))
    }
}

fn map() -> LowRankAffineMap {
    LowRankAffineMap::new(
        OwnedBuffer::from_fn(2, |i| [1.0, -2.0][i]),
        OwnedBuffer::from_fn(2, |i| [2.0, 0.5][i]),
        OwnedBuffer::new(0),
        OwnedBuffer::new(0),
    )
    .unwrap()
}

#[test]
fn transported_value_gradient_and_measure_match_finite_difference() {
    let transformed = FrozenTransportTarget::new(&StandardNormalTarget, map()).unwrap();
    let point = [0.3, -0.7];
    let mut gradient = [0.0; 2];
    let value = evaluate(&transformed, &point, &mut gradient).unwrap();
    let expected_x = [1.6, -2.35];
    let expected_value =
        -0.5 * (expected_x[0] * expected_x[0] / 4.0 + expected_x[1] * expected_x[1]);
    assert!((value - expected_value).abs() < 1e-14); // determinant is one
    let h = 1e-6;
    for coordinate in 0..2 {
        let mut plus = point;
        let mut minus = point;
        plus[coordinate] += h;
        minus[coordinate] -= h;
        let numerical = (evaluate(&transformed, &plus, &mut [0.0; 2]).unwrap()
            - evaluate(&transformed, &minus, &mut [0.0; 2]).unwrap())
            / (2.0 * h);
        assert!((gradient[coordinate] - numerical).abs() < 1e-9);
    }
}

#[test]
fn frozen_evaluation_reuses_storage() {
    let transformed = FrozenTransportTarget::new(&StandardNormalTarget, map()).unwrap();
    let point = [0.3, -0.7];
    let mut gradient = [0.0; 2];
    let measured = allocation_counter::measure(|| {
        for _ in 0..32 {
            evaluate(&transformed, &point, &mut gradient).unwrap();
        }
    });
    assert_eq!(measured.count_total, 0);
}

struct PositiveSupport;
impl alea_core::target::TargetSpace for PositiveSupport {
    type Error = Never;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for PositiveSupport {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Never> {
        if q[0] <= 0. {
            return Ok(f64::NEG_INFINITY);
        }
        g[0] = -q[0];
        g[1] = -q[1];
        Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
    }
}
#[test]
fn affine_support_zero_remains_raw_zero_and_never_a_valid_cached_point() {
    let target = FrozenTransportTarget::new(&PositiveSupport, map()).unwrap();
    let mut gradient = [7.; 2];
    assert_eq!(
        target.logp_grad(&[-1., 0.], &mut gradient).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(gradient.iter().all(|x| x.is_nan()));
    assert!(matches!(
        evaluate(&target, &[-1., 0.], &mut gradient),
        Err(alea_core::target::EvaluationError::NonFiniteLogDensity)
    ));
    let mut point = alea_core::target::PointState::new(&target, OwnedBuffer::new(2)).unwrap();
    let before = point.clone();
    let mut workspace = alea_core::target::EvaluationWorkspace::new(2);
    assert!(point.try_update(&[-1., 0.], &mut workspace).is_err());
    assert_eq!(point.position(), before.position());
    assert_eq!(point.gradient(), before.gradient());
    assert_eq!(point.log_density(), before.log_density());
}

struct IncompleteScore;
impl alea_core::target::TargetSpace for IncompleteScore {
    type Error = Never;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for IncompleteScore {
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Never> {
        g[0] = 0.;
        Ok(0.)
    }
}
#[test]
fn support_zero_recovery_does_not_hide_invalid_finite_point_scores() {
    let target = FrozenTransportTarget::new(&IncompleteScore, map()).unwrap();
    let mut gradient = [7.; 2];
    assert!(matches!(
        evaluate(&target, &[0., 0.], &mut gradient),
        Err(alea_core::target::EvaluationError::NonFiniteGradient { .. })
    ));
    assert!(gradient.iter().all(|x| x.is_nan()));
}

struct BrokenTarget;
impl alea_core::target::TargetSpace for BrokenTarget {
    type Error = Never;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for BrokenTarget {
    fn logp_grad(&self, _: &[f64], _: &mut [f64]) -> Result<f64, Never> {
        Err(Never)
    }
}
#[test]
fn transported_backend_errors_retain_the_concrete_source_type() {
    let target = FrozenTransportTarget::new(&BrokenTarget, map()).unwrap();
    let mut gradient = [7.; 2];
    assert!(matches!(
        target.logp_grad(&[0., 0.], &mut gradient),
        Err(alea_core::transport::FrozenTransportError::Evaluation(
            alea_core::target::EvaluationError::Model(Never)
        ))
    ));
    assert_eq!(gradient, [7.; 2]);
}

struct NumericalOutput(f64);
impl alea_core::target::TargetSpace for NumericalOutput {
    type Error = Never;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for NumericalOutput {
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Never> {
        g.fill(0.);
        Ok(self.0)
    }
}
#[test]
fn affine_numerical_outputs_and_map_overflow_remain_outer_numerical_errors() {
    for value in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        let raw = NumericalOutput(value);
        let target = FrozenTransportTarget::new(&raw, map()).unwrap();
        assert!(matches!(
            evaluate(&target, &[0.; 2], &mut [7.; 2]),
            Err(alea_core::target::EvaluationError::NonFiniteLogDensity)
        ));
    }
    let huge = LowRankAffineMap::new(
        OwnedBuffer::new(2),
        OwnedBuffer::from_fn(2, |i| if i == 0 { 1e300 } else { 1. }),
        OwnedBuffer::new(0),
        OwnedBuffer::new(0),
    )
    .unwrap();
    let target = FrozenTransportTarget::new(&StandardNormalTarget, huge).unwrap();
    assert!(matches!(
        evaluate(&target, &[1e300, 0.], &mut [7.; 2]),
        Err(alea_core::target::EvaluationError::NonFiniteLogDensity)
    ));
}
