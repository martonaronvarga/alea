use alea_core::distribution::{
    ContinuousDistribution, Distribution, DistributionError, EventShape, ParameterScore, Support,
};
use alea_downstream_conformance::assert_distribution_conformance;

struct ExternalNormal {
    location: f64,
}
impl Distribution for ExternalNormal {
    type Event = f64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn log_prob(&self, event: &f64) -> Result<f64, DistributionError> {
        if event.is_nan() {
            return Err(DistributionError::Observation);
        }
        Ok(-0.5 * (event - self.location).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
    }
}
impl ContinuousDistribution for ExternalNormal {
    fn event_score(&self, event: f64) -> Result<f64, DistributionError> {
        if !event.is_finite() {
            return Err(DistributionError::Observation);
        }
        Ok(self.location - event)
    }
}
impl ParameterScore for ExternalNormal {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, event: &f64, output: &mut [f64]) -> Result<(), DistributionError> {
        if output.len() != 1 {
            return Err(DistributionError::Shape);
        }
        if !event.is_finite() {
            return Err(DistributionError::Observation);
        }
        output[0] = event - self.location;
        Ok(())
    }
}
#[test]
fn external_normalized_law_and_optional_scores_compose() {
    let law = ExternalNormal { location: 0.2 };
    let norm = -0.5 * (2.0 * std::f64::consts::PI).ln();
    assert_distribution_conformance(
        &law,
        &[
            (0.2, norm),
            (1.2, norm - 0.5),
            (f64::INFINITY, f64::NEG_INFINITY),
        ],
        1e-14,
    );
    assert_eq!(law.event_score(1.2).unwrap(), -1.0);
    let mut score = [0.0];
    law.parameter_score(&1.2, &mut score).unwrap();
    assert_eq!(score, [1.0]);
    assert!(law.log_prob(&f64::NAN).is_err());
    assert!(law.parameter_score(&1.2, &mut []).is_err());
}
