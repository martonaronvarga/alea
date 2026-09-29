use super::{Wiener4, Wiener4Params, WienerObservation};
use alea_core::model::OpaquePrimitive;
use thiserror::Error;

/// One Wiener observation with the existing analytic series derivatives.
/// Parameters are `[alpha, tau, beta, delta]`; transforms belong to the layout.
pub struct WienerPrimitive {
    observation: WienerObservation,
    tolerance: f64,
}
#[derive(Clone, Copy, Debug, Error)]
pub enum WienerPrimitiveError {
    #[error("invalid Wiener observation or series tolerance")]
    Configuration,
    #[error("Wiener parameters are outside support or the series failed")]
    Evaluation,
}
impl WienerPrimitive {
    pub fn new(
        observation: WienerObservation,
        tolerance: f64,
    ) -> Result<Self, WienerPrimitiveError> {
        if !observation.rt.is_finite()
            || observation.rt <= 0.0
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || tolerance >= 1.0
        {
            return Err(WienerPrimitiveError::Configuration);
        }
        Ok(Self {
            observation,
            tolerance,
        })
    }
}
impl OpaquePrimitive<4> for WienerPrimitive {
    type Error = WienerPrimitiveError;
    fn value_gradient(&self, input: &[f64; 4]) -> Result<(f64, [f64; 4]), Self::Error> {
        let params = Wiener4Params::from_array(*input);
        let evaluation = Wiener4.fused(&self.observation, &params, self.tolerance);
        let gradient = evaluation.grad.to_array();
        if !evaluation.log_prob.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(WienerPrimitiveError::Evaluation);
        }
        Ok((evaluation.log_prob, gradient))
    }
}
