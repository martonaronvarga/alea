//! Explicit value/VJP boundary for opaque FFI or specialized numerical kernels.
//! This boundary is composed outside Enzyme IR; it does not register arbitrary
//! foreign calls with the compiler's internal derivative registry.
use super::ConstrainedModel;
use crate::dist::wiener::{Wiener4, Wiener4Params, WienerObservation};
use thiserror::Error;

/// A small opaque primitive supplies its primal and derivative together.
/// Implementations may call FFI but must preserve the fixed mathematical target.
pub trait OpaquePrimitive<const N: usize> {
    type Error: std::error::Error + 'static;
    fn value_gradient(&self, input: &[f64; N]) -> Result<(f64, [f64; N]), Self::Error>;
}

/// Fixed-arity custom derivative adapter. It never asks Enzyme to differentiate
/// through the opaque implementation. Compose with an Enzyme model via `SumModel`.
pub struct PrimitiveModel<P, const N: usize>(pub P);

#[derive(Debug, Error)]
pub enum PrimitiveError<E: std::error::Error + 'static> {
    #[error("opaque primitive dimension mismatch")]
    Dimension,
    #[error("opaque primitive failed: {0}")]
    Evaluation(#[source] E),
}
impl<P: OpaquePrimitive<N>, const N: usize> ConstrainedModel for PrimitiveModel<P, N> {
    type Error = PrimitiveError<P::Error>;
    fn dimension(&self) -> usize {
        N
    }
    fn logp_grad(&self, input: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let input = input.try_into().map_err(|_| PrimitiveError::Dimension)?;
        if gradient.len() != N {
            return Err(PrimitiveError::Dimension);
        }
        let (value, derivative) = self
            .0
            .value_gradient(input)
            .map_err(PrimitiveError::Evaluation)?;
        gradient.copy_from_slice(&derivative);
        Ok(value)
    }
}

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
