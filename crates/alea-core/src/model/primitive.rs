//! Explicit value/VJP boundary for opaque FFI or specialized numerical kernels.
//! This boundary is composed outside Enzyme IR; it does not register arbitrary
//! foreign calls with the compiler's internal derivative registry.
use super::ConstrainedModel;
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
