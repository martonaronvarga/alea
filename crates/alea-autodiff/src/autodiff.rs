//! Adapter for `std::autodiff::autodiff_reverse` / Enzyme generated functions.
//!
//! The differentiated function takes `&[f64]` with activity `Duplicated` and
//! returns `f64` with activity `Active`. Its generated derivative has signature
//! `fn(&[f64], &mut [f64], f64) -> f64`. Mark constant data `Const` and bind it
//! with a closure. Generate AD at the model definition, not through validation,
//! allocation, or error-handling code. Use the pinned `.#autodiff` release shell.
#![forbid(unsafe_code)]

use alea_core::model::ConstrainedModel;
use thiserror::Error;

pub struct EnzymeModel<F> {
    dimension: usize,
    derivative: F,
}
impl<F: Fn(&[f64], &mut [f64], f64) -> f64> EnzymeModel<F> {
    /// `derivative` must be the reverse derivative of the desired scalar model,
    /// accepting an output seed and accumulating into its input shadow.
    pub fn new(dimension: usize, derivative: F) -> Self {
        Self {
            dimension,
            derivative,
        }
    }
}
#[derive(Clone, Copy, Debug, Error)]
pub enum AutodiffError {
    #[error("autodiff input or shadow dimension mismatch")]
    Dimension,
    #[error("autodiff input, value, or derivative is not finite")]
    NonFinite,
}
impl<F: Fn(&[f64], &mut [f64], f64) -> f64> ConstrainedModel for EnzymeModel<F> {
    type Error = AutodiffError;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        if q.len() != self.dimension || g.len() != self.dimension {
            return Err(AutodiffError::Dimension);
        }
        if q.iter().any(|v| !v.is_finite()) {
            return Err(AutodiffError::NonFinite);
        }
        // Enzyme accumulates adjoints: zero on EVERY call, even after failure.
        g.fill(0.0);
        let value = (self.derivative)(q, g, 1.0);
        if !value.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err(AutodiffError::NonFinite);
        }
        Ok(value)
    }
}
