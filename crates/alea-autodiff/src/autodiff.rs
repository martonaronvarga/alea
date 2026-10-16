//! Adapter for `std::autodiff::autodiff_reverse` / Enzyme generated functions.
//!
//! The differentiated function takes `&[f64]` with activity `Duplicated` and
//! returns `f64` with activity `Active`. Its generated derivative has signature
//! `fn(&[f64], &mut [f64], f64) -> f64`. Mark constant data `Const` and bind it
//! with a closure. Generate AD at the model definition, not through validation,
//! allocation, or error-handling code. Use the pinned `.#autodiff` release shell.
#![forbid(unsafe_code)]

use alea_core::{
    capability::{HessianVector, PotentialHessianDerivative},
    model::ConstrainedModel,
    target::LogDensityGradient,
};
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

/// Adapter for Enzyme-generated first-, second-, and third-order operations.
///
/// Alea does not differentiate validation or allocation code. Generate the
/// callbacks at the scalar model boundary, then bind them here. `H` writes the
/// potential Hessian-vector product and `D` writes the row-major potential
/// Hessian plus its row-major coordinate-derivative blocks.
pub struct EnzymeManifoldModel<F, H, D> {
    dimension: usize,
    gradient: F,
    hvp: H,
    tensors: D,
}

impl<F, H, D> EnzymeManifoldModel<F, H, D> {
    /// Binds generated derivative callbacks without allocation or type erasure.
    pub fn new(dimension: usize, gradient: F, hvp: H, tensors: D) -> Self {
        Self {
            dimension,
            gradient,
            hvp,
            tensors,
        }
    }
}

impl<F, H, D> LogDensityGradient for EnzymeManifoldModel<F, H, D>
where
    F: Fn(&[f64], &mut [f64], f64) -> f64,
    H: Fn(&[f64], &[f64], &mut [f64]),
    D: Fn(&[f64], &mut [f64], &mut [f64]),
{
    type Error = AutodiffError;

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        if q.len() != self.dimension || gradient.len() != self.dimension {
            return Err(AutodiffError::Dimension);
        }
        if q.iter().any(|value| !value.is_finite()) {
            return Err(AutodiffError::NonFinite);
        }
        gradient.fill(0.0);
        let value = (self.gradient)(q, gradient, 1.0);
        if !value.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(AutodiffError::NonFinite);
        }
        Ok(value)
    }
}

impl<F, H, D> HessianVector for EnzymeManifoldModel<F, H, D>
where
    F: Fn(&[f64], &mut [f64], f64) -> f64,
    H: Fn(&[f64], &[f64], &mut [f64]),
    D: Fn(&[f64], &mut [f64], &mut [f64]),
{
    fn potential_hvp(
        &self,
        position: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        if position.len() != self.dimension
            || vector.len() != self.dimension
            || output.len() != self.dimension
        {
            return Err(AutodiffError::Dimension);
        }
        if position
            .iter()
            .chain(vector)
            .any(|value| !value.is_finite())
        {
            return Err(AutodiffError::NonFinite);
        }
        output.fill(0.0);
        (self.hvp)(position, vector, output);
        if output.iter().any(|x| !x.is_finite()) {
            return Err(AutodiffError::NonFinite);
        }
        Ok(())
    }
}

impl<F, H, D> PotentialHessianDerivative for EnzymeManifoldModel<F, H, D>
where
    F: Fn(&[f64], &mut [f64], f64) -> f64,
    H: Fn(&[f64], &[f64], &mut [f64]),
    D: Fn(&[f64], &mut [f64], &mut [f64]),
{
    fn potential_hessian_and_derivatives(
        &self,
        position: &[f64],
        hessian: &mut [f64],
        derivatives: &mut [f64],
    ) -> Result<(), Self::Error> {
        let Some(matrix_len) = self.dimension.checked_mul(self.dimension) else {
            return Err(AutodiffError::Dimension);
        };
        let Some(derivative_len) = matrix_len.checked_mul(self.dimension) else {
            return Err(AutodiffError::Dimension);
        };
        if matrix_len > isize::MAX as usize / size_of::<f64>()
            || derivative_len > isize::MAX as usize / size_of::<f64>()
        {
            return Err(AutodiffError::Dimension);
        }
        if position.len() != self.dimension
            || hessian.len() != matrix_len
            || derivatives.len() != derivative_len
        {
            return Err(AutodiffError::Dimension);
        }
        if position.iter().any(|value| !value.is_finite()) {
            return Err(AutodiffError::NonFinite);
        }
        hessian.fill(0.0);
        derivatives.fill(0.0);
        (self.tensors)(position, hessian, derivatives);
        if hessian
            .iter()
            .chain(derivatives.iter())
            .any(|x| !x.is_finite())
        {
            return Err(AutodiffError::NonFinite);
        }
        Ok(())
    }
}
