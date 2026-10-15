//! Frozen, measure-correct target reparameterization.

use std::{cell::RefCell, error::Error};

use alea_math::{
    affine::{AffineMapError, LowRankAffineMap},
    buffer::OwnedBuffer,
};
use thiserror::Error;

use crate::target::{EvaluationError, LogDensityGradient, evaluate};

/// Failure while evaluating a frozen transported target.
#[derive(Debug, Error)]
pub enum FrozenTransportError<E: Error + 'static> {
    /// The target dimension changed or an evaluation buffer has the wrong shape.
    #[error("transported target dimension mismatch")]
    Dimension,
    /// Recursive use attempted to borrow the same chain-local workspace.
    #[error("transported target workspace is already in use")]
    WorkspaceBusy,
    /// The frozen affine map failed numerically.
    #[error(transparent)]
    Affine(#[from] AffineMapError),
    /// The underlying target failed or produced invalid output.
    #[error(transparent)]
    Evaluation(#[from] EvaluationError<E>),
    /// The transformed log density overflowed despite finite constituents.
    #[error("transported log density is not finite")]
    NonFiniteLogDensity,
}

struct Workspace {
    target_position: OwnedBuffer,
    target_score: OwnedBuffer,
    latent_score: OwnedBuffer,
}

/// A target evaluated through an immutable affine map.
///
/// For `x=A z+b`, this evaluates `log p(x)+log|det A|` and returns
/// `A^T grad_x log p(x)`. The map cannot train or mutate during retained draws.
/// One adapter is intended per chain because its reusable workspace is not `Sync`.
pub struct FrozenTransportTarget<'a, T: LogDensityGradient + ?Sized> {
    target: &'a T,
    map: LowRankAffineMap,
    workspace: RefCell<Workspace>,
}

impl<'a, T: LogDensityGradient + ?Sized> FrozenTransportTarget<'a, T> {
    /// Binds a frozen map to a target of the same dimension.
    ///
    /// # Errors
    /// Returns [`FrozenTransportError::Dimension`] before allocating workspace
    /// when the dimensions differ.
    pub fn new(
        target: &'a T,
        map: LowRankAffineMap,
    ) -> Result<Self, FrozenTransportError<T::Error>> {
        let dimension = map.dimension();
        if target.dimension() != dimension {
            return Err(FrozenTransportError::Dimension);
        }
        Ok(Self {
            target,
            map,
            workspace: RefCell::new(Workspace {
                target_position: OwnedBuffer::new(dimension),
                target_score: OwnedBuffer::new(dimension),
                latent_score: OwnedBuffer::new(dimension),
            }),
        })
    }

    /// Immutable map used by this target.
    pub fn map(&self) -> &LowRankAffineMap {
        &self.map
    }

    /// Underlying target in original coordinates.
    pub fn target(&self) -> &T {
        self.target
    }

    /// Maps latent coordinates to original target coordinates.
    pub fn to_target(&self, latent: &[f64], target: &mut [f64]) -> Result<(), AffineMapError> {
        self.map.forward(latent, target)
    }
}

impl<T: LogDensityGradient + ?Sized> LogDensityGradient for FrozenTransportTarget<'_, T> {
    type Error = FrozenTransportError<T::Error>;

    fn dimension(&self) -> usize {
        self.map.dimension()
    }

    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        if position.len() != self.dimension()
            || gradient.len() != self.dimension()
            || self.target.dimension() != self.dimension()
        {
            return Err(FrozenTransportError::Dimension);
        }
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| FrozenTransportError::WorkspaceBusy)?;
        let Workspace {
            target_position,
            target_score,
            latent_score,
        } = &mut *workspace;
        self.map.forward(position, target_position)?;
        let log_density = evaluate(self.target, target_position, target_score)?;
        self.map.pullback(target_score, latent_score)?;
        let transformed = log_density + self.map.log_abs_det();
        if !transformed.is_finite() {
            return Err(FrozenTransportError::NonFiniteLogDensity);
        }
        gradient.copy_from_slice(latent_score);
        Ok(transformed)
    }
}
