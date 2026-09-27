//! Fallible fused evaluation and target-bound, coherent cached points.
//!
//! This protocol is additive: the legacy density/state traits remain available
//! while samplers migrate. No unsafe code or type-erased model dispatch is needed.
#![forbid(unsafe_code)]

use std::error::Error;

use thiserror::Error;

use crate::{buffer::OwnedBuffer, density::FusedLogDensity};

/// A semantically pure fused log-density/gradient in unconstrained coordinates.
///
/// The dimension and mathematical target must remain fixed while borrowed by a
/// [`PointState`]. Interior mutability may support counters or scratch, but must
/// not change the density/gradient at a position. Implementations must overwrite
/// every gradient entry on success. On failure, gradient contents are unspecified.
/// Implementations should reuse chain-local scratch rather than allocate per call.
///
/// [`evaluate`] checks dimensions, finite inputs, and complete finite outputs.
/// The raw method is available to adapters; callers must honor its dimensions.
pub trait LogDensityGradient {
    /// The original model/backend error, retained rather than formatted or boxed.
    type Error: Error + 'static;

    /// Number of unconstrained coordinates (constant for this target).
    fn dimension(&self) -> usize;

    /// Writes the full gradient and returns the log density in one evaluation.
    ///
    /// # Errors
    /// Returns the implementation's model-domain or backend failure. Use
    /// [`evaluate`] for uniform shape and non-finite validation around this call.
    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error>;
}

/// Both evaluation buffers must have exactly the target dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error(
    "target dimension {expected} requires matching buffers, got position {position} and gradient {gradient}"
)]
pub struct DimensionError {
    /// Target dimension.
    pub expected: usize,
    /// Actual position length.
    pub position: usize,
    /// Actual gradient (or workspace) length.
    pub gradient: usize,
}

impl DimensionError {
    #[inline]
    fn check(expected: usize, position: usize, gradient: usize) -> Result<(), Self> {
        if position != expected || gradient != expected {
            return Err(Self {
                expected,
                position,
                gradient,
            });
        }
        Ok(())
    }
}

/// Checked evaluation failures. Model errors retain their concrete source type.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EvaluationError<E: Error + 'static> {
    /// Shape failure detected before model invocation or output mutation.
    #[error(transparent)]
    Dimension(#[from] DimensionError),
    /// A cached point's target violated the fixed-dimension contract.
    #[error("target dimension changed from {expected} to {actual}")]
    TargetDimensionChanged { expected: usize, actual: usize },
    /// Non-finite input detected before invoking the model.
    #[error("position entry {index} is not finite")]
    NonFinitePosition { index: usize },
    /// The backend/model's original error, without allocation for wrapping it.
    #[error("target evaluation failed: {0}")]
    Model(#[source] E),
    /// Includes negative infinity: support failures are not valid cached points.
    #[error("target returned a non-finite log density")]
    NonFiniteLogDensity,
    /// Includes coordinates that the model did not overwrite.
    #[error("gradient entry {index} is missing or not finite")]
    NonFiniteGradient { index: usize },
}

/// Performs one checked fused evaluation into caller-owned gradient storage.
///
/// The gradient is filled with NaNs before calling the target so partially
/// written successful outputs cannot reuse a previous gradient. This costs one
/// linear write pass; output validation costs one linear read pass. Neither the
/// wrapper nor error wrapping allocates. The target itself may allocate.
///
/// # Errors
/// Returns typed dimension, non-finite input/output, or original model errors.
/// Shape/input failures leave the gradient untouched. After model invocation,
/// the gradient may be partially written on error or unwinding. Use [`PointState`]
/// for a transactional cached point. Target panics propagate; they are not errors.
#[inline]
pub fn evaluate<T: LogDensityGradient + ?Sized>(
    target: &T,
    position: &[f64],
    gradient: &mut [f64],
) -> Result<f64, EvaluationError<T::Error>> {
    DimensionError::check(target.dimension(), position.len(), gradient.len())?;
    if let Some(index) = position.iter().position(|x| !x.is_finite()) {
        return Err(EvaluationError::NonFinitePosition { index });
    }
    gradient.fill(f64::NAN);
    let log_density = target
        .logp_grad(position, gradient)
        .map_err(EvaluationError::Model)?;
    if !log_density.is_finite() {
        return Err(EvaluationError::NonFiniteLogDensity);
    }
    if let Some(index) = gradient.iter().position(|x| !x.is_finite()) {
        return Err(EvaluationError::NonFiniteGradient { index });
    }
    Ok(log_density)
}

/// Explicit dimension adapter for the old infallible [`FusedLogDensity`] trait.
///
/// The supplied dimension must match the legacy model's meaning; this adapter
/// cannot infer it or recover model errors already discarded by the old API.
/// It never calls separate density/gradient methods or catches target panics.
#[derive(Debug)]
pub struct FusedAdapter<'a, D: ?Sized> {
    density: &'a D,
    dimension: usize,
}

impl<'a, D: ?Sized> FusedAdapter<'a, D> {
    /// Borrows a legacy model with an explicitly declared fixed dimension.
    pub fn new(density: &'a D, dimension: usize) -> Self {
        Self { density, dimension }
    }
}

impl<D> LogDensityGradient for FusedAdapter<'_, D>
where
    D: FusedLogDensity<Point = [f64], Gradient = [f64]> + ?Sized,
{
    type Error = DimensionError;

    #[inline]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        DimensionError::check(self.dimension, position.len(), gradient.len())?;
        Ok(self.density.log_prob_and_grad(position, gradient))
    }
}

/// Chain-local gradient scratch reused for transactional point updates.
///
/// Failed evaluations may dirty this workspace but never the cached point.
/// Scratch has no public cache or mutable-gradient interface.
#[derive(Debug)]
pub struct EvaluationWorkspace {
    gradient: OwnedBuffer,
}

impl EvaluationWorkspace {
    /// Allocates one initialized, 64-byte-aligned gradient buffer.
    pub fn new(dimension: usize) -> Self {
        Self {
            gradient: OwnedBuffer::new(dimension),
        }
    }

    /// Fixed number of gradient entries available to an evaluation.
    pub fn dimension(&self) -> usize {
        self.gradient.len()
    }
}

/// Finite position, log density, and gradient bound to the target that produced them.
///
/// The target is borrowed for the point's lifetime. Only read-only cache access is
/// exposed; updates evaluate into separate scratch before committing all fields.
/// No implementation of the legacy mutable state traits is provided. The purity
/// contract of [`LogDensityGradient`] still applies to interior-mutable targets.
/// "Atomic" here means a coherent transaction, not lock-free synchronization.
///
/// # Examples
/// ```
/// use kernels::{buffer::OwnedBuffer, dist::Gaussian, target::{
///     EvaluationWorkspace, FusedAdapter, PointState,
/// }};
/// let target = FusedAdapter::new(&Gaussian, 2);
/// let mut point = PointState::new(&target, OwnedBuffer::from_fn(2, |_| 0.0))?;
/// let mut scratch = EvaluationWorkspace::new(2);
/// point.try_update(&[1.0, 2.0], &mut scratch)?;
/// assert_eq!(point.log_density(), -2.5);
/// assert_eq!(point.gradient(), &[-1.0, -2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Cached coordinates cannot be independently mutated:
/// ```compile_fail
/// # use kernels::{buffer::OwnedBuffer, dist::Gaussian, target::{FusedAdapter, PointState}};
/// let target = FusedAdapter::new(&Gaussian, 1);
/// let mut point = PointState::new(&target, OwnedBuffer::new(1)).unwrap();
/// point.position()[0] = 7.0;
/// ```
#[derive(Debug)]
pub struct PointState<'a, T: LogDensityGradient + ?Sized> {
    target: &'a T,
    position: OwnedBuffer,
    gradient: OwnedBuffer,
    log_density: f64,
}

impl<T: LogDensityGradient + ?Sized> Clone for PointState<'_, T> {
    /// Copies the complete cache and retains the same target borrow.
    fn clone(&self) -> Self {
        Self {
            target: self.target,
            position: self.position.clone(),
            gradient: self.gradient.clone(),
            log_density: self.log_density,
        }
    }

    /// Reuses both allocations when dimensions agree; copies one coherent point.
    fn clone_from(&mut self, source: &Self) {
        if self.dimension() != source.dimension() {
            *self = source.clone();
            return;
        }
        self.position.copy_from_slice(&source.position);
        self.gradient.copy_from_slice(&source.gradient);
        self.log_density = source.log_density;
        self.target = source.target;
    }
}

impl<'a, T: LogDensityGradient + ?Sized> PointState<'a, T> {
    /// Takes ownership of aligned coordinates and evaluates the initial cache once.
    ///
    /// # Errors
    /// Returns [`EvaluationError`] for invalid input, model failure, or invalid
    /// output. No point is constructed unless the entire cache is finite.
    pub fn new(target: &'a T, position: OwnedBuffer) -> Result<Self, EvaluationError<T::Error>> {
        DimensionError::check(target.dimension(), position.len(), position.len())?;
        let mut gradient = OwnedBuffer::new(position.len());
        let log_density = evaluate(target, &position, &mut gradient)?;
        Ok(Self {
            target,
            position,
            gradient,
            log_density,
        })
    }

    /// Replaces the point using one fused evaluation and reusable gradient scratch.
    ///
    /// Success copies coordinates and swaps the gradient allocation. No allocation
    /// or resize is performed by this method. All cached fields remain unchanged
    /// on an error or a target panic (when unwinding). The workspace can be reused
    /// after failure. Updates retain the point's target, not a per-update argument;
    /// cloning/replacing a whole point copies its target and complete cache together.
    ///
    /// # Errors
    /// Returns [`EvaluationError`] on dimension, input, model, or output failure.
    /// A target that changes dimension is rejected before evaluation.
    pub fn try_update(
        &mut self,
        position: &[f64],
        workspace: &mut EvaluationWorkspace,
    ) -> Result<(), EvaluationError<T::Error>> {
        let actual = self.target.dimension();
        if actual != self.dimension() {
            return Err(EvaluationError::TargetDimensionChanged {
                expected: self.dimension(),
                actual,
            });
        }
        let log_density = evaluate(self.target, position, &mut workspace.gradient)?;
        // All lengths and values are validated. No user code or allocation occurs
        // during commit, so an evaluation cannot expose half-updated cache fields.
        self.position.copy_from_slice(position);
        std::mem::swap(&mut self.gradient, &mut workspace.gradient);
        self.log_density = log_density;
        Ok(())
    }

    /// Fixed point dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.position.len()
    }

    /// The shared target borrow associated with this complete cache.
    #[inline]
    pub fn target(&self) -> &'a T {
        self.target
    }

    /// Coordinates corresponding to the cached density and gradient.
    #[inline]
    pub fn position(&self) -> &[f64] {
        &self.position
    }

    /// Gradient at exactly [`Self::position`].
    #[inline]
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    /// Finite log density at exactly [`Self::position`].
    #[inline]
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
}
