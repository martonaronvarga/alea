//! Derivative-free target capability and transactional density-only cache.
//! This is independent of the fused derivative capability: random-walk methods
//! must not require derivatives or pay for computing them.
#![deny(missing_docs)]
use crate::target::EvaluationError;
use alea_math::buffer::OwnedBuffer;

/// Semantically pure, fixed-dimensional log density in unconstrained coordinates.
/// The mathematical target and its dimension must remain fixed while borrowed.
pub trait LogDensity {
    /// Original model or backend error retained by checked evaluation.
    type Error: std::error::Error + 'static;
    /// Number of coordinates in each retained state.
    fn dimension(&self) -> usize;
    /// Returns the log density. Domain/backend failures retain their concrete type.
    fn logp(&self, position: &[f64]) -> Result<f64, Self::Error>;
}

/// Checks shape and finiteness without allocation or gradient computation.
pub fn evaluate<T: LogDensity + ?Sized>(
    target: &T,
    q: &[f64],
) -> Result<f64, EvaluationError<T::Error>> {
    if q.len() != target.dimension() {
        return Err(EvaluationError::DensityDimension {
            expected: target.dimension(),
            actual: q.len(),
        });
    }
    if let Some(index) = q.iter().position(|x| !x.is_finite()) {
        return Err(EvaluationError::NonFinitePosition { index });
    }
    let value = target.logp(q).map_err(EvaluationError::Model)?;
    if !value.is_finite() {
        return Err(EvaluationError::NonFiniteLogDensity);
    }
    Ok(value)
}

/// Coherent read-only position and density, with no fabricated gradient cache.
#[derive(Debug)]
#[must_use = "a cached density point has no effect unless it is retained"]
pub struct DensityPoint<'a, T: LogDensity + ?Sized> {
    target: &'a T,
    position: OwnedBuffer,
    log_density: f64,
}
impl<'a, T: LogDensity + ?Sized> DensityPoint<'a, T> {
    /// Evaluates and binds an initial position to `target`.
    ///
    /// # Errors
    /// Returns [`EvaluationError`] for a shape, input, model, or output failure.
    pub fn new(target: &'a T, position: OwnedBuffer) -> Result<Self, EvaluationError<T::Error>> {
        let log_density = evaluate(target, &position)?;
        Ok(Self {
            target,
            position,
            log_density,
        })
    }
    #[must_use]
    /// Target borrowed by this cached point.
    pub fn target(&self) -> &'a T {
        self.target
    }
    #[must_use]
    /// Coherent retained position.
    pub fn position(&self) -> &[f64] {
        &self.position
    }
    #[must_use]
    /// Number of coordinates in each retained state.
    pub fn dimension(&self) -> usize {
        self.position.len()
    }
    #[must_use]
    /// Log density cached at the retained position.
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
    /// Evaluates before publishing either field. Errors and panics preserve this point.
    ///
    /// # Errors
    /// Returns [`EvaluationError`] if the target dimension changed or if the new
    /// position fails checked evaluation.
    pub fn try_update(&mut self, position: &[f64]) -> Result<(), EvaluationError<T::Error>> {
        if self.target.dimension() != self.dimension() {
            return Err(EvaluationError::TargetDimensionChanged {
                expected: self.dimension(),
                actual: self.target.dimension(),
            });
        }
        let log_density = evaluate(self.target, position)?;
        self.position.copy_from_slice(position);
        self.log_density = log_density;
        Ok(())
    }
}
impl<T: LogDensity + ?Sized> Clone for DensityPoint<'_, T> {
    fn clone(&self) -> Self {
        Self {
            target: self.target,
            position: self.position.clone(),
            log_density: self.log_density,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        if self.position.len() == source.position.len() {
            self.position.copy_from_slice(&source.position);
        } else {
            self.position = source.position.clone();
        }
        self.target = source.target;
        self.log_density = source.log_density;
    }
}
