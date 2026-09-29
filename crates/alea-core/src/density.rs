//! Derivative-free target capability and transactional density-only cache.
//! This is independent of the fused derivative capability: random-walk methods
//! must not require derivatives or pay for computing them.
use crate::target::EvaluationError;
use alea_math::buffer::OwnedBuffer;

/// Semantically pure, fixed-dimensional log density in unconstrained coordinates.
/// The mathematical target and its dimension must remain fixed while borrowed.
pub trait LogDensity {
    type Error: std::error::Error + 'static;
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
pub struct DensityPoint<'a, T: LogDensity + ?Sized> {
    target: &'a T,
    position: OwnedBuffer,
    log_density: f64,
}
impl<'a, T: LogDensity + ?Sized> DensityPoint<'a, T> {
    pub fn new(target: &'a T, position: OwnedBuffer) -> Result<Self, EvaluationError<T::Error>> {
        let log_density = evaluate(target, &position)?;
        Ok(Self {
            target,
            position,
            log_density,
        })
    }
    pub fn target(&self) -> &'a T {
        self.target
    }
    pub fn position(&self) -> &[f64] {
        &self.position
    }
    pub fn dimension(&self) -> usize {
        self.position.len()
    }
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
    /// Evaluates before publishing either field. Errors and panics preserve this point.
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
