//! Normalized probability laws, with separate event, batch, derivative and RNG contracts.
//!
//! A legitimate support zero is `Ok(NEG_INFINITY)`. Invalid parameters, NaN
//! observations and numerical/backend failures are errors. Unnormalized factors
//! implement [`crate::model::ConstrainedModel`], never this trait by implication.

use rand::Rng;

/// The shape of one observation; a batch is a slice of these events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventShape {
    Scalar,
    Vector(usize),
    Matrix { rows: usize, columns: usize },
}

/// Reference measure and support of a normalized law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Support {
    Real,
    Positive,
    UnitInterval,
    /// Unit sphere with respect to surface area.
    UnitSphere,
    /// Unit ball with respect to volume.
    UnitBall,
    Count,
    /// Signed integers with counting measure.
    Integer,
    Finite {
        outcomes: usize,
    },
    RealVector,
    Simplex,
    CorrelationCholesky,
    /// Correlation matrices with respect to their free off-diagonal entries.
    CorrelationMatrix,
    /// A finite interval in the real line (bounds belong to the law).
    Interval,
    /// A half-line with a law-specific finite lower endpoint.
    LowerBounded,
    /// Angles in radians on one period, with respect to arc length.
    Circle,
    /// Symmetric positive-definite matrices, with respect to their free entries.
    PositiveDefinite,
}

/// Checked construction/evaluation/sampling failure, distinct from support zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DistributionError {
    #[error("invalid distribution parameter: {0}")]
    Parameter(&'static str),
    #[error("invalid observation")]
    Observation,
    #[error("distribution event, derivative or batch shape mismatch")]
    Shape,
    #[error("distribution calculation is outside representable precision")]
    Numerical,
    #[error("sampling exhausted its bounded rejection budget")]
    SamplingBudget,
}

/// A normalized density/PMF with a fixed event representation and reference measure.
pub trait Distribution {
    type Event: ?Sized;
    fn event_shape(&self) -> EventShape;
    fn support(&self) -> Support;
    /// Evaluates a normalized density or mass.
    ///
    /// # Errors
    /// Returns invalid observation, shape or numerical errors. Support zeros are
    /// successful negative infinity, including valid impossible discrete events.
    fn log_prob(&self, event: &Self::Event) -> Result<f64, DistributionError>;
}

/// Optional derivatives with respect to continuous distribution parameters.
/// Parameter order is documented by each implementation. Boundary derivatives
/// may be undefined even when the probability itself is defined.
pub trait ParameterScore: Distribution {
    fn parameter_count(&self) -> usize;
    /// Writes the complete parameter score on success.
    ///
    /// # Errors
    /// Returns shape, boundary or numerical errors; output is scratch on failure.
    fn parameter_score(
        &self,
        event: &Self::Event,
        output: &mut [f64],
    ) -> Result<(), DistributionError>;
}

/// Optional score of a scalar continuous event. Discrete events have no such score.
pub trait ContinuousDistribution: Distribution<Event = f64> {
    /// # Errors
    /// Returns an error where the event derivative is undefined or non-finite.
    fn event_score(&self, event: f64) -> Result<f64, DistributionError>;
}

/// Optional generative capability. Sampling is separate from target evaluation.
pub trait Sample: Distribution {
    type Draw;
    /// # Errors
    /// Returns a numerical or bounded rejection-budget error.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Self::Draw, DistributionError>;
}

/// Evaluates independent scalar events; this is not a multivariate event.
///
/// # Errors
/// Rejects a shape mismatch before writing. On an evaluation error, output is
/// scratch and may contain a completed prefix.
pub fn log_prob_batch<D: Distribution<Event = f64>>(
    distribution: &D,
    observations: &[f64],
    output: &mut [f64],
) -> Result<(), DistributionError> {
    if observations.len() != output.len() {
        return Err(DistributionError::Shape);
    }
    for (event, value) in observations.iter().zip(output) {
        let logp = distribution.log_prob(event)?;
        if logp.is_nan() {
            return Err(DistributionError::Numerical);
        }
        *value = logp;
    }
    Ok(())
}
