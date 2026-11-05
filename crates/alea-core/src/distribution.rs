//! Normalized probability laws, with separate event, batch, derivative and RNG contracts.
//!
//! A legitimate support zero is `Ok(NEG_INFINITY)`. Invalid parameters, NaN
//! observations and numerical/backend failures are errors. Unnormalized factors
//! implement [`crate::model::ConstrainedModel`], never this trait by implication.

use rand::Rng;
use std::borrow::Borrow;

/// The shape of one observation; a batch is a slice of these events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventShape {
    Scalar,
    /// An event with both continuous and discrete scalar coordinates.
    Mixed {
        continuous: usize,
        discrete: usize,
    },
    Vector(usize),
    Matrix {
        rows: usize,
        columns: usize,
    },
}

/// Geometric domain of an event. This does not identify its reference measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Support {
    /// A union of component domains sharing a reference measure.
    Union,
    Real,
    Positive,
    UnitInterval,
    /// Unit sphere.
    UnitSphere,
    /// Unit ball.
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
    /// Correlation matrices.
    CorrelationMatrix,
    /// A finite interval in the real line (bounds belong to the law).
    Interval,
    /// A half-line with a law-specific finite lower endpoint.
    LowerBounded,
    /// Support endpoints depend on the law's parameters.
    ParameterDependent,
    /// Angles in radians on one period.
    Circle,
    /// Symmetric positive-definite matrices.
    PositiveDefinite,
}

/// Measure relative to which a probability density or mass is evaluated.
/// Dimensions describe independent coordinates, not the number of stored entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceMeasure {
    /// Lebesgue measure in ordinary Cartesian coordinates.
    Lebesgue { dimension: usize },
    /// Counting measure on the discrete event space.
    Counting,
    /// Product of Lebesgue measure and counting measure.
    LebesgueCounting { dimension: usize },
    /// Lebesgue measure on the first K−1 simplex coordinates.
    SimplexCoordinates { components: usize },
    /// Surface area on a unit sphere in the stated ambient dimension.
    SphereSurface { ambient_dimension: usize },
    /// Lebesgue measure on the free strict-lower entries of a correlation factor.
    CorrelationCholeskyCoordinates { order: usize },
    /// Lebesgue measure on free off-diagonal correlation entries.
    CorrelationCoordinates { order: usize },
    /// Lebesgue measure on independent entries of a symmetric matrix.
    SymmetricMatrixCoordinates { order: usize },
}
impl ReferenceMeasure {
    /// Number of continuous coordinates; counting measure has no such dimension.
    /// Returns `None` if invalid/overflowing metadata cannot describe that space.
    pub fn dimension(self) -> Option<usize> {
        match self {
            Self::Lebesgue { dimension } | Self::LebesgueCounting { dimension } => Some(dimension),
            Self::Counting => None,
            Self::SimplexCoordinates { components } => components.checked_sub(1),
            Self::SphereSurface { ambient_dimension } => ambient_dimension.checked_sub(1),
            Self::CorrelationCholeskyCoordinates { order }
            | Self::CorrelationCoordinates { order } => {
                order.checked_mul(order.checked_sub(1)?)?.checked_div(2)
            }
            Self::SymmetricMatrixCoordinates { order } => {
                order.checked_mul(order.checked_add(1)?)?.checked_div(2)
            }
        }
    }
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
    type Error: std::error::Error + Send + Sync + From<DistributionError> + 'static;
    type Event: ?Sized;
    fn event_shape(&self) -> EventShape;
    fn support(&self) -> Support;
    fn reference_measure(&self) -> ReferenceMeasure;
    /// Evaluates a normalized density or mass.
    ///
    /// # Errors
    /// Returns invalid observation, shape or numerical errors. Support zeros are
    /// successful negative infinity, including valid impossible discrete events.
    fn log_prob(&self, event: &Self::Event) -> Result<f64, Self::Error>;
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
    fn parameter_score(&self, event: &Self::Event, output: &mut [f64]) -> Result<(), Self::Error>;

    /// Evaluates a scalar density and both kinds of score together.
    /// Parameter scores use the same order as [`Self::parameter_score`]. The
    /// returned pair is `(log_probability, event_score)`; output is scratch on
    /// failure. Implementations may share work but must preserve the separate
    /// density/event/parameter evaluation's error and numerical contracts.
    ///
    /// # Errors
    /// Rejects the output shape first, then preserves density, event-score and
    /// parameter-score failures, in that order.
    fn log_prob_event_parameter_score(
        &self,
        event: f64,
        output: &mut [f64],
    ) -> Result<(f64, f64), Self::Error>
    where
        Self: ScalarEventScore + Sized,
    {
        if output.len() != self.parameter_count() {
            return Err(DistributionError::Shape.into());
        }
        let value = self.log_prob(&event)?;
        let event_score = self.event_score(event)?;
        self.parameter_score(&event, output)?;
        Ok((value, event_score))
    }
}

/// Optional score of a scalar continuous event. Discrete events have no such score.
pub trait ScalarEventScore: Distribution<Event = f64> {
    /// # Errors
    /// Returns an error where the event derivative is undefined or non-finite.
    fn event_score(&self, event: f64) -> Result<f64, Self::Error>;
}

/// Optional generative capability. Sampling is separate from target evaluation.
pub trait Sample: Distribution {
    type Draw: Borrow<Self::Event>;
    /// # Errors
    /// Returns a numerical or bounded rejection-budget error.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Self::Draw, Self::Error>;
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
) -> Result<(), D::Error> {
    if observations.len() != output.len() {
        return Err(DistributionError::Shape.into());
    }
    for (event, value) in observations.iter().zip(output) {
        let logp = distribution.log_prob(event)?;
        if logp.is_nan() {
            return Err(DistributionError::Numerical.into());
        }
        *value = logp;
    }
    Ok(())
}

#[cfg(test)]
mod combined_score_tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    struct ExternalLaw {
        fail: Cell<u8>,
        visits: RefCell<Vec<u8>>,
    }
    impl ExternalLaw {
        fn visit(&self, stage: u8) -> Result<(), DistributionError> {
            self.visits.borrow_mut().push(stage);
            if self.fail.get() == stage {
                Err(DistributionError::Numerical)
            } else {
                Ok(())
            }
        }
    }
    impl Distribution for ExternalLaw {
        type Event = f64;
        type Error = DistributionError;
        fn event_shape(&self) -> EventShape {
            EventShape::Scalar
        }
        fn support(&self) -> Support {
            Support::Real
        }
        fn reference_measure(&self) -> ReferenceMeasure {
            ReferenceMeasure::Lebesgue { dimension: 1 }
        }
        fn log_prob(&self, _: &f64) -> Result<f64, Self::Error> {
            self.visit(1)?;
            Ok(-2.)
        }
    }
    impl ScalarEventScore for ExternalLaw {
        fn event_score(&self, _: f64) -> Result<f64, Self::Error> {
            self.visit(2)?;
            Ok(3.)
        }
    }
    impl ParameterScore for ExternalLaw {
        fn parameter_count(&self) -> usize {
            1
        }
        fn parameter_score(&self, _: &f64, out: &mut [f64]) -> Result<(), Self::Error> {
            self.visit(3)?;
            out[0] = 4.;
            Ok(())
        }
    }
    #[test]
    fn downstream_default_preserves_shape_error_order_and_dyn_compatibility() {
        let law = ExternalLaw {
            fail: Cell::new(0),
            visits: RefCell::new(Vec::new()),
        };
        // The new provided scalar method must not break multivariate/dyn users.
        let erased: &dyn ParameterScore<Event = f64, Error = DistributionError> = &law;
        assert_eq!(erased.parameter_count(), 1);
        assert!(law.log_prob_event_parameter_score(0., &mut []).is_err());
        assert!(law.visits.borrow().is_empty());
        for stage in 1..=3 {
            law.fail.set(stage);
            law.visits.borrow_mut().clear();
            assert!(law.log_prob_event_parameter_score(0., &mut [0.]).is_err());
            assert_eq!(*law.visits.borrow(), (1..=stage).collect::<Vec<_>>());
        }
        law.fail.set(0);
        let mut score = [0.];
        assert_eq!(
            law.log_prob_event_parameter_score(0., &mut score).unwrap(),
            (-2., 3.)
        );
        assert_eq!(score, [4.]);
    }
}
