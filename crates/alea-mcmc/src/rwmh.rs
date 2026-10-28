//! Fixed-scale derivative-free random-walk Metropolis.
//! Warmup is an explicit controller; transitions never adapt implicitly.
#![deny(missing_docs)]
use crate::{
    MarkovChain,
    config::{ProposalScale, SamplerConfigError},
};
use alea_core::{
    density::{DensityPoint, LogDensity},
    target::EvaluationError,
};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, MetricError},
    random::NormalGenerator,
};
use rand::{Rng, RngExt};

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
/// Validated fixed-scale random-walk proposal settings.
pub struct RwmhOptions {
    proposal_scale: ProposalScale,
}
impl RwmhOptions {
    /// Validates a finite, strictly positive proposal scale.
    ///
    /// # Errors
    /// Returns [`SamplerConfigError`] when `proposal_scale` is non-finite or not positive.
    pub fn new(proposal_scale: f64) -> Result<Self, SamplerConfigError> {
        Ok(Self {
            proposal_scale: ProposalScale::try_from(proposal_scale)?,
        })
    }
    /// Validated proposal scale.
    pub fn proposal_scale(self) -> ProposalScale {
        self.proposal_scale
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
/// Numerical proposal failures recorded as rejections rather than model errors.
pub enum NumericalRejection {
    /// The proposed position was nonfinite.
    Position,
    /// The proposed log density was nonfinite.
    LogDensity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
/// Owned summary of one completed random-walk proposal.
pub struct RwmhTransition {
    /// Number of accepted proposals.
    pub accepted: bool,
    /// Metropolis acceptance probability for this proposal.
    pub acceptance_probability: f64,
    /// Numerical failure that rejected the proposal, if any.
    pub numerical_rejection: Option<NumericalRejection>,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
/// Construction or target-evaluation failure with the original model error retained.
pub enum RwmhError<E: std::error::Error + 'static> {
    #[error("random-walk Metropolis requires a nonempty target")]
    /// Random-walk Metropolis requires a nonempty target.
    EmptyTarget,
    #[error("proposal metric dimension {metric} differs from target dimension {target}")]
    /// Proposal metric dimension `metric` differs from target dimension `target`.
    MetricDimension {
        /// Target dimension.
        target: usize,
        /// Metric dimension.
        metric: usize,
    },
    #[error(transparent)]
    /// Underlying metric construction or operation failed.
    Metric(#[from] MetricError),
    #[error(transparent)]
    /// Checked target evaluation failed; its concrete source error is retained.
    Evaluation(#[from] EvaluationError<E>),
}

/// Target-bound random-walk chain. `metric` is the proposal covariance before
/// multiplication by the squared step size, not an HMC inverse mass.
pub struct Rwmh<'a, T: LogDensity + ?Sized, M> {
    point: DensityPoint<'a, T>,
    proposal: DensityPoint<'a, T>,
    noise: OwnedBuffer,
    position: OwnedBuffer,
    metric: M,
    options: RwmhOptions,
    normal: NormalGenerator,
}
impl<'a, T: LogDensity + ?Sized, M: EuclideanMetric> Rwmh<'a, T, M> {
    /// Validates dimensions and evaluates the initial cache before constructing a chain.
    ///
    /// # Errors
    /// Returns [`RwmhError`] for an empty target, metric/target dimension mismatch,
    /// invalid initial point, or target evaluation failure.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: RwmhOptions,
    ) -> Result<Self, RwmhError<T::Error>> {
        let dim = target.dimension();
        if dim == 0 {
            return Err(RwmhError::EmptyTarget);
        }
        if metric.dimension() != dim {
            return Err(RwmhError::MetricDimension {
                target: dim,
                metric: metric.dimension(),
            });
        }
        let point = DensityPoint::new(target, position)?;
        Ok(Self {
            proposal: point.clone(),
            point,
            metric,
            options,
            normal: NormalGenerator::default(),
            noise: OwnedBuffer::new(dim),
            position: OwnedBuffer::new(dim),
        })
    }
    /// Coherent retained point after the last completed transition.
    pub fn point(&self) -> &DensityPoint<'a, T> {
        &self.point
    }
    /// Current validated fixed proposal options.
    pub fn options(&self) -> RwmhOptions {
        self.options
    }
    /// Chain-local normal state required in addition to the uniform RNG for replay.
    pub fn normal_generator(&self) -> &NormalGenerator {
        &self.normal
    }
    /// Restores the normal cache when rebuilding a frozen checkpoint.
    pub fn with_normal_generator(mut self, normal: NormalGenerator) -> Self {
        self.normal = normal;
        self
    }
    /// Installs an already validated scale; cache and RNG are untouched.
    pub fn set_proposal_scale(&mut self, proposal_scale: ProposalScale) {
        self.options.proposal_scale = proposal_scale;
    }
    /// Re-evaluates and transactionally installs a new retained position.
    ///
    /// # Errors
    /// Returns [`RwmhError::Evaluation`] without changing the live point if the
    /// position or target evaluation is invalid.
    pub fn set_position(&mut self, position: &[f64]) -> Result<(), RwmhError<T::Error>> {
        self.point.try_update(position)?;
        Ok(())
    }
    /// Returns typed rejection for numerical failure; model errors propagate.
    /// Errors and panics leave the live point unchanged. RNG consumption is not rolled back.
    ///
    /// # Errors
    /// Returns [`RwmhError`] for a changed target/metric dimension, metric failure,
    /// or non-numerical target evaluation failure.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<RwmhTransition, RwmhError<T::Error>> {
        let dimension = self.point.target().dimension();
        if dimension != self.point.dimension() {
            return Err(EvaluationError::TargetDimensionChanged {
                expected: self.point.dimension(),
                actual: dimension,
            }
            .into());
        }
        if self.metric.dimension() != dimension {
            return Err(RwmhError::MetricDimension {
                target: dimension,
                metric: self.metric.dimension(),
            });
        }
        for z in self.noise.iter_mut() {
            *z = self.normal.sample(rng);
        }
        self.metric
            .sample_momentum(&self.noise, &mut self.position)?;
        for (p, q) in self.position.iter_mut().zip(self.point.position()) {
            *p = q + self.options.proposal_scale.value() * *p;
        }
        if let Err(error) = self.proposal.try_update(&self.position) {
            let reason = match error {
                EvaluationError::NonFinitePosition { .. } => NumericalRejection::Position,
                EvaluationError::NonFiniteLogDensity => NumericalRejection::LogDensity,
                error => return Err(error.into()),
            };
            return Ok(RwmhTransition {
                accepted: false,
                acceptance_probability: 0.0,
                numerical_rejection: Some(reason),
            });
        }
        let probability = (self.proposal.log_density() - self.point.log_density())
            .min(0.0)
            .exp();
        let accepted = rng.random::<f64>() < probability;
        if accepted {
            self.point.clone_from(&self.proposal);
        }
        Ok(RwmhTransition {
            accepted,
            acceptance_probability: probability,
            numerical_rejection: None,
        })
    }
}
impl<T: LogDensity + ?Sized, M: EuclideanMetric> MarkovChain for Rwmh<'_, T, M> {
    type Error = RwmhError<T::Error>;
    type Transition = RwmhTransition;
    fn position(&self) -> &[f64] {
        self.point.position()
    }
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Rwmh::step(self, rng)
    }
}
