//! Fixed-scale derivative-free random-walk Metropolis.
//! Warmup is an explicit controller; transitions never adapt implicitly.
use crate::{
    MarkovChain,
    config::{SamplerConfigError, StepSize},
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

#[derive(Debug, Clone, Copy)]
pub struct RwmhOptions {
    step_size: StepSize,
}
impl RwmhOptions {
    pub fn new(step_size: f64) -> Result<Self, SamplerConfigError> {
        Ok(Self {
            step_size: StepSize::try_from(step_size)?,
        })
    }
    pub fn step_size(self) -> StepSize {
        self.step_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalRejection {
    Position,
    LogDensity,
}

#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RwmhTransition {
    pub accepted: bool,
    pub acceptance_probability: f64,
    pub numerical_rejection: Option<NumericalRejection>,
}

#[derive(Debug, thiserror::Error)]
pub enum RwmhError<E: std::error::Error + 'static> {
    #[error("random-walk Metropolis requires a nonempty target")]
    EmptyTarget,
    #[error("proposal metric dimension {metric} differs from target dimension {target}")]
    MetricDimension { target: usize, metric: usize },
    #[error(transparent)]
    Metric(#[from] MetricError),
    #[error(transparent)]
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
    pub fn point(&self) -> &DensityPoint<'a, T> {
        &self.point
    }
    pub fn options(&self) -> RwmhOptions {
        self.options
    }
    /// Installs an already validated scale; cache and RNG are untouched.
    pub fn set_step_size(&mut self, step_size: StepSize) {
        self.options.step_size = step_size;
    }
    pub fn set_position(&mut self, position: &[f64]) -> Result<(), RwmhError<T::Error>> {
        self.point.try_update(position)?;
        Ok(())
    }
    /// Returns typed rejection for numerical failure; model errors propagate.
    /// Errors and panics leave the live point unchanged. RNG consumption is not rolled back.
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
            *p = q + self.options.step_size.value() * *p;
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
