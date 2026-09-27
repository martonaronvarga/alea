//! Target-bound fixed-length HMC with fallible evaluation and coherent caches.
#![forbid(unsafe_code)]

use std::{error::Error, num::NonZeroUsize};

pub use crate::hamiltonian::Divergence;
use crate::hamiltonian::{PhaseError, PhaseWorkspace, SignedStep};
use kernels::{
    buffer::OwnedBuffer,
    metric::{Metric, MetricError},
    target::{EvaluationError, LogDensityGradient, PointState},
};
use rand::{Rng, RngExt};
use thiserror::Error;

/// Invalid HMC configuration, detected before allocating a chain or evaluating it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum HmcConfigError {
    /// Step size must be strictly positive and finite.
    #[error("step size must be finite and positive")]
    StepSize,
    /// A transition must attempt at least one leapfrog step.
    #[error("leapfrog count must be nonzero")]
    LeapfrogCount,
    /// The absolute endpoint energy-error limit must be positive and finite.
    #[error("energy error limit must be finite and positive")]
    EnergyErrorLimit,
}

/// Finite, strictly positive integration step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepSize(f64);

impl TryFrom<f64> for StepSize {
    type Error = HmcConfigError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() || value <= 0.0 {
            return Err(HmcConfigError::StepSize);
        }
        Ok(Self(value))
    }
}

impl StepSize {
    /// The validated step size.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Validated immutable settings for a fixed-length trajectory.
#[derive(Debug, Clone, Copy)]
pub struct HmcOptions {
    step_size: StepSize,
    leapfrog_steps: NonZeroUsize,
    max_energy_error: f64,
}

impl HmcOptions {
    /// Validates a step size/count with an absolute endpoint energy-error limit of 1000.
    ///
    /// # Errors
    /// Returns [`HmcConfigError`] for a non-positive/non-finite step or zero count.
    pub fn new(step_size: f64, leapfrog_steps: usize) -> Result<Self, HmcConfigError> {
        Ok(Self {
            step_size: step_size.try_into()?,
            leapfrog_steps: NonZeroUsize::new(leapfrog_steps)
                .ok_or(HmcConfigError::LeapfrogCount)?,
            max_energy_error: 1000.0,
        })
    }

    /// Sets the symmetric absolute endpoint energy-error cutoff.
    ///
    /// # Errors
    /// Returns [`HmcConfigError::EnergyErrorLimit`] for non-finite/non-positive input.
    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, HmcConfigError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(HmcConfigError::EnergyErrorLimit);
        }
        self.max_energy_error = limit;
        Ok(self)
    }

    /// Validated positive step size.
    pub fn step_size(self) -> StepSize {
        self.step_size
    }
    /// Fixed nonzero trajectory length in leapfrog steps.
    pub fn leapfrog_steps(self) -> NonZeroUsize {
        self.leapfrog_steps
    }
    /// Maximum allowed absolute endpoint Hamiltonian error.
    pub fn max_energy_error(self) -> f64 {
        self.max_energy_error
    }
}

impl Default for HmcOptions {
    fn default() -> Self {
        Self {
            step_size: StepSize(0.1),
            leapfrog_steps: NonZeroUsize::new(10).expect("ten is nonzero"),
            max_energy_error: 1000.0,
        }
    }
}

/// Typed outcome of a transition, including rejected numerical trajectories.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct HmcTransition {
    /// Whether position, gradient, and density were committed together.
    pub accepted: bool,
    /// Metropolis probability; zero for any divergent trajectory.
    pub acceptance_probability: f64,
    /// Number of attempted steps, including a failing step if present.
    pub leapfrog_steps: usize,
    /// Finite initial Hamiltonian, if it could be computed.
    pub initial_energy: Option<f64>,
    /// Finite endpoint Hamiltonian, if integration reached one.
    pub proposal_energy: Option<f64>,
    /// Finite `proposal_energy - initial_energy`, when available.
    pub energy_error: Option<f64>,
    /// Explicit numerical rejection reason, separate from ordinary MH rejection.
    pub divergence: Option<Divergence>,
}

impl HmcTransition {
    fn new() -> Self {
        Self {
            accepted: false,
            acceptance_probability: 0.0,
            leapfrog_steps: 0,
            initial_energy: None,
            proposal_energy: None,
            energy_error: None,
            divergence: None,
        }
    }

    fn divergent(mut self, reason: Divergence) -> Self {
        self.divergence = Some(reason);
        self
    }

    fn failed<E: Error + 'static>(
        self,
        error: PhaseError<E>,
        leapfrog_step: usize,
    ) -> Result<Self, HmcError<E>> {
        match error {
            PhaseError::Divergence(reason) => Ok(self.divergent(reason)),
            PhaseError::Metric(source) => Err(HmcError::Metric(source)),
            PhaseError::Evaluation(source) => Err(HmcError::Evaluation {
                leapfrog_step,
                source,
            }),
        }
    }
}

/// Construction, configuration-contract, or target/backend failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HmcError<E: Error + 'static> {
    /// HMC requires at least one coordinate, unlike the general target protocol.
    #[error("HMC requires a nonempty target")]
    EmptyTarget,
    /// Metric and target dimensions disagree.
    #[error("metric dimension {metric} does not match target dimension {target}")]
    MetricDimension { target: usize, metric: usize },
    /// A metric operation rejected its buffers.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// Original evaluation failure. Step zero denotes construction/reset/contract checking.
    #[error("evaluation failed at leapfrog step {leapfrog_step}: {source}")]
    Evaluation {
        /// One-based attempted leapfrog step, or zero outside integration.
        leapfrog_step: usize,
        /// Original model error chain or evaluation contract failure.
        #[source]
        source: EvaluationError<E>,
    },
}

/// Fixed-length HMC owning one live cache, one proposal, and reusable aligned scratch.
///
/// Construction evaluates the target once. A successful `L`-step trajectory then
/// requires exactly `L` fused evaluations, with no starting refresh. The target
/// and metric must remain semantically fixed; only read-only live-point access is
/// provided. Rejection, error, and unwinding never mutate the live point. RNG state
/// is consumed on attempted transitions and is not rolled back.
///
/// This is additive to legacy [`super::hmc::Hmc`], not an implementation of the old
/// boolean-returning `Kernel` trait. There is no adaptation or NUTS tree here.
///
/// # Examples
/// ```
/// use kernels::{buffer::OwnedBuffer, dist::Gaussian, metric::IdentityMetric, target::FusedAdapter};
/// use rand::{rngs::SmallRng, SeedableRng};
/// use runtime::{HmcChain, HmcOptions};
/// let target = FusedAdapter::new(&Gaussian, 2);
/// let mut chain = HmcChain::new(&target, OwnedBuffer::new(2),
///     IdentityMetric::new(2), HmcOptions::new(0.1, 8)?)?;
/// let info = chain.step(&mut SmallRng::seed_from_u64(42))?;
/// assert_eq!(info.leapfrog_steps, 8);
/// assert!(chain.point().log_density().is_finite());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct HmcChain<'a, T: LogDensityGradient + ?Sized, M> {
    point: PointState<'a, T>,
    proposal: PhaseWorkspace<'a, T>,
    metric: M,
    options: HmcOptions,
}

impl<'a, T: LogDensityGradient + ?Sized, M: Metric> HmcChain<'a, T, M> {
    /// Validates dimensions, evaluates the initial point once, and allocates scratch.
    ///
    /// # Errors
    /// Returns [`HmcError`] for empty targets, metric mismatch, or initial evaluation failure.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: HmcOptions,
    ) -> Result<Self, HmcError<T::Error>> {
        let dim = target.dimension();
        if dim == 0 {
            return Err(HmcError::EmptyTarget);
        }
        if metric.dim() != dim {
            return Err(HmcError::MetricDimension {
                target: dim,
                metric: metric.dim(),
            });
        }
        let point = PointState::new(target, position).map_err(|source| HmcError::Evaluation {
            leapfrog_step: 0,
            source,
        })?;
        Ok(Self {
            proposal: PhaseWorkspace::new(&point),
            point,
            metric,
            options,
        })
    }

    /// Current coherent cached point, with no mutable state escape hatch.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }
    /// Immutable validated settings used for each transition.
    pub fn options(&self) -> HmcOptions {
        self.options
    }

    /// Explicitly resets the position with one transactional target evaluation.
    ///
    /// # Errors
    /// Returns the original [`EvaluationError`], preserving the old point on failure.
    pub fn set_position(&mut self, position: &[f64]) -> Result<(), EvaluationError<T::Error>> {
        self.proposal.update_point(&mut self.point, position)
    }

    /// Attempts a transition without allocating in the sampler itself.
    ///
    /// Numeric failures return a rejected [`HmcTransition`] with a divergence.
    /// The live point is committed by swapping the entire validated proposal only
    /// on acceptance. Model/backend errors are not silently turned into rejections.
    ///
    /// # Errors
    /// Returns [`HmcError`] for metric/target dimension changes or concrete model
    /// failures. All such errors leave the live cache unchanged. Model/metric panics
    /// propagate; with unwinding the live cache also remains unchanged.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<HmcTransition, HmcError<T::Error>> {
        let dim = self.point.dimension();
        let actual = self.point.target().dimension();
        if actual != dim {
            return Err(HmcError::Evaluation {
                leapfrog_step: 0,
                source: EvaluationError::TargetDimensionChanged {
                    expected: dim,
                    actual,
                },
            });
        }
        if self.metric.dim() != dim {
            return Err(HmcError::MetricDimension {
                target: dim,
                metric: self.metric.dim(),
            });
        }
        let mut info = HmcTransition::new();
        let mut phase = match self.proposal.start(&self.point, &self.metric, rng) {
            Ok(phase) => phase,
            Err(error) => return info.failed(error, 0),
        };
        let initial = match phase.energy() {
            Ok(energy) => energy,
            Err(error) => return info.failed(error, 0),
        };
        info.initial_energy = Some(initial);
        // HmcOptions can only contain a finite positive step size.
        let eps = SignedStep::new(self.options.step_size.value()).expect("validated step size");
        for step in 1..=self.options.leapfrog_steps.get() {
            info.leapfrog_steps = step;
            phase = match phase.step(eps) {
                Ok(phase) => phase,
                Err(error) => return info.failed(error, step),
            };
        }
        let proposed = match phase.energy() {
            Ok(energy) => energy,
            Err(error) => return info.failed(error, info.leapfrog_steps),
        };
        info.proposal_energy = Some(proposed);
        let error = proposed - initial;
        if !error.is_finite() {
            return Ok(info.divergent(Divergence::Energy));
        }
        info.energy_error = Some(error);
        if error.abs() > self.options.max_energy_error {
            return Ok(info.divergent(Divergence::EnergyErrorLimit));
        }
        info.acceptance_probability = (-error).min(0.0).exp();
        info.accepted = rng.random::<f64>() < info.acceptance_probability;
        if info.accepted {
            phase.commit(&mut self.point);
        }
        Ok(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::{
        dist::Gaussian,
        metric::{CholeskyFactor, DenseMetric, DiagonalMetric, IdentityMetric},
        target::FusedAdapter,
    };
    use rand::{SeedableRng, rngs::SmallRng};

    fn buffer(values: &[f64]) -> OwnedBuffer {
        OwnedBuffer::from_fn(values.len(), |i| values[i])
    }

    fn check_momentum<M: Metric>(metric: M) {
        let target = FusedAdapter::new(&Gaussian, 2);
        let mut chain =
            HmcChain::new(&target, OwnedBuffer::new(2), metric, HmcOptions::default()).unwrap();
        let mut rng = SmallRng::seed_from_u64(823);
        let mut replay = rng.clone();
        let z: [f64; 2] = std::array::from_fn(|_| crate::random::standard_normal(&mut replay));
        let info = chain.step(&mut rng).unwrap();
        let expected = 0.5 * z.iter().map(|z| z * z).sum::<f64>();
        assert!((info.initial_energy.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn initial_kinetic_energy_uses_metric_correct_momentum() {
        check_momentum(IdentityMetric::new(2));
        check_momentum(DiagonalMetric::new(buffer(&[4.0, 9.0])).unwrap());
        check_momentum(DenseMetric::new(
            CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 3.0])).unwrap(),
        ));
    }
}
