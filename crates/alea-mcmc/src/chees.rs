//! Static randomized-length HMC with path data for ChEES warmup.
//!
//! Adaptation lives in [`crate::adapt::CheesEnsembleAdapter`]. This kernel can
//! execute a shared warmup plan explicitly; its [`MarkovChain`] transition draws
//! a fresh uniform integration length from frozen parameters.

use std::error::Error;

use alea_core::target::{LogDensityGradient, PointState};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric},
};
use rand::{Rng, RngExt};

use crate::{
    MarkovChain,
    adapt::{CheesObservation, CheesPlan, FrozenCheesParameters},
    canonical::metropolis,
    hamiltonian::{Divergence, PhaseError, PhaseWorkspace, SignedStep},
    hmc::{HmcConfigError, HmcError},
    integrator::{Leapfrog, SymmetricSplitting},
};

/// Fixed retained-sampling settings for randomized static HMC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheesHmcOptions {
    frozen: FrozenCheesParameters,
    max_energy_error: f64,
}

impl CheesHmcOptions {
    /// Wraps parameters returned by completed ChEES warmup.
    pub fn new(frozen: FrozenCheesParameters) -> Self {
        Self {
            frozen,
            max_energy_error: 1000.0,
        }
    }

    /// Sets the symmetric absolute endpoint energy-error diagnostic limit.
    ///
    /// # Errors
    /// Rejects nonpositive and nonfinite limits.
    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, HmcConfigError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(HmcConfigError::EnergyErrorLimit);
        }
        self.max_energy_error = limit;
        Ok(self)
    }

    /// Frozen ChEES parameters.
    pub fn frozen(self) -> FrozenCheesParameters {
        self.frozen
    }
}

/// One randomized static-HMC transition.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct CheesHmcTransition {
    /// Whether the endpoint cache was committed.
    pub accepted: bool,
    /// Endpoint MH probability, zero for numerical divergence.
    pub acceptance_probability: f64,
    /// Shared/adapted or retained/randomized trajectory plan used.
    pub plan: CheesPlan,
    /// Initial Hamiltonian, if available.
    pub initial_energy: Option<f64>,
    /// Endpoint Hamiltonian, if available.
    pub proposal_energy: Option<f64>,
    /// Endpoint minus initial Hamiltonian.
    pub energy_error: Option<f64>,
    /// Numerical rejection reason.
    pub divergence: Option<Divergence>,
}

impl CheesHmcTransition {
    fn new(plan: CheesPlan) -> Self {
        Self {
            accepted: false,
            acceptance_probability: 0.0,
            plan,
            initial_energy: None,
            proposal_energy: None,
            energy_error: None,
            divergence: None,
        }
    }
}

/// Frozen randomized static HMC. No adaptation state is retained here.
#[derive(Debug)]
pub struct CheesHmc<'a, T: LogDensityGradient + ?Sized, M, I = Leapfrog> {
    point: PointState<'a, T>,
    proposal: PhaseWorkspace<'a, T>,
    metric: M,
    integrator: I,
    options: CheesHmcOptions,
    last_initial: OwnedBuffer,
    last_proposal: OwnedBuffer,
    last_endpoint_velocity: OwnedBuffer,
}

impl<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric> CheesHmc<'a, T, M> {
    /// Constructs the retained kernel with leapfrog.
    ///
    /// # Errors
    /// Rejects empty/mismatched dimensions and failed initial evaluation.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: CheesHmcOptions,
    ) -> Result<Self, HmcError<T::Error>> {
        Self::new_with_integrator(target, position, metric, options, Leapfrog)
    }
}

impl<'a, T, M, I> CheesHmc<'a, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: SymmetricSplitting,
{
    /// Constructs the retained kernel with a validated symmetric integrator.
    ///
    /// # Errors
    /// Rejects empty/mismatched dimensions and failed initial evaluation.
    pub fn new_with_integrator(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: CheesHmcOptions,
        integrator: I,
    ) -> Result<Self, HmcError<T::Error>> {
        let dimension = target.dimension();
        if dimension == 0 {
            return Err(HmcError::EmptyTarget);
        }
        if metric.dimension() != dimension {
            return Err(HmcError::MetricDimension {
                target: dimension,
                metric: metric.dimension(),
            });
        }
        let point = PointState::new(target, position).map_err(|source| HmcError::Evaluation {
            integration_step: 0,
            source,
        })?;
        Ok(Self {
            proposal: PhaseWorkspace::new(&point),
            last_initial: OwnedBuffer::new(dimension),
            last_proposal: OwnedBuffer::new(dimension),
            last_endpoint_velocity: OwnedBuffer::new(dimension),
            point,
            metric,
            integrator,
            options,
        })
    }

    /// Current coherent cache.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }

    /// Metric used to turn endpoint momentum into the ChEES path velocity.
    pub fn metric(&self) -> &M {
        &self.metric
    }

    /// Installs parameters returned by a completed shared warmup without target
    /// reevaluation or RNG consumption. Call this at the warmup/sampling boundary,
    /// before any retained draw.
    pub fn install_frozen(&mut self, frozen: FrozenCheesParameters) {
        self.options.frozen = frozen;
    }

    /// Executes an externally shared warmup plan without changing frozen options.
    ///
    /// # Errors
    /// Propagates concrete metric/model failures. Such failures leave the live
    /// point unchanged; RNG and path scratch are not rolled back.
    pub fn step_with_plan<R: Rng + ?Sized>(
        &mut self,
        plan: CheesPlan,
        rng: &mut R,
    ) -> Result<CheesHmcTransition, HmcError<T::Error>> {
        self.last_initial.copy_from_slice(self.point.position());
        self.last_proposal.copy_from_slice(self.point.position());
        self.last_endpoint_velocity.fill(0.0);
        let mut transition = CheesHmcTransition::new(plan);
        let mut phase = match self.proposal.start(&self.point, &self.metric, rng) {
            Ok(phase) => phase,
            Err(error) => return phase_failure(transition, error, 0),
        };
        let initial = match phase.energy() {
            Ok(energy) => energy,
            Err(error) => return phase_failure(transition, error, 0),
        };
        transition.initial_energy = Some(initial);
        let step = SignedStep::new(plan.step_size.value())
            .expect("ChEES plan contains a validated positive step");
        for index in 1..=plan.integration_steps {
            phase = match phase.step_with(step, &self.integrator) {
                Ok(phase) => phase,
                Err(error) => return phase_failure(transition, error, index),
            };
        }
        let proposed = match phase.energy() {
            Ok(energy) => energy,
            Err(error) => return phase_failure(transition, error, plan.integration_steps),
        };
        transition.proposal_energy = Some(proposed);
        transition.energy_error = Some(proposed - initial);
        self.last_proposal.copy_from_slice(phase.point().position());
        self.metric
            .velocity(phase.momentum(), &mut self.last_endpoint_velocity)
            .map_err(HmcError::Metric)?;
        transition.acceptance_probability =
            match metropolis(initial, proposed, self.options.max_energy_error) {
                Ok(probability) => probability,
                Err(reason) => {
                    transition.divergence = Some(reason);
                    return Ok(transition);
                }
            };
        transition.accepted = rng.random::<f64>() < transition.acceptance_probability;
        if transition.accepted {
            phase.commit(&mut self.point);
        }
        Ok(transition)
    }
}

impl<'a, T, I> CheesHmc<'a, T, IdentityMetric, I>
where
    T: LogDensityGradient + ?Sized,
    I: SymmetricSplitting,
{
    /// Forms the paper's identity-coordinate warmup observation from the most
    /// recent transition. Divergences report the initial point and zero velocity.
    /// For nonidentity metrics, callers must instead construct
    /// [`CheesObservation`] in coherently whitened coordinates.
    pub fn observation<'s>(&'s self, transition: &CheesHmcTransition) -> CheesObservation<'s> {
        CheesObservation {
            initial: &self.last_initial,
            proposal: &self.last_proposal,
            endpoint_velocity: &self.last_endpoint_velocity,
            acceptance_probability: transition.acceptance_probability,
            trajectory_time: transition.plan.trajectory_time,
        }
    }
}

fn phase_failure<E: Error + 'static>(
    mut transition: CheesHmcTransition,
    error: PhaseError<E>,
    integration_step: usize,
) -> Result<CheesHmcTransition, HmcError<E>> {
    match error {
        PhaseError::Divergence(reason) => {
            transition.divergence = Some(reason);
            Ok(transition)
        }
        PhaseError::Metric(source) => Err(HmcError::Metric(source)),
        PhaseError::Evaluation(source) => Err(HmcError::Evaluation {
            integration_step,
            source,
        }),
    }
}

impl<T, M, I> MarkovChain for CheesHmc<'_, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: SymmetricSplitting,
{
    type Error = HmcError<T::Error>;
    type Transition = CheesHmcTransition;

    fn position(&self) -> &[f64] {
        self.point.position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        let plan = self.options.frozen.sample_plan(rng);
        self.step_with_plan(plan, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapt::{CheesCriterion, CheesEnsembleAdapter};
    use alea_distributions::StandardNormalTarget;
    use alea_math::metric::IdentityMetric;
    use rand::{SeedableRng, rngs::SmallRng};

    fn frozen(dimension: usize) -> FrozenCheesParameters {
        CheesEnsembleAdapter::new(
            dimension,
            0.2.try_into().unwrap(),
            1.0,
            0.2,
            2.0,
            CheesCriterion::Global,
        )
        .unwrap()
        .finish()
        .unwrap()
    }

    #[test]
    fn plan_execution_exposes_unselected_endpoint_and_keeps_cache_coherent() {
        let target = StandardNormalTarget::new(2);
        let mut chain = CheesHmc::new(
            &target,
            OwnedBuffer::from_fn(2, |i| [1.0, -0.5][i]),
            IdentityMetric::new(2),
            CheesHmcOptions::new(frozen(2)),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(41);
        let plan = CheesPlan {
            step_size: 0.2.try_into().unwrap(),
            integration_steps: 5,
            trajectory_time: 1.0,
        };
        let transition = chain.step_with_plan(plan, &mut rng).unwrap();
        let observation = chain.observation(&transition);
        assert_eq!(observation.initial, &[1.0, -0.5]);
        assert!(observation.proposal.iter().all(|x| x.is_finite()));
        assert!(observation.endpoint_velocity.iter().all(|x| x.is_finite()));
        let expected = -0.5 * chain.position().iter().map(|x| x * x).sum::<f64>();
        assert!((chain.point().log_density() - expected).abs() < 1e-14);
    }

    #[test]
    fn retained_kernel_randomizes_lengths_without_adaptation_state() {
        let target = StandardNormalTarget::new(1);
        let mut chain = CheesHmc::new(
            &target,
            OwnedBuffer::new(1),
            IdentityMetric::new(1),
            CheesHmcOptions::new(frozen(1)),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(7);
        let mut saw_different = false;
        let first = chain.step(&mut rng).unwrap().plan.integration_steps;
        for _ in 0..32 {
            saw_different |= chain.step(&mut rng).unwrap().plan.integration_steps != first;
        }
        assert!(saw_different);
    }
}
