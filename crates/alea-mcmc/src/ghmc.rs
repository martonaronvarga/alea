//! Persistent-momentum generalized HMC and two-stage delayed rejection.
//!
//! Both kernels make the momentum convention explicit. A trajectory proposal is
//! the involution `R * Phi`, where `Phi` is a reversible canonical integrator and
//! `R` flips momentum. After the accept/reject decision, momentum is flipped once
//! unconditionally. Consequently an accepted transition keeps the integrated
//! momentum while a rejected transition keeps the refreshed momentum with its
//! sign reversed. Target/backend failures do not publish either change.
#![forbid(unsafe_code)]

use std::{error::Error, num::NonZeroUsize};

use alea_core::target::{EvaluationError, LogDensityGradient, PointState};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, MetricError},
};
use rand::{Rng, RngExt};
use thiserror::Error;

use crate::{
    MarkovChain,
    canonical::metropolis,
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
    hmc::{Hmc, HmcConfigError, HmcError, HmcOptions, StepSize},
    integrator::{Integrator, Leapfrog},
};

/// Invalid persistent-momentum kernel settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GhmcConfigError {
    /// A shared fixed-HMC setting is invalid.
    #[error(transparent)]
    Hmc(#[from] HmcConfigError),
    /// Persistence is the coefficient on the old momentum and must be in `[0, 1)`.
    #[error("momentum persistence must be finite and in [0, 1)")]
    Persistence,
    /// The second delayed-rejection step is the first step divided by this factor.
    #[error("delayed-rejection scale factor must be finite and greater than one")]
    RetryScale,
}

/// Validated options for persistent-momentum GHMC.
#[derive(Debug, Clone, Copy)]
pub struct GhmcOptions {
    hmc: HmcOptions,
    persistence: f64,
}

impl GhmcOptions {
    /// Creates a fixed-length GHMC kernel.
    ///
    /// `persistence` is the coefficient `c` in
    /// `p <- c p + sqrt(1-c^2) eta`, with `eta ~ N(0, M)`.
    ///
    /// # Errors
    /// Rejects invalid HMC settings or persistence outside `[0, 1)`.
    pub fn new(
        step_size: f64,
        integration_steps: usize,
        persistence: f64,
    ) -> Result<Self, GhmcConfigError> {
        if !persistence.is_finite() || !(0.0..1.0).contains(&persistence) {
            return Err(GhmcConfigError::Persistence);
        }
        Ok(Self {
            hmc: HmcOptions::new(step_size, integration_steps)?,
            persistence,
        })
    }

    /// Sets the absolute endpoint energy-error diagnostic limit.
    ///
    /// # Errors
    /// Rejects a non-positive or non-finite limit.
    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, GhmcConfigError> {
        self.hmc = self.hmc.with_max_energy_error(limit)?;
        Ok(self)
    }

    /// Validated integration step.
    pub fn step_size(self) -> StepSize {
        self.hmc.step_size()
    }

    /// Number of complete integration macrosteps per proposal.
    pub fn integration_steps(self) -> NonZeroUsize {
        self.hmc.integration_steps()
    }

    /// Coefficient multiplying the previous momentum during refreshment.
    pub fn persistence(self) -> f64 {
        self.persistence
    }

    /// Absolute endpoint energy-error diagnostic limit.
    pub fn max_energy_error(self) -> f64 {
        self.hmc.max_energy_error()
    }
}

impl Default for GhmcOptions {
    fn default() -> Self {
        Self {
            hmc: HmcOptions::default(),
            persistence: 0.9,
        }
    }
}

/// Result of one GHMC transition.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct GhmcTransition {
    /// Whether the position and integrated momentum were committed.
    pub accepted: bool,
    /// Endpoint Metropolis probability; zero for a numerical divergence.
    pub acceptance_probability: f64,
    /// Attempted complete integration macrosteps.
    pub integration_steps: usize,
    /// Hamiltonian after partial momentum refreshment.
    pub initial_energy: Option<f64>,
    /// Proposed endpoint Hamiltonian.
    pub proposal_energy: Option<f64>,
    /// Proposed minus initial Hamiltonian.
    pub energy_error: Option<f64>,
    /// Numerical rejection reason, distinct from a backend failure.
    pub divergence: Option<Divergence>,
}

impl GhmcTransition {
    fn empty() -> Self {
        Self {
            accepted: false,
            acceptance_probability: 0.0,
            integration_steps: 0,
            initial_energy: None,
            proposal_energy: None,
            energy_error: None,
            divergence: None,
        }
    }
}

/// Construction or concrete metric/model failure in GHMC and DR-G-HMC.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GhmcError<E: Error + 'static> {
    /// A derived warmup retry step cannot be represented.
    #[error(transparent)]
    Configuration(#[from] HmcConfigError),
    /// Initial point construction failed.
    #[error(transparent)]
    Initialization(#[from] HmcError<E>),
    /// A metric operation failed.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// Initial or refreshed momentum became non-finite after a metric operation.
    #[error("momentum entry {index} is not finite")]
    NonFiniteMomentum { index: usize },
    /// A target evaluation failed at an attempted integration macrostep.
    #[error("evaluation failed at integration step {integration_step}: {source}")]
    Evaluation {
        /// One-based attempted macrostep, or zero before integration.
        integration_step: usize,
        /// Original checked model error.
        #[source]
        source: EvaluationError<E>,
    },
}

fn concrete_phase_error<E: Error + 'static>(
    error: PhaseError<E>,
    integration_step: usize,
) -> Result<Divergence, GhmcError<E>> {
    match error {
        PhaseError::Divergence(reason) => Ok(reason),
        PhaseError::Metric(source) => Err(GhmcError::Metric(source)),
        PhaseError::Evaluation(source) => Err(GhmcError::Evaluation {
            integration_step,
            source,
        }),
    }
}

#[derive(Debug)]
struct PersistentCore<'a, T: LogDensityGradient + ?Sized, M, I> {
    live: PhaseState<'a, T>,
    refreshed_state: PhaseState<'a, T>,
    proposal: PhaseWorkspace<'a, T>,
    metric: M,
    integrator: I,
    standard_normal: OwnedBuffer,
    refreshed: OwnedBuffer,
}

impl<'a, T, M, I> PersistentCore<'a, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    fn new<R: Rng + ?Sized>(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        hmc: HmcOptions,
        integrator: I,
        rng: &mut R,
    ) -> Result<Self, GhmcError<T::Error>> {
        let chain = Hmc::new_with_integrator(target, position, metric, hmc, integrator)?;
        let (point, metric, integrator) = chain.into_parts();
        let dim = point.dimension();
        let mut standard_normal = OwnedBuffer::new(dim);
        for value in standard_normal.iter_mut() {
            *value = alea_math::random::standard_normal(rng);
        }
        let mut momentum = OwnedBuffer::new(dim);
        metric.sample_momentum(&standard_normal, &mut momentum)?;
        let live = PhaseState::new(point, momentum).map_err(|error| match error {
            PhaseError::Metric(source) => GhmcError::Metric(source),
            PhaseError::Evaluation(source) => GhmcError::Evaluation {
                integration_step: 0,
                source,
            },
            PhaseError::Divergence(Divergence::Momentum { index }) => {
                GhmcError::NonFiniteMomentum { index }
            }
            PhaseError::Divergence(_) => unreachable!("construction only validates momentum"),
        })?;
        let refreshed_state = live.clone();
        Ok(Self {
            proposal: PhaseWorkspace::new(live.point()),
            refreshed: OwnedBuffer::new(dim),
            live,
            refreshed_state,
            metric,
            integrator,
            standard_normal,
        })
    }

    fn refresh<R: Rng + ?Sized>(
        &mut self,
        persistence: f64,
        rng: &mut R,
    ) -> Result<(), GhmcError<T::Error>> {
        for value in self.standard_normal.iter_mut() {
            *value = alea_math::random::standard_normal(rng);
        }
        self.metric
            .sample_momentum(&self.standard_normal, &mut self.refreshed)?;
        let innovation = (1.0 - persistence * persistence).sqrt();
        for ((next, previous), noise) in self
            .refreshed
            .iter_mut()
            .zip(self.live.momentum())
            .zip(self.standard_normal.iter_mut())
        {
            // Reuse the standard-normal allocation as a temporary. The sampled
            // metric momentum already lives in `next`.
            *noise = *next;
            *next = persistence * previous + innovation * *noise;
        }
        if let Some(index) = self.refreshed.iter().position(|p| !p.is_finite()) {
            return Err(GhmcError::NonFiniteMomentum { index });
        }
        self.refreshed_state.clone_from(&self.live);
        self.refreshed_state
            .replace_momentum(&self.refreshed)
            .expect("validated finite same-sized refreshed momentum");
        Ok(())
    }

    fn reject_momentum(&mut self) {
        for momentum in self.refreshed.iter_mut() {
            *momentum = -*momentum;
        }
        self.live
            .replace_momentum(&self.refreshed)
            .expect("validated finite same-sized refreshed momentum");
    }

    fn point(&self) -> &PointState<'a, T> {
        self.live.point()
    }
}

/// Persistent-momentum generalized HMC with ordinary reversible MH correction.
#[derive(Debug)]
pub struct Ghmc<'a, T: LogDensityGradient + ?Sized, M, I = Leapfrog> {
    core: PersistentCore<'a, T, M, I>,
    options: GhmcOptions,
}

impl<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric> Ghmc<'a, T, M> {
    /// Constructs GHMC and samples its initial explicit momentum from `N(0, M)`.
    ///
    /// # Errors
    /// Rejects invalid dimensions, initial evaluation, or metric operations.
    pub fn new<R: Rng + ?Sized>(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: GhmcOptions,
        rng: &mut R,
    ) -> Result<Self, GhmcError<T::Error>> {
        Self::new_with_integrator(target, position, metric, options, Leapfrog, rng)
    }
}

impl<'a, T, M, I> Ghmc<'a, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    /// Constructs GHMC with a validated canonical integrator.
    ///
    /// # Errors
    /// Rejects invalid dimensions, initial evaluation, or metric operations.
    pub fn new_with_integrator<R: Rng + ?Sized>(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: GhmcOptions,
        integrator: I,
        rng: &mut R,
    ) -> Result<Self, GhmcError<T::Error>> {
        Ok(Self {
            core: PersistentCore::new(target, position, metric, options.hmc, integrator, rng)?,
            options,
        })
    }

    /// Current coherent position cache.
    pub fn point(&self) -> &PointState<'a, T> {
        self.core.point()
    }

    /// Explicit persistent momentum carried between transitions.
    pub fn momentum(&self) -> &[f64] {
        self.core.live.momentum()
    }

    /// Validated immutable kernel settings.
    pub fn options(&self) -> GhmcOptions {
        self.options
    }

    pub(crate) fn set_step_size(&mut self, step: StepSize) {
        self.options.hmc = HmcOptions::new(step.value(), self.options.integration_steps().get())
            .expect("validated step/count")
            .with_max_energy_error(self.options.max_energy_error())
            .expect("validated cutoff");
    }

    /// Attempts one persistent-momentum transition without sampler allocation.
    ///
    /// Rejection stores the negative refreshed momentum. Acceptance stores the
    /// unflipped integrated momentum. Backend errors publish neither.
    ///
    /// # Errors
    /// Propagates concrete target and metric failures.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<GhmcTransition, GhmcError<T::Error>> {
        let mut info = GhmcTransition::empty();
        self.core.refresh(self.options.persistence, rng)?;
        let mut phase = match self.core.proposal.start_with_momentum(
            self.core.refreshed_state.point(),
            &self.core.metric,
            self.core.refreshed_state.momentum(),
        ) {
            Ok(phase) => phase,
            Err(error) => {
                let reason = concrete_phase_error(error, 0)?;
                self.core.reject_momentum();
                info.divergence = Some(reason);
                return Ok(info);
            }
        };
        let initial = match phase.energy() {
            Ok(energy) => energy,
            Err(error) => {
                let reason = concrete_phase_error(error, 0)?;
                self.core.reject_momentum();
                info.divergence = Some(reason);
                return Ok(info);
            }
        };
        info.initial_energy = Some(initial);
        let step = SignedStep::new(self.options.step_size().value())
            .expect("validated positive GHMC step");
        for index in 1..=self.options.integration_steps().get() {
            info.integration_steps = index;
            phase = match phase.step_with(step, &self.core.integrator) {
                Ok(phase) => phase,
                Err(error) => {
                    let reason = concrete_phase_error(error, index)?;
                    self.core.reject_momentum();
                    info.divergence = Some(reason);
                    return Ok(info);
                }
            };
        }
        let proposed = match phase.energy() {
            Ok(energy) => energy,
            Err(error) => {
                let reason = concrete_phase_error(error, info.integration_steps)?;
                self.core.reject_momentum();
                info.divergence = Some(reason);
                return Ok(info);
            }
        };
        info.proposal_energy = Some(proposed);
        info.energy_error = Some(proposed - initial);
        info.acceptance_probability =
            match metropolis(initial, proposed, self.options.max_energy_error()) {
                Ok(probability) => probability,
                Err(reason) => {
                    self.core.reject_momentum();
                    info.divergence = Some(reason);
                    return Ok(info);
                }
            };
        info.accepted = rng.random::<f64>() < info.acceptance_probability;
        if info.accepted {
            phase
                .save_into(&mut self.core.live)
                .expect("live GHMC state has the proposal dimension");
        } else {
            self.core.reject_momentum();
        }
        Ok(info)
    }
}

/// Settings for two-stage delayed-rejection GHMC.
#[derive(Debug, Clone, Copy)]
pub struct DrGhmcOptions {
    first_step: StepSize,
    second_step: StepSize,
    integration_steps: NonZeroUsize,
    retry_scale: f64,
    persistence: f64,
    max_energy_error: f64,
}

impl DrGhmcOptions {
    /// Creates a two-stage kernel. Stage two uses `first_step / retry_scale`.
    /// Each stage initially attempts one complete leapfrog macrostep.
    ///
    /// # Errors
    /// Rejects invalid step, persistence, or retry scale.
    pub fn new(
        first_step: f64,
        retry_scale: f64,
        persistence: f64,
    ) -> Result<Self, GhmcConfigError> {
        if !retry_scale.is_finite() || retry_scale <= 1.0 {
            return Err(GhmcConfigError::RetryScale);
        }
        if !persistence.is_finite() || !(0.0..1.0).contains(&persistence) {
            return Err(GhmcConfigError::Persistence);
        }
        let first_step: StepSize = first_step.try_into()?;
        let second_step: StepSize = (first_step.value() / retry_scale).try_into()?;
        Ok(Self {
            first_step,
            second_step,
            integration_steps: NonZeroUsize::new(1).expect("one is nonzero"),
            retry_scale,
            persistence,
            max_energy_error: 1000.0,
        })
    }

    /// Sets the absolute energy-error diagnostic applied to every actual and ghost map.
    ///
    /// # Errors
    /// Rejects a non-positive or non-finite limit.
    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, GhmcConfigError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(HmcConfigError::EnergyErrorLimit.into());
        }
        self.max_energy_error = limit;
        Ok(self)
    }

    /// Uses the same positive leapfrog count for both proposal stages.
    ///
    /// # Errors
    /// Rejects zero.
    pub fn with_integration_steps(mut self, steps: usize) -> Result<Self, GhmcConfigError> {
        self.integration_steps =
            NonZeroUsize::new(steps).ok_or(HmcConfigError::IntegrationCount)?;
        Ok(self)
    }

    /// First-stage step size.
    pub fn first_step(self) -> StepSize {
        self.first_step
    }

    /// Geometric divisor used for the second-stage step.
    pub fn retry_scale(self) -> f64 {
        self.retry_scale
    }

    /// Second-stage step size after applying the geometric divisor.
    pub fn second_step(self) -> StepSize {
        self.second_step
    }

    /// Complete macrosteps per actual or reverse-ghost proposal map.
    pub fn integration_steps(self) -> NonZeroUsize {
        self.integration_steps
    }

    /// Coefficient on the old momentum during partial refreshment.
    pub fn persistence(self) -> f64 {
        self.persistence
    }
}

/// One delayed-rejection stage diagnostic.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrGhmcStage {
    /// Actual macrosteps attempted by this stage, excluding reverse ghosts.
    pub integration_steps: usize,
    /// Whether this stage's actual proposal was attempted.
    pub attempted: bool,
    /// Correct delayed-rejection probability, zero for a divergence.
    pub acceptance_probability: f64,
    /// Endpoint minus initial Hamiltonian when both exist.
    pub energy_error: Option<f64>,
    /// Numerical failure of this actual proposal.
    pub divergence: Option<Divergence>,
}

/// Result of one two-stage DR-G-HMC transition.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct DrGhmcTransition {
    /// Macrosteps spent evaluating the reverse first-stage ghost.
    pub ghost_integration_steps: usize,
    /// Whether either proposal was accepted.
    pub accepted: bool,
    /// One-based accepted stage, or `None` after rejection.
    pub accepted_stage: Option<usize>,
    /// First-stage diagnostics.
    pub first: DrGhmcStage,
    /// Second-stage diagnostics; not attempted when stage one accepts.
    pub second: DrGhmcStage,
    /// Actual and reverse-ghost macrosteps evaluated by this transition.
    pub integration_steps: usize,
    /// Hamiltonian after partial momentum refreshment, if finite.
    pub initial_energy: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct MapResult {
    initial: Option<f64>,
    proposed: Option<f64>,
    divergence: Option<Divergence>,
    attempted_steps: usize,
}

fn proposal_map<'a, T, M, I>(
    workspace: &mut PhaseWorkspace<'a, T>,
    start: &PhaseState<'a, T>,
    destination: &mut PhaseState<'a, T>,
    metric: &M,
    integrator: &I,
    step: StepSize,
    integration_steps: NonZeroUsize,
) -> Result<MapResult, GhmcError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    let mut phase = match workspace.start_from_phase(start, metric) {
        Ok(phase) => phase,
        Err(error) => {
            return Ok(MapResult {
                initial: None,
                proposed: None,
                divergence: Some(concrete_phase_error(error, 0)?),
                attempted_steps: 0,
            });
        }
    };
    let initial = match phase.energy() {
        Ok(energy) => energy,
        Err(error) => {
            return Ok(MapResult {
                initial: None,
                proposed: None,
                divergence: Some(concrete_phase_error(error, 0)?),
                attempted_steps: 0,
            });
        }
    };
    let signed = SignedStep::new(step.value()).expect("validated positive DR-G-HMC step");
    for index in 1..=integration_steps.get() {
        phase = match phase.step_with(signed, integrator) {
            Ok(phase) => phase,
            Err(error) => {
                return Ok(MapResult {
                    initial: Some(initial),
                    proposed: None,
                    divergence: Some(concrete_phase_error(error, index)?),
                    attempted_steps: index,
                });
            }
        };
    }
    let proposed = match phase.energy() {
        Ok(energy) => energy,
        Err(error) => {
            return Ok(MapResult {
                initial: Some(initial),
                proposed: None,
                divergence: Some(concrete_phase_error(error, integration_steps.get())?),
                attempted_steps: integration_steps.get(),
            });
        }
    };
    phase
        .save_into(destination)
        .expect("DR-G-HMC scratch snapshots share one dimension");
    destination
        .flip_momentum()
        .expect("negating finite proposal momentum stays finite");
    Ok(MapResult {
        initial: Some(initial),
        proposed: Some(proposed),
        divergence: None,
        attempted_steps: integration_steps.get(),
    })
}

fn map_probability(result: MapResult, limit: f64) -> (f64, Option<Divergence>) {
    if let Some(reason) = result.divergence {
        return (0.0, Some(reason));
    }
    match metropolis(
        result.initial.expect("valid map has initial energy"),
        result.proposed.expect("valid map has endpoint energy"),
        limit,
    ) {
        Ok(probability) => (probability, None),
        Err(reason) => (0.0, Some(reason)),
    }
}

/// Two-stage delayed-rejection generalized HMC.
///
/// Both stages start from the same partially refreshed phase. The second-stage
/// acceptance includes the reverse first-stage rejection probability. This is
/// deterministic delayed rejection; no optional probabilistic-retry factor is
/// hidden in the ratio.
#[derive(Debug)]
pub struct DrGhmc<'a, T: LogDensityGradient + ?Sized, M, I = Leapfrog> {
    core: PersistentCore<'a, T, M, I>,
    ghost_workspace: PhaseWorkspace<'a, T>,
    first: PhaseState<'a, T>,
    second: PhaseState<'a, T>,
    reverse_first: PhaseState<'a, T>,
    options: DrGhmcOptions,
}

impl<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric> DrGhmc<'a, T, M> {
    /// Constructs two-stage DR-G-HMC and samples the explicit initial momentum.
    ///
    /// # Errors
    /// Rejects invalid dimensions, initial evaluation, or metric operations.
    pub fn new<R: Rng + ?Sized>(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: DrGhmcOptions,
        rng: &mut R,
    ) -> Result<Self, GhmcError<T::Error>> {
        Self::new_with_integrator(target, position, metric, options, Leapfrog, rng)
    }
}

impl<'a, T, M, I> DrGhmc<'a, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    /// Constructs two-stage DR-G-HMC with a validated canonical integrator.
    ///
    /// # Errors
    /// Rejects invalid dimensions, initial evaluation, or metric operations.
    pub fn new_with_integrator<R: Rng + ?Sized>(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: DrGhmcOptions,
        integrator: I,
        rng: &mut R,
    ) -> Result<Self, GhmcError<T::Error>> {
        let hmc = HmcOptions::new(options.first_step.value(), 1)
            .expect("DR-G-HMC options retain a validated first step")
            .with_max_energy_error(options.max_energy_error)
            .expect("DR-G-HMC options retain a validated energy-error limit");
        let core = PersistentCore::new(target, position, metric, hmc, integrator, rng)?;
        Ok(Self {
            ghost_workspace: PhaseWorkspace::new(core.live.point()),
            first: core.live.clone(),
            second: core.live.clone(),
            reverse_first: core.live.clone(),
            core,
            options,
        })
    }

    /// Current coherent position cache.
    pub fn point(&self) -> &PointState<'a, T> {
        self.core.point()
    }

    /// Explicit persistent momentum carried between transitions.
    pub fn momentum(&self) -> &[f64] {
        self.core.live.momentum()
    }

    /// Validated immutable settings for both delayed-rejection stages.
    pub fn options(&self) -> DrGhmcOptions {
        self.options
    }

    pub(crate) fn set_step_size(&mut self, step: StepSize) -> Result<(), GhmcError<T::Error>> {
        let second = (step.value() / self.options.retry_scale)
            .try_into()
            .map_err(GhmcError::Configuration)?;
        self.options.first_step = step;
        self.options.second_step = second;
        Ok(())
    }

    /// Attempts one two-stage transition with a bounded one-step reverse ghost.
    ///
    /// # Errors
    /// Propagates concrete metric/model failures from actual and reverse maps.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<DrGhmcTransition, GhmcError<T::Error>> {
        self.core.refresh(self.options.persistence, rng)?;
        let mut transition = DrGhmcTransition {
            ghost_integration_steps: 0,
            accepted: false,
            accepted_stage: None,
            first: DrGhmcStage::default(),
            second: DrGhmcStage::default(),
            integration_steps: 0,
            initial_energy: None,
        };

        let first = proposal_map(
            &mut self.core.proposal,
            &self.core.refreshed_state,
            &mut self.first,
            &self.core.metric,
            &self.core.integrator,
            self.options.first_step,
            self.options.integration_steps,
        )?;
        transition.integration_steps += first.attempted_steps;
        transition.first.integration_steps = first.attempted_steps;
        transition.initial_energy = first.initial;
        transition.first.attempted = true;
        transition.first.energy_error = first
            .initial
            .zip(first.proposed)
            .map(|(initial, proposed)| proposed - initial);
        let (first_probability, first_divergence) =
            map_probability(first, self.options.max_energy_error);
        transition.first.acceptance_probability = first_probability;
        transition.first.divergence = first_divergence;
        if rng.random::<f64>() < first_probability {
            self.core.live.clone_from(&self.first);
            self.core
                .live
                .flip_momentum()
                .expect("accepted finite momentum can be negated");
            transition.accepted = true;
            transition.accepted_stage = Some(1);
            return Ok(transition);
        }

        let second = proposal_map(
            &mut self.core.proposal,
            &self.core.refreshed_state,
            &mut self.second,
            &self.core.metric,
            &self.core.integrator,
            self.options.second_step,
            self.options.integration_steps,
        )?;
        transition.integration_steps += second.attempted_steps;
        transition.second.integration_steps = second.attempted_steps;
        transition.second.attempted = true;
        transition.second.energy_error = second
            .initial
            .zip(second.proposed)
            .map(|(initial, proposed)| proposed - initial);
        if let Some(reason) = second.divergence {
            transition.second.divergence = Some(reason);
            self.core.reject_momentum();
            return Ok(transition);
        }
        let initial = second.initial.expect("valid second map has initial energy");
        let proposed = second
            .proposed
            .expect("valid second map has endpoint energy");
        if let Err(reason) = metropolis(initial, proposed, self.options.max_energy_error) {
            transition.second.divergence = Some(reason);
            self.core.reject_momentum();
            return Ok(transition);
        }

        let reverse = proposal_map(
            &mut self.ghost_workspace,
            &self.second,
            &mut self.reverse_first,
            &self.core.metric,
            &self.core.integrator,
            self.options.first_step,
            self.options.integration_steps,
        )?;
        transition.integration_steps += reverse.attempted_steps;
        transition.ghost_integration_steps = reverse.attempted_steps;
        let (reverse_probability, _) = map_probability(reverse, self.options.max_energy_error);
        let second_probability = delayed_rejection_probability(
            initial - proposed,
            first_probability,
            reverse_probability,
        );
        transition.second.acceptance_probability = second_probability;
        if rng.random::<f64>() < second_probability {
            self.core.live.clone_from(&self.second);
            self.core
                .live
                .flip_momentum()
                .expect("accepted finite momentum can be negated");
            transition.accepted = true;
            transition.accepted_stage = Some(2);
        } else {
            self.core.reject_momentum();
        }
        Ok(transition)
    }
}

impl<T, M, I> MarkovChain for DrGhmc<'_, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    type Error = GhmcError<T::Error>;
    type Transition = DrGhmcTransition;

    fn position(&self) -> &[f64] {
        self.point().position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        DrGhmc::step(self, rng)
    }
}

impl<T, M, I> MarkovChain for Ghmc<'_, T, M, I>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: Integrator,
{
    type Error = GhmcError<T::Error>;
    type Transition = GhmcTransition;

    fn position(&self) -> &[f64] {
        self.point().position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Ghmc::step(self, rng)
    }
}

/// Stable log acceptance for a two-stage deterministic delayed-rejection move.
///
/// The proposal maps must each be volume-preserving involutions. `forward_first`
/// is the first-stage probability from `x`; `reverse_first` is the first-stage
/// probability from the second-stage proposal. The ratio is
/// `pi(y2)/pi(x) * (1-reverse_first)/(1-forward_first)`.
pub fn delayed_rejection_probability(
    log_density_ratio: f64,
    forward_first: f64,
    reverse_first: f64,
) -> f64 {
    if !log_density_ratio.is_finite()
        || !forward_first.is_finite()
        || !reverse_first.is_finite()
        || !(0.0..=1.0).contains(&forward_first)
        || !(0.0..=1.0).contains(&reverse_first)
        || forward_first >= 1.0
    {
        return 0.0;
    }
    if reverse_first >= 1.0 {
        return 0.0;
    }
    let log_probability = log_density_ratio + (-reverse_first).ln_1p() - (-forward_first).ln_1p();
    log_probability.min(0.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alea_distributions::StandardNormalTarget;
    use alea_math::metric::IdentityMetric;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn delayed_rejection_ratio_matches_direct_scalar_oracle() {
        for (ratio, forward, reverse) in [
            (0.7_f64.ln(), 0.2, 0.4),
            (2.0_f64.ln(), 0.7, 0.1),
            ((1e-200_f64).ln(), 0.999_999, 0.25),
        ] {
            let expected = (ratio.exp() * (1.0 - reverse) / (1.0 - forward)).min(1.0);
            let actual = delayed_rejection_probability(ratio, forward, reverse);
            assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.max(1.0));
        }
        assert_eq!(delayed_rejection_probability(0.0, 1.0, 0.0), 0.0);
        assert_eq!(delayed_rejection_probability(0.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn ghmc_numerical_rejection_keeps_negative_refreshed_momentum() {
        let target = StandardNormalTarget::new(2);
        let options = GhmcOptions::new(0.9, 1, 0.0)
            .unwrap()
            .with_max_energy_error(1e-16)
            .unwrap();
        let mut rng = SmallRng::seed_from_u64(812);
        let mut replay = rng.clone();
        // Constructor samples and stores an initial momentum first.
        for _ in 0..2 {
            let _ = alea_math::random::standard_normal(&mut replay);
        }
        let refreshed: [f64; 2] =
            std::array::from_fn(|_| alea_math::random::standard_normal(&mut replay));
        let mut chain = Ghmc::new(
            &target,
            OwnedBuffer::from_fn(2, |i| [1.0, -0.4][i]),
            IdentityMetric::new(2),
            options,
            &mut rng,
        )
        .unwrap();
        let before = chain.point().position().to_vec();
        let transition = chain.step(&mut rng).unwrap();
        assert!(!transition.accepted);
        assert_eq!(transition.divergence, Some(Divergence::EnergyErrorLimit));
        assert_eq!(chain.point().position(), before);
        for (&actual, expected) in chain.momentum().iter().zip(refreshed) {
            assert!((actual + expected).abs() < 1e-15);
        }
    }

    #[test]
    fn delayed_rejection_scalar_ratio_obeys_pairwise_balance_oracle() {
        let log_ratio = -0.7;
        let forward = 0.35;
        let reverse = 0.2;
        let alpha_xy = delayed_rejection_probability(log_ratio, forward, reverse);
        let alpha_yx = delayed_rejection_probability(-log_ratio, reverse, forward);
        let left = (1.0 - forward) * alpha_xy;
        let right = log_ratio.exp() * (1.0 - reverse) * alpha_yx;
        assert!((left - right).abs() < 2e-15);
    }

    #[test]
    fn staged_rejection_executes_configured_actual_and_reverse_work() {
        let target = StandardNormalTarget::new(2);
        let options = DrGhmcOptions::new(2.0, 4.0, 0.0)
            .unwrap()
            .with_integration_steps(3)
            .unwrap();
        assert_eq!(options.integration_steps().get(), 3);
        let mut exercised = false;
        for seed in 0..64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut chain = DrGhmc::new(
                &target,
                OwnedBuffer::from_fn(2, |i| [0.7, -0.3][i]),
                IdentityMetric::new(2),
                options,
                &mut rng,
            )
            .unwrap();
            let transition = chain.step(&mut rng).unwrap();
            if transition.second.attempted {
                assert!(matches!(transition.integration_steps, 6 | 9));
                exercised = true;
                break;
            }
        }
        assert!(exercised, "reference seeds never reached delayed rejection");
    }
}
