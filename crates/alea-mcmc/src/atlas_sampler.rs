//! Opt-in full ATLAS with preliminary, delayed and failure branches.
//!
//! Acceptance follows arXiv:2410.21587v1 Appendix E.2. The delayed ratio uses
//! the actual forward selection densities q(n|x), not their Hastings ratios.
//! Local curvature uses a deterministic bounded HVP policy, rather than random
//! BFGS/power-iteration auxiliaries. All baseline geometry and empirical trajectory
//! lengths are frozen. This research implementation is outside the reference facade.
use crate::{
    Hmc, MarkovChain,
    config::StepSize,
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
};
use alea_core::{
    capability::{HessianVector, HvpError, evaluate_hvp},
    target::{EvaluationError, PointState},
};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, MetricError},
};
use rand::{Rng, RngExt};
use std::{error::Error, num::NonZeroUsize};

/// Validated fixed ATLAS policy: constant-trajectory2, probabilistic1, one retry.
#[derive(Debug, Clone, Copy)]
pub struct AtlasOptions {
    baseline: StepSize,
    maximum: NonZeroUsize,
    sigma: f64,
}
impl AtlasOptions {
    /// Baseline step and maximum trajectory work (3..=1024 macrosteps per leg).
    /// Local lognormal width is log(1.2); offsets are uniform on[.33,.66).
    /// # Errors
    /// Rejects invalid steps or work budgets.
    pub fn new(step: f64, maximum: usize) -> Result<Self, AtlasConfigError> {
        if !(3..=1024).contains(&maximum) {
            return Err(AtlasConfigError);
        }
        Ok(Self {
            baseline: step.try_into().map_err(|_| AtlasConfigError)?,
            maximum: NonZeroUsize::new(maximum).expect("positive"),
            sigma: 1.2_f64.ln(),
        })
    }
    /// Fixed preliminary step size.
    pub fn baseline_step(self) -> StepSize {
        self.baseline
    }
    /// Maximum macrosteps in any forward, reverse or ghost leg.
    pub fn maximum_steps(self) -> usize {
        self.maximum.get()
    }
}
/// Invalid geometry, trajectory distribution or bounded resource policy.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("invalid ATLAS configuration or trajectory distribution")]
pub struct AtlasConfigError;
/// Concrete failures propagate without publishing a partial live state.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AtlasError<E: Error + 'static> {
    /// Invalid initial frozen policy.
    #[error(transparent)]
    Configuration(#[from] AtlasConfigError),
    /// Target/metric/phase error with its original source type.
    #[error(transparent)]
    Phase(#[from] PhaseError<E>),
    /// Concrete initial target error.
    #[error(transparent)]
    Target(#[from] EvaluationError<E>),
    /// Local HVP capability failure.
    #[error(transparent)]
    Curvature(#[from] HvpError<E>),
    /// Metric action failure.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// Work accounting overflow.
    #[error("ATLAS work counter overflow")]
    CounterOverflow,
}
/// Branch producing the final attempted proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtlasStage {
    /// Baseline GIST/NoUT proposal.
    Preliminary,
    /// Rejected supported preliminary proposal, followed by one locally tuned retry.
    Delayed,
    /// Too-short preliminary trajectory, followed by an empirical-length retry.
    Failure,
}
/// Explicit rejection reason. A valid earlier phase is never a partial live commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtlasRejection {
    /// Ordinary MH rejection.
    Metropolis,
    /// The preliminary proposal has no reverse selection support.
    SubUTurn,
    /// The reverse delayed proposal belongs to a different branch.
    ReverseBranch,
    /// A local step implies zero or excessive trajectory work.
    WorkLimit,
    /// Nonfinite integration or step arithmetic.
    Divergence(Divergence),
}
/// All work includes discarded trajectories and reverse/ghost legs.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct AtlasTransition {
    /// Whether a complete proposal replaced the live point.
    pub accepted: bool,
    /// Branch of the final attempted proposal.
    pub stage: AtlasStage,
    /// Conditional MH probability in that branch (not a marginal acceptance rate).
    pub acceptance_probability: f64,
    /// Preliminary trajectory length, including its stopping attempt.
    pub preliminary_length: usize,
    /// Actually attempted macrosteps across every leg.
    pub integration_steps: usize,
    /// Macrosteps in actual preliminary and delayed forward proposals.
    pub forward_integration_steps: usize,
    /// Macrosteps checking reverse support of the actual preliminary proposal.
    pub reverse_integration_steps: usize,
    /// Macrosteps constructing delayed-branch ghosts and their reverse checks.
    pub ghost_integration_steps: usize,
    /// Actual HVP evaluations for forward and reverse local step densities.
    pub hvp_calls: usize,
    /// Initial Hamiltonian.
    pub initial_energy: Option<f64>,
    /// Final proposed Hamiltonian, when finite.
    pub proposal_energy: Option<f64>,
    /// Final proposal minus initial energy.
    pub energy_error: Option<f64>,
    /// Rejection reason; absent on acceptance.
    pub rejection: Option<AtlasRejection>,
}
impl AtlasTransition {
    fn new() -> Self {
        Self {
            accepted: false,
            stage: AtlasStage::Preliminary,
            acceptance_probability: 0.,
            preliminary_length: 0,
            integration_steps: 0,
            forward_integration_steps: 0,
            reverse_integration_steps: 0,
            ghost_integration_steps: 0,
            hvp_calls: 0,
            initial_energy: None,
            proposal_energy: None,
            energy_error: None,
            rejection: None,
        }
    }
}
#[derive(Debug)]
struct Curvature {
    direction: OwnedBuffer,
    hessian: OwnedBuffer,
    first: OwnedBuffer,
    second: OwnedBuffer,
}
impl Curvature {
    fn new(d: usize) -> Self {
        Self {
            direction: OwnedBuffer::new(d),
            hessian: OwnedBuffer::new(d),
            first: OwnedBuffer::new(d),
            second: OwnedBuffer::new(d),
        }
    }
    fn mean<T: HessianVector + ?Sized, M: EuclideanMetric>(
        &mut self,
        point: &PointState<'_, T>,
        metric: &M,
        options: AtlasOptions,
        work: &mut AtlasTransition,
    ) -> Result<f64, AtlasError<T::Error>> {
        let d = point.dimension();
        let mut estimate = 0_f64;
        // Deterministic starts make q(epsilon|phase) reproducible in both directions.
        // Missing a direction affects efficiency, not the accounted proposal law.
        for restart in 0..2 {
            for (i, x) in self.direction.iter_mut().enumerate() {
                *x = if restart == 0 {
                    f64::from(i == 0)
                } else {
                    if i % 2 == 0 { 1. } else { -1. }
                } / (d as f64).sqrt();
            }
            for _ in 0..3 {
                evaluate_hvp(
                    point.target(),
                    point.position(),
                    &self.direction,
                    &mut self.hessian,
                )?;
                metric.velocity(&self.hessian, &mut self.first)?;
                evaluate_hvp(
                    point.target(),
                    point.position(),
                    &self.first,
                    &mut self.hessian,
                )?;
                metric.velocity(&self.hessian, &mut self.second)?;
                work.hvp_calls += 2;
                let norm = norm(&self.second);
                if norm == 0. {
                    self.direction.fill(0.);
                    self.direction[0] = 1.;
                } else if norm.is_finite() {
                    for (x, y) in self.direction.iter_mut().zip(self.second.iter()) {
                        *x = y / norm;
                    }
                } else {
                    return Ok(2. * options.baseline.value() / 1000.);
                }
            }
            evaluate_hvp(
                point.target(),
                point.position(),
                &self.direction,
                &mut self.hessian,
            )?;
            metric.velocity(&self.hessian, &mut self.first)?;
            work.hvp_calls += 1;
            estimate = estimate.max(norm(&self.first));
        }
        let baseline = options.baseline.value();
        let mean = if estimate > 0. {
            (0.5 / estimate.sqrt()).min(0.5 * baseline)
        } else {
            0.5 * baseline
        };
        Ok(if mean >= baseline / 1000. && mean.is_finite() {
            mean
        } else {
            2. * baseline / 1000.
        })
    }
}
fn norm(x: &[f64]) -> f64 {
    let scale = x.iter().fold(0_f64, |a, b| a.max(b.abs()));
    if scale == 0. {
        0.
    } else {
        scale * x.iter().map(|v| (v / scale).powi(2)).sum::<f64>().sqrt()
    }
}

/// Full bounded ATLAS research chain. Adaptation controllers never enter it.
#[derive(Debug)]
pub struct Atlas<'a, T: HessianVector + ?Sized, M: EuclideanMetric> {
    point: PointState<'a, T>,
    metric: M,
    options: AtlasOptions,
    lengths: Box<[usize]>,
    origin: PhaseState<'a, T>,
    candidate: PhaseState<'a, T>,
    proposal: PhaseState<'a, T>,
    ghost: PhaseState<'a, T>,
    history: Vec<PhaseState<'a, T>>,
    workspace: PhaseWorkspace<'a, T>,
    curvature: Curvature,
}
// Private auxiliary-variates seam for independent deterministic proposal fixtures.
trait ProposalChoices {
    fn offset(&mut self) -> f64;
    fn index(&mut self, range: std::ops::Range<usize>) -> usize;
    fn local_step(&mut self, mean: f64, sigma: f64) -> f64;
    fn uniform(&mut self) -> f64;
}
struct RandomChoices<'a, R: ?Sized>(&'a mut R);
impl<R: Rng + ?Sized> ProposalChoices for RandomChoices<'_, R> {
    fn offset(&mut self) -> f64 {
        self.0.random_range(0.33..0.66)
    }
    fn index(&mut self, range: std::ops::Range<usize>) -> usize {
        self.0.random_range(range)
    }
    fn local_step(&mut self, mean: f64, sigma: f64) -> f64 {
        let mu = mean.ln() - 0.5 * sigma * sigma;
        (mu + sigma * alea_math::random::standard_normal(self.0)).exp()
    }
    fn uniform(&mut self) -> f64 {
        self.0.random::<f64>()
    }
}

impl<'a, T: HessianVector + ?Sized, M: EuclideanMetric> Atlas<'a, T, M> {
    /// Consume a frozen leapfrog HMC cache and an empirical trajectory distribution.
    /// Repeated lengths encode empirical probability mass. No target call/RNG use.
    /// # Errors
    /// Rejects empty, excessive or invalid lengths and unaddressable history.
    pub fn from_hmc(
        chain: Hmc<'a, T, M>,
        options: AtlasOptions,
        lengths: &[usize],
    ) -> Result<Self, AtlasError<T::Error>> {
        validate_lengths(lengths, options)?;
        let (point, metric, _) = chain.into_parts();
        let d = point.dimension();
        d.checked_mul(options.maximum.get())
            .and_then(|n| n.checked_mul(3 * size_of::<f64>()))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(AtlasConfigError)?;
        let origin = PhaseState::new(point.clone(), OwnedBuffer::new(d))?;
        Ok(Self {
            candidate: origin.clone(),
            proposal: origin.clone(),
            ghost: origin.clone(),
            history: (0..options.maximum.get()).map(|_| origin.clone()).collect(),
            workspace: PhaseWorkspace::new(&point),
            curvature: Curvature::new(d),
            origin,
            point,
            metric,
            options,
            lengths: lengths.into(),
        })
    }
    /// Coherent live point.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }
    /// Frozen baseline geometry.
    pub fn metric(&self) -> &M {
        &self.metric
    }
    /// Frozen baseline step and resource controls.
    pub fn options(&self) -> AtlasOptions {
        self.options
    }
    /// Frozen empirical lengths, with repetitions preserving probability mass.
    pub fn trajectory_lengths(&self) -> &[usize] {
        &self.lengths
    }
    /// Execute all ATLAS branches using fixed baseline geometry/distribution.
    /// # Errors
    /// Backend and shape/capability errors retain their concrete source and leave
    /// the live point unchanged. Numerical support failures are explicit rejections.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<AtlasTransition, AtlasError<T::Error>> {
        let mut phase = self.workspace.start(&self.point, &self.metric, rng)?;
        let initial = phase.energy()?;
        phase.save_into(&mut self.origin)?;
        self.propose(initial, &mut RandomChoices(rng))
    }
    fn propose(
        &mut self,
        initial: f64,
        choices: &mut impl ProposalChoices,
    ) -> Result<AtlasTransition, AtlasError<T::Error>> {
        let mut result = AtlasTransition::new();
        result.initial_energy = Some(initial);
        let offset = choices.offset();
        let length = trace(
            &self.origin,
            &self.metric,
            self.options,
            &mut self.workspace,
            &mut self.history,
            &mut result,
        )?;
        result.forward_integration_steps = result.integration_steps;
        result.preliminary_length = length;
        let failure = length < 3;
        let mut preliminary_log_accept = f64::NEG_INFINITY;
        let mut preliminary_log_density = 0.;
        let n = if failure {
            self.lengths[choices.index(0..self.lengths.len())]
        } else {
            let lower = lower(length, offset);
            let n = choices.index(lower..length);
            preliminary_log_density = selection_log_density(length, offset, n);
            self.candidate.clone_from(&self.history[n]);
            self.candidate.flip_momentum()?;
            let reverse = trace(
                &self.candidate,
                &self.metric,
                self.options,
                &mut self.workspace,
                &mut self.history,
                &mut result,
            )?;
            let reverse_log = if reverse >= 3 {
                selection_log_density(reverse, offset, n)
            } else {
                f64::NEG_INFINITY
            };
            result.reverse_integration_steps =
                result.integration_steps - result.forward_integration_steps;
            let energy = energy(&self.candidate, &self.metric, &mut self.workspace)?;
            preliminary_log_accept =
                (-energy + result.initial_energy.expect("initial energy") + reverse_log
                    - preliminary_log_density)
                    .min(0.);
            result.proposal_energy = Some(energy);
            result.energy_error = Some(energy - result.initial_energy.expect("initial"));
            result.acceptance_probability = preliminary_log_accept.exp();
            if choices.uniform() < result.acceptance_probability {
                self.point.clone_from(self.candidate.point());
                result.accepted = true;
                return Ok(result);
            }
            if !reverse_log.is_finite() {
                result.rejection = Some(AtlasRejection::SubUTurn);
                return Ok(result);
            }
            n
        };
        result.stage = if failure {
            AtlasStage::Failure
        } else {
            AtlasStage::Delayed
        };
        result.acceptance_probability = 0.;
        let mean =
            self.curvature
                .mean(self.origin.point(), &self.metric, self.options, &mut result)?;
        let sigma = self.options.sigma;
        let local_step = choices.local_step(mean, sigma);
        if !local_step.is_finite() || local_step <= 0. {
            result.rejection = Some(AtlasRejection::Divergence(Divergence::Energy));
            return Ok(result);
        }
        let count = (n as f64 * self.options.baseline.value() / local_step).floor();
        let count = if failure {
            count.clamp(3., self.options.maximum.get() as f64)
        } else {
            count
        };
        if count < 1. || count > self.options.maximum.get() as f64 {
            result.rejection = Some(AtlasRejection::WorkLimit);
            return Ok(result);
        }
        let before_forward = result.integration_steps;
        let forward_divergence = fixed(
            &self.origin,
            &self.metric,
            local_step,
            count as usize,
            &mut self.workspace,
            &mut self.proposal,
            &mut result,
        )?;
        result.forward_integration_steps += result.integration_steps - before_forward;
        if let Some(reason) = forward_divergence {
            result.rejection = Some(AtlasRejection::Divergence(reason));
            return Ok(result);
        }
        self.proposal.flip_momentum()?;
        let proposed = energy(&self.proposal, &self.metric, &mut self.workspace)?;
        result.proposal_energy = Some(proposed);
        result.energy_error = Some(proposed - result.initial_energy.expect("initial"));
        let before_ghost = result.integration_steps;
        let ghost_length = trace(
            &self.proposal,
            &self.metric,
            self.options,
            &mut self.workspace,
            &mut self.history,
            &mut result,
        )?;
        result.ghost_integration_steps += result.integration_steps - before_ghost;
        let mut log_ratio = result.initial_energy.expect("initial") - proposed;
        if failure {
            if ghost_length >= 3 {
                result.rejection = Some(AtlasRejection::ReverseBranch);
                return Ok(result);
            }
        } else {
            let ghost_log_density = selection_log_density(ghost_length, offset, n);
            if ghost_length < 3 || !ghost_log_density.is_finite() {
                result.rejection = Some(AtlasRejection::ReverseBranch);
                return Ok(result);
            }
            self.ghost.clone_from(&self.history[n]);
            self.ghost.flip_momentum()?;
            let before_ghost = result.integration_steps;
            let reverse = trace(
                &self.ghost,
                &self.metric,
                self.options,
                &mut self.workspace,
                &mut self.history,
                &mut result,
            )?;
            result.ghost_integration_steps += result.integration_steps - before_ghost;
            let reverse_log = if reverse >= 3 {
                selection_log_density(reverse, offset, n)
            } else {
                f64::NEG_INFINITY
            };
            if !reverse_log.is_finite() {
                result.rejection = Some(AtlasRejection::ReverseBranch);
                return Ok(result);
            }
            let ghost_energy = energy(&self.ghost, &self.metric, &mut self.workspace)?;
            let ghost_accept = (-ghost_energy + proposed + reverse_log - ghost_log_density).min(0.);
            log_ratio += delayed_selection_ratio(
                preliminary_log_density,
                ghost_log_density,
                preliminary_log_accept,
                ghost_accept,
            );
        }
        let reverse_mean = self.curvature.mean(
            self.proposal.point(),
            &self.metric,
            self.options,
            &mut result,
        )?;
        log_ratio += log_step_density(local_step, reverse_mean, sigma)
            - log_step_density(local_step, mean, sigma);
        result.acceptance_probability = if log_ratio.is_nan() {
            0.
        } else {
            log_ratio.min(0.).exp()
        };
        result.accepted = choices.uniform() < result.acceptance_probability;
        if result.accepted {
            self.point.clone_from(self.proposal.point());
        } else {
            result.rejection = Some(AtlasRejection::Metropolis);
        }
        Ok(result)
    }
}
impl<T: HessianVector + ?Sized, M: EuclideanMetric> MarkovChain for Atlas<'_, T, M> {
    type Error = AtlasError<T::Error>;
    type Transition = AtlasTransition;
    fn position(&self) -> &[f64] {
        self.point.position()
    }
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Self::step(self, rng)
    }
}
fn validate_lengths(lengths: &[usize], options: AtlasOptions) -> Result<(), AtlasConfigError> {
    if lengths.is_empty()
        || lengths.len() > 4096
        || lengths.iter().any(|n| *n < 3 || *n > options.maximum.get())
    {
        Err(AtlasConfigError)
    } else {
        Ok(())
    }
}
fn lower(length: usize, offset: f64) -> usize {
    ((length as f64 * offset).floor() as usize).max(1)
}
fn selection_log_density(length: usize, offset: f64, n: usize) -> f64 {
    let lo = lower(length, offset);
    if n < lo || n >= length {
        f64::NEG_INFINITY
    } else {
        -((length - lo) as f64).ln()
    }
}
fn log_reject(log_accept: f64) -> f64 {
    (-log_accept.exp_m1()).ln()
}
fn delayed_selection_ratio(
    forward: f64,
    reverse: f64,
    accept_forward: f64,
    accept_reverse: f64,
) -> f64 {
    reverse - forward + log_reject(accept_reverse) - log_reject(accept_forward)
}
fn log_step_density(step: f64, mean: f64, sigma: f64) -> f64 {
    let z = (step.ln() - mean.ln() + 0.5 * sigma * sigma) / sigma;
    -0.5 * z * z - step.ln() - sigma.ln() - 0.918_938_533_204_672_7
}
fn energy<'a, T: alea_core::target::LogDensityGradient + ?Sized, M: EuclideanMetric>(
    phase: &PhaseState<'a, T>,
    metric: &M,
    workspace: &mut PhaseWorkspace<'a, T>,
) -> Result<f64, AtlasError<T::Error>> {
    Ok(workspace.start_from_phase(phase, metric)?.energy()?)
}
fn trace<'a, T: HessianVector + ?Sized, M: EuclideanMetric>(
    origin: &PhaseState<'a, T>,
    metric: &M,
    options: AtlasOptions,
    workspace: &mut PhaseWorkspace<'a, T>,
    history: &mut [PhaseState<'a, T>],
    work: &mut AtlasTransition,
) -> Result<usize, AtlasError<T::Error>> {
    let mut phase = workspace.start_from_phase(origin, metric)?;
    let initial = phase.energy()?;
    let mut distance = 0.;
    for (index, snapshot) in history.iter_mut().enumerate() {
        phase.save_into(snapshot)?;
        work.integration_steps = work
            .integration_steps
            .checked_add(1)
            .ok_or(AtlasError::CounterOverflow)?;
        phase = match phase
            .step(SignedStep::new(options.baseline.value()).expect("positive baseline"))
        {
            Ok(next) => next,
            Err(PhaseError::Divergence(_)) => return Ok(index + 1),
            Err(error) => return Err(error.into()),
        };
        let next_energy = match phase.energy() {
            Ok(value) => value,
            Err(PhaseError::Divergence(_)) => return Ok(index + 1),
            Err(error) => return Err(error.into()),
        };
        let next_distance = phase
            .point()
            .position()
            .iter()
            .zip(origin.point().position())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        if (initial - next_energy).abs() > 10.
            || !next_distance.is_finite()
            || next_distance <= distance
        {
            return Ok(index + 1);
        }
        distance = next_distance;
    }
    Ok(history.len())
}
fn fixed<'a, T: HessianVector + ?Sized, M: EuclideanMetric>(
    origin: &PhaseState<'a, T>,
    metric: &M,
    step: f64,
    count: usize,
    workspace: &mut PhaseWorkspace<'a, T>,
    destination: &mut PhaseState<'a, T>,
    work: &mut AtlasTransition,
) -> Result<Option<Divergence>, AtlasError<T::Error>> {
    let mut phase = workspace.start_from_phase(origin, metric)?;
    for _ in 0..count {
        work.integration_steps = work
            .integration_steps
            .checked_add(1)
            .ok_or(AtlasError::CounterOverflow)?;
        phase = match phase.step(SignedStep::new(step).expect("positive finite local step")) {
            Ok(next) => next,
            Err(PhaseError::Divergence(reason)) => return Ok(Some(reason)),
            Err(error) => return Err(error.into()),
        };
    }
    phase.save_into(destination)?;
    Ok(None)
}

/// Finite ATLAS trajectory learning after geometry/step warmup has finished.
#[derive(Debug)]
#[must_use = "consume warmup before retaining draws"]
pub struct AtlasWarmup<'a, T: HessianVector + ?Sized, M: EuclideanMetric> {
    chain: Atlas<'a, T, M>,
    iterations: usize,
}
/// Warmup work and the resulting empirical distribution size.
#[derive(Debug, Clone, Copy)]
pub struct AtlasWarmupReport {
    /// Actual forward-proposal macrosteps across warmup.
    pub forward_integration_steps: usize,
    /// Preliminary reverse-support macrosteps across warmup.
    pub reverse_integration_steps: usize,
    /// Reverse/ghost macrosteps across delayed warmup proposals.
    pub ghost_integration_steps: usize,
    /// Completed adaptation-only transitions.
    pub iterations: usize,
    /// All forward/reverse/ghost macrosteps during adaptation.
    pub integration_steps: usize,
    /// Actual adaptation HVP calls.
    pub hvp_calls: usize,
    /// Trimmed empirical trajectory observations.
    pub trajectory_observations: usize,
}
/// Frozen full ATLAS and finite trajectory learning costs.
pub type AtlasWarmupResult<'a, T, M> = Result<
    (Atlas<'a, T, M>, AtlasWarmupReport),
    AtlasError<<T as alea_core::target::LogDensityGradient>::Error>,
>;
impl<'a, T: HessianVector + ?Sized, M: EuclideanMetric> AtlasWarmup<'a, T, M> {
    /// Start from frozen NUTS/HMC geometry, with a provisional length8 (or budget).
    /// # Errors
    /// Same resource/shape failures as [`Atlas::from_hmc`].
    pub fn from_hmc(
        chain: Hmc<'a, T, M>,
        options: AtlasOptions,
        iterations: usize,
    ) -> Result<Self, AtlasError<T::Error>> {
        Ok(Self {
            chain: Atlas::from_hmc(chain, options, &[8.min(options.maximum.get())])?,
            iterations,
        })
    }
    /// Record preliminary NoUT lengths while moving, then freeze a10–90% trimmed
    /// empirical distribution. At most4096 most recent observations are retained.
    /// # Errors
    /// Propagates typed kernel failures and checked work-count overflow.
    pub fn run<R: Rng + ?Sized>(mut self, rng: &mut R) -> AtlasWarmupResult<'a, T, M> {
        let mut lengths = Vec::with_capacity(self.iterations.min(4096));
        let mut report = AtlasWarmupReport {
            forward_integration_steps: 0,
            reverse_integration_steps: 0,
            ghost_integration_steps: 0,
            iterations: 0,
            integration_steps: 0,
            hvp_calls: 0,
            trajectory_observations: 0,
        };
        for index in 0..self.iterations {
            let transition = self.chain.step(rng)?;
            report.iterations += 1;
            report.forward_integration_steps = report
                .forward_integration_steps
                .checked_add(transition.forward_integration_steps)
                .ok_or(AtlasError::CounterOverflow)?;
            report.reverse_integration_steps = report
                .reverse_integration_steps
                .checked_add(transition.reverse_integration_steps)
                .ok_or(AtlasError::CounterOverflow)?;
            report.ghost_integration_steps = report
                .ghost_integration_steps
                .checked_add(transition.ghost_integration_steps)
                .ok_or(AtlasError::CounterOverflow)?;
            report.integration_steps = report
                .integration_steps
                .checked_add(transition.integration_steps)
                .ok_or(AtlasError::CounterOverflow)?;
            report.hvp_calls = report
                .hvp_calls
                .checked_add(transition.hvp_calls)
                .ok_or(AtlasError::CounterOverflow)?;
            let length = transition
                .preliminary_length
                .clamp(3, self.chain.options.maximum.get());
            if lengths.len() < 4096 {
                lengths.push(length);
            } else {
                lengths[index % 4096] = length;
            }
        }
        if !lengths.is_empty() {
            lengths.sort_unstable();
            let trim = lengths.len() / 10;
            let end = lengths.len() - trim;
            self.chain.lengths = lengths[trim..end].into();
        }
        report.trajectory_observations = self.chain.lengths.len();
        Ok((self.chain, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delayed_density_ratio_balances_unequal_selection_and_rejection_masses() {
        let h_x = 2.;
        let h_y = 3.;
        let q_x = 0.1_f64;
        let q_y = 0.25_f64;
        let a_x = 0.3_f64;
        let a_y = 0.8_f64;
        let f_x = 0.7_f64;
        let f_y = 0.4_f64;
        let log_ratio = h_x - h_y
            + (f_y / f_x).ln()
            + delayed_selection_ratio(q_x.ln(), q_y.ln(), a_x.ln(), a_y.ln());
        let forward = (-h_x).exp() * q_x * (1. - a_x) * f_x * log_ratio.min(0.).exp();
        let reverse = (-h_y).exp() * q_y * (1. - a_y) * f_y * (-log_ratio).min(0.).exp();
        assert!((forward - reverse).abs() < 1e-16);
        assert_eq!(log_reject(0.), f64::NEG_INFINITY);
        assert!((log_reject(-1e-15) - 1e-15_f64.ln()).abs() < 1e-13);
    }
    struct Harmonic;
    impl alea_core::target::LogDensityGradient for Harmonic {
        type Error = std::convert::Infallible;
        fn dimension(&self) -> usize {
            1
        }
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
            g[0] = -q[0];
            Ok(-0.5 * q[0] * q[0])
        }
    }
    impl HessianVector for Harmonic {
        fn potential_hvp(&self, _: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Self::Error> {
            out[0] = v[0];
            Ok(())
        }
    }
    struct FixedChoices {
        offset: f64,
        index: f64,
        step: f64,
        uniforms: [f64; 2],
        used: usize,
    }
    impl ProposalChoices for FixedChoices {
        fn offset(&mut self) -> f64 {
            self.offset
        }
        fn index(&mut self, r: std::ops::Range<usize>) -> usize {
            r.start + (self.index * (r.end - r.start) as f64).floor() as usize
        }
        fn local_step(&mut self, _: f64, _: f64) -> f64 {
            self.step
        }
        fn uniform(&mut self) -> f64 {
            let u = self.uniforms[self.used];
            self.used += 1;
            u
        }
    }
    #[test]
    fn independent_harmonic_proposal_fixtures_cover_every_branch_and_support_boundary() {
        use alea_math::metric::IdentityMetric;
        let target = Harmonic;
        let mut stages = [0_usize; 3];
        let mut reasons = std::collections::BTreeSet::new();
        for (index, line) in include_str!("../tests/fixtures/atlas/harmonic-proposals.csv")
            .lines()
            .skip(1)
            .enumerate()
        {
            let f: Vec<_> = line.split(',').collect();
            let v = |i: usize| f[i].parse::<f64>().unwrap();
            let cap = v(3) as usize;
            let h = Hmc::new(
                &target,
                OwnedBuffer::from_fn(1, |_| v(0)),
                IdentityMetric::new(1),
                crate::HmcOptions::new(v(2), 1).unwrap(),
            )
            .unwrap();
            let mut chain = Atlas::from_hmc(
                h,
                AtlasOptions::new(v(2), cap).unwrap(),
                &[3, 4.min(cap), 5.min(cap)],
            )
            .unwrap();
            chain.origin =
                PhaseState::new(chain.point.clone(), OwnedBuffer::from_fn(1, |_| v(1))).unwrap();
            let mut choices = FixedChoices {
                offset: v(4),
                index: v(5),
                step: v(6),
                uniforms: [v(7), v(8)],
                used: 0,
            };
            let o = chain
                .propose(0.5 * (v(0) * v(0) + v(1) * v(1)), &mut choices)
                .unwrap();
            assert_eq!(format!("{:?}", o.stage), f[11], "case {index}");
            let reason = o
                .rejection
                .map_or_else(|| "None".to_owned(), |r| format!("{r:?}"));
            assert_eq!(reason, f[12], "case {index}");
            reasons.insert(reason);
            assert_eq!(o.accepted, v(10) == 1., "case {index}");
            assert!(
                (o.acceptance_probability - v(13)).abs() < 1e-10,
                "case {index}, actual={}, expected={}",
                o.acceptance_probability,
                v(13)
            );
            assert!(
                (chain.point.position()[0] - v(9)).abs() < 1e-10,
                "case {index}"
            );
            assert_eq!(
                [
                    o.forward_integration_steps,
                    o.reverse_integration_steps,
                    o.ghost_integration_steps,
                    o.hvp_calls
                ],
                [
                    v(14) as usize,
                    v(15) as usize,
                    v(16) as usize,
                    v(17) as usize
                ],
                "case {index}"
            );
            assert_eq!(
                o.integration_steps,
                o.forward_integration_steps
                    + o.reverse_integration_steps
                    + o.ghost_integration_steps
            );
            stages[match o.stage {
                AtlasStage::Preliminary => 0,
                AtlasStage::Delayed => 1,
                AtlasStage::Failure => 2,
            }] += 1;
        }
        assert!(stages.into_iter().all(|n| n > 0));
        for name in ["SubUTurn", "ReverseBranch", "WorkLimit", "Metropolis"] {
            assert!(reasons.contains(name));
        }
    }
}
