//! Opt-in M10 integration kernels with explicit capability and cost contracts.
//!
//! These kernels intentionally do not implement NUTS and are not selected by
//! default warmup. Each transition is Metropolis corrected and publishes only a
//! complete endpoint; model failures and numerical divergences never mutate the
//! live cached point.

use std::{error::Error, num::NonZeroUsize};

use alea_core::{
    capability::{
        GaussianResidual, HessianVector, HvpError, SplitForceError, SplitLogDensityGradient,
        evaluate_cheap_score, evaluate_expensive_score, evaluate_hvp, evaluate_residual_score,
    },
    target::{EvaluationError, EvaluationWorkspace, LogDensityGradient, PointState},
};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric, MetricError},
};
use rand::{Rng, RngExt};

use crate::{
    MarkovChain,
    canonical::metropolis,
    hamiltonian::Divergence,
    hmc::{HmcConfigError, StepSize},
};

/// Validated fixed-trajectory settings shared by advanced integration kernels.
#[derive(Debug, Clone, Copy)]
pub struct AdvancedHmcOptions {
    step_size: StepSize,
    integration_steps: NonZeroUsize,
    max_energy_error: f64,
}

impl AdvancedHmcOptions {
    pub fn new(step_size: f64, integration_steps: usize) -> Result<Self, HmcConfigError> {
        Ok(Self {
            step_size: step_size.try_into()?,
            integration_steps: NonZeroUsize::new(integration_steps)
                .ok_or(HmcConfigError::IntegrationCount)?,
            max_energy_error: 1000.0,
        })
    }

    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, HmcConfigError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(HmcConfigError::EnergyErrorLimit);
        }
        self.max_energy_error = limit;
        Ok(self)
    }

    pub fn step_size(self) -> StepSize {
        self.step_size
    }

    pub fn integration_steps(self) -> NonZeroUsize {
        self.integration_steps
    }
}

/// Exact derivative/evaluation work performed by one attempted transition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdvancedEvaluationCounters {
    pub full: usize,
    pub cheap: usize,
    pub expensive: usize,
    pub residual: usize,
    pub hvp: usize,
}

/// Common Metropolis outcome for the advanced kernels.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct AdvancedHmcTransition {
    pub accepted: bool,
    pub acceptance_probability: f64,
    pub integration_steps: usize,
    pub initial_energy: Option<f64>,
    pub proposal_energy: Option<f64>,
    pub energy_error: Option<f64>,
    pub divergence: Option<Divergence>,
    pub evaluations: AdvancedEvaluationCounters,
}

impl AdvancedHmcTransition {
    fn new() -> Self {
        Self {
            accepted: false,
            acceptance_probability: 0.0,
            integration_steps: 0,
            initial_energy: None,
            proposal_energy: None,
            energy_error: None,
            divergence: None,
            evaluations: AdvancedEvaluationCounters::default(),
        }
    }
}

/// Construction, metric, capability or backend failure. Ordinary floating-point
/// trajectory failures are reported as rejected transitions instead.
#[derive(Debug, thiserror::Error)]
pub enum AdvancedHmcError<E: Error + 'static> {
    #[error("advanced HMC requires a nonempty target")]
    EmptyTarget,
    #[error("metric dimension {metric} does not match target dimension {target}")]
    MetricDimension { target: usize, metric: usize },
    #[error(transparent)]
    Metric(#[from] MetricError),
    #[error(transparent)]
    Evaluation(#[from] EvaluationError<E>),
    #[error(transparent)]
    SplitForce(#[from] SplitForceError<E>),
    #[error(transparent)]
    Hvp(#[from] HvpError<E>),
    #[error("Gaussian reference center/precision is malformed")]
    GaussianReference,
    #[error("inner cheap-force step count must be nonzero")]
    InnerSteps,
}

fn finite(values: &[f64]) -> Result<(), Divergence> {
    values
        .iter()
        .position(|value| !value.is_finite())
        .map_or(Ok(()), |index| Err(Divergence::Position { index }))
}

fn momentum_finite(values: &[f64]) -> Result<(), Divergence> {
    values
        .iter()
        .position(|value| !value.is_finite())
        .map_or(Ok(()), |index| Err(Divergence::Momentum { index }))
}

fn reject_numerical<E: Error + 'static>(
    transition: &mut AdvancedHmcTransition,
    error: AdvancedHmcError<E>,
) -> Result<(), AdvancedHmcError<E>> {
    let divergence = match error {
        AdvancedHmcError::Evaluation(error) => match error {
            EvaluationError::NonFinitePosition { index } => Divergence::Position { index },
            EvaluationError::NonFiniteLogDensity => Divergence::LogDensity,
            EvaluationError::NonFiniteGradient { index } => Divergence::Gradient { index },
            other => return Err(AdvancedHmcError::Evaluation(other)),
        },
        AdvancedHmcError::SplitForce(error) => match error {
            SplitForceError::NonFiniteOutput { index, .. } => Divergence::Gradient { index },
            SplitForceError::Evaluation(EvaluationError::NonFinitePosition { index }) => {
                Divergence::Position { index }
            }
            SplitForceError::Evaluation(EvaluationError::NonFiniteLogDensity) => {
                Divergence::LogDensity
            }
            SplitForceError::Evaluation(EvaluationError::NonFiniteGradient { index }) => {
                Divergence::Gradient { index }
            }
            other => return Err(AdvancedHmcError::SplitForce(other)),
        },
        AdvancedHmcError::Hvp(error) => match error {
            HvpError::NonFiniteVector { index } | HvpError::NonFiniteOutput { index } => {
                Divergence::Gradient { index }
            }
            HvpError::Evaluation(EvaluationError::NonFinitePosition { index }) => {
                Divergence::Position { index }
            }
            HvpError::Evaluation(EvaluationError::NonFiniteLogDensity) => Divergence::LogDensity,
            HvpError::Evaluation(EvaluationError::NonFiniteGradient { index }) => {
                Divergence::Gradient { index }
            }
            other => return Err(AdvancedHmcError::Hvp(other)),
        },
        other => return Err(other),
    };
    transition.divergence = Some(divergence);
    Ok(())
}

fn exact_diagonal_gaussian_flow(
    center: &[f64],
    precision: &[f64],
    position: &mut [f64],
    momentum: &mut [f64],
    time: f64,
) {
    debug_assert_eq!(center.len(), position.len());
    debug_assert_eq!(precision.len(), position.len());
    debug_assert_eq!(momentum.len(), position.len());
    for index in 0..position.len() {
        let omega = precision[index].sqrt();
        let angle = omega * time;
        let (sin, cos) = angle.sin_cos();
        let displacement = position[index] - center[index];
        let old_momentum = momentum[index];
        position[index] = center[index] + cos * displacement + sin * old_momentum / omega;
        momentum[index] = cos * old_momentum - omega * sin * displacement;
    }
}

fn kinetic<M: EuclideanMetric>(
    metric: &M,
    momentum: &[f64],
    velocity: &mut [f64],
) -> Result<f64, MetricError> {
    metric.velocity(momentum, velocity)?;
    Ok(0.5
        * momentum
            .iter()
            .zip(velocity.iter())
            .map(|(p, v)| p * v)
            .sum::<f64>())
}

fn finish_transition<R: Rng + ?Sized>(
    transition: &mut AdvancedHmcTransition,
    initial: f64,
    proposed: f64,
    max_energy_error: f64,
    rng: &mut R,
) {
    transition.proposal_energy = Some(proposed);
    let error = proposed - initial;
    if !proposed.is_finite() || !error.is_finite() {
        transition.divergence = Some(Divergence::Energy);
        return;
    }
    transition.energy_error = Some(error);
    match metropolis(initial, proposed, max_energy_error) {
        Ok(probability) => {
            transition.acceptance_probability = probability;
            transition.accepted = rng.random::<f64>() < probability;
        }
        Err(reason) => transition.divergence = Some(reason),
    }
}

#[derive(Debug)]
struct Workspace<'a, T: LogDensityGradient + ?Sized> {
    proposal: PointState<'a, T>,
    evaluation: EvaluationWorkspace,
    position: OwnedBuffer,
    momentum: OwnedBuffer,
    velocity: OwnedBuffer,
    first: OwnedBuffer,
    second: OwnedBuffer,
}

impl<'a, T: LogDensityGradient + ?Sized> Workspace<'a, T> {
    fn new(point: &PointState<'a, T>) -> Self {
        let dimension = point.dimension();
        Self {
            proposal: point.clone(),
            evaluation: EvaluationWorkspace::new(dimension),
            position: OwnedBuffer::new(dimension),
            momentum: OwnedBuffer::new(dimension),
            velocity: OwnedBuffer::new(dimension),
            first: OwnedBuffer::new(dimension),
            second: OwnedBuffer::new(dimension),
        }
    }

    fn prepare<M: EuclideanMetric, R: Rng + ?Sized>(
        &mut self,
        live: &PointState<'a, T>,
        metric: &M,
        rng: &mut R,
    ) -> Result<(), AdvancedHmcError<T::Error>> {
        self.proposal.clone_from(live);
        self.position.copy_from_slice(live.position());
        for value in self.velocity.iter_mut() {
            *value = alea_math::random::standard_normal(rng);
        }
        metric.sample_momentum(&self.velocity, &mut self.momentum)?;
        Ok(())
    }
}

/// Nested cheap/expensive-force HMC. One trajectory performs `L + 1`
/// expensive evaluations and `L * (inner + 1)` cheap evaluations; endpoint
/// density evaluation is counted separately.
#[derive(Debug)]
pub struct SplitForceHmc<'a, T: SplitLogDensityGradient + ?Sized, M> {
    point: PointState<'a, T>,
    workspace: Workspace<'a, T>,
    metric: M,
    options: AdvancedHmcOptions,
    inner_steps: NonZeroUsize,
}

impl<'a, T: SplitLogDensityGradient + ?Sized, M: EuclideanMetric> SplitForceHmc<'a, T, M> {
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: AdvancedHmcOptions,
        inner_steps: usize,
    ) -> Result<Self, AdvancedHmcError<T::Error>> {
        if target.dimension() == 0 {
            return Err(AdvancedHmcError::EmptyTarget);
        }
        if metric.dimension() != target.dimension() {
            return Err(AdvancedHmcError::MetricDimension {
                target: target.dimension(),
                metric: metric.dimension(),
            });
        }
        let inner_steps = NonZeroUsize::new(inner_steps).ok_or(AdvancedHmcError::InnerSteps)?;
        let point = PointState::new(target, position)?;
        let workspace = Workspace::new(&point);
        Ok(Self {
            point,
            workspace,
            metric,
            options,
            inner_steps,
        })
    }

    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }

    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<AdvancedHmcTransition, AdvancedHmcError<T::Error>> {
        let mut transition = AdvancedHmcTransition::new();
        self.workspace.prepare(&self.point, &self.metric, rng)?;
        let initial = -self.point.log_density()
            + kinetic(
                &self.metric,
                &self.workspace.momentum,
                &mut self.workspace.velocity,
            )?;
        transition.initial_energy = Some(initial);
        if !initial.is_finite() {
            transition.divergence = Some(Divergence::Energy);
            return Ok(transition);
        }
        let h = self.options.step_size.value();
        let inner = self.inner_steps.get();
        let delta = h / inner as f64;
        if let Err(error) = evaluate_expensive_score(
            self.point.target(),
            &self.workspace.position,
            &mut self.workspace.second,
        ) {
            reject_numerical(&mut transition, AdvancedHmcError::SplitForce(error))?;
            return Ok(transition);
        }
        transition.evaluations.expensive += 1;
        for (momentum, score) in self
            .workspace
            .momentum
            .iter_mut()
            .zip(self.workspace.second.iter())
        {
            *momentum += 0.5 * h * score;
        }
        for outer in 0..self.options.integration_steps.get() {
            if let Err(error) = evaluate_cheap_score(
                self.point.target(),
                &self.workspace.position,
                &mut self.workspace.first,
            ) {
                reject_numerical(&mut transition, AdvancedHmcError::SplitForce(error))?;
                return Ok(transition);
            }
            transition.evaluations.cheap += 1;
            for (momentum, score) in self
                .workspace
                .momentum
                .iter_mut()
                .zip(self.workspace.first.iter())
            {
                *momentum += 0.5 * delta * score;
            }
            for inner_index in 0..inner {
                self.metric
                    .velocity(&self.workspace.momentum, &mut self.workspace.velocity)?;
                for ((position, velocity), next) in self
                    .workspace
                    .position
                    .iter()
                    .zip(self.workspace.velocity.iter())
                    .zip(self.workspace.first.iter_mut())
                {
                    *next = *position + delta * velocity;
                }
                if let Err(reason) = finite(&self.workspace.first) {
                    transition.divergence = Some(reason);
                    return Ok(transition);
                }
                self.workspace
                    .position
                    .copy_from_slice(&self.workspace.first);
                if let Err(error) = evaluate_cheap_score(
                    self.point.target(),
                    &self.workspace.position,
                    &mut self.workspace.first,
                ) {
                    reject_numerical(&mut transition, AdvancedHmcError::SplitForce(error))?;
                    return Ok(transition);
                }
                transition.evaluations.cheap += 1;
                let kick = if inner_index + 1 == inner { 0.5 } else { 1.0 };
                for (momentum, score) in self
                    .workspace
                    .momentum
                    .iter_mut()
                    .zip(self.workspace.first.iter())
                {
                    *momentum += kick * delta * score;
                }
            }
            if let Err(error) = evaluate_expensive_score(
                self.point.target(),
                &self.workspace.position,
                &mut self.workspace.second,
            ) {
                reject_numerical(&mut transition, AdvancedHmcError::SplitForce(error))?;
                return Ok(transition);
            }
            transition.evaluations.expensive += 1;
            let kick = if outer + 1 == self.options.integration_steps.get() {
                0.5
            } else {
                1.0
            };
            for (momentum, score) in self
                .workspace
                .momentum
                .iter_mut()
                .zip(self.workspace.second.iter())
            {
                *momentum += kick * h * score;
            }
            transition.integration_steps = outer + 1;
        }
        if let Err(reason) = momentum_finite(&self.workspace.momentum) {
            transition.divergence = Some(reason);
            return Ok(transition);
        }
        if let Err(error) = self
            .workspace
            .proposal
            .try_update(&self.workspace.position, &mut self.workspace.evaluation)
        {
            reject_numerical(&mut transition, AdvancedHmcError::Evaluation(error))?;
            return Ok(transition);
        }
        transition.evaluations.full = 1;
        let proposed = -self.workspace.proposal.log_density()
            + kinetic(
                &self.metric,
                &self.workspace.momentum,
                &mut self.workspace.velocity,
            )?;
        finish_transition(
            &mut transition,
            initial,
            proposed,
            self.options.max_energy_error,
            rng,
        );
        if transition.accepted {
            std::mem::swap(&mut self.point, &mut self.workspace.proposal);
        }
        Ok(transition)
    }
}

impl<T: SplitLogDensityGradient + ?Sized, M: EuclideanMetric> MarkovChain
    for SplitForceHmc<'_, T, M>
{
    type Error = AdvancedHmcError<T::Error>;
    type Transition = AdvancedHmcTransition;
    fn position(&self) -> &[f64] {
        self.point.position()
    }
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Self::step(self, rng)
    }
}

/// HMC with a residual half-kick around the exact diagonal-Gaussian flow.
#[derive(Debug)]
pub struct GaussianSplitHmc<'a, T: GaussianResidual + ?Sized> {
    point: PointState<'a, T>,
    workspace: Workspace<'a, T>,
    metric: IdentityMetric,
    options: AdvancedHmcOptions,
}

impl<'a, T: GaussianResidual + ?Sized> GaussianSplitHmc<'a, T> {
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        options: AdvancedHmcOptions,
    ) -> Result<Self, AdvancedHmcError<T::Error>> {
        let dimension = target.dimension();
        if dimension == 0 {
            return Err(AdvancedHmcError::EmptyTarget);
        }
        if target.gaussian_center().len() != dimension
            || target.gaussian_precision().len() != dimension
            || target
                .gaussian_center()
                .iter()
                .any(|value| !value.is_finite())
            || target
                .gaussian_precision()
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(AdvancedHmcError::GaussianReference);
        }
        let point = PointState::new(target, position)?;
        let workspace = Workspace::new(&point);
        Ok(Self {
            point,
            workspace,
            metric: IdentityMetric::new(dimension),
            options,
        })
    }

    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }

    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<AdvancedHmcTransition, AdvancedHmcError<T::Error>> {
        let mut transition = AdvancedHmcTransition::new();
        self.workspace.prepare(&self.point, &self.metric, rng)?;
        let initial = -self.point.log_density()
            + kinetic(
                &self.metric,
                &self.workspace.momentum,
                &mut self.workspace.velocity,
            )?;
        transition.initial_energy = Some(initial);
        if !initial.is_finite() {
            transition.divergence = Some(Divergence::Energy);
            return Ok(transition);
        }
        let h = self.options.step_size.value();
        if let Err(error) = evaluate_residual_score(
            self.point.target(),
            &self.workspace.position,
            &mut self.workspace.first,
        ) {
            reject_numerical(&mut transition, AdvancedHmcError::SplitForce(error))?;
            return Ok(transition);
        }
        transition.evaluations.residual = 1;
        for (momentum, score) in self
            .workspace
            .momentum
            .iter_mut()
            .zip(self.workspace.first.iter())
        {
            *momentum += 0.5 * h * score;
        }
        for step in 0..self.options.integration_steps.get() {
            exact_diagonal_gaussian_flow(
                self.point.target().gaussian_center(),
                self.point.target().gaussian_precision(),
                &mut self.workspace.position,
                &mut self.workspace.momentum,
                h,
            );
            if let Err(reason) = finite(&self.workspace.position) {
                transition.divergence = Some(reason);
                return Ok(transition);
            }
            if let Err(error) = evaluate_residual_score(
                self.point.target(),
                &self.workspace.position,
                &mut self.workspace.first,
            ) {
                reject_numerical(&mut transition, AdvancedHmcError::SplitForce(error))?;
                return Ok(transition);
            }
            transition.evaluations.residual += 1;
            let kick = if step + 1 == self.options.integration_steps.get() {
                0.5
            } else {
                1.0
            };
            for (momentum, score) in self
                .workspace
                .momentum
                .iter_mut()
                .zip(self.workspace.first.iter())
            {
                *momentum += kick * h * score;
            }
            transition.integration_steps = step + 1;
        }
        if let Err(reason) = momentum_finite(&self.workspace.momentum) {
            transition.divergence = Some(reason);
            return Ok(transition);
        }
        if let Err(error) = self
            .workspace
            .proposal
            .try_update(&self.workspace.position, &mut self.workspace.evaluation)
        {
            reject_numerical(&mut transition, AdvancedHmcError::Evaluation(error))?;
            return Ok(transition);
        }
        transition.evaluations.full = 1;
        let proposed = -self.workspace.proposal.log_density()
            + kinetic(
                &self.metric,
                &self.workspace.momentum,
                &mut self.workspace.velocity,
            )?;
        finish_transition(
            &mut transition,
            initial,
            proposed,
            self.options.max_energy_error,
            rng,
        );
        if transition.accepted {
            std::mem::swap(&mut self.point, &mut self.workspace.proposal);
        }
        Ok(transition)
    }
}

impl<T: GaussianResidual + ?Sized> MarkovChain for GaussianSplitHmc<'_, T> {
    type Error = AdvancedHmcError<T::Error>;
    type Transition = AdvancedHmcTransition;
    fn position(&self) -> &[f64] {
        self.point.position()
    }
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Self::step(self, rng)
    }
}

/// Analytic HVP or reversible Hessian-free realization of the five-stage
/// fourth-order force-gradient composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceGradientMode {
    AnalyticHvp,
    HessianFree,
}

/// Metropolis-corrected fourth-order force-gradient HMC.
#[derive(Debug)]
pub struct ForceGradientHmc<'a, T: HessianVector + ?Sized, M> {
    point: PointState<'a, T>,
    workspace: Workspace<'a, T>,
    metric: M,
    options: AdvancedHmcOptions,
    mode: ForceGradientMode,
}

impl<'a, T: HessianVector + ?Sized, M: EuclideanMetric> ForceGradientHmc<'a, T, M> {
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: AdvancedHmcOptions,
        mode: ForceGradientMode,
    ) -> Result<Self, AdvancedHmcError<T::Error>> {
        if target.dimension() == 0 {
            return Err(AdvancedHmcError::EmptyTarget);
        }
        if metric.dimension() != target.dimension() {
            return Err(AdvancedHmcError::MetricDimension {
                target: target.dimension(),
                metric: metric.dimension(),
            });
        }
        let point = PointState::new(target, position)?;
        let workspace = Workspace::new(&point);
        Ok(Self {
            point,
            workspace,
            metric,
            options,
            mode,
        })
    }

    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }

    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<AdvancedHmcTransition, AdvancedHmcError<T::Error>> {
        let mut transition = AdvancedHmcTransition::new();
        self.workspace.prepare(&self.point, &self.metric, rng)?;
        let initial = -self.point.log_density()
            + kinetic(
                &self.metric,
                &self.workspace.momentum,
                &mut self.workspace.velocity,
            )?;
        transition.initial_energy = Some(initial);
        if !initial.is_finite() {
            transition.divergence = Some(Divergence::Energy);
            return Ok(transition);
        }
        let h = self.options.step_size.value();
        for step in 0..self.options.integration_steps.get() {
            if let Err(error) = self.macrostep(h, &mut transition.evaluations) {
                reject_numerical(&mut transition, error)?;
                return Ok(transition);
            }
            if let Err(reason) = momentum_finite(&self.workspace.momentum) {
                transition.divergence = Some(reason);
                return Ok(transition);
            }
            transition.integration_steps = step + 1;
        }
        let proposed = -self.workspace.proposal.log_density()
            + kinetic(
                &self.metric,
                &self.workspace.momentum,
                &mut self.workspace.velocity,
            )?;
        finish_transition(
            &mut transition,
            initial,
            proposed,
            self.options.max_energy_error,
            rng,
        );
        if transition.accepted {
            std::mem::swap(&mut self.point, &mut self.workspace.proposal);
        }
        Ok(transition)
    }

    fn macrostep(
        &mut self,
        h: f64,
        counters: &mut AdvancedEvaluationCounters,
    ) -> Result<(), AdvancedHmcError<T::Error>> {
        for (momentum, score) in self
            .workspace
            .momentum
            .iter_mut()
            .zip(self.workspace.proposal.gradient())
        {
            *momentum += h * score / 6.0;
        }
        self.drift_and_evaluate(0.5 * h)?;
        counters.full += 1;
        match self.mode {
            ForceGradientMode::AnalyticHvp => {
                self.metric.velocity(
                    self.workspace.proposal.gradient(),
                    &mut self.workspace.velocity,
                )?;
                evaluate_hvp(
                    self.point.target(),
                    self.workspace.proposal.position(),
                    &self.workspace.velocity,
                    &mut self.workspace.second,
                )?;
                counters.hvp += 1;
                for ((momentum, score), hvp) in self
                    .workspace
                    .momentum
                    .iter_mut()
                    .zip(self.workspace.proposal.gradient())
                    .zip(self.workspace.second.iter())
                {
                    *momentum += (2.0 * h / 3.0) * score - (h * h * h / 36.0) * hvp;
                }
            }
            ForceGradientMode::HessianFree => {
                self.metric.velocity(
                    self.workspace.proposal.gradient(),
                    &mut self.workspace.velocity,
                )?;
                for ((temporary, position), velocity) in self
                    .workspace
                    .position
                    .iter_mut()
                    .zip(self.workspace.proposal.position())
                    .zip(self.workspace.velocity.iter())
                {
                    *temporary = *position + (h * h / 24.0) * velocity;
                }
                alea_core::target::evaluate(
                    self.point.target(),
                    &self.workspace.position,
                    &mut self.workspace.second,
                )?;
                counters.full += 1;
                for (momentum, score) in self
                    .workspace
                    .momentum
                    .iter_mut()
                    .zip(self.workspace.second.iter())
                {
                    *momentum += (2.0 * h / 3.0) * score;
                }
            }
        }
        self.drift_and_evaluate(0.5 * h)?;
        counters.full += 1;
        for (momentum, score) in self
            .workspace
            .momentum
            .iter_mut()
            .zip(self.workspace.proposal.gradient())
        {
            *momentum += h * score / 6.0;
        }
        Ok(())
    }

    fn drift_and_evaluate(&mut self, scale: f64) -> Result<(), AdvancedHmcError<T::Error>> {
        self.metric
            .velocity(&self.workspace.momentum, &mut self.workspace.velocity)?;
        for ((next, position), velocity) in self
            .workspace
            .position
            .iter_mut()
            .zip(self.workspace.proposal.position())
            .zip(self.workspace.velocity.iter())
        {
            *next = *position + scale * velocity;
        }
        self.workspace
            .proposal
            .try_update(&self.workspace.position, &mut self.workspace.evaluation)?;
        Ok(())
    }
}

impl<T: HessianVector + ?Sized, M: EuclideanMetric> MarkovChain for ForceGradientHmc<'_, T, M> {
    type Error = AdvancedHmcError<T::Error>;
    type Transition = AdvancedHmcTransition;
    fn position(&self) -> &[f64] {
        self.point.position()
    }
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Self::step(self, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    struct Normal;
    impl LogDensityGradient for Normal {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            1
        }
        fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
            gradient[0] = -q[0];
            Ok(-0.5 * q[0] * q[0])
        }
    }
    impl HessianVector for Normal {
        fn potential_hvp(
            &self,
            _: &[f64],
            vector: &[f64],
            output: &mut [f64],
        ) -> Result<(), Self::Error> {
            output.copy_from_slice(vector);
            Ok(())
        }
    }

    fn force_gradient_map(mode: ForceGradientMode, q: f64, p: f64, h: f64) -> (f64, f64) {
        let target = Normal;
        let mut chain = ForceGradientHmc::new(
            &target,
            OwnedBuffer::from_fn(1, |_| q),
            IdentityMetric::new(1),
            AdvancedHmcOptions::new(h.abs(), 1).unwrap(),
            mode,
        )
        .unwrap();
        chain.workspace.momentum[0] = p;
        let mut counters = AdvancedEvaluationCounters::default();
        chain.macrostep(h, &mut counters).unwrap();
        (
            chain.workspace.proposal.position()[0],
            chain.workspace.momentum[0],
        )
    }

    #[test]
    fn exact_gaussian_flow_matches_closed_orbit_and_reverses() {
        let center: [f64; 2] = [0.3, -0.2];
        let precision: [f64; 2] = [1.0, 9.0];
        let mut q: [f64; 2] = [0.7, -0.8];
        let mut p: [f64; 2] = [-0.4, 0.5];
        let original_q = q;
        let original_p = p;
        let initial = 0.5
            * (precision[0] * (q[0] - center[0]).powi(2)
                + precision[1] * (q[1] - center[1]).powi(2)
                + p[0] * p[0]
                + p[1] * p[1]);
        exact_diagonal_gaussian_flow(&center, &precision, &mut q, &mut p, 0.37);
        let final_energy = 0.5
            * (precision[0] * (q[0] - center[0]).powi(2)
                + precision[1] * (q[1] - center[1]).powi(2)
                + p[0] * p[0]
                + p[1] * p[1]);
        assert!((final_energy - initial).abs() < 5e-16);
        exact_diagonal_gaussian_flow(&center, &precision, &mut q, &mut p, -0.37);
        for (actual, expected) in q.iter().zip(original_q) {
            assert!((actual - expected).abs() < 2e-16);
        }
        for (actual, expected) in p.iter().zip(original_p) {
            assert!((actual - expected).abs() < 2e-16);
        }
    }

    #[test]
    fn force_gradient_variants_reverse_and_preserve_unit_volume() {
        for mode in [
            ForceGradientMode::AnalyticHvp,
            ForceGradientMode::HessianFree,
        ] {
            let (q, p) = force_gradient_map(mode, 0.7, -0.4, 0.19);
            let (reversed_q, reversed_p) = force_gradient_map(mode, q, p, -0.19);
            assert!((reversed_q - 0.7).abs() < 2e-14);
            assert!((reversed_p + 0.4).abs() < 2e-14);

            let delta = 1e-6;
            let (q_plus, p_plus) = force_gradient_map(mode, 0.7 + delta, -0.4, 0.19);
            let (q_minus, p_minus) = force_gradient_map(mode, 0.7 - delta, -0.4, 0.19);
            let dq_dq = (q_plus - q_minus) / (2.0 * delta);
            let dp_dq = (p_plus - p_minus) / (2.0 * delta);
            let (q_plus, p_plus) = force_gradient_map(mode, 0.7, -0.4 + delta, 0.19);
            let (q_minus, p_minus) = force_gradient_map(mode, 0.7, -0.4 - delta, 0.19);
            let dq_dp = (q_plus - q_minus) / (2.0 * delta);
            let dp_dp = (p_plus - p_minus) / (2.0 * delta);
            let determinant = dq_dq * dp_dp - dq_dp * dp_dq;
            assert!((determinant - 1.0).abs() < 2e-9, "{mode:?}: {determinant}");
        }
    }
}
