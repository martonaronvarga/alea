//! Exactness-pending Riemannian HMC. This kernel is not validated exact-target inference.
//!
//! The non-separable Hamiltonian is integrated by the generalized leapfrog.
//! Implicit stages run a fixed, bounded number of iterations, so forward work is
//! deterministic. This does not by itself guarantee that a finite-tolerance
//! reverse solve selects the same implicit branch. Failed forward solves reject
//! the proposal and publish an explicit event; this research sampler does not yet
//! enforce a per-proposal involution check by default. The opt-in Newton/reverse
//! profile checks every reversed phase and half-momentum; finite solver volume
//! preservation remains unresolved and the type remains exactness-pending.

mod fisher;
mod softabs;

use std::{error::Error, num::NonZeroUsize};

use alea_core::{
    capability::{PositionMetric, PositionMetricError, evaluate_position_metric},
    target::{EvaluationError, EvaluationWorkspace, LogDensityGradient, PointState},
};
use alea_math::buffer::OwnedBuffer;
use rand::{Rng, RngExt};

use crate::{MarkovChain, canonical::metropolis, config::StepSize, hamiltonian::Divergence};

pub use fisher::{FisherMetricConfigError, HierarchicalNormalFisherMetric, LogisticFisherMetric};
pub use softabs::{SoftAbsConfigError, SoftAbsError, SoftAbsMetric, SoftAbsOptions};

/// Which implicit equation failed its fixed-point postcondition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplicitStage {
    Momentum,
    Position,
}

/// Explicit bounded-solve failure emitted by a rejected RMHMC transition.
#[derive(Debug, Clone, Copy)]
pub struct ImplicitSolveEvent {
    pub integration_step: usize,
    pub stage: ImplicitStage,
    pub iterations: usize,
    pub scaled_residual: f64,
    pub tolerance: f64,
}

/// Why an RMHMC proposal was rejected before the Metropolis decision.
#[derive(Debug, Clone, Copy)]
pub enum RmhmcTermination {
    Divergence(Divergence),
    ImplicitSolve(ImplicitSolveEvent),
    /// Reverse replay failed at a boundary or intermediate half-momentum.
    ReverseCheck {
        integration_step: usize,
        scaled_error: f64,
        tolerance: f64,
    },
}

/// Exact operation counts for one attempted RMHMC transition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RmhmcCosts {
    pub target_evaluations: usize,
    pub metric_derivative_evaluations: usize,
    pub factorizations: usize,
    pub triangular_solves: usize,
    pub inverse_actions: usize,
    pub dynamics_steps: usize,
    pub momentum_iterations: usize,
    pub position_iterations: usize,
    /// Dense Newton residual Jacobians formed, including reverse work.
    pub newton_jacobians: usize,
    /// Pivoted dense linear systems solved by Newton.
    pub newton_linear_solves: usize,
    /// Completed replayed macrosteps.
    pub reverse_steps: usize,
}

/// Explicit bounded implicit solve policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImplicitSolver {
    /// Historical fixed-count Picard iteration.
    #[default]
    FixedPoint,
    /// Newton solves with analytic residual Jacobians and actual equation checks.
    Newton,
}

/// Validated generalized-leapfrog controls.
#[derive(Debug, Clone, Copy)]
pub struct RmhmcOptions {
    step_size: StepSize,
    integration_steps: NonZeroUsize,
    implicit_iterations: NonZeroUsize,
    implicit_tolerance: f64,
    max_energy_error: f64,
    solver: ImplicitSolver,
    reverse_tolerance: Option<f64>,
}

impl RmhmcOptions {
    /// Constructs fixed trajectory and fixed-point budgets.
    ///
    /// # Errors
    /// Rejects zero/non-finite controls and non-positive tolerances.
    pub fn new(
        step_size: f64,
        integration_steps: usize,
        implicit_iterations: usize,
        implicit_tolerance: f64,
    ) -> Result<Self, RmhmcConfigError> {
        let step_size = StepSize::try_from(step_size).map_err(|_| RmhmcConfigError::StepSize)?;
        let integration_steps =
            NonZeroUsize::new(integration_steps).ok_or(RmhmcConfigError::IntegrationSteps)?;
        let implicit_iterations =
            NonZeroUsize::new(implicit_iterations).ok_or(RmhmcConfigError::ImplicitIterations)?;
        if !implicit_tolerance.is_finite() || implicit_tolerance <= 0.0 {
            return Err(RmhmcConfigError::ImplicitTolerance);
        }
        Ok(Self {
            step_size,
            integration_steps,
            implicit_iterations,
            implicit_tolerance,
            max_energy_error: 1000.0,
            solver: ImplicitSolver::FixedPoint,
            reverse_tolerance: None,
        })
    }

    /// Select Newton with the supplied iteration budget/tolerance from `new`.
    /// This alone does not imply reversibility or volume preservation.
    pub fn with_newton_solver(mut self) -> Self {
        self.solver = ImplicitSolver::Newton;
        self
    }
    /// Require full-proposal reverse replay and intermediate half-momentum checks.
    /// # Errors
    /// Requires finite positive tolerance and at most1024 macrosteps of history.
    pub fn with_reverse_check(mut self, tolerance: f64) -> Result<Self, RmhmcConfigError> {
        if !tolerance.is_finite() || tolerance <= 0. || self.integration_steps.get() > 1024 {
            return Err(RmhmcConfigError::ReverseCheck);
        }
        self.reverse_tolerance = Some(tolerance);
        Ok(self)
    }
    /// Frozen implicit solve algorithm.
    pub fn implicit_solver(self) -> ImplicitSolver {
        self.solver
    }
    /// Explicit reverse-support tolerance, when enabled.
    pub fn reverse_tolerance(self) -> Option<f64> {
        self.reverse_tolerance
    }

    /// Sets the finite absolute Hamiltonian-error rejection threshold.
    ///
    /// # Errors
    /// Rejects non-finite or non-positive thresholds.
    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, RmhmcConfigError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(RmhmcConfigError::EnergyErrorLimit);
        }
        self.max_energy_error = limit;
        Ok(self)
    }

    pub fn step_size(self) -> f64 {
        self.step_size.value()
    }

    pub fn integration_steps(self) -> NonZeroUsize {
        self.integration_steps
    }

    pub fn implicit_iterations(self) -> NonZeroUsize {
        self.implicit_iterations
    }

    pub fn implicit_tolerance(self) -> f64 {
        self.implicit_tolerance
    }
}

/// RMHMC configuration error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RmhmcConfigError {
    #[error("RMHMC step size must be finite and positive")]
    StepSize,
    #[error("RMHMC integration step count must be nonzero")]
    IntegrationSteps,
    #[error("RMHMC implicit iteration count must be nonzero")]
    ImplicitIterations,
    #[error("RMHMC implicit tolerance must be finite and positive")]
    ImplicitTolerance,
    #[error("RMHMC energy-error limit must be finite and positive")]
    EnergyErrorLimit,
    #[error("RMHMC reverse check requires positive finite tolerance and at most1024 macrosteps")]
    ReverseCheck,
}

/// Construction or backend failure. Numerical integration failures are rejected
/// transitions and appear in [`RmhmcTransition::termination`].
#[derive(Debug, thiserror::Error)]
pub enum RmhmcError<TE: Error + 'static, ME: Error + 'static> {
    #[error("RMHMC requires a nonempty target")]
    EmptyTarget,
    #[error("target dimension {target} differs from metric dimension {metric}")]
    Dimension { target: usize, metric: usize },
    #[error("RMHMC storage size overflow")]
    SizeOverflow,
    #[error(transparent)]
    Target(#[from] EvaluationError<TE>),
    #[error(transparent)]
    Metric(#[from] PositionMetricError<ME>),
    #[error("position metric is not positive definite at pivot {pivot}")]
    NotPositiveDefinite { pivot: usize },
}

/// Outcome and measured work of one RMHMC proposal.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RmhmcTransition {
    pub accepted: bool,
    pub acceptance_probability: f64,
    pub integration_steps: usize,
    pub initial_energy: Option<f64>,
    pub proposal_energy: Option<f64>,
    pub energy_error: Option<f64>,
    pub termination: Option<RmhmcTermination>,
    pub costs: RmhmcCosts,
    /// None when disabled; replay begins false and becomes true only after every check.
    pub reverse_passed: Option<bool>,
    /// Maximum scaled boundary/half-momentum replay discrepancy.
    pub maximum_reverse_error: Option<f64>,
}

impl RmhmcTransition {
    fn new() -> Self {
        Self {
            accepted: false,
            acceptance_probability: 0.0,
            integration_steps: 0,
            initial_energy: None,
            proposal_energy: None,
            energy_error: None,
            termination: None,
            costs: RmhmcCosts::default(),
            reverse_passed: None,
            maximum_reverse_error: None,
        }
    }
}

#[derive(Debug)]
struct MetricState {
    dimension: usize,
    matrix: OwnedBuffer,
    derivatives: OwnedBuffer,
    lower: OwnedBuffer,
    inverse: OwnedBuffer,
    log_det: f64,
}

impl MetricState {
    fn new(dimension: usize) -> Result<Self, RmhmcConfigErrorInternal> {
        let matrix_len = dimension
            .checked_mul(dimension)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
            .ok_or(RmhmcConfigErrorInternal)?;
        let derivative_len = matrix_len
            .checked_mul(dimension)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
            .ok_or(RmhmcConfigErrorInternal)?;
        Ok(Self {
            dimension,
            matrix: OwnedBuffer::new(matrix_len),
            derivatives: OwnedBuffer::new(derivative_len),
            lower: OwnedBuffer::new(matrix_len),
            inverse: OwnedBuffer::new(matrix_len),
            log_det: 0.0,
        })
    }

    fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.dimension, other.dimension);
        self.matrix.copy_from_slice(&other.matrix);
        self.derivatives.copy_from_slice(&other.derivatives);
        self.lower.copy_from_slice(&other.lower);
        self.inverse.copy_from_slice(&other.inverse);
        self.log_det = other.log_det;
    }

    fn velocity(&self, momentum: &[f64], velocity: &mut [f64]) {
        let d = self.dimension;
        for (row, output) in velocity.iter_mut().enumerate() {
            *output = self.inverse[row * d..(row + 1) * d]
                .iter()
                .zip(momentum)
                .map(|(entry, value)| entry * value)
                .sum();
        }
    }

    fn sample_momentum(&self, normal: &[f64], momentum: &mut [f64]) {
        let d = self.dimension;
        for (row, output) in momentum.iter_mut().enumerate() {
            *output = (0..=row)
                .map(|column| self.lower[row * d + column] * normal[column])
                .sum();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RmhmcConfigErrorInternal;

fn evaluate_metric<TE: Error + 'static, M: PositionMetric + ?Sized>(
    provider: &M,
    position: &[f64],
    state: &mut MetricState,
    provider_workspace: &mut [f64],
    solve_scratch: &mut [f64],
    costs: &mut RmhmcCosts,
) -> Result<(), RmhmcError<TE, M::Error>> {
    evaluate_position_metric(
        provider,
        position,
        &mut state.matrix,
        &mut state.derivatives,
        provider_workspace,
    )?;
    costs.metric_derivative_evaluations += 1;
    factor_and_invert(state, solve_scratch, costs)?;
    Ok(())
}

fn factor_and_invert<TE: Error + 'static, ME: Error + 'static>(
    state: &mut MetricState,
    scratch: &mut [f64],
    costs: &mut RmhmcCosts,
) -> Result<(), RmhmcError<TE, ME>> {
    let d = state.dimension;
    state.lower.fill(0.0);
    for row in 0..d {
        for column in 0..=row {
            let correction = (0..column)
                .map(|k| state.lower[row * d + k] * state.lower[column * d + k])
                .sum::<f64>();
            let value = state.matrix[row * d + column] - correction;
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(RmhmcError::NotPositiveDefinite { pivot: row });
                }
                state.lower[row * d + column] = value.sqrt();
            } else {
                state.lower[row * d + column] = value / state.lower[column * d + column];
            }
        }
    }
    state.log_det = 2.0 * (0..d).map(|i| state.lower[i * d + i].ln()).sum::<f64>();
    costs.factorizations += 1;

    state.inverse.fill(0.0);
    for column in 0..d {
        scratch.fill(0.0);
        for row in 0..d {
            let rhs = if row == column { 1.0 } else { 0.0 };
            let correction = (0..row)
                .map(|k| state.lower[row * d + k] * scratch[k])
                .sum::<f64>();
            scratch[row] = (rhs - correction) / state.lower[row * d + row];
        }
        for row in (0..d).rev() {
            let correction = ((row + 1)..d)
                .map(|k| state.lower[k * d + row] * state.inverse[k * d + column])
                .sum::<f64>();
            state.inverse[row * d + column] =
                (scratch[row] - correction) / state.lower[row * d + row];
        }
        costs.triangular_solves += 2;
    }
    if state
        .inverse
        .iter()
        .chain(std::iter::once(&state.log_det))
        .any(|value| !value.is_finite())
    {
        return Err(RmhmcError::NotPositiveDefinite {
            pivot: d.saturating_sub(1),
        });
    }
    Ok(())
}

#[derive(Debug)]
struct ReversePhase {
    position: OwnedBuffer,
    momentum: OwnedBuffer,
    half: OwnedBuffer,
}
impl ReversePhase {
    fn new(d: usize) -> Self {
        Self {
            position: OwnedBuffer::new(d),
            momentum: OwnedBuffer::new(d),
            half: OwnedBuffer::new(d),
        }
    }
}

#[derive(Debug)]
struct Workspace<'a, T: LogDensityGradient + ?Sized> {
    proposal: PointState<'a, T>,
    evaluation: EvaluationWorkspace,
    metric_proposal: MetricState,
    metric_trial: MetricState,
    metric_workspace: OwnedBuffer,
    solve_scratch: OwnedBuffer,
    normal: OwnedBuffer,
    momentum: OwnedBuffer,
    momentum_guess: OwnedBuffer,
    momentum_next: OwnedBuffer,
    position_guess: OwnedBuffer,
    position_next: OwnedBuffer,
    velocity: OwnedBuffer,
    velocity_initial: OwnedBuffer,
    force: OwnedBuffer,
    newton_matrix: OwnedBuffer,
    reverse_point: PointState<'a, T>,
    reverse_metric: MetricState,
    reverse_momentum: OwnedBuffer,
    reverse_history: Vec<ReversePhase>,
}

impl<'a, T: LogDensityGradient + ?Sized> Workspace<'a, T> {
    fn new<M: PositionMetric + ?Sized>(
        point: &PointState<'a, T>,
        metric: &MetricState,
        provider: &M,
    ) -> Result<Self, RmhmcError<T::Error, M::Error>> {
        let d = point.dimension();
        let metric_workspace_len = provider.workspace_len();
        if metric_workspace_len > isize::MAX as usize / size_of::<f64>() {
            return Err(RmhmcError::SizeOverflow);
        }
        let mut metric_proposal = MetricState::new(d).map_err(|_| RmhmcError::SizeOverflow)?;
        metric_proposal.copy_from(metric);
        Ok(Self {
            proposal: point.clone(),
            evaluation: EvaluationWorkspace::new(d),
            metric_proposal,
            metric_trial: MetricState::new(d).map_err(|_| RmhmcError::SizeOverflow)?,
            metric_workspace: OwnedBuffer::new(metric_workspace_len),
            solve_scratch: OwnedBuffer::new(d),
            normal: OwnedBuffer::new(d),
            momentum: OwnedBuffer::new(d),
            momentum_guess: OwnedBuffer::new(d),
            momentum_next: OwnedBuffer::new(d),
            position_guess: OwnedBuffer::new(d),
            position_next: OwnedBuffer::new(d),
            velocity: OwnedBuffer::new(d),
            velocity_initial: OwnedBuffer::new(d),
            force: OwnedBuffer::new(d),
            newton_matrix: OwnedBuffer::new(d * d),
            reverse_point: point.clone(),
            reverse_metric: MetricState::new(d).map_err(|_| RmhmcError::SizeOverflow)?,
            reverse_momentum: OwnedBuffer::new(d),
            reverse_history: Vec::new(),
        })
    }
}

/// Research energy-only MH kernel with unresolved reverse-support and volume laws.
#[derive(Debug)]
#[must_use = "this research kernel has unresolved reversal and volume obligations"]
pub struct ExactnessPendingRmhmc<'a, T: LogDensityGradient + ?Sized, M: PositionMetric + ?Sized> {
    point: PointState<'a, T>,
    metric: MetricState,
    metric_provider: &'a M,
    workspace: Workspace<'a, T>,
    options: RmhmcOptions,
}

impl<'a, T, M> ExactnessPendingRmhmc<'a, T, M>
where
    T: LogDensityGradient + ?Sized,
    M: PositionMetric + ?Sized,
{
    /// Constructs and validates the initial target/metric cache.
    ///
    /// # Errors
    /// Returns target, metric, dimension, storage, or factorization failures.
    pub fn new(
        target: &'a T,
        metric_provider: &'a M,
        position: OwnedBuffer,
        options: RmhmcOptions,
    ) -> Result<Self, RmhmcError<T::Error, M::Error>> {
        let dimension = target.dimension();
        if dimension == 0 {
            return Err(RmhmcError::EmptyTarget);
        }
        if metric_provider.dimension() != dimension {
            return Err(RmhmcError::Dimension {
                target: dimension,
                metric: metric_provider.dimension(),
            });
        }
        let metric_workspace_len = metric_provider.workspace_len();
        if metric_workspace_len > isize::MAX as usize / size_of::<f64>() {
            return Err(RmhmcError::SizeOverflow);
        }
        let point = PointState::new(target, position)?;
        let mut metric = MetricState::new(dimension).map_err(|_| RmhmcError::SizeOverflow)?;
        let mut metric_workspace = OwnedBuffer::new(metric_workspace_len);
        let mut solve_scratch = OwnedBuffer::new(dimension);
        let mut construction_costs = RmhmcCosts::default();
        evaluate_metric(
            metric_provider,
            point.position(),
            &mut metric,
            &mut metric_workspace,
            &mut solve_scratch,
            &mut construction_costs,
        )?;
        let mut workspace = Workspace::new(&point, &metric, metric_provider)?;
        if options.reverse_tolerance.is_some() {
            workspace.reverse_history = (0..options.integration_steps.get())
                .map(|_| ReversePhase::new(dimension))
                .collect();
        }
        Ok(Self {
            point,
            metric,
            metric_provider,
            workspace,
            options,
        })
    }

    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }

    pub fn options(&self) -> RmhmcOptions {
        self.options
    }

    pub(crate) fn install_options(&mut self, options: RmhmcOptions) {
        if options.reverse_tolerance.is_some()
            && self.workspace.reverse_history.len() < options.integration_steps.get()
        {
            self.workspace
                .reverse_history
                .resize_with(options.integration_steps.get(), || {
                    ReversePhase::new(self.point.dimension())
                });
        }
        self.options = options;
    }

    /// Attempts one full-momentum-refresh RMHMC transition.
    ///
    /// # Errors
    /// Backend, shape, metric, or factorization failures retain their concrete
    /// types and never publish a partial proposal. Bounded implicit failures are
    /// ordinary rejected transitions with an explicit event.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<RmhmcTransition, RmhmcError<T::Error, M::Error>> {
        let mut transition = RmhmcTransition::new();
        self.workspace.proposal.clone_from(&self.point);
        self.workspace.metric_proposal.copy_from(&self.metric);
        for value in self.workspace.normal.iter_mut() {
            *value = alea_math::random::standard_normal(rng);
        }
        self.metric
            .sample_momentum(&self.workspace.normal, &mut self.workspace.momentum);
        let initial = hamiltonian(
            &self.point,
            &self.metric,
            &self.workspace.momentum,
            &mut self.workspace.velocity,
            &mut transition.costs,
        );
        transition.initial_energy = Some(initial);
        if !initial.is_finite() {
            transition.termination = Some(RmhmcTermination::Divergence(Divergence::Energy));
            return Ok(transition);
        }
        let step_size = self.options.step_size.value();
        for step in 0..self.options.integration_steps.get() {
            if self.options.reverse_tolerance.is_some() {
                let saved = &mut self.workspace.reverse_history[step];
                saved
                    .position
                    .copy_from_slice(self.workspace.proposal.position());
                saved.momentum.copy_from_slice(&self.workspace.momentum);
            }
            if let Some(termination) =
                self.integrate_step(step, step_size, &mut transition.costs)?
            {
                transition.termination = Some(termination);
                return Ok(transition);
            }
            if self.options.reverse_tolerance.is_some() {
                self.workspace.reverse_history[step]
                    .half
                    .copy_from_slice(&self.workspace.momentum_guess);
            }
            transition.integration_steps = step + 1;
        }
        if let Some(tolerance) = self.options.reverse_tolerance
            && let Some(termination) = self.reverse_replay(tolerance, &mut transition)?
        {
            transition.termination = Some(termination);
            return Ok(transition);
        }
        let proposed = hamiltonian(
            &self.workspace.proposal,
            &self.workspace.metric_proposal,
            &self.workspace.momentum,
            &mut self.workspace.velocity,
            &mut transition.costs,
        );
        transition.proposal_energy = Some(proposed);
        let error = proposed - initial;
        if !proposed.is_finite() || !error.is_finite() {
            transition.termination = Some(RmhmcTermination::Divergence(Divergence::Energy));
            return Ok(transition);
        }
        transition.energy_error = Some(error);
        match metropolis(initial, proposed, self.options.max_energy_error) {
            Ok(probability) => {
                transition.acceptance_probability = probability;
                transition.accepted = rng.random::<f64>() < probability;
            }
            Err(reason) => {
                transition.termination = Some(RmhmcTermination::Divergence(reason));
            }
        }
        if transition.accepted {
            std::mem::swap(&mut self.point, &mut self.workspace.proposal);
            std::mem::swap(&mut self.metric, &mut self.workspace.metric_proposal);
        }
        Ok(transition)
    }

    fn integrate_step(
        &mut self,
        integration_step: usize,
        step_size: f64,
        costs: &mut RmhmcCosts,
    ) -> Result<Option<RmhmcTermination>, RmhmcError<T::Error, M::Error>> {
        if self.options.solver == ImplicitSolver::Newton {
            return self.integrate_newton(integration_step, step_size, costs);
        }
        let iterations = self.options.implicit_iterations.get();
        self.workspace
            .momentum_guess
            .copy_from_slice(&self.workspace.momentum);
        let mut residual = f64::INFINITY;
        for _ in 0..iterations {
            metric_force(
                self.workspace.proposal.gradient(),
                &self.workspace.metric_proposal,
                &self.workspace.momentum_guess,
                &mut self.workspace.velocity,
                &mut self.workspace.force,
                costs,
            );
            for ((next, initial), force) in self
                .workspace
                .momentum_next
                .iter_mut()
                .zip(self.workspace.momentum.iter())
                .zip(self.workspace.force.iter())
            {
                *next = *initial + 0.5 * step_size * force;
            }
            residual = scaled_residual(
                &self.workspace.momentum_guess,
                &self.workspace.momentum_next,
            );
            std::mem::swap(
                &mut self.workspace.momentum_guess,
                &mut self.workspace.momentum_next,
            );
            costs.momentum_iterations += 1;
        }
        if !residual.is_finite() || residual > self.options.implicit_tolerance {
            return Ok(Some(RmhmcTermination::ImplicitSolve(ImplicitSolveEvent {
                integration_step,
                stage: ImplicitStage::Momentum,
                iterations,
                scaled_residual: residual,
                tolerance: self.options.implicit_tolerance,
            })));
        }

        self.workspace.metric_proposal.velocity(
            &self.workspace.momentum_guess,
            &mut self.workspace.velocity_initial,
        );
        costs.inverse_actions += 1;
        self.workspace
            .position_guess
            .copy_from_slice(self.workspace.proposal.position());
        residual = f64::INFINITY;
        for _ in 0..iterations {
            evaluate_metric(
                self.metric_provider,
                &self.workspace.position_guess,
                &mut self.workspace.metric_trial,
                &mut self.workspace.metric_workspace,
                &mut self.workspace.solve_scratch,
                costs,
            )?;
            self.workspace
                .metric_trial
                .velocity(&self.workspace.momentum_guess, &mut self.workspace.velocity);
            costs.inverse_actions += 1;
            for (((next, initial), initial_velocity), trial_velocity) in self
                .workspace
                .position_next
                .iter_mut()
                .zip(self.workspace.proposal.position())
                .zip(self.workspace.velocity_initial.iter())
                .zip(self.workspace.velocity.iter())
            {
                *next = *initial + 0.5 * step_size * (initial_velocity + trial_velocity);
            }
            residual = scaled_residual(
                &self.workspace.position_guess,
                &self.workspace.position_next,
            );
            std::mem::swap(
                &mut self.workspace.position_guess,
                &mut self.workspace.position_next,
            );
            costs.position_iterations += 1;
        }
        if !residual.is_finite() || residual > self.options.implicit_tolerance {
            return Ok(Some(RmhmcTermination::ImplicitSolve(ImplicitSolveEvent {
                integration_step,
                stage: ImplicitStage::Position,
                iterations,
                scaled_residual: residual,
                tolerance: self.options.implicit_tolerance,
            })));
        }

        self.finish_step(step_size, costs)
    }

    fn integrate_newton(
        &mut self,
        integration_step: usize,
        h: f64,
        costs: &mut RmhmcCosts,
    ) -> Result<Option<RmhmcTermination>, RmhmcError<T::Error, M::Error>> {
        let d = self.point.dimension();
        let tolerance = self.options.implicit_tolerance;
        self.workspace
            .momentum_guess
            .copy_from_slice(&self.workspace.momentum);
        let mut residual = f64::INFINITY;
        let mut converged = false;
        let mut used = 0;
        for iteration in 0..self.options.implicit_iterations.get() {
            used = iteration + 1;
            costs.momentum_iterations += 1;
            metric_force(
                self.workspace.proposal.gradient(),
                &self.workspace.metric_proposal,
                &self.workspace.momentum_guess,
                &mut self.workspace.velocity,
                &mut self.workspace.force,
                costs,
            );
            for i in 0..d {
                self.workspace.momentum_next[i] =
                    self.workspace.momentum[i] + 0.5 * h * self.workspace.force[i];
            }
            residual = scaled_residual(
                &self.workspace.momentum_guess,
                &self.workspace.momentum_next,
            );
            if !self.workspace.momentum_next.iter().all(|x| x.is_finite()) {
                break;
            }
            if residual <= tolerance {
                converged = true;
                break;
            }
            // R=p_half-p0-h/2*force(q0,p_half); dforce_i/dp_j
            // = (G^-1 * dG_i * velocity)_j, using symmetric dG_i and G^-1.
            let metric = &self.workspace.metric_proposal;
            for i in 0..d {
                for j in 0..d {
                    let mut derivative = 0.;
                    for row in 0..d {
                        for column in 0..d {
                            derivative += metric.inverse[j * d + row]
                                * metric.derivatives[i * d * d + row * d + column]
                                * self.workspace.velocity[column];
                        }
                    }
                    self.workspace.newton_matrix[i * d + j] =
                        f64::from(i == j) - 0.5 * h * derivative;
                }
            }
            for i in 0..d {
                self.workspace.momentum_next[i] =
                    self.workspace.momentum_guess[i] - self.workspace.momentum_next[i];
            }
            costs.newton_jacobians += 1;
            if !solve_newton(
                &mut self.workspace.newton_matrix,
                &mut self.workspace.momentum_next,
                d,
            ) {
                break;
            }
            costs.newton_linear_solves += 1;
            for i in 0..d {
                self.workspace.momentum_guess[i] -= self.workspace.momentum_next[i];
            }
            if self.workspace.momentum_guess.iter().any(|x| !x.is_finite()) {
                break;
            }
        }
        if !converged {
            return Ok(Some(RmhmcTermination::ImplicitSolve(ImplicitSolveEvent {
                integration_step,
                stage: ImplicitStage::Momentum,
                iterations: used,
                scaled_residual: residual,
                tolerance,
            })));
        }
        self.workspace.metric_proposal.velocity(
            &self.workspace.momentum_guess,
            &mut self.workspace.velocity_initial,
        );
        costs.inverse_actions += 1;
        self.workspace
            .position_guess
            .copy_from_slice(self.workspace.proposal.position());
        converged = false;
        residual = f64::INFINITY;
        for iteration in 0..self.options.implicit_iterations.get() {
            used = iteration + 1;
            costs.position_iterations += 1;
            evaluate_metric(
                self.metric_provider,
                &self.workspace.position_guess,
                &mut self.workspace.metric_trial,
                &mut self.workspace.metric_workspace,
                &mut self.workspace.solve_scratch,
                costs,
            )?;
            self.workspace
                .metric_trial
                .velocity(&self.workspace.momentum_guess, &mut self.workspace.velocity);
            costs.inverse_actions += 1;
            for i in 0..d {
                self.workspace.position_next[i] = self.workspace.proposal.position()[i]
                    + 0.5 * h * (self.workspace.velocity_initial[i] + self.workspace.velocity[i]);
            }
            residual = scaled_residual(
                &self.workspace.position_guess,
                &self.workspace.position_next,
            );
            if !self.workspace.position_next.iter().all(|x| x.is_finite()) {
                break;
            }
            if residual <= tolerance {
                converged = true;
                break;
            }
            let metric = &self.workspace.metric_trial;
            // R=q1-q0-h/2(v0+v1); dv_i/dq_j=-(G^-1*dG_j*v1)_i.
            for i in 0..d {
                for j in 0..d {
                    let mut derivative = 0.;
                    for row in 0..d {
                        for column in 0..d {
                            derivative += metric.inverse[i * d + row]
                                * metric.derivatives[j * d * d + row * d + column]
                                * self.workspace.velocity[column];
                        }
                    }
                    self.workspace.newton_matrix[i * d + j] =
                        f64::from(i == j) + 0.5 * h * derivative;
                }
            }
            for i in 0..d {
                self.workspace.position_next[i] =
                    self.workspace.position_guess[i] - self.workspace.position_next[i];
            }
            costs.newton_jacobians += 1;
            if !solve_newton(
                &mut self.workspace.newton_matrix,
                &mut self.workspace.position_next,
                d,
            ) {
                break;
            }
            costs.newton_linear_solves += 1;
            for i in 0..d {
                self.workspace.position_guess[i] -= self.workspace.position_next[i];
            }
            if self.workspace.position_guess.iter().any(|x| !x.is_finite()) {
                break;
            }
        }
        if !converged {
            return Ok(Some(RmhmcTermination::ImplicitSolve(ImplicitSolveEvent {
                integration_step,
                stage: ImplicitStage::Position,
                iterations: used,
                scaled_residual: residual,
                tolerance,
            })));
        }
        self.finish_step(h, costs)
    }

    fn reverse_replay(
        &mut self,
        tolerance: f64,
        transition: &mut RmhmcTransition,
    ) -> Result<Option<RmhmcTermination>, RmhmcError<T::Error, M::Error>> {
        transition.reverse_passed = Some(false);
        transition.maximum_reverse_error = Some(0.);
        self.workspace
            .reverse_point
            .clone_from(&self.workspace.proposal);
        self.workspace
            .reverse_metric
            .copy_from(&self.workspace.metric_proposal);
        self.workspace
            .reverse_momentum
            .copy_from_slice(&self.workspace.momentum);
        for p in self.workspace.momentum.iter_mut() {
            *p = -*p;
        }
        for reverse_step in 0..self.options.integration_steps.get() {
            let index = self.options.integration_steps.get() - 1 - reverse_step;
            if self
                .integrate_step(
                    reverse_step,
                    self.options.step_size.value(),
                    &mut transition.costs,
                )?
                .is_some()
            {
                transition.maximum_reverse_error = Some(f64::INFINITY);
                return Ok(Some(RmhmcTermination::ReverseCheck {
                    integration_step: index,
                    scaled_error: f64::INFINITY,
                    tolerance,
                }));
            }
            transition.costs.reverse_steps += 1;
            let expected = &self.workspace.reverse_history[index];
            let error = scaled_residual(self.workspace.proposal.position(), &expected.position)
                .max(scaled_reversed_residual(
                    &self.workspace.momentum,
                    &expected.momentum,
                ))
                .max(scaled_reversed_residual(
                    &self.workspace.momentum_guess,
                    &expected.half,
                ));
            transition.maximum_reverse_error = Some(
                transition
                    .maximum_reverse_error
                    .expect("replay started")
                    .max(error),
            );
            if !error.is_finite() || error > tolerance {
                return Ok(Some(RmhmcTermination::ReverseCheck {
                    integration_step: index,
                    scaled_error: error,
                    tolerance,
                }));
            }
        }
        // Publish only the complete forward cache after the whole replay passed.
        self.workspace
            .proposal
            .clone_from(&self.workspace.reverse_point);
        self.workspace
            .metric_proposal
            .copy_from(&self.workspace.reverse_metric);
        self.workspace
            .momentum
            .copy_from_slice(&self.workspace.reverse_momentum);
        transition.reverse_passed = Some(true);
        Ok(None)
    }

    fn finish_step(
        &mut self,
        step_size: f64,
        costs: &mut RmhmcCosts,
    ) -> Result<Option<RmhmcTermination>, RmhmcError<T::Error, M::Error>> {
        evaluate_metric(
            self.metric_provider,
            &self.workspace.position_guess,
            &mut self.workspace.metric_trial,
            &mut self.workspace.metric_workspace,
            &mut self.workspace.solve_scratch,
            costs,
        )?;
        if let Err(error) = self.workspace.proposal.try_update(
            &self.workspace.position_guess,
            &mut self.workspace.evaluation,
        ) {
            return match error {
                EvaluationError::NonFinitePosition { index } => {
                    Ok(Some(RmhmcTermination::Divergence(Divergence::Position {
                        index,
                    })))
                }
                EvaluationError::NonFiniteLogDensity => {
                    Ok(Some(RmhmcTermination::Divergence(Divergence::LogDensity)))
                }
                EvaluationError::NonFiniteGradient { index } => {
                    Ok(Some(RmhmcTermination::Divergence(Divergence::Gradient {
                        index,
                    })))
                }
                other => Err(RmhmcError::Target(other)),
            };
        }
        costs.target_evaluations += 1;
        std::mem::swap(
            &mut self.workspace.metric_proposal,
            &mut self.workspace.metric_trial,
        );
        metric_force(
            self.workspace.proposal.gradient(),
            &self.workspace.metric_proposal,
            &self.workspace.momentum_guess,
            &mut self.workspace.velocity,
            &mut self.workspace.force,
            costs,
        );
        for ((momentum, half), force) in self
            .workspace
            .momentum
            .iter_mut()
            .zip(self.workspace.momentum_guess.iter())
            .zip(self.workspace.force.iter())
        {
            *momentum = *half + 0.5 * step_size * force;
        }
        costs.dynamics_steps += 1;
        if let Some(index) = self
            .workspace
            .momentum
            .iter()
            .position(|value| !value.is_finite())
        {
            return Ok(Some(RmhmcTermination::Divergence(Divergence::Momentum {
                index,
            })));
        }
        Ok(None)
    }
}

impl<T, M> MarkovChain for ExactnessPendingRmhmc<'_, T, M>
where
    T: LogDensityGradient + ?Sized,
    M: PositionMetric + ?Sized,
{
    type Error = RmhmcError<T::Error, M::Error>;
    type Transition = RmhmcTransition;

    fn position(&self) -> &[f64] {
        self.point.position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Self::step(self, rng)
    }
}

// Pivoted dense Newton solves; no allocation or silent regularization.
fn solve_newton(matrix: &mut [f64], rhs: &mut [f64], d: usize) -> bool {
    if matrix.iter().chain(rhs.iter()).any(|x| !x.is_finite()) {
        return false;
    }
    for column in 0..d {
        let Some(pivot) = (column..d).max_by(|a, b| {
            matrix[*a * d + column]
                .abs()
                .total_cmp(&matrix[*b * d + column].abs())
        }) else {
            return false;
        };
        if matrix[pivot * d + column] == 0. {
            return false;
        }
        if pivot != column {
            for j in column..d {
                matrix.swap(pivot * d + j, column * d + j);
            }
            rhs.swap(pivot, column);
        }
        for row in column + 1..d {
            let factor = matrix[row * d + column] / matrix[column * d + column];
            for j in column + 1..d {
                matrix[row * d + j] -= factor * matrix[column * d + j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for row in (0..d).rev() {
        for column in row + 1..d {
            rhs[row] -= matrix[row * d + column] * rhs[column];
        }
        rhs[row] /= matrix[row * d + row];
    }
    rhs.iter().all(|x| x.is_finite())
}
fn scaled_reversed_residual(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a + b).abs() / (1. + a.abs().max(b.abs())))
        .fold(0., f64::max)
}

fn scaled_residual(previous: &[f64], next: &[f64]) -> f64 {
    previous
        .iter()
        .zip(next)
        .map(|(a, b)| (a - b).abs() / (1.0 + a.abs().max(b.abs())))
        .fold(0.0, f64::max)
}

fn metric_force(
    score: &[f64],
    metric: &MetricState,
    momentum: &[f64],
    velocity: &mut [f64],
    force: &mut [f64],
    costs: &mut RmhmcCosts,
) {
    let d = metric.dimension;
    metric.velocity(momentum, velocity);
    costs.inverse_actions += 1;
    for coordinate in 0..d {
        let derivative = &metric.derivatives[coordinate * d * d..(coordinate + 1) * d * d];
        let mut trace = 0.0;
        let mut quadratic = 0.0;
        for row in 0..d {
            for column in 0..d {
                trace += metric.inverse[row * d + column] * derivative[column * d + row];
                quadratic += velocity[row] * derivative[row * d + column] * velocity[column];
            }
        }
        force[coordinate] = score[coordinate] - 0.5 * trace + 0.5 * quadratic;
    }
}

fn hamiltonian<T: LogDensityGradient + ?Sized>(
    point: &PointState<'_, T>,
    metric: &MetricState,
    momentum: &[f64],
    velocity: &mut [f64],
    costs: &mut RmhmcCosts,
) -> f64 {
    metric.velocity(momentum, velocity);
    costs.inverse_actions += 1;
    -point.log_density()
        + 0.5 * metric.log_det
        + 0.5
            * momentum
                .iter()
                .zip(velocity.iter())
                .map(|(p, v)| p * v)
                .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    struct Normal;
    impl alea_core::target::TargetSpace for Normal {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            1
        }
    }
    impl LogDensityGradient for Normal {
        fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
            gradient[0] = -q[0];
            Ok(-0.5 * q[0] * q[0])
        }
    }

    struct ExponentialMetric;
    impl PositionMetric for ExponentialMetric {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            1
        }
        fn metric_and_derivatives(
            &self,
            q: &[f64],
            metric: &mut [f64],
            derivatives: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), Self::Error> {
            metric[0] = q[0].exp();
            derivatives[0] = metric[0];
            Ok(())
        }
    }

    fn deterministic_map(q: f64, p: f64, step: f64) -> (f64, f64) {
        let target = Normal;
        let metric = ExponentialMetric;
        let options = RmhmcOptions::new(step.abs(), 1, 24, 1.0e-13).unwrap();
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::from_fn(1, |_| q), options)
                .unwrap();
        chain.workspace.proposal.clone_from(&chain.point);
        chain.workspace.metric_proposal.copy_from(&chain.metric);
        chain.workspace.momentum[0] = p;
        let mut costs = RmhmcCosts::default();
        assert!(chain.integrate_step(0, step, &mut costs).unwrap().is_none());
        (
            chain.workspace.proposal.position()[0],
            chain.workspace.momentum[0],
        )
    }

    #[test]
    fn generalized_leapfrog_signed_reversal_and_volume() {
        let original = (0.2, -0.35);
        let forward = deterministic_map(original.0, original.1, 0.04);
        let reverse = deterministic_map(forward.0, forward.1, -0.04);
        assert!((reverse.0 - original.0).abs() < 2.0e-12);
        assert!((reverse.1 - original.1).abs() < 2.0e-12);

        let delta = 1.0e-6;
        let q_plus = deterministic_map(original.0 + delta, original.1, 0.04);
        let q_minus = deterministic_map(original.0 - delta, original.1, 0.04);
        let p_plus = deterministic_map(original.0, original.1 + delta, 0.04);
        let p_minus = deterministic_map(original.0, original.1 - delta, 0.04);
        let dq_dq = (q_plus.0 - q_minus.0) / (2.0 * delta);
        let dp_dq = (q_plus.1 - q_minus.1) / (2.0 * delta);
        let dq_dp = (p_plus.0 - p_minus.0) / (2.0 * delta);
        let dp_dp = (p_plus.1 - p_minus.1) / (2.0 * delta);
        let determinant = dq_dq * dp_dp - dq_dp * dp_dq;
        assert!((determinant - 1.0).abs() < 2.0e-8, "det={determinant}");
    }

    #[test]
    fn singular_newton_system_rejects_without_publishing() {
        let target = Normal;
        let metric = ExponentialMetric;
        let options = RmhmcOptions::new(1., 1, 100, 1e-12)
            .unwrap()
            .with_newton_solver();
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::new(1), options).unwrap();
        chain.workspace.proposal.clone_from(&chain.point);
        chain.workspace.metric_proposal.copy_from(&chain.metric);
        chain.workspace.momentum[0] = 2.;
        let mut costs = RmhmcCosts::default();
        assert!(matches!(
            chain.integrate_step(0, 1., &mut costs).unwrap(),
            Some(RmhmcTermination::ImplicitSolve(ImplicitSolveEvent {
                stage: ImplicitStage::Momentum,
                iterations: 1,
                ..
            }))
        ));
        assert_eq!(costs.newton_jacobians, 1);
        assert_eq!(costs.newton_linear_solves, 0);
        assert_eq!(chain.point().position(), &[0.]);
        let mut singular = [1., 2., 2., 4.];
        let mut rhs = [1., 2.];
        assert!(!solve_newton(&mut singular, &mut rhs, 2));
    }
    #[test]
    fn solver_tolerance_ladder_matches_independent_decimal_maps_and_failures() {
        for (row, line) in include_str!("../tests/fixtures/rmhmc/solver-ladder.csv")
            .lines()
            .skip(1)
            .enumerate()
        {
            let f: Vec<_> = line.split(',').collect();
            let v = |i: usize| f[i].parse::<f64>().unwrap();
            let target = Normal;
            let metric = ExponentialMetric;
            let options = RmhmcOptions::new(v(2), 1, v(5) as usize, v(4)).unwrap();
            let options = if f[3] == "newton" {
                options.with_newton_solver()
            } else {
                options
            };
            let mut chain = ExactnessPendingRmhmc::new(
                &target,
                &metric,
                OwnedBuffer::from_fn(1, |_| v(0)),
                options,
            )
            .unwrap();
            chain.workspace.proposal.clone_from(&chain.point);
            chain.workspace.metric_proposal.copy_from(&chain.metric);
            chain.workspace.momentum[0] = v(1);
            let mut costs = RmhmcCosts::default();
            let result = chain.integrate_step(0, v(2), &mut costs).unwrap();
            let stage = match result {
                None => "Success",
                Some(RmhmcTermination::ImplicitSolve(e)) => match e.stage {
                    ImplicitStage::Momentum => "Momentum",
                    ImplicitStage::Position => "Position",
                },
                _ => panic!("unexpected numerical failure"),
            };
            assert_eq!(stage, f[6], "row{row}");
            assert_eq!(costs.momentum_iterations, v(10) as usize, "row{row}");
            assert_eq!(costs.position_iterations, v(11) as usize, "row{row}");
            if result.is_none() {
                for (actual, want) in [
                    chain.workspace.proposal.position()[0],
                    chain.workspace.momentum[0],
                    chain.workspace.momentum_guess[0],
                ]
                .into_iter()
                .zip([v(7), v(8), v(9)])
                {
                    assert!(
                        (actual - want).abs() < 2e-11 * (1. + want.abs()),
                        "row{row}: actual={actual}, expected={want}"
                    );
                }
            }
            assert_eq!(chain.point().position(), &[v(0)]);
        }
    }
    #[test]
    fn multiple_implicit_roots_are_valid_but_do_not_define_one_selected_map() {
        let mut branches = std::collections::BTreeSet::new();
        for line in include_str!("../tests/fixtures/rmhmc/multiple-roots.csv")
            .lines()
            .skip(1)
        {
            let f: Vec<_> = line.split(',').collect();
            let v = |i: usize| f[i].parse::<f64>().unwrap();
            let (q, p, h, half, end, momentum) = (v(0), v(1), v(2), v(4), v(5), v(6));
            let force = |q: f64, p: f64| -q - 0.5 + 0.5 * (-q).exp() * p * p;
            let half_rhs = p + h / 2. * force(q, half);
            let end_rhs = q + h / 2. * half * ((-q).exp() + (-end).exp());
            assert!((half - half_rhs).abs() < 1e-10 * (1. + half.abs()));
            assert!((end - end_rhs).abs() < 1e-10 * (1. + end.abs()));
            assert!(
                (momentum - half - h / 2. * force(end, half)).abs() < 1e-10 * (1. + momentum.abs())
            );
            branches.insert(f[3]);
        }
        assert!(branches.contains("near-left"));
        assert!(branches.contains("near-right"));
        assert!(branches.contains("far-unique"));
    }

    #[test]
    fn newton_matches_independent_decimal_implicit_roots() {
        for row in include_str!("../tests/fixtures/rmhmc/exponential-decimal.csv").lines() {
            let v: Vec<f64> = row.split(',').map(|x| x.parse().unwrap()).collect();
            let target = Normal;
            let metric = ExponentialMetric;
            let options = RmhmcOptions::new(v[2], 1, 100, 1e-12)
                .unwrap()
                .with_newton_solver();
            let mut chain = ExactnessPendingRmhmc::new(
                &target,
                &metric,
                OwnedBuffer::from_fn(1, |_| v[0]),
                options,
            )
            .unwrap();
            chain.workspace.proposal.clone_from(&chain.point);
            chain.workspace.metric_proposal.copy_from(&chain.metric);
            chain.workspace.momentum[0] = v[1];
            let mut costs = RmhmcCosts::default();
            assert!(chain.integrate_step(0, v[2], &mut costs).unwrap().is_none());
            for (actual, expected) in [
                chain.workspace.proposal.position()[0],
                chain.workspace.momentum[0],
                chain.workspace.momentum_guess[0],
            ]
            .into_iter()
            .zip(&v[3..])
            {
                assert!(
                    (actual - expected).abs() < 3e-12,
                    "actual={actual}, expected={expected}"
                );
            }
            assert!(costs.newton_linear_solves > 0);
            assert!(costs.momentum_iterations < 100);
        }
    }

    #[test]
    fn full_reverse_replay_detects_corrupted_intermediate_half_momentum() {
        let target = Normal;
        let metric = ExponentialMetric;
        let options = RmhmcOptions::new(0.04, 1, 100, 1e-12)
            .unwrap()
            .with_newton_solver()
            .with_reverse_check(1e-8)
            .unwrap();
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::from_fn(1, |_| 0.2), options)
                .unwrap();
        chain.workspace.proposal.clone_from(&chain.point);
        chain.workspace.metric_proposal.copy_from(&chain.metric);
        chain.workspace.momentum[0] = -0.35;
        chain.workspace.reverse_history[0].position[0] = 0.2;
        chain.workspace.reverse_history[0].momentum[0] = -0.35;
        let mut t = RmhmcTransition::new();
        assert!(
            chain
                .integrate_step(0, 0.04, &mut t.costs)
                .unwrap()
                .is_none()
        );
        chain.workspace.reverse_history[0].half[0] = chain.workspace.momentum_guess[0] + 1e-4;
        assert!(matches!(
            chain.reverse_replay(1e-8, &mut t).unwrap(),
            Some(RmhmcTermination::ReverseCheck { .. })
        ));
        assert_eq!(chain.point().position(), &[0.2]);
        assert_eq!(t.reverse_passed, Some(false));
        assert!(t.maximum_reverse_error.unwrap() > 1e-8);
        assert_eq!(t.costs.reverse_steps, 1);
    }

    #[test]
    fn finite_map_can_pass_full_tolerant_replay_and_still_change_volume() {
        let map = |q: f64, p: f64| {
            let target = Normal;
            let metric = ExponentialMetric;
            let options = RmhmcOptions::new(0.4, 1, 100, 0.2)
                .unwrap()
                .with_newton_solver()
                .with_reverse_check(0.2)
                .unwrap();
            let mut chain = ExactnessPendingRmhmc::new(
                &target,
                &metric,
                OwnedBuffer::from_fn(1, |_| q),
                options,
            )
            .unwrap();
            chain.workspace.proposal.clone_from(&chain.point);
            chain.workspace.metric_proposal.copy_from(&chain.metric);
            chain.workspace.momentum[0] = p;
            chain.workspace.reverse_history[0].position[0] = q;
            chain.workspace.reverse_history[0].momentum[0] = p;
            let mut t = RmhmcTransition::new();
            assert!(
                chain
                    .integrate_step(0, 0.4, &mut t.costs)
                    .unwrap()
                    .is_none()
            );
            chain.workspace.reverse_history[0].half[0] = chain.workspace.momentum_guess[0];
            assert!(chain.reverse_replay(0.2, &mut t).unwrap().is_none());
            assert_eq!(t.reverse_passed, Some(true));
            (
                chain.workspace.proposal.position()[0],
                chain.workspace.momentum[0],
            )
        };
        let (q, p, delta) = (0.2, -0.35, 1e-6);
        let a = map(q + delta, p);
        let b = map(q - delta, p);
        let c = map(q, p + delta);
        let d = map(q, p - delta);
        let det = ((a.0 - b.0) * (c.1 - d.1) - (c.0 - d.0) * (a.1 - b.1)) / (4. * delta * delta);
        let analytic = 1. + 0.2 * (-q).exp() * p;
        assert!((det - analytic).abs() < 1e-8);
        assert!((det - 1.).abs() > 0.04);
    }

    #[test]
    fn finite_newton_termination_does_not_supply_a_unit_volume_contract() {
        let map = |q: f64, p: f64| {
            let target = Normal;
            let metric = ExponentialMetric;
            let options = RmhmcOptions::new(0.4, 1, 100, 0.1)
                .unwrap()
                .with_newton_solver();
            let mut chain = ExactnessPendingRmhmc::new(
                &target,
                &metric,
                OwnedBuffer::from_fn(1, |_| q),
                options,
            )
            .unwrap();
            chain.workspace.proposal.clone_from(&chain.point);
            chain.workspace.metric_proposal.copy_from(&chain.metric);
            chain.workspace.momentum[0] = p;
            assert!(
                chain
                    .integrate_step(0, 0.4, &mut RmhmcCosts::default())
                    .unwrap()
                    .is_none()
            );
            (
                chain.workspace.proposal.position()[0],
                chain.workspace.momentum[0],
            )
        };
        let q = 0.2;
        let p = -0.35;
        let delta = 1e-6;
        let a = map(q + delta, p);
        let b = map(q - delta, p);
        let c = map(q, p + delta);
        let d = map(q, p - delta);
        let determinant =
            ((a.0 - b.0) * (c.1 - d.1) - (c.0 - d.0) * (a.1 - b.1)) / (4. * delta * delta);
        let oracle = 1. + 0.2 * (-q).exp() * p;
        assert!((determinant - oracle).abs() < 1e-8);
        assert!((determinant - 1.).abs() > 0.04);
    }

    #[test]
    fn dense_log_determinant_inverse_and_action_match_two_by_two_oracle() {
        let mut metric = MetricState::new(2).unwrap();
        metric.matrix.copy_from_slice(&[4.0, 1.0, 1.0, 3.0]);
        let mut scratch = [0.0; 2];
        let mut costs = RmhmcCosts::default();
        factor_and_invert::<Infallible, Infallible>(&mut metric, &mut scratch, &mut costs).unwrap();
        assert!((metric.log_det - 11.0_f64.ln()).abs() < 2.0e-15);
        let expected = [3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0];
        for (actual, expected) in metric.inverse.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-16);
        }
        let mut velocity = [0.0; 2];
        metric.velocity(&[2.0, -1.0], &mut velocity);
        assert!((velocity[0] - 7.0 / 11.0).abs() < 2.0e-16);
        assert!((velocity[1] + 6.0 / 11.0).abs() < 2.0e-16);
        assert_eq!(costs.factorizations, 1);
        assert_eq!(costs.triangular_solves, 4);
    }

    #[test]
    fn implicit_failure_is_explicit_and_does_not_publish() {
        let target = Normal;
        let metric = ExponentialMetric;
        let options = RmhmcOptions::new(0.5, 1, 1, 1.0e-16).unwrap();
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::from_fn(1, |_| 0.4), options)
                .unwrap();
        let before = chain.point().position().to_vec();
        chain.workspace.proposal.clone_from(&chain.point);
        chain.workspace.metric_proposal.copy_from(&chain.metric);
        chain.workspace.momentum[0] = 0.7;
        let mut costs = RmhmcCosts::default();
        let event = chain.integrate_step(3, 0.5, &mut costs).unwrap();
        assert!(matches!(
            event,
            Some(RmhmcTermination::ImplicitSolve(ImplicitSolveEvent {
                integration_step: 3,
                ..
            }))
        ));
        assert_eq!(chain.point().position(), before);
    }
}
