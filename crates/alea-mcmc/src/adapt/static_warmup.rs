//! Finite step adaptation with already frozen geometry for static research kernels.
use super::{AdaptationError, DualAveraging};
use crate::{
    Hmc,
    config::{AcceptanceTarget, StepSize},
    ghmc::{DrGhmc, Ghmc},
    hmc::HmcError,
    integrator::SymmetricSplitting,
};
use alea_core::target::LogDensityGradient;
use alea_math::metric::EuclideanMetric;
use rand::Rng;
use std::error::Error;

/// Bounded step adaptation; geometry and trajectory counts are already fixed.
#[derive(Debug, Clone, Copy)]
pub struct StaticWarmupOptions {
    iterations: usize,
    target: AcceptanceTarget,
    minimum: StepSize,
    maximum: StepSize,
}
impl StaticWarmupOptions {
    /// Default target0.8 and step bounds1e-12..1e6. Zero iterations preserve the
    /// supplied step, including values outside these bounds, without RNG use.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            target: 0.8.try_into().expect("valid target"),
            minimum: 1e-12.try_into().expect("valid bound"),
            maximum: 1e6.try_into().expect("valid bound"),
        }
    }
    /// Set the adaptation target without changing retained kernel behavior.
    pub fn with_target(mut self, target: AcceptanceTarget) -> Self {
        self.target = target;
        self
    }
    /// Set finite ordered positive bounds for warmup updates.
    /// # Errors
    /// Rejects bounds with minimum greater than or equal to maximum.
    pub fn with_step_bounds(
        mut self,
        minimum: StepSize,
        maximum: StepSize,
    ) -> Result<Self, AdaptationError> {
        if minimum.value() >= maximum.value() {
            return Err(AdaptationError::Configuration);
        }
        self.minimum = minimum;
        self.maximum = maximum;
        Ok(self)
    }
}

/// Counts describe the complete warmup cost, including delayed ghost work.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct StaticWarmupReport {
    /// Completed transitions.
    pub iterations: usize,
    /// Accepted transitions, including second-stage DRGHMC acceptances.
    pub accepted: usize,
    /// Transitions containing at least one numerical divergence.
    pub divergences: usize,
    /// Attempted macrosteps including reverse/ghost integration.
    pub integration_attempts: usize,
    /// Actual first-stage macrosteps, excluding delayed and ghost proposals.
    pub first_stage_integration_steps: usize,
    /// Actual delayed-proposal macrosteps.
    pub delayed_integration_steps: usize,
    /// Actual reverse-ghost macrosteps.
    pub ghost_integration_steps: usize,
    /// Controller updates limited by configured step bounds.
    pub bounded_step_updates: usize,
    /// Final smoothed, bounded step.
    pub step_size: StepSize,
}

/// Concrete kernel errors retain their type; failed warmup returns no chain.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StaticWarmupError<E: Error + 'static> {
    /// A kernel/target/metric operation failed.
    #[error("static warmup kernel failed: {0}")]
    SamplerConfig(#[source] E),
    /// Invalid acceptance or controller arithmetic.
    #[error(transparent)]
    Adaptation(#[from] AdaptationError),
    /// Integration-work count overflowed.
    #[error("static warmup work counter overflow")]
    CounterOverflow,
}

/// Frozen static chain and its finite adaptation work, with concrete kernel errors.
pub type StaticWarmupResult<C, E> = Result<(C, StaticWarmupReport), StaticWarmupError<E>>;

pub(super) struct Observation {
    pub(super) acceptance: f64,
    pub(super) accepted: bool,
    pub(super) divergent: bool,
    pub(super) steps: usize,
    pub(super) delayed_steps: usize,
    pub(super) ghost_steps: usize,
}
pub(super) fn drive<R: Rng + ?Sized, E: Error + 'static>(
    initial: StepSize,
    options: StaticWarmupOptions,
    rng: &mut R,
    mut step: impl FnMut(StepSize, &mut R) -> Result<Observation, E>,
) -> Result<StaticWarmupReport, StaticWarmupError<E>> {
    let mut report = StaticWarmupReport {
        iterations: 0,
        accepted: 0,
        divergences: 0,
        integration_attempts: 0,
        first_stage_integration_steps: 0,
        delayed_integration_steps: 0,
        ghost_integration_steps: 0,
        bounded_step_updates: 0,
        step_size: initial,
    };
    if options.iterations == 0 {
        return Ok(report);
    }
    let mut controller = DualAveraging::new(initial, options.target);
    let bound = |s: StepSize| {
        s.value()
            .clamp(options.minimum.value(), options.maximum.value())
    };
    let mut current = initial;
    for _ in 0..options.iterations {
        let o = step(current, rng).map_err(StaticWarmupError::SamplerConfig)?;
        report.iterations += 1;
        report.accepted += usize::from(o.accepted);
        report.divergences += usize::from(o.divergent);
        report.integration_attempts = report
            .integration_attempts
            .checked_add(o.steps)
            .ok_or(StaticWarmupError::CounterOverflow)?;
        let first = o
            .steps
            .checked_sub(o.delayed_steps)
            .and_then(|n| n.checked_sub(o.ghost_steps))
            .ok_or(StaticWarmupError::CounterOverflow)?;
        report.first_stage_integration_steps = report
            .first_stage_integration_steps
            .checked_add(first)
            .ok_or(StaticWarmupError::CounterOverflow)?;
        report.delayed_integration_steps = report
            .delayed_integration_steps
            .checked_add(o.delayed_steps)
            .ok_or(StaticWarmupError::CounterOverflow)?;
        report.ghost_integration_steps = report
            .ghost_integration_steps
            .checked_add(o.ghost_steps)
            .ok_or(StaticWarmupError::CounterOverflow)?;
        let proposed = controller.update(o.acceptance)?;
        let value = bound(proposed);
        report.bounded_step_updates += usize::from(value != proposed.value());
        current = value.try_into().expect("bounded step");
    }
    let proposed = controller.finish();
    let value = bound(proposed);
    report.bounded_step_updates += usize::from(value != proposed.value());
    report.step_size = value.try_into().expect("bounded step");
    Ok(report)
}

/// Adapt static HMC steps with frozen geometry and integration count.
/// # Errors
/// Propagates typed HMC/backend, adaptation and counter errors; no chain escapes.
pub fn warmup_hmc_step_size<
    'a,
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: SymmetricSplitting,
    R: Rng + ?Sized,
>(
    mut chain: Hmc<'a, T, M, I>,
    options: StaticWarmupOptions,
    rng: &mut R,
) -> StaticWarmupResult<Hmc<'a, T, M, I>, HmcError<T::Error>> {
    let report = drive(chain.options().step_size(), options, rng, |step, rng| {
        chain.set_step_size(step);
        let o = chain.step(rng)?;
        Ok(Observation {
            acceptance: o.acceptance_probability,
            accepted: o.accepted,
            divergent: o.divergence.is_some(),
            steps: o.integration_steps,
            delayed_steps: 0,
            ghost_steps: 0,
        })
    })?;
    chain.set_step_size(report.step_size);
    Ok((chain, report))
}

/// Adapt persistent GHMC without changing its metric, persistence or trajectory count.
/// # Errors
/// Propagates concrete kernel/backend, controller and counter errors.
pub fn warmup_ghmc_step_size<
    'a,
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: SymmetricSplitting,
    R: Rng + ?Sized,
>(
    mut chain: Ghmc<'a, T, M, I>,
    options: StaticWarmupOptions,
    rng: &mut R,
) -> StaticWarmupResult<Ghmc<'a, T, M, I>, crate::ghmc::GhmcError<T::Error>> {
    let report = drive(chain.options().step_size(), options, rng, |step, rng| {
        chain.set_step_size(step);
        let o = chain.step(rng)?;
        Ok(Observation {
            acceptance: o.acceptance_probability,
            accepted: o.accepted,
            divergent: o.divergence.is_some(),
            steps: o.integration_steps,
            delayed_steps: 0,
            ghost_steps: 0,
        })
    })?;
    chain.set_step_size(report.step_size);
    Ok((chain, report))
}

/// Adapt DRGHMC using first-stage MH probability, including rejected first stages.
/// Actual accepted counts and work include delayed proposals and reverse ghosts.
/// # Errors
/// Rejects a step whose derived retry step is unrepresentable; propagates kernel,
/// backend, adaptation and counter errors. No partly tuned chain escapes.
pub fn warmup_drghmc_step_size<
    'a,
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: SymmetricSplitting,
    R: Rng + ?Sized,
>(
    mut chain: DrGhmc<'a, T, M, I>,
    options: StaticWarmupOptions,
    rng: &mut R,
) -> StaticWarmupResult<DrGhmc<'a, T, M, I>, crate::ghmc::GhmcError<T::Error>> {
    let report = drive(chain.options().first_step(), options, rng, |step, rng| {
        chain.set_step_size(step)?;
        let o = chain.step(rng)?;
        Ok(Observation {
            acceptance: o.first.acceptance_probability,
            accepted: o.accepted,
            divergent: o.first.divergence.is_some() || o.second.divergence.is_some(),
            steps: o.integration_steps,
            delayed_steps: o.second.integration_steps,
            ghost_steps: o.ghost_integration_steps,
        })
    })?;
    chain
        .set_step_size(report.step_size)
        .map_err(StaticWarmupError::SamplerConfig)?;
    Ok((chain, report))
}
