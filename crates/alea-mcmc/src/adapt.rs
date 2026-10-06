//! Explicit step-size warmup, separate from stationary sampling.
//! RWMH scale adaptation and windowed HMC warmup, separate from retained draws.
mod covariance;
mod fisher;
mod step;
mod window;
use crate::{
    Rwmh,
    config::{AcceptanceTarget, StepSize},
    rwmh::RwmhError,
};
use alea_core::density::LogDensity;
use alea_math::metric::EuclideanMetric;
pub use covariance::{CovarianceError, MetricKind, OnlineCovariance, WarmupMetric};
pub use fisher::{
    FisherError, FisherHmcWarmup, FisherMetricAdapter, FisherOptions, FisherWarmupError,
    FisherWarmupResult, MetricInitialization, WeightedFisherMoments,
};
pub use step::{DiminishingSchedule, StepAdaptation, StepObservation, StepSizeController};
pub use window::{
    HmcWarmup, HmcWarmupReport, HmcWarmupResult, SearchOptions, SearchResult, WarmupConfigError,
    WarmupError, WarmupOptions, WarmupStage, WindowSchedule, find_reasonable_step_size,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AdaptationError {
    #[error("invalid diminishing schedule or optimizer configuration")]
    Configuration,
    #[error("backend failures cannot update adaptation")]
    BackendFailure,
    #[error("nonfinite step adaptation arithmetic")]
    Numerical,
    #[error("acceptance probability must be finite and between zero and one")]
    AcceptanceProbability,
    #[error("adaptation iteration count overflow")]
    IterationOverflow,
}

/// Dual averaging with gamma=0.05, t0=10, kappa=0.75.
/// Scales are clamped to the finite normal range in log space.
#[derive(Debug, Clone)]
pub struct DualAveraging {
    mu: f64,
    h_bar: f64,
    log_average: f64,
    iterations: usize,
    target: AcceptanceTarget,
}
impl DualAveraging {
    pub fn new(initial: StepSize, target: AcceptanceTarget) -> Self {
        Self {
            mu: 10.0_f64.ln() + initial.value().ln(),
            h_bar: 0.0,
            log_average: initial.value().ln(),
            iterations: 0,
            target,
        }
    }
    /// Invalid input leaves all controller state unchanged.
    pub fn update(&mut self, acceptance: f64) -> Result<StepSize, AdaptationError> {
        if !acceptance.is_finite() || !(0.0..=1.0).contains(&acceptance) {
            return Err(AdaptationError::AcceptanceProbability);
        }
        let iterations = self
            .iterations
            .checked_add(1)
            .ok_or(AdaptationError::IterationOverflow)?;
        let m = iterations as f64;
        let weight = 1.0 / (m + 10.0);
        self.h_bar = (1.0 - weight) * self.h_bar + weight * (self.target.value() - acceptance);
        let log_step = self.mu - m.sqrt() / 0.05 * self.h_bar;
        let decay = m.powf(-0.75);
        self.log_average = decay * log_step + (1.0 - decay) * self.log_average;
        self.iterations = iterations;
        Ok(scale(log_step))
    }
    /// Consumes the controller; no adaptation state accompanies stationary draws.
    pub fn finish(self) -> StepSize {
        scale(self.log_average)
    }
}
fn scale(log_step: f64) -> StepSize {
    // ln(MAX).exp() can round up; clamp again after exponentiation.
    let value = log_step
        .clamp(f64::MIN_POSITIVE.ln(), f64::MAX.ln())
        .exp()
        .clamp(f64::MIN_POSITIVE, f64::MAX);
    StepSize::try_from(value).expect("clamped finite positive step size")
}

#[derive(Debug, Clone, Copy)]
pub struct WarmupSummary {
    pub iterations: usize,
    pub accepted: usize,
    pub step_size: StepSize,
}
impl WarmupSummary {
    pub fn acceptance_rate(self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.accepted as f64 / self.iterations as f64
        }
    }
}

/// Runs explicit RWMH warmup. `None` runs fixed-scale burn-in.
/// Zero iterations leave the initial scale unchanged, including subnormal scales.
/// Errors stop at the last completed transition; the chain remains usable.
pub fn warmup_rwmh<T, M, R>(
    chain: &mut Rwmh<'_, T, M>,
    iterations: usize,
    target: Option<AcceptanceTarget>,
    rng: &mut R,
) -> Result<WarmupSummary, RwmhError<T::Error>>
where
    T: LogDensity + ?Sized,
    M: EuclideanMetric,
    R: rand::Rng + ?Sized,
{
    let mut adapter = target.map(|target| DualAveraging::new(chain.options().step_size(), target));
    let mut accepted = 0;
    for _ in 0..iterations {
        let transition = chain.step(rng)?;
        accepted += usize::from(transition.accepted);
        if let Some(adapter) = &mut adapter {
            chain.set_step_size(
                adapter.update(transition.acceptance_probability).expect(
                    "RWMH produces a finite probability and iterations are bounded by usize",
                ),
            );
        }
    }
    if iterations > 0
        && let Some(adapter) = adapter
    {
        chain.set_step_size(adapter.finish());
    }
    Ok(WarmupSummary {
        iterations,
        accepted,
        step_size: chain.options().step_size(),
    })
}
