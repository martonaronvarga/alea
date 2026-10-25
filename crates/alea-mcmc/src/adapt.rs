//! Explicit step-size warmup, separate from stationary sampling.
//! RWMH scale adaptation and windowed HMC warmup, separate from retained draws.
#![deny(missing_docs)]
mod chees;
mod covariance;
// Research APIs have their own documentation and release schedule.
#[allow(missing_docs)]
mod fisher;
// Research APIs have their own documentation and release schedule.
mod isokinetic;
#[allow(missing_docs)]
mod joint;
mod joint_nuts;
// Research APIs have their own documentation and release schedule.
#[allow(missing_docs)]
mod manifold;
mod nuts;
mod pathfinder;
mod static_warmup;
mod step;
mod transport;
mod window;
pub use crate::atlas::{CurvatureProbeConfigError, CurvatureProbeError, CurvatureProbeOptions};
use crate::{
    Rwmh,
    config::{AcceptanceTarget, StepSize},
    rwmh::RwmhError,
};
use alea_core::density::LogDensity;
use alea_math::metric::EuclideanMetric;
pub use chees::{
    CheesConfigError, CheesCriterion, CheesEnsembleAdapter, CheesObservation, CheesPlan,
    CheesUpdate, FrozenCheesParameters,
};
pub use covariance::{CovarianceError, MetricKind, OnlineCovariance, WarmupMetric};
pub use fisher::{
    FisherError, FisherHmcWarmup, FisherMetricAdapter, FisherNutsWarmup, FisherNutsWarmupError,
    FisherNutsWarmupResult, FisherOptions, FisherWarmupError, FisherWarmupResult,
    MetricInitialization, WeightedFisherMoments,
};
pub use isokinetic::{
    MclmcWarmupOptions, MclmcWarmupReport, MclmcWarmupResult, warmup_mams_step_size,
    warmup_mclmc_step_size,
};
pub use joint::{
    JointHmcWarmup, JointWarmupError, JointWarmupOptions, JointWarmupReport, JointWarmupResult,
};
pub use joint_nuts::{
    JointNutsWarmup, JointNutsWarmupError, JointNutsWarmupReport, JointNutsWarmupResult,
};
pub use manifold::{
    FrozenManifoldParameters, ManifoldBayesTuner, ManifoldObservation, ManifoldTunerOptions,
    ManifoldTuningError, RmhmcManifoldWarmup, wang_diminishing_probability,
};
pub use nuts::{
    ControlledNutsWarmupResult, NutsWarmup, NutsWarmupError, NutsWarmupObservation,
    NutsWarmupReport, NutsWarmupResult, find_reasonable_nuts_step_size,
};
pub use pathfinder::{
    ImportanceResampling, MultiPathfinderOptions, PathfinderConfigError, PathfinderError,
    PathfinderPathReport, PathfinderPathStatus, PathfinderReport, PathfinderResult, PsisStatus,
    ScoreInitializationOptions, multi_pathfinder, score_initialize,
};
pub use static_warmup::{
    StaticWarmupError, StaticWarmupOptions, StaticWarmupReport, StaticWarmupResult,
    warmup_drghmc_step_size, warmup_ghmc_step_size, warmup_hmc_step_size,
};
pub use step::{DiminishingSchedule, StepAdaptation, StepObservation, StepSizeController};
pub use transport::{
    AffineTransportLearner, TransportLearningError, TransportLearningOptions, TransportWarmupError,
    TransportWarmupReport, TransportWarmupResult, learn_affine_transport,
};
pub use window::{
    ControlledHmcWarmupResult, HmcWarmup, HmcWarmupReport, HmcWarmupResult, SearchOptions,
    SearchResult, WarmupConfigError, WarmupError, WarmupOptions, WarmupStage, WindowSchedule,
    find_reasonable_step_size,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
/// Invalid adaptation observation, configuration or arithmetic.
pub enum AdaptationError {
    #[error("invalid diminishing schedule or optimizer configuration")]
    /// Invalid diminishing schedule or optimizer configuration.
    Configuration,
    #[error("backend failures cannot update adaptation")]
    /// Backend failures cannot update adaptation.
    BackendFailure,
    #[error("nonfinite step adaptation arithmetic")]
    /// Nonfinite step adaptation arithmetic.
    Numerical,
    #[error("acceptance probability must be finite and between zero and one")]
    /// Acceptance probability must be finite and between zero and one.
    AcceptanceProbability,
    #[error("adaptation iteration count overflow")]
    /// Adaptation iteration count overflow.
    IterationOverflow,
}

/// Dual averaging with gamma=0.05, t0=10, kappa=0.75.
/// Scales are clamped to the finite normal range in log space.
#[derive(Debug, Clone)]
#[must_use = "dual averaging has no effect unless it is updated and finished"]
pub struct DualAveraging {
    mu: f64,
    h_bar: f64,
    log_average: f64,
    iterations: usize,
    target: AcceptanceTarget,
}
impl DualAveraging {
    /// Constructs dual averaging from a validated initial scale and acceptance target.
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
        self.update_inner(acceptance, None)
    }
    fn update_capped(
        &mut self,
        acceptance: f64,
        maximum: StepSize,
    ) -> Result<StepSize, AdaptationError> {
        self.update_inner(acceptance, Some(maximum))
    }
    fn update_inner(
        &mut self,
        acceptance: f64,
        maximum: Option<StepSize>,
    ) -> Result<StepSize, AdaptationError> {
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
        let mut log_step = self.mu - m.sqrt() / 0.05 * self.h_bar;
        if let Some(maximum) = maximum {
            log_step = log_step.min(maximum.value().ln());
        }
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

#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
/// Completed RWMH warmup counts and the resulting fixed scale.
pub struct WarmupSummary {
    /// Number of completed warmup transitions.
    pub iterations: usize,
    /// Number of accepted proposals.
    pub accepted: usize,
    /// Validated proposal or integration step size.
    pub step_size: StepSize,
}
impl WarmupSummary {
    #[must_use]
    /// Fraction of accepted warmup proposals, or zero for an empty warmup.
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
///
/// # Errors
/// Propagates [`RwmhError`] from the first failed transition.
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
