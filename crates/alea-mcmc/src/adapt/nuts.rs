//! Native NUTS window adaptation, separate from frozen retained sampling.
use std::error::Error;

use alea_core::target::LogDensityGradient;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric},
};
use rand::Rng;

use super::{
    AdaptationError, CovarianceError, DualAveraging, OnlineCovariance, SearchOptions, SearchResult,
    WarmupMetric, WarmupOptions, WarmupStage, WindowSchedule,
};
use crate::{
    Nuts, NutsOptions, NutsTransition,
    config::StepSize,
    integrator::{Integrator, Leapfrog},
    nuts::{NutsError, NutsTermination},
};
mod gradient;

/// Failure never publishes a partly adapted sampling chain.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NutsWarmupError<E: Error + 'static> {
    /// Original typed kernel, target or metric error.
    #[error(transparent)]
    Nuts(#[from] NutsError<E>),
    /// Covariance construction failed without installing a partial estimate.
    #[error(transparent)]
    Covariance(#[from] CovarianceError),
    /// Invalid acceptance observation or unrepresentable controller arithmetic.
    #[error(transparent)]
    Adaptation(#[from] AdaptationError),
    /// No acceptable bracket was found within the declared finite work budget.
    #[error("NUTS step-size search did not bracket acceptance within its limits")]
    SearchExhausted,
    /// Integration/search counters cannot be represented.
    #[error("NUTS warmup counter overflow")]
    CounterOverflow,
}

/// Warmup-only observations; none imply convergence or a valid typical-set start.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct NutsWarmupReport {
    /// Completed transitions, excluding search probes.
    pub iterations: usize,
    /// Transitions selecting a non-origin phase, not Metropolis acceptance count.
    pub moved: usize,
    /// Transitions stopped by numerical divergence, possibly with a valid candidate.
    pub divergences: usize,
    /// Transitions reaching the configured maximum depth.
    pub depth_saturations: usize,
    /// Successfully fitted geometry updates, excluding gradient-score initialization.
    pub metric_updates: usize,
    /// One-macrostep search proposals.
    pub search_probes: usize,
    /// Attempted macrosteps, including probes; not an actual gradient-call count.
    pub integration_attempts: usize,
    /// Step updates limited by the explicit search bounds, or default search bounds.
    pub bounded_step_updates: usize,
    /// Frozen integration step.
    pub step_size: StepSize,
    /// Initial finite unconstrained log density.
    pub initial_log_density: f64,
    /// Final finite unconstrained log density.
    pub final_log_density: f64,
    /// Smallest finite log density of an observed live warmup state.
    pub minimum_log_density: f64,
    /// Largest finite log density of an observed live warmup state.
    pub maximum_log_density: f64,
}

/// Borrowed post-transition observation. Retaining positions requires explicit copying.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct NutsWarmupObservation<'a> {
    /// One-based completed transition count.
    pub completed: usize,
    /// Stage that generated this transition.
    pub stage: WarmupStage,
    /// Coherent live position, including repeats after rejection.
    pub position: &'a [f64],
    /// Log density at that same position.
    pub log_density: f64,
    /// Step used for this transition, before the next adaptation update.
    pub step_size: StepSize,
    /// Full tree acceptance and work statistics.
    pub transition: NutsTransition,
}

/// A complete frozen NUTS chain and its warmup report.
pub type NutsWarmupResult<'a, T, I = Leapfrog> = Result<
    (Nuts<'a, T, WarmupMetric, I>, NutsWarmupReport),
    NutsWarmupError<<T as LogDensityGradient>::Error>,
>;
/// Cancellation discards the controller without publishing a sampling chain.
pub type ControlledNutsWarmupResult<'a, T, I = Leapfrog> = Result<
    Option<(Nuts<'a, T, WarmupMetric, I>, NutsWarmupReport)>,
    NutsWarmupError<<T as LogDensityGradient>::Error>,
>;

/// Bounded one-macrostep search with fresh momentum at the same live point.
/// Finite energy decreases are acceptable, including decreases beyond the energy
/// cutoff. Returns the acceptable side of the bracket. Settings/cache are unchanged;
/// target counters and RNG consumption are not rolled back.
///
/// # Errors
/// Propagates concrete kernel/backend failures and reports bounded exhaustion.
pub fn find_reasonable_nuts_step_size<T, M, R, I>(
    chain: &mut Nuts<'_, T, M, I>,
    options: SearchOptions,
    rng: &mut R,
) -> Result<SearchResult, NutsWarmupError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    R: Rng + ?Sized,
    I: Integrator,
{
    let mut step = chain
        .options()
        .step_size()
        .value()
        .clamp(options.min.value(), options.max.value());
    let increasing =
        chain.probe(step.try_into().expect("bounded step"), rng)? >= options.target.value();
    for probes in 2..=options.max_probes {
        let candidate = if increasing { step * 2.0 } else { step * 0.5 }
            .clamp(options.min.value(), options.max.value());
        if candidate == step {
            return Err(NutsWarmupError::SearchExhausted);
        }
        let acceptable = chain.probe(candidate.try_into().expect("bounded step"), rng)?
            >= options.target.value();
        if acceptable != increasing {
            return Ok(SearchResult {
                step_size: (if increasing { step } else { candidate })
                    .try_into()
                    .expect("bounded step"),
                probes,
            });
        }
        step = candidate;
    }
    Err(NutsWarmupError::SearchExhausted)
}

/// Consuming covariance/dual-averaging adaptation using actual NUTS transitions.
/// Metric fitting is cold window-boundary work. The returned kernel contains no
/// estimator, observer or controller. Target dimension and geometry stay fixed
/// within each trajectory. Zero warmup consumes no RNG and preserves the initial step.
#[derive(Debug)]
#[must_use = "warmup must finish before retained sampling"]
pub struct NutsWarmup<'a, T: LogDensityGradient + ?Sized, I = Leapfrog> {
    pub(super) chain: Nuts<'a, T, WarmupMetric, I>,
    pub(super) covariance: Option<OnlineCovariance>,
    options: WarmupOptions,
    gradient: Option<gradient::Estimate>,
    gradient_final_restart: bool,
}

impl<'a, T: LogDensityGradient + ?Sized> NutsWarmup<'a, T> {
    /// Construct leapfrog NUTS warmup with identity initial geometry.
    /// # Errors
    /// Rejects initial target/shape errors or unaddressable covariance/tree storage.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        nuts: NutsOptions,
        options: WarmupOptions,
    ) -> Result<Self, NutsWarmupError<T::Error>> {
        Self::new_with_integrator(target, position, nuts, options, Leapfrog)
    }
}

impl<'a, T: LogDensityGradient + ?Sized, I: Integrator> NutsWarmup<'a, T, I> {
    /// Construct with a fixed sealed symmetric integrator.
    /// # Errors
    /// Same initial-state and storage errors as [`Self::new`].
    pub fn new_with_integrator(
        target: &'a T,
        position: OwnedBuffer,
        nuts: NutsOptions,
        options: WarmupOptions,
        integrator: I,
    ) -> Result<Self, NutsWarmupError<T::Error>> {
        let chain = Nuts::new_with_integrator(
            target,
            position,
            WarmupMetric::Identity(IdentityMetric::new(target.dimension())),
            nuts,
            integrator,
        )?;
        let covariance = if options.iterations >= 20 {
            Some(OnlineCovariance::new(
                chain.point().dimension(),
                options.kind,
            )?)
        } else {
            None
        };
        Ok(Self {
            chain,
            covariance,
            options,
            gradient: None,
            gradient_final_restart: true,
        })
    }

    /// Select gradient-informed diagonal geometry instead of covariance windows.
    /// This replaces the configured metric kind with diagonal estimation. Paired
    /// live-state moments include repeated states, use growing background windows,
    /// and freeze after 85% of warmup. Late metric tuning uses symmetric tree
    /// acceptance; final fixed-geometry tuning restarts search and uses ordinary
    /// tree acceptance. The returned NUTS
    /// kernel has no adaptive state. Zero warmup
    /// preserves the supplied step and identity metric without consuming RNG.
    /// # Errors
    /// Rejects unaddressable paired-moment storage before beginning adaptation.
    pub fn with_gradient_diagonal(mut self) -> Result<Self, NutsWarmupError<T::Error>> {
        self.gradient = Some(gradient::Estimate::new(self.chain.point().dimension())?);
        self.covariance = None;
        self.options.kind = super::MetricKind::Diagonal;
        Ok(self)
    }

    /// Configure the final search/dual-averaging restart of gradient diagonal
    /// warmup. Enabled by default. Disabling it retains the continuous adaptation
    /// history and late symmetric acceptance, as in nuts-rs. Geometry still
    /// freezes at 85%; covariance warmup keeps its own window restarts.
    pub fn with_gradient_final_restart(mut self, enabled: bool) -> Self {
        self.gradient_final_restart = enabled;
        self
    }

    /// Freeze a complete chain; no warmup state escapes on error.
    /// # Errors
    /// Propagates target, covariance, step-search and counter errors.
    pub fn run<R: Rng + ?Sized>(self, rng: &mut R) -> NutsWarmupResult<'a, T, I> {
        self.drive(rng, |_| true, |_| {})
            .map(|out| out.expect("unconditional warmup never cancels"))
    }

    /// Observe completed states without retaining them inside the controller.
    /// The callback runs once per completed transition before metric updates.
    /// A callback panic publishes no chain. It does not roll back target/RNG effects.
    /// # Errors
    /// Same numerical/backend failures as [`Self::run`].
    pub fn run_observed<R: Rng + ?Sized>(
        self,
        rng: &mut R,
        observe: impl FnMut(NutsWarmupObservation<'_>),
    ) -> NutsWarmupResult<'a, T, I> {
        self.drive(rng, |_| true, observe)
            .map(|out| out.expect("unconditional warmup never cancels"))
    }

    /// Cancel before search or at a completed-transition boundary. A search or
    /// trajectory already in progress finishes before cancellation is observed.
    /// `false` returns `None` without a partly adapted or restartable chain.
    /// # Errors
    /// Same numerical/backend failures as [`Self::run`].
    pub fn run_controlled<R: Rng + ?Sized>(
        self,
        rng: &mut R,
        proceed: impl FnMut(usize) -> bool,
    ) -> ControlledNutsWarmupResult<'a, T, I> {
        self.drive(rng, proceed, |_| {})
    }

    fn drive<R: Rng + ?Sized>(
        mut self,
        rng: &mut R,
        mut proceed: impl FnMut(usize) -> bool,
        mut observe: impl FnMut(NutsWarmupObservation<'_>),
    ) -> ControlledNutsWarmupResult<'a, T, I> {
        let initial = self.chain.point().log_density();
        let mut report = NutsWarmupReport {
            iterations: 0,
            moved: 0,
            divergences: 0,
            depth_saturations: 0,
            metric_updates: 0,
            search_probes: 0,
            integration_attempts: 0,
            bounded_step_updates: 0,
            step_size: self.chain.options().step_size(),
            initial_log_density: initial,
            final_log_density: initial,
            minimum_log_density: initial,
            maximum_log_density: initial,
        };
        if self.options.iterations == 0 {
            return Ok(Some((self.chain, report)));
        }
        if !proceed(0) {
            return Ok(None);
        }
        if let Some(estimate) = self.gradient.take() {
            return self.drive_gradient(estimate, rng, proceed, observe, report);
        }
        self.search(rng, &mut report)?;
        let mut adapter = DualAveraging::new(self.chain.options().step_size(), self.options.target);
        for stage in WindowSchedule::new(self.options.iterations) {
            if !proceed(report.iterations) {
                return Ok(None);
            }
            let transition = self.chain.step(rng)?;
            let step_size = transition.step_size;
            report.iterations += 1;
            report.moved += usize::from(transition.accepted);
            report.divergences += usize::from(matches!(
                transition.termination,
                NutsTermination::Divergence(_)
            ));
            report.depth_saturations +=
                usize::from(transition.termination == NutsTermination::MaxDepth);
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(NutsWarmupError::CounterOverflow)?;
            report.final_log_density = self.chain.point().log_density();
            report.minimum_log_density = report.minimum_log_density.min(report.final_log_density);
            report.maximum_log_density = report.maximum_log_density.max(report.final_log_density);
            observe(NutsWarmupObservation {
                completed: report.iterations,
                stage,
                position: self.chain.point().position(),
                log_density: report.final_log_density,
                step_size,
                transition,
            });
            if !proceed(report.iterations) {
                return Ok(None);
            }
            let step = adapter.update(transition.acceptance_probability)?;
            if let WarmupStage::Slow { window_end } = stage {
                let covariance = self
                    .covariance
                    .as_mut()
                    .expect("slow stage has covariance storage");
                covariance.update(self.chain.point().position())?;
                if window_end {
                    let metric = covariance.metric()?;
                    self.chain.replace_metric(metric)?;
                    covariance.reset();
                    let step = self.bound(adapter.finish(), &mut report);
                    self.chain.set_step_size(step);
                    self.search(rng, &mut report)?;
                    adapter =
                        DualAveraging::new(self.chain.options().step_size(), self.options.target);
                    report.metric_updates += 1;
                    continue;
                }
            }
            let step = self.bound(step, &mut report);
            self.chain.set_step_size(step);
        }
        let step = self.bound(adapter.finish(), &mut report);
        self.chain.set_step_size(step);
        report.step_size = step;
        Ok(Some((self.chain, report)))
    }

    fn bound(&self, step: StepSize, report: &mut NutsWarmupReport) -> StepSize {
        let bounds = self.options.search.unwrap_or_default();
        let value = step.value().clamp(bounds.min.value(), bounds.max.value());
        report.bounded_step_updates += usize::from(value != step.value());
        value.try_into().expect("bounded positive step")
    }

    fn search<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        report: &mut NutsWarmupReport,
    ) -> Result<(), NutsWarmupError<T::Error>> {
        if let Some(options) = self.options.search {
            let result = find_reasonable_nuts_step_size(&mut self.chain, options, rng)?;
            report.search_probes = report
                .search_probes
                .checked_add(result.probes)
                .ok_or(NutsWarmupError::CounterOverflow)?;
            report.integration_attempts = report
                .integration_attempts
                .checked_add(result.probes)
                .ok_or(NutsWarmupError::CounterOverflow)?;
            self.chain.set_step_size(result.step_size);
        }
        Ok(())
    }
}
