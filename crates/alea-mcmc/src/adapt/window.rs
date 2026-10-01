//! Stan-style window scheduling and an owning warmup-to-sampling boundary.
use super::{
    AdaptationError, CovarianceError, DualAveraging, MetricKind, OnlineCovariance, WarmupMetric,
};
use crate::integrator::{Integrator, Leapfrog};
use crate::{
    Hmc, HmcOptions,
    config::{AcceptanceTarget, StepSize},
    hmc::HmcError,
};
use alea_core::target::LogDensityGradient;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric},
};
use rand::Rng;
use std::error::Error;

/// A transition's adaptation stage. Only slow stages contribute covariance data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupStage {
    /// Only adapt step size while approaching the typical set.
    InitialFast,
    /// Adapt step size and accumulate covariance; replace metric at the boundary.
    Slow { window_end: bool },
    /// Retune step size with the last, frozen metric.
    FinalFast,
}

/// Constant-space Stan/BlackJAX default schedule (75/25/50, doubling windows).
/// Below 20 transitions only step size is adapted. Below 150, buffers become
/// floor(15% n), floor(10% n), with one remaining slow window.
#[derive(Debug, Clone)]
pub struct WindowSchedule {
    total: usize,
    next: usize,
    initial: usize,
    final_start: usize,
    window_end: usize,
    window_size: usize,
}
impl WindowSchedule {
    /// Construct the default schedule without allocating, including for zero steps.
    pub fn new(total: usize) -> Self {
        let (initial, final_size, size) = if total < 20 {
            (total, 0, 0)
        } else if total < 150 {
            let initial = total * 15 / 100;
            let final_size = total / 10;
            (initial, final_size, total - initial - final_size)
        } else {
            (75, 50, 25)
        };
        let final_start = total - final_size;
        let remaining = final_start - initial;
        let size = if size <= remaining / 3 {
            size
        } else {
            remaining
        };
        Self {
            total,
            next: 0,
            initial,
            final_start,
            window_end: initial + size,
            window_size: size,
        }
    }
}
impl Iterator for WindowSchedule {
    type Item = WarmupStage;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.total {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(if index < self.initial {
            WarmupStage::InitialFast
        } else if index >= self.final_start {
            WarmupStage::FinalFast
        } else {
            let window_end = self.next == self.window_end;
            if window_end && self.next < self.final_start {
                let remaining = self.final_start - self.next;
                let size = self.window_size.saturating_mul(2);
                self.window_size = if size <= remaining / 3 {
                    size
                } else {
                    remaining
                };
                self.window_end = self.next + self.window_size;
            }
            WarmupStage::Slow { window_end }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.next;
        (remaining, Some(remaining))
    }
}
impl ExactSizeIterator for WindowSchedule {}
impl std::iter::FusedIterator for WindowSchedule {}

/// Validated limits for doubling/halving one-step acceptance probes.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    min: StepSize,
    max: StepSize,
    max_probes: usize,
    target: AcceptanceTarget,
}
impl SearchOptions {
    /// Rejects invalid bounds and fewer than two probes before evaluating a target.
    pub fn new(
        min: f64,
        max: f64,
        max_probes: usize,
        target: f64,
    ) -> Result<Self, WarmupConfigError> {
        let min = StepSize::try_from(min).map_err(|_| WarmupConfigError::Search)?;
        let max = StepSize::try_from(max).map_err(|_| WarmupConfigError::Search)?;
        if min.value() >= max.value() || max_probes < 2 {
            return Err(WarmupConfigError::Search);
        }
        let target = AcceptanceTarget::try_from(target).map_err(|_| WarmupConfigError::Search)?;
        Ok(Self {
            min,
            max,
            max_probes,
            target,
        })
    }
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self::new(1e-12, 1e6, 80, 0.8).expect("valid search defaults")
    }
}

/// Invalid user-provided warmup configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WarmupConfigError {
    #[error("invalid step-size search bounds, probe count, or acceptance target")]
    Search,
}

/// Warmup policy; no settings can become invalid after construction.
#[derive(Debug, Clone, Copy)]
pub struct WarmupOptions {
    iterations: usize,
    kind: MetricKind,
    target: AcceptanceTarget,
    search: Option<SearchOptions>,
}
impl WarmupOptions {
    /// Default diagonal adaptation, acceptance target 0.8 and bounded search.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            kind: MetricKind::Diagonal,
            target: 0.8.try_into().expect("valid acceptance default"),
            search: Some(SearchOptions::default()),
        }
    }
    #[must_use]
    pub fn with_metric(mut self, kind: MetricKind) -> Self {
        self.kind = kind;
        self
    }
    #[must_use]
    pub fn with_target(mut self, target: AcceptanceTarget) -> Self {
        self.target = target;
        self
    }
    /// `None` uses the supplied initial step and resets dual averaging to the
    /// smoothed step after each metric update (BlackJAX 1.5 behavior).
    #[must_use]
    pub fn with_search(mut self, search: Option<SearchOptions>) -> Self {
        self.search = search;
        self
    }
}

/// Explicit errors; failure never returns a partially adapted sampling chain.
#[derive(Debug, thiserror::Error)]
pub enum WarmupError<E: Error + 'static> {
    #[error(transparent)]
    Hmc(#[from] HmcError<E>),
    #[error(transparent)]
    Covariance(#[from] CovarianceError),
    #[error(transparent)]
    Adaptation(#[from] AdaptationError),
    #[error("step-size search did not bracket acceptance within its limits")]
    SearchExhausted,
    #[error("warmup evaluation counter overflow")]
    CounterOverflow,
}

/// An acceptable bracket endpoint and the number of momentum/trajectory probes.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub step_size: StepSize,
    pub probes: usize,
}

/// Search using one complete macrostep per probe with fresh momentum. Each probe starts
/// at the same live point and does not commit it or alter the chain's options.
/// Returns the acceptable side of the bracket. RNG is consumed, not rolled back.
/// Numerical divergences count as zero acceptance; model/backend errors propagate.
/// Exhaustion is an explicit error, never an unbounded loop or silent fallback.
pub fn find_reasonable_step_size<T, M, R, I>(
    chain: &mut Hmc<'_, T, M, I>,
    options: SearchOptions,
    rng: &mut R,
) -> Result<SearchResult, WarmupError<T::Error>>
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
            return Err(WarmupError::SearchExhausted);
        }
        let acceptable = chain.probe(candidate.try_into().expect("bounded step"), rng)?
            >= options.target.value();
        if acceptable != increasing {
            let value = if increasing { step } else { candidate };
            return Ok(SearchResult {
                step_size: value.try_into().expect("bounded step"),
                probes,
            });
        }
        step = candidate;
    }
    Err(WarmupError::SearchExhausted)
}

/// Counters include warmup only. No draws are retained by the controller.
#[derive(Debug, Clone, Copy)]
pub struct HmcWarmupReport {
    pub iterations: usize,
    pub accepted: usize,
    pub divergences: usize,
    pub metric_updates: usize,
    pub search_probes: usize,
    /// Attempted macrosteps, including search; excludes the initial evaluation.
    /// This is not an actual gradient-call count, especially after partial failures.
    pub integration_attempts: usize,
    pub step_size: StepSize,
}

/// Successful warmup returns a fixed kernel and its warmup-only counters.
pub type HmcWarmupResult<'a, T, I = Leapfrog> = Result<
    (Hmc<'a, T, WarmupMetric, I>, HmcWarmupReport),
    WarmupError<<T as LogDensityGradient>::Error>,
>;

/// Owning, non-sampling warmup controller. Only successful completion returns a
/// fixed [`Hmc`]; it contains no adapter and implements the runtime chain protocol.
/// Observation updates and ordinary transitions allocate nothing after construction.
/// Metric construction allocates at slow-window boundaries only.
///
/// ```
/// use alea_distributions::Gaussian;
/// use alea_math::buffer::OwnedBuffer;
/// use alea_mcmc::{HmcOptions, adapt::{HmcWarmup, WarmupOptions}};
/// use rand::{SeedableRng, rngs::SmallRng};
/// let target = Gaussian::new(2);
/// let warmup = HmcWarmup::new(&target, OwnedBuffer::new(2),
///     HmcOptions::new(0.1, 5)?, WarmupOptions::new(200))?;
/// let mut rng = SmallRng::seed_from_u64(7);
/// let (mut chain, report) = warmup.run(&mut rng)?;
/// let transition = chain.step(&mut rng)?;
/// assert_eq!(chain.options().step_size(), report.step_size);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// Warmup itself cannot be passed to a retained-draw runner:
/// ```compile_fail
/// use alea_core::target::LogDensityGradient;
/// use alea_mcmc::{MarkovChain, adapt::HmcWarmup};
/// fn sample<C: MarkovChain>(_: C) {}
/// fn unfinished<T: LogDensityGradient>(warmup: HmcWarmup<'_, T>) {
///     sample(warmup);
/// }
/// ```
#[derive(Debug)]
#[must_use = "warmup must run to completion before retained sampling"]
pub struct HmcWarmup<'a, T: LogDensityGradient + ?Sized, I = Leapfrog> {
    chain: Hmc<'a, T, WarmupMetric, I>,
    covariance: Option<OnlineCovariance>,
    options: WarmupOptions,
}
impl<'a, T: LogDensityGradient + ?Sized> HmcWarmup<'a, T> {
    /// Construct reference leapfrog warmup.
    /// # Errors
    /// Propagates invalid dimensions, initial evaluation or storage errors.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        hmc: HmcOptions,
        options: WarmupOptions,
    ) -> Result<Self, WarmupError<T::Error>> {
        Self::new_with_integrator(target, position, hmc, options, Leapfrog)
    }
}
impl<'a, T: LogDensityGradient + ?Sized, I: Integrator> HmcWarmup<'a, T, I> {
    /// Construct the initial kernel and its reusable adaptation workspace.
    /// No covariance storage is allocated for fewer than 20 transitions.
    ///
    /// # Errors
    /// Rejects invalid target/position dimensions, initial evaluation failures,
    /// and covariance storage overflow. Allocation exhaustion follows Rust's allocator.
    pub fn new_with_integrator(
        target: &'a T,
        position: OwnedBuffer,
        hmc: HmcOptions,
        options: WarmupOptions,
        integrator: I,
    ) -> Result<Self, WarmupError<T::Error>> {
        // HMC validates supplied position shape before covariance allocation.
        let chain = Hmc::new_with_integrator(
            target,
            position,
            WarmupMetric::Identity(IdentityMetric::new(target.dimension())),
            hmc,
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
        })
    }

    /// Consume warmup, freeze the smoothed step, and discard all adaptation state.
    /// Zero iterations preserve the initial step and consume no RNG. On failure
    /// no chain is published; the caller still owns its target and RNG.
    ///
    /// # Errors
    /// Propagates backend errors, failed bounded search, unrepresentable covariance
    /// or metric estimates, and counter overflow. Numerical trajectory divergences
    /// are ordinary rejected states and contribute zero acceptance to adaptation.
    pub fn run<R: Rng + ?Sized>(mut self, rng: &mut R) -> HmcWarmupResult<'a, T, I> {
        let mut report = HmcWarmupReport {
            iterations: self.options.iterations,
            accepted: 0,
            divergences: 0,
            metric_updates: 0,
            search_probes: 0,
            integration_attempts: 0,
            step_size: self.chain.options().step_size(),
        };
        if self.options.iterations == 0 {
            return Ok((self.chain, report));
        }
        self.search(rng, &mut report)?;
        let mut adapter = DualAveraging::new(self.chain.options().step_size(), self.options.target);
        for stage in WindowSchedule::new(self.options.iterations) {
            let transition = self.chain.step(rng)?;
            report.accepted += usize::from(transition.accepted);
            report.divergences += usize::from(transition.divergence.is_some());
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(WarmupError::CounterOverflow)?;
            let step = adapter.update(transition.acceptance_probability)?;
            if let WarmupStage::Slow { window_end } = stage {
                let covariance = self
                    .covariance
                    .as_mut()
                    .expect("slow schedule has covariance storage");
                covariance.update(self.chain.point().position())?;
                if window_end {
                    // Build/validate entirely before publishing the replacement.
                    let metric = covariance.metric()?;
                    covariance.reset();
                    self.chain.replace_metric(metric)?;
                    self.chain.set_step_size(adapter.finish());
                    self.search(rng, &mut report)?;
                    adapter =
                        DualAveraging::new(self.chain.options().step_size(), self.options.target);
                    report.metric_updates += 1;
                    continue;
                }
            }
            self.chain.set_step_size(step);
        }
        self.chain.set_step_size(adapter.finish());
        report.step_size = self.chain.options().step_size();
        Ok((self.chain, report))
    }

    fn search<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        report: &mut HmcWarmupReport,
    ) -> Result<(), WarmupError<T::Error>> {
        if let Some(options) = self.options.search {
            let result = find_reasonable_step_size(&mut self.chain, options, rng)?;
            report.search_probes = report
                .search_probes
                .checked_add(result.probes)
                .ok_or(WarmupError::CounterOverflow)?;
            report.integration_attempts = report
                .integration_attempts
                .checked_add(result.probes)
                .ok_or(WarmupError::CounterOverflow)?;
            self.chain.set_step_size(result.step_size);
        }
        Ok(())
    }
}
