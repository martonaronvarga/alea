//! Experimental paired-score Fisher warmup; covariance warmup remains the oracle.
use super::{
    DiminishingSchedule, HmcWarmupReport, SearchOptions, StepAdaptation, StepObservation,
    StepSizeController, WarmupError, find_reasonable_step_size,
};
mod weighted;
use crate::config::AcceptanceTarget;
use crate::integrator::{Integrator, Leapfrog};
use crate::{Hmc, HmcOptions};
use alea_core::target::LogDensityGradient;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{LowRankDiagonalMetric, MetricError},
};
use rand::Rng;
use std::error::Error;
pub use weighted::WeightedFisherMoments;

// Log of a positive sum, including a zero scatter represented by -infinity.
// The ridge argument is finite, so the larger log is always finite.
fn log_add_positive(a: f64, b: f64) -> f64 {
    let larger = a.max(b);
    larger + (a.min(b) - larger).exp().ln_1p()
}

/// Paired-estimator failures do not partially update the observation history.
#[derive(Debug, thiserror::Error)]
pub enum FisherError {
    #[error("invalid Fisher dimensions, window length, rank or ridge")]
    Configuration,
    #[error("non-finite or unrepresentable paired Fisher observation")]
    Numerical,
    #[error("Fisher observation count overflow")]
    CountOverflow,
    #[error(transparent)]
    Metric(#[from] MetricError),
    #[cfg(feature = "faer")]
    #[error(transparent)]
    Fit(#[from] alea_math::fisher::FisherFitError),
}

#[derive(Debug)]
struct Moments {
    count: usize,
    // q mean, score mean, q scatter, score scatter, each contiguous.
    values: OwnedBuffer,
    next: OwnedBuffer,
}
impl Moments {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            values: OwnedBuffer::new(4 * dim),
            next: OwnedBuffer::new(4 * dim),
        }
    }
    fn reset(&mut self) {
        self.count = 0;
        self.values.fill(0.0);
    }
    fn prepare(&mut self, q: &[f64], score: &[f64], reset: bool) -> Result<(), FisherError> {
        let count = if reset { 0 } else { self.count };
        let n = count.checked_add(1).ok_or(FisherError::CountOverflow)? as f64;
        let d = q.len();
        for (axis, data) in [q, score].into_iter().enumerate() {
            for (i, &value) in data.iter().enumerate() {
                let mean_index = axis * d + i;
                let scatter_index = (axis + 2) * d + i;
                let mean = if reset { 0.0 } else { self.values[mean_index] };
                let scatter = if reset {
                    0.0
                } else {
                    self.values[scatter_index]
                };
                let delta = value - mean;
                self.next[mean_index] = mean + delta / n;
                self.next[scatter_index] = scatter + delta * (delta * ((n - 1.0) / n));
            }
        }
        if self.next.iter().any(|v| !v.is_finite()) {
            return Err(FisherError::Numerical);
        }
        Ok(())
    }
    fn commit(&mut self, reset: bool) {
        self.count = if reset { 1 } else { self.count + 1 };
        std::mem::swap(&mut self.values, &mut self.next);
    }
}

/// Foreground/background paired Welford estimates with bounded optional history.
/// Every observation is an actual chain state and its cached log-density score,
/// including repeated states after rejection. No target calls or allocations.
#[derive(Debug)]
pub struct FisherMetricAdapter {
    dim: usize,
    period: usize,
    seen: usize,
    foreground: Moments,
    background: Moments,
    positions: OwnedBuffer,
    scores: OwnedBuffer,
    retained: usize,
    cursor: usize,
}
impl FisherMetricAdapter {
    /// Allocate O(d) moments and optional O(d * capacity) paired ring storage.
    /// `capacity=0` is appropriate for diagonal-only adaptation.
    ///
    /// # Errors
    /// Rejects zero dimension/period and storage arithmetic overflow.
    pub fn new(dim: usize, period: usize, capacity: usize) -> Result<Self, FisherError> {
        let max = isize::MAX as usize / size_of::<f64>();
        let history = dim
            .checked_mul(capacity)
            .filter(|&n| n <= max)
            .ok_or(FisherError::Configuration)?;
        if dim == 0 || period == 0 || dim > max / 4 {
            return Err(FisherError::Configuration);
        }
        Ok(Self {
            dim,
            period,
            seen: 0,
            foreground: Moments::new(dim),
            background: Moments::new(dim),
            positions: OwnedBuffer::new(history),
            scores: OwnedBuffer::new(history),
            retained: 0,
            cursor: 0,
        })
    }
    /// Reset the transient estimators/ring without reallocating storage.
    /// # Errors
    /// A zero period is invalid and leaves the estimator unchanged.
    pub fn reset(&mut self, period: usize) -> Result<(), FisherError> {
        if period == 0 {
            return Err(FisherError::Configuration);
        }
        self.period = period;
        self.seen = 0;
        self.foreground.reset();
        self.background.reset();
        self.retained = 0;
        self.cursor = 0;
        Ok(())
    }
    /// Validate and atomically incorporate one paired observation.
    /// # Errors
    /// Shape, non-finite input, overflow or non-finite moments leave history intact.
    pub fn observe(&mut self, q: &[f64], score: &[f64]) -> Result<(), FisherError> {
        if q.len() != self.dim || score.len() != self.dim {
            return Err(FisherError::Configuration);
        }
        if q.iter().chain(score).any(|v| !v.is_finite()) {
            return Err(FisherError::Numerical);
        }
        let seen = self.seen.checked_add(1).ok_or(FisherError::CountOverflow)?;
        let boundary = self.seen > 0 && self.seen.is_multiple_of(self.period);
        let swap = boundary && self.seen / self.period >= 2;
        // Prepare both updates first. At boundaries the old background becomes
        // foreground; the old foreground becomes a freshly reset background.
        if swap {
            self.background.prepare(q, score, false)?;
            self.foreground.prepare(q, score, true)?;
            self.background.commit(false);
            self.foreground.commit(true);
            std::mem::swap(&mut self.foreground, &mut self.background);
        } else {
            self.foreground.prepare(q, score, false)?;
            self.background.prepare(q, score, boundary)?;
            self.foreground.commit(false);
            self.background.commit(boundary);
        }
        self.seen = seen;
        let capacity = (self.positions.len() / self.dim).min(self.period.saturating_mul(2));
        if capacity > 0 {
            let range = self.cursor * self.dim..(self.cursor + 1) * self.dim;
            self.positions[range.clone()].copy_from_slice(q);
            self.scores[range].copy_from_slice(score);
            self.cursor = (self.cursor + 1) % capacity;
            self.retained = (self.retained + 1).min(capacity);
        }
        Ok(())
    }
    /// Effective number of observations in the foreground estimate.
    pub fn count(&self) -> usize {
        self.foreground.count
    }
    /// Write `sigma_i = (C_ii/F_ii)^(1/4)` without allocating.
    /// A ridge in coordinates standardized by `fallback` handles constant draws.
    /// Scales are explicitly bounded to [1e-10,1e10].
    /// # Errors
    /// Rejects shapes/ridge/invalid fallback or unrepresentable moments. Output
    /// contents are unspecified on numerical failure; the estimator is unchanged.
    pub fn scales_into(
        &self,
        fallback: &[f64],
        ridge: f64,
        out: &mut [f64],
    ) -> Result<(), FisherError> {
        if fallback.len() != self.dim || out.len() != self.dim || !ridge.is_finite() || ridge <= 0.0
        {
            return Err(FisherError::Configuration);
        }
        for i in 0..self.dim {
            let scale = fallback[i];
            if !scale.is_finite() || !(1e-10..=1e10).contains(&scale) {
                return Err(FisherError::Configuration);
            }
            // Standardized scatters may overflow/underflow even when their
            // regularized ratio and the final bounded scale are representable.
            let log_scale = scale.ln();
            let log_ridge = ridge.ln();
            let c = log_add_positive(
                self.foreground.values[2 * self.dim + i].ln() - 2.0 * log_scale,
                log_ridge,
            );
            let f = log_add_positive(
                self.foreground.values[3 * self.dim + i].ln() + 2.0 * log_scale,
                log_ridge,
            );
            let sigma = (log_scale + 0.25 * (c - f))
                .clamp(1e-10_f64.ln(), 1e10_f64.ln())
                .exp();
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(FisherError::Numerical);
            }
            out[i] = sigma.clamp(1e-10, 1e10);
        }
        Ok(())
    }
    /// Cold-path spectral fit from the bounded paired history.
    /// # Errors
    /// Requires at least two retained samples and valid fitting parameters.
    #[cfg(feature = "faer")]
    pub fn fit_low_rank(
        &self,
        scales: &[f64],
        ridge: f64,
        threshold: f64,
        max_rank: usize,
    ) -> Result<LowRankDiagonalMetric, FisherError> {
        if scales.len() != self.dim {
            return Err(FisherError::Configuration);
        }
        Ok(alea_math::fisher::fit_low_rank(
            &self.positions[..self.retained * self.dim],
            &self.scores[..self.retained * self.dim],
            scales,
            ridge,
            threshold,
            max_rank,
        )?)
    }
}

/// CPU Fisher schedule: default periods 10 then 80, geometry frozen for the
/// final 15%; dual averaging is the reference step controller. Alternative
/// weighted moments and scalar optimizers are explicit warmup-only experiments.
#[derive(Debug, Clone, Copy)]
pub struct FisherOptions {
    iterations: usize,
    max_rank: usize,
    initialization: MetricInitialization,
    target: AcceptanceTarget,
    step_adaptation: StepAdaptation,
    weights: Option<DiminishingSchedule>,
    period: usize,
    history: usize,
}

/// Initialization is independent of paired-score estimation. A single tail score
/// is not a reliable curvature estimate, even for a standard Gaussian.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetricInitialization {
    /// Conservative default until paired observations provide geometric evidence.
    #[default]
    Identity,
    /// Experimental historical score rule; clipping bounds numerics, not quality.
    ClippedScore,
}
impl MetricInitialization {
    /// Writes coordinate scales for the initial inverse mass without allocating.
    /// # Errors
    /// Shape/nonfinite-score errors preserve output. The score rule clips mass to
    /// [1e-20,1e20], with mass one for an exactly zero score.
    pub fn scales_into(self, score: &[f64], out: &mut [f64]) -> Result<(), FisherError> {
        if score.len() != out.len() {
            return Err(FisherError::Configuration);
        }
        if score.iter().any(|s| !s.is_finite()) {
            return Err(FisherError::Numerical);
        }
        for (&score, scale) in score.iter().zip(out) {
            *scale = if self == Self::Identity || score == 0.0 {
                1.0
            } else {
                score
                    .abs()
                    .clamp(1e-20, 1e20)
                    .sqrt()
                    .recip()
                    .clamp(1e-10, 1e10)
            };
        }
        Ok(())
    }
}
impl FisherOptions {
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            max_rank: 0,
            initialization: MetricInitialization::Identity,
            target: 0.8.try_into().expect("valid acceptance target"),
            step_adaptation: StepAdaptation::default(),
            weights: None,
            period: 80,
            history: 160,
        }
    }
    /// Set the step-controller acceptance target (default 0.8), including the
    /// final fixed-geometry adaptation phase. A higher target generally reduces
    /// step size but does not guarantee divergence-free sampling. With fixed
    /// step counts it also shortens trajectories, potentially reducing mixing.
    /// This does not change the search criterion or the divergence threshold.
    ///
    /// ```
    /// use alea_mcmc::adapt::FisherOptions;
    /// let options = FisherOptions::new(1000).with_target(0.95.try_into()?);
    /// # let _ = options;
    /// # Ok::<(), alea_mcmc::config::SamplerConfigError>(())
    /// ```
    #[must_use]
    pub fn with_target(mut self, target: AcceptanceTarget) -> Self {
        self.target = target;
        self
    }
    /// Select initialization separately from subsequent Fisher estimation.
    #[must_use]
    pub fn with_initialization(mut self, initialization: MetricInitialization) -> Self {
        self.initialization = initialization;
        self
    }
    /// Choose a warmup-only step controller; defaults to dual averaging.
    #[must_use]
    pub fn with_step_adaptation(mut self, settings: StepAdaptation) -> Self {
        self.step_adaptation = settings;
        self
    }
    /// Use normalized diminishing weighted moments for diagonal geometry.
    /// Combining this with nonzero low-rank fitting is rejected at construction:
    /// a bounded unweighted ring is not the same weighted estimator.
    #[must_use]
    pub fn with_weighted_moments(mut self, schedule: DiminishingSchedule) -> Self {
        self.weights = Some(schedule);
        self
    }
    /// Bound the main window period and paired history (default 80/160).
    /// The initial window remains ten observations. History bounds storage and
    /// fitting work independently of the output rank cap.
    /// # Errors
    /// Requires positive period and capacity >= 2.
    pub fn with_window_budget(
        mut self,
        period: usize,
        history: usize,
    ) -> Result<Self, FisherError> {
        if period == 0 || history < 2 {
            return Err(FisherError::Configuration);
        }
        self.period = period;
        self.history = history;
        Ok(self)
    }
    /// Opt into low-rank fits every window. Zero selects diagonal adaptation.
    /// Rank is validated against the target dimension during construction.
    #[cfg(feature = "faer")]
    #[must_use]
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.max_rank = rank;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FisherWarmupError<E: Error + 'static> {
    #[error(transparent)]
    Warmup(#[from] WarmupError<E>),
    #[error(transparent)]
    Fisher(#[from] FisherError),
}
pub type FisherWarmupResult<'a, T, I = Leapfrog> = Result<
    (Hmc<'a, T, LowRankDiagonalMetric, I>, HmcWarmupReport),
    FisherWarmupError<<T as LogDensityGradient>::Error>,
>;

/// Owning warmup controller; consuming `run` returns a frozen ordinary HMC chain.
#[derive(Debug)]
#[must_use = "run warmup before retaining draws"]
pub struct FisherHmcWarmup<'a, T: LogDensityGradient + ?Sized, I = Leapfrog> {
    chain: Hmc<'a, T, LowRankDiagonalMetric, I>,
    adapter: FisherMetricAdapter,
    initial_scales: OwnedBuffer,
    scales: OwnedBuffer,
    options: FisherOptions,
    weighted: Option<WeightedFisherMoments>,
}
impl<'a, T: LogDensityGradient + ?Sized> FisherHmcWarmup<'a, T> {
    /// Construct experimental Fisher warmup with leapfrog.
    /// # Errors
    /// Propagates shape/rank/storage and initial evaluation errors.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        hmc: HmcOptions,
        options: FisherOptions,
    ) -> Result<Self, FisherWarmupError<T::Error>> {
        Self::new_with_integrator(target, position, hmc, options, Leapfrog)
    }
}
impl<'a, T: LogDensityGradient + ?Sized, I: Integrator> FisherHmcWarmup<'a, T, I> {
    /// Initialize identity geometry by default; clipped-score initialization is an
    /// explicit experiment, separate from subsequent paired Fisher estimation.
    /// # Errors
    /// Rejects shape/rank errors and propagates target errors without boxing.
    pub fn new_with_integrator(
        target: &'a T,
        position: OwnedBuffer,
        hmc: HmcOptions,
        options: FisherOptions,
        integrator: I,
    ) -> Result<Self, FisherWarmupError<T::Error>> {
        let d = target.dimension();
        if options.max_rank > d || d == 0 || (options.max_rank > 0 && options.weights.is_some()) {
            return Err(FisherError::Configuration.into());
        }
        let adapter = FisherMetricAdapter::new(
            d,
            10,
            if options.max_rank > 0 {
                options.history
            } else {
                0
            },
        )?;
        let weighted = options
            .weights
            .map(|schedule| WeightedFisherMoments::new(d, schedule))
            .transpose()?;
        let metric = LowRankDiagonalMetric::new(
            OwnedBuffer::from_fn(d, |_| 1.0),
            OwnedBuffer::new(0),
            OwnedBuffer::new(0),
        )
        .map_err(FisherError::from)?;
        let mut chain = Hmc::new_with_integrator(target, position, metric, hmc, integrator)
            .map_err(WarmupError::from)?;
        let mut initial_scales = OwnedBuffer::new(d);
        options
            .initialization
            .scales_into(chain.point().gradient(), &mut initial_scales)?;
        if options.iterations > 0 {
            chain
                .metric_mut()
                .set_scales(&initial_scales)
                .map_err(FisherError::from)?;
        }
        Ok(Self {
            chain,
            adapter,
            scales: initial_scales.clone(),
            initial_scales,
            options,
            weighted,
        })
    }
    /// Run adaptation, then discard all moments and paired history.
    /// # Errors
    /// Returns typed backend, bounded-search, adaptation or Fisher fitting errors.
    /// No partially adapted chain escapes on error. RNG consumption is not undone.
    pub fn run<R: Rng + ?Sized>(self, rng: &mut R) -> FisherWarmupResult<'a, T, I> {
        self.run_observed(rng, |_, _, _, _| {})
    }

    // Private static-dispatch observation seam for controller contract tests.
    // The public path monomorphizes a no-op: no observer storage or allocations.
    fn run_observed<R: Rng + ?Sized>(
        mut self,
        rng: &mut R,
        mut observe: impl FnMut(usize, &Self, &crate::HmcTransition, &HmcWarmupReport),
    ) -> FisherWarmupResult<'a, T, I> {
        let n = self.options.iterations;
        let mut report = HmcWarmupReport {
            iterations: n,
            accepted: 0,
            divergences: 0,
            metric_updates: 0,
            search_probes: 0,
            integration_attempts: 0,
            step_size: self.chain.options().step_size(),
        };
        if n == 0 {
            return Ok((self.chain, report));
        }
        let search = find_reasonable_step_size(&mut self.chain, SearchOptions::default(), rng)?;
        self.chain.set_step_size(search.step_size);
        report.search_probes = search.probes;
        report.integration_attempts = search.probes;
        let target = self.options.target;
        let mut step_adapter =
            StepSizeController::new(search.step_size, target, self.options.step_adaptation);
        // Integer arithmetic avoids overflow for any usize iteration budget.
        let switch = n / 10 * 3 + (n % 10) * 3 / 10;
        let freeze = n / 100 * 85 + (n % 100) * 85 / 100;
        for i in 0..n {
            if i == switch {
                self.adapter.reset(self.options.period)?;
            }
            if i == freeze {
                step_adapter = StepSizeController::new(
                    self.chain.options().step_size(),
                    target,
                    self.options.step_adaptation,
                );
            }
            let transition = self.chain.step(rng).map_err(WarmupError::from)?;
            report.accepted += usize::from(transition.accepted);
            report.divergences += usize::from(transition.divergence.is_some());
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(WarmupError::CounterOverflow)?;
            if i < freeze {
                if let Some(weighted) = &mut self.weighted {
                    weighted
                        .observe(self.chain.point().position(), self.chain.point().gradient())?;
                    weighted.scales_into(&self.initial_scales, 1e-5, &mut self.scales)?;
                } else {
                    self.adapter
                        .observe(self.chain.point().position(), self.chain.point().gradient())?;
                    self.adapter
                        .scales_into(&self.initial_scales, 1e-5, &mut self.scales)?;
                }
                self.chain
                    .metric_mut()
                    .set_scales(&self.scales)
                    .map_err(FisherError::from)?;
                #[cfg(feature = "faer")]
                if self.options.max_rank > 0
                    && (self.adapter.seen.is_multiple_of(self.adapter.period) || i + 1 == freeze)
                    && self.adapter.retained >= 2
                {
                    let metric = self.adapter.fit_low_rank(
                        &self.scales,
                        1e-5,
                        2.0,
                        self.options.max_rank,
                    )?;
                    self.chain
                        .replace_metric(metric)
                        .map_err(WarmupError::from)?;
                }
                report.metric_updates += 1;
            }
            let step = step_adapter
                .update(StepObservation::from(&transition))
                .map_err(WarmupError::from)?;
            self.chain.set_step_size(step);
            observe(i, &self, &transition, &report);
        }
        self.chain.set_step_size(step_adapter.finish());
        report.step_size = self.chain.options().step_size();
        Ok((self.chain, report))
    }
}

#[cfg(test)]
mod tests;
