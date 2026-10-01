//! Experimental paired-score Fisher warmup; covariance warmup remains the oracle.
use super::{
    DualAveraging, HmcWarmupReport, SearchOptions, WarmupError, find_reasonable_step_size,
};
use crate::integrator::{Integrator, Leapfrog};
use crate::{Hmc, HmcOptions};
use alea_core::target::LogDensityGradient;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{LowRankDiagonalMetric, MetricError},
};
use rand::Rng;
use std::error::Error;

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
            let c = self.foreground.values[2 * self.dim + i] / scale / scale + ridge;
            let f = self.foreground.values[3 * self.dim + i] * scale * scale + ridge;
            let sigma = (scale.ln() + 0.25 * (c.ln() - f.ln())).exp();
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

/// Experimental CPU Fisher schedule: periods 10 then 80, geometry frozen for
/// the final 15%; dual averaging (not nutpie's Adam) adapts the step size.
#[derive(Debug, Clone, Copy)]
pub struct FisherOptions {
    iterations: usize,
    max_rank: usize,
    initialization: MetricInitialization,
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
        }
    }
    /// Select initialization separately from subsequent Fisher estimation.
    #[must_use]
    pub fn with_initialization(mut self, initialization: MetricInitialization) -> Self {
        self.initialization = initialization;
        self
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
        if options.max_rank > d || d == 0 {
            return Err(FisherError::Configuration.into());
        }
        let adapter = FisherMetricAdapter::new(d, 10, if options.max_rank > 0 { 160 } else { 0 })?;
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
        })
    }
    /// Run adaptation, then discard all moments and paired history.
    /// # Errors
    /// Returns typed backend, bounded-search, adaptation or Fisher fitting errors.
    /// No partially adapted chain escapes on error. RNG consumption is not undone.
    pub fn run<R: Rng + ?Sized>(mut self, rng: &mut R) -> FisherWarmupResult<'a, T, I> {
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
        let target = 0.8.try_into().expect("valid acceptance target");
        let mut step_adapter = DualAveraging::new(search.step_size, target);
        // Integer arithmetic avoids overflow for any usize iteration budget.
        let switch = n / 10 * 3 + (n % 10) * 3 / 10;
        let freeze = n / 100 * 85 + (n % 100) * 85 / 100;
        for i in 0..n {
            if i == switch {
                self.adapter.reset(80)?;
            }
            if i == freeze {
                step_adapter = DualAveraging::new(self.chain.options().step_size(), target);
            }
            let transition = self.chain.step(rng).map_err(WarmupError::from)?;
            report.accepted += usize::from(transition.accepted);
            report.divergences += usize::from(transition.divergence.is_some());
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(WarmupError::CounterOverflow)?;
            if i < freeze {
                self.adapter
                    .observe(self.chain.point().position(), self.chain.point().gradient())?;
                self.adapter
                    .scales_into(&self.initial_scales, 1e-5, &mut self.scales)?;
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
                .update(transition.acceptance_probability)
                .map_err(WarmupError::from)?;
            self.chain.set_step_size(step);
        }
        self.chain.set_step_size(step_adapter.finish());
        report.step_size = self.chain.options().step_size();
        Ok((self.chain, report))
    }
}
