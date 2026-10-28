//! Warmup-only step tuning for already preconditioned isokinetic targets.
use super::static_warmup::{Observation, drive};
use super::{StaticWarmupError, StaticWarmupOptions, StaticWarmupResult};
use crate::{
    config::StepSize,
    mams::{ApproximateMclmc, Mams, MamsError},
};
use alea_core::target::LogDensityGradient;
use rand::Rng;

/// Corrected MAMS uses its actual MH probability with default target0.8.
/// Geometry is fixed; an explicitly configured trajectory duration updates the
/// integration count with the step. Both freeze before retained sampling.
/// # Errors
/// Preserves concrete backend, controller and work-counter failures.
pub fn warmup_mams_step_size<'a, T: LogDensityGradient + ?Sized, R: Rng + ?Sized>(
    mut chain: Mams<'a, T>,
    options: StaticWarmupOptions,
    rng: &mut R,
) -> StaticWarmupResult<Mams<'a, T>, MamsError<T::Error>> {
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
/// Explicitly approximate error-variance tuning policy for this second-order
/// symmetric isokinetic integrator. Squared local correction error scales as h^6
/// at a fixed macrostep count, as checked by a deterministic refinement fixture.
#[derive(Debug, Clone, Copy)]
pub struct MclmcWarmupOptions {
    iterations: usize,
    target_per_dimension: f64,
    minimum: StepSize,
    maximum: StepSize,
}
impl MclmcWarmupOptions {
    /// Default per-dimension centered-error variance target5e-4 and bounds1e-8..10.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            target_per_dimension: 5e-4,
            minimum: 1e-8.try_into().expect("positive"),
            maximum: 10.0.try_into().expect("positive"),
        }
    }
    /// Change the positive finite centered-error variance target.
    /// # Errors
    /// Rejects nonpositive/nonfinite targets.
    pub fn with_target(mut self, target: f64) -> Result<Self, super::AdaptationError> {
        if !target.is_finite() || target <= 0. {
            return Err(super::AdaptationError::Configuration);
        }
        self.target_per_dimension = target;
        Ok(self)
    }
}
/// Warmup error observations, with no implication of an exact retained target.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MclmcWarmupReport {
    /// Attempted warmup transitions, including recovered numerical failures.
    pub iterations: usize,
    /// Actual macrosteps simulated.
    pub integration_attempts: usize,
    /// Frozen smoothed step.
    pub step_size: StepSize,
    /// Per-dimension centered correction-error variance target.
    pub target_per_dimension: f64,
    /// Exponent of squared local error in step size.
    pub error_power: f64,
    /// Most recent centered error variance per dimension at the observed step.
    pub observed_error_per_dimension: f64,
    /// Warmup updates limited by the declared bounds.
    pub bounded_step_updates: usize,
    /// Numerical proposals discarded only during warmup; retained sampling never retries.
    pub recovered_numerical_failures: usize,
}
/// Frozen approximate chain and explicit error-variance adaptation diagnostics.
pub type MclmcWarmupResult<'a, T> = Result<
    (ApproximateMclmc<'a, T>, MclmcWarmupReport),
    StaticWarmupError<MamsError<<T as alea_core::target::TargetSpace>::Error>>,
>;

/// Adapt an explicitly approximate chain without introducing an MH correction.
/// Refresh length and macrostep count remain fixed. The scalar controller averages
/// normalized error coefficients rather than pretending acceptance exists.
/// # Errors
/// Preserves typed kernel errors; unrepresentable error/controller arithmetic fails.
pub fn warmup_mclmc_step_size<'a, T: LogDensityGradient + ?Sized, R: Rng + ?Sized>(
    mut chain: ApproximateMclmc<'a, T>,
    options: MclmcWarmupOptions,
    rng: &mut R,
) -> MclmcWarmupResult<'a, T> {
    // Options are deliberately queried through the frozen public getter.
    let initial = chain.options().step_size();
    let mut report = MclmcWarmupReport {
        iterations: 0,
        integration_attempts: 0,
        step_size: initial,
        target_per_dimension: options.target_per_dimension,
        error_power: 6.,
        observed_error_per_dimension: 0.,
        bounded_step_updates: 0,
        recovered_numerical_failures: 0,
    };
    if options.iterations == 0 {
        return Ok((chain, report));
    }
    let dimension = chain.point().dimension() as f64;
    let mut coefficients = ErrorVariance::default();
    let mut log_average = initial.value().ln();
    for index in 0..options.iterations {
        let result = chain.step(rng);
        report.iterations += 1;
        report.integration_attempts = report
            .integration_attempts
            .checked_add(chain.attempted_steps())
            .ok_or(StaticWarmupError::CounterOverflow)?;
        let step = chain.options().step_size().value();
        let o = match result {
            Ok(value) => value,
            Err(error) if error.is_numerical() => {
                report.recovered_numerical_failures += 1;
                let smaller = (0.5 * step).max(options.minimum.value());
                report.bounded_step_updates += usize::from(smaller == step);
                chain.set_step_size(smaller.try_into().expect("positive smaller step"));
                log_average = smaller.ln();
                coefficients = ErrorVariance::default();
                continue;
            }
            Err(error) => return Err(StaticWarmupError::SamplerConfig(error)),
        };
        let error = o.correction_error / dimension.sqrt();
        coefficients.update(error, step);
        // Center the signed h^-3 coefficients. The logarithmic common scale
        // preserves their variance without forming error^2 / h^6.
        let log_variance = coefficients.log_variance();
        let proposed = if let Some(variance) = log_variance {
            ((options.target_per_dimension.ln() - variance) / 6.)
                .clamp((step * 0.5).ln(), (step * 1.1).ln())
                .exp()
        } else {
            step * 1.1
        };
        if !proposed.is_finite() {
            return Err(StaticWarmupError::Adaptation(
                super::AdaptationError::Numerical,
            ));
        }
        let bounded = proposed.clamp(options.minimum.value(), options.maximum.value());
        report.bounded_step_updates += usize::from(bounded != proposed);
        let weight = ((index + 1) as f64).powf(-0.75);
        log_average = (1. - weight) * log_average + weight * bounded.ln();
        report.observed_error_per_dimension =
            log_variance.map_or(0., |v| (v + 6. * step.ln()).exp());
        if !report.observed_error_per_dimension.is_finite() {
            return Err(StaticWarmupError::Adaptation(
                super::AdaptationError::Numerical,
            ));
        }
        chain.set_step_size(bounded.try_into().expect("bounded step"));
    }
    if report.recovered_numerical_failures == report.iterations {
        return Err(StaticWarmupError::Adaptation(
            super::AdaptationError::Numerical,
        ));
    }
    let step = log_average
        .exp()
        .clamp(options.minimum.value(), options.maximum.value())
        .try_into()
        .expect("bounded step");
    chain.set_step_size(step);
    report.step_size = step;
    Ok((chain, report))
}

// Exponentially weighted centered moments in a shared logarithmic scale.
#[derive(Default)]
struct ErrorVariance {
    scale: Option<f64>,
    mean: f64,
    variance: f64,
    observations: usize,
}
impl ErrorVariance {
    fn update(&mut self, error: f64, step: f64) {
        let value = if error == 0. {
            0.
        } else {
            let magnitude = error.abs().ln() - 3. * step.ln();
            let scale = self.scale.map_or(magnitude, |old| old.max(magnitude));
            if let Some(old) = self.scale {
                let rescale = (old - scale).exp();
                self.mean *= rescale;
                self.variance *= rescale * rescale;
            }
            self.scale = Some(scale);
            error.signum() * (magnitude - scale).exp()
        };
        if self.observations == 0 {
            self.mean = value;
        } else {
            let delta = value - self.mean;
            self.mean += 0.1 * delta;
            self.variance = 0.9 * (self.variance + 0.1 * delta * delta);
        }
        self.observations += 1;
    }
    fn log_variance(&self) -> Option<f64> {
        self.scale
            .filter(|_| self.variance > 0.)
            .map(|s| 2. * s + self.variance.ln())
    }
}
#[cfg(test)]
mod tests {
    use super::ErrorVariance;
    #[test]
    fn centered_scaled_error_variance_matches_independent_recurrence() {
        let mut moments = ErrorVariance::default();
        let values = [3., 3., 4., -2., 5., 0., 1.];
        let (mut mean, mut variance) = (values[0], 0.);
        for (index, value) in values.into_iter().enumerate() {
            moments.update(value, 0.5);
            if index > 0 {
                let delta = value - mean;
                mean += 0.1 * delta;
                variance = 0.9 * (variance + 0.1 * delta * delta);
            }
        }
        let actual = (moments.log_variance().unwrap() + 6. * 0.5_f64.ln()).exp();
        assert!((actual - variance).abs() < 1e-12);
        let mut constant = ErrorVariance::default();
        for _ in 0..10 {
            constant.update(3., 0.5);
        }
        assert!(constant.log_variance().is_none());
    }
}
