//! Warmup-only controllers sharing the error target minus observed acceptance.
use super::{AdaptationError, DualAveraging, scale};
use crate::config::{AcceptanceTarget, StepSize};
use crate::{HmcTransition, hmc::Divergence};

/// A power-law schedule with divergent sum and convergent squared sum.
/// These scalar conditions alone do not establish adaptive-MCMC ergodicity.
#[derive(Debug, Clone, Copy)]
pub struct DiminishingSchedule {
    offset: f64,
    exponent: f64,
}
impl DiminishingSchedule {
    /// # Errors
    /// Requires finite offset >= 1 and 1/2 < exponent <= 1.
    pub fn new(offset: f64, exponent: f64) -> Result<Self, AdaptationError> {
        if !offset.is_finite()
            || offset < 1.0
            || !exponent.is_finite()
            || exponent <= 0.5
            || exponent > 1.0
        {
            return Err(AdaptationError::Configuration);
        }
        Ok(Self { offset, exponent })
    }
    /// Weight at a zero-based iteration; positive for every usize index.
    pub fn weight(self, iteration: usize) -> f64 {
        (self.offset + iteration as f64).powf(-self.exponent)
    }
}

/// Explicit policy for the statistic consumed by a scalar warmup controller.
#[derive(Debug, Clone, Copy)]
pub enum StepObservation {
    /// MH probability, including nonzero probabilities on rejected proposals.
    Acceptance(f64),
    /// Numerical divergence contributes zero acceptance.
    Divergence(Divergence),
    /// A backend failure must not be treated as numerical rejection.
    /// The caller retains/propagates the original backend error separately.
    BackendFailure,
}
impl From<&HmcTransition> for StepObservation {
    fn from(transition: &HmcTransition) -> Self {
        match transition.divergence {
            Some(reason) => Self::Divergence(reason),
            None => Self::Acceptance(transition.acceptance_probability),
        }
    }
}
impl StepObservation {
    fn acceptance(self) -> Result<f64, AdaptationError> {
        match self {
            Self::Acceptance(a) if a.is_finite() && (0.0..=1.0).contains(&a) => Ok(a),
            Self::Acceptance(_) => Err(AdaptationError::AcceptanceProbability),
            Self::Divergence(_) => Ok(0.0),
            Self::BackendFailure => Err(AdaptationError::BackendFailure),
        }
    }
}

/// Validated scalar optimizer settings. Dual averaging remains the default.
/// RM and Adam are experimental warmup policies, not convergence guarantees;
/// the tested nonlinear targets can exhibit substantially more divergences.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepAdaptation {
    method: Method,
}
#[derive(Debug, Clone, Copy, Default)]
enum Method {
    #[default]
    DualAveraging,
    RobbinsMonro {
        rate: f64,
        schedule: DiminishingSchedule,
    },
    Adam {
        rate: f64,
        schedule: DiminishingSchedule,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}
impl StepAdaptation {
    /// # Errors
    /// Requires a finite positive initial learning rate no greater than one.
    pub fn robbins_monro(
        rate: f64,
        schedule: DiminishingSchedule,
    ) -> Result<Self, AdaptationError> {
        validate_rate(rate)?;
        Ok(Self {
            method: Method::RobbinsMonro { rate, schedule },
        })
    }
    /// Bias-corrected Adam with an explicitly decaying learning rate.
    /// Uses `target - acceptance` in log-step coordinates, as Robbins–Monro does.
    /// The second moment is represented by its square root to avoid underflow.
    /// # Errors
    /// Requires rate in (0,1], beta1/beta2 in [0,1), and finite normal epsilon > 0.
    pub fn adam(
        rate: f64,
        schedule: DiminishingSchedule,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Result<Self, AdaptationError> {
        validate_rate(rate)?;
        if !beta1.is_finite()
            || !beta2.is_finite()
            || !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2)
            || !epsilon.is_normal()
            || epsilon <= 0.0
        {
            return Err(AdaptationError::Configuration);
        }
        Ok(Self {
            method: Method::Adam {
                rate,
                schedule,
                beta1,
                beta2,
                epsilon,
            },
        })
    }
}
fn validate_rate(rate: f64) -> Result<(), AdaptationError> {
    if !rate.is_finite() || rate <= 0.0 || rate > 1.0 {
        Err(AdaptationError::Configuration)
    } else {
        Ok(())
    }
}

/// Transactional, allocation-free warmup controller. Consume it before sampling.
#[derive(Debug, Clone)]
pub struct StepSizeController {
    settings: StepAdaptation,
    target: AcceptanceTarget,
    dual: DualAveraging,
    iterations: usize,
    log_step: f64,
    first: f64,
    // Store sqrt(v), not v: squaring a valid tiny acceptance error can underflow.
    second: f64,
}
impl StepSizeController {
    /// Initialize a warmup controller; no observations or RNG are consumed.
    pub fn new(initial: StepSize, target: AcceptanceTarget, settings: StepAdaptation) -> Self {
        Self {
            settings,
            target,
            dual: DualAveraging::new(initial, target),
            iterations: 0,
            log_step: initial.value().ln(),
            first: 0.0,
            second: 0.0,
        }
    }
    /// # Errors
    /// Invalid statistics, backend failures, overflow and nonfinite arithmetic
    /// leave all optimizer state unchanged. Numerical divergence is not an error.
    pub fn update(&mut self, observation: StepObservation) -> Result<StepSize, AdaptationError> {
        let acceptance = observation.acceptance()?;
        if matches!(self.settings.method, Method::DualAveraging) {
            return self.dual.update(acceptance);
        }
        let iterations = self
            .iterations
            .checked_add(1)
            .ok_or(AdaptationError::IterationOverflow)?;
        let error = self.target.value() - acceptance;
        let (first, second, update) = match self.settings.method {
            Method::DualAveraging => unreachable!("handled above"),
            Method::RobbinsMonro { rate, schedule } => {
                (0.0, 0.0, rate * schedule.weight(self.iterations) * error)
            }
            Method::Adam {
                rate,
                schedule,
                beta1,
                beta2,
                epsilon,
            } => {
                let first = beta1 * self.first + (1.0 - beta1) * error;
                let second = (beta2.sqrt() * self.second).hypot((1.0 - beta2).sqrt() * error);
                // expm1 avoids cancellation for beta close to one, including t=1.
                let correction = |beta: f64| -(iterations as f64 * beta.ln()).exp_m1();
                let m = first / correction(beta1);
                let rms = second / correction(beta2).sqrt();
                (
                    first,
                    second,
                    rate * schedule.weight(self.iterations) * (m / (rms + epsilon)),
                )
            }
        };
        let log_step = self.log_step - update;
        if !log_step.is_finite() || !first.is_finite() || !second.is_finite() {
            return Err(AdaptationError::Numerical);
        }
        let step = scale(log_step);
        self.log_step = step.value().ln();
        self.first = first;
        self.second = second;
        self.iterations = iterations;
        Ok(step)
    }
    /// Final step; dual averaging uses its smoothed estimate, RM/Adam their
    /// bounded final iterate. No controller state accompanies the result.
    pub fn finish(self) -> StepSize {
        match self.settings.method {
            Method::DualAveraging => self.dual.finish(),
            _ => scale(self.log_step),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adam_normalization_does_not_square_tiny_errors() {
        let schedule = DiminishingSchedule::new(1.0, 0.75).unwrap();
        let settings = StepAdaptation::adam(1.0, schedule, 0.0, 0.0, f64::MIN_POSITIVE).unwrap();
        let mut c = StepSizeController::new(
            1.0.try_into().unwrap(),
            f64::MIN_POSITIVE.try_into().unwrap(),
            settings,
        );
        // m = sqrt(v) = epsilon, so the first normalized update is exactly 1/2.
        let actual = c.update(StepObservation::Acceptance(0.0)).unwrap().value();
        assert!((actual - (-0.5_f64).exp()).abs() < 1e-14);
    }
    #[test]
    fn count_overflow_is_atomic_and_extreme_initial_steps_stay_bounded() {
        let schedule = DiminishingSchedule::new(1.0, 0.75).unwrap();
        for settings in [
            StepAdaptation::robbins_monro(1.0, schedule).unwrap(),
            StepAdaptation::adam(1.0, schedule, 0.0, 0.0, f64::MIN_POSITIVE).unwrap(),
        ] {
            let mut c =
                StepSizeController::new(0.1.try_into().unwrap(), 0.8.try_into().unwrap(), settings);
            c.iterations = usize::MAX;
            let before = (
                c.log_step.to_bits(),
                c.first.to_bits(),
                c.second.to_bits(),
                c.iterations,
            );
            assert_eq!(
                c.update(StepObservation::Acceptance(0.0)),
                Err(AdaptationError::IterationOverflow)
            );
            assert_eq!(
                (
                    c.log_step.to_bits(),
                    c.first.to_bits(),
                    c.second.to_bits(),
                    c.iterations
                ),
                before
            );
            for value in [f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX] {
                let mut c = StepSizeController::new(
                    value.try_into().unwrap(),
                    0.8.try_into().unwrap(),
                    settings,
                );
                for a in [0.8, 0.0, 1.0] {
                    let step = c.update(StepObservation::Acceptance(a)).unwrap().value();
                    assert!(step.is_normal() && step > 0.0);
                }
            }
        }
    }
}
