//! Central finite differences for diagnostics only, never a sampler AD fallback.
#![forbid(unsafe_code)]
use crate::target::{EvaluationError, LogDensityGradient, evaluate};
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub struct GradientCheckOptions {
    pub relative_step: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}
impl Default for GradientCheckOptions {
    fn default() -> Self {
        Self {
            relative_step: 1e-5,
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-5,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GradientComponent {
    pub index: usize,
    pub analytic: f64,
    pub finite_difference: f64,
    pub absolute_error: f64,
    pub passed: bool,
}
#[derive(Debug)]
pub struct GradientReport {
    pub log_density: f64,
    pub components: Vec<GradientComponent>,
}
impl GradientReport {
    pub fn passed(&self) -> bool {
        self.components.iter().all(|c| c.passed)
    }
}
#[derive(Debug, Error)]
pub enum GradientCheckError<E: std::error::Error + 'static> {
    #[error("invalid finite-difference step or tolerances")]
    Options,
    #[error("finite-difference displacement is not representable at coordinate {index}")]
    Displacement { index: usize },
    #[error(transparent)]
    Evaluation(#[from] EvaluationError<E>),
}

/// Evaluate the supplied target at the center and at 2N nearby positions.
/// Allocations here are intentional: this is an explicit diagnostic API.
pub fn check_gradient<T: LogDensityGradient + ?Sized>(
    target: &T,
    position: &[f64],
    options: GradientCheckOptions,
) -> Result<GradientReport, GradientCheckError<T::Error>> {
    if !options.relative_step.is_finite()
        || options.relative_step <= 0.0
        || !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance < 0.0
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance < 0.0
    {
        return Err(GradientCheckError::Options);
    }
    let mut analytic = vec![0.0; position.len()];
    let log_density = evaluate(target, position, &mut analytic)?;
    let mut q = position.to_vec();
    let mut scratch = vec![0.0; position.len()];
    let mut components = Vec::with_capacity(position.len());
    for i in 0..q.len() {
        let h = options.relative_step * position[i].abs().max(1.0);
        let plus = position[i] + h;
        let minus = position[i] - h;
        if !plus.is_finite()
            || !minus.is_finite()
            || !(plus - minus).is_finite()
            || plus == position[i]
            || minus == position[i]
        {
            return Err(GradientCheckError::Displacement { index: i });
        }
        q[i] = plus;
        let fp = evaluate(target, &q, &mut scratch)?;
        q[i] = minus;
        let fm = evaluate(target, &q, &mut scratch)?;
        q[i] = position[i];
        let finite_difference = (fp - fm) / (plus - minus);
        let absolute_error = (finite_difference - analytic[i]).abs();
        let passed = finite_difference.is_finite()
            && absolute_error
                <= options.absolute_tolerance
                    + options.relative_tolerance * finite_difference.abs().max(analytic[i].abs());
        components.push(GradientComponent {
            index: i,
            analytic: analytic[i],
            finite_difference,
            absolute_error,
            passed,
        });
    }
    Ok(GradientReport {
        log_density,
        components,
    })
}
