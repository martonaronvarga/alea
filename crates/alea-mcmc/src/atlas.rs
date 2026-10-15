//! Bounded HVP curvature probes for opt-in ATLAS-style local-step experiments.
//!
//! This is deliberately not presented as a complete ATLAS transition. It
//! provides the bounded derivative-work component needed to benchmark local
//! curvature schedules without silently changing a stationary HMC kernel.

use alea_core::{
    capability::{HessianVector, HvpError, evaluate_hvp},
    target::PointState,
};
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{EuclideanMetric, MetricError};
use rand::Rng;

/// Validated work and step bounds for a local curvature probe.
#[derive(Debug, Clone, Copy)]
pub struct AtlasProbeOptions {
    restarts: usize,
    iterations: usize,
    safety: f64,
    min_step: f64,
    max_step: f64,
}

/// Invalid bounded-work or local-step settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid ATLAS curvature-probe configuration")]
pub struct AtlasConfigError;

impl AtlasProbeOptions {
    /// Creates a bounded power-iteration experiment on `H^2`.
    ///
    /// # Errors
    /// Requires nonzero restarts/iterations, positive finite safety, and ordered
    /// positive finite step bounds.
    pub fn new(
        restarts: usize,
        iterations: usize,
        safety: f64,
        min_step: f64,
        max_step: f64,
    ) -> Result<Self, AtlasConfigError> {
        if restarts == 0
            || iterations == 0
            || !safety.is_finite()
            || safety <= 0.0
            || !min_step.is_finite()
            || !max_step.is_finite()
            || min_step <= 0.0
            || min_step > max_step
        {
            return Err(AtlasConfigError);
        }
        let calls_per_restart = iterations
            .checked_mul(2)
            .and_then(|calls| calls.checked_add(1))
            .ok_or(AtlasConfigError)?;
        restarts
            .checked_mul(calls_per_restart)
            .ok_or(AtlasConfigError)?;
        Ok(Self {
            restarts,
            iterations,
            safety,
            min_step,
            max_step,
        })
    }

    /// Exact upper bound on HVP calls made by one probe.
    pub fn hvp_budget(self) -> usize {
        self.restarts * (2 * self.iterations + 1)
    }
}

/// Result of a bounded local-curvature experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasProbe {
    /// Largest observed `||H v||` over normalized power iterates. This is a
    /// lower estimate of spectral curvature, not a certified upper bound.
    pub curvature_estimate: f64,
    /// Safety-scaled local step clamped to configured bounds.
    pub local_step: f64,
    /// Exact HVP calls consumed.
    pub hvp_count: usize,
}

/// Configuration or checked target-capability failure.
#[derive(Debug, thiserror::Error)]
pub enum AtlasProbeError<E: std::error::Error + 'static> {
    /// Invalid work/step bounds or arithmetic overflow.
    #[error("invalid ATLAS curvature-probe configuration")]
    Configuration,
    /// Checked potential-Hessian vector evaluation failed.
    #[error(transparent)]
    Hvp(#[from] HvpError<E>),
    /// Metric action failed while forming the generalized curvature operator.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// Random direction generation or norm arithmetic became nonfinite.
    #[error("ATLAS curvature probe produced a nonfinite direction")]
    Numerical,
}

/// Estimates local spectral curvature with exactly bounded potential HVP work.
///
/// Applying `H` twice makes the power iteration target `H^2`, so negative
/// curvature is not discarded. The result remains a lower estimate; callers
/// must not treat it as a certified stability bound.
///
/// # Errors
/// Propagates checked HVP/model errors and nonfinite numerical work.
pub fn probe_local_curvature<T: HessianVector + ?Sized, R: Rng + ?Sized>(
    point: &PointState<'_, T>,
    options: AtlasProbeOptions,
    rng: &mut R,
) -> Result<AtlasProbe, AtlasProbeError<T::Error>> {
    let dimension = point.dimension();
    if dimension == 0 {
        return Err(AtlasProbeError::Configuration);
    }
    let mut direction = OwnedBuffer::new(dimension);
    let mut first = OwnedBuffer::new(dimension);
    let mut second = OwnedBuffer::new(dimension);
    let mut estimate = 0.0_f64;
    let mut calls = 0_usize;
    for _ in 0..options.restarts {
        for value in direction.iter_mut() {
            *value = alea_math::random::standard_normal(rng);
        }
        normalize(&mut direction)?;
        for _ in 0..options.iterations {
            evaluate_hvp(point.target(), point.position(), &direction, &mut first)?;
            evaluate_hvp(point.target(), point.position(), &first, &mut second)?;
            calls += 2;
            let norm = scaled_norm(&second);
            if norm == 0.0 {
                direction.fill(0.0);
                direction[0] = 1.0;
            } else if norm.is_finite() {
                for (direction, value) in direction.iter_mut().zip(second.iter()) {
                    *direction = *value / norm;
                }
            } else {
                return Err(AtlasProbeError::Numerical);
            }
        }
        evaluate_hvp(point.target(), point.position(), &direction, &mut first)?;
        calls += 1;
        estimate = estimate.max(scaled_norm(&first));
    }
    if !estimate.is_finite() {
        return Err(AtlasProbeError::Numerical);
    }
    let unconstrained = if estimate == 0.0 {
        options.max_step
    } else {
        options.safety / estimate.sqrt()
    };
    let local_step = unconstrained.clamp(options.min_step, options.max_step);
    Ok(AtlasProbe {
        curvature_estimate: estimate,
        local_step,
        hvp_count: calls,
    })
}

/// Metric-aware curvature estimate for the generalized oscillator operator
/// `M^-1 H`. As with [`probe_local_curvature`], this is an observed lower
/// estimate and requires explicit inflation before use as a policy bound.
///
/// # Errors
/// Rejects mismatched/empty geometry and propagates checked metric/HVP failures.
pub fn probe_metric_curvature<T: HessianVector + ?Sized, M: EuclideanMetric, R: Rng + ?Sized>(
    point: &PointState<'_, T>,
    metric: &M,
    options: AtlasProbeOptions,
    rng: &mut R,
) -> Result<AtlasProbe, AtlasProbeError<T::Error>> {
    let dimension = point.dimension();
    if dimension == 0 || metric.dimension() != dimension {
        return Err(AtlasProbeError::Configuration);
    }
    let mut direction = OwnedBuffer::new(dimension);
    let mut hessian = OwnedBuffer::new(dimension);
    let mut first = OwnedBuffer::new(dimension);
    let mut second = OwnedBuffer::new(dimension);
    let mut estimate = 0.0_f64;
    let mut calls = 0_usize;
    for _ in 0..options.restarts {
        for value in direction.iter_mut() {
            *value = alea_math::random::standard_normal(rng);
        }
        normalize(&mut direction)?;
        for _ in 0..options.iterations {
            evaluate_hvp(point.target(), point.position(), &direction, &mut hessian)?;
            metric.velocity(&hessian, &mut first)?;
            evaluate_hvp(point.target(), point.position(), &first, &mut hessian)?;
            metric.velocity(&hessian, &mut second)?;
            calls += 2;
            let norm = scaled_norm(&second);
            if norm == 0.0 {
                direction.fill(0.0);
                direction[0] = 1.0;
            } else if norm.is_finite() {
                for (direction, value) in direction.iter_mut().zip(second.iter()) {
                    *direction = *value / norm;
                }
            } else {
                return Err(AtlasProbeError::Numerical);
            }
        }
        evaluate_hvp(point.target(), point.position(), &direction, &mut hessian)?;
        metric.velocity(&hessian, &mut first)?;
        calls += 1;
        estimate = estimate.max(scaled_norm(&first));
    }
    if !estimate.is_finite() {
        return Err(AtlasProbeError::Numerical);
    }
    let unconstrained = if estimate == 0.0 {
        options.max_step
    } else {
        options.safety / estimate.sqrt()
    };
    Ok(AtlasProbe {
        curvature_estimate: estimate,
        local_step: unconstrained.clamp(options.min_step, options.max_step),
        hvp_count: calls,
    })
}

fn scaled_norm(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    scale
        * values
            .iter()
            .map(|value| (value / scale) * (value / scale))
            .sum::<f64>()
            .sqrt()
}

fn normalize<E: std::error::Error + 'static>(values: &mut [f64]) -> Result<(), AtlasProbeError<E>> {
    let norm = scaled_norm(values);
    if !norm.is_finite() || norm == 0.0 {
        return Err(AtlasProbeError::Numerical);
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alea_distributions::Gaussian;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn identity_hessian_has_unit_curvature_and_exact_budget() {
        let target = Gaussian::new(5);
        let point = PointState::new(&target, OwnedBuffer::new(5)).unwrap();
        let options = AtlasProbeOptions::new(3, 4, 0.8, 0.01, 2.0).unwrap();
        let probe =
            probe_local_curvature(&point, options, &mut SmallRng::seed_from_u64(71)).unwrap();
        assert!((probe.curvature_estimate - 1.0).abs() < 1e-14);
        assert!((probe.local_step - 0.8).abs() < 1e-14);
        assert_eq!(probe.hvp_count, options.hvp_budget());
    }
}
