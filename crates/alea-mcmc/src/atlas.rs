//! Bounded HVP curvature probes for opt-in joint warmup experiments.
//!
//! This internal module provides the bounded derivative-work component used to
//! select a frozen integration policy without silently changing a stationary
//! HMC kernel.

use alea_core::{
    capability::{HvpError, PotentialHessianVector, evaluate_hvp},
    target::PointState,
};
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{EuclideanMetric, MetricError};
use rand::Rng;

/// Validated work bounds for a local curvature probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct CurvatureProbeOptions {
    restarts: usize,
    iterations: usize,
}

/// Invalid bounded-work settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid curvature-probe configuration")]
pub struct CurvatureProbeConfigError;

impl CurvatureProbeOptions {
    /// Creates a bounded power-iteration experiment.
    ///
    /// # Errors
    /// Requires nonzero restart and iteration budgets whose exact HVP count is
    /// representable in `usize`.
    pub fn new(restarts: usize, iterations: usize) -> Result<Self, CurvatureProbeConfigError> {
        if restarts == 0 || iterations == 0 {
            return Err(CurvatureProbeConfigError);
        }
        let calls_per_restart = iterations
            .checked_mul(2)
            .and_then(|calls| calls.checked_add(1))
            .ok_or(CurvatureProbeConfigError)?;
        restarts
            .checked_mul(calls_per_restart)
            .ok_or(CurvatureProbeConfigError)?;
        Ok(Self {
            restarts,
            iterations,
        })
    }

    /// Exact upper bound on HVP calls made by one probe.
    #[must_use]
    pub fn hvp_budget(self) -> usize {
        self.restarts * (2 * self.iterations + 1)
    }
}

/// Result of a bounded local-curvature experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CurvatureProbe {
    /// Largest observed `||H v||` over normalized power iterates. This is a
    /// lower estimate of spectral curvature, not a certified upper bound.
    pub(crate) curvature_estimate: f64,
    /// Exact HVP calls consumed.
    pub(crate) hvp_count: usize,
}

/// Configuration or checked target-capability failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CurvatureProbeError<E: std::error::Error + 'static> {
    /// Invalid work/step bounds or arithmetic overflow.
    #[error("invalid curvature-probe configuration")]
    Configuration,
    /// Checked potential-Hessian vector evaluation failed.
    #[error(transparent)]
    Hvp(#[from] HvpError<E>),
    /// Metric action failed while forming the generalized curvature operator.
    #[error(transparent)]
    Metric(#[from] MetricError),
    /// Random direction generation or norm arithmetic became nonfinite.
    #[error("curvature probe produced a nonfinite direction")]
    Numerical,
}

/// Metric-aware curvature estimate for the generalized oscillator operator
/// `M^-1 H`. This is an observed lower estimate and requires explicit inflation
/// before use as a policy bound.
///
/// # Errors
/// Rejects mismatched/empty geometry and propagates checked metric/HVP failures.
pub(crate) fn probe_metric_curvature<
    T: PotentialHessianVector + ?Sized,
    M: EuclideanMetric,
    R: Rng + ?Sized,
>(
    point: &PointState<'_, T>,
    metric: &M,
    options: CurvatureProbeOptions,
    rng: &mut R,
) -> Result<CurvatureProbe, CurvatureProbeError<T::Error>> {
    let dimension = point.dimension();
    if dimension == 0 || metric.dimension() != dimension {
        return Err(CurvatureProbeError::Configuration);
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
                return Err(CurvatureProbeError::Numerical);
            }
        }
        evaluate_hvp(point.target(), point.position(), &direction, &mut hessian)?;
        metric.velocity(&hessian, &mut first)?;
        calls += 1;
        estimate = estimate.max(scaled_norm(&first));
    }
    if !estimate.is_finite() {
        return Err(CurvatureProbeError::Numerical);
    }
    Ok(CurvatureProbe {
        curvature_estimate: estimate,
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

fn normalize<E: std::error::Error + 'static>(
    values: &mut [f64],
) -> Result<(), CurvatureProbeError<E>> {
    let norm = scaled_norm(values);
    if !norm.is_finite() || norm == 0.0 {
        return Err(CurvatureProbeError::Numerical);
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alea_distributions::StandardNormalTarget;
    use alea_math::metric::IdentityMetric;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn identity_hessian_has_unit_curvature_and_exact_budget() {
        let target = StandardNormalTarget::new(5);
        let point = PointState::new(&target, OwnedBuffer::new(5)).unwrap();
        let options = CurvatureProbeOptions::new(3, 4).unwrap();
        let probe = probe_metric_curvature(
            &point,
            &IdentityMetric::new(5),
            options,
            &mut SmallRng::seed_from_u64(71),
        )
        .unwrap();
        assert!((probe.curvature_estimate - 1.0).abs() < 1e-14);
        assert_eq!(probe.hvp_count, options.hvp_budget());
    }
}
