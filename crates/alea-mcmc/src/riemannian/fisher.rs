//! Analytic expected-Fisher metrics for selected model families.

use std::convert::Infallible;

use alea_core::capability::PositionMetric;
use alea_math::buffer::OwnedBuffer;

/// Invalid analytic Fisher-metric configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FisherMetricConfigError {
    #[error("Fisher metric dimension and row count must be nonzero")]
    Empty,
    #[error("Fisher metric storage size overflow")]
    SizeOverflow,
    #[error("Fisher metric storage length is inconsistent")]
    StorageLength,
    #[error("Fisher metric input contains a non-finite value")]
    NonFinite,
    #[error("Fisher metric precision must be finite and positive")]
    InvalidPrecision,
}

/// Expected posterior Fisher metric for binary logistic regression.
///
/// Rows of the design matrix are contiguous. The metric is
/// `X^T diag(sigmoid(Xq) * (1 - sigmoid(Xq))) X + diag(prior_precision)`.
#[derive(Debug)]
pub struct LogisticFisherMetric {
    rows: usize,
    dimension: usize,
    design: OwnedBuffer,
    prior_precision: OwnedBuffer,
}

impl LogisticFisherMetric {
    /// Takes owned row-major design and diagonal prior precision.
    ///
    /// # Errors
    /// Rejects empty, overflowing, malformed, non-finite, or non-positive input.
    pub fn new(
        rows: usize,
        dimension: usize,
        design: OwnedBuffer,
        prior_precision: OwnedBuffer,
    ) -> Result<Self, FisherMetricConfigError> {
        if rows == 0 || dimension == 0 {
            return Err(FisherMetricConfigError::Empty);
        }
        let expected = rows
            .checked_mul(dimension)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
            .ok_or(FisherMetricConfigError::SizeOverflow)?;
        if design.len() != expected || prior_precision.len() != dimension {
            return Err(FisherMetricConfigError::StorageLength);
        }
        if design.iter().any(|value| !value.is_finite()) {
            return Err(FisherMetricConfigError::NonFinite);
        }
        if prior_precision
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(FisherMetricConfigError::InvalidPrecision);
        }
        Ok(Self {
            rows,
            dimension,
            design,
            prior_precision,
        })
    }
}

impl PositionMetric for LogisticFisherMetric {
    type Error = Infallible;

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn metric_and_derivatives(
        &self,
        position: &[f64],
        metric: &mut [f64],
        derivatives: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), Self::Error> {
        let d = self.dimension;
        metric.fill(0.0);
        derivatives.fill(0.0);
        for (index, precision) in self.prior_precision.iter().enumerate() {
            metric[index * d + index] = *precision;
        }
        for row in self.design.chunks_exact(d).take(self.rows) {
            let eta = row.iter().zip(position).map(|(x, q)| x * q).sum::<f64>();
            let probability = if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                let exponential = eta.exp();
                exponential / (1.0 + exponential)
            };
            let weight = probability * (1.0 - probability);
            let weight_derivative = weight * (1.0 - 2.0 * probability);
            for (j, xj) in row.iter().enumerate() {
                for (k, xk) in row.iter().enumerate() {
                    metric[j * d + k] += weight * xj * xk;
                    for (coordinate, xc) in row.iter().enumerate() {
                        derivatives[coordinate * d * d + j * d + k] +=
                            weight_derivative * xc * xj * xk;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Expected posterior Fisher metric for a normal hierarchy.
///
/// Coordinates are `(mu, log_scale, theta_0, ..., theta_{n-1})`, with
/// `theta_i ~ Normal(mu, exp(log_scale))` and unit-structure observations whose
/// precision is supplied explicitly. Expected cross-information with log scale
/// is zero; the log-scale information is `2n` plus its prior precision.
#[derive(Debug, Clone, Copy)]
pub struct HierarchicalNormalFisherMetric {
    groups: usize,
    mean_prior_precision: f64,
    log_scale_prior_precision: f64,
    observation_precision: f64,
}

impl HierarchicalNormalFisherMetric {
    /// Constructs a strictly positive-definite hierarchical Fisher metric.
    ///
    /// # Errors
    /// Rejects zero groups or non-finite/non-positive precisions.
    pub fn new(
        groups: usize,
        mean_prior_precision: f64,
        log_scale_prior_precision: f64,
        observation_precision: f64,
    ) -> Result<Self, FisherMetricConfigError> {
        if groups == 0 {
            return Err(FisherMetricConfigError::Empty);
        }
        if [
            mean_prior_precision,
            log_scale_prior_precision,
            observation_precision,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(FisherMetricConfigError::InvalidPrecision);
        }
        let dimension = groups
            .checked_add(2)
            .ok_or(FisherMetricConfigError::SizeOverflow)?;
        let matrix_len = dimension
            .checked_mul(dimension)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
            .ok_or(FisherMetricConfigError::SizeOverflow)?;
        matrix_len
            .checked_mul(dimension)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
            .ok_or(FisherMetricConfigError::SizeOverflow)?;
        Ok(Self {
            groups,
            mean_prior_precision,
            log_scale_prior_precision,
            observation_precision,
        })
    }
}

impl PositionMetric for HierarchicalNormalFisherMetric {
    type Error = Infallible;

    fn dimension(&self) -> usize {
        self.groups + 2
    }

    fn metric_and_derivatives(
        &self,
        position: &[f64],
        metric: &mut [f64],
        derivatives: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), Self::Error> {
        let d = self.dimension();
        let inverse_variance = (-2.0 * position[1]).exp();
        metric.fill(0.0);
        derivatives.fill(0.0);
        metric[0] = self.mean_prior_precision + self.groups as f64 * inverse_variance;
        metric[d + 1] = self.log_scale_prior_precision + 2.0 * self.groups as f64;
        let derivative_offset = d * d;
        derivatives[derivative_offset] = -2.0 * self.groups as f64 * inverse_variance;
        for group in 0..self.groups {
            let theta = group + 2;
            metric[theta] = -inverse_variance;
            metric[theta * d] = -inverse_variance;
            metric[theta * d + theta] = inverse_variance + self.observation_precision;

            derivatives[derivative_offset + theta] = 2.0 * inverse_variance;
            derivatives[derivative_offset + theta * d] = 2.0 * inverse_variance;
            derivatives[derivative_offset + theta * d + theta] = -2.0 * inverse_variance;
        }
        Ok(())
    }
}
