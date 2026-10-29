//! Fused fixed-design regression likelihoods. One node per batch, not per row.
//!
//! Design rows include an intercept column only if supplied. Offsets are on the
//! linear-predictor scale: for Poisson exposure e>0 supply ln(e), not e. Captured
//! data are shared across cloned batches. These analytic nodes do not export an
//! Enzyme primal. A returned factor is joint over its rows; pointwise units must
//! be declared explicitly, for example by supplying separate batches.
#![deny(missing_docs)]
use super::{E, error};
use crate::scalar::Bernoulli;
use alea_core::model::assembly::{Expr, ModelBuilder, Parameter, Positive};
use alea_math::numeric::{CompensatedSum, first_nonfinite, sigmoid};
use std::sync::Arc;
use thiserror::Error;

/// Invalid fixed regression data or diagnostic input; no rows are silently dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RegressionDataError {
    /// At least one observation is required.
    #[error("empty regression observations")]
    Empty,
    /// Design storage or offsets do not match response dimensions.
    #[error("regression data dimensions do not agree")]
    Shape,
    /// A predictor, offset, real response or diagnostic direction is not finite.
    #[error("nonfinite regression {field} at index {index}")]
    NonFinite {
        /// Predictor, offset, observation or diagnostic direction.
        field: &'static str,
        /// Flat index in that field.
        index: usize,
    },
    /// Standardization needs an explicit column containing exactly one in every row.
    #[error("standardization requires column {column} to be a unit intercept")]
    Intercept {
        /// Requested intercept column in the original design.
        column: usize,
    },
    /// Count conversion would lose integer precision.
    #[error("Poisson count exceeds exact binary64 integers at row {index}")]
    CountPrecision {
        /// Observation index.
        index: usize,
    },
}
/// Owned row-major design and typed observations, shared cheaply when cloned.
/// Shape/predictor checks occur at construction; family-specific response checks
/// occur when attaching a factor. Rank and estimator existence are not inferred.
#[derive(Debug, Clone)]
pub struct RegressionData<Y> {
    columns: usize,
    design: Arc<[f64]>,
    observations: Arc<[Y]>,
    offsets: Arc<[f64]>,
}
impl<Y> RegressionData<Y> {
    /// Constructs an explicitly sized design with zero offsets. Zero columns
    /// permit an offset-only likelihood; empty observations are rejected.
    /// # Errors
    /// Rejects empty responses, shape overflow/mismatch and nonfinite predictors.
    pub fn new(
        columns: usize,
        design: Vec<f64>,
        observations: Vec<Y>,
    ) -> Result<Self, RegressionDataError> {
        if observations.is_empty() {
            return Err(RegressionDataError::Empty);
        }
        if columns.checked_mul(observations.len()) != Some(design.len()) {
            return Err(RegressionDataError::Shape);
        }
        finite(&design, "predictor")?;
        let offsets = vec![0.; observations.len()].into();
        Ok(Self {
            columns,
            design: design.into(),
            observations: observations.into(),
            offsets,
        })
    }
    /// Replaces zero/default offsets with explicit linear-predictor offsets.
    /// # Errors
    /// Rejects a mismatched length or nonfinite offsets, including log(0) exposures.
    pub fn with_offsets(mut self, offsets: Vec<f64>) -> Result<Self, RegressionDataError> {
        if offsets.len() != self.observations.len() {
            return Err(RegressionDataError::Shape);
        }
        finite(&offsets, "offset")?;
        self.offsets = offsets.into();
        Ok(self)
    }
    /// Number of observations.
    pub fn rows(&self) -> usize {
        self.observations.len()
    }
    /// Number of explicitly supplied coefficient columns.
    pub fn columns(&self) -> usize {
        self.columns
    }
    /// Flat row-major predictors.
    pub fn design(&self) -> &[f64] {
        &self.design
    }
    /// Responses in row order.
    pub fn observations(&self) -> &[Y] {
        &self.observations
    }
    /// Additive linear-predictor offsets.
    pub fn offsets(&self) -> &[f64] {
        &self.offsets
    }
    fn linear_predictor(&self, row: usize, beta: &[f64]) -> f64 {
        let mut sum = CompensatedSum::default();
        sum.add(self.offsets[row]);
        for (x, b) in self.design[row * self.columns..(row + 1) * self.columns]
            .iter()
            .zip(beta)
        {
            sum.add_product(*x, *b);
        }
        sum.total()
    }
}
fn finite(values: &[f64], field: &'static str) -> Result<(), RegressionDataError> {
    if let Some(index) = first_nonfinite(values) {
        return Err(RegressionDataError::NonFinite { field, index });
    }
    Ok(())
}
fn check_columns<Y>(data: &RegressionData<Y>, beta: &[Expr]) -> Result<(), E> {
    if data.columns != beta.len() {
        Err(E::Shape)
    } else {
        Ok(())
    }
}
/// Normal observations with a shared positive residual standard deviation.
/// The local score is with respect to supplied coefficient expressions and sigma;
/// the model's parameter chart supplies its own pullback. No priors are added.
/// # Errors
/// Rejects nonfinite responses, wrong coefficient counts and foreign handles.
/// Evaluation rejects nonpositive scales or nonrepresentable density/scores.
pub fn normal_regression(
    builder: &mut ModelBuilder,
    beta: &[Expr],
    sigma: &Parameter<Positive>,
    data: RegressionData<f64>,
) -> Result<Expr, E> {
    check_columns(&data, beta)?;
    finite(data.observations(), "observation").map_err(error)?;
    let mut inputs = beta.to_vec();
    inputs.push(sigma.expr());
    builder.custom_with_scratch(
        &inputs,
        vec![CompensatedSum::default(); data.columns + 1],
        move |x, g, scores| {
            let p = data.columns;
            let sigma = x[p];
            if !sigma.is_finite() || sigma <= 0. {
                return Err(E::Numerical);
            }
            let normalizer = -sigma.ln() - 0.5 * std::f64::consts::TAU.ln();
            let mut total = CompensatedSum::default();
            scores.fill(CompensatedSum::default());
            for i in 0..data.rows() {
                let eta = data.linear_predictor(i, &x[..p]);
                let z = (data.observations[i] - eta) / sigma;
                let value = -0.5 * z * z + normalizer;
                let mean_score = z / sigma;
                total.add(value);
                for (s, a) in scores[..p].iter_mut().zip(&data.design[i * p..(i + 1) * p]) {
                    s.add(a * mean_score);
                }
                scores[p].add((z * z - 1.) / sigma);
            }
            finish(total, scores, g)
        },
    )
}
/// Bernoulli observations with a logit link, preserving representable tail scores.
/// # Errors
/// Rejects wrong coefficient counts/foreign handles and nonrepresentable evaluation.
pub fn bernoulli_logit_regression(
    builder: &mut ModelBuilder,
    beta: &[Expr],
    data: RegressionData<bool>,
) -> Result<Expr, E> {
    check_columns(&data, beta)?;
    builder.custom_with_scratch(
        beta,
        vec![CompensatedSum::default(); data.columns],
        move |x, g, scores| {
            let mut total = CompensatedSum::default();
            scores.fill(CompensatedSum::default());
            for i in 0..data.rows() {
                let eta = data.linear_predictor(i, x);
                if !eta.is_finite() {
                    return Err(E::Numerical);
                }
                let y = data.observations[i];
                total.add(Bernoulli::logit_log_prob(eta, y).map_err(error)?);
                let residual = if y { sigmoid(-eta) } else { -sigmoid(eta) };
                for (s, a) in scores
                    .iter_mut()
                    .zip(&data.design[i * data.columns..(i + 1) * data.columns])
                {
                    s.add(a * residual);
                }
            }
            finish(total, scores, g)
        },
    )
}
/// Poisson observations with a log-rate link. Offsets are log exposures when used.
/// Evaluates `y*eta-exp(eta)-ln(y!)` directly, retaining finite log probabilities
/// when exp(eta) underflows. Counts above 2^53 are explicitly rejected.
/// # Errors
/// Rejects count precision loss, coefficient mismatches and foreign handles.
/// Overflowing rates, predictors, sums or scores are numerical errors.
pub fn poisson_log_regression(
    builder: &mut ModelBuilder,
    beta: &[Expr],
    data: RegressionData<u64>,
) -> Result<Expr, E> {
    check_columns(&data, beta)?;
    let mut normalizers = Vec::with_capacity(data.rows());
    for (index, &y) in data.observations.iter().enumerate() {
        if y > 1 << 53 {
            return Err(error(RegressionDataError::CountPrecision { index }));
        }
        normalizers.push(statrs::function::gamma::ln_gamma(y as f64 + 1.));
    }
    builder.custom_with_scratch(
        beta,
        vec![CompensatedSum::default(); data.columns],
        move |x, g, scores| {
            let mut total = CompensatedSum::default();
            scores.fill(CompensatedSum::default());
            for (i, normalizer) in normalizers.iter().enumerate() {
                let eta = data.linear_predictor(i, x);
                let rate = eta.exp();
                if !eta.is_finite() || !rate.is_finite() {
                    return Err(E::Numerical);
                }
                let y = data.observations[i] as f64;
                total.add(y * eta - rate - normalizer);
                for (s, a) in scores
                    .iter_mut()
                    .zip(&data.design[i * data.columns..(i + 1) * data.columns])
                {
                    s.add(a * (y - rate));
                }
            }
            finish(total, scores, g)
        },
    )
}
fn finish(total: CompensatedSum, scores: &[CompensatedSum], g: &mut [f64]) -> Result<f64, E> {
    let value = total.total();
    if !value.is_finite() || scores.iter().any(|s| !s.total().is_finite()) {
        return Err(E::Numerical);
    }
    for (out, score) in g.iter_mut().zip(scores) {
        *out = score.total();
    }
    Ok(value)
}

/// Likelihood-specific existence diagnostics, separate from numerical convergence.
pub mod diagnostics;

mod standardization;
pub use standardization::*;
