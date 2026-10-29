//! Sequential thin-SVD least squares through the optional Faer backend.
#![deny(missing_docs)]
use crate::numeric::{CompensatedSum, euclidean_norm, first_nonfinite};
use faer::{
    Mat, Par,
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
    linalg::svd,
};
use thiserror::Error;
/// Invalid data, policy or nonrepresentable factorization/solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LeastSquaresError {
    /// Shapes overflow/mismatch, no rows, or tolerance is outside (0,1).
    #[error("invalid least-squares shape or tolerance")]
    Shape,
    /// Input entries are not finite.
    #[error("nonfinite least-squares input")]
    NonFinite,
    /// Factorization, coefficients or residual norm are not representable.
    #[error("least-squares numerical failure")]
    Numerical,
}
/// Truncated-SVD solution and explicit numerical rank evidence.
/// Rank depends on units/scaling and the declared relative cutoff; it is not an
/// exact algebraic-rank certificate. No statistical covariance is inferred.
#[derive(Debug, Clone)]
pub struct LeastSquares {
    /// Minimum-norm coefficients in the retained singular subspace.
    pub coefficients: Vec<f64>,
    /// Singular values of X/max|X|, descending; data scale is retained separately.
    pub scaled_singular_values: Vec<f64>,
    /// Matrix scale max|X|, or one for a zero matrix.
    pub design_scale: f64,
    /// Number of singular values strictly above the declared cutoff.
    pub numerical_rank: usize,
    /// Relative rank cutoff supplied by the caller.
    pub relative_tolerance: f64,
    /// Absolute cutoff in scaled singular-value units.
    pub scaled_cutoff: f64,
    /// Stable Euclidean norm of the residual on the original data scale.
    pub residual_norm: f64,
    /// Rows minus numerical rank; not a claim of statistical independence.
    pub residual_degrees_of_freedom: usize,
}
/// Solves row-major X beta ~= y with a thin SVD, never normal equations.
/// Uses max-absolute input scaling and explicitly sequential Faer kernels without
/// mutating Faer's global parallelism. Zero columns are valid; zero rows are not.
/// # Errors
/// Rejects invalid inputs/cutoff, failed SVD and nonrepresentable outputs.
pub fn solve(
    rows: usize,
    columns: usize,
    design: &[f64],
    response: &[f64],
    relative_tolerance: f64,
) -> Result<LeastSquares, LeastSquaresError> {
    #[cfg(feature = "tracing")]
    let _span = tracing::debug_span!(target: "alea::numerics", "least_squares", rows = rows, columns = columns).entered();
    #[cfg(feature = "tracing")]
    tracing::debug!(target: "alea::numerics", schema_version = 1_u64, event = "started");
    let outcome = solve_inner(rows, columns, design, response, relative_tolerance);
    #[cfg(feature = "tracing")]
    match &outcome {
        Ok(result) => {
            tracing::debug!(target: "alea::numerics", schema_version = 1_u64, event = "finished", numerical_rank = result.numerical_rank, rank_policy = "relative_singular_value")
        }
        Err(_) => {
            tracing::error!(target: "alea::numerics", schema_version = 1_u64, event = "failed")
        }
    }
    outcome
}
fn solve_inner(
    rows: usize,
    columns: usize,
    design: &[f64],
    response: &[f64],
    relative_tolerance: f64,
) -> Result<LeastSquares, LeastSquaresError> {
    if rows == 0
        || rows.checked_mul(columns) != Some(design.len())
        || response.len() != rows
        || !relative_tolerance.is_finite()
        || relative_tolerance <= 0.
        || relative_tolerance >= 1.
        || rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(size_of::<f64>()))
            .is_none()
    {
        return Err(LeastSquaresError::Shape);
    }
    if first_nonfinite(design).is_some() || first_nonfinite(response).is_some() {
        return Err(LeastSquaresError::NonFinite);
    }
    let design_scale = design.iter().fold(0_f64, |m, x| m.max(x.abs()));
    let design_scale = if design_scale == 0. { 1. } else { design_scale };
    let response_scale = response.iter().fold(0_f64, |m, x| m.max(x.abs()));
    let response_scale = if response_scale == 0. {
        1.
    } else {
        response_scale
    };
    let k = rows.min(columns);
    let mut coefficients = vec![0.; columns];
    let mut singular_values = Vec::new();
    let mut rank = 0;
    let mut cutoff = 0.;
    if k > 0 {
        let matrix = Mat::from_fn(rows, columns, |i, j| design[i * columns + j] / design_scale);
        let mut u = Mat::zeros(rows, k);
        let mut v = Mat::zeros(columns, k);
        let mut singular = Diag::zeros(k);
        let mut scratch = MemBuffer::new(svd::svd_scratch::<f64>(
            rows,
            columns,
            svd::ComputeSvdVectors::Thin,
            svd::ComputeSvdVectors::Thin,
            Par::Seq,
            Default::default(),
        ));
        svd::svd(
            matrix.as_ref(),
            singular.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            Par::Seq,
            MemStack::new(&mut scratch),
            Default::default(),
        )
        .map_err(|_| LeastSquaresError::Numerical)?;
        singular_values = (0..k).map(|i| singular[i]).collect();
        if singular_values.iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(LeastSquaresError::Numerical);
        }
        cutoff = relative_tolerance * singular_values[0];
        rank = singular_values.iter().filter(|&&s| s > cutoff).count();
        let mut actions = vec![0.; k];
        for j in 0..k {
            if singular_values[j] > cutoff {
                let mut dot = CompensatedSum::default();
                for i in 0..rows {
                    dot.add(u[(i, j)] * (response[i] / response_scale));
                }
                actions[j] = dot.total() / singular_values[j];
            }
        }
        for i in 0..columns {
            let mut dot = CompensatedSum::default();
            for j in 0..k {
                dot.add(v[(i, j)] * actions[j]);
            }
            coefficients[i] = rescale(dot.total(), response_scale, design_scale);
        }
    }
    if first_nonfinite(&coefficients).is_some() {
        return Err(LeastSquaresError::Numerical);
    }
    let residuals: Vec<_> = (0..rows)
        .map(|i| {
            let mut residual = CompensatedSum::default();
            residual.add(response[i]);
            for j in 0..columns {
                residual.add(-design[i * columns + j] * coefficients[j]);
            }
            residual.total()
        })
        .collect();
    let norm = euclidean_norm(&residuals);
    if !norm.is_finite() {
        return Err(LeastSquaresError::Numerical);
    }
    Ok(LeastSquares {
        coefficients,
        scaled_singular_values: singular_values,
        design_scale,
        numerical_rank: rank,
        relative_tolerance,
        scaled_cutoff: cutoff,
        residual_norm: norm,
        residual_degrees_of_freedom: rows - rank,
    })
}
fn rescale(value: f64, numerator: f64, denominator: f64) -> f64 {
    let (v, a) = libm::frexp(value);
    let (n, b) = libm::frexp(numerator);
    let (d, c) = libm::frexp(denominator);
    libm::scalbn(v * n / d, a + b - c)
}
