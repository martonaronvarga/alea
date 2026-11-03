//! Sequential thin-SVD least squares through the optional Faer backend.
#![deny(missing_docs)]
use crate::matrix::{MatrixError, RowMajor, cholesky_in_place};
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
    /// Covariance was requested but the supplied rank policy discards a column.
    #[error("least-squares covariance requires full column rank")]
    RankDeficient,
    /// Original rounded-covariance Cholesky failure, including its pivot index.
    #[error("least-squares covariance factorization failed: {0}")]
    Covariance(#[from] MatrixError),
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
    let outcome =
        solve_inner(rows, columns, design, response, relative_tolerance, None).map(|(fit, _)| fit);
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
/// Solves once and computes row-major `RSS/divisor * (X^T X)^-1` from the same
/// thin SVD. No normal equations, explicit inverse or rank truncation for this
/// covariance path. The divisor is explicit; this function assigns no statistical
/// meaning to it. Residual scaling precedes the Gram calculation, avoiding an
/// unscaled inverse-design covariance intermediate. Positive-residual covariance
/// also requires a checked Cholesky on its max-absolute-scaled rounded entries.
/// No full-range or exact-rank guarantee.
/// # Errors
/// Rejects invalid shapes/policy, zero or non-exactly representable divisor,
/// deficient numerical column rank and nonrepresentable factor/solve/covariance.
pub fn solve_with_residual_covariance(
    rows: usize,
    columns: usize,
    design: &[f64],
    response: &[f64],
    relative_tolerance: f64,
    divisor: usize,
) -> Result<(LeastSquares, Vec<f64>), LeastSquaresError> {
    if divisor == 0
        || u64::try_from(divisor).map_or(true, |n| n > (1_u64 << 53))
        || columns
            .checked_mul(columns)
            .is_none_or(|n| n > isize::MAX as usize / size_of::<f64>())
    {
        return Err(LeastSquaresError::Shape);
    }
    solve_inner(
        rows,
        columns,
        design,
        response,
        relative_tolerance,
        Some(divisor),
    )
}
fn solve_inner(
    rows: usize,
    columns: usize,
    design: &[f64],
    response: &[f64],
    relative_tolerance: f64,
    divisor: Option<usize>,
) -> Result<(LeastSquares, Vec<f64>), LeastSquaresError> {
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
    let mut covariance_vectors = None;
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
        if divisor.is_some() && rank != columns {
            return Err(LeastSquaresError::RankDeficient);
        }
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
        if divisor.is_some() {
            covariance_vectors = Some(v);
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
    let mut covariance = Vec::new();
    if let (Some(divisor), Some(mut vectors)) = (divisor, covariance_vectors) {
        let residual_scale = norm / (divisor as f64).sqrt();
        if norm > 0. && residual_scale == 0. {
            return Err(LeastSquaresError::Numerical);
        }
        for i in 0..columns {
            for j in 0..columns {
                let original = vectors[(i, j)];
                let value = rescale(original / singular_values[j], residual_scale, design_scale);
                if !value.is_finite() || (norm > 0. && original != 0. && value == 0.) {
                    return Err(LeastSquaresError::Numerical);
                }
                vectors[(i, j)] = value;
            }
        }
        covariance.resize(columns * columns, 0.);
        for i in 0..columns {
            for j in 0..=i {
                let mut sum = CompensatedSum::default();
                for k in 0..columns {
                    sum.add_product(vectors[(i, k)], vectors[(j, k)]);
                }
                let value = sum.total();
                if !value.is_finite() || (norm > 0. && i == j && value <= 0.) {
                    return Err(LeastSquaresError::Numerical);
                }
                covariance[i * columns + j] = value;
                covariance[j * columns + i] = value;
            }
        }
        if norm > 0. && columns > 0 {
            let scale = covariance.iter().fold(0_f64, |a, x| a.max(x.abs()));
            let mut check: Vec<_> = covariance.iter().map(|x| x / scale).collect();
            if covariance
                .iter()
                .zip(&check)
                .any(|(a, b)| *a != 0. && *b == 0.)
            {
                return Err(LeastSquaresError::Numerical);
            }
            cholesky_in_place::<RowMajor>(columns, &mut check)?;
        }
    }
    Ok((
        LeastSquares {
            coefficients,
            scaled_singular_values: singular_values,
            design_scale,
            numerical_rank: rank,
            relative_tolerance,
            scaled_cutoff: cutoff,
            residual_norm: norm,
            residual_degrees_of_freedom: rows - rank,
        },
        covariance,
    ))
}
fn rescale(value: f64, numerator: f64, denominator: f64) -> f64 {
    let (v, a) = libm::frexp(value);
    let (n, b) = libm::frexp(numerator);
    let (d, c) = libm::frexp(denominator);
    libm::scalbn(v * n / d, a + b - c)
}
