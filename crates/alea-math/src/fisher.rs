//! Cold-path regularized Fisher fitting in the joint position/score subspace.
//! No dense ambient d-by-d covariance is formed unless the data span all d axes.
use crate::{
    buffer::OwnedBuffer,
    metric::{LowRankDiagonalMetric, MetricError},
};
use faer::{Mat, Side};

/// Invalid or numerically unrepresentable Fisher fit.
#[derive(Debug, thiserror::Error)]
pub enum FisherFitError {
    #[error("invalid Fisher data shape or fitting options")]
    Shape,
    #[error("non-finite or non-positive-definite Fisher calculation")]
    Numerical,
    #[error(transparent)]
    Metric(#[from] MetricError),
}

fn power(matrix: &Mat<f64>, exponent: f64) -> Result<Mat<f64>, FisherFitError> {
    if (0..matrix.ncols()).any(|j| (0..matrix.nrows()).any(|i| !matrix[(i, j)].is_finite())) {
        return Err(FisherFitError::Numerical);
    }
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| FisherFitError::Numerical)?;
    let values = eigen.S().column_vector();
    if (0..values.nrows()).any(|i| !values[i].is_finite() || values[i] <= 0.0) {
        return Err(FisherFitError::Numerical);
    }
    let u = eigen.U();
    let scaled = Mat::from_fn(u.nrows(), u.ncols(), |i, j| {
        u[(i, j)] * values[j].powf(exponent)
    });
    let result = &scaled * u.transpose();
    if (0..result.ncols()).any(|j| (0..result.nrows()).any(|i| !result[(i, j)].is_finite())) {
        return Err(FisherFitError::Numerical);
    }
    Ok(result)
}

/// Fits centered paired row-major positions and log-density scores.
///
/// In standardized coordinates, uses **scatter** matrices `C = X X^T + ridge I`
/// and `F = Y Y^T + ridge I`, matching the nuts-rs scatter convention. Solves
/// `G F G = C` in the joint data subspace. Outside that subspace G is identity.
/// Retains eigenvalues outside `[1/threshold, threshold]`, prioritizing largest
/// absolute log eigenvalue if `max_rank` truncates the solution. This truncation
/// is a heuristic, not a globally optimal rank-constrained Fisher minimizer.
///
/// Cold path: allocates O(d min(d,2n) + min(d,2n)^2 + d n) storage and may factor
/// a full-rank reduced matrix. `max_rank` bounds output rank, not fitting cost;
/// callers must bound the number of observations too.
///
/// # Errors
/// Rejects invalid dimensions/options, non-finite data, failed decompositions or
/// invalid output geometry. Degenerate constant data returns diagonal geometry.
pub fn fit_low_rank(
    positions: &[f64],
    scores: &[f64],
    scales: &[f64],
    ridge: f64,
    threshold: f64,
    max_rank: usize,
) -> Result<LowRankDiagonalMetric, FisherFitError> {
    let d = scales.len();
    if d == 0
        || positions.len() != scores.len()
        || !positions.len().is_multiple_of(d)
        || positions.len() / d < 2
        || !ridge.is_finite()
        || ridge <= 0.0
        || !threshold.is_finite()
        || threshold < 1.0
        || max_rank > d
    {
        return Err(FisherFitError::Shape);
    }
    let n = positions.len() / d;
    let empty = || {
        LowRankDiagonalMetric::new(
            OwnedBuffer::from_fn(d, |i| scales[i]),
            OwnedBuffer::new(0),
            OwnedBuffer::new(0),
        )
    };
    // Validate scales even when all data are degenerate.
    let diagonal = empty()?;
    let mut x = Mat::zeros(d, n);
    let mut y = Mat::zeros(d, n);
    for i in 0..d {
        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        for row in 0..n {
            mean_x += (positions[row * d + i] - mean_x) / (row + 1) as f64;
            mean_y += (scores[row * d + i] - mean_y) / (row + 1) as f64;
        }
        for row in 0..n {
            x[(i, row)] = (positions[row * d + i] - mean_x) / scales[i];
            y[(i, row)] = (scores[row * d + i] - mean_y) * scales[i];
            if !x[(i, row)].is_finite() || !y[(i, row)].is_finite() {
                return Err(FisherFitError::Numerical);
            }
        }
    }
    // Twice-reorthogonalized MGS handles rank-deficient/repeated draws explicitly.
    let mut basis = Vec::<f64>::new();
    let mut candidate = vec![0.0; d];
    for data in [&x, &y] {
        for col in 0..n {
            if basis.len() / d == d {
                break;
            }
            for i in 0..d {
                candidate[i] = data[(i, col)];
            }
            let original = candidate.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !original.is_finite() {
                return Err(FisherFitError::Numerical);
            }
            for _ in 0..2 {
                for u in basis.chunks_exact(d) {
                    let dot: f64 = u.iter().zip(&candidate).map(|(a, b)| a * b).sum();
                    for (v, &u) in candidate.iter_mut().zip(u) {
                        *v -= dot * u;
                    }
                }
            }
            let norm = candidate.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() {
                return Err(FisherFitError::Numerical);
            }
            if norm > original * 1e-10 && norm > 0.0 {
                basis.extend(candidate.iter().map(|v| v / norm));
            }
        }
    }
    let k = basis.len() / d;
    if k == 0 || max_rank == 0 {
        return Ok(diagonal);
    }
    let q = Mat::from_fn(d, k, |i, j| basis[j * d + i]);
    let xp = q.transpose() * &x;
    let yp = q.transpose() * &y;
    let mut c = &xp * xp.transpose();
    let mut f = &yp * yp.transpose();
    for i in 0..k {
        c[(i, i)] += ridge;
        f[(i, i)] += ridge;
    }
    let fhalf = power(&f, 0.5)?;
    let finvhalf = power(&f, -0.5)?;
    let middle = &fhalf * &c * &fhalf;
    let g = &finvhalf * power(&middle, 0.5)? * &finvhalf;
    if (0..k).any(|j| (0..k).any(|i| !g[(i, j)].is_finite())) {
        return Err(FisherFitError::Numerical);
    }
    let eig = g
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| FisherFitError::Numerical)?;
    let values = eig.S().column_vector();
    if (0..k).any(|i| !values[i].is_finite() || values[i] <= 0.0) {
        return Err(FisherFitError::Numerical);
    }
    let mut retained: Vec<_> = (0..k)
        .filter(|&i| values[i] < threshold.recip() || values[i] > threshold)
        .collect();
    retained.sort_by(|&i, &j| values[j].ln().abs().total_cmp(&values[i].ln().abs()));
    retained.truncate(max_rank);
    let vectors = &q * eig.U();
    Ok(LowRankDiagonalMetric::new(
        OwnedBuffer::from_fn(d, |i| scales[i]),
        OwnedBuffer::from_fn(d * retained.len(), |index| {
            vectors[(index % d, retained[index / d])]
        }),
        OwnedBuffer::from_fn(retained.len(), |i| values[retained[i]]),
    )?)
}
