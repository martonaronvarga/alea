//! Cold-path regularized Fisher fitting in the joint position/score subspace.
//! No dense ambient d-by-d covariance is formed unless the data span all d axes.
use crate::{
    buffer::OwnedBuffer,
    metric::{LowRankDiagonalMetric, MetricError},
};
use faer::{
    Accum, Mat, MatRef, Par,
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
};

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

// Fits are chain-local, usually small reduced systems. Explicit sequential
// kernels avoid nested worker barriers and never mutate Faer's global policy.
fn product(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> Mat<f64> {
    let mut result = Mat::zeros(lhs.nrows(), rhs.ncols());
    faer::linalg::matmul::matmul(result.as_mut(), Accum::Replace, lhs, rhs, 1.0, Par::Seq);
    result
}

struct Eigen {
    vectors: Mat<f64>,
    values: Diag<f64>,
}

fn eigen(matrix: &Mat<f64>) -> Result<Eigen, FisherFitError> {
    if (0..matrix.ncols()).any(|j| (0..matrix.nrows()).any(|i| !matrix[(i, j)].is_finite())) {
        return Err(FisherFitError::Numerical);
    }
    let n = matrix.nrows();
    let mut vectors = Mat::zeros(n, n);
    let mut values = Diag::zeros(n);
    let mut scratch = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        Par::Seq,
        Default::default(),
    ));
    faer::linalg::evd::self_adjoint_evd(
        matrix.as_ref(),
        values.as_mut(),
        Some(vectors.as_mut()),
        Par::Seq,
        MemStack::new(&mut scratch),
        Default::default(),
    )
    .map_err(|_| FisherFitError::Numerical)?;
    let values_ref = values.column_vector();
    if (0..n).any(|i| !values_ref[i].is_finite() || values_ref[i] <= 0.0) {
        return Err(FisherFitError::Numerical);
    }
    Ok(Eigen { vectors, values })
}

fn power(eigen: &Eigen, exponent: f64) -> Result<Mat<f64>, FisherFitError> {
    let values = eigen.values.column_vector();
    let u = eigen.vectors.as_ref();
    // One power per eigenvalue, not per matrix entry. Keep the same powf
    // operation (rather than a reciprocal/sqrt rewrite) for numerical behavior.
    let mut scaled = Mat::zeros(u.nrows(), u.ncols());
    for j in 0..u.ncols() {
        let factor = values[j].powf(exponent);
        for i in 0..u.nrows() {
            scaled[(i, j)] = u[(i, j)] * factor;
        }
    }
    let result = product(scaled.as_ref(), u.transpose());
    if (0..result.ncols()).any(|j| (0..result.nrows()).any(|i| !result[(i, j)].is_finite())) {
        return Err(FisherFitError::Numerical);
    }
    Ok(result)
}

// If C = L L^T and B = F^(1/2) L = U Sigma V^T, then
// (F^(1/2) C F^(1/2))^(1/2) = U Sigma U^T. Taking the SVD of B
// avoids squaring its condition number before recovering small singular values.
fn sandwich_root(mut c: Mat<f64>, fhalf: &Mat<f64>) -> Result<Mat<f64>, FisherFitError> {
    use faer::linalg::{cholesky::llt::factor, svd};

    let n = c.nrows();
    let mut scratch = MemBuffer::new(factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Default::default(),
    ));
    factor::cholesky_in_place(
        c.as_mut(),
        Default::default(), // No dynamic regularization or spectral clipping.
        Par::Seq,
        MemStack::new(&mut scratch),
        Default::default(),
    )
    .map_err(|_| FisherFitError::Numerical)?;
    // The factorization writes only the lower triangle; ordinary products read both.
    for j in 0..n {
        for i in 0..j {
            c[(i, j)] = 0.0;
        }
    }
    let b = product(fhalf.as_ref(), c.as_ref());
    if (0..n).any(|j| (0..n).any(|i| !b[(i, j)].is_finite())) {
        return Err(FisherFitError::Numerical);
    }
    let mut vectors = Mat::zeros(n, n);
    let mut values = Diag::zeros(n);
    let mut scratch = MemBuffer::new(svd::svd_scratch::<f64>(
        n,
        n,
        svd::ComputeSvdVectors::Full,
        svd::ComputeSvdVectors::No,
        Par::Seq,
        Default::default(),
    ));
    svd::svd(
        b.as_ref(),
        values.as_mut(),
        Some(vectors.as_mut()),
        None,
        Par::Seq,
        MemStack::new(&mut scratch),
        Default::default(),
    )
    .map_err(|_| FisherFitError::Numerical)?;
    let singular_values = values.column_vector();
    if (0..n).any(|i| !singular_values[i].is_finite() || singular_values[i] <= 0.0) {
        return Err(FisherFitError::Numerical);
    }
    power(&Eigen { vectors, values }, 1.0)
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
/// The middle square root uses Cholesky/SVD factors to avoid squaring their
/// conditioning in a formed sandwich product. Reduced decompositions/products
/// execute sequentially within each chain, without
/// accessing or modifying Faer's process-global parallelism configuration.
/// A common power-of-two normalization of data and ridge precedes squared norms
/// and scatter formation; this changes neither the real-arithmetic objective nor
/// the relative regularization.
///
/// # Errors
/// Rejects invalid dimensions/options, non-finite data, failed decompositions or
/// invalid output geometry, including a ridge lost to underflow during common
/// normalization. Degenerate constant data returns diagonal geometry.
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
    // Scale before norms and Gram products: normalizing the scatters afterwards
    // cannot recover squares already lost to overflow or subnormal rounding.
    // (X,Y,ridge) -> (X/a,Y/a,ridge/a^2) leaves G F G = C unchanged.
    // Including sqrt(ridge) bounds the rescaled ridge and makes amplitude normal
    // even for the smallest positive f64 ridge. Round DOWN to a power of two
    // so data rescaling introduces no significand rounding for normal results.
    let mut amplitude = ridge.sqrt();
    for j in 0..n {
        for i in 0..d {
            amplitude = amplitude.max(x[(i, j)].abs()).max(y[(i, j)].abs());
        }
    }
    let amplitude = f64::from_bits(amplitude.to_bits() & (0x7ff_u64 << 52));
    // Do not form a^2 (which itself may overflow/underflow), and do not silently
    // discard regularization if the input's dynamic range is unrepresentable.
    let ridge = (ridge / amplitude) / amplitude;
    if !ridge.is_finite() || ridge <= 0.0 {
        return Err(FisherFitError::Numerical);
    }
    for j in 0..n {
        for i in 0..d {
            x[(i, j)] /= amplitude;
            y[(i, j)] /= amplitude;
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
    let xp = product(q.transpose(), x.as_ref());
    let yp = product(q.transpose(), y.as_ref());
    let mut c = product(xp.as_ref(), xp.transpose());
    let mut f = product(yp.as_ref(), yp.transpose());
    for i in 0..k {
        c[(i, i)] += ridge;
        f[(i, i)] += ridge;
    }
    // G F G = C is unchanged when both regularized scatters are divided by
    // the same positive number. Keep the nested square-root product away from
    // avoidable overflow/underflow while preserving relative regularization.
    let mut common_scale = 0.0_f64;
    for j in 0..k {
        for i in 0..k {
            if !c[(i, j)].is_finite() || !f[(i, j)].is_finite() {
                return Err(FisherFitError::Numerical);
            }
            common_scale = common_scale.max(c[(i, j)].abs()).max(f[(i, j)].abs());
        }
    }
    if common_scale <= 0.0 {
        return Err(FisherFitError::Numerical);
    }
    for j in 0..k {
        for i in 0..k {
            c[(i, j)] /= common_scale;
            f[(i, j)] /= common_scale;
        }
    }
    // Both powers share exactly the same SPD eigensystem.
    let f_eigen = eigen(&f)?;
    let fhalf = power(&f_eigen, 0.5)?;
    let finvhalf = power(&f_eigen, -0.5)?;
    let middle_root = sandwich_root(c, &fhalf)?;
    let g = product(
        product(finvhalf.as_ref(), middle_root.as_ref()).as_ref(),
        finvhalf.as_ref(),
    );
    let eig = eigen(&g)?;
    let values = eig.values.column_vector();
    let mut retained: Vec<_> = (0..k)
        .filter(|&i| values[i] < threshold.recip() || values[i] > threshold)
        .collect();
    retained.sort_by(|&i, &j| values[j].ln().abs().total_cmp(&values[i].ln().abs()));
    retained.truncate(max_rank);
    let vectors = product(q.as_ref(), eig.vectors.as_ref());
    Ok(LowRankDiagonalMetric::new(
        OwnedBuffer::from_fn(d, |i| scales[i]),
        OwnedBuffer::from_fn(d * retained.len(), |index| {
            vectors[(index % d, retained[index / d])]
        }),
        OwnedBuffer::from_fn(retained.len(), |i| values[retained[i]]),
    )?)
}
