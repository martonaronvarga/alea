//! Online centered moments and cold-path conversion from covariance to mass.
use alea_math::{
    buffer::OwnedBuffer,
    metric::{
        CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric, MetricError,
    },
};

/// Diagonal adaptation is the default; dense adaptation is explicitly opt-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetricKind {
    /// Linear-memory variance adaptation; does not learn rotations.
    #[default]
    Diagonal,
    /// Quadratic-memory covariance adaptation with cubic window-boundary work.
    Dense,
}

/// Invalid covariance input or an unrepresentable estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CovarianceError {
    #[error("covariance requires a nonzero, addressable dimension")]
    Dimension,
    #[error("covariance observation has the wrong dimension")]
    Shape,
    #[error("covariance arithmetic is nonfinite")]
    NonFinite,
    #[error("covariance requires at least two observations")]
    TooFewSamples,
    #[error("covariance sample count overflow")]
    CountOverflow,
    #[error("regularized covariance is not representably positive definite")]
    NotPositiveDefinite,
    #[error(transparent)]
    Metric(#[from] MetricError),
}

/// Allocation-free Welford updates after construction, with transactional failures.
/// Dense moments use symmetric full column-major storage. Rejected Markov states
/// must be included as repeated observations; omitting them biases the estimate.
#[derive(Debug)]
pub struct OnlineCovariance {
    kind: MetricKind,
    count: usize,
    mean: OwnedBuffer,
    m2: OwnedBuffer,
    delta: OwnedBuffer,
    next_mean: OwnedBuffer,
    next_m2: OwnedBuffer,
}

impl OnlineCovariance {
    /// Allocate reusable observation scratch.
    ///
    /// # Errors
    /// Rejects zero dimensions and storage-size overflow before allocating.
    pub fn new(dimension: usize, kind: MetricKind) -> Result<Self, CovarianceError> {
        let length = match kind {
            MetricKind::Diagonal => Some(dimension),
            MetricKind::Dense => dimension.checked_mul(dimension),
        }
        .filter(|&n| n > 0 && n <= isize::MAX as usize / size_of::<f64>())
        .ok_or(CovarianceError::Dimension)?;
        Ok(Self {
            kind,
            count: 0,
            mean: OwnedBuffer::new(dimension),
            m2: OwnedBuffer::new(length),
            delta: OwnedBuffer::new(dimension),
            next_mean: OwnedBuffer::new(dimension),
            next_m2: OwnedBuffer::new(length),
        })
    }

    /// Number of successfully incorporated observations in this window.
    pub fn count(&self) -> usize {
        self.count
    }
    /// Current mean (zero before the first observation).
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Update centered moments. Invalid/overflowing observations leave the
    /// published mean, moments and count unchanged (scratch may change).
    ///
    /// # Errors
    /// Rejects shape mismatch, nonfinite arithmetic and count overflow.
    pub fn update(&mut self, point: &[f64]) -> Result<(), CovarianceError> {
        if point.len() != self.mean.len() {
            return Err(CovarianceError::Shape);
        }
        let count = self
            .count
            .checked_add(1)
            .ok_or(CovarianceError::CountOverflow)?;
        let n = self.mean.len();
        for (i, &x) in point.iter().enumerate() {
            self.delta[i] = x - self.mean[i];
            self.next_mean[i] = self.mean[i] + self.delta[i] / count as f64;
        }
        if self
            .delta
            .iter()
            .chain(self.next_mean.iter())
            .any(|v| !v.is_finite())
        {
            return Err(CovarianceError::NonFinite);
        }
        // delta * delta^T * (n-1)/n is algebraically Welford's update and
        // preserves symmetry exactly instead of computing both triangles separately.
        let weight = self.count as f64 / count as f64;
        match self.kind {
            MetricKind::Diagonal => {
                for i in 0..n {
                    self.next_m2[i] = self.m2[i] + (self.delta[i] * weight) * self.delta[i];
                }
            }
            MetricKind::Dense => {
                for col in 0..n {
                    for row in col..n {
                        let value =
                            self.m2[row + col * n] + (self.delta[row] * weight) * self.delta[col];
                        self.next_m2[row + col * n] = value;
                        self.next_m2[col + row * n] = value;
                    }
                }
            }
        }
        if self.next_m2.iter().any(|v| !v.is_finite()) {
            return Err(CovarianceError::NonFinite);
        }
        std::mem::swap(&mut self.mean, &mut self.next_mean);
        std::mem::swap(&mut self.m2, &mut self.next_m2);
        self.count = count;
        Ok(())
    }

    /// Clear a window without reallocating. Windows deliberately have no memory.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean.fill(0.0);
        self.m2.fill(0.0);
    }

    /// Stan/BlackJAX regularization: n/(n+5) * sample_cov + 0.005/(n+5) I.
    /// The result is **inverse mass**, not momentum covariance. Dense layout is
    /// column-major. This cold window-boundary operation allocates its result.
    ///
    /// # Errors
    /// Requires at least two observations and finite resulting entries.
    pub fn regularized_covariance(&self) -> Result<OwnedBuffer, CovarianceError> {
        if self.count < 2 {
            return Err(CovarianceError::TooFewSamples);
        }
        let count = self.count as f64;
        let weight = count / (count + 5.0);
        let ridge = 0.005 / (count + 5.0);
        let n = self.mean.len();
        let covariance = OwnedBuffer::from_fn(self.m2.len(), |i| {
            weight * (self.m2[i] / (count - 1.0))
                + if self.kind == MetricKind::Diagonal || i % (n + 1) == 0 {
                    ridge
                } else {
                    0.0
                }
        });
        if covariance.iter().any(|v| !v.is_finite()) {
            return Err(CovarianceError::NonFinite);
        }
        Ok(covariance)
    }

    /// Build a validated momentum mass M = inverse(regularized covariance).
    /// Dense inversion/factorization allocates only at window boundaries; the
    /// resulting sampler reuses the existing SIMD/Faer/OpenBLAS metric kernels.
    ///
    /// # Errors
    /// Propagates insufficient/nonfinite covariance and rejects an unrepresentable
    /// inverse or factor. The estimator and any installed metric are unchanged.
    pub fn metric(&self) -> Result<WarmupMetric, CovarianceError> {
        let covariance = self.regularized_covariance()?;
        match self.kind {
            MetricKind::Diagonal => {
                let mass = OwnedBuffer::from_fn(covariance.len(), |i| 1.0 / covariance[i]);
                Ok(WarmupMetric::Diagonal(DiagonalMetric::new(mass)?))
            }
            MetricKind::Dense => {
                let n = self.mean.len();
                let factor = cholesky(covariance, n)?;
                let covariance_metric = DenseMetric::new(CholeskyFactor::new_lower(n, factor)?);
                let mut mass = OwnedBuffer::new(n * n);
                let mut basis = OwnedBuffer::new(n);
                for col in 0..n {
                    basis.fill(0.0);
                    basis[col] = 1.0;
                    covariance_metric.velocity(&basis, &mut mass[col * n..(col + 1) * n])?;
                }
                // Roundoff in independent solves can differ across the diagonal.
                for col in 0..n {
                    for row in col + 1..n {
                        let value = 0.5 * mass[row + col * n] + 0.5 * mass[col + row * n];
                        mass[row + col * n] = value;
                        mass[col + row * n] = value;
                    }
                }
                Ok(WarmupMetric::Dense(DenseMetric::new(
                    CholeskyFactor::new_lower(n, cholesky(mass, n)?)?,
                )))
            }
        }
    }
}

fn cholesky(mut matrix: OwnedBuffer, n: usize) -> Result<OwnedBuffer, CovarianceError> {
    for col in 0..n {
        for row in col..n {
            let mut value = matrix[row + col * n];
            for k in 0..col {
                value -= matrix[row + k * n] * matrix[col + k * n];
            }
            if row == col {
                if !value.is_finite() || value <= 0.0 {
                    return Err(CovarianceError::NotPositiveDefinite);
                }
                matrix[row + col * n] = value.sqrt();
            } else {
                value /= matrix[col + col * n];
                if !value.is_finite() {
                    return Err(CovarianceError::NonFinite);
                }
                matrix[row + col * n] = value;
            }
        }
        for row in 0..col {
            matrix[row + col * n] = 0.0;
        }
    }
    Ok(matrix)
}

/// Frozen metric selected by warmup, using the canonical optimized math kernels.
#[derive(Debug)]
pub enum WarmupMetric {
    Identity(IdentityMetric),
    Diagonal(DiagonalMetric),
    Dense(DenseMetric),
}
impl EuclideanMetric for WarmupMetric {
    fn dimension(&self) -> usize {
        match self {
            Self::Identity(m) => m.dimension(),
            Self::Diagonal(m) => m.dimension(),
            Self::Dense(m) => m.dimension(),
        }
    }
    fn velocity(&self, p: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        match self {
            Self::Identity(m) => m.velocity(p, out),
            Self::Diagonal(m) => m.velocity(p, out),
            Self::Dense(m) => m.velocity(p, out),
        }
    }
    fn sample_momentum(&self, z: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        match self {
            Self::Identity(m) => m.sample_momentum(z, out),
            Self::Diagonal(m) => m.sample_momentum(z, out),
            Self::Dense(m) => m.sample_momentum(z, out),
        }
    }
    fn log_det(&self) -> f64 {
        match self {
            Self::Identity(m) => m.log_det(),
            Self::Diagonal(m) => m.log_det(),
            Self::Dense(m) => m.log_det(),
        }
    }
}
