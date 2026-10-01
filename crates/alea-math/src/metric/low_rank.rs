//! Spectral low-rank corrections to a diagonal inverse mass.
use super::{EuclideanMetric, MetricError, check_vector_lengths};
use crate::buffer::OwnedBuffer;

/// `G = M^-1 = S (I + U diag(lambda - 1) U^T) S`.
///
/// `S` contains coordinate scales (not variances), `U` has orthonormal columns,
/// and all eigenvalues are positive. Values below one represent contractions;
/// unlike a positive-only `D + V V^T`, this represents both signs of correction.
/// Storage and operations cost O(d r + d); operations allocate nothing.
/// Extremely ill-conditioned spectral corrections are rejected, not clamped.
/// Finite inputs/scales can still overflow in operations; callers check results.
#[derive(Debug)]
pub struct LowRankDiagonalMetric {
    scales: OwnedBuffer,
    basis: OwnedBuffer,
    eigenvalues: OwnedBuffer,
    momentum_correction: OwnedBuffer,
}

impl LowRankDiagonalMetric {
    /// Validates coordinate scales, column-major directions and eigenvalues.
    /// Empty eigenvalues/basis give a diagonal inverse mass `S^2`.
    ///
    /// # Errors
    /// Rejects shape/size errors, non-representable positive scales/eigenvalues,
    /// non-finite directions and a non-orthonormal basis (absolute tolerance 1e-10).
    /// Also rejects corrections whose conservative rounding/basis-error estimate
    /// exceeds 1e-6: `kappa * (rank * max_gram_error + 16 eps (dim + rank))`,
    /// where `kappa = max(1,lambda_max)/min(1,lambda_min)`. This is an explicit
    /// supported-precision policy, not a rigorous error bound for arbitrary inputs.
    pub fn new(
        scales: OwnedBuffer,
        basis: OwnedBuffer,
        eigenvalues: OwnedBuffer,
    ) -> Result<Self, MetricError> {
        let dim = scales.len();
        let rank = eigenvalues.len();
        if rank > dim {
            return Err(MetricError::InvalidBasis);
        }
        let expected = dim
            .checked_mul(rank)
            .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
            .ok_or(MetricError::DimensionOverflow { dim })?;
        if basis.len() != expected {
            return Err(MetricError::StorageLength {
                expected,
                actual: basis.len(),
            });
        }
        Self::validate_scales(&scales)?;
        for (index, &value) in eigenvalues.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 || !value.recip().is_finite() {
                return Err(MetricError::InvalidDiagonal { index });
            }
        }
        let mut max_gram_error = 0.0_f64;
        for j in 0..rank {
            for i in 0..=j {
                let dot: f64 = basis[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(&basis[j * dim..(j + 1) * dim])
                    .map(|(a, b)| a * b)
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                if !dot.is_finite() || (dot - expected).abs() > 1e-10 {
                    return Err(MetricError::InvalidBasis);
                }
                max_gram_error = max_gram_error.max((dot - expected).abs());
            }
        }
        if rank > 0 {
            // I + U (lambda-1) U^T and its inverse square root both subtract
            // nearly equal quantities for extreme spectra. Even a basis within
            // the absolute orthogonality tolerance can then produce a negative G.
            let smallest = eigenvalues.iter().copied().fold(1.0_f64, f64::min);
            let largest = eigenvalues.iter().copied().fold(1.0_f64, f64::max);
            let condition = largest / smallest;
            let rounding = 16.0 * f64::EPSILON * (dim as f64 + rank as f64);
            let estimated_error = condition * (rank as f64 * max_gram_error + rounding);
            if !estimated_error.is_finite() || estimated_error > 1e-6 {
                return Err(MetricError::IllConditionedCorrection);
            }
        }
        let momentum_correction =
            OwnedBuffer::from_fn(rank, |i| eigenvalues[i].sqrt().recip() - 1.0);
        Ok(Self {
            scales,
            basis,
            eigenvalues,
            momentum_correction,
        })
    }

    fn validate_scales(scales: &[f64]) -> Result<(), MetricError> {
        for (index, &value) in scales.iter().enumerate() {
            let variance = value * value;
            if value <= 0.0
                || !variance.is_finite()
                || variance <= 0.0
                || !variance.recip().is_finite()
            {
                return Err(MetricError::InvalidDiagonal { index });
            }
        }
        Ok(())
    }

    /// Replaces scales without allocating; all validation precedes mutation.
    /// Directions and eigenvalues remain unchanged. For adaptation only: never
    /// change geometry within a trajectory or after warmup has been frozen.
    ///
    /// # Errors
    /// Rejects a mismatched length or invalid scale, leaving this metric unchanged.
    pub fn set_scales(&mut self, scales: &[f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension(), scales, &self.scales)?;
        Self::validate_scales(scales)?;
        self.scales.copy_from_slice(scales);
        Ok(())
    }

    /// Coordinate scales of the inverse mass.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }
    /// Orthonormal directions in column-major order.
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }
    /// Positive eigenvalues in standardized coordinates.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
    /// Number of retained directions.
    pub fn rank(&self) -> usize {
        self.eigenvalues.len()
    }
}

impl EuclideanMetric for LowRankDiagonalMetric {
    fn dimension(&self) -> usize {
        self.scales.len()
    }

    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        let dim = self.dimension();
        check_vector_lengths(dim, src, dst)?;
        // Accumulate in standardized coordinates, then apply the outer S.
        for ((out, &p), &s) in dst.iter_mut().zip(src).zip(self.scales.iter()) {
            *out = s * p;
        }
        for j in 0..self.rank() {
            let u = &self.basis[j * dim..(j + 1) * dim];
            let dot: f64 = u
                .iter()
                .zip(src)
                .zip(self.scales.iter())
                .map(|((&u, &p), &s)| u * (s * p))
                .sum();
            let correction = (self.eigenvalues[j] - 1.0) * dot;
            for (out, &u) in dst.iter_mut().zip(u) {
                *out += correction * u;
            }
        }
        for (out, &s) in dst.iter_mut().zip(self.scales.iter()) {
            *out *= s;
        }
        Ok(())
    }

    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        let dim = self.dimension();
        check_vector_lengths(dim, src, dst)?;
        dst.copy_from_slice(src);
        for j in 0..self.rank() {
            let u = &self.basis[j * dim..(j + 1) * dim];
            let dot: f64 = u.iter().zip(src).map(|(u, z)| u * z).sum();
            let correction = self.momentum_correction[j] * dot;
            for (out, &u) in dst.iter_mut().zip(u) {
                *out += correction * u;
            }
        }
        for (out, &s) in dst.iter_mut().zip(self.scales.iter()) {
            *out /= s;
        }
        Ok(())
    }

    fn log_det(&self) -> f64 {
        -2.0 * self.scales.iter().map(|s| s.ln()).sum::<f64>()
            - self.eigenvalues.iter().map(|v| v.ln()).sum::<f64>()
    }
}
