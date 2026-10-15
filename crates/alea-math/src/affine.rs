//! Frozen affine maps with diagonal scaling and spectral low-rank correction.

use crate::buffer::OwnedBuffer;
use thiserror::Error;

/// Invalid affine-map structure or operation dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AffineMapError {
    /// Shift and coordinate scales differ in length.
    #[error("affine shift and scale dimensions differ")]
    Dimension,
    /// Low-rank storage is not `dimension * rank`.
    #[error("invalid affine low-rank basis storage")]
    BasisStorage,
    /// A translation coordinate is non-finite.
    #[error("affine shift entry {index} must be finite")]
    NonFiniteShift { index: usize },
    /// A scale or spectral eigenvalue is not finite and strictly positive.
    #[error("affine scale or eigenvalue {index} must be finite and positive")]
    InvalidScale { index: usize },
    /// A basis entry is non-finite or the columns are not orthonormal.
    #[error("affine low-rank directions are not orthonormal")]
    InvalidBasis,
    /// Spectral correction magnifies supported basis error beyond the precision budget.
    #[error("affine low-rank correction exceeds the floating-point conditioning budget")]
    IllConditionedCorrection,
    /// Source and destination buffers must match the map dimension.
    #[error("affine operation dimension mismatch")]
    VectorDimension,
    /// A finite input produced an unrepresentable output.
    #[error("affine operation produced a non-finite value")]
    Numerical,
}

/// `x = shift + S [I + U diag(sqrt(lambda)-1) U^T] z`.
///
/// `S` is positive diagonal and the columns of `U` are orthonormal. The inverse,
/// transpose action and log determinant cost `O(d r + d)` and allocate nothing.
/// The type exposes no mutation, making a constructed map suitable for retained
/// sampling after a separate warmup learner has been consumed.
#[derive(Debug, Clone)]
pub struct LowRankAffineMap {
    shift: OwnedBuffer,
    scales: OwnedBuffer,
    basis: OwnedBuffer,
    eigenvalues: OwnedBuffer,
    forward: OwnedBuffer,
    inverse: OwnedBuffer,
    log_abs_det: f64,
}

impl LowRankAffineMap {
    /// Validates and takes ownership of column-major spectral storage.
    ///
    /// # Errors
    /// Rejects inconsistent dimensions, non-positive parameters, non-finite
    /// values and basis columns that are not orthonormal to absolute tolerance
    /// `1e-10`.
    pub fn new(
        shift: OwnedBuffer,
        scales: OwnedBuffer,
        basis: OwnedBuffer,
        eigenvalues: OwnedBuffer,
    ) -> Result<Self, AffineMapError> {
        let dim = shift.len();
        let rank = eigenvalues.len();
        if scales.len() != dim {
            return Err(AffineMapError::Dimension);
        }
        if rank > dim || dim.checked_mul(rank) != Some(basis.len()) {
            return Err(AffineMapError::BasisStorage);
        }
        for (index, &value) in scales.iter().chain(eigenvalues.iter()).enumerate() {
            if !value.is_finite() || value <= 0.0 || !value.recip().is_finite() {
                return Err(AffineMapError::InvalidScale { index });
            }
        }
        if let Some(index) = shift.iter().position(|value| !value.is_finite()) {
            return Err(AffineMapError::NonFiniteShift { index });
        }
        if basis.iter().any(|value| !value.is_finite()) {
            return Err(AffineMapError::InvalidBasis);
        }
        let mut max_gram_error = 0.0_f64;
        for right in 0..rank {
            let u = &basis[right * dim..(right + 1) * dim];
            for left in 0..=right {
                let v = &basis[left * dim..(left + 1) * dim];
                let dot = u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                if !dot.is_finite() || (dot - expected).abs() > 1e-10 {
                    return Err(AffineMapError::InvalidBasis);
                }
                max_gram_error = max_gram_error.max((dot - expected).abs());
            }
        }
        if rank > 0 {
            let smallest = eigenvalues.iter().copied().fold(1.0_f64, f64::min);
            let largest = eigenvalues.iter().copied().fold(1.0_f64, f64::max);
            let condition = largest / smallest;
            let rounding = 16.0 * f64::EPSILON * (dim as f64 + rank as f64);
            let estimated_error = condition * (rank as f64 * max_gram_error + rounding);
            if !estimated_error.is_finite() || estimated_error > 1e-6 {
                return Err(AffineMapError::IllConditionedCorrection);
            }
        }
        let log_abs_det = scales.iter().map(|value| value.ln()).sum::<f64>()
            + 0.5 * eigenvalues.iter().map(|value| value.ln()).sum::<f64>();
        if !log_abs_det.is_finite() {
            return Err(AffineMapError::Numerical);
        }
        let forward = OwnedBuffer::from_fn(rank, |i| eigenvalues[i].sqrt() - 1.0);
        let inverse = OwnedBuffer::from_fn(rank, |i| eigenvalues[i].sqrt().recip() - 1.0);
        Ok(Self {
            shift,
            scales,
            basis,
            eigenvalues,
            forward,
            inverse,
            log_abs_det,
        })
    }

    /// Number of input and output coordinates.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.shift.len()
    }

    /// Number of corrected standardized directions.
    #[inline]
    pub fn rank(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Translation in target coordinates.
    pub fn shift(&self) -> &[f64] {
        &self.shift
    }

    /// Positive target-coordinate scales.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Column-major orthonormal directions in standardized coordinates.
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }

    /// Positive covariance multipliers in corrected directions.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Constant log absolute determinant of the forward Jacobian.
    #[inline]
    pub fn log_abs_det(&self) -> f64 {
        self.log_abs_det
    }

    fn check(&self, src: &[f64], dst: &[f64]) -> Result<(), AffineMapError> {
        if src.len() != self.dimension() || dst.len() != self.dimension() {
            Err(AffineMapError::VectorDimension)
        } else {
            Ok(())
        }
    }

    /// Applies the forward map without allocating.
    pub fn forward(&self, latent: &[f64], target: &mut [f64]) -> Result<(), AffineMapError> {
        self.check(latent, target)?;
        target.copy_from_slice(latent);
        let dim = self.dimension();
        for k in 0..self.rank() {
            let u = &self.basis[k * dim..(k + 1) * dim];
            let dot = u.iter().zip(latent).map(|(a, b)| a * b).sum::<f64>();
            let coefficient = self.forward[k] * dot;
            for (value, &direction) in target.iter_mut().zip(u) {
                *value += coefficient * direction;
            }
        }
        for ((value, &scale), &shift) in target
            .iter_mut()
            .zip(self.scales.iter())
            .zip(self.shift.iter())
        {
            *value = shift + scale * *value;
        }
        if target.iter().any(|value| !value.is_finite()) {
            return Err(AffineMapError::Numerical);
        }
        Ok(())
    }

    /// Applies the exact inverse without allocating.
    pub fn inverse(&self, target: &[f64], latent: &mut [f64]) -> Result<(), AffineMapError> {
        self.check(target, latent)?;
        for (((value, &coordinate), &shift), &scale) in latent
            .iter_mut()
            .zip(target)
            .zip(self.shift.iter())
            .zip(self.scales.iter())
        {
            *value = (coordinate - shift) / scale;
        }
        let dim = self.dimension();
        for k in 0..self.rank() {
            let u = &self.basis[k * dim..(k + 1) * dim];
            let dot = u.iter().zip(latent.iter()).map(|(a, b)| a * b).sum::<f64>();
            let coefficient = self.inverse[k] * dot;
            for (value, &direction) in latent.iter_mut().zip(u) {
                *value += coefficient * direction;
            }
        }
        if latent.iter().any(|value| !value.is_finite()) {
            return Err(AffineMapError::Numerical);
        }
        Ok(())
    }

    /// Applies the transpose Jacobian to a target-coordinate covector.
    pub fn pullback(
        &self,
        target_score: &[f64],
        latent_score: &mut [f64],
    ) -> Result<(), AffineMapError> {
        self.check(target_score, latent_score)?;
        for ((value, &score), &scale) in latent_score
            .iter_mut()
            .zip(target_score)
            .zip(self.scales.iter())
        {
            *value = scale * score;
        }
        let dim = self.dimension();
        for k in 0..self.rank() {
            let u = &self.basis[k * dim..(k + 1) * dim];
            let dot = u
                .iter()
                .zip(latent_score.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let coefficient = self.forward[k] * dot;
            for (value, &direction) in latent_score.iter_mut().zip(u) {
                *value += coefficient * direction;
            }
        }
        if latent_score.iter().any(|value| !value.is_finite()) {
            return Err(AffineMapError::Numerical);
        }
        Ok(())
    }
}
