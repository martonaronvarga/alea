//! Checked information-matrix factorization, independent of statistical targets.
//! No ridge, pseudoinverse, eigenvalue clipping or global parallelism mutation.
#![deny(missing_docs)]
use crate::{
    matrix::{ColumnMajor, MatrixError, RowMajor, SquareMatrixRef, cholesky_in_place},
    numeric::CompensatedSum,
};
use faer::{
    Mat, Par,
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{evd, triangular_solve::solve_lower_triangular_in_place},
};

/// Explicit numerical admission policy, dependent on the supplied coordinate units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationOptions {
    /// Require min/max eigenvalue strictly above this cutoff, in (0,1).
    pub relative_eigenvalue_cutoff: f64,
    /// Maximum pair asymmetry divided by max|matrix|, in [0,1).
    /// Zero requires exact symmetry; no absolute unit-dependent floor is added.
    pub relative_symmetry_tolerance: f64,
}
impl InformationOptions {
    /// Checks the declared cutoffs without evaluating or allocating a matrix.
    /// # Errors
    /// Requires eigenvalue cutoff in (0,1) and symmetry tolerance in [0,1).
    pub fn validate(self) -> Result<(), InformationError> {
        if !self.relative_eigenvalue_cutoff.is_finite()
            || self.relative_eigenvalue_cutoff <= 0.
            || self.relative_eigenvalue_cutoff >= 1.
            || !self.relative_symmetry_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.relative_symmetry_tolerance)
        {
            return Err(InformationError::Invalid);
        }
        Ok(())
    }
}
/// Invalid data/policy, failed factorization or unsupported numerical conditioning.
#[derive(Debug, thiserror::Error)]
pub enum InformationError {
    /// Invalid shape, empty dimension, overflowing storage or invalid policy.
    #[error("invalid information shape or numerical policy")]
    Invalid,
    /// Nonfinite input matrix or projection.
    #[error("nonfinite information input")]
    NonFinite,
    /// Matrix-scaled pair asymmetry exceeds the explicit tolerance.
    #[error("information matrix is asymmetric at ({row}, {column})")]
    Asymmetric {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// A nonpositive eigenvalue prevents unregularized covariance estimation.
    #[error("information matrix is not positive definite")]
    NotPositiveDefinite,
    /// Positive spectrum fails the explicit relative cutoff; no rank truncation.
    #[error("information reciprocal condition {reciprocal_condition:e} does not exceed cutoff")]
    IllConditioned {
        /// Estimated min/max eigenvalue ratio in the supplied units.
        reciprocal_condition: f64,
    },
    /// Original checked lower-factorization error.
    #[error(transparent)]
    Factor(#[from] MatrixError),
    /// Faer's symmetric eigensolver did not converge.
    #[error("information eigensolver did not converge")]
    EigenvalueConvergence,
    /// A normalized input, solve or covariance cannot be represented reliably.
    #[error("information calculation is not representable")]
    Numerical,
}
/// Immutable full-rank numerical factor, not a statistical validity certificate.
/// Eigenvalues are estimated by Faer on max-absolute-scaled input. Units affect
/// conditioning; the cutoff is not an exact algebraic rank or error certificate.
#[derive(Debug, Clone)]
pub struct InformationMatrix {
    lower: Mat<f64>,
    scale: f64,
    reciprocal_condition: f64,
    options: InformationOptions,
}
impl InformationMatrix {
    /// Validates both triangles, averages admitted symmetric pairs, estimates the
    /// spectrum and factors without regularization. All kernels are sequential.
    /// # Errors
    /// Rejects invalid/nonfinite/asymmetric, non-SPD, ill-conditioned or
    /// nonrepresentable input. Near-symmetric averaging is explicit policy, not
    /// evidence that the supplied matrix is a correct Hessian.
    pub fn new(
        matrix: SquareMatrixRef<'_, RowMajor>,
        options: InformationOptions,
    ) -> Result<Self, InformationError> {
        let d = matrix.order();
        options.validate()?;
        if d == 0 || !storage(d, d) {
            return Err(InformationError::Invalid);
        }
        let input = matrix.as_slice();
        if input.iter().any(|x| !x.is_finite()) {
            return Err(InformationError::NonFinite);
        }
        let scale = input.iter().fold(0_f64, |a, x| a.max(x.abs()));
        if scale == 0. {
            return Err(InformationError::NotPositiveDefinite);
        }
        let mut normalized = vec![0.; input.len()];
        for i in 0..d {
            for j in 0..=i {
                let a = input[i * d + j];
                let b = input[j * d + i];
                let aa = a / scale;
                let bb = b / scale;
                if (options.relative_symmetry_tolerance == 0. && a != b)
                    || (aa - bb).abs() > options.relative_symmetry_tolerance
                {
                    return Err(InformationError::Asymmetric { row: i, column: j });
                }
                if (a != 0. && aa == 0.) || (b != 0. && bb == 0.) {
                    return Err(InformationError::Numerical);
                }
                let average = if aa == bb { aa } else { 0.5 * aa + 0.5 * bb };
                normalized[i + j * d] = average;
                normalized[j + i * d] = average;
            }
        }
        let dense = Mat::from_fn(d, d, |i, j| normalized[i + j * d]);
        let mut eigenvalues = Diag::zeros(d);
        let mut scratch = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(
            d,
            evd::ComputeEigenvectors::No,
            Par::Seq,
            Default::default(),
        ));
        evd::self_adjoint_evd(
            dense.as_ref(),
            eigenvalues.as_mut(),
            None,
            Par::Seq,
            MemStack::new(&mut scratch),
            Default::default(),
        )
        .map_err(|_| InformationError::EigenvalueConvergence)?;
        let values = eigenvalues.column_vector();
        let mut smallest = f64::INFINITY;
        let mut largest = 0_f64;
        for i in 0..d {
            let value = values[i];
            if !value.is_finite() {
                return Err(InformationError::Numerical);
            }
            if value <= 0. {
                return Err(InformationError::NotPositiveDefinite);
            }
            smallest = smallest.min(value);
            largest = largest.max(value);
        }
        let reciprocal_condition = smallest / largest;
        if reciprocal_condition <= options.relative_eigenvalue_cutoff {
            return Err(InformationError::IllConditioned {
                reciprocal_condition,
            });
        }
        cholesky_in_place::<ColumnMajor>(d, &mut normalized)?;
        let lower = Mat::from_fn(d, d, |i, j| normalized[i + j * d]);
        Ok(Self {
            lower,
            scale,
            reciprocal_condition,
            options,
        })
    }
    /// Number of information coordinates.
    pub fn dimension(&self) -> usize {
        self.lower.nrows()
    }
    /// Original max-absolute matrix scale, retained separately from the factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }
    /// Estimated min/max eigenvalue ratio in the original supplied units.
    pub fn reciprocal_condition(&self) -> f64 {
        self.reciprocal_condition
    }
    /// Explicit policy used for symmetry and conditioning admission.
    pub fn options(&self) -> InformationOptions {
        self.options
    }
    /// Computes row-major covariance by triangular solves, not an inverse routine.
    /// # Errors
    /// Rejects nonrepresentable solves or variances, without substituting zeros.
    pub fn covariance(&self) -> Result<Vec<f64>, InformationError> {
        let d = self.dimension();
        let mut identity = vec![0.; d * d];
        for i in 0..d {
            identity[i * d + i] = 1.;
        }
        self.project(d, &identity)
    }
    /// Computes `J I^-1 J^T` for a row-major `rows`-by-dimension Jacobian.
    /// Solves `L W = J^T / sqrt(scale)` and forms its Gram matrix; never uses
    /// a previously rounded dense covariance as an intermediate. Zero Jacobian
    /// rows have zero variance. Nonzero rows with underflowed variance reject.
    /// # Errors
    /// Rejects invalid shapes/nonfinite inputs or nonrepresentable intermediate
    /// divisions, solves and Gram entries. This is not a full-range enclosure.
    pub fn project(&self, rows: usize, jacobian: &[f64]) -> Result<Vec<f64>, InformationError> {
        let d = self.dimension();
        if !storage(rows, d) || !storage(rows, rows) || rows.checked_mul(d) != Some(jacobian.len())
        {
            return Err(InformationError::Invalid);
        }
        if jacobian.iter().any(|x| !x.is_finite()) {
            return Err(InformationError::NonFinite);
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        let root = self.scale.sqrt();
        let mut whitened = Mat::from_fn(d, rows, |i, j| jacobian[j * d + i] / root);
        for j in 0..rows {
            for i in 0..d {
                if !whitened[(i, j)].is_finite()
                    || (jacobian[j * d + i] != 0. && whitened[(i, j)] == 0.)
                {
                    return Err(InformationError::Numerical);
                }
            }
        }
        solve_lower_triangular_in_place(self.lower.as_ref(), whitened.as_mut(), Par::Seq);
        if (0..rows).any(|j| (0..d).any(|i| !whitened[(i, j)].is_finite())) {
            return Err(InformationError::Numerical);
        }
        let mut result = vec![0.; rows * rows];
        for i in 0..rows {
            for j in 0..=i {
                let mut sum = CompensatedSum::default();
                for k in 0..d {
                    sum.add_product(whitened[(k, i)], whitened[(k, j)]);
                }
                let value = sum.total();
                if !value.is_finite()
                    || (i == j
                        && value <= 0.
                        && jacobian[i * d..(i + 1) * d].iter().any(|x| *x != 0.))
                {
                    return Err(InformationError::Numerical);
                }
                result[i * rows + j] = value;
                result[j * rows + i] = value;
            }
        }
        Ok(result)
    }
}
fn storage(rows: usize, columns: usize) -> bool {
    rows.checked_mul(columns)
        .is_some_and(|n| n <= isize::MAX as usize / size_of::<f64>())
}
