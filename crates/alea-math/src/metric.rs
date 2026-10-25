//! Validated Euclidean momentum mass matrices: `p ~ N(0, M)`, velocity `M^-1 p`.
//!
//! Constructors validate structure once. Operations reuse caller-owned output
//! storage and check lengths before writing. Finite inputs can still overflow for
//! ill-conditioned metrics; samplers must check numerical results.
#![deny(missing_docs)]

use crate::buffer::OwnedBuffer;
use thiserror::Error;

mod low_rank;
pub use low_rank::LowRankDiagonalMetric;
mod structured;
pub use structured::{BlockDiagonalMetric, BlockMetric, SparseCholeskyMetric};

/// Invalid mass-matrix structure or metric operation dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MetricError {
    /// The retained directions must be an orthonormal family of at most `dim` vectors.
    #[error("low-rank directions are not orthonormal")]
    InvalidBasis,
    /// Spectral corrections amplify rounding/basis error beyond the supported budget.
    #[error("low-rank correction exceeds the floating-point conditioning budget")]
    IllConditionedCorrection,
    /// The square matrix cannot fit in a Rust slice.
    #[error("matrix dimension {dim} exceeds the supported storage size")]
    DimensionOverflow {
        /// Requested matrix dimension.
        dim: usize,
    },
    /// Storage does not contain exactly `dim * dim` entries.
    #[error("matrix storage length mismatch: expected {expected}, got {actual}")]
    StorageLength {
        /// Required dimension or storage length.
        expected: usize,
        /// Observed dimension or storage length.
        actual: usize,
    },
    /// A diagonal mass or factor pivot is not finite and strictly positive.
    #[error("diagonal entry {index} must be finite and positive")]
    InvalidDiagonal {
        /// Zero-based index of the invalid entry.
        index: usize,
    },
    /// A factor entry is NaN or infinite.
    #[error("factor entry at row {row}, column {column} must be finite")]
    NonFiniteFactor {
        /// Zero-based matrix row.
        row: usize,
        /// Zero-based matrix column.
        column: usize,
    },
    /// Full column-major storage must have zero entries above its diagonal.
    #[error("factor entry at row {row}, column {column} must be zero")]
    NonZeroUpperTriangle {
        /// Zero-based matrix row.
        row: usize,
        /// Zero-based matrix column.
        column: usize,
    },
    /// Neither short nor oversized vector buffers are accepted.
    #[error(
        "metric dimension {expected} requires matching vectors, got source {source_len} and destination {destination_len}"
    )]
    VectorLength {
        /// Required dimension or storage length.
        expected: usize,
        /// Actual source vector length.
        source_len: usize,
        /// Actual destination vector length.
        destination_len: usize,
    },
    /// Compressed sparse-column offsets or row indices are malformed.
    #[error("invalid sparse Cholesky structure")]
    InvalidSparseStructure,
}

#[inline]
fn check_vector_lengths(dim: usize, src: &[f64], dst: &[f64]) -> Result<(), MetricError> {
    if src.len() != dim || dst.len() != dim {
        return Err(MetricError::VectorLength {
            expected: dim,
            source_len: src.len(),
            destination_len: dst.len(),
        });
    }
    Ok(())
}

#[cfg(feature = "openblas")]
#[inline]
fn assert_vector_lengths(dim: usize, src: &[f64], dst: &[f64]) {
    assert_eq!(src.len(), dim, "metric source dimension mismatch");
    assert_eq!(dst.len(), dim, "metric destination dimension mismatch");
}

fn matrix_len(dim: usize) -> Result<usize, MetricError> {
    dim.checked_mul(dim)
        .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
        .ok_or(MetricError::DimensionOverflow { dim })
}

#[cfg(feature = "openblas")]
#[inline]
fn dense_apply_inverse_openblas(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    assert_vector_lengths(dim, src, dst);
    assert_eq!(Some(lower_col_major.len()), dim.checked_mul(dim));
    if dim == 0 {
        return;
    }
    let n = i32::try_from(dim).expect("validated f64 square matrix fits the BLAS dimension");

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    // SAFETY: lengths are checked in release builds before mutation. The matrix
    // holds dim*dim initialized f64s, dst holds dim disjoint writable f64s, and
    // n=lda=dim>0 fits i32. Unit stride stays within dst. Both synchronous calls
    // borrow these buffers only for the call; no pointers escape.
    unsafe {
        cblas::dtrsv(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::None,
            cblas::Diagonal::Generic,
            n,
            lower_col_major,
            n,
            dst,
            1,
        );
        cblas::dtrsv(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::Ordinary,
            cblas::Diagonal::Generic,
            n,
            lower_col_major,
            n,
            dst,
            1,
        );
    }
}

#[cfg(feature = "openblas")]
#[inline]
fn dense_apply_sqrt_openblas(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    assert_vector_lengths(dim, src, dst);
    assert_eq!(Some(lower_col_major.len()), dim.checked_mul(dim));
    if dim == 0 {
        return;
    }
    let n = i32::try_from(dim).expect("validated f64 square matrix fits the BLAS dimension");

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    // SAFETY: release-mode checks establish dim*dim readable matrix entries and
    // dim disjoint writable vector entries. n=lda=dim>0 fits i32 and incx=1.
    // The synchronous call retains no pointers; Rust borrows cover its lifetime.
    unsafe {
        cblas::dtrmv(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::None,
            cblas::Diagonal::Generic,
            n,
            lower_col_major,
            n,
            dst,
            1,
        );
    }
}

#[cfg(all(
    feature = "simd",
    any(test, not(any(feature = "faer", feature = "openblas")))
))]
#[inline]
fn dense_apply_inverse_simd(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use std::simd::Simd;
    use std::simd::num::SimdFloat;

    // Portable lanes do not imply AVX-512/AVX2 on the baseline CPU target.
    // Separate multiply/add avoids scalar software-FMA calls when the compiler
    // cannot assume hardware FMA, and matches the scalar kernel's arithmetic.
    const LANES: usize = 8;
    type Vf = Simd<f64, LANES>;

    debug_assert_eq!(lower_col_major.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    // Forward solve: L y = b
    //
    // Column-oriented algorithm. For column k of L (contiguous in col-major):
    //   dst[k]      /= L[k, k]
    //   dst[k+1..]  -= col_k[k+1..] * dst[k]        ← contiguous SIMD update
    //
    for k in 0..dim {
        // col_k: full column k, length dim, contiguous in memory
        let col_k = &lower_col_major[k * dim..][..dim];

        let diag = col_k[k];
        dst[k] /= diag;
        let xk = dst[k];

        let start = k + 1;
        if start >= dim {
            continue;
        }

        let col_rest = &col_k[start..]; // L[start..dim, k] — contiguous
        let dst_rest = &mut dst[start..]; // same range in dst
        let n = dst_rest.len();
        let vxk = Vf::splat(xk);

        let mut i = 0;

        while i + LANES <= n {
            let lv = Vf::from_slice(&col_rest[i..]);
            let dv = Vf::from_slice(&dst_rest[i..]);
            (dv - lv * vxk).copy_to_slice(&mut dst_rest[i..]);
            i += LANES;
        }
        // Scalar tail
        while i < n {
            dst_rest[i] -= col_rest[i] * xk;
            i += 1;
        }
    }

    // Backward solve: Lᵀ x = y
    //
    // Lᵀ[i, j] = L[j, i] = lower_col_major[j + i*dim] = col_i[j]
    // Column i of L is contiguous and covers rows 0..dim.
    //
    // Back-substitution:
    //   for i = dim-1 downto 0:
    //     x[i] = (y[i] - Σ_{j>i} col_i[j] · x[j]) / L[i,i]
    //
    // split_at_mut(i+1) separates the write target (dst[i]) from the
    // read range (dst[i+1..dim]) without unsafe, satisfying the borrow checker.
    for i in (0..dim).rev() {
        let col_i = &lower_col_major[i * dim..][..dim]; // col i of L — contiguous

        // dst[..i+1]: write target includes dst[i]
        // dst[i+1..]: read-only source for the dot product
        let (dst_lo, dst_hi) = dst.split_at_mut(i + 1);

        let acc = if !dst_hi.is_empty() {
            let col_rest = &col_i[i + 1..]; // L[i+1..dim, i] = Lᵀ[i, i+1..dim]
            let n = dst_hi.len();
            let mut j = 0;

            let mut vs0 = Vf::splat(0.0);
            let mut vs1 = Vf::splat(0.0);
            let mut vs2 = Vf::splat(0.0);
            let mut vs3 = Vf::splat(0.0);
            while j + 4 * LANES <= n {
                vs0 += Vf::from_slice(&col_rest[j..]) * Vf::from_slice(&dst_hi[j..]);
                vs1 +=
                    Vf::from_slice(&col_rest[j + LANES..]) * Vf::from_slice(&dst_hi[j + LANES..]);
                vs2 += Vf::from_slice(&col_rest[j + 2 * LANES..])
                    * Vf::from_slice(&dst_hi[j + 2 * LANES..]);
                vs3 += Vf::from_slice(&col_rest[j + 3 * LANES..])
                    * Vf::from_slice(&dst_hi[j + 3 * LANES..]);
                j += 4 * LANES;
            }
            let vsum = (vs0 + vs1) + (vs2 + vs3);
            let mut s = vsum.reduce_sum();
            while j < n {
                s += col_rest[j] * dst_hi[j];
                j += 1;
            }
            s
        } else {
            0.0
        };

        dst_lo[i] = (dst_lo[i] - acc) / col_i[i];
    }
}

#[cfg(all(
    feature = "simd",
    any(test, not(any(feature = "faer", feature = "openblas")))
))]
#[inline]
fn dense_apply_sqrt_simd(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use std::simd::Simd;

    const LANES: usize = 8;
    const BLOCK: usize = 16;
    type Vf = Simd<f64, LANES>;

    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    dst.fill(0.0);

    let mut jb = 0;
    while jb < dim {
        let jend = (jb + BLOCK).min(dim);

        for j in jb..jend {
            let zj = Vf::splat(src[j]);
            let col = &lower_col_major[j * dim..(j + 1) * dim];

            let mut i = j;
            while i + LANES <= dim {
                let c = Vf::from_slice(&col[i..i + LANES]);
                let d = Vf::from_slice(&dst[i..i + LANES]);

                let updated = zj * c + d;
                updated.copy_to_slice(&mut dst[i..i + LANES]);

                i += LANES;
            }

            while i < dim {
                dst[i] += col[i] * src[j];
                i += 1;
            }
        }

        jb = jend;
    }
}

#[cfg(all(feature = "faer", not(feature = "openblas")))]
#[inline]
fn dense_apply_inverse_faer(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use faer::linalg::triangular_solve::{
        solve_lower_triangular_in_place, solve_upper_triangular_in_place,
    };
    use faer::{MatMut, MatRef, Par};

    debug_assert_eq!(lower_col_major.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    let l = MatRef::from_column_major_slice(lower_col_major, dim, dim);
    solve_lower_triangular_in_place(
        l,
        MatMut::from_column_major_slice_mut(dst, dim, 1),
        Par::Seq,
    );

    let lt = l.transpose();
    solve_upper_triangular_in_place(
        lt,
        MatMut::from_column_major_slice_mut(dst, dim, 1),
        Par::Seq,
    );
}

#[cfg(all(feature = "faer", not(feature = "openblas")))]
#[inline]
fn dense_apply_sqrt_faer(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use faer::linalg::matmul::triangular::{BlockStructure, matmul_with_conj};
    use faer::{Accum, Conj, MatMut, MatRef, Par};

    debug_assert_eq!(lower_col_major.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    let lhs = MatRef::from_column_major_slice(lower_col_major, dim, dim);
    let rhs = MatRef::from_column_major_slice(src, dim, 1);
    let dst = MatMut::from_column_major_slice_mut(dst, dim, 1);

    matmul_with_conj(
        dst,
        BlockStructure::Rectangular,
        Accum::Replace,
        lhs,
        BlockStructure::TriangularLower,
        Conj::No,
        rhs,
        BlockStructure::Rectangular,
        Conj::No,
        1.0,
        Par::Seq,
    );
}

#[allow(dead_code)]
#[inline]
fn dense_apply_inverse_default(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    const BLOCK: usize = 32;

    let mut jb = 0;
    while jb < dim {
        let jend = (jb + BLOCK).min(dim);

        for i in jb..jend {
            let mut acc = dst[i];

            let mut kb = 0;
            while kb < jb {
                let kend = (kb + BLOCK).min(jb);
                let mut k = kb;
                while k < kend {
                    acc -= lower_col_major[i + k * dim] * dst[k];
                    k += 1;
                }
                kb = kend;
            }

            let mut k = jb;
            while k < i {
                acc -= lower_col_major[i + k * dim] * dst[k];
                k += 1;
            }

            dst[i] = acc / lower_col_major[i + i * dim];
        }

        jb = jend;
    }

    let mut jb = dim;
    while jb > 0 {
        let j0 = jb.saturating_sub(BLOCK);

        for i in (j0..jb).rev() {
            let mut acc = dst[i];

            let mut kb = jb;
            while kb < dim {
                let kend = (kb + BLOCK).min(dim);
                let mut k = kb;
                while k < kend {
                    acc -= lower_col_major[k + i * dim] * dst[k];
                    k += 1;
                }
                kb = kend;
            }

            let mut k = i + 1;
            while k < jb {
                acc -= lower_col_major[k + i * dim] * dst[k];
                k += 1;
            }

            dst[i] = acc / lower_col_major[i + i * dim];
        }

        jb = j0;
    }
}

#[allow(dead_code)]
#[inline]
fn dense_apply_sqrt_default(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    dst.fill(0.0);

    const BLOCK: usize = 32;

    let mut jb = 0;
    while jb < dim {
        let jend = (jb + BLOCK).min(dim);

        for j in jb..jend {
            let zj = src[j];
            let col = &lower_col_major[j * dim..(j + 1) * dim];

            let mut i = j;
            while i + 4 <= dim {
                dst[i] += col[i] * zj;
                dst[i + 1] += col[i + 1] * zj;
                dst[i + 2] += col[i + 2] * zj;
                dst[i + 3] += col[i + 3] * zj;
                i += 4;
            }

            while i < dim {
                dst[i] += col[i] * zj;
                i += 1;
            }
        }

        jb = jend;
    }
}

/// An externally implementable constant Euclidean momentum mass matrix.
///
/// For a symmetric positive-definite mass `M = L L^T`, momentum sampling maps
/// `z ~ N(0, I)` to `p = L z`, velocity is `M^-1 p`, kinetic energy is
/// `p^T M^-1 p / 2`, and [`EuclideanMetric::log_det`] is `log(det(M))`.
/// The dimension and matrix must remain fixed while borrowed.
///
/// Every operation must check all vector dimensions before mutating output and
/// must reuse caller storage. Aliasing source and output is not expressible through
/// the safe interface. Finite values may still overflow for ill-conditioned
/// metrics; samplers validate numerical results before publishing a transition.
pub trait EuclideanMetric {
    /// Fixed coordinate dimension of this mass matrix.
    fn dimension(&self) -> usize;

    /// Writes `M^-1 p` into `velocity`.
    ///
    /// # Errors
    /// Returns [`MetricError::VectorLength`] without mutating `velocity` when
    /// either slice does not have [`Self::dimension`] entries.
    fn velocity(&self, momentum: &[f64], velocity: &mut [f64]) -> Result<(), MetricError>;

    /// Maps a standard-normal draw to momentum distributed as `N(0, M)`.
    ///
    /// # Errors
    /// Returns [`MetricError::VectorLength`] without mutating `momentum` when
    /// either slice does not have [`Self::dimension`] entries.
    fn sample_momentum(
        &self,
        standard_normal: &[f64],
        momentum: &mut [f64],
    ) -> Result<(), MetricError>;

    /// Returns `log(det(M))`, which must be finite for a valid metric.
    fn log_det(&self) -> f64;

    /// Computes `p^T M^-1 p / 2` using caller-owned scratch.
    ///
    /// # Errors
    /// Returns the error from [`Self::velocity`]. Shape errors leave `scratch`
    /// untouched.
    fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> Result<f64, MetricError> {
        self.kinetic_energy_with_velocity(momentum, scratch)
    }

    /// Computes kinetic energy and writes the matching `M^-1 p` into `velocity`.
    /// Factor representations may share intermediate products to evaluate energy
    /// as a sum of squares. Neither geometry nor momentum may change between this
    /// call and use of the returned velocity.
    ///
    /// # Errors
    /// Returns the error from [`Self::velocity`]. Shape errors leave output unchanged.
    fn kinetic_energy_with_velocity(
        &self,
        momentum: &[f64],
        velocity: &mut [f64],
    ) -> Result<f64, MetricError> {
        self.velocity(momentum, velocity)?;
        Ok(0.5
            * momentum
                .iter()
                .zip(velocity)
                .map(|(p, v)| p * *v)
                .sum::<f64>())
    }
}

/// Unit mass matrix, including the empty zero-dimensional matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct IdentityMetric {
    dim: usize,
}

impl IdentityMetric {
    /// Creates a unit mass matrix of the given dimension without allocating.
    #[inline]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EuclideanMetric for IdentityMetric {
    #[inline]
    fn dimension(&self) -> usize {
        self.dim
    }
    #[inline]
    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dim, src, dst)?;
        dst.copy_from_slice(src);
        Ok(())
    }

    #[inline]
    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dim, src, dst)?;
        dst.copy_from_slice(src);
        Ok(())
    }

    fn kinetic_energy_with_velocity(
        &self,
        momentum: &[f64],
        velocity: &mut [f64],
    ) -> Result<f64, MetricError> {
        check_vector_lengths(self.dim, momentum, velocity)?;
        if momentum.len() < 16 {
            velocity.copy_from_slice(momentum);
            return Ok(0.5
                * momentum
                    .iter()
                    .map(|momentum| momentum * momentum)
                    .sum::<f64>());
        }
        Ok(pulp::Arch::new().dispatch(DiagonalEnergy::<true> {
            p: momentum,
            out: velocity,
            inverse: &[],
            sqrt: &[],
        }))
    }

    #[inline]
    fn log_det(&self) -> f64 {
        0.0
    }
}

/// Positive finite diagonal entries of the momentum mass matrix `M`.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DiagonalMetric {
    diag: OwnedBuffer,
}

impl DiagonalMetric {
    /// Frozen diagonal masses, in coordinate order. Useful for lossless restart.
    pub fn as_slice(&self) -> &[f64] {
        &self.diag
    }
    /// Takes ownership of diagonal masses without copying or changing them.
    /// Empty storage represents the zero-dimensional matrix.
    ///
    /// # Errors
    /// Returns [`MetricError::InvalidDiagonal`] for the first non-finite or
    /// non-positive entry. Small positive values are not silently clamped.
    ///
    /// # Examples
    /// ```
    /// use alea_math::buffer::OwnedBuffer;
    /// use alea_math::metric::{DiagonalMetric, EuclideanMetric, MetricError};
    /// let masses = OwnedBuffer::from_fn(2, |i| [4.0, 9.0][i]);
    /// let metric = DiagonalMetric::new(masses)?;
    /// let mut momentum = [0.0; 2];
    /// metric.sample_momentum(&[1.0, 2.0], &mut momentum)?;
    /// assert_eq!(momentum, [2.0, 6.0]);
    /// # Ok::<(), MetricError>(())
    /// ```
    pub fn new(diag: OwnedBuffer) -> Result<Self, MetricError> {
        for (index, &d) in diag.iter().enumerate() {
            if !d.is_finite() || d <= 0.0 {
                return Err(MetricError::InvalidDiagonal { index });
            }
        }

        Ok(Self { diag })
    }
}

impl EuclideanMetric for DiagonalMetric {
    #[inline]
    fn dimension(&self) -> usize {
        self.diag.len()
    }

    #[inline]
    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension(), src, dst)?;
        for ((out, x), d) in dst.iter_mut().zip(src).zip(self.diag.iter()) {
            *out = *x / *d;
        }
        Ok(())
    }

    #[inline]
    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension(), src, dst)?;
        for ((out, x), d) in dst.iter_mut().zip(src).zip(self.diag.iter()) {
            *out = *x * d.sqrt();
        }
        Ok(())
    }

    #[inline]
    fn log_det(&self) -> f64 {
        self.diag.iter().map(|x| x.ln()).sum()
    }
}

/// Validated lower-triangular `L` in full column-major storage.
/// [`DenseMetric`] interprets it as a mass factor; [`InverseDenseMetric`] as an
/// inverse-mass factor. The owning metric determines the convention.
/// This validates a supplied factor; it does not factorize a mass matrix or
/// certify its condition number or the representability of every operation.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct CholeskyFactor {
    dim: usize,
    lower_col_major: OwnedBuffer,
}

impl CholeskyFactor {
    /// Takes ownership of a factor without copying or altering entries.
    /// All entries must be finite, the diagonal strictly positive, and the upper
    /// triangle exactly zero (signed zero is accepted). Dimension zero is valid.
    ///
    /// # Errors
    /// Returns a typed error for size overflow, incorrect storage length, invalid
    /// diagonal, non-finite entries, or a nonzero upper triangle.
    pub fn new_lower(dim: usize, lower_col_major: OwnedBuffer) -> Result<Self, MetricError> {
        let expected = matrix_len(dim)?;
        if lower_col_major.len() != expected {
            return Err(MetricError::StorageLength {
                expected,
                actual: lower_col_major.len(),
            });
        }
        for column in 0..dim {
            for row in 0..dim {
                let value = lower_col_major[row + column * dim];
                if row == column && (!value.is_finite() || value <= 0.0) {
                    return Err(MetricError::InvalidDiagonal { index: row });
                }
                if !value.is_finite() {
                    return Err(MetricError::NonFiniteFactor { row, column });
                }
                if row < column && value != 0.0 {
                    return Err(MetricError::NonZeroUpperTriangle { row, column });
                }
            }
        }
        Ok(Self {
            dim,
            lower_col_major,
        })
    }

    #[inline]
    /// Dimension of the square factor.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    #[inline]
    /// Immutable full column-major factor storage.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.lower_col_major.as_slice()
    }
}

/// Dense momentum mass matrix represented by its validated Cholesky factor.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DenseMetric {
    factor: CholeskyFactor,
}

impl DenseMetric {
    /// Constructs `M = L L^T` from an already validated factor.
    pub fn new(factor: CholeskyFactor) -> Self {
        Self { factor }
    }
    #[inline]
    /// Borrows the immutable factor, preserving its validation invariants.
    pub fn factor(&self) -> &CholeskyFactor {
        &self.factor
    }
}

impl EuclideanMetric for DenseMetric {
    #[inline]
    fn dimension(&self) -> usize {
        self.factor.dim
    }

    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        let n = self.factor.dim;
        check_vector_lengths(n, src, dst)?;
        if n == 0 {
            return Ok(());
        }
        let l = self.factor.as_slice();
        #[cfg(feature = "openblas")]
        {
            dense_apply_inverse_openblas(n, l, src, dst);
        }
        #[cfg(all(not(feature = "openblas"), feature = "faer"))]
        {
            dense_apply_inverse_faer(n, l, src, dst);
        }
        #[cfg(all(not(feature = "openblas"), not(feature = "faer"), feature = "simd"))]
        {
            dense_apply_inverse_simd(n, l, src, dst);
        }
        #[cfg(all(
            not(feature = "openblas"),
            not(feature = "faer"),
            not(feature = "simd")
        ))]
        {
            dense_apply_inverse_default(n, l, src, dst);
        }
        Ok(())
    }

    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        let n = self.factor.dim;
        check_vector_lengths(n, src, dst)?;
        if n == 0 {
            return Ok(());
        }
        let l = self.factor.as_slice();
        #[cfg(feature = "openblas")]
        {
            dense_apply_sqrt_openblas(n, l, src, dst);
        }
        #[cfg(all(not(feature = "openblas"), feature = "faer"))]
        {
            dense_apply_sqrt_faer(n, l, src, dst);
        }
        #[cfg(all(not(feature = "openblas"), not(feature = "faer"), feature = "simd"))]
        {
            dense_apply_sqrt_simd(n, l, src, dst);
        }
        #[cfg(all(
            not(feature = "openblas"),
            not(feature = "faer"),
            not(feature = "simd")
        ))]
        {
            dense_apply_sqrt_default(n, l, src, dst);
        }
        Ok(())
    }

    #[inline]
    fn log_det(&self) -> f64 {
        let n = self.factor.dim;
        let l = self.factor.as_slice();
        let mut sum = 0.0;
        for i in 0..n {
            sum += l[i + i * n].ln();
        }
        2.0 * sum
    }
}

/// Diagonal inverse momentum mass `G = M^-1`, with cached square-root scales.
/// This is distinct from [`DiagonalMetric`], whose input entries are masses.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct InverseDiagonalMetric {
    inverse: OwnedBuffer,
    sqrt_inverse: OwnedBuffer,
    momentum_scale: OwnedBuffer,
}

impl InverseDiagonalMetric {
    /// Takes positive finite inverse masses. Construction caches both scales;
    /// every hot operation reuses caller-owned storage.
    /// # Errors
    /// Rejects the first nonpositive or nonfinite entry without clamping it.
    pub fn new(inverse: OwnedBuffer) -> Result<Self, MetricError> {
        for (index, value) in inverse.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(MetricError::InvalidDiagonal { index });
            }
        }
        let sqrt_inverse = OwnedBuffer::from_fn(inverse.len(), |i| inverse[i].sqrt());
        let momentum_scale = OwnedBuffer::from_fn(inverse.len(), |i| 1.0 / sqrt_inverse[i]);
        Ok(Self {
            inverse,
            sqrt_inverse,
            momentum_scale,
        })
    }
    /// Frozen inverse masses, in coordinate order; sufficient for lossless restart.
    pub fn as_slice(&self) -> &[f64] {
        &self.inverse
    }
    /// Replace inverse masses and their cached factors without allocating.
    /// Validation precedes every write, so an error preserves all previous values.
    /// The owning sampler must keep geometry fixed within a trajectory.
    /// # Errors
    /// Rejects a wrong length or any nonpositive/nonfinite inverse mass.
    pub fn set_inverse_masses(&mut self, inverse: &[f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.inverse.len(), inverse, &self.inverse)?;
        for (index, value) in inverse.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(MetricError::InvalidDiagonal { index });
            }
        }
        for (((out, sqrt), scale), value) in self
            .inverse
            .iter_mut()
            .zip(self.sqrt_inverse.iter_mut())
            .zip(self.momentum_scale.iter_mut())
            .zip(inverse)
        {
            *out = *value;
            *sqrt = value.sqrt();
            *scale = 1.0 / *sqrt;
        }
        Ok(())
    }
}

impl EuclideanMetric for InverseDiagonalMetric {
    #[inline]
    fn dimension(&self) -> usize {
        self.inverse.len()
    }
    #[inline]
    fn velocity(&self, p: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension(), p, out)?;
        for ((out, p), g) in out.iter_mut().zip(p).zip(self.inverse.iter()) {
            *out = p * g;
        }
        Ok(())
    }
    #[inline]
    fn sample_momentum(&self, z: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension(), z, out)?;
        for ((out, z), scale) in out.iter_mut().zip(z).zip(self.momentum_scale.iter()) {
            *out = z * scale;
        }
        Ok(())
    }
    fn kinetic_energy_with_velocity(&self, p: &[f64], out: &mut [f64]) -> Result<f64, MetricError> {
        check_vector_lengths(self.dimension(), p, out)?;
        if p.len() >= 16 {
            return Ok(pulp::Arch::new().dispatch(DiagonalEnergy::<false> {
                p,
                out,
                inverse: &self.inverse,
                sqrt: &self.sqrt_inverse,
            }));
        }
        let mut energy = 0.0;
        for (((out, p), g), sqrt) in out
            .iter_mut()
            .zip(p)
            .zip(self.inverse.iter())
            .zip(self.sqrt_inverse.iter())
        {
            let u = p * sqrt;
            energy += (0.5 * u) * u;
            *out = p * g;
        }
        Ok(energy)
    }
    fn log_det(&self) -> f64 {
        -self.inverse.iter().map(|g| g.ln()).sum::<f64>()
    }
}

// Nonnegative factored kinetic terms use independent SIMD accumulators. Scaling
// each term by 1/2 before summing avoids overflow of an otherwise representable
// kinetic energy and reduces the summation error bound versus one long chain.
struct DiagonalEnergy<'a, const IDENTITY: bool> {
    p: &'a [f64],
    out: &'a mut [f64],
    inverse: &'a [f64],
    sqrt: &'a [f64],
}
impl<const IDENTITY: bool> pulp::WithSimd for DiagonalEnergy<'_, IDENTITY> {
    type Output = f64;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
        let (p, pt) = S::as_simd_f64s(self.p);
        let (out, ot) = S::as_mut_simd_f64s(self.out);
        let (g, gt) = S::as_simd_f64s(if IDENTITY { self.p } else { self.inverse });
        let (sqrt, st) = S::as_simd_f64s(if IDENTITY { self.p } else { self.sqrt });
        let half = simd.splat_f64s(0.5);
        let mut sums = [simd.splat_f64s(0.0); 4];
        let (pc, pr) = p.as_chunks::<4>();
        let (oc, or) = out.as_chunks_mut::<4>();
        let (gc, gr) = g.as_chunks::<4>();
        let (sc, sr) = sqrt.as_chunks::<4>();
        for (((p, out), g), sqrt) in pc.iter().zip(oc).zip(gc).zip(sc) {
            for i in 0..4 {
                let u = if IDENTITY {
                    p[i]
                } else {
                    simd.mul_f64s(p[i], sqrt[i])
                };
                out[i] = if IDENTITY {
                    p[i]
                } else {
                    simd.mul_f64s(p[i], g[i])
                };
                let e = simd.mul_f64s(simd.mul_f64s(half, u), u);
                sums[i] = simd.add_f64s(sums[i], e);
            }
        }
        for (((p, out), g), sqrt) in pr.iter().zip(or).zip(gr).zip(sr) {
            let u = if IDENTITY {
                *p
            } else {
                simd.mul_f64s(*p, *sqrt)
            };
            *out = if IDENTITY { *p } else { simd.mul_f64s(*p, *g) };
            sums[0] = simd.add_f64s(sums[0], simd.mul_f64s(simd.mul_f64s(half, u), u));
        }
        let mut energy = simd.reduce_sum_f64s(simd.add_f64s(
            simd.add_f64s(sums[0], sums[1]),
            simd.add_f64s(sums[2], sums[3]),
        ));
        for (i, (p, out)) in pt.iter().zip(ot).enumerate() {
            let u = if IDENTITY { *p } else { p * st[i] };
            *out = if IDENTITY { *p } else { p * gt[i] };
            energy += (0.5 * u) * u;
        }
        energy
    }
}

/// Dense inverse momentum mass `G = M^-1 = L L^T`.
/// Stores the factor of G directly, without constructing or factorizing M.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct InverseDenseMetric {
    factor: CholeskyFactor,
}
impl InverseDenseMetric {
    /// Owns an already validated lower factor of inverse mass G.
    pub fn new(factor: CholeskyFactor) -> Self {
        Self { factor }
    }
    /// Borrows the exact frozen inverse-mass factor, including for checkpointing.
    pub fn factor(&self) -> &CholeskyFactor {
        &self.factor
    }

    fn transpose_product(&self, p: &[f64], out: &mut [f64]) {
        let n = self.dimension();
        let l = self.factor.as_slice();
        for (col, out) in out.iter_mut().enumerate() {
            *out = l[col + col * n..(col + 1) * n]
                .iter()
                .zip(&p[col..])
                .map(|(l, p)| l * p)
                .sum();
        }
    }
    fn lower_product_in_place(&self, out: &mut [f64]) {
        let n = self.dimension();
        let l = self.factor.as_slice();
        // Descending rows leave every required source entry untouched.
        for row in (0..n).rev() {
            let mut value = 0.0;
            for col in 0..=row {
                value += l[row + col * n] * out[col];
            }
            out[row] = value;
        }
    }
}
impl EuclideanMetric for InverseDenseMetric {
    #[inline]
    fn dimension(&self) -> usize {
        self.factor.dimension()
    }
    fn velocity(&self, p: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension(), p, out)?;
        self.transpose_product(p, out);
        self.lower_product_in_place(out);
        Ok(())
    }
    fn kinetic_energy_with_velocity(&self, p: &[f64], out: &mut [f64]) -> Result<f64, MetricError> {
        check_vector_lengths(self.dimension(), p, out)?;
        self.transpose_product(p, out);
        let energy = out.iter().map(|u| (0.5 * u) * u).sum();
        self.lower_product_in_place(out);
        Ok(energy)
    }
    fn sample_momentum(&self, z: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        let n = self.dimension();
        check_vector_lengths(n, z, out)?;
        let l = self.factor.as_slice();
        for row in (0..n).rev() {
            let mut value = z[row];
            for col in row + 1..n {
                value -= l[col + row * n] * out[col];
            }
            out[row] = value / l[row + row * n];
        }
        Ok(())
    }
    fn log_det(&self) -> f64 {
        -2.0 * (0..self.dimension())
            .map(|i| self.factor.as_slice()[i + i * self.dimension()].ln())
            .sum::<f64>()
    }
}

#[cfg(all(test, feature = "simd"))]
mod simd_tests {
    use super::*;

    #[test]
    fn simd_matches_scalar_for_tails_and_unaligned_subslices() {
        for n in [0, 1, 2, 7, 8, 9, 31, 32, 33, 65] {
            let lower = OwnedBuffer::from_fn(n * n, |k| {
                let (i, j) = (k % n, k / n);
                if i == j {
                    2.0 + i as f64 / 100.0
                } else if i > j {
                    0.01 * (i + j + 1) as f64
                } else {
                    0.0
                }
            });
            let input = OwnedBuffer::from_fn(n + 1, |i| (i as f64 + 1.0) / 3.0);
            let mut output = OwnedBuffer::new(n + 1);
            let mut expected = vec![0.0; n];
            // Eight-byte offsets intentionally remove the 64-byte base alignment.
            let src = &input[1..];
            let dst = &mut output[1..];
            dense_apply_sqrt_default(n, &lower, src, &mut expected);
            dense_apply_sqrt_simd(n, &lower, src, dst);
            for (actual, expected) in dst.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-11 * expected.abs().max(1.0));
            }
            dense_apply_inverse_default(n, &lower, src, &mut expected);
            dense_apply_inverse_simd(n, &lower, src, dst);
            for (actual, expected) in dst.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-11 * expected.abs().max(1.0));
            }
        }
    }
}
