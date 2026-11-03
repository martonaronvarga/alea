//! Checked row-major GEMM adapters. Accelerated reductions are not compensated
//! and need not reproduce scalar reduction bits. No global thread policy changes.

/// Execution choice for dense matrix products. Optional variants require the
/// corresponding Cargo feature; enabling a feature does not change the default.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatrixKernel {
    /// Ordered compensated reference arithmetic.
    #[default]
    Compensated,
    /// Faer matrix kernels, explicitly sequential.
    #[cfg(feature = "faer")]
    Faer,
    /// LP64 CBLAS GEMM. Thread count belongs to the host BLAS configuration.
    #[cfg(feature = "openblas")]
    Blas,
}

/// Attempts `out = op(a) * op(b)` (or adds the product when `add` is true).
/// Dimensions describe the logical product `(m, k, n)`; transpose flags describe
/// the physical row-major operands. Returns false without writes for the
/// compensated policy, invalid lengths/overflow, or unsupported BLAS dimensions.
/// Empty products are handled without calling foreign code. Numerical admission
/// of inputs/outputs belongs to the caller; no clipping or finite repair occurs.
pub fn try_gemm(
    kernel: MatrixKernel,
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    dimensions: (usize, usize, usize),
    transpose: (bool, bool),
    add: bool,
) -> bool {
    let (m, k, n) = dimensions;
    if kernel == MatrixKernel::Compensated
        || m.checked_mul(k) != Some(a.len())
        || k.checked_mul(n) != Some(b.len())
        || m.checked_mul(n) != Some(out.len())
    {
        return false;
    }
    if m == 0 || n == 0 || k == 0 {
        if !add {
            out.fill(0.);
        }
        return true;
    }
    let (ta, tb) = transpose;
    let _ = (ta, tb); // Both optional backends can be disabled.
    match kernel {
        MatrixKernel::Compensated => false,
        #[cfg(feature = "faer")]
        MatrixKernel::Faer => {
            use faer::{Accum, MatMut, MatRef, Par};
            let a = if ta {
                MatRef::from_row_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_row_major_slice(a, m, k)
            };
            let b = if tb {
                MatRef::from_row_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_row_major_slice(b, k, n)
            };
            faer::linalg::matmul::matmul(
                MatMut::from_row_major_slice_mut(out, m, n),
                if add { Accum::Add } else { Accum::Replace },
                a,
                b,
                1.,
                Par::Seq,
            );
            true
        }
        #[cfg(feature = "openblas")]
        MatrixKernel::Blas => {
            use cblas::{Layout, Transpose, dgemm};
            let (Ok(m), Ok(k), Ok(n)) = (i32::try_from(m), i32::try_from(k), i32::try_from(n))
            else {
                return false;
            };
            // SAFETY: checked exact slice lengths and positive LP64 dimensions.
            // Physical row widths are lda=(ta?m:k), ldb=(tb?k:n), ldc=n.
            // Rust borrows guarantee output cannot overlap either input. All
            // initialized buffers stay borrowed until this synchronous call ends.
            unsafe {
                dgemm(
                    Layout::RowMajor,
                    if ta {
                        Transpose::Ordinary
                    } else {
                        Transpose::None
                    },
                    if tb {
                        Transpose::Ordinary
                    } else {
                        Transpose::None
                    },
                    m,
                    n,
                    k,
                    1.,
                    a,
                    if ta { m } else { k },
                    b,
                    if tb { k } else { n },
                    if add { 1. } else { 0. },
                    out,
                    n,
                );
            }
            true
        }
    }
}
