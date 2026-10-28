//! Dense square matrix layouts and allocation-free lower Cholesky factorization.
//! Layout is part of the type. Factorization reads the lower triangle of a
//! caller-validated symmetric matrix; it performs no symmetrization or repair.
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}
/// Dense square matrix storage convention.
pub trait MatrixLayout: sealed::Sealed {
    fn index(order: usize, row: usize, column: usize) -> usize;
}
/// Contiguous matrix rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMajor;
/// Contiguous matrix columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnMajor;
impl sealed::Sealed for RowMajor {}
impl sealed::Sealed for ColumnMajor {}
impl MatrixLayout for RowMajor {
    fn index(order: usize, row: usize, column: usize) -> usize {
        row * order + column
    }
}
impl MatrixLayout for ColumnMajor {
    fn index(order: usize, row: usize, column: usize) -> usize {
        row + column * order
    }
}
/// Shape or arithmetic failure. Numerical failure leaves scratch partially written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MatrixError {
    #[error("square matrix storage does not match its order")]
    Shape,
    #[error("matrix pivot {index} is not finite and positive")]
    NotPositiveDefinite { index: usize },
    #[error("matrix operation is not finite at ({row}, {column})")]
    NonFinite { row: usize, column: usize },
}
/// Immutable dense square matrix with an explicit layout and checked shape.
#[derive(Debug, Clone, Copy)]
pub struct SquareMatrixRef<'a, L: MatrixLayout> {
    values: &'a [f64],
    order: usize,
    layout: PhantomData<L>,
}
impl<'a, L: MatrixLayout> SquareMatrixRef<'a, L> {
    /// # Errors
    /// Rejects incorrect storage length and overflowing dimensions.
    pub fn new(order: usize, values: &'a [f64]) -> Result<Self, MatrixError> {
        if order.checked_mul(order) != Some(values.len()) {
            return Err(MatrixError::Shape);
        }
        Ok(Self {
            values,
            order,
            layout: PhantomData,
        })
    }
    pub fn order(&self) -> usize {
        self.order
    }
    pub fn as_slice(&self) -> &'a [f64] {
        self.values
    }
    /// Returns an entry, or `None` outside the matrix.
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        if row < self.order && column < self.order {
            Some(self.values[L::index(self.order, row, column)])
        } else {
            None
        }
    }
}
/// Computes a lower Cholesky factor in place using sequential residual subtraction.
/// The caller establishes symmetry; only lower entries are consumed. Upper
/// entries are zero on success. Zero order is valid. Shape errors precede writes.
/// # Errors
/// Reports invalid shape, a non-positive/nonfinite pivot, or a nonfinite factor.
pub fn cholesky_in_place<L: MatrixLayout>(
    order: usize,
    values: &mut [f64],
) -> Result<(), MatrixError> {
    if order.checked_mul(order) != Some(values.len()) {
        return Err(MatrixError::Shape);
    }
    for i in 0..order {
        for j in 0..=i {
            let mut residual = values[L::index(order, i, j)];
            for k in 0..j {
                residual -= values[L::index(order, i, k)] * values[L::index(order, j, k)];
            }
            let entry = if i == j {
                if residual <= 0. || !residual.is_finite() {
                    return Err(MatrixError::NotPositiveDefinite { index: i });
                }
                residual.sqrt()
            } else {
                residual / values[L::index(order, j, j)]
            };
            if !entry.is_finite() {
                return Err(MatrixError::NonFinite { row: i, column: j });
            }
            values[L::index(order, i, j)] = entry;
        }
        for j in i + 1..order {
            values[L::index(order, i, j)] = 0.;
        }
    }
    Ok(())
}
