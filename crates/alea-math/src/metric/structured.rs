//! Block and sparse constant Euclidean metrics without dense materialization.

use super::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, MetricError, check_vector_lengths,
};
use crate::buffer::OwnedBuffer;

/// One independently factorized block of a block-diagonal mass matrix.
#[derive(Debug)]
pub enum BlockMetric {
    /// Positive diagonal mass entries.
    Diagonal(DiagonalMetric),
    /// Dense Cholesky factor local to this block.
    Dense(DenseMetric),
}

impl BlockMetric {
    /// Validates a diagonal block.
    pub fn diagonal(masses: OwnedBuffer) -> Result<Self, MetricError> {
        DiagonalMetric::new(masses).map(Self::Diagonal)
    }

    /// Uses a previously validated dense Cholesky block.
    pub fn dense(factor: CholeskyFactor) -> Self {
        Self::Dense(DenseMetric::new(factor))
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Diagonal(metric) => metric.dimension(),
            Self::Dense(metric) => metric.dimension(),
        }
    }

    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        match self {
            Self::Diagonal(metric) => metric.velocity(src, dst),
            Self::Dense(metric) => metric.velocity(src, dst),
        }
    }

    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        match self {
            Self::Diagonal(metric) => metric.sample_momentum(src, dst),
            Self::Dense(metric) => metric.sample_momentum(src, dst),
        }
    }

    fn log_det(&self) -> f64 {
        match self {
            Self::Diagonal(metric) => metric.log_det(),
            Self::Dense(metric) => metric.log_det(),
        }
    }
}

/// A mass matrix composed from independent diagonal or dense blocks.
///
/// Storage is the sum of block storage rather than a `d*d` matrix. Actions are
/// dispatched once per block and remain allocation-free.
#[derive(Debug)]
pub struct BlockDiagonalMetric {
    blocks: Box<[BlockMetric]>,
    dimension: usize,
}

impl BlockDiagonalMetric {
    /// Validates the total dimension and takes ownership of all blocks.
    pub fn new(blocks: Vec<BlockMetric>) -> Result<Self, MetricError> {
        let mut dimension = 0_usize;
        for block in &blocks {
            dimension = dimension
                .checked_add(block.dimension())
                .ok_or(MetricError::DimensionOverflow { dim: usize::MAX })?;
        }
        Ok(Self {
            blocks: blocks.into_boxed_slice(),
            dimension,
        })
    }

    /// Read-only block decomposition.
    pub fn blocks(&self) -> &[BlockMetric] {
        &self.blocks
    }
}

impl EuclideanMetric for BlockDiagonalMetric {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension, src, dst)?;
        let mut offset = 0;
        for block in &self.blocks {
            let end = offset + block.dimension();
            block.velocity(&src[offset..end], &mut dst[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension, src, dst)?;
        let mut offset = 0;
        for block in &self.blocks {
            let end = offset + block.dimension();
            block.sample_momentum(&src[offset..end], &mut dst[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    fn log_det(&self) -> f64 {
        self.blocks.iter().map(BlockMetric::log_det).sum()
    }
}

/// Sparse lower Cholesky factor `L` in compressed sparse-column storage.
///
/// The represented mass is `M = L L^T`. Every column must begin with its
/// positive diagonal entry, followed by strictly increasing row indices below
/// the diagonal. Momentum sampling and triangular solves cost `O(nnz)` and do
/// not construct a dense matrix.
#[derive(Debug)]
pub struct SparseCholeskyMetric {
    dimension: usize,
    column_offsets: Box<[usize]>,
    row_indices: Box<[usize]>,
    values: OwnedBuffer,
}

impl SparseCholeskyMetric {
    /// Validates canonical lower-CSC storage.
    ///
    /// # Errors
    /// Rejects malformed offsets, missing/invalid diagonal entries, unsorted or
    /// out-of-triangle row indices, and non-finite values.
    pub fn new(
        dimension: usize,
        column_offsets: Box<[usize]>,
        row_indices: Box<[usize]>,
        values: OwnedBuffer,
    ) -> Result<Self, MetricError> {
        if column_offsets.len() != dimension.checked_add(1).unwrap_or(0)
            || column_offsets.first().copied() != Some(0)
            || column_offsets.last().copied() != Some(values.len())
            || row_indices.len() != values.len()
        {
            return Err(MetricError::InvalidSparseStructure);
        }
        for column in 0..dimension {
            let start = column_offsets[column];
            let end = column_offsets[column + 1];
            if start >= end || end > values.len() || row_indices[start] != column {
                return Err(MetricError::InvalidSparseStructure);
            }
            let diagonal = values[start];
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(MetricError::InvalidDiagonal { index: column });
            }
            let mut previous = column;
            for index in start + 1..end {
                let row = row_indices[index];
                if row <= previous || row >= dimension {
                    return Err(MetricError::InvalidSparseStructure);
                }
                if !values[index].is_finite() {
                    return Err(MetricError::NonFiniteFactor { row, column });
                }
                previous = row;
            }
        }
        Ok(Self {
            dimension,
            column_offsets,
            row_indices,
            values,
        })
    }

    /// Number of explicitly stored Cholesky entries.
    pub fn nonzeros(&self) -> usize {
        self.values.len()
    }

    /// Canonical CSC column offsets.
    pub fn column_offsets(&self) -> &[usize] {
        &self.column_offsets
    }

    /// Canonical CSC row indices.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Cholesky entries corresponding to [`Self::row_indices`].
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl EuclideanMetric for SparseCholeskyMetric {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension, src, dst)?;
        dst.copy_from_slice(src);
        for column in 0..self.dimension {
            let start = self.column_offsets[column];
            let end = self.column_offsets[column + 1];
            dst[column] /= self.values[start];
            let solved = dst[column];
            for index in start + 1..end {
                dst[self.row_indices[index]] -= self.values[index] * solved;
            }
        }
        for column in (0..self.dimension).rev() {
            let start = self.column_offsets[column];
            let end = self.column_offsets[column + 1];
            let mut residual = dst[column];
            for index in start + 1..end {
                residual -= self.values[index] * dst[self.row_indices[index]];
            }
            dst[column] = residual / self.values[start];
        }
        Ok(())
    }

    fn sample_momentum(&self, src: &[f64], dst: &mut [f64]) -> Result<(), MetricError> {
        check_vector_lengths(self.dimension, src, dst)?;
        dst.fill(0.0);
        for (column, &coefficient) in src.iter().enumerate() {
            for index in self.column_offsets[column]..self.column_offsets[column + 1] {
                dst[self.row_indices[index]] += self.values[index] * coefficient;
            }
        }
        Ok(())
    }

    fn log_det(&self) -> f64 {
        2.0 * (0..self.dimension)
            .map(|column| self.values[self.column_offsets[column]].ln())
            .sum::<f64>()
    }
}
