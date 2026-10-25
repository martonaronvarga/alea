#![deny(missing_docs)]
use thiserror::Error;

/// Typed, allocation-free transform error vocabulary for parameter transforms.
/// Numerical failures do not clamp a parameter onto its support boundary.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformError {
    #[error("transform dimensions overflow or do not define a valid shape")]
    /// Transform dimensions overflow or do not define a valid shape.
    InvalidDimension,
    #[error("transform rounded to the boundary at index {index}")]
    /// Transform rounded to the boundary at index `index`.
    PrecisionLoss {
        /// Zero-based index of the invalid entry.
        index: usize,
    },
    #[error("transform dimension mismatch: expected {expected}, got {actual}")]
    /// Transform dimension mismatch: expected `expected`, got `actual`.
    Dimension {
        /// Required dimension or storage length.
        expected: usize,
        /// Observed dimension or storage length.
        actual: usize,
    },
    #[error("transform bounds are invalid at index {index}")]
    /// Transform bounds are invalid at index `index`.
    InvalidBounds {
        /// Zero-based index of the invalid entry.
        index: usize,
    },
    #[error("transform input is outside its domain at index {index}")]
    /// Transform input is outside its domain at index `index`.
    Domain {
        /// Zero-based index of the invalid entry.
        index: usize,
    },
    #[error("transform input is non-finite at index {index}")]
    /// Transform input is non-finite at index `index`.
    NonFiniteInput {
        /// Zero-based index of the invalid entry.
        index: usize,
    },
    #[error("transform output is non-finite at index {index}")]
    /// Transform output is non-finite at index `index`.
    NonFiniteOutput {
        /// Zero-based index of the invalid entry.
        index: usize,
    },
    #[error("transform log-Jacobian is non-finite")]
    /// Transform log-Jacobian is non-finite.
    NonFiniteJacobian,
}
