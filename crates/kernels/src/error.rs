use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProbError {
    #[error("Invalid parameters provided: {0}")]
    InvalidParameters(String),

    #[error("Buffer allocation or layout failed: {0}")]
    BufferError(String),

    #[error("Numerical instability encountered: {0}")]
    NumericalError(String),

    #[error("Point outside distribution support: {0}")]
    OutOfSupport(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("MCMC sampler error: {0}")]
    SamplerError(String),
}

pub type Result<T> = std::result::Result<T, ProbError>;

/// Typed, allocation-free transform error vocabulary for the M2 model boundary.
/// Numerical failures do not clamp a parameter onto its support boundary.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformError {
    #[error("transform dimensions overflow or do not define a valid shape")]
    InvalidDimension,
    #[error("transform rounded to the boundary at index {index}")]
    PrecisionLoss { index: usize },
    #[error("transform dimension mismatch: expected {expected}, got {actual}")]
    Dimension { expected: usize, actual: usize },
    #[error("transform bounds are invalid at index {index}")]
    InvalidBounds { index: usize },
    #[error("transform input is outside its domain at index {index}")]
    Domain { index: usize },
    #[error("transform input is non-finite at index {index}")]
    NonFiniteInput { index: usize },
    #[error("transform output is non-finite at index {index}")]
    NonFiniteOutput { index: usize },
    #[error("transform log-Jacobian is non-finite")]
    NonFiniteJacobian,
}
