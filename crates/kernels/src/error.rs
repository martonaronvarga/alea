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
