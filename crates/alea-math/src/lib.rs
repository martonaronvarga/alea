//! Aligned storage, Euclidean metrics and numerical primitives. No model or FFI dependency.
#![cfg_attr(
    all(
        feature = "simd",
        any(test, not(any(feature = "faer", feature = "openblas")))
    ),
    feature(portable_simd)
)]
pub mod affine;
pub mod buffer;
#[cfg(feature = "faer")]
pub mod fisher;
pub mod interpolation;
pub mod metric;
pub mod numeric;
pub mod random;

/// Validated storage and constant Euclidean geometry for reference inference.
///
/// Structured, low-rank, fitted, and position-dependent geometry remains
/// available through its named modules without inheriting this adoption status.
pub mod reference {
    pub use crate::buffer::OwnedBuffer;
    pub use crate::metric::{
        CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
        InverseDenseMetric, InverseDiagonalMetric, MetricError,
    };
}
