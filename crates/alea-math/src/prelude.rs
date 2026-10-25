//! Common imports; experimental APIs remain in their named modules.
pub use crate::buffer::OwnedBuffer;
pub use crate::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
    InverseDenseMetric, InverseDiagonalMetric, MetricError,
};
