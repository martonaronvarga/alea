//! Common probability laws and observation models.
pub use crate::Gaussian;
pub use crate::wiener::{
    BatchStrategy, Boundary, Quadrature, Wiener4, Wiener4Params, Wiener5, Wiener5Params, Wiener7,
    Wiener7Params, WienerBatchModel, WienerBatchWorkspace, WienerModel, WienerObservation,
    WienerObservations, WienerOptions,
};
pub use alea_core::distribution::{ContinuousDistribution, Distribution};
