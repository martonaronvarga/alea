//! Fallible fused targets, transactional state and constrained model composition.
#![forbid(unsafe_code)]
pub mod capability;
pub mod density;
pub mod distribution;
mod error;
pub mod model;
pub mod state_space;
pub mod target;
pub mod transform;
pub mod transport;

/// Validated target and transform contracts used by Alea's reference inference path.
///
/// This additive namespace identifies the current S2 component boundary. It does
/// not imply semver stability while the workspace remains pre-release. Optional
/// derivative, state-space, and learned-transport capabilities stay in their
/// explicitly named modules and are not promoted through this facade.
pub mod reference {
    pub use crate::density::{DensityPoint, LogDensity, evaluate as evaluate_density};
    pub use crate::target::{
        DimensionError, EvaluationError, EvaluationWorkspace, LogDensityGradient, PointState,
        evaluate,
    };
    pub use crate::transform::{ParameterBlock, ParameterLayout, Transform, TransformError};
}
