#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "branch-hints", feature(likely_unlikely))]
#[cfg(feature = "std-autodiff")]
compile_error!(
    "std-autodiff is not implemented: a validated Enzyme compiler and target adapter are required (roadmap M2); use analytic LogDensityGradient targets"
);
pub mod buffer;
pub mod density;
pub mod dist;
pub mod error;
pub mod extension;
pub mod kernel;
pub mod metric;
pub mod numeric;
pub mod proposal;
pub mod state;
pub mod state_space;
pub mod target;
