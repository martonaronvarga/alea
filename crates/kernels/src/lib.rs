#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "branch-hints", feature(likely_unlikely))]
#[cfg(feature = "std-autodiff")]
pub mod autodiff;
pub mod buffer;
pub mod density;
pub mod dist;
pub mod error;
pub mod extension;
pub mod gradient_check;
pub mod kernel;
pub mod metric;
pub mod model;
pub mod numeric;
pub mod proposal;
pub mod state;
pub mod state_space;
pub mod target;
pub mod transform;
