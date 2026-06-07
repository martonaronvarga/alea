#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "branch-hints", feature(likely_unlikely))]
#![cfg_attr(feature = "std-autodiff", feature(autodiff))]
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
