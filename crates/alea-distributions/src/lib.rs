//! Analytic distributions and DDM observation models.
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "branch-hints", feature(likely_unlikely))]
pub mod error;
pub mod gaussian;
pub mod latent;
pub mod wiener;
pub use gaussian::Gaussian;
