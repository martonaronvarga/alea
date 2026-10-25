//! Analytic distributions and DDM observation models.
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "branch-hints", feature(likely_unlikely))]
mod error;
pub mod gaussian;
pub mod latent;
pub mod mixture;
pub mod modeling;
pub mod multivariate;
pub mod scalar;
pub mod wiener;
pub use gaussian::Gaussian;
