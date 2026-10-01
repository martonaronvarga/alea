//! Aligned storage, Euclidean metrics and numerical primitives. No model or FFI dependency.
#![cfg_attr(
    all(
        feature = "simd",
        any(test, not(any(feature = "faer", feature = "openblas")))
    ),
    feature(portable_simd)
)]
pub mod buffer;
#[cfg(feature = "faer")]
pub mod fisher;
pub mod interpolation;
pub mod metric;
pub mod numeric;
pub mod random;
