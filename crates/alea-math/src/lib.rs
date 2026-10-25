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

/// Common imports for ordinary Alea workflows.
///
/// This is a convenience prelude, not a release or stability boundary. The
/// original owning modules remain available. Experimental methods require
/// explicit imports from their documented modules.
pub mod prelude;
