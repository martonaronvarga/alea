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

/// Common imports for ordinary Alea workflows.
///
/// This is a convenience prelude, not a release or stability boundary. The
/// original owning modules remain available. Experimental methods require
/// explicit imports from their documented modules.
pub mod prelude;
