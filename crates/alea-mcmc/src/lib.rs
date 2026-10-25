//! Target-bound Markov chains with typed, fallible transitions.
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![forbid(unsafe_code)]
pub mod adapt;
/// Experimental advanced integrators and joint tuning. See the module documentation and maturity ledger.
pub mod advanced;
mod atlas;
/// Experimental Atlas proposals. See the module documentation and maturity ledger.
pub mod atlas_sampler;
/// Experimental batched Hamiltonian integration. See the module documentation and maturity ledger.
pub mod batch;
mod canonical;
/// Experimental shared ChEES adaptation. See the module documentation and maturity ledger.
pub mod chees;
pub mod config;
/// Experimental persistent and delayed-rejection dynamics. See the module documentation and maturity ledger.
pub mod ghmc;
pub mod hamiltonian;
pub mod hmc;
pub mod integrator;
/// Experimental isokinetic dynamics; MCLMC is explicitly approximate. See the module documentation and maturity ledger.
pub mod mams;
pub mod nuts;
/// Experimental manifold dynamics; RMHMC exactness remains pending. See the module documentation and maturity ledger.
pub mod riemannian;
pub mod rwmh;
mod trajectory_weight;
/// Experimental locally refined WALNUTS transitions. See the module documentation and maturity ledger.
pub mod walnuts;
pub use alea_core::target::LogDensityGradient;
pub use hmc::{Hmc, HmcOptions, HmcTransition};
pub use nuts::{Nuts, NutsOptions, NutsTransition};
pub use rwmh::{Rwmh, RwmhOptions, RwmhTransition};

/// An externally implementable, target-bound Markov chain.
///
/// A chain owns its current point and reusable workspace; the target is bound at
/// construction. The position dimension must remain fixed and every published
/// coordinate must be finite. A successful step publishes exactly one coherent
/// retained state, including a repeat after rejection. An error must leave the
/// previous position coherent and observable. RNG consumption need not roll back.
pub trait MarkovChain {
    /// Original transition failure, preserved by runners.
    type Error: std::error::Error + 'static;
    /// Sampler-specific metadata for one completed transition.
    type Transition;
    /// Current coherent retained position.
    fn position(&self) -> &[f64];
    /// Attempts one transition using caller-owned randomness.
    ///
    /// # Errors
    /// Returns the original transition failure without publishing partial state.
    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R)
    -> Result<Self::Transition, Self::Error>;
}

/// Explicitly approximate-target kernels.
///
/// These methods deliberately omit an exact-target correction. Keeping them under
/// a named boundary prevents approximate inference from looking interchangeable
/// with the exact-target reference surface.
pub mod approximate {
    pub use crate::mams::{
        ApproximateMclmc, ApproximateMclmcOptions, ApproximateMclmcTransition, ApproximateTarget,
    };
}

/// Common imports for ordinary Alea workflows.
///
/// This is a convenience prelude, not a release or stability boundary. The
/// original owning modules remain available. Experimental methods require
/// explicit imports from their documented modules.
pub mod prelude;
