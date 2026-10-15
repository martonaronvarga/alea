//! Target-bound Markov chains with typed, fallible transitions.
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![forbid(unsafe_code)]
pub mod adapt;
mod canonical;
pub mod config;
pub mod hamiltonian;
pub mod hmc;
pub mod integrator;
pub mod nuts;
pub mod rwmh;
mod trajectory_weight;
pub mod walnuts;
pub use hmc::{Hmc, HmcOptions, HmcTransition};
pub use nuts::{Nuts, NutsOptions, NutsTransition};
pub use rwmh::{Rwmh, RwmhOptions, RwmhTransition};
pub use walnuts::{
    Walnuts, WalnutsConfigError, WalnutsError, WalnutsMicrostepDistribution, WalnutsOptions,
    WalnutsTermination, WalnutsTransition,
};

/// A chain owns its current point and reusable workspace; the target is bound at construction.
/// Implementations must retain a coherent live point after rejection or error.
pub trait MarkovChain {
    type Error: std::error::Error + 'static;
    type Transition;
    fn position(&self) -> &[f64];
    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R)
    -> Result<Self::Transition, Self::Error>;
}
