//! Target-bound Markov chains with typed, fallible transitions.
#![forbid(unsafe_code)]
pub mod adapt;
mod canonical;
pub mod config;
pub mod hamiltonian;
pub mod hmc;
pub mod integrator;
pub mod rwmh;
pub use hmc::{Hmc, HmcOptions, HmcTransition};
pub use rwmh::{Rwmh, RwmhOptions, RwmhTransition};

/// A chain owns its current point and reusable workspace; the target is bound at construction.
/// Implementations must retain a coherent live point after rejection or error.
pub trait MarkovChain {
    type Error: std::error::Error + 'static;
    type Transition;
    fn position(&self) -> &[f64];
    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R)
    -> Result<Self::Transition, Self::Error>;
}
