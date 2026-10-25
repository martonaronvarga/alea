//! Target-bound Markov chains with typed, fallible transitions.
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![forbid(unsafe_code)]
pub mod adapt;
pub mod advanced;
mod atlas;
pub mod atlas_sampler;
pub mod batch;
mod canonical;
pub mod chees;
pub mod config;
pub mod ghmc;
pub mod hamiltonian;
pub mod hmc;
pub mod integrator;
pub mod mams;
pub mod nuts;
pub mod riemannian;
pub mod rwmh;
mod trajectory_weight;
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

/// Validated exact-target kernels and finite warmup used by the reference path.
///
/// This namespace is intentionally narrower than the crate root. In particular it
/// excludes Fisher fitting, WALNUTS policy, learned geometry, alternative dynamics,
/// approximate MCLMC, advanced integration, and Riemannian research. The aliases
/// share the existing implementations and carry no separate numerical behavior.
pub mod reference {
    pub use crate::MarkovChain;
    pub use crate::adapt::{
        AdaptationError, ControlledHmcWarmupResult, CovarianceError, DualAveraging, HmcWarmup,
        HmcWarmupReport, HmcWarmupResult, MetricKind, OnlineCovariance, SearchOptions,
        SearchResult, WarmupConfigError, WarmupError, WarmupMetric, WarmupOptions, WarmupStage,
        WarmupSummary, WindowSchedule, find_reasonable_step_size, warmup_rwmh,
    };
    pub use crate::adapt::{
        ControlledNutsWarmupResult, NutsWarmup, NutsWarmupError, NutsWarmupObservation,
        NutsWarmupReport, NutsWarmupResult, find_reasonable_nuts_step_size,
    };
    pub use crate::config::{AcceptanceTarget, SamplerConfigError, TreeDepth};
    pub use crate::hamiltonian::Divergence;
    pub use crate::hmc::{Hmc, HmcConfigError, HmcError, HmcOptions, HmcTransition, StepSize};
    pub use crate::nuts::{
        Nuts, NutsConfigError, NutsError, NutsOptions, NutsTermination, NutsTransition,
    };
    pub use crate::rwmh::{NumericalRejection, Rwmh, RwmhError, RwmhOptions, RwmhTransition};
}
