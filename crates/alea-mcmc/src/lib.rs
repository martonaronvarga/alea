//! Target-bound Markov chains with typed, fallible transitions.
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![forbid(unsafe_code)]
pub mod adapt;
pub mod advanced;
pub mod atlas;
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
pub use advanced::{
    AdvancedEvaluationCounters, AdvancedHmcError, AdvancedHmcOptions, AdvancedHmcTransition,
    ForceGradientHmc, ForceGradientMode, GaussianSplitHmc, SplitForceHmc,
};
pub use alea_core::target::LogDensityGradient;
pub use atlas::{
    AtlasConfigError, AtlasProbe, AtlasProbeError, AtlasProbeOptions, probe_local_curvature,
    probe_metric_curvature,
};
pub use batch::{BatchExecutionReport, BatchMachineError, BatchedLeapfrog};
pub use chees::{CheesHmc, CheesHmcOptions, CheesHmcTransition};
pub use ghmc::{
    DrGhmc, DrGhmcOptions, DrGhmcStage, DrGhmcTransition, Ghmc, GhmcConfigError, GhmcError,
    GhmcOptions, GhmcTransition, delayed_rejection_probability,
};
pub use hmc::{Hmc, HmcOptions, HmcTransition};
pub use mams::{
    ApproximateMclmc, ApproximateMclmcOptions, ApproximateMclmcTransition, ApproximateTarget, Mams,
    MamsDivergence, MamsError, MamsOptions, MamsTransition,
};
pub use nuts::{Nuts, NutsOptions, NutsTransition};
pub use riemannian::{
    FisherMetricConfigError, HierarchicalNormalFisherMetric, ImplicitSolveEvent, ImplicitStage,
    LogisticFisherMetric, Rmhmc, RmhmcConfigError, RmhmcCosts, RmhmcError, RmhmcOptions,
    RmhmcTermination, RmhmcTransition, SoftAbsConfigError, SoftAbsError, SoftAbsMetric,
    SoftAbsOptions,
};
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

/// Validated exact-target kernels and finite warmup used by the reference path.
///
/// This namespace is intentionally narrower than the crate root. In particular it
/// excludes Fisher fitting, WALNUTS policy, learned geometry, alternative dynamics,
/// approximate MCLMC, advanced integration, and Riemannian research. The aliases
/// share the existing implementations and carry no separate numerical behavior.
pub mod reference {
    pub use crate::MarkovChain;
    pub use crate::adapt::{
        AdaptationError, CovarianceError, DualAveraging, HmcWarmup, HmcWarmupReport,
        HmcWarmupResult, MetricKind, OnlineCovariance, SearchOptions, SearchResult,
        WarmupConfigError, WarmupError, WarmupMetric, WarmupOptions, WarmupStage, WarmupSummary,
        WindowSchedule, find_reasonable_step_size, warmup_rwmh,
    };
    pub use crate::config::{AcceptanceTarget, SamplerConfigError, TreeDepth};
    pub use crate::hamiltonian::Divergence;
    pub use crate::hmc::{Hmc, HmcConfigError, HmcError, HmcOptions, HmcTransition, StepSize};
    pub use crate::nuts::{
        Nuts, NutsConfigError, NutsError, NutsOptions, NutsTermination, NutsTransition,
    };
    pub use crate::rwmh::{NumericalRejection, Rwmh, RwmhError, RwmhOptions, RwmhTransition};
}
