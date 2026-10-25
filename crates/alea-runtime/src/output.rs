//! Versioned, borrowed streaming records; no serialization or storage is imposed.
use alea_mcmc::{
    HmcTransition, NutsTransition, RwmhTransition,
    approximate::ApproximateMclmcTransition,
    chees::CheesHmcTransition,
    ghmc::{DrGhmcTransition, GhmcTransition},
    mams::MamsTransition,
    nuts::NutsTermination,
    walnuts::{WalnutsTermination, WalnutsTransition},
};

/// Version of the draw/transition record contract.
pub const SCHEMA_VERSION: u32 = 1;

/// Stable logical chain identity, also the ChaCha stream identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainId(pub u64);

/// Warmup records are never included in retained-draw diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Discarded transitions (adaptive warmup must be performed explicitly).
    Warmup,
    /// Fixed-geometry retained transitions, including repeats after rejection.
    Sampling,
}

/// Common transition metadata. Unsupported quantities are absent, not zero.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TransitionStats {
    /// Selection/Metropolis outcome according to the sampler's typed transition.
    pub accepted: bool,
    /// Mean transition acceptance probability.
    pub acceptance_probability: f64,
    /// Hamiltonian at the start, after fresh momentum sampling; never energy error.
    pub energy: Option<f64>,
    /// Hamiltonian divergence, absent for density-only kernels.
    pub divergent: Option<bool>,
    /// Maximum-depth termination, absent for non-tree kernels.
    pub depth_saturated: Option<bool>,
    /// Attempted complete integrator macrosteps (not gradient evaluations).
    pub integration_steps: Option<usize>,
    /// Completed tree depth, where meaningful.
    pub tree_depth: Option<usize>,
}

/// Externally implementable, side-effect-free adapter for monitoring.
///
/// The original transition remains available to sinks. Repeated projection of one
/// transition must return identical metadata without evaluating a target, consuming
/// randomness, or mutating sampler state. Unsupported quantities are `None`, not
/// invented zeros. Probabilities are finite and lie in `[0, 1]`; present energy is
/// finite and represents the transition's initial Hamiltonian.
pub trait DiagnosticTransition {
    /// Returns common metadata without evaluating the target or consuming RNG.
    fn diagnostic_stats(&self) -> TransitionStats;
}
impl DiagnosticTransition for TransitionStats {
    fn diagnostic_stats(&self) -> TransitionStats {
        *self
    }
}
impl DiagnosticTransition for HmcTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.acceptance_probability,
            energy: self.initial_energy,
            divergent: Some(self.divergence.is_some()),
            integration_steps: Some(self.integration_steps),
            ..TransitionStats::default()
        }
    }
}
impl DiagnosticTransition for CheesHmcTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.acceptance_probability,
            energy: self.initial_energy,
            divergent: Some(self.divergence.is_some()),
            integration_steps: Some(self.plan.integration_steps),
            ..TransitionStats::default()
        }
    }
}
impl DiagnosticTransition for GhmcTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.acceptance_probability,
            energy: self.initial_energy,
            divergent: Some(self.divergence.is_some()),
            integration_steps: Some(self.integration_steps),
            ..TransitionStats::default()
        }
    }
}
impl DiagnosticTransition for DrGhmcTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        let acceptance_probability = if self.second.attempted {
            self.second.acceptance_probability
        } else {
            self.first.acceptance_probability
        };
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability,
            energy: self.initial_energy,
            divergent: Some(self.first.divergence.is_some() || self.second.divergence.is_some()),
            integration_steps: Some(self.integration_steps),
            ..TransitionStats::default()
        }
    }
}
impl DiagnosticTransition for MamsTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.acceptance_probability,
            divergent: Some(self.divergence.is_some()),
            integration_steps: Some(self.integration_steps),
            ..TransitionStats::default()
        }
    }
}
impl DiagnosticTransition for ApproximateMclmcTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            // The unadjusted map always publishes its finite endpoint. This is
            // not a Metropolis probability or an exact-target guarantee.
            accepted: true,
            acceptance_probability: 1.0,
            integration_steps: Some(self.integration_steps),
            ..TransitionStats::default()
        }
    }
}
impl DiagnosticTransition for NutsTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.acceptance_probability,
            energy: self.initial_energy,
            divergent: Some(matches!(self.termination, NutsTermination::Divergence(_))),
            depth_saturated: Some(self.termination == NutsTermination::MaxDepth),
            integration_steps: Some(self.integration_steps),
            tree_depth: Some(self.tree_depth),
        }
    }
}
impl DiagnosticTransition for WalnutsTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            // WALNUTS uses biased-progressive selection with the exact
            // forward/reverse support correction, not endpoint Metropolis.
            acceptance_probability: 1.0,
            energy: self.initial_energy,
            divergent: Some(matches!(
                self.termination,
                WalnutsTermination::Divergence(_) | WalnutsTermination::RefinementLimit
            )),
            depth_saturated: Some(self.termination == WalnutsTermination::MaxDepth),
            integration_steps: Some(self.microsteps),
            tree_depth: Some(self.tree_depth),
        }
    }
}
impl DiagnosticTransition for RwmhTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.acceptance_probability,
            ..TransitionStats::default()
        }
    }
}

/// One completed transition. Index is zero-based within phase and logical chain.
/// Positions include rejected repeats; references cannot escape the sink call.
#[derive(Debug)]
pub struct Draw<'a, T> {
    /// Output contract version.
    pub schema_version: u32,
    /// Independent RNG stream identity.
    pub chain: ChainId,
    /// Warmup or retained sampling.
    pub phase: Phase,
    /// Zero-based index within the phase.
    pub index: usize,
    /// Coherent committed chain position.
    pub position: &'a [f64],
    /// Original, sampler-specific typed statistics.
    pub transition: &'a T,
    /// Standardized monitoring metadata.
    pub stats: TransitionStats,
}
