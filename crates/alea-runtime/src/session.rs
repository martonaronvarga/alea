//! Validated reference inference sessions with explicit resource and replay policy.
//!
//! Targets are constructed inside each chain worker. A compiled model may therefore
//! use chain-local `RefCell` scratch without being `Sync`. No adaptation survives
//! warmup. See `docs/m15-sessions.md` for durable format and deterministic scope.
#![deny(missing_docs)]
mod state;
pub mod storage;
pub use state::{Checkpoint, CheckpointError, FrozenGeometry, Provenance};

use crate::{
    ensemble::{Execution, chain_rng},
    monitor::{Report, Snapshot, Window},
    output::{ChainId, DiagnosticTransition, Phase, TransitionStats},
};
use alea_core::{
    density::LogDensity,
    target::{EvaluationError, LogDensityGradient},
};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric, random::NormalGenerator};
use alea_mcmc::{
    Hmc, HmcOptions, Nuts, NutsOptions, Rwmh, RwmhOptions,
    adapt::{
        DualAveraging, HmcWarmup, MetricKind, NutsWarmup, NutsWarmupError, SearchOptions,
        WarmupError, WarmupMetric, WarmupOptions,
    },
    config::AcceptanceTarget,
    hmc::HmcError,
    nuts::NutsError,
    rwmh::RwmhError,
};
use chacha20::ChaCha12Rng;
use rayon::prelude::*;
use std::{
    error::Error,
    panic::{AssertUnwindSafe, catch_unwind},
    time::{Duration, Instant},
};
use storage::{Sink, SinkError};
use tokio_util::sync::CancellationToken;

// Owned persistence evolves independently of the borrowed hot-path record schema.
const OWNED_SCHEMA_VERSION: u32 = 1;

/// Explicit reference kernel. NUTS uses native NUTS during covariance warmup.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Kernel {
    /// Frozen fixed-trajectory HMC.
    Hmc(HmcOptions),
    /// Frozen multinomial NUTS.
    Nuts(NutsOptions),
    /// Density-only random-walk Metropolis with identity proposal covariance.
    Rwmh(RwmhOptions),
}
impl Kernel {
    fn compatible(self, other: Self) -> bool {
        match (self, other) {
            (Self::Hmc(a), Self::Hmc(b)) => {
                a.integration_steps() == b.integration_steps()
                    && a.max_energy_error() == b.max_energy_error()
            }
            (Self::Nuts(a), Self::Nuts(b)) => {
                a.max_depth() == b.max_depth() && a.max_energy_error() == b.max_energy_error()
            }
            (Self::Rwmh(_), Self::Rwmh(_)) => true,
            _ => false,
        }
    }
}

/// Finite adaptation policy, separate from retained sampling.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Warmup {
    /// Use the supplied fixed geometry and step immediately.
    None,
    /// Reference windowed covariance and dual averaging, consumed before sampling.
    Covariance {
        /// Number of warmup transitions (not including bounded search probes).
        iterations: usize,
        /// Diagonal or dense covariance estimator.
        metric: MetricKind,
        /// Explicit mean acceptance target.
        target: AcceptanceTarget,
        /// Optional bounded search at initialization and metric boundaries.
        search: Option<SearchOptions>,
    },
    /// Identity-proposal RWMH scale adaptation, with no gradient evaluations.
    RwmhScale {
        /// Number of discarded adaptive transitions.
        iterations: usize,
        /// Explicit dual-averaging target.
        target: AcceptanceTarget,
    },
}
impl Warmup {
    fn iterations(self) -> usize {
        match self {
            Self::None => 0,
            Self::Covariance { iterations, .. } | Self::RwmhScale { iterations, .. } => iterations,
        }
    }
}

/// Coordinate observable. Names describe unconstrained sampler coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observable {
    /// Unique user-facing name, for example `unconstrained.beta[0]`.
    pub name: String,
    /// Index in the target's unconstrained point.
    pub coordinate: usize,
}

/// Hard admission limits. Target-owned scratch and sink-owned storage are caller budgets.
#[derive(Debug, Clone, Copy)]
pub struct Resources {
    /// Maximum logical chains.
    pub chains: usize,
    /// Maximum threads in a supplied Rayon pool (sequential uses one).
    pub threads: usize,
    /// Maximum target dimension.
    pub dimension: usize,
    /// Maximum NUTS depth; trajectory work can grow exponentially with this limit.
    pub tree_depth: usize,
    /// Maximum fixed-HMC macrosteps per transition.
    pub integration_steps: usize,
    /// Maximum warmup transitions per chain; searches are separately bounded.
    pub warmup: usize,
    /// Maximum floats in all diagnostic rings.
    pub diagnostic_values: usize,
    /// Maximum bytes in collected record/coordinate storage, excluding allocator overhead.
    pub stored_bytes: usize,
    /// Conservative aggregate sampler/covariance float budget, excluding targets.
    pub working_values: usize,
}
impl Default for Resources {
    fn default() -> Self {
        Self {
            chains: 16,
            threads: 16,
            dimension: 1024,
            tree_depth: 12,
            integration_steps: 10_000,
            warmup: 100_000,
            diagnostic_values: 1_048_576,
            stored_bytes: 268_435_456,
            working_values: 16_777_216,
        }
    }
}

/// Explicit final diagnostic warning thresholds. No diagnostic stops adaptation or sampling.
#[derive(Debug, Clone, Copy)]
pub struct Diagnostics {
    /// Contiguous retained draws per chain, at least four.
    pub window: usize,
    /// Rank/folded R-hat warning threshold, finite and at least one.
    pub maximum_rhat: f64,
    /// Bulk and tail ESS warning threshold, finite and positive.
    pub minimum_ess: f64,
}
impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            window: 512,
            maximum_rhat: 1.01,
            minimum_ess: 100.0,
        }
    }
}

/// A completed owned record. Independent of borrowed, sampler-specific hot-path draws.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OwnedDraw {
    /// Owned draw schema version, currently one.
    pub schema_version: u32,
    /// Logical stream.
    pub chain: ChainId,
    /// Always sampling for session sinks; warmup is discarded.
    pub phase: Phase,
    /// Absolute zero-based retained index, continuing across restart.
    pub index: usize,
    /// Coherent point, including rejected repeats.
    pub position: Vec<f64>,
    /// Common transition metadata, never fabricated unsupported values.
    pub stats: TransitionStats,
    /// Numerical proposal failure (also reported separately for density-only RWMH).
    pub numerical_rejection: bool,
}

/// Outcome precedence is failure, numerical exhaustion, cancellation, warning, success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    /// Budget completed and final diagnostic gates passed.
    Success,
    /// Budget completed, with unavailable/failed diagnostics or adverse transitions.
    DiagnosticWarning,
    /// User token or wall-clock deadline was observed at a safe boundary.
    Cancelled,
    /// Explicit numerical limit or configured consecutive-divergence budget exhausted.
    NumericalExhaustion,
    /// Target, worker, contract or output backend failed.
    BackendFailure,
}

/// Original typed failures remain attached to the affected chain.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Failure<E: Error + 'static> {
    /// Factory failed before constructing the target.
    #[error("target factory failed: {0}")]
    Target(#[source] E),
    /// HMC construction or transition failure.
    #[error(transparent)]
    Hmc(#[from] HmcError<E>),
    /// NUTS construction or transition failure.
    #[error(transparent)]
    Nuts(#[from] NutsError<E>),
    /// RWMH construction or transition failure.
    #[error(transparent)]
    Rwmh(#[from] RwmhError<E>),
    /// Warmup did not freeze successfully.
    #[error(transparent)]
    Warmup(#[from] WarmupError<E>),
    /// Native NUTS warmup did not freeze successfully.
    #[error(transparent)]
    NutsWarmup(#[from] NutsWarmupError<E>),
    /// Sink could not accept a completed draw.
    #[error(transparent)]
    Sink(#[from] SinkError),
    /// A factory/target worker unwound. Abort/OOM cannot be caught.
    #[error("chain worker panicked")]
    WorkerPanic,
    /// Observer unwound after the transition committed.
    #[error("draw observer panicked")]
    ObserverPanic,
    /// Target dimension differs or the selected target capability is incompatible.
    #[error("session target or kernel contract mismatch")]
    Contract,
    /// Re-evaluated frozen point differs from checkpoint cache bits.
    #[error("checkpoint target evaluation differs; restart refused")]
    RestartMismatch,
    /// Explicit retained numerical rejection streak limit.
    #[error("consecutive numerical rejection budget exhausted")]
    NumericalLimit,
}
impl<E: Error + 'static> Failure<E> {
    fn numerical(&self) -> bool {
        fn evaluation<E: Error + 'static>(e: &EvaluationError<E>) -> bool {
            matches!(
                e,
                EvaluationError::NonFinitePosition { .. }
                    | EvaluationError::NonFiniteLogDensity
                    | EvaluationError::NonFiniteGradient { .. }
            )
        }
        fn hmc<E: Error + 'static>(e: &HmcError<E>) -> bool {
            matches!(e,HmcError::Evaluation{source,..} if evaluation(source))
        }
        match self {
            Self::NutsWarmup(NutsWarmupError::Nuts(e)) => match e {
                NutsError::Initialization(e) => hmc(e),
                NutsError::Phase { source, .. } => match source {
                    alea_mcmc::hamiltonian::PhaseError::Evaluation(e) => evaluation(e),
                    alea_mcmc::hamiltonian::PhaseError::Divergence(_) => true,
                    _ => false,
                },
                _ => false,
            },
            Self::NutsWarmup(_) => true,
            Self::NumericalLimit => true,
            Self::Hmc(e) => hmc(e),
            Self::Rwmh(RwmhError::Evaluation(e)) => evaluation(e),
            Self::Nuts(NutsError::Initialization(e)) => hmc(e),
            Self::Nuts(NutsError::Phase { source, .. }) => match source {
                alea_mcmc::hamiltonian::PhaseError::Evaluation(e) => evaluation(e),
                alea_mcmc::hamiltonian::PhaseError::Divergence(_) => true,
                _ => false,
            },
            Self::Warmup(WarmupError::Hmc(e)) => hmc(e),
            Self::Warmup(
                WarmupError::SearchExhausted
                | WarmupError::Covariance(_)
                | WarmupError::Adaptation(_)
                | WarmupError::CounterOverflow,
            ) => true,
            _ => false,
        }
    }
}

/// Input validation fails before factories, sinks or RNG streams are touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid session configuration or resource budget")]
pub struct ConfigurationError;

/// Chain-local result; valid draws survive all recoverable failure paths.
#[derive(Debug)]
#[non_exhaustive]
pub struct ChainOutcome<E: Error + 'static> {
    /// Logical identity, ordered in the outcome independently of scheduling.
    pub chain: ChainId,
    /// Completed warmup transitions in this invocation.
    pub warmup_completed: usize,
    /// Absolute retained completion count, including checkpoint history.
    pub completed: usize,
    /// Draws from this invocation when collection was requested.
    pub draws: Vec<OwnedDraw>,
    /// Last completed draw rejected by a sink, even if also collected.
    pub undelivered: Option<OwnedDraw>,
    /// Frozen restart point; absent when warmup failed or was cancelled.
    pub checkpoint: Option<Checkpoint>,
    /// Original failure; never masquerades as ordinary rejection.
    pub failure: Option<Failure<E>>,
    /// Safe-boundary cancellation observed by this chain.
    pub cancelled: bool,
    /// Numerical rejected transitions in this invocation, including RWMH.
    pub numerical_rejections: usize,
}

/// Numerical and resource policy recorded with the outcome, including explicit
/// warmup choices. Durable draw headers carry the common replay identity; persist
/// this plan alongside them when a complete experiment specification is needed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RunPlan {
    /// Requested kernel before any adaptation.
    pub kernel: Kernel,
    /// Requested finite adaptation policy.
    pub warmup: Warmup,
    /// Absolute retained end index.
    pub draws: usize,
    /// Starting unconstrained points for this invocation.
    pub initial: Vec<Vec<f64>>,
    /// Absolute retained starting indices (zero for fresh runs).
    pub starting_indices: Vec<usize>,
    /// Unconstrained coordinate labels.
    pub observables: Vec<Observable>,
    /// Contiguous diagnostic window and warning gates.
    pub diagnostics: Diagnostics,
    /// Admission limits.
    pub resources: Resources,
    /// Whether the outcome collects new draws.
    pub collected: bool,
    /// Cooperative elapsed-time limit.
    pub wall_time: Option<Duration>,
    /// Consecutive numerical rejection limit, preserved across restart.
    pub numerical_limit: usize,
}

/// Complete structured result. Streaming samples live with the sink; failed delivery
/// is returned explicitly. Diagnostic ranges cover this invocation only on restart.
#[derive(Debug)]
#[non_exhaustive]
pub struct RunOutcome<E: Error + 'static> {
    /// Explicit terminal classification.
    pub status: Status,
    /// Target/seed/implementation identity needed to interpret durable records.
    pub provenance: Provenance,
    /// Explicit numerical, output and resource policy for this invocation.
    pub plan: RunPlan,
    /// Logical input-order chain results, including partial work.
    pub chains: Vec<ChainOutcome<E>>,
    /// Final bounded-window diagnostics; undefined estimates remain errors.
    pub diagnostics: Report,
    /// Sink finalization error, including an observer panic at finalization.
    pub sink_failure: Option<SinkError>,
}

/// Session specification under construction. Defaults perform no adaptation;
/// covariance warmup must be selected explicitly.
pub struct SessionBuilder {
    target_id: String,
    dimension: usize,
    initial: Vec<Vec<f64>>,
    kernel: Kernel,
    warmup: Warmup,
    draws: usize,
    seed: [u8; 32],
    observables: Vec<Observable>,
    diagnostics: Diagnostics,
    resources: Resources,
    collect: bool,
    cancel: CancellationToken,
    wall_time: Option<Duration>,
    numerical_limit: usize,
    restart: Vec<Checkpoint>,
}
impl SessionBuilder {
    /// `target_id` must identify the mathematical model/data and executable build
    /// for replay, not merely a display name. Initial points are unconstrained.
    pub fn new(target_id: impl Into<String>, dimension: usize, initial: Vec<Vec<f64>>) -> Self {
        Self {
            target_id: target_id.into(),
            dimension,
            initial,
            kernel: Kernel::Nuts(NutsOptions::new(0.1, 8).expect("valid default")),
            warmup: Warmup::None,
            draws: 1000,
            seed: [0; 32],
            observables: Vec::new(),
            diagnostics: Diagnostics::default(),
            resources: Resources::default(),
            collect: true,
            cancel: CancellationToken::new(),
            wall_time: None,
            numerical_limit: 100,
            restart: Vec::new(),
        }
    }
    /// Select the explicit frozen reference kernel.
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
    /// Select finite adaptation consumed before retention.
    pub fn warmup(mut self, warmup: Warmup) -> Self {
        self.warmup = warmup;
        self
    }
    /// Absolute retained end index per chain; restart continues toward this budget.
    pub fn draws(mut self, draws: usize) -> Self {
        self.draws = draws;
        self
    }
    /// Seed plus logical chain ID determines the independent uniform stream.
    pub fn seed(mut self, seed: [u8; 32]) -> Self {
        self.seed = seed;
        self
    }
    /// Explicit coordinate observables. Empty selects all unconstrained coordinates.
    pub fn observables(mut self, observables: Vec<Observable>) -> Self {
        self.observables = observables;
        self
    }
    /// Bounded final diagnostics and warning gates.
    pub fn diagnostics(mut self, diagnostics: Diagnostics) -> Self {
        self.diagnostics = diagnostics;
        self
    }
    /// Admission budgets, including pool size and output storage.
    pub fn resources(mut self, resources: Resources) -> Self {
        self.resources = resources;
        self
    }
    /// Store all new draws in the outcome, or stream with bounded sampler memory.
    pub fn collect(mut self, collect: bool) -> Self {
        self.collect = collect;
        self
    }
    /// Poll a caller-owned cancellation token at safe numerical boundaries.
    pub fn cancellation(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }
    /// Cooperative duration from `run` entry, including factory/warmup/output time.
    pub fn wall_time(mut self, duration: Duration) -> Self {
        self.wall_time = Some(duration);
        self
    }
    /// Stop after this many consecutive numerical rejections; must be nonzero.
    pub fn numerical_limit(mut self, limit: usize) -> Self {
        self.numerical_limit = limit;
        self
    }
    /// Continue frozen checkpoints. No warmup is rerun; initial coordinates are
    /// replaced by the checkpoints. Only retained-phase checkpoints are supported.
    pub fn restart(mut self, checkpoints: Vec<Checkpoint>) -> Self {
        self.restart = checkpoints;
        self.warmup = Warmup::None;
        self
    }

    /// Validate dimensions, adaptation/kernel compatibility, resource products,
    /// names and all checkpoint identities before any numerical work.
    /// # Errors
    /// Returns [`ConfigurationError`] without creating targets or output files.
    pub fn build(mut self) -> Result<Session, ConfigurationError> {
        let r = self.resources;
        let n = self.initial.len();
        let d = self.dimension;
        let depth = match self.kernel {
            Kernel::Nuts(o) => o.max_depth().value(),
            _ => 0,
        };
        let steps = match self.kernel {
            Kernel::Hmc(o) => o.integration_steps().get(),
            _ => 1,
        };
        if d == 0
            || d > r.dimension
            || n == 0
            || n > r.chains
            || r.threads == 0
            || depth > r.tree_depth
            || steps > r.integration_steps
            || self.target_id.is_empty()
            || self.target_id.len() > 4096
            || self.draws == 0
            || self.numerical_limit == 0
            || self.warmup.iterations() > r.warmup
            || self
                .initial
                .iter()
                .any(|q| q.len() != d || q.iter().any(|x| !x.is_finite()))
        {
            return Err(ConfigurationError);
        }
        if !matches!(
            (self.kernel, self.warmup),
            (_, Warmup::None)
                | (Kernel::Hmc(_) | Kernel::Nuts(_), Warmup::Covariance { .. })
                | (Kernel::Rwmh(_), Warmup::RwmhScale { .. })
        ) {
            return Err(ConfigurationError);
        }
        if self.observables.is_empty() {
            self.observables = (0..d)
                .map(|i| Observable {
                    name: format!("unconstrained.q[{i}]"),
                    coordinate: i,
                })
                .collect();
        }
        if self.observables.iter().enumerate().any(|(i, o)| {
            o.name.is_empty()
                || o.coordinate >= d
                || self.observables[..i]
                    .iter()
                    .any(|p| p.name == o.name || p.coordinate == o.coordinate)
        }) {
            return Err(ConfigurationError);
        }
        let diagnostic = self.diagnostics;
        if diagnostic.window < 4
            || !diagnostic.maximum_rhat.is_finite()
            || diagnostic.maximum_rhat < 1.0
            || !diagnostic.minimum_ess.is_finite()
            || diagnostic.minimum_ess <= 0.0
        {
            return Err(ConfigurationError);
        }
        let product = |a: usize, b: usize, c: usize, limit: usize| {
            a.checked_mul(b)
                .and_then(|v| v.checked_mul(c))
                .is_some_and(|v| v <= limit && v <= isize::MAX as usize)
        };
        let per_dimension = 13_usize
            .checked_mul(depth + 2)
            .and_then(|v| v.checked_add(64))
            .ok_or(ConfigurationError)?;
        let dense = self.restart.iter().any(|c| {
            matches!(
                c.geometry,
                FrozenGeometry::Dense { .. } | FrozenGeometry::InverseDense { .. }
            )
        }) || matches!(
            self.warmup,
            Warmup::Covariance {
                metric: MetricKind::Dense,
                ..
            }
        );
        let working = d
            .checked_mul(per_dimension)
            .and_then(|v| {
                if dense {
                    d.checked_mul(d)
                        .and_then(|m| m.checked_mul(8))
                        .and_then(|m| v.checked_add(m))
                } else {
                    Some(v)
                }
            })
            .ok_or(ConfigurationError)?;
        let record_bytes = d
            .checked_mul(size_of::<f64>())
            .and_then(|v| v.checked_add(size_of::<OwnedDraw>()))
            .ok_or(ConfigurationError)?;
        let diagnostic_bytes = self
            .observables
            .len()
            .checked_mul(64)
            .and_then(|v| v.checked_add(size_of::<Option<f64>>()))
            .ok_or(ConfigurationError)?;
        if !product(
            n,
            diagnostic.window,
            diagnostic_bytes,
            isize::MAX as usize / 8,
        ) || !product(
            n,
            diagnostic.window,
            self.observables.len(),
            r.diagnostic_values,
        ) || !product(n, working, 1, r.working_values)
            || !product(n, working, 8, isize::MAX as usize)
            || (self.collect && !product(n, self.draws, record_bytes, r.stored_bytes))
        {
            return Err(ConfigurationError);
        }
        let provenance = Provenance::new(self.target_id.clone(), d, self.seed);
        if !self.restart.is_empty() {
            if !matches!(self.warmup, Warmup::None) || self.restart.len() != n {
                return Err(ConfigurationError);
            }
            for (i, cp) in self.restart.iter().enumerate() {
                if cp.provenance != provenance
                    || cp.chain != ChainId(i as u64)
                    || cp.completed > self.draws
                    || !self.kernel.compatible(cp.kernel)
                    || !cp.valid()
                {
                    return Err(ConfigurationError);
                }
                // Dense geometry is admitted even when no new warmup is requested.
                if !product(n, cp.geometry.values(), 1, r.working_values) {
                    return Err(ConfigurationError);
                }
            }
        }
        Ok(Session {
            spec: self,
            provenance,
        })
    }
}

/// A validated, reusable run specification. Each invocation has independent state;
/// concurrent invocations require separate non-overwriting sinks.
pub struct Session {
    spec: SessionBuilder,
    provenance: Provenance,
}
impl Session {
    /// Validated provenance for sink headers and checkpoint compatibility.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    /// Recorded numerical and resource policy.
    pub fn plan(&self) -> RunPlan {
        RunPlan {
            kernel: self.spec.kernel,
            warmup: self.spec.warmup,
            draws: self.spec.draws,
            initial: (0..self.spec.initial.len())
                .map(|i| {
                    self.spec
                        .restart
                        .get(i)
                        .map_or_else(|| self.spec.initial[i].clone(), |cp| cp.position.clone())
                })
                .collect(),
            starting_indices: (0..self.spec.initial.len())
                .map(|i| self.spec.restart.get(i).map_or(0, |cp| cp.completed))
                .collect(),
            observables: self.spec.observables.clone(),
            diagnostics: self.spec.diagnostics,
            resources: self.spec.resources,
            collected: self.spec.collect,
            wall_time: self.spec.wall_time,
            numerical_limit: self.spec.numerical_limit,
        }
    }
    /// Selected names in the same order as final diagnostic coordinates.
    pub fn observables(&self) -> &[Observable] {
        &self.spec.observables
    }

    /// Run HMC/NUTS, constructing an independent target/model inside each worker.
    /// A target need not be `Send` or `Sync`; only the factory and returned errors
    /// cross worker boundaries. The caller owns the Rayon pool and sink.
    /// # Errors
    /// Pool admission/capability errors return before work; runtime failures are
    /// retained with valid draws in [`RunOutcome`].
    pub fn run<T, F, E, S>(
        &self,
        factory: F,
        execution: Execution<'_>,
        sink: &S,
    ) -> Result<RunOutcome<E>, ConfigurationError>
    where
        T: LogDensityGradient<Error = E>,
        F: Fn(ChainId) -> Result<T, E> + Sync,
        E: Error + Send + 'static,
        S: Sink,
    {
        if matches!(self.spec.kernel, Kernel::Rwmh(_)) {
            return Err(ConfigurationError);
        }
        self.execute(execution, sink, |id, out, window, control| {
            let target = factory(id).map_err(Failure::Target)?;
            if target.dimension() != self.spec.dimension {
                return Err(Failure::Contract);
            }
            let mut rng = self.rng(id);
            let cp = self.spec.restart.get(id.0 as usize);
            let chain = if let Some(cp) = cp {
                let geometry = cp
                    .geometry
                    .restore()
                    .map_err(|_| Failure::RestartMismatch)?;
                let chain = GradientChain::new(&target, buffer(&cp.position), geometry, cp.kernel)?;
                if !cp.matches(chain.position(), chain.logp(), chain.gradient()) {
                    return Err(Failure::RestartMismatch);
                }
                chain
            } else {
                let options = match self.spec.warmup {
                    Warmup::Covariance {
                        iterations,
                        metric,
                        target,
                        search,
                    } => WarmupOptions::new(iterations)
                        .with_metric(metric)
                        .with_target(target)
                        .with_search(search),
                    _ => WarmupOptions::new(0),
                };
                match self.spec.kernel {
                    Kernel::Hmc(o) => {
                        let result = HmcWarmup::new(
                            &target,
                            buffer(&self.spec.initial[id.0 as usize]),
                            o,
                            options,
                        )?
                        .run_controlled(&mut rng, |completed| {
                            out.warmup_completed = completed;
                            !control.stopped()
                        })?;
                        let Some((chain, report)) = result else {
                            out.cancelled = true;
                            return Ok(());
                        };
                        out.warmup_completed = report.iterations;
                        GradientChain::Hmc(Box::new(chain))
                    }
                    Kernel::Nuts(o) => {
                        let result = NutsWarmup::new(
                            &target,
                            buffer(&self.spec.initial[id.0 as usize]),
                            o,
                            options,
                        )?
                        .run_controlled(&mut rng, |completed| {
                            out.warmup_completed = completed;
                            !control.stopped()
                        })?;
                        let Some((chain, report)) = result else {
                            out.cancelled = true;
                            return Ok(());
                        };
                        out.warmup_completed = report.iterations;
                        GradientChain::Nuts(Box::new(chain))
                    }
                    _ => unreachable!(),
                }
            };
            self.sample(chain, &mut rng, out, window, control, sink)
        })
    }

    /// Run density-only RWMH without requesting or fabricating a gradient.
    /// # Errors
    /// Same admission and runtime outcome contract as [`Self::run`].
    pub fn run_density<T, F, E, S>(
        &self,
        factory: F,
        execution: Execution<'_>,
        sink: &S,
    ) -> Result<RunOutcome<E>, ConfigurationError>
    where
        T: LogDensity<Error = E>,
        F: Fn(ChainId) -> Result<T, E> + Sync,
        E: Error + Send + 'static,
        S: Sink,
    {
        let Kernel::Rwmh(options) = self.spec.kernel else {
            return Err(ConfigurationError);
        };
        self.execute(execution, sink, |id, out, window, control| {
            let target = factory(id).map_err(Failure::Target)?;
            if target.dimension() != self.spec.dimension {
                return Err(Failure::Contract);
            }
            let mut rng = self.rng(id);
            let cp = self.spec.restart.get(id.0 as usize);
            let (q, options, normal) = if let Some(cp) = cp {
                let Kernel::Rwmh(options) = cp.kernel else {
                    return Err(Failure::RestartMismatch);
                };
                (
                    &cp.position,
                    options,
                    NormalGenerator::from_spare(cp.normal_spare).ok_or(Failure::RestartMismatch)?,
                )
            } else {
                (
                    &self.spec.initial[id.0 as usize],
                    options,
                    NormalGenerator::default(),
                )
            };
            let mut chain = Rwmh::new(
                &target,
                buffer(q),
                IdentityMetric::new(self.spec.dimension),
                options,
            )?
            .with_normal_generator(normal);
            if let Some(cp) = cp {
                if !cp.matches(chain.point().position(), chain.point().log_density(), &[]) {
                    return Err(Failure::RestartMismatch);
                }
            } else if let Warmup::RwmhScale { iterations, target } = self.spec.warmup {
                let mut adapter = DualAveraging::new(options.step_size(), target);
                for _ in 0..iterations {
                    if control.stopped() {
                        out.cancelled = true;
                        return Ok(());
                    }
                    let transition = chain.step(&mut rng)?;
                    out.warmup_completed += 1;
                    chain.set_step_size(
                        adapter
                            .update(transition.acceptance_probability)
                            .map_err(|e| Failure::Warmup(WarmupError::Adaptation(e)))?,
                    );
                }
                if iterations > 0 {
                    chain.set_step_size(adapter.finish());
                }
            }
            self.sample(DensityChain(chain), &mut rng, out, window, control, sink)
        })
    }

    fn rng(&self, id: ChainId) -> ChaCha12Rng {
        let mut rng = chain_rng(self.spec.seed, id);
        if let Some(cp) = self.spec.restart.get(id.0 as usize) {
            rng.set_word_pos(cp.word_position);
        }
        rng
    }

    fn execute<E, S, F>(
        &self,
        execution: Execution<'_>,
        sink: &S,
        work: F,
    ) -> Result<RunOutcome<E>, ConfigurationError>
    where
        E: Error + Send + 'static,
        S: Sink,
        F: Fn(ChainId, &mut ChainOutcome<E>, &mut Window, &Control) -> Result<(), Failure<E>>
            + Sync,
    {
        if matches!(execution,Execution::Parallel(pool) if pool.current_num_threads()>self.spec.resources.threads)
        {
            return Err(ConfigurationError);
        }
        let control = Control {
            cancel: &self.spec.cancel,
            deadline: self
                .spec
                .wall_time
                .map(|d| Instant::now().checked_add(d).ok_or(ConfigurationError))
                .transpose()?,
        };
        let coordinates: Vec<_> = self.spec.observables.iter().map(|o| o.coordinate).collect();
        let start_failure = match catch_unwind(AssertUnwindSafe(|| sink.start(&self.provenance))) {
            Ok(result) => result.err(),
            Err(_) => Some(SinkError::ObserverPanic),
        };
        let chain_work = |i: usize| {
            let id = ChainId(i as u64);
            let mut out = ChainOutcome {
                chain: id,
                warmup_completed: 0,
                completed: self.spec.restart.get(i).map_or(0, |c| c.completed),
                draws: if self.spec.collect {
                    Vec::with_capacity(
                        self.spec.draws - self.spec.restart.get(i).map_or(0, |c| c.completed),
                    )
                } else {
                    Vec::new()
                },
                undelivered: None,
                checkpoint: self.spec.restart.get(i).cloned(),
                failure: None,
                cancelled: false,
                numerical_rejections: 0,
            };
            let mut window = Window::new(id, self.spec.diagnostics.window, coordinates.len());
            if start_failure.is_some() {
                // Admission to the sink failed; factories are never invoked.
            } else if control.stopped() {
                out.cancelled = true;
            } else {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    work(id, &mut out, &mut window, &control)
                }));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => out.failure = Some(e),
                    Err(_) => out.failure = Some(Failure::WorkerPanic),
                }
            }
            (out, window)
        };
        let results: Vec<_> = match execution {
            Execution::Sequential => (0..self.spec.initial.len()).map(chain_work).collect(),
            Execution::Parallel(pool) => pool.install(|| {
                (0..self.spec.initial.len())
                    .into_par_iter()
                    .map(chain_work)
                    .collect()
            }),
        };
        let (chains, windows): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let progress = chains
            .iter()
            .map(|c| crate::ensemble::ChainProgress {
                chain: c.chain,
                warmup: c.warmup_completed,
                draws: c.completed,
            })
            .collect::<Vec<_>>();
        let failed = chains.iter().any(|c| c.failure.is_some());
        let mut snapshot = Snapshot::new(&windows, &progress, &coordinates, true);
        snapshot.failed = failed;
        let mut diagnostics = Report::calculate(&snapshot);
        for (i, chain) in diagnostics.chains.iter_mut().enumerate() {
            let offset = self.spec.restart.get(i).map_or(0, |cp| cp.completed);
            chain.range.start += offset;
            chain.range.end += offset;
        }
        if diagnostics
            .chains
            .iter()
            .any(|c| c.range != diagnostics.chains[0].range)
        {
            diagnostics
                .diagnostics
                .fill(Err(crate::diagnostics::modern::Error::Shape));
        }
        let finish_failure = match catch_unwind(AssertUnwindSafe(|| sink.finish())) {
            Ok(result) => result.err(),
            Err(_) => Some(SinkError::ObserverPanic),
        };
        let sink_failure = start_failure.or(finish_failure);
        diagnostics.failed |= sink_failure.is_some();
        let status = if sink_failure.is_some()
            || chains
                .iter()
                .any(|c| c.failure.as_ref().is_some_and(|e| !e.numerical()))
        {
            Status::BackendFailure
        } else if failed {
            Status::NumericalExhaustion
        } else if chains.iter().any(|c| c.cancelled) {
            Status::Cancelled
        } else if chains.iter().any(|c| c.numerical_rejections > 0)
            || diagnostics
                .chains
                .iter()
                .any(|c| c.counts.divergences > 0 || c.counts.depth_saturations > 0)
            || diagnostics.diagnostics.iter().any(|d| match d {
                Ok(d) => {
                    d.rhat > self.spec.diagnostics.maximum_rhat
                        || d.bulk_ess < self.spec.diagnostics.minimum_ess
                        || d.tail_ess < self.spec.diagnostics.minimum_ess
                }
                Err(_) => true,
            })
        {
            Status::DiagnosticWarning
        } else {
            Status::Success
        };
        Ok(RunOutcome {
            status,
            provenance: self.provenance.clone(),
            plan: self.plan(),
            chains,
            diagnostics,
            sink_failure,
        })
    }

    fn sample<C, E, S>(
        &self,
        mut chain: C,
        rng: &mut ChaCha12Rng,
        out: &mut ChainOutcome<E>,
        window: &mut Window,
        control: &Control,
        sink: &S,
    ) -> Result<(), Failure<E>>
    where
        C: Sampler<E>,
        E: Error + 'static,
        S: Sink,
    {
        let coordinates: Vec<_> = self.spec.observables.iter().map(|o| o.coordinate).collect();
        let mut streak = self
            .spec
            .restart
            .get(out.chain.0 as usize)
            .map_or(0, |cp| cp.numerical_streak);
        let result = (|| {
            if streak >= self.spec.numerical_limit {
                return Err(Failure::NumericalLimit);
            }
            while out.completed < self.spec.draws {
                if control.stopped() {
                    out.cancelled = true;
                    break;
                }
                let (stats, numerical_rejection) =
                    match catch_unwind(AssertUnwindSafe(|| chain.step(rng))) {
                        Ok(r) => r?,
                        Err(_) => return Err(Failure::WorkerPanic),
                    };
                let draw = OwnedDraw {
                    schema_version: OWNED_SCHEMA_VERSION,
                    chain: out.chain,
                    phase: Phase::Sampling,
                    index: out.completed,
                    position: chain.position().to_vec(),
                    stats,
                    numerical_rejection,
                };
                out.completed += 1;
                out.numerical_rejections += usize::from(numerical_rejection);
                window.push(&draw.position, &coordinates, stats);
                if self.spec.collect {
                    out.draws.push(draw.clone());
                }
                streak = if numerical_rejection { streak + 1 } else { 0 };
                let delivered = catch_unwind(AssertUnwindSafe(|| sink.draw(&draw)));
                match delivered {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        out.undelivered = Some(draw);
                        return Err(Failure::Sink(e));
                    }
                    Err(_) => {
                        out.undelivered = Some(draw);
                        return Err(Failure::ObserverPanic);
                    }
                }
                if streak >= self.spec.numerical_limit {
                    return Err(Failure::NumericalLimit);
                }
            }
            Ok(())
        })();
        let mut checkpoint =
            chain.checkpoint(self.provenance.clone(), out.chain, out.completed, rng);
        checkpoint.numerical_streak = streak;
        out.checkpoint = Some(checkpoint);
        result
    }
}

struct Control<'a> {
    cancel: &'a CancellationToken,
    deadline: Option<Instant>,
}
impl Control<'_> {
    fn stopped(&self) -> bool {
        self.cancel.is_cancelled() || self.deadline.is_some_and(|t| Instant::now() >= t)
    }
}
fn buffer(q: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(q.len(), |i| q[i])
}

trait Sampler<E: Error + 'static> {
    fn step(&mut self, rng: &mut ChaCha12Rng) -> Result<(TransitionStats, bool), Failure<E>>;
    fn position(&self) -> &[f64];
    fn logp(&self) -> f64;
    fn gradient(&self) -> &[f64];
    fn checkpoint(
        &self,
        provenance: Provenance,
        id: ChainId,
        completed: usize,
        rng: &ChaCha12Rng,
    ) -> Checkpoint;
}
enum GradientChain<'a, T: LogDensityGradient> {
    Hmc(Box<Hmc<'a, T, WarmupMetric>>),
    Nuts(Box<Nuts<'a, T, WarmupMetric>>),
}
impl<'a, T: LogDensityGradient> GradientChain<'a, T> {
    fn new(
        target: &'a T,
        q: OwnedBuffer,
        metric: WarmupMetric,
        kernel: Kernel,
    ) -> Result<Self, Failure<T::Error>> {
        Ok(match kernel {
            Kernel::Hmc(o) => Self::Hmc(Box::new(Hmc::new(target, q, metric, o)?)),
            Kernel::Nuts(o) => {
                let options = HmcOptions::new(o.step_size().value(), 1)
                    .expect("validated step")
                    .with_max_energy_error(o.max_energy_error())
                    .expect("validated energy");
                Self::Nuts(Box::new(Nuts::from_hmc(
                    Hmc::new(target, q, metric, options)?,
                    o.max_depth(),
                )?))
            }
            _ => return Err(Failure::Contract),
        })
    }
}
impl<T: LogDensityGradient> Sampler<T::Error> for GradientChain<'_, T> {
    fn step(
        &mut self,
        rng: &mut ChaCha12Rng,
    ) -> Result<(TransitionStats, bool), Failure<T::Error>> {
        let stats = match self {
            Self::Hmc(c) => c.step(rng)?.diagnostic_stats(),
            Self::Nuts(c) => c.step(rng)?.diagnostic_stats(),
        };
        Ok((stats, stats.divergent == Some(true)))
    }
    fn position(&self) -> &[f64] {
        match self {
            Self::Hmc(c) => c.point().position(),
            Self::Nuts(c) => c.point().position(),
        }
    }
    fn logp(&self) -> f64 {
        match self {
            Self::Hmc(c) => c.point().log_density(),
            Self::Nuts(c) => c.point().log_density(),
        }
    }
    fn gradient(&self) -> &[f64] {
        match self {
            Self::Hmc(c) => c.point().gradient(),
            Self::Nuts(c) => c.point().gradient(),
        }
    }
    fn checkpoint(
        &self,
        provenance: Provenance,
        id: ChainId,
        completed: usize,
        rng: &ChaCha12Rng,
    ) -> Checkpoint {
        let (kernel, metric) = match self {
            Self::Hmc(c) => (Kernel::Hmc(c.options()), c.metric()),
            Self::Nuts(c) => (Kernel::Nuts(c.options()), c.metric()),
        };
        Checkpoint {
            provenance,
            chain: id,
            completed,
            word_position: rng.get_word_pos(),
            kernel,
            geometry: FrozenGeometry::capture(metric),
            position: self.position().to_vec(),
            logp: self.logp(),
            gradient: self.gradient().to_vec(),
            normal_spare: None,
            numerical_streak: 0,
        }
    }
}
struct DensityChain<'a, T: LogDensity>(Rwmh<'a, T, IdentityMetric>);
impl<T: LogDensity> Sampler<T::Error> for DensityChain<'_, T> {
    fn step(
        &mut self,
        rng: &mut ChaCha12Rng,
    ) -> Result<(TransitionStats, bool), Failure<T::Error>> {
        let transition = self.0.step(rng)?;
        Ok((
            transition.diagnostic_stats(),
            transition.numerical_rejection.is_some(),
        ))
    }
    fn position(&self) -> &[f64] {
        self.0.point().position()
    }
    fn logp(&self) -> f64 {
        self.0.point().log_density()
    }
    fn gradient(&self) -> &[f64] {
        &[]
    }
    fn checkpoint(
        &self,
        provenance: Provenance,
        id: ChainId,
        completed: usize,
        rng: &ChaCha12Rng,
    ) -> Checkpoint {
        Checkpoint {
            provenance,
            chain: id,
            completed,
            word_position: rng.get_word_pos(),
            kernel: Kernel::Rwmh(self.0.options()),
            geometry: FrozenGeometry::Identity(self.position().len()),
            position: self.position().to_vec(),
            logp: self.logp(),
            gradient: Vec::new(),
            normal_spare: self.0.normal_generator().spare(),
            numerical_streak: 0,
        }
    }
}
