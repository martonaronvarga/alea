//! Frozen checkpoint values; adaptive state is deliberately not representable.
use super::{SamplerConfig, buffer};
use crate::output::ChainId;
use alea_math::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
    InverseDenseMetric, InverseDiagonalMetric, MetricError,
};
use alea_mcmc::adapt::WarmupMetric;
use std::path::Path;

/// Identity attached to a run, durable draw stream and every checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Provenance {
    /// Durable schema version.
    pub schema_version: u32,
    /// Caller-supplied model/data/executable identity. Replay requires it unchanged.
    pub target_id: String,
    /// Package implementation identifier; no cross-build replay guarantee.
    pub implementation: String,
    /// Unconstrained target dimension.
    pub dimension: usize,
    /// Full uniform seed; each chain has its logical stream.
    pub seed: [u8; 32],
}
impl Provenance {
    pub(super) fn new(target_id: String, dimension: usize, seed: [u8; 32]) -> Self {
        Self {
            schema_version: 1,
            target_id,
            implementation: concat!(
                "alea-runtime/",
                env!("CARGO_PKG_VERSION"),
                ";session/2;chacha12/0.10;rand/0.10;numerics/3"
            )
            .into(),
            dimension,
            seed,
        }
    }
    pub(super) fn valid(&self) -> bool {
        self.valid_format() && self == &Self::new(self.target_id.clone(), self.dimension, self.seed)
    }
    pub(super) fn valid_format(&self) -> bool {
        self.schema_version == 1
            && self.dimension > 0
            && !self.target_id.is_empty()
            && self.target_id.len() <= 4096
            && !self.implementation.is_empty()
            && self.implementation.len() <= 4096
    }
}

/// Lossless frozen geometry with explicit mass or inverse-mass representation.
/// RWMH supports identity only here.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FrozenGeometry {
    /// Identity with explicit dimension.
    Identity(usize),
    /// Diagonal masses, not inverse masses or standard deviations.
    Diagonal(Vec<f64>),
    /// Lower Cholesky factor in full column-major storage; upper triangle zero.
    Dense {
        /// Matrix dimension.
        dimension: usize,
        /// Frozen factor entries; never refactorized on restart.
        lower: Vec<f64>,
    },
    /// Positive diagonal inverse masses G, distinct from the legacy mass variant.
    InverseDiagonal(Vec<f64>),
    /// Exact lower Cholesky factor of inverse mass G, in full column-major order.
    InverseDense {
        /// Matrix dimension.
        dimension: usize,
        /// Frozen inverse-mass factor entries; never inverted on restart.
        lower: Vec<f64>,
    },
}
impl FrozenGeometry {
    pub(super) fn capture(metric: &WarmupMetric) -> Self {
        match metric {
            WarmupMetric::Identity(m) => Self::Identity(m.dimension()),
            WarmupMetric::Diagonal(m) => Self::Diagonal(m.as_slice().to_vec()),
            WarmupMetric::Dense(m) => Self::Dense {
                dimension: m.dimension(),
                lower: m.factor().as_slice().to_vec(),
            },
            WarmupMetric::InverseDiagonal(m) => Self::InverseDiagonal(m.as_slice().to_vec()),
            WarmupMetric::InverseDense(m) => Self::InverseDense {
                dimension: m.dimension(),
                lower: m.factor().as_slice().to_vec(),
            },
            _ => unreachable!("session only constructs reference covariance metrics"),
        }
    }
    pub(super) fn restore(&self) -> Result<WarmupMetric, MetricError> {
        Ok(match self {
            Self::Identity(d) => WarmupMetric::Identity(IdentityMetric::new(*d)),
            Self::Diagonal(v) => WarmupMetric::Diagonal(DiagonalMetric::new(buffer(v))?),
            Self::Dense { dimension, lower } => WarmupMetric::Dense(DenseMetric::new(
                CholeskyFactor::new_lower(*dimension, buffer(lower))?,
            )),
            Self::InverseDiagonal(v) => {
                WarmupMetric::InverseDiagonal(InverseDiagonalMetric::new(buffer(v))?)
            }
            Self::InverseDense { dimension, lower } => WarmupMetric::InverseDense(
                InverseDenseMetric::new(CholeskyFactor::new_lower(*dimension, buffer(lower))?),
            ),
        })
    }
    pub(super) fn values(&self) -> usize {
        match self {
            Self::Identity(_) => 0,
            Self::Diagonal(v) | Self::InverseDiagonal(v) => v.len(),
            Self::Dense { lower, .. } | Self::InverseDense { lower, .. } => lower.len(),
        }
    }
    pub(super) fn dimension(&self) -> usize {
        match self {
            Self::Identity(d)
            | Self::Dense { dimension: d, .. }
            | Self::InverseDense { dimension: d, .. } => *d,
            Self::Diagonal(v) | Self::InverseDiagonal(v) => v.len(),
        }
    }
}

/// Version-two retained-boundary restart state. Fields are read-only through
/// accessors and validated on decode/admission. Mid-warmup checkpoints are not
/// supported: no adaptive controller or partially estimated geometry is saved.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub(super) provenance: Provenance,
    pub(super) chain: ChainId,
    pub(super) completed: usize,
    pub(super) word_position: u128,
    pub(super) kernel: SamplerConfig,
    pub(super) geometry: FrozenGeometry,
    pub(super) position: Vec<f64>,
    pub(super) logp: f64,
    pub(super) gradient: Vec<f64>,
    pub(super) normal_spare: Option<f64>,
    pub(super) numerical_streak: usize,
}
impl Checkpoint {
    /// Model, seed and implementation identity.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    /// Logical chain and RNG stream identity.
    pub fn chain(&self) -> ChainId {
        self.chain
    }
    /// Absolute retained completion count; the next emitted index.
    pub fn completed(&self) -> usize {
        self.completed
    }
    /// Coherent saved unconstrained point.
    pub fn position(&self) -> &[f64] {
        &self.position
    }
    /// Frozen kernel settings, including adapted step size.
    pub fn sampler(&self) -> SamplerConfig {
        self.kernel
    }
    /// Frozen momentum/proposal geometry.
    pub fn geometry(&self) -> &FrozenGeometry {
        &self.geometry
    }
    /// Write a new checkpoint, flush and sync file contents. Existing paths are
    /// never overwritten. I/O failure can leave a partial file for inspection.
    /// # Errors
    /// Returns I/O, byte-budget or format failure. See the documented crash scope.
    pub fn write_new(
        &self,
        path: impl AsRef<Path>,
        maximum_bytes: usize,
    ) -> Result<(), CheckpointError> {
        super::storage::write_checkpoint(self, path.as_ref(), maximum_bytes)
    }
    /// Decode with a hard file-size budget, preserving every f64 bit and RNG word.
    /// # Errors
    /// Rejects unknown versions/phases, corruption, truncation, nonfinite caches,
    /// invalid geometry, trailing data and budgets before running a model.
    pub fn read(path: impl AsRef<Path>, maximum_bytes: usize) -> Result<Self, CheckpointError> {
        super::storage::read_checkpoint(path.as_ref(), maximum_bytes)
    }
    pub(super) fn matches(&self, q: &[f64], logp: f64, gradient: &[f64]) -> bool {
        bits(&self.position, q)
            && self.logp.to_bits() == logp.to_bits()
            && bits(&self.gradient, gradient)
    }
    pub(super) fn valid(&self) -> bool {
        let d = self.provenance.dimension;
        self.provenance.valid()
            && self.position.len() == d
            && self.position.iter().all(|x| x.is_finite())
            && self.logp.is_finite()
            && self.geometry.dimension() == d
            && self.geometry.restore().is_ok()
            && self.word_position < (1_u128 << 68)
            && self.numerical_streak <= self.completed
            && self.normal_spare.is_none_or(f64::is_finite)
            && match self.kernel {
                SamplerConfig::Rwmh(_) => {
                    self.gradient.is_empty() && matches!(self.geometry, FrozenGeometry::Identity(_))
                }
                _ => {
                    self.gradient.len() == d
                        && self.gradient.iter().all(|x| x.is_finite())
                        && self.normal_spare.is_none()
                }
            }
    }
}
fn bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Durable checkpoint validation failure. Never attempts numerical recovery.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// Filesystem or truncated read failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Invalid schema, phase, state, checksum or trailing data.
    #[error("invalid or unsupported checkpoint")]
    Format,
    /// File or decoded value exceeds its explicit byte budget.
    #[error("checkpoint exceeds byte budget")]
    Budget,
}
