//! Streamed, named sampling tables with a sealed JSON analysis manifest.
#![deny(missing_docs)]
mod data;
mod schema;
use crate::output::{ChainId, Draw, Phase, TransitionStats};
use alea_core::{
    model::{
        assembly::{PointwiseError, PointwiseLikelihood},
        prediction::NamedOutput,
    },
    transform::TransformError,
};
pub use data::{ArrayValues, DataArray, DataAxis, ObservedData, PosteriorPredictive};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, BufWriter, Write},
};

/// Version of the CSV/JSON analysis contract; independent of checkpoint storage.
pub const SCHEMA_VERSION: u32 = 2;

/// Caller-supplied identities and method settings; not inferred from type names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Exact model identity, preferably a content digest.
    pub model: String,
    /// Exact input-data identity, preferably a content digest.
    pub data: String,
    /// Source/build identity, including revision and relevant toolchain/features.
    pub source: String,
    /// Inference method, such as NUTS; not an exactness or convergence claim.
    pub method: String,
    /// Declared inference settings and seeds in a deterministic key order.
    pub settings: BTreeMap<String, String>,
}

/// Sampling metadata and optional likelihood reporting, validated before output.
/// Native writing requires neither Python nor an external analysis package.
pub struct AnalysisSpec {
    /// Constrained names and chart layout; values are attached in the CSV rows.
    pub parameters: NamedOutput<()>,
    /// Source, model, data and inference identities.
    pub provenance: Provenance,
    /// All planned logical chains, including any that produce no rows.
    pub chains: Vec<ChainId>,
    /// Whether to include the free-coordinate vector alongside physical values.
    pub save_unconstrained: bool,
    /// Scalar generated quantities, supplied in this order on every successful row.
    pub predictions: Vec<String>,
    /// Optional physical units keyed by exact CSV value-column name.
    pub units: BTreeMap<String, String>,
    /// A validated predictive partition. Its coordinate schema must match parameters.
    pub likelihood: Option<PointwiseLikelihood>,
    /// Typed observations with explicit predictive observation IDs.
    pub observed_data: Vec<ObservedData>,
    /// Typed model inputs which are not modeled as random observations.
    pub constant_data: Vec<DataArray>,
    /// Explicit replicated-outcome bindings; other predictions remain generic quantities.
    pub posterior_predictive: Vec<PosteriorPredictive>,
}
impl AnalysisSpec {
    /// Starts an analysis without pointwise likelihoods or generated quantities.
    pub fn new(parameters: NamedOutput<()>, provenance: Provenance, chains: Vec<ChainId>) -> Self {
        Self {
            parameters,
            provenance,
            chains,
            save_unconstrained: false,
            predictions: Vec::new(),
            units: BTreeMap::new(),
            likelihood: None,
            observed_data: Vec::new(),
            constant_data: Vec::new(),
            posterior_predictive: Vec::new(),
        }
    }
    /// Uses a pointwise model's exact named layout without copying graph/data.
    /// # Errors
    /// Propagates invalid named coordinate metadata.
    pub fn from_likelihood(
        likelihood: PointwiseLikelihood,
        provenance: Provenance,
        chains: Vec<ChainId>,
    ) -> Result<Self, AnalysisError> {
        let parameters = NamedOutput::new(
            likelihood.model().coordinates(),
            likelihood.model().layout().clone(),
            (),
        )?;
        let mut spec = Self::new(parameters, provenance, chains);
        spec.likelihood = Some(likelihood);
        Ok(spec)
    }
}

/// Why a planned chain stopped. Completion does not imply convergence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainStatus {
    /// The declared computation finished; reporting failures remain counted.
    Completed,
    /// Deliberately stopped before finishing, with an explicit reason.
    Interrupted(String),
    /// Failed, including chains which never produced a draw.
    Failed(String),
}

/// Schema, ordering, numerical or stream failure during analysis export.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    /// Invalid schema, row, logical index or final chain declaration.
    #[error("invalid analysis record: {0}")]
    Invalid(&'static str),
    /// A previous I/O failure/unwind makes continuing this stream unsafe.
    #[error("analysis stream is poisoned after an incomplete write")]
    Poisoned,
    /// Original transform/metadata failure; admitted draw failures are recorded.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Original pointwise failure; no successful row was written.
    #[error(transparent)]
    Pointwise(#[from] PointwiseError),
    /// Original output-stream failure. Discard the unsealed artifact.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// JSON manifest serialization failure. Discard the unsealed artifact.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Default)]
struct PhaseCount {
    first: Option<u64>,
    last: Option<u64>,
    rows: u64,
    failures: u64,
}
#[derive(Default)]
struct ChainCount {
    warmup: PhaseCount,
    sampling: PhaseCount,
}

/// Bounded sampling CSV stream, sealed with SHA-256 and a final JSON manifest.
///
/// Rows preserve logical chain IDs, warmup/sampling phase and absolute indices.
/// No draws are buffered across calls. Numeric fields are finite binary64 values
/// formatted for exact round trips; absent optional statistics are empty fields.
/// Invalid logical keys leave counters unchanged; admitted payload errors become
/// failure rows. Any I/O error poisons the writer; a partial stream must not be retried or treated as a valid artifact.
#[must_use]
pub struct AnalysisWriter<W: Write> {
    csv: BufWriter<W>,
    spec: AnalysisSpec,
    columns: Vec<String>,
    counts: BTreeMap<u64, ChainCount>,
    hash: Sha256,
    poisoned: bool,
    parameters: Vec<f64>,
    scratch: Vec<f64>,
    likelihood: Vec<f64>,
    prediction_domains: Vec<data::PredictionDomain>,
}
impl<W: Write> AnalysisWriter<W> {
    /// Validates metadata and writes a quoted UTF-8 CSV header with CRLF records.
    /// # Errors
    /// Rejects invalid identities, chain/quantity names, mismatched likelihood
    /// schemas or unknown unit columns before touching the output stream.
    pub fn new(csv: W, spec: AnalysisSpec) -> Result<Self, AnalysisError> {
        let columns = schema::validate(&spec)?;
        let counts = spec
            .chains
            .iter()
            .map(|id| (id.0, ChainCount::default()))
            .collect();
        let mut writer = Self {
            csv: BufWriter::new(csv),
            parameters: vec![0.; spec.parameters.layout().constrained_dimension()],
            scratch: vec![0.; spec.parameters.layout().scratch_dimension()],
            likelihood: vec![0.; spec.likelihood.as_ref().map_or(0, |p| p.units().len())],
            prediction_domains: data::prediction_domains(&spec),
            spec,
            columns,
            counts,
            hash: Sha256::new(),
            poisoned: false,
        };
        let header = encode_csv(&writer.columns);
        writer.write_bytes(&header)?;
        Ok(writer)
    }

    /// Immutable schema defining column order and numerical meaning.
    pub fn spec(&self) -> &AnalysisSpec {
        &self.spec
    }

    /// Writes a committed draw, constraining parameters without a Jacobian and
    /// evaluating the declared pointwise units. Predictions follow schema order.
    /// # Errors
    /// Rejects invalid versions/keys before writing. Once the logical key is
    /// admitted, shape/statistic/numerical failures write a failure record and
    /// consume that index before returning the original error. Do not retry it.
    /// I/O errors or callback panics poison the stream; panics propagate.
    pub fn write_draw<T>(
        &mut self,
        draw: &Draw<'_, T>,
        predictions: &[f64],
    ) -> Result<(), AnalysisError> {
        if self.poisoned {
            return Err(AnalysisError::Poisoned);
        }
        if draw.schema_version != crate::output::SCHEMA_VERSION {
            return Err(AnalysisError::Invalid("draw schema version"));
        }
        let index =
            u64::try_from(draw.index).map_err(|_| AnalysisError::Invalid("draw index overflow"))?;
        self.validate_key(draw.chain, draw.phase, index)?;
        // Leave poison set if user model code unwinds during preparation.
        self.poisoned = true;
        let prepared = self.prepare_draw(draw, predictions);
        self.poisoned = false;
        let row = match prepared {
            Ok(row) => row,
            Err(error) => {
                self.write_failure(draw.chain, draw.phase, index, &error.to_string())?;
                return Err(error);
            }
        };
        self.write_bytes(&encode_csv(&row))?;
        self.commit_key(draw.chain, draw.phase, index, false);
        Ok(())
    }

    /// Preserves an unsuccessful reporting/evaluation attempt with its original
    /// logical index and explanation. All numeric payload fields are absent.
    /// # Errors
    /// Rejects empty errors and invalid/duplicate indices. I/O failure poisons output.
    pub fn write_failure(
        &mut self,
        chain: ChainId,
        phase: Phase,
        index: u64,
        error: &str,
    ) -> Result<(), AnalysisError> {
        self.validate_key(chain, phase, index)?;
        if error.is_empty() {
            return Err(AnalysisError::Invalid("empty failure reason"));
        }
        let mut row = vec![
            chain.0.to_string(),
            phase_name(phase).into(),
            index.to_string(),
            "failure".into(),
            error.into(),
        ];
        row.resize(self.columns.len(), String::new());
        self.write_bytes(&encode_csv(&row))?;
        self.commit_key(chain, phase, index, true);
        Ok(())
    }

    /// Flushes CSV, then writes and flushes the final manifest. Every planned
    /// chain needs exactly one status, even if it produced no rows. Neither CSV
    /// nor a partially written manifest is valid on error. No files are created.
    /// # Errors
    /// Rejects missing/duplicate/unknown statuses or empty failure reasons;
    /// preserves CSV/manifest I/O errors. Does not perform fsync or atomic rename.
    pub fn finish<M: Write>(
        mut self,
        mut manifest: M,
        statuses: &[(ChainId, ChainStatus)],
    ) -> Result<W, AnalysisError> {
        if self.poisoned {
            return Err(AnalysisError::Poisoned);
        }
        let statuses = schema::statuses(&self.spec, statuses)?;
        self.poisoned = true;
        self.csv.flush()?;
        let digest = format!("{:x}", self.hash.clone().finalize());
        let value = schema::manifest(&self.spec, &self.columns, &self.counts, &statuses, &digest);
        serde_json::to_writer_pretty(&mut manifest, &value)?;
        manifest.write_all(b"\n")?;
        manifest.flush()?;
        self.csv
            .into_inner()
            .map_err(|e| AnalysisError::Io(e.into_error()))
    }

    fn prepare_draw<T>(
        &mut self,
        draw: &Draw<'_, T>,
        predictions: &[f64],
    ) -> Result<Vec<String>, AnalysisError> {
        if draw.position.len() != self.spec.parameters.layout().unconstrained_dimension()
            || predictions.len() != self.spec.predictions.len()
        {
            return Err(AnalysisError::Invalid("coordinate or prediction shape"));
        }
        if draw
            .position
            .iter()
            .chain(predictions)
            .any(|x| !x.is_finite())
        {
            return Err(AnalysisError::Invalid("nonfinite coordinate or prediction"));
        }
        if !self
            .prediction_domains
            .iter()
            .zip(predictions)
            .all(|(domain, value)| domain.contains(*value))
        {
            return Err(AnalysisError::Invalid(
                "predictive value outside observed datatype domain",
            ));
        }
        validate_stats(draw.stats)?;
        self.spec.parameters.layout().constrain_values(
            draw.position,
            &mut self.parameters,
            &mut self.scratch,
        )?;
        if let Some(likelihood) = &self.spec.likelihood {
            likelihood.evaluate(draw.position, &mut self.likelihood)?;
        }
        let mut row = vec![
            draw.chain.0.to_string(),
            phase_name(draw.phase).into(),
            draw.index.to_string(),
            "draw".into(),
            String::new(),
        ];
        row.extend(self.parameters.iter().map(ToString::to_string));
        if self.spec.save_unconstrained {
            row.extend(draw.position.iter().map(ToString::to_string));
        }
        row.extend(stats_fields(draw.stats));
        row.extend(
            self.likelihood
                .iter()
                .chain(predictions)
                .map(ToString::to_string),
        );
        Ok(row)
    }
    fn validate_key(&self, chain: ChainId, phase: Phase, index: u64) -> Result<(), AnalysisError> {
        if self.poisoned {
            return Err(AnalysisError::Poisoned);
        }
        let count = self
            .counts
            .get(&chain.0)
            .ok_or(AnalysisError::Invalid("unplanned chain"))?;
        if phase == Phase::Warmup && count.sampling.rows != 0 {
            return Err(AnalysisError::Invalid("warmup after sampling"));
        }
        let count = match phase {
            Phase::Warmup => &count.warmup,
            Phase::Sampling => &count.sampling,
        };
        if count
            .last
            .is_some_and(|last| last.checked_add(1) != Some(index))
            || count.rows == u64::MAX
        {
            return Err(AnalysisError::Invalid(
                "nonconsecutive or overflowing draw index",
            ));
        }
        Ok(())
    }
    fn commit_key(&mut self, chain: ChainId, phase: Phase, index: u64, failure: bool) {
        let count = self.counts.get_mut(&chain.0).expect("validated chain");
        let count = match phase {
            Phase::Warmup => &mut count.warmup,
            Phase::Sampling => &mut count.sampling,
        };
        count.first.get_or_insert(index);
        count.last = Some(index);
        count.rows += 1;
        count.failures += u64::from(failure);
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), AnalysisError> {
        self.poisoned = true;
        self.csv.write_all(bytes)?;
        self.hash.update(bytes);
        self.poisoned = false;
        Ok(())
    }
}
fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Warmup => "warmup",
        Phase::Sampling => "sampling",
    }
}
fn encode_csv(fields: &[String]) -> Vec<u8> {
    let mut result = Vec::new();
    for (i, value) in fields.iter().enumerate() {
        if i != 0 {
            result.push(b',');
        }
        result.push(b'"');
        for &byte in value.as_bytes() {
            result.push(byte);
            if byte == b'"' {
                result.push(byte);
            }
        }
        result.push(b'"');
    }
    result.extend_from_slice(b"\r\n");
    result
}
fn validate_stats(s: TransitionStats) -> Result<(), AnalysisError> {
    if !s.acceptance_probability.is_finite()
        || !(0. ..=1.).contains(&s.acceptance_probability)
        || s.energy.is_some_and(|e| !e.is_finite())
    {
        return Err(AnalysisError::Invalid("invalid transition statistics"));
    }
    Ok(())
}
fn stats_fields(s: TransitionStats) -> [String; 7] {
    [
        u8::from(s.accepted).to_string(),
        s.acceptance_probability.to_string(),
        s.energy.map(|x| x.to_string()).unwrap_or_default(),
        s.divergent
            .map(|x| u8::from(x).to_string())
            .unwrap_or_default(),
        s.depth_saturated
            .map(|x| u8::from(x).to_string())
            .unwrap_or_default(),
        s.integration_steps
            .map(|x| x.to_string())
            .unwrap_or_default(),
        s.tree_depth.map(|x| x.to_string()).unwrap_or_default(),
    ]
}
