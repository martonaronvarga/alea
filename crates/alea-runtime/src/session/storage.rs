//! Versioned owned streams and non-overwriting durable files.
//!
//! Channels apply fail-fast backpressure: a full queue is an explicit error with
//! the undelivered draw returned in the run outcome. No numerical worker blocks
//! on an async executor, subscriber or detached I/O thread.
use super::{Checkpoint, CheckpointError, FrozenGeometry, Kernel, OwnedDraw, Provenance};
use crate::output::{ChainId, Phase, TransitionStats};
use alea_mcmc::{HmcOptions, NutsOptions, RwmhOptions};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    path::Path,
    sync::{
        Mutex,
        mpsc::{SyncSender, TrySendError},
    },
};

/// Shared synchronous observer invoked after each completed retained transition.
/// Cross-chain interleaving is unspecified; each chain is strictly ordered.
/// Implementations bound their own memory/time. Panics are caught by the session
/// (the process panic hook still runs); process abort/OOM cannot be contained.
pub trait Sink: Sync {
    /// Bind a run identity before invoking any target factory.
    /// # Errors
    /// Returns incompatible provenance or backend admission failure.
    fn start(&self, _provenance: &Provenance) -> Result<(), SinkError> {
        Ok(())
    }
    /// Accept one owned-format record by borrow. Return errors without hiding loss.
    /// # Errors
    /// Reports backend failure or explicit output budget/backpressure exhaustion.
    fn draw(&self, draw: &OwnedDraw) -> Result<(), SinkError>;
    /// Drain/flush/join owned resources after all numerical workers have joined.
    /// # Errors
    /// Reports incomplete finalization; already emitted draws remain valid.
    fn finish(&self) -> Result<(), SinkError> {
        Ok(())
    }
}

/// Discard output while optionally collecting it in the session outcome.
#[derive(Debug, Clone, Copy, Default)]
pub struct Discard;
impl Sink for Discard {
    fn draw(&self, _: &OwnedDraw) -> Result<(), SinkError> {
        Ok(())
    }
}

/// Adapter for a thread-safe caller observer; no global subscriber is installed.
pub struct Observer<F>(pub F);
impl<F> Sink for Observer<F>
where
    F: Fn(&OwnedDraw) -> Result<(), SinkError> + Sync,
{
    fn draw(&self, draw: &OwnedDraw) -> Result<(), SinkError> {
        (self.0)(draw)
    }
}

/// Bounded channel observer with explicit full/disconnected semantics.
pub struct ChannelSink {
    sender: SyncSender<OwnedDraw>,
}
impl ChannelSink {
    /// Create a bounded stream. Capacity zero is rejected (no rendezvous races).
    /// # Errors
    /// Returns [`SinkError::Budget`] for zero capacity.
    pub fn bounded(
        capacity: usize,
    ) -> Result<(Self, std::sync::mpsc::Receiver<OwnedDraw>), SinkError> {
        if capacity == 0
            || capacity
                .checked_mul(size_of::<OwnedDraw>())
                .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(SinkError::Budget);
        }
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
        Ok((Self { sender }, receiver))
    }
}
impl Sink for ChannelSink {
    fn draw(&self, draw: &OwnedDraw) -> Result<(), SinkError> {
        self.sender.try_send(draw.clone()).map_err(|e| match e {
            TrySendError::Full(_) => SinkError::Backpressure,
            TrySendError::Disconnected(_) => SinkError::Disconnected,
        })
    }
}

/// Output failure with no rollback of completed transitions.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SinkError {
    /// Original filesystem/backend I/O error.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// Queue is full; the run returns the completed undelivered draw.
    #[error("bounded draw queue is full")]
    Backpressure,
    /// All channel subscribers were dropped.
    #[error("draw subscriber disconnected")]
    Disconnected,
    /// Explicit durable output byte budget exhausted.
    #[error("output storage budget exhausted")]
    Budget,
    /// An earlier write failed; continuing could corrupt the record stream.
    #[error("durable sink is failed or poisoned")]
    Failed,
    /// A run-admission or finalization callback panicked.
    #[error("sink lifecycle callback panicked")]
    ObserverPanic,
    /// Malformed/incompatible owned record.
    #[error("invalid owned draw or stream format")]
    Format,
}

struct FileState {
    file: File,
    bytes: usize,
    failed: bool,
    finished: bool,
}
/// A create-new, checksummed binary stream. Each frame is written under one lock;
/// `finish` flushes and syncs file contents after all workers join. An I/O error
/// poisons the stream; complete preceding frames remain readable. Directory-entry
/// persistence across power loss depends on caller filesystem/directory syncing.
pub struct DurableSink {
    state: Mutex<FileState>,
    provenance: Provenance,
    maximum_bytes: usize,
}
impl DurableSink {
    /// Create a new path with a versioned provenance header; never truncate/append.
    /// # Errors
    /// Returns existing-path/I/O failure or an insufficient header byte budget.
    pub fn create_new(
        path: impl AsRef<Path>,
        provenance: &Provenance,
        maximum_bytes: usize,
    ) -> Result<Self, SinkError> {
        if !provenance.valid() {
            return Err(SinkError::Format);
        }
        let mut header = Vec::new();
        header.extend_from_slice(b"ALEADW01");
        let mut payload = Vec::new();
        put_provenance(&mut payload, provenance);
        frame(&mut header, &payload);
        if header.len() > maximum_bytes {
            return Err(SinkError::Budget);
        }
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&header)?;
        Ok(Self {
            state: Mutex::new(FileState {
                file,
                bytes: header.len(),
                failed: false,
                finished: false,
            }),
            provenance: provenance.clone(),
            maximum_bytes,
        })
    }
}
impl Sink for DurableSink {
    fn start(&self, provenance: &Provenance) -> Result<(), SinkError> {
        if self.state.lock().map_err(|_| SinkError::Failed)?.finished {
            return Err(SinkError::Failed);
        }
        if provenance != &self.provenance {
            return Err(SinkError::Format);
        }
        Ok(())
    }
    fn draw(&self, draw: &OwnedDraw) -> Result<(), SinkError> {
        if !valid_draw(draw, self.provenance.dimension) {
            return Err(SinkError::Format);
        }
        let mut payload = Vec::new();
        put_draw(&mut payload, draw);
        let mut bytes = Vec::new();
        frame(&mut bytes, &payload);
        let mut state = self.state.lock().map_err(|_| SinkError::Failed)?;
        if state.failed || state.finished {
            return Err(SinkError::Failed);
        }
        let next = state
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= self.maximum_bytes)
            .ok_or(SinkError::Budget)?;
        if let Err(e) = state.file.write_all(&bytes) {
            state.failed = true;
            return Err(SinkError::Io(e));
        }
        state.bytes = next;
        Ok(())
    }
    fn finish(&self) -> Result<(), SinkError> {
        let mut state = self.state.lock().map_err(|_| SinkError::Failed)?;
        if state.failed {
            return Err(SinkError::Failed);
        }
        if state.finished {
            return Ok(());
        }
        if let Err(e) = state.file.flush().and_then(|()| state.file.sync_all()) {
            state.failed = true;
            return Err(SinkError::Io(e));
        }
        state.finished = true;
        Ok(())
    }
}

/// Incremental reader retaining at most one bounded frame. A truncated/corrupt
/// tail is an error after the complete prefix has been returned.
pub struct DrawReader<R> {
    reader: R,
    provenance: Provenance,
    maximum_frame_bytes: usize,
    failed: bool,
}
impl<R: Read> DrawReader<R> {
    /// Read the versioned provenance header with an allocation bound.
    /// # Errors
    /// Rejects malformed/truncated headers, unknown versions and oversized frames.
    pub fn new(mut reader: R, maximum_frame_bytes: usize) -> Result<Self, SinkError> {
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"ALEADW01" {
            return Err(SinkError::Format);
        }
        let payload = read_frame(&mut reader, maximum_frame_bytes)?.ok_or(SinkError::Format)?;
        let mut input = Input(&payload);
        let provenance = input.provenance().ok_or(SinkError::Format)?;
        // Draw data remains readable across numerical revisions; replay admission
        // separately requires the exact current implementation identity.
        if !input.0.is_empty() || !provenance.valid_format() {
            return Err(SinkError::Format);
        }
        Ok(Self {
            reader,
            provenance,
            maximum_frame_bytes,
            failed: false,
        })
    }
    /// Identity common to every record in this stream.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    /// Read the next complete record; clean EOF alone returns `None`.
    /// # Errors
    /// Returns malformed/checksum/truncation/budget errors without inventing draws.
    pub fn next_draw(&mut self) -> Result<Option<OwnedDraw>, SinkError> {
        if self.failed {
            return Err(SinkError::Failed);
        }
        let result = self.read_next();
        self.failed = result.is_err();
        result
    }
    fn read_next(&mut self) -> Result<Option<OwnedDraw>, SinkError> {
        let Some(payload) = read_frame(&mut self.reader, self.maximum_frame_bytes)? else {
            return Ok(None);
        };
        let mut input = Input(&payload);
        let draw = input.draw().ok_or(SinkError::Format)?;
        if !input.0.is_empty() || !valid_draw(&draw, self.provenance.dimension) {
            return Err(SinkError::Format);
        }
        Ok(Some(draw))
    }
}
fn valid_draw(draw: &OwnedDraw, dimension: usize) -> bool {
    let s = draw.stats;
    draw.schema_version == super::OWNED_SCHEMA_VERSION
        && draw.phase == Phase::Sampling
        && draw.position.len() == dimension
        && draw.position.iter().all(|v| v.is_finite())
        && s.acceptance_probability.is_finite()
        && (0.0..=1.0).contains(&s.acceptance_probability)
        && s.energy.is_none_or(f64::is_finite)
}
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, b| {
        (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}
fn frame(out: &mut Vec<u8>, payload: &[u8]) {
    put_u64(out, payload.len() as u64);
    out.extend_from_slice(payload);
    put_u64(out, checksum(payload));
}
fn read_frame(reader: &mut impl Read, maximum_bytes: usize) -> Result<Option<Vec<u8>>, SinkError> {
    let mut length = [0; 8];
    if reader.read(&mut length[..1])? == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut length[1..])?;
    let n = usize::try_from(u64::from_le_bytes(length)).map_err(|_| SinkError::Budget)?;
    if n > maximum_bytes || n > isize::MAX as usize {
        return Err(SinkError::Budget);
    }
    let mut payload = vec![0; n];
    reader.read_exact(&mut payload)?;
    let mut hash = [0; 8];
    reader.read_exact(&mut hash)?;
    if u64::from_le_bytes(hash) != checksum(&payload) {
        return Err(SinkError::Format);
    }
    Ok(Some(payload))
}
fn put_u64(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn put_usize(out: &mut Vec<u8>, n: usize) {
    put_u64(out, n as u64);
}
fn put_f64(out: &mut Vec<u8>, n: f64) {
    put_u64(out, n.to_bits());
}
fn put_string(out: &mut Vec<u8>, s: &str) {
    put_usize(out, s.len());
    out.extend_from_slice(s.as_bytes());
}
fn put_vec(out: &mut Vec<u8>, values: &[f64]) {
    put_usize(out, values.len());
    for &v in values {
        put_f64(out, v);
    }
}
fn put_option<T: Copy>(out: &mut Vec<u8>, v: Option<T>, put: impl Fn(&mut Vec<u8>, T)) {
    out.push(u8::from(v.is_some()));
    if let Some(v) = v {
        put(out, v);
    }
}
fn put_bool(out: &mut Vec<u8>, v: bool) {
    out.push(u8::from(v));
}
fn put_provenance(out: &mut Vec<u8>, p: &Provenance) {
    put_u64(out, u64::from(p.schema_version));
    put_string(out, &p.target_id);
    put_string(out, &p.implementation);
    put_usize(out, p.dimension);
    out.extend_from_slice(&p.seed);
}
fn put_draw(out: &mut Vec<u8>, d: &OwnedDraw) {
    put_u64(out, u64::from(d.schema_version));
    put_u64(out, d.chain.0);
    out.push(1);
    put_usize(out, d.index);
    put_vec(out, &d.position);
    let s = d.stats;
    put_bool(out, s.accepted);
    put_f64(out, s.acceptance_probability);
    put_option(out, s.energy, put_f64);
    put_option(out, s.divergent, put_bool);
    put_option(out, s.depth_saturated, put_bool);
    put_option(out, s.integration_steps, put_usize);
    put_option(out, s.tree_depth, put_usize);
    put_bool(out, d.numerical_rejection);
}
struct Input<'a>(&'a [u8]);
impl<'a> Input<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (a, b) = self.0.split_at_checked(n)?;
        self.0 = b;
        Some(a)
    }
    fn byte(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn usize(&mut self) -> Option<usize> {
        usize::try_from(self.u64()?).ok()
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }
    fn bool(&mut self) -> Option<bool> {
        match self.byte()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    fn option<T>(&mut self, get: impl Fn(&mut Self) -> Option<T>) -> Option<Option<T>> {
        match self.byte()? {
            0 => Some(None),
            1 => Some(Some(get(self)?)),
            _ => None,
        }
    }
    fn string(&mut self) -> Option<String> {
        let n = self.usize()?;
        if n > 4096 {
            return None;
        }
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
    fn vec(&mut self) -> Option<Vec<f64>> {
        let n = self.usize()?;
        let bytes = self.take(n.checked_mul(8)?)?;
        Some(
            bytes
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().expect("eight bytes")))
                .collect(),
        )
    }
    fn provenance(&mut self) -> Option<Provenance> {
        Some(Provenance {
            schema_version: u32::try_from(self.u64()?).ok()?,
            target_id: self.string()?,
            implementation: self.string()?,
            dimension: self.usize()?,
            seed: self.take(32)?.try_into().ok()?,
        })
    }
    fn kernel(&mut self) -> Option<Kernel> {
        let kind = self.byte()?;
        let step = self.f64()?;
        match kind {
            0 => {
                let steps = self.usize()?;
                let energy = self.f64()?;
                Some(Kernel::Hmc(
                    HmcOptions::new(step, steps)
                        .ok()?
                        .with_max_energy_error(energy)
                        .ok()?,
                ))
            }
            1 => {
                let depth = self.usize()?;
                let energy = self.f64()?;
                Some(Kernel::Nuts(
                    NutsOptions::new(step, depth)
                        .ok()?
                        .with_max_energy_error(energy)
                        .ok()?,
                ))
            }
            2 => Some(Kernel::Rwmh(RwmhOptions::new(step).ok()?)),
            _ => None,
        }
    }
    fn geometry(&mut self) -> Option<FrozenGeometry> {
        match self.byte()? {
            0 => Some(FrozenGeometry::Identity(self.usize()?)),
            1 => Some(FrozenGeometry::Diagonal(self.vec()?)),
            2 => Some(FrozenGeometry::Dense {
                dimension: self.usize()?,
                lower: self.vec()?,
            }),
            3 => Some(FrozenGeometry::InverseDiagonal(self.vec()?)),
            4 => Some(FrozenGeometry::InverseDense {
                dimension: self.usize()?,
                lower: self.vec()?,
            }),
            _ => None,
        }
    }
    fn draw(&mut self) -> Option<OwnedDraw> {
        let schema_version = u32::try_from(self.u64()?).ok()?;
        let chain = ChainId(self.u64()?);
        if self.byte()? != 1 {
            return None;
        }
        Some(OwnedDraw {
            schema_version,
            chain,
            phase: Phase::Sampling,
            index: self.usize()?,
            position: self.vec()?,
            stats: TransitionStats {
                accepted: self.bool()?,
                acceptance_probability: self.f64()?,
                energy: self.option(Self::f64)?,
                divergent: self.option(Self::bool)?,
                depth_saturated: self.option(Self::bool)?,
                integration_steps: self.option(Self::usize)?,
                tree_depth: self.option(Self::usize)?,
            },
            numerical_rejection: self.bool()?,
        })
    }
    fn checkpoint(&mut self) -> Option<Checkpoint> {
        let provenance = self.provenance()?;
        if self.byte()? != 1 {
            return None;
        }
        Some(Checkpoint {
            provenance,
            chain: ChainId(self.u64()?),
            completed: self.usize()?,
            word_position: u128::from_le_bytes(self.take(16)?.try_into().ok()?),
            kernel: self.kernel()?,
            geometry: self.geometry()?,
            position: self.vec()?,
            logp: self.f64()?,
            gradient: self.vec()?,
            normal_spare: self.option(Self::f64)?,
            numerical_streak: self.usize()?,
        })
    }
}
fn checkpoint_payload(cp: &Checkpoint) -> Vec<u8> {
    let mut out = Vec::new();
    put_provenance(&mut out, &cp.provenance);
    out.push(1);
    put_u64(&mut out, cp.chain.0);
    put_usize(&mut out, cp.completed);
    out.extend_from_slice(&cp.word_position.to_le_bytes());
    match cp.kernel {
        Kernel::Hmc(o) => {
            out.push(0);
            put_f64(&mut out, o.step_size().value());
            put_usize(&mut out, o.integration_steps().get());
            put_f64(&mut out, o.max_energy_error());
        }
        Kernel::Nuts(o) => {
            out.push(1);
            put_f64(&mut out, o.step_size().value());
            put_usize(&mut out, o.max_depth().value());
            put_f64(&mut out, o.max_energy_error());
        }
        Kernel::Rwmh(o) => {
            out.push(2);
            put_f64(&mut out, o.step_size().value());
        }
    }
    match &cp.geometry {
        FrozenGeometry::Identity(d) => {
            out.push(0);
            put_usize(&mut out, *d);
        }
        FrozenGeometry::Diagonal(v) => {
            out.push(1);
            put_vec(&mut out, v);
        }
        FrozenGeometry::Dense { dimension, lower } => {
            out.push(2);
            put_usize(&mut out, *dimension);
            put_vec(&mut out, lower);
        }
        FrozenGeometry::InverseDiagonal(v) => {
            out.push(3);
            put_vec(&mut out, v);
        }
        FrozenGeometry::InverseDense { dimension, lower } => {
            out.push(4);
            put_usize(&mut out, *dimension);
            put_vec(&mut out, lower);
        }
    }
    put_vec(&mut out, &cp.position);
    put_f64(&mut out, cp.logp);
    put_vec(&mut out, &cp.gradient);
    put_option(&mut out, cp.normal_spare, put_f64);
    put_usize(&mut out, cp.numerical_streak);
    out
}
pub(super) fn write_checkpoint(
    cp: &Checkpoint,
    path: &Path,
    maximum_bytes: usize,
) -> Result<(), CheckpointError> {
    if !cp.valid() {
        return Err(CheckpointError::Format);
    }
    let mut bytes = b"ALEACP01".to_vec();
    frame(&mut bytes, &checkpoint_payload(cp));
    if bytes.len() > maximum_bytes {
        return Err(CheckpointError::Budget);
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}
pub(super) fn read_checkpoint(
    path: &Path,
    maximum_bytes: usize,
) -> Result<Checkpoint, CheckpointError> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(
            u64::try_from(maximum_bytes)
                .map_err(|_| CheckpointError::Budget)?
                .checked_add(1)
                .ok_or(CheckpointError::Budget)?,
        )
        .read_to_end(&mut bytes)?;
    if bytes.len() > maximum_bytes {
        return Err(CheckpointError::Budget);
    }
    if bytes.get(..8) != Some(b"ALEACP01") {
        return Err(CheckpointError::Format);
    }
    let mut reader = &bytes[8..];
    let payload = read_frame(&mut reader, maximum_bytes)
        .map_err(|e| match e {
            SinkError::Io(e) => CheckpointError::Io(e),
            SinkError::Budget => CheckpointError::Budget,
            _ => CheckpointError::Format,
        })?
        .ok_or(CheckpointError::Format)?;
    let mut input = Input(&payload);
    let cp = input.checkpoint().ok_or(CheckpointError::Format)?;
    if !input.0.is_empty() || !reader.is_empty() || !cp.valid() {
        return Err(CheckpointError::Format);
    }
    Ok(cp)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn checkpoint() -> Checkpoint {
        Checkpoint {
            provenance: Provenance::new("fixture".into(), 1, [15; 32]),
            chain: ChainId(0),
            completed: 3,
            word_position: 25,
            kernel: Kernel::Hmc(HmcOptions::new(0.2, 3).unwrap()),
            geometry: FrozenGeometry::Identity(1),
            position: vec![0.0],
            logp: 0.0,
            gradient: vec![0.0],
            normal_spare: None,
            numerical_streak: 0,
        }
    }
    #[test]
    fn warmup_phase_unknown_schema_invalid_geometry_and_rng_are_rejected() {
        let cp = checkpoint();
        let mut payload = checkpoint_payload(&cp);
        let mut prefix = Vec::new();
        put_provenance(&mut prefix, &cp.provenance);
        payload[prefix.len()] = 0; // Warmup cannot be decoded as a retained checkpoint.
        assert!(Input(&payload).checkpoint().is_none());
        let mut cp = cp.clone();
        cp.word_position = 1_u128 << 68;
        assert!(!cp.valid());
        cp.word_position = 0;
        cp.geometry = FrozenGeometry::Diagonal(vec![-1.0]);
        assert!(!cp.valid());
        cp.geometry = FrozenGeometry::Identity(1);
        cp.provenance.schema_version = 2;
        assert!(!cp.valid());
        cp.provenance.schema_version = 1;
        cp.gradient[0] = f64::NAN;
        assert!(!cp.valid());
        let bytes = u64::MAX.to_le_bytes();
        assert!(matches!(
            read_frame(&mut bytes.as_slice(), 1024),
            Err(SinkError::Budget)
        ));
    }

    #[test]
    fn geometry_tags_preserve_mass_and_inverse_mass_factors_without_conversion() {
        for geometry in [
            FrozenGeometry::Identity(1),
            FrozenGeometry::Diagonal(vec![4.]),
            FrozenGeometry::Dense {
                dimension: 1,
                lower: vec![2.],
            },
            FrozenGeometry::InverseDiagonal(vec![0.25]),
            FrozenGeometry::InverseDense {
                dimension: 1,
                lower: vec![0.5],
            },
        ] {
            let mut cp = checkpoint();
            cp.geometry = geometry;
            assert!(cp.valid());
            let bytes = checkpoint_payload(&cp);
            let decoded = Input(&bytes).checkpoint().unwrap();
            assert_eq!(decoded, cp);
            assert_eq!(
                FrozenGeometry::capture(&decoded.geometry.restore().unwrap()),
                cp.geometry
            );
        }
        let mut cp = checkpoint();
        cp.provenance.implementation = cp
            .provenance
            .implementation
            .replace(";numerics/3", ";numerics/2");
        assert!(
            !cp.valid(),
            "a prior numerical implementation cannot silently resume"
        );
        cp = checkpoint();
        cp.geometry = FrozenGeometry::InverseDiagonal(vec![0.]);
        assert!(!cp.valid());
        cp.geometry = FrozenGeometry::InverseDense {
            dimension: 1,
            lower: vec![-1.],
        };
        assert!(!cp.valid());
    }

    #[test]
    fn older_draw_streams_remain_readable_without_allowing_checkpoint_replay() {
        let mut cp = checkpoint();
        cp.provenance.implementation = cp
            .provenance
            .implementation
            .replace(";numerics/3", ";numerics/2");
        let mut bytes = b"ALEADW01".to_vec();
        let mut payload = Vec::new();
        put_provenance(&mut payload, &cp.provenance);
        frame(&mut bytes, &payload);
        let reader = DrawReader::new(bytes.as_slice(), 4096).unwrap();
        assert_eq!(reader.provenance(), &cp.provenance);
        assert!(!cp.valid());
    }
    #[test]
    fn backend_write_failure_poisoning_prevents_continuing_a_partial_stream() {
        // A read-only handle is a deterministic injected filesystem write error.
        let file = File::open(env!("CARGO_MANIFEST_DIR").to_owned() + "/Cargo.toml").unwrap();
        let sink = DurableSink {
            state: Mutex::new(FileState {
                file,
                bytes: 0,
                failed: false,
                finished: false,
            }),
            provenance: checkpoint().provenance,
            maximum_bytes: 1000,
        };
        let draw = OwnedDraw {
            schema_version: 1,
            chain: ChainId(0),
            phase: Phase::Sampling,
            index: 0,
            position: vec![0.0],
            stats: TransitionStats::default(),
            numerical_rejection: false,
        };
        assert!(matches!(sink.draw(&draw), Err(SinkError::Io(_))));
        assert!(matches!(sink.draw(&draw), Err(SinkError::Failed)));
        assert!(matches!(sink.finish(), Err(SinkError::Failed)));
    }
}
