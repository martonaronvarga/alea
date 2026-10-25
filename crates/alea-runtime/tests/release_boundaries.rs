//! Release input-fuzzing and operational soak boundaries, independent of moments.
use alea_core::density::LogDensity;
use alea_mcmc::reference::RwmhOptions;
use alea_runtime::{
    ensemble::Execution,
    session::{
        Checkpoint, Kernel, SessionBuilder, Status,
        storage::{Discard, DrawReader, DurableSink},
    },
};
use std::{
    convert::Infallible,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Normal;
impl LogDensity for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
    fn logp(&self, q: &[f64]) -> Result<f64, Self::Error> {
        Ok(-0.5 * q[0] * q[0])
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "alea-release-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn builder(draws: usize) -> SessionBuilder {
    SessionBuilder::new("release-normal-v1", 1, vec![vec![0.0]])
        .seed([17; 32])
        .kernel(Kernel::Rwmh(RwmhOptions::new(0.5).unwrap()))
        .draws(draws)
        .collect(false)
}

#[test]
fn bounded_readers_survive_truncations_and_seeded_mutations_of_valid_streams() {
    let directory = Directory::new();
    let path = directory.0.join("draws");
    let session = builder(8).build().unwrap();
    let sink = DurableSink::create_new(&path, session.provenance(), 65_536).unwrap();
    let outcome = session
        .run_density(|_| Ok(Normal), Execution::Sequential, &sink)
        .unwrap();
    let bytes = std::fs::read(path).unwrap();
    let scan = |input: &[u8]| {
        if let Ok(mut reader) = DrawReader::new(input, 4096) {
            let mut frames = 0;
            while let Ok(Some(_)) = reader.next_draw() {
                frames += 1;
                assert!(
                    frames <= input.len(),
                    "decoder did not make bounded progress"
                );
            }
        }
    };
    for length in 0..=bytes.len() {
        scan(&bytes[..length]);
    }
    let checkpoint_path = directory.0.join("checkpoint");
    outcome.chains[0]
        .checkpoint
        .as_ref()
        .unwrap()
        .write_new(&checkpoint_path, 4096)
        .unwrap();
    let checkpoint = std::fs::read(&checkpoint_path).unwrap();
    let corrupted_path = directory.0.join("mutated-checkpoint");
    let mut random = 0x17_ab_23_cd_u64;
    for iteration in 0..4096 {
        // A fixed integer PRNG makes the byte corpus independent of rand changes.
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let mut mutated = bytes.clone();
        let at = random as usize % mutated.len();
        mutated[at] ^= 1 << (iteration % 8);
        scan(&mutated);
        if iteration < 256 {
            let mut mutated = checkpoint.clone();
            let at = random as usize % mutated.len();
            mutated[at] ^= 1 << (iteration % 8);
            std::fs::write(&corrupted_path, &mutated).unwrap();
            assert!(Checkpoint::read(&corrupted_path, 4096).is_err());
        }
    }
}

#[test]
#[ignore = "operational release soak: 409,600 retained steps across 200 frozen restarts"]
fn repeated_frozen_restart_matches_one_uninterrupted_stream_with_bounded_collection() {
    const CYCLES: usize = 200;
    const CHUNK: usize = 2048;
    let mut checkpoint = None;
    for cycle in 1..=CYCLES {
        let mut spec = builder(cycle * CHUNK);
        if let Some(cp) = checkpoint.take() {
            spec = spec.restart(vec![cp]);
        }
        let result = spec
            .build()
            .unwrap()
            .run_density(|_| Ok(Normal), Execution::Sequential, &Discard)
            .unwrap();
        assert!(matches!(
            result.status,
            Status::Success | Status::DiagnosticWarning
        ));
        assert_eq!(result.chains[0].completed, cycle * CHUNK);
        checkpoint = result.chains.into_iter().next().unwrap().checkpoint;
    }
    let uninterrupted = builder(CYCLES * CHUNK)
        .build()
        .unwrap()
        .run_density(|_| Ok(Normal), Execution::Sequential, &Discard)
        .unwrap();
    assert_eq!(checkpoint, uninterrupted.chains[0].checkpoint);
}
