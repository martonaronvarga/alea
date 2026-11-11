//! Shared cold benchmark configuration; no instrumentation in measured kernels.
use std::path::{Path, PathBuf};

/// Keep result identities disjoint across packages and executable targets.
pub fn criterion() -> criterion::Criterion {
    let root = std::env::var_os("CRITERION_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let target = std::env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
                    let workspace = manifest
                        .ancestors()
                        .find(|p| {
                            std::fs::read_to_string(p.join("Cargo.toml")).is_ok_and(|s| {
                                s.starts_with("[workspace]\n") || s.contains("\n[workspace]\n")
                            })
                        })
                        .unwrap_or(manifest);
                    workspace.join("target")
                });
            target.join("criterion")
        });
    criterion::Criterion::default().output_directory(
        &root
            .join(env!("CARGO_PKG_NAME"))
            .join(env!("CARGO_CRATE_NAME")),
    )
}

/// Cold fixed-work preflight shared by native and generated benchmark paths.
#[allow(dead_code)]
pub mod fixed_work {
    /// FNV-1a identity for within-run matching. Source sealing uses the runner's
    /// existing cryptographic manifest.
    pub fn identity(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    }

    /// Five fixed positive states, in coordinate declaration order. The fixture
    /// declares whether its evaluation coordinates are constrained.
    pub fn states(dimension: usize) -> Vec<Vec<f64>> {
        (0..5)
            .map(|state| {
                (0..dimension)
                    .map(|coordinate| 0.1 + coordinate as f64 * 0.01 + state as f64 * 0.025)
                    .collect()
            })
            .collect()
    }

    /// Mathematical and execution identities supplied by the existing fixture.
    pub struct Fixture<'a> {
        pub name: &'a str,
        pub model: u64,
        pub coordinates: u64,
        pub query: &'a str,
        pub algorithm: &'a str,
        pub arithmetic: &'a str,
        pub provider: &'a str,
    }

    /// Executes one first call and four steady calls, emitting receipts before
    /// propagating errors. Counts describe public fused target calls, including
    /// failed attempts. Internal compiler/validator calls are not inferred.
    pub fn run<E>(
        fixture: &Fixture<'_>,
        route: &str,
        bank: &[Vec<f64>],
        target_bytes: Option<usize>,
        mut evaluate: impl FnMut(&[f64], &mut [f64]) -> Result<f64, E>,
    ) -> Result<Vec<f64>, E> {
        assert_eq!(bank.len(), 5);
        let dimension = bank[0].len();
        assert!(bank.iter().all(|q| q.len() == dimension));
        let bank_bytes: Vec<_> = bank
            .iter()
            .flatten()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let bank_id = identity(&bank_bytes);
        let mut gradient = vec![0.; dimension];
        let mut answers = Vec::with_capacity(5 * (dimension + 1));
        let timing = std::env::var_os("ALEA_FIXED_WORK_TIMING").is_some();
        if timing {
            assert!(std::env::var_os("SLURM_JOB_ID").is_some());
            assert!(std::env::var_os("SLURMD_NODENAME").is_some());
        }
        for (phase, states) in [("first", &bank[..1]), ("steady", &bank[1..])] {
            let start = timing.then(std::time::Instant::now);
            let mut calls = 0;
            let mut completed = 0;
            let emit = |calls, completed, status, reason| {
                let elapsed = start.map(|start| start.elapsed().as_nanos());
                let target_bytes = target_bytes
                    .map(|bytes| bytes.to_string())
                    .unwrap_or_else(|| "unknown".into());
                eprintln!(
                    "ALEA_FIXED_WORK_V1\t{}\t{route}\t{phase}\t{}\t{}\t{}\t{}\t{:016x}\t{:016x}\t{bank_id:016x}\t{calls}\t{completed}\t{}\t{target_bytes}\t{status}\t{reason}",
                    fixture.name,
                    fixture.query,
                    fixture.algorithm,
                    fixture.arithmetic,
                    fixture.provider,
                    fixture.model,
                    fixture.coordinates,
                    (bank.len() + 1) * dimension * size_of::<f64>(),
                );
                if let Some(elapsed) = elapsed {
                    eprintln!(
                        "ALEA_PHASE_ELAPSED_V1\t{}\t{route}\t{phase}\t{elapsed}",
                        fixture.name
                    );
                }
            };
            for q in states {
                calls += 1;
                gradient.fill(f64::NAN);
                let value = match evaluate(q, &mut gradient) {
                    Ok(value) => value,
                    Err(error) => {
                        emit(calls, completed, "failed", "target-error");
                        return Err(error);
                    }
                };
                if !value.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
                    emit(calls, completed, "failed", "nonfinite-output");
                    panic!("fixed-work target returned non-finite output");
                }
                completed += 1;
                answers.push(value);
                answers.extend_from_slice(&gradient);
            }
            emit(calls, completed, "supported", "executed");
        }
        Ok(answers)
    }
}
