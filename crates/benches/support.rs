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
