# Unbuilt research archive

M0 preserves exploratory Wiener implementations, C++ comparison programs and
their local compiled outputs here, outside Cargo source discovery. Nothing in
this directory is a supported library implementation or part of CI.

`manifest.sha256` records original paths and bytes at archival. Source files keep
their basenames in `wiener/`; local executable artifacts live in `bin/` and are
ignored by Git. They were moved, not deleted, and can be moved back if needed.
Old relative includes/build commands may need adjustment before reuse.

The active implementation remains `crates/kernels/src/dist/wiener/`. No active
Wiener source or experimental input data is removed by this archive operation.
