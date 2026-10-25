# Workflow validation evidence

The [receipt](receipt.json) records the exact scope and hashes. The full nine-crate workspace passed 440 tests with one ignored, all-target checks, Clippy and rustdoc. Affected SIMD/Faer/OpenBLAS contracts and 10 Miri tests passed. 6938 public API items, five neutral variants and two Faer snapshots were reviewed. The benchmark suite passed 44 tests with two optional skips; the compiled third-AD oracle separately executed both tests on dusk.

The CI commands ran locally in pinned Nix environments. Hosted GitHub Actions was not dispatched. One preservation test exposed a stale historical-file binding; the original failing log, strict repair test and resumed passing gate log are all retained. No historical timing artifact was rewritten. The original 1560-job v2 evidence reproduces identically with current and archived readers.

Numerical validation is bounded evidence, not general sampler superiority or a finite-RMHMC exactness proof.

The executed [Stan reference](stan-adaptation-reference/README.md) adds 400 recurrence and 1711 schedule observations. Four short-budget window-end differences remain explicit. These fixtures do not claim whole-workflow equivalence.
