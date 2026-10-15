# M0/M1 completion record — 2026-09-27

Both milestones meet their documented exit gates on the tested x86-64 Linux/Nix
environment. This closes repository stabilization and the safe core, not the
subsequent algorithm, statistical-validation or Enzyme milestones.

## Reproducible validation

Run from the repository root with the existing `flake.lock` and `Cargo.lock`:

| Command after `nix develop … --command` | Environment | Result |
|---|---|---|
| `bash tools/ci/check.sh` | `.#stable` | 144 tests + 6 doctests; format/check/Clippy/docs clean |
| `bash tools/ci/check.sh kernels/faer,runtime/experimental` | `.#stable` | 144 tests + 6 doctests; all gates clean |
| `bash tools/ci/check.sh kernels/simd,kernels/branch-hints,memory/simd` | default | 145 tests + 6 doctests; all gates clean |
| `bash tools/ci/check.sh kernels/simd,kernels/faer,kernels/openblas,runtime/experimental` | `.#blas` | 145 tests + 6 doctests; all gates clean |
| `cargo test --manifest-path crates/Cargo.toml --workspace --release --locked` | `.#stable` | 144 tests + 6 doctests |
| `bash tools/ci/miri.sh` | default | 64 passed; 6 intentionally ignored |

Native check scripts use `--all-targets` for compilation and Clippy, `--locked`,
and deny Clippy/rustdoc warnings. The Miri script enables strict provenance and
symbolic alignment checking. The six Miri exclusions are one native C roundtrip
and five long statistical tests; native checks execute them. Two legacy raw-C
tests and the native allocation/harness binaries are compile-gated out of Miri.
The Nix Miri driver prints its informational warning that `RUSTC` is ignored;
there were no Miri failures. Updated RWMH boundary tests were also rerun separately.

The unsupported `std-autodiff` feature fails with exit 101 and its intentional
M2 prerequisite diagnostic. A local C allocation-failure harness run under
AddressSanitizer and UndefinedBehaviorSanitizer completed without findings.
The checked C harness runs in every native workspace test job; sanitizer execution
was local supplementary validation, not a claimed sanitizer CI job.

The GitHub workflow is installed and its constituent commands ran locally.
No remote Actions run, branch-protection change, commit, or push is asserted.

## Implemented contracts

- Workspace metadata/dependencies/lints are centralized. Unfinished APIs are
  gated, supported backend features compose, and toolchain/source inputs are pinned.
  See [M0 baseline](stabilization.md).
- Aligned storage remains initialized and 64-byte base-aligned; no home-grown
  unsafe allocator was restored. The fused fallible target protocol and checked
  point cache remain the supported model boundary.
- `AcceptanceTarget` and `TreeDepth` complement `StepSize`; immutable private
  payloads prevent invalid settings after construction. Tree depth guarantees
  representable counts, not an affordable resource budget or implemented NUTS.
- `Rwmh::try_new`, `try_step`, and `try_sample` add typed configuration, dimension
  and storage-overflow errors. Compatibility methods remain panicking wrappers.
  Public legacy configuration is revalidated at fallible entry points.
- `TransformError` establishes the allocation-free error vocabulary. Transform
  algorithms/Jacobians are explicitly M2, not claimed implemented here.
- [Public phase evolution and persistent snapshots](hamiltonian-phase.md) preserve
  the complete target binding, cached point and momentum. Errors/panics cannot
  expose a partially advanced phase. Endpoint snapshots can be reused without
  allocating; NUTS tree construction/selection remains M5.
- [The unsafe/FFI audit](unsafe-audit.md) closes the repository-owned unsafe
  inventory and fixes callback unwinding, uninitialized output references,
  unchecked casts/shapes, and foreign failure-path leaks.

## Performance evidence and limits

Allocation counters stay at zero for repeated checked target evaluation, modern
HMC transitions (including dense masses and divergent endpoints), and new phase
restart/step/save/clone/update paths through dimension 129. The safe cubature
`_into` wrapper performs no Rust heap allocation on the measured success path;
C numerical integration still allocates and has separate ownership tests.

No claim is made that aligned storage always beats `Vec`, that these safety fixes
improve throughput, or that scalar/SIMD/Faer/BLAS have identical performance.
Existing benchmark records remain the baseline; later performance gates require
controlled end-to-end timing and ESS/sec measurements. Modern convergence
diagnostics and production-ready sampler certification are not M1 deliverables.

Research sources and four local binaries were moved into `archive/`, not deleted.
All 14 archived entries were checked against the recovery manifest. The cubature
submodule remains unmodified; generated C fixes live in normal repository files.

## Next boundary

M2 starts with parameter layout/transforms and the validated `std::autodiff`/Enzyme
compiler/adapter path. Stable analytic targets remain usable throughout. M3 still
needs its remaining external statistical and performance gates before adaptation,
NUTS, ChEES-HMC or RMHMC are presented as complete.
