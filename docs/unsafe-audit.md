# M1 unsafe and foreign-boundary audit

Audited 2026-09-27. Scope: all repository-owned, compiled Rust `unsafe` code,
including optional backends, plus the C contracts those boundaries depend on.
This is not a formal verification of transitive dependencies or every upstream C
entry point. Uncompiled research files do not constitute supported APIs.

## Inventory and evidence

| Boundary | Invariants | Validation |
|---|---|---|
| `memory::OwnedBuffer` | Initialized storage, 64-byte base alignment, no raw-parts escape | Safe wrapper over pinned aligned-vec; six strict Miri tests |
| `kernels::metric` OpenBLAS calls | Full initialized column-major lower factor; checked lengths; `i32` dimensions; LP64 ABI; disjoint borrows; synchronous calls | Build-time C ABI guard, independent matrix references, combined-backend native tests; scalar metric equivalents under Miri |
| `ffi::cubature_raw` | ABI matches vendored `cubature.h`; enum discriminants match C; `size_t` is `usize` | Header/source audit and native integration tests |
| `ffi::cubature` | Checked dimensions/bounds/tolerances/output sizes; matching live userdata; serial callbacks; no pointer retention; initialized output; no unwinding through C | Direct production-trampoline tests under strict Miri, native error/panic roundtrips, foreign allocation-failure harness |
| FFI test callbacks | Matching userdata and initialized outputs before Rust references | Safety comments and native comparisons |

The workspace denies unsafe operations outside explicit unsafe blocks, undocumented
unsafe blocks, and missing unsafe-function safety docs. Target evaluation, phase
evolution, validated configuration and aligned storage forbid unsafe Rust.
No repository-owned unsafe allocation implementation remains.

## Cubature findings and fixes

The upstream source is pinned by the cubature submodule at
`4ba70a7d595a00cfb15b02763013e0088013adb7`. The Rust boundary exposes only synchronous,
serial `hcubature`, not the vector/parallel entry points.

1. The previous safe function-pointer wrapper could invoke a callback whose
   userdata contract did not match its `T`. Both `hcubature_fn` and
   `hcubature_fn_into` are now **unsafe**. Prefer the safe closure API.
2. C scratch can be uninitialized. The generic trampoline initializes every raw
   output element to NaN **before** forming a mutable slice; incomplete/non-finite
   successful writes are rejected. This also permits a safe closure to read its
   output slice without observing uninitialized memory.
3. The trampoline catches unwinding, stores the panic payload without dropping it
   in C, latches failure, and returns a failure code. Rust resumes the panic only
   after C returns. `panic=abort` still aborts; this cannot recover from aborts or
   panicking panic hooks. Caller-provided raw callbacks must not unwind.
4. Safe entry points revalidate publicly constructed `Bounds`, finite ordered
   endpoints, finite centers/widths, tolerances, dimensions and output lengths.
   The Genz–Malik point count and four-region allocation capacity are checked
   against C unsigned indexing and Rust addressability limits. Oversized settings
   fail before output allocation. Null callbacks are rejected.
5. Generic stack-owned callback state replaces an internally introduced trait
   object. Borrowed closures cannot escape the integration call. No state is
   global; independent calls have independent callback state.
6. Inspection found that upstream callback failures leaked pending/queued regions,
   and failed heap reallocations lost the old allocation. The build applies
   exact-match, fail-closed fixes from `crates/ffi/cubature_patch.rs` into `OUT_DIR`;
   the submodule is untouched. Region ownership transfers clear the source slot;
   all remaining owners are destroyed on failure. Reallocations retain their old
   allocation on failure. Heap indices use `size_t`, growth is checked, and the
   region count is bounded for upstream unsigned iteration. Upstream source
   changes require reviewing these patches. The upstream GPL notice is retained.

`tests/cubature_cleanup.c` compiles against that exact generated source and tracks
every C allocation. In dimensions 1–3 it injects failure at each allocation in a
baseline integration and at callback positions 1–100, including after region
splits. Every run must finish with zero live allocations. The native Rust test
executes this binary; it is not linked into production Rust. A local ASan/UBSan
run of the same harness also passed without findings. Foreign integration is
not represented as Miri coverage.

## Error and compatibility behavior

- `CubatureError` is now a typed, non-exhaustive `thiserror` enum, not an integer
  tuple. Closure error codes are retained separately from backend error codes.
- `Bounds::try_new` is the fallible constructor; `Bounds::new` remains a panicking
  compatibility convenience. `Bounds::dim` returns `usize`, avoiding truncation.
- Validation failures do not invoke callbacks or mutate result buffers. Results
  after backend/callback failures or panics are unspecified; do not consume them.
- C success after an evaluation budget is exhausted is not a convergence claim;
  callers must inspect error estimates. `max_eval=0` retains upstream unlimited
  evaluation semantics, subject to checked storage/index limits and available memory.
- Non-finite aggregate results/error estimates are rejected by the safe wrapper.
- Rust wrapper success paths allocate zero bytes with caller-provided buffers.
  The C integrator still allocates. Malloc fault injection checks cleanup; it is
  not evidence of allocation-free integration or an end-to-end speedup.

## Reproduction and limits

From the repository root:

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml -p ffi --locked
nix develop --command bash tools/ci/miri.sh
nix develop .#blas --command bash tools/ci/check.sh kernels/simd,kernels/faer,kernels/openblas,runtime/experimental
```

Miri executes the production Rust trampoline with representative C-like raw
buffers, including `MaybeUninit`, and the supported pure-Rust metric/target/phase
paths. It cannot execute arbitrary C/OpenBLAS. Native C failure injection,
sanitizers and backend reference tests are complementary, not interchangeable.
No audit here proves third-party code safe under every platform, allocator failure,
resource-exhaustion scenario or malicious `unsafe` caller.
