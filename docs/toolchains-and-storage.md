# Toolchains and SIMD storage

## Reproducible development

Run from the repository root. `flake.lock` pins Fenix, Nixpkgs, and their component
hashes. The default shell uses the complete nightly profile, so rustc, Clippy,
Miri, rust-src, rustfmt, and rust-analyzer come from one compiler date, not Fenix's
potentially mixed-date `latest` profile.

```sh
nix develop
# Or run noninteractively, without entering another shell:
nix develop --command cargo clippy --manifest-path crates/Cargo.toml --workspace --all-targets --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml --workspace --all-targets --locked
nix build .#checks.x86_64-linux.toolchain --no-link
```

Shells:

| Shell | Intended use |
|---|---|
| default / nightly | Ordinary development, Clippy, Miri, portable SIMD |
| stable | Stable-feature compilation and Clippy; no Miri or portable SIMD |
| blas | Nightly plus LP64 OpenBLAS for explicit `--features openblas` |
| full | Nightly plus optional Stan, Python/R, and profiling tools |
| autodiff | Content-pinned upstream nightly-2026-06-23 + matching Enzyme; release/fat-LTO M2 gate |

No shell needs `.nix-rust` or `.nix-cmdstan` links. The old links are preserved but
ignored. There is no nested `exec zsh`, automatic `RUSTC_BOOTSTRAP`, forced AVX2,
global BLAS link flag, or global autodiff flag. Native CPU tuning is an explicit
benchmark choice, not a prerequisite for opening the shell.
The opt-in BLAS/full shells provide matching LP64 headers/library and a scoped
Linux runtime library path; see the [metric backend contract](metric-contract.md).

The selected nightly is `2026-04-17`, rustc `1.97.0-nightly`, commit
`7af3402cda75aaead39f72516fd6cbb2f3ee0dbd`, LLVM `22.1.2`. Update the Fenix lock
deliberately, then rerun Clippy, Miri, tests, SIMD checks, and benchmarks.
The separately verified stable toolchain is Rust `1.95.0` (`59807616e`).

### Miri

```sh
nix develop --command cargo miri setup
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo miri test --manifest-path crates/Cargo.toml -p memory --lib --locked
```

`MIRI_LIB_SRC` points at this compiler's standard-library sources. Do not set
`MIRI_SYSROOT` to the ordinary native sysroot: Miri builds its own. The first setup
downloads standard-library build dependencies and populates the user's Miri cache.

The `memory` crate isolates the production buffer and its tests from C cubature and
other foreign dependencies. It is not a copied implementation; `kernels::buffer`
re-exports it. Passing this suite does **not** audit the remaining FFI or the entire
`aligned-vec` dependency API.

### Autodiff boundary

M2 supersedes the source-build experiment with upstream prebuilt nightly-2026-06-23
and its matching Enzyme component, using a content-hashed manifest in `flake.nix`.
The shell uses the workspace release profile's fat LTO and scopes AD flags to this
opt-in environment. `nix/enzyme.nix` and `nix/rust-toolchain.nix` are historical,
unused recipes, not the current compiler path. Actual generated derivatives now
execute in the [M2 validation gate](m2-model-boundary.md), including hybrid Wiener
custom derivatives. The compiler feature remains experimental; other platforms
and arbitrary opaque calls inside AD are not claimed validated.

## Storage contract

`OwnedBuffer<T>` wraps `aligned_vec::AVec<T, ConstAlign<64>>`. Its base alignment is
`max(64, align_of::<T>())`, including after growth and cloning. Elements are
contiguous with no per-element padding. The wrapper has the same size as
`Vec<f64>` on the tested platform and introduces no local unsafe blocks.

- `new` and `resize` expose only initialized elements; removed elements are dropped.
- `from_fn` constructs final values directly instead of zeroing and then overwriting.
- Shrinking/clearing retains capacity. Keep workspaces alive across transitions.
- Generic drop, panic cleanup, zero-size, and over-alignment tests remain required.
- Raw-parts and unchecked length manipulation are not part of the public API.
- The wrapper only uses the dependency's constructor, reserve/push/pop, truncate,
  clone, and slice APIs. Changes to this boundary require renewed Miri tests.

### Is Vec faster?

There is no universal ordering. `Vec<f64>` guarantees ordinary `f64` alignment,
not 32-/64-byte base alignment; it can nevertheless be used by auto-vectorized
loops and safe `Simd::from_slice` loads. The old allocator exposed uninitialized
values as Rust references and is not a valid performance baseline.

Alignment is a storage guarantee, not a guarantee of faster generated code. A
slice boundary can hide alignment from the optimizer. An explicitly aligned SIMD
view can preserve it, but arbitrary offsets and tails still need handling. For
64-byte-aligned `f64` data, an offset of `i` elements remains 64-byte aligned only
when `i % 8 == 0`. Dense matrix columns need a suitable stride; triangular solver
subranges often start at unaligned offsets even when the allocation is aligned.

The current metric kernels use safe `Simd::from_slice`. Do not replace these with
aligned-only pointer loads merely because the allocation base is aligned.

### Reproducing the comparison

```sh
nix develop --command env RUSTFLAGS='-Ctarget-cpu=native' \
  cargo bench --manifest-path crates/Cargo.toml --target-dir crates/target/buffer-bench \
  -p memory --bench storage --features simd --locked -- --noplot
```

The benchmark compares the same slice-based AXPY loop over `Vec` and aligned
storage, explicit four-lane SIMD loads over both, and aligned SIMD block views.
Allocation/initialization/destruction is a separate group. Sizes include a short
tail, small vectors, cache-resident working sets, and a working set beyond this
machine's last-level cache. Inputs and outputs are black-boxed; implementation
agreement is checked before timing. Actual pointer offsets are printed.

Run on an otherwise idle machine. CPU frequency, thermal state, allocator, ISA,
and working-set size matter. Record distributions, not only point estimates.
AXPY is a storage microbenchmark, not evidence of improved ESS/sec or faster
Wiener evaluation. HMC/metric end-to-end comparisons remain roadmap gates.

### Local results: 2026-09-24

Machine: Intel Core i5-8250U, x86-64 Linux, AVX2/FMA, 6 MiB last-level cache.
Compiler: the pinned nightly above, fat LTO and `-Ctarget-cpu=native`. Measurements
were repeated after concurrent validation builds finished: 30 samples, 0.5 s
warmup and 2 s measurement per case. CPU frequency was not locked. These are
single-machine exploratory results, not a cross-machine performance guarantee.

Mean times for the **same auto-vectorizable slice loop** (95% bootstrap intervals):

| Elements | Vec | 64-byte aligned |
|---:|---:|---:|
| 31 | 13.0 ns [12.8, 13.3] | 13.2 ns [12.9, 13.9] |
| 32 | 9.04 ns [8.52, 9.70] | 8.67 ns [8.62, 8.71] |
| 1,024 | 218 ns [214, 225] | 190 ns [189, 191] |
| 32,768 | 17.9 us [17.8, 18.0] | 17.1 us [16.9, 17.4] |
| 1,048,576 | 1.48 ms [1.45, 1.50] | 1.46 ms [1.44, 1.49] |

The 1,024-element aligned case used about 13% less time in this run; the small
and largest cases have overlapping intervals. Vec happened to be 64-byte aligned
for both 31-element inputs; at 1,024 elements the Vec input/output offsets modulo
64 were 48/16. Allocator luck is not an API guarantee.

Allocation/init/drop told a different story: at 32 elements Vec averaged 30.3 ns
versus 35.9 ns aligned; at 1,024, 381 ns versus 419 ns. This supports reusing
aligned workspaces instead of allocating in every numerical step.

Explicit SIMD was not universally faster: at 1,024 elements the indexed
`from_slice` variants averaged about 714–717 ns, and the aligned block variant
244 ns, versus 190 ns for the aligned auto-vectorizable loop. Loop shape, bounds
checks, unrolling, and instruction selection still need profiling; alignment alone
does not choose the best kernel. At the largest size the aligned block variant
averaged 1.37 ms. Do not infer that SIMD is inherently slow or that 4 lanes is
optimal on every CPU.

All 35 estimates, including explicit SIMD and allocation cases, are preserved in
[the CSV](benchmarks/storage-2026-09-24.csv). Values are Criterion **mean** estimates,
not necessarily the slope estimates printed in its terminal summary.

## Validation record

- `nix flake check --no-build`: passed on x86-64 Linux (evaluation only).
- Nix `checks.x86_64-linux.toolchain`: built and passed.
- Default/nightly workspace tests: 76 passed, plus 4 doctests and benchmark smoke tests.
- Stable 1.95.0: workspace/all-targets check passed.
- Workspace Clippy: passed with existing warnings; memory/all-targets/all-features
  Clippy passed with `-D warnings`.
- Miri: all 6 storage tests passed with strict provenance/symbolic alignment.
- Portable SIMD: 58 kernel tests (including offset/tail regression) and 6 HMC tests passed.
- Rust workspace formatting and edited Nix-file formatting passed.

Subsequent LP64 OpenBLAS/metric validation is recorded in the
[metric contract](metric-contract.md).

Not validated here: a full sandboxed application package build, the full research
shell, other OS/architectures, or compilation/execution of the custom
autodiff toolchain. When packaging from a Git flake, newly added source files must
be tracked by Git; Nix's Git source filter excludes untracked files.

References: [Fenix profiles](https://github.com/nix-community/fenix#usage),
[Miri configuration](https://github.com/rust-lang/miri#common-problems),
[Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html),
[portable SIMD](https://doc.rust-lang.org/std/simd/struct.Simd.html),
[aligned-vec](https://docs.rs/aligned-vec/0.6.4/aligned_vec/).
