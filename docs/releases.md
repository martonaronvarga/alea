# Release and compatibility policy

Alea is pre-alpha. Every crate remains unpublished, and no API has completed an
S3 release-candidate cycle. This policy defines how a crate can graduate; it does
not declare the workspace stable. The [maturity policy](maturity-and-promotion.md)
and machine-readable public API inventory continue to own component status.

## Independent release units

The package split and upstream attributions are specified in
[LICENSING.md](../LICENSING.md): six reusable libraries use MIT OR Apache-2.0;
ffi/distributions/cli use GPL-3.0-or-later because they include or depend on the
GPL native cubature path. Every package includes the appropriate full license
texts and third-party notices. Metadata and actual archive checks enforce this.

| Crate | First intended release scope | Present release constraints |
|---|---|---|
| `alea-math` | Initialized aligned buffers; identity, diagonal and dense Euclidean metrics | Reference APIs have documentation enforcement; platform results and candidate cycle pending |
| `alea-core` | Checked density/score, coherent point caches, transforms and layouts | Same; model assembly and higher-order capabilities graduate separately |
| `alea-mcmc` | Fixed HMC/RWMH, multinomial NUTS, explicit covariance warmup | Same; named research modules retain their existing maturity and exactness limits |
| `alea-runtime` | Reference sessions, deterministic logical chain identity, bounded output and diagnostics | Same; checkpoint v2 and draw-stream v1 compatibility require a candidate cycle |
| `alea-distributions` | Individually validated normalized laws and models | GPL native dependency; candidate evidence pending |
| `alea-autodiff` | Analytic adapters and explicitly enabled Enzyme | Compiler/plugin pins and derivative orders graduate independently |
| `alea-ffi` | Audited synchronous cubature | Vendored GPL-2.0-or-later sources, native toolchain and sanitizer evidence |
| `alea-smc` | No stable surface proposed | Contracts only; filters and PMCMC are not implemented |
| `alea-cli` | No stable inference CLI proposed | Experiment commands do not define a stable user workflow or schema |

Use explicit `reference` imports in core/math/MCMC/runtime. Cargo defaults enable
no research feature. Research algorithms require named imports, and approximate
or exactness-pending algorithms retain their explicit type names. A crate's first
stable release must either place experimental APIs behind a clearly documented
unstable feature with a compatibility exclusion, or include them in its semver
promise. The present namespace split alone does not establish a Rust semver
exception. No research export is being frozen by this preparation milestone.

## Compiler, features and platforms

Rust edition 2024, MSRV **1.95.0**. The dependency lock is part of source releases.
The portable CI lane installs 1.95.0 and current stable independently. MSRV changes
require release notes and a minor-version release while pre-1.0; after 1.0 they
require advance notice for at least one minor release and 90 days. Patch releases
preserve their declared MSRV. Consumers of libraries should resolve dependencies
with an MSRV-aware Cargo resolver and report lock-resolution failures separately.

| Host/target | Reference configuration | Evidence required before support claim |
|---|---|---|
| Linux x86_64 GNU | Scalar stable/MSRV; pinned nightly extras | Native tests, docs, fresh project, Miri and FFI sanitizers |
| Linux aarch64 GNU | Scalar stable/MSRV | Native portable CI and fresh project |
| macOS x86_64 | Scalar stable/MSRV | Native portable CI and fresh project |
| macOS aarch64 | Scalar stable/MSRV | Native portable CI and fresh project |
| Windows x86_64 MSVC | Scalar stable/MSRV | Native portable CI and C compiler integration |
| Other architectures, wasm, musl, no_std, GPU | Unclaimed | Separate execution and backend evidence |

These are the proposed support lanes. Workflow configuration is not evidence of a
passing run. The release notes must link successful runs for the candidate source
revision. Floating-point replay is only promised within the same implementation,
target semantics, compiler/options, dependency versions and numerical backend.
Thread scheduling does not change logical chain seeds; it can change interleaved
output order. No cross-CPU or cross-backend bitwise identity is promised.

| Feature | Crates | Toolchain and resource contract |
|---|---|---|
| Default (empty) | All | Stable scalar float64; no CPU-specific flags required |
| `faer` | math, MCMC, runtime | Stable, optional numerical backend and research fitting; explicit caller-owned parallelism |
| `simd` | math, MCMC, distributions | Pinned nightly portable SIMD; CPU instruction selection remains the compiler's baseline |
| `openblas` | math, MCMC | LP64 OpenBLAS development/runtime libraries; avoid nested BLAS threads |
| `branch-hints` | distributions | Pinned nightly; no statistical or API change |
| `std-autodiff` | autodiff, CLI | Separately pinned matching nightly/Enzyme component, release/LTO execution required |

Do not use `--all-features` on stable. Use `nix develop .#stable`, the default
nightly shell, `.#blas`, or `.#autodiff` as appropriate. Their content pins are in
`flake.lock`; no undeclared compiler installation is needed for the default build.
Optional backends remain outside the first reference compatibility promise.

## Error, panic, determinism and cost contracts

Checked constructors reject invalid dimensions, budgets, nonfinite parameters and
incompatible checkpoint identities before numerical work. Model/backend errors
retain their concrete source types. Point caches commit atomically after successful
evaluation. Scratch and consumed RNG are not rolled back on error or unwind.
Raw target callbacks must honor buffer lengths and overwrite their advertised
outputs. A callback panic propagates at low-level target/chain APIs; sessions
capture supported worker/observer panics after joining workers. Process abort,
allocation failure and an indefinitely blocked user callback cannot be contained.
Method-specific `Errors`, `Panics` and `Safety` sections remain authoritative.

Storage is O(d) for vectors/diagonal geometry, O(d²) for dense geometry and
O(d × tree depth) for reference NUTS scratch. Dense factorization is cold O(d³)
work. A depth bound is not a wall-time guarantee. Collection additionally costs
O(chains × draws × d); streaming with collection disabled bounds retained output
memory. Diagnostic windows, channel capacities, output bytes and session resources
are explicit budgets. Cancellation/deadlines are observed between bounded steps,
not by interrupting arbitrary user code. Frozen checkpoints do not encode adaptive
warmup state. Durable paths use create-new semantics and preserve readable complete
frames on failure; power-loss directory durability requires filesystem support.

## Fresh downstream workflow

From a repository checkout or verified unpacked source release:

```sh
cargo fetch --manifest-path crates/Cargo.toml --locked
python3 tools/release/fresh.py .
```

The runner creates a separate temporary project with its own manifest and lock,
copies [the complete example](../tools/release/examples/reference.rs), and runs four
analytic Gaussian chains with covariance warmup and frozen NUTS using only public
APIs. It resolves and fetches that consumer's own dependency graph, then executes
offline. A workspace fetch alone may omit independently resolved packages or
platform features; use `--offline` only when the consumer graph is already cached.
Production dependencies
for this example require no native cubature, Enzyme, Nix, repository test fixtures
or test-only helper crates. See [modeling](m14-modeling.md) and
[sessions](m15-sessions.md) for native model and durable cancel/resume workflows.

For native regression and positive-parameter models, diagnostics, bounded SBC,
cancellation and exact durable recovery, run the fuller independent consumer:

```sh
python3 tools/release/fresh.py . --workflow --output /path/to/new-workflow-output
```

It retains `report.json`, a checkpoint and durable continuation draws. The
[candidate protocol](reference-candidate-protocol.md) declares its seeds, budgets,
analytic checks and descriptive performance observations. The package verifier
runs this same workflow against the contents of actual `.crate` archives.

## Source artifacts and signing

```sh
python3 tools/release/source.py /tmp/alea-source.tar.gz --signing-key /path/to/maintainer-key.pub
# With the signer's identity/key already trusted independently:
ssh-keygen -Y verify -f /path/to/trusted-allowed-signers -I maintainer@example.org \
  -n alea-source-v1 -s /tmp/alea-source.tar.gz.sha256.sig < /tmp/alea-source.tar.gz.sha256
cd /tmp
sha256sum -c alea-source.tar.gz.sha256
python3 /path/to/trusted/alea/tools/release/verify.py /tmp/alea-source.tar.gz \
  --allowed-signers /path/to/trusted-allowed-signers --identity maintainer@example.org
```

The source writer includes tracked and new source plus initialized submodule
contents, normalizes timestamps/ownership/modes, and uses deterministic gzip
headers. A manifest hashes every included input. Agent state, build directories,
local keys and unrelated files are excluded by an explicit directory allowlist.
Never put credentials in source directories. Existing artifact paths are refused.
Two builds with the pinned Python/zlib and identical source bytes/modes produce
identical archive bytes;
compiler-dependent executable reproducibility is a separate claim. Changing a
source file changes the archive digest. A valid signature attests to the digest,
not to platform execution or scientific validity.

Signing is optional for local preparation and **mandatory for publication**. Supply
an established maintainer identity; do not generate a replacement release key as
part of a candidate run. No key is configured by this repository. Retain the
archive, digest, detached signature, independently distributed signing key,
platform logs and compatibility report together. The tool never publishes or
changes Git state. Registry packaging must also be verified from each crate's
actual `cargo package` output before lifting its `publish = false` guard.

SSH signer authorization includes a namespace. A Git-only `allowed_signers` entry
does not authorize verification in `alea-source-v1`. Use a separate independently
trusted source-signers record for that namespace, anchored to the established
maintainer public key; preserve any restrictions on the Git signer record. A
successful source signature is preparation evidence, not a candidate-cycle result.

For unpublished workspace packages, run
`python3 tools/release/packages.py /tmp/alea-package-verification` after fetching
dependencies. It assembles all nine `.crate` archives without package-local locks,
unpacks them and builds a fresh consumer using local registry patches. This
handles the existing cyclic dev fixtures without pretending that unpublished
siblings already exist on crates.io. Source releases retain the workspace lock.
The command verifies package contents and consumption, not registry publication.

## Candidate and stable release procedure

1. Select the crates and exact intended surface. Verify package licensing and
   retain all dependency notices. Update independent versions and release notes.
2. Run reference portable MSRV/stable lanes, the pinned feature matrix, Miri,
   native ASan/UBSan, property/parser suites and operational soaks. Retain failures.
3. Validate the API inventory and run `tools/release/semver.sh BASELINE_SOURCE`
   against an independently retained accepted source baseline. The wrapper calls
   [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) for every
   reference crate. First-candidate bootstrap needs an explicit reviewed baseline;
   comparing the candidate with itself is not compatibility evidence.
4. Reproduce the fresh-project workflow from the archived source, verify package
   contents and make two matching source archives. Sign the accepted artifact.
5. Record candidate source digest, per-crate API/feature/schema scope, successful
   platform runs, open defects, downstream feedback and soak results. Complete an
   S3 candidate review with at least one independent downstream exercise. A
   correctness/unsoundness/data-loss/schema defect resets the cycle
   after its repair and requires a new candidate artifact.
6. Close the candidate only when there are no unresolved release-blocking defects.
   Publish each eligible crate's compatibility scope and window; graduate that
   crate alone. Do not infer S4 from a green local suite or from other crates.

The platform labels follow [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Configured CI runs are not replaced by local cross-compilation. Optional research,
native FFI and Enzyme components can remain pre-alpha while the reference surface
progresses. Benchmark superiority is not a release gate; practical reliability and
descriptive performance evidence are sufficient for baseline assessment.

For a future 1.x reference release, semver covers every public item unless an
explicitly named unstable feature excludes it. Maintain the current stable minor
and its predecessor with correctness/security fixes for at least 12 months after
the successor's release. Deprecations remain usable for at least two minor
releases and 90 days, and removals require a major version. Breaking checkpoint or
draw-stream changes require a new schema version and a migration reader or
explicit typed rejection; never reinterpret old bytes under a new meaning.

## Compiled-target ABI

The [ABI document](compiled-target-abi.md) remains a design specification. No
versioned Alea compiled-target ABI has two real production consumers. BridgeStan
benchmark callbacks and analytic Rust traits are distinct existing boundaries,
not consumers of that proposed ABI. Implement it only after two concrete consumers
require it, with ownership, destructor pairing, panic containment, size/version
negotiation, concurrency and compatibility tests. No empty ABI is added for M17.
