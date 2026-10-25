# Release and compatibility policy

Alea's first stable release is being prepared as **0.1.0** for `alea-math`,
`alea-core`, `alea-mcmc` and `alea-runtime`. The [release review](releases/0.1.0/README.md)
records its exact candidate and outstanding gates. Preparation is not publication:
all registry guards remain closed until the signed candidate has passed its review.
S4 starts only when the reviewed signed artifacts are published. No crate has
completed that step yet.

The [previous candidate](releases/2026-10-25-reference-candidate/README.md) is
historical evidence for its frozen revision. The current ordinary-module/prelude
API deliberately breaks that unpublished revision. There are no compatibility
aliases. The first published artifact establishes the compatibility baseline.
Release qualification belongs to a crate/artifact review; the
[maturity policy](maturity-and-promotion.md) separately records numerical evidence.

## Independent release units

The package split and upstream attributions are specified in
[LICENSING.md](../LICENSING.md): six reusable libraries use MIT OR Apache-2.0;
ffi/distributions/cli use GPL-3.0-or-later because they include or depend on the
GPL native cubature path. Every package includes the appropriate full license
texts and third-party notices. Metadata and actual archive checks enforce this.

| Crate | First intended release scope | Present release constraints |
|---|---|---|
| `alea-math` | Initialized aligned buffers; identity, diagonal and dense Euclidean metrics | Prior candidate reviewed; crate-wide stable compatibility and publication remain open |
| `alea-core` | Checked density/score, coherent point caches, transforms and layouts | Prior candidate reviewed; model assembly and higher-order capabilities graduate separately |
| `alea-mcmc` | Fixed HMC/RWMH, multinomial NUTS, explicit covariance warmup | Prior candidate reviewed; named research modules retain their maturity and exactness limits |
| `alea-runtime` | Reference sessions, deterministic logical chain identity, bounded output and diagnostics | Prior candidate reviewed; reviewed checkpoint v2/draw-stream v1 scope, no cross-build replay promise |
| `alea-distributions` | Individually validated normalized laws and models | GPL native dependency; candidate evidence pending |
| `alea-autodiff` | Analytic adapters and explicitly enabled Enzyme | Compiler/plugin pins and derivative orders graduate independently |
| `alea-ffi` | Audited synchronous cubature | Vendored GPL-2.0-or-later sources, native toolchain and sanitizer evidence |
| `alea-smc` | No stable surface proposed | Contracts only; filters and PMCMC are not implemented |
| `alea-cli` | No stable inference CLI proposed | Experiment commands do not define a stable user workflow or schema |

Use normal owning modules or each crate's curated `prelude`. Cargo defaults enable
no research feature. Research algorithms require named imports; approximate and
exactness-pending algorithms retain their explicit type names.

For the four release crates, the compatibility promise covers **every public item
and Cargo feature**, including experimental APIs. An experimental label describes
numerical evidence and intended use; it is not permission to break a public Rust
API within a compatible release. No unstable feature, release namespace or hidden
compatibility alias is introduced. Research numerical maturity and default choices
remain unchanged. Optional nightly/native backends retain their documented
compiler and system requirements, independently of Rust API compatibility.

### Compatibility and maintenance window

The first published 0.1.0 artifact starts the baseline. Versions 0.1.x preserve
source compatibility, documented trait laws, errors, feature names and the declared
MSRV. A breaking change requires 0.2.0 or a later incompatible release, migration
notes, and an explicit API/semver review. The former unpublished reference paths
are outside this baseline. API snapshots inventory the full public surface;
`cargo-semver-checks` compares later releases with the published source, not with
the candidate itself.

Maintain the current compatible release line and its predecessor with correctness
and security fixes for at least 12 months after the successor's publication.
Deprecations stay usable for at least two minor releases and 90 days; removals
require an incompatible version. If an urgent soundness/security repair cannot
preserve behavior, publish the advisory and the smallest necessary correction,
and document its compatibility impact. Numerical bug fixes may change results;
bitwise identity is only promised under the replay conditions below.

After 1.0, incompatible API removals require a major version. This policy does not
make unchecked research methods exact or turn S0/S1 mechanisms into validated
algorithms. Report security issues through [SECURITY.md](../SECURITY.md); ordinary
bugs use the issue tracker with version, features, platform and a reproducer.

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
   comparing the candidate with itself is not compatibility evidence. For the
   initial unpublished-to-0.1.0 transition only, pass `--initial`: the wrapper
   verifies the actual old API snapshots against the digest-bound breaking review
   and reports no semver compatibility claim. Workflow dispatch exposes this as
   `initial_release`; the first `v0.1.0` tag uses that mode explicitly. Future
   releases use the independently retained published baseline and real semver checks.
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

Breaking checkpoint or draw-stream changes require a new schema version and a
migration reader or explicit typed rejection; never reinterpret old bytes under a
new meaning. Current checkpoints use `ALEACP02`, `session/2`, and `numerics/3`;
draw streams and owned records use schema 1. Cross-build replay remains unclaimed.

## Compiled-target ABI

The [ABI document](compiled-target-abi.md) remains a design specification. No
versioned Alea compiled-target ABI has two real production consumers. BridgeStan
benchmark callbacks and analytic Rust traits are distinct existing boundaries,
not consumers of that proposed ABI. Implement it only after two concrete consumers
require it, with ownership, destructor pairing, panic containment, size/version
negotiation, concurrency and compatibility tests. No empty ABI is added for M17.
