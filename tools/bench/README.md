# Manifest-driven inference experiments

The [native-backend extension](../../docs/m16-native-backends.md) adds all twenty
native Rust/Enzyme and JAX ports, cross-backend conformance, shared-target Rust
comparisons, serial/parallel observations and bounded research configurations.
Use `alea-bench native campaign OUTPUT --models REFERENCE_MODELS` after preparing
the pinned compiled reference models. Historical BridgeStan bundles stay separate.

`alea-bench` is the M16 experiment utility. The Rust executable dispatches to the
standard-library Python package `alea_bench`; its private worker/probe/analyzer
modes use Alea's reference NUTS and validated runtime diagnostics. No library API,
numerical default, or dependency edge is added.

From the repository root with recursive submodules available:

```sh
nix develop .#stable
cargo build --manifest-path crates/Cargo.toml -p alea-cli --bin alea-bench --locked
python3 tools/bench/alea-bench prepare tools/bench/protocols/reference-v1.json target/engines
python3 tools/bench/alea-bench seal tools/bench/protocols/reference-v1.json target/manifest.json
python3 tools/bench/alea-bench run target/manifest.json target/engines/engines.json target/run
python3 tools/bench/alea-bench verify target/run
python3 tools/bench/alea-bench report target/run --output target/second-report.json
python3 tools/bench/alea-bench track target/run --output target/tracking.json
```

Alternatively, `reproduce MANIFEST NEW_DIRECTORY` builds all pinned engines and
executes the entire experiment from that manifest. Source bytes must match its
seal. All output paths are create-new. Compilers, dependency fetches and builds
run before measurements. `prepare` needs Nix, network access and a C++ toolchain;
jobs require Linux GNU time and taskset. The offline CI tests need only Python,
GNU time/taskset and a built debug `alea-bench`.

The version-one structural contract is in `schemas/v1.json`; `schema.py` enforces
cross-field budgets, types and digest seals. `manifest.py`, `runner.py` and
`report.py` are reusable without the CLI. `generate_corpus.py --check` and
`schema_document.py --check` reject accidental generated-file drift. Version one
implements the roadmap's V1 accounting and V2 diagnostic requirements; those
names do not denote two competing JSON schema versions.

See [the method and evidence limits](../../docs/m16-experiments.md) and the checked
[M16 evidence](../../docs/benchmarks/m16/) for reproducing a published bundle.


## Descriptive PosteriorDB extension

The v2 pipeline measures 20 PosteriorDB posteriors and 26 configurations, including
Fisher, ChEES, WALNUTS, isokinetics, SoftAbs/manifold RMHMC, transport,
ATLAS-component and Pathfinder workflows. It produces descriptive diagnostics,
reference discrepancies and costs without promotion gates. Process completion uses
Linux pidfd notification and nanosecond timestamps instead of timeout polling.

```sh
python3 tools/bench/alea-bench pdb experiment tools/bench/protocols/posteriordb-v2.json target/new-pdb-experiment
python3 tools/bench/alea-bench pdb verify target/new-pdb-experiment/run
python3 tools/bench/alea-bench pdb pack target/new-pdb-experiment/run target/new-pdb-bundle
python3 tools/bench/alea-bench pdb unpack target/new-pdb-bundle target/restored-pdb
```

See [method, budgets and accounting](../../docs/m16-posteriordb.md) before comparing
engines or experimental kernels. The standalone Rust worker does not alter the
production crate graph. JSON protocol schema: `schemas/v2.json`; offline contract
and artifact readers live alongside the existing v1 implementation.
