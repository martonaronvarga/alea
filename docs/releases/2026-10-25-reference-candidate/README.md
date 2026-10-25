# Reference candidate review — 2026-10-25

Candidate code: [`148c22cab4517395f8a65568ca18c0a90eaa4bc5`](https://github.com/martonaronvarga/alea/commit/148c22cab4517395f8a65568ca18c0a90eaa4bc5).
Accepted baseline: `df17c0a0af7ab8aabc7753304e3beacd6d8c7135`.
The code revision is frozen separately from this subsequent evidence and policy
review, so the signed archive has no self-referential digest. Its preparation-time
READMEs describe the then-current S2 status; this review records the subsequent
S3 qualification without changing the archived implementation.

The reference candidate cycle completes the practical workflow, portability,
compatibility and artifact checks below. Four explicit reference namespaces
qualify as **S3 release candidates**. Every crate remains unpublished and pre-alpha;
there is no S4 compatibility commitment or registry release. Numerical
implementations retain their S2 validation classifications. Research algorithms,
Enzyme qualification and optional backends retain their separate maturity and
exactness obligations. Constructor defaults remain unchanged.

| Candidate namespace | Reviewed composition |
|---|---|
| `alea_core::reference` | Checked target/density, coherent point state, transforms and layouts |
| `alea_math::reference` | Initialized storage and constant Euclidean metric algebra |
| `alea_mcmc::reference` | Fixed HMC/RWMH, multinomial NUTS and finite covariance warmup |
| `alea_runtime::reference` | Sessions, finite warmup, diagnostics, bounded output and frozen recovery |

These are candidate import boundaries, not blanket graduation of everything in
the four crates. Actual Rust semver checks include all public exports of each
crate, including named research exports; namespaces alone create no semver
exemption. The new NUTS-only gradient-diagonal session profile is an explicit
option. Covariance warmup, generalized turning and zero jitter remain defaults.

## Executed gates

The [predeclared protocol](../../reference-candidate-protocol.md) fixes models,
seeds, budgets, tolerances and descriptive timing procedures.

| Gate | Result and evidence |
|---|---|
| Native platforms | All ten MSRV 1.95/current-stable lanes passed: Linux x86_64/aarch64, macOS Intel/ARM and Windows MSVC. Each runs contracts/docs and two independent consumers. [Release run](https://github.com/martonaronvarga/alea/actions/runs/40200300753), [retained jobs](evidence/release-jobs.json). |
| Optional features and AD | Scalar, Faer, nightly SIMD/branch hints, combined OpenBLAS and compiled Enzyme lanes passed. [Rust run](https://github.com/martonaronvarga/alea/actions/runs/40200300691), [retained jobs](evidence/rust-jobs.json). |
| Memory/ownership | All six strict-provenance and symbolic-alignment Miri partitions passed; the complete storage partition also passed locally. [Local log](evidence/miri-storage-local.log). |
| Native failures and operations | ASan/UBSan, parser mutations, configuration/transform properties and the 409,600-step/200-restart soak passed in the release run. |
| Rust compatibility | All four reference crates pass actual `cargo-semver-checks` against the accepted baseline. [Log](evidence/semver.log), [API/schema adjudication](evidence/api-review.json). Five session-warmup declarations were added; none removed. |
| Package consumption | All nine actual `.crate` archives have verified licenses/notices and a fresh public-API consumer; the full workflow also passes against unpacked packages. [Package hashes](evidence/packages.json), [workflow](evidence/packaged-workflow.json). All nine hashes match CI packaging. |
| Source consumption | The same full workflow passes from the verified unpacked signed source. [Results](evidence/archived-workflow.json). Two local archives, a no-Git reconstruction and the independent CI archive have identical SHA-256. |

The [artifact index](evidence/github-artifacts.json) and
[verification receipt](evidence/artifact-verification.json) bind downloaded ZIPs
to GitHub's SHA-256 digests. Copies of all ten native workflow reports and both
Linux timing reports are retained under [evidence/native](evidence/native).
The [manifest](manifest.json) hashes every retained evidence file. The final
[review checks](evidence/review-checks.log) include maturity/exactness mutations,
API compatibility, metadata and artifact contracts. Source archives,
crate archives, ZIPs and durable draw/checkpoint files are retained outside the
checkout at `../alea-candidates/reference-0.1-2026-10-25/148c22c/`; there is no
permanent public artifact download service. CI artifacts also have retention
limits. Back up that directory independently of the working disk.

## Downstream results

The consumer has its own project, manifest and dependency lock and uses only
public core/math/MCMC/runtime APIs. It constructs a native Gaussian regression
and a positive Gamma precision model, runs warmup and four-chain inference,
checks analytic moments and modern diagnostics, then cancels and resumes through
create-new durable output. This is an **agent-executed independent downstream
exercise**; no external-human adoption or feedback is claimed.

Each native lane passed the preset mean/second-moment and diagnostic checks.
The archived-source run observed:

| Model/profile | Finding |
|---|---|
| Regression, covariance/default NUTS | R-hat 1.00020/1.00085; bulk ESS 7,913/7,996 from 8,192 retained draws |
| Regression, explicit gradient/position/jitter profile | R-hat 1.00052/1.00002; bulk ESS 8,608/8,097 |
| Positive Gamma(3,2) precision | Mean 1.52918 against 1.5; second moment about the true mean 0.76056 against variance 0.75; one transform Jacobian |
| Durable recovery | Cancel after 64 draws, resume 192: every resumed draw and final point/RNG/policy checkpoint equals the uninterrupted 256-draw run; existing output preserved |
| Bounded regression SBC | All 32 prior-predictive datasets retained; coefficient bins `[3,9,8,12]` and `[10,8,10,4]`, chi-square 5.25 and 3.0 against preset 16.27 |

Raw simulated data, truth and ranks are included in every workflow report. These
bounded checks cover two small native models and one transform, not arbitrary
posterior geometry, broad model coverage or a universal calibration guarantee.
ESS may exceed the draw count for negatively correlated draws. Consumer timings
are debug-build workflow observations and are not release performance rankings.

## Descriptive performance observations

Five alternating paired runs per dimension compare the candidate with the accepted
baseline on each native Linux host, using the same compiler and independently
resolved matching external dependencies. Builds are excluded. Dimensions 2/16/64
use a fixed identity Gaussian, explicit step/depth/jitter/turning and 5000 draws.
Work counts and draw checksums match between versions on each host. These probes
measure the unchanged frozen kernel, not adaptation, modeling or Enzyme speed.

| Native host | Dimension | Median candidate/baseline time | Observed paired range |
|---|---:|---:|---:|
| Linux x86_64 | 2 | 0.9929 | 0.9753–1.0084 |
| Linux x86_64 | 16 | 0.9909 | 0.9771–0.9914 |
| Linux x86_64 | 64 | 0.9761 | 0.9728–0.9877 |
| Linux aarch64 | 2 | 0.9930 | 0.9918–1.0003 |
| Linux aarch64 | 16 | 0.9960 | 0.9952–1.0008 |
| Linux aarch64 | 64 | 0.9986 | 0.9917–1.0017 |

Lower ratios mean less time. These initial observed bands indicate comparable
cost for this cycle; they are not confidence intervals, hard gates, general
speedup claims or a new nuts-rs/Stan/PosteriorDB comparison. The earlier
[native parity report](../../benchmarks/walltime-parity-2026-10-23/README.md)
continues to own those competitor results. All raw pairs, counts, compiler,
host and source-input hashes are in the two `alea-observations/report.json` files.

## Defects repaired and compatibility

| Observed problem | Final behavior |
|---|---|
| [Reproduced NUTS recovery failure](evidence/checkpoint-regression-before.log): jitter/turning omitted, including during reconstruction | Checkpoint v2 stores and checks both; construction preserves the configured kernel. Exact recovery is tested for default and explicit policies. |
| Extending the warmup enum initially shifted an existing variant discriminant | The new non-exhaustive variant is appended; all four real Rust semver checks pass. |
| Cold consumers/workspace CI lacked independently resolved dependencies | Consumers fetch their own graph, workspace CI fetches the complete lock, then both execute offline. |
| Windows selected the WSL shell shim and checked-out native sources used CRLF | Invoke the active Git Bash explicitly; normalize only the owned generated C build copy, retaining strict patch guards and the untouched upstream source. |
| New stable Rust deprecated the atomic helper used for model identity | Equivalent checked relaxed CAS loop works on MSRV and stable with the same overflow behavior. |
| macOS retained 64 bytes while the measured session owner remained live | Measure complete construction/drop ownership lifetime; retain strict zero-live-bytes and bounded-peak assertions. |
| Benchmark fixture assumed an already existing root target directory | Own repository-local temporary scratch and register cleanup immediately, including setup failures. |
| The complete serial Miri job exceeded its 20-minute budget | [Verified partition](evidence/miri-partition.json) of all 20 existing commands and flags across six independent jobs; retain interpreter logs. No test was removed. |
| Storage Miri still timed out after partitioning | Keep exhaustive native lengths through 129; interpreter lengths through 33 retain every error position, four alignments and four nonfinite values across scalar/chunk/dispatch boundaries. The replacement storage job passes. |
| CI failure logs required authenticated downloads and cancellation could lose buffered output | Flush each retained line before exposing it to the console; regression test reads the log while the child is still running. Preserve escaped bounded public failure annotations. |

Earlier failed attempts, including the [serial Miri timeout](https://github.com/martonaronvarga/alea/actions/runs/40149070533) and [partitioned storage timeout](https://github.com/martonaronvarga/alea/actions/runs/40150548155), remain recorded in CI; final successful runs above are
bound to the repaired code revision. No statistical seeds or acceptance limits
were relaxed to pass a platform. A cold independently fetched consumer and the
same workflow from all actual packages also pass locally.

The intentional pre-alpha binary break is **`ALEACP02`, implementation
`session/2`**. V1 checkpoints reject with `CheckpointError::Format`; restart from
the original model/configuration and retain old draws as data. Draw-stream schema 1
remains readable. Numerical-kernel identity remains `numerics/3`; the shared
owned-record provenance schema remains1. These are separate version axes.
No mid-warmup restart or cross-compiler/backend bitwise replay is promised.
See [session migration and resource limits](../../m15-sessions.md).

## Signed candidate and remaining scope

Source archive SHA-256:
`d89d4a6720db3e0196cd3efffa28e423a673a2926be58b5088b61f4a71ddfa6f`
(1,417 source files). The [digest](evidence/alea-source.tar.gz.sha256) and
[detached signature](evidence/alea-source.tar.gz.sha256.sig) verify in SSH namespace
`alea-source-v1`, identity `git@martonaronvarga.dev`, against the established
maintainer ED25519 fingerprint
`SHA256:0nc+zHneaGaj321FxCvtrTWhtqb4NKYJUPBD3k/lFE0`.
The retained [public key](evidence/source-signing-key.pub) is verification material;
trust its identity independently. No replacement signing key was generated.

No unresolved release-blocking defect was observed within this cycle's declared
scope. External-human feedback, longer field use, crate-wide stable API exclusions
for research exports, and actual per-crate publication/support commitments remain
future S4 work. The versioned compiled-target ABI still awaits two real consumers.
Fisher/ChEES/Atlas/Pathfinder/manifold/isokinetic research does not graduate from
this review; RMHMC remains `ExactnessPendingRmhmc`. A newly found correctness,
data-loss, unsoundness or schema defect resets the affected candidate cycle.
