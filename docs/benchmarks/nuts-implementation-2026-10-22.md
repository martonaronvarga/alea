# Implemented NUTS overhead and geometry improvements

The concrete engineering changes from the [reference review](../nuts-reference-review-2026-10-22.md)
are implemented. Native Gaussian NUTS takes approximately **24–30% less sampling
time** in the paired comparison. The fixed-work tree fixture takes **30–33% less
time**, and dense metric fitting takes **71–77% less time**. These are Alea
before/after measurements on one host, with the earlier uncommitted modeling and
Enzyme work included in both source versions. They are not new Stan/nuts-rs
comparisons or broad posterior ESS/second results.

**What changed.** NUTS now stores immutable phase snapshots in a private,
preallocated pool. Trees retain index handles for endpoints and candidates;
creating a leaf copies only its momentum sum. This removes the three complete
position/gradient/momentum copies previously made at every leaf. Binary carries,
generalized full-tree and cross-subtree turning, biased outer selection and typed
failure behavior remain. The pool resets all scratch ownership at each trajectory,
including after a model or metric panic, while the live point remains separate.

Endpoint energy admission now also supplies the corresponding cached velocity.
Turning reuses it, and leaf pairs omit their duplicate cross checks. Larger trees
still evaluate every cross criterion so a turn cannot conceal a numerical overflow
in another criterion. Finite kinetic energy alone cannot admit a nonfinite velocity.
The nine instrumented dimension/depth cases now use **3, 15 and 511 velocity calls**
for **1, 7 and 255 leapfrog edges**, respectively. Previously those counts were
9, 57 and 2,041. New gradient counts remain one per complete leapfrog edge.
[Executed counts](nuts-implementation-2026-10-22/metric-counts.csv).

Acceptance statistics now sum bounded contributions and divide once, preserving
failed-edge zero contributions and finite energy-cutoff accounting. Selection
probabilities exactly zero or one avoid an RNG draw. These changes alter rounding
and seeded paths; the independent recursive oracle checks the resulting complete
tree, selection, work, stopping and RNG consumption.

Learned geometry has explicit inverse-mass types and enum variants. Diagonal
geometry caches inverse masses and square-root scales. Dense fitting factors the
estimated `G = M^-1` once, instead of forming `M` and factoring it again. For
`G = L L^T`, velocity is `L(L^T p)`, kinetic energy is the half squared norm of
`L^T p`, and momentum sampling solves `L^T p = z`. Energy and velocity share the
intermediate product. Existing public mass types retain their conventions.

Frozen checkpoint tags 0–2 retain their original mass meanings; tags 3–4 explicitly
store inverse geometry. Restore retains exact inverse-mass/factor bits and
recomputes only deterministic diagonal scales. Numerical identity `numerics/2`
rejects older checkpoints, while older draw streams remain readable. End-to-end
tests cover uninterrupted versus restarted HMC/NUTS with both learned diagonal
and dense geometry. [Checkpoint contract](../m15-sessions.md).

The pool reserves `3 * depth + 2` phase slots. It keeps `O(dimension * depth)`
storage and allocation-free transitions. Reserved floating-point payload changes
from `(10 * depth + 34) * dimension` to `(13 * depth + 19) * dimension` entries:
about 8% more at depth 8 and 11% more at depth 10. This excludes metadata,
allocator overhead, geometry and temporary construction storage; it is not an RSS
measurement. The session resource bound accounts for the enlarged workspace.

**Paired timing evidence.** Intel i5-8250U, logical CPU 0, serial alternating
executables, seven rounds per case. No builds or tests ran during measurement.
Every outcome is retained, including the slower pairs. Protocols bind source and
executable hashes before timing; those identities remain unchanged afterward.
Process orchestration uses blocking waits and `perf_counter_ns`, with Rust
`Instant` phase clocks rather than timeout polling.

| Native Gaussian dimension | Median current/baseline phase time | Recorded reduction | Actual gradient-call ratio |
| --- | ---: | ---: | ---: |
| 2 | 0.6968 | 30.3% | 1.00058 |
| 16 | 0.7506 | 24.9% | 0.99983 |
| 64 | 0.7579 | 24.2% | 1.00005 |

The unchanged native harness uses default features, the workspace release/fat-LTO
profile and stable Rust 1.95. Each process runs 32,768 fixed-geometry draws from an
isotropic Gaussian, step 0.3, depth 8 and start 0.1. The phase clock includes
construction and draw collection. All 21 paired phase ratios favor the new binary;
ranges are 0.661–0.818, 0.710–0.758 and 0.749–0.773. Calls differ slightly after RNG
consumption changes, so actual seconds per call are also retained; their median
ratios are 0.6964, 0.7507 and 0.7579. The seven repetitions reuse one declared seed,
so this measures timing variation rather than independent inference uncertainty.
[Protocol](nuts-implementation-2026-10-22/gaussian-protocol.json),
[42 process outcomes](nuts-implementation-2026-10-22/gaussian.jsonl),
[paired results](nuts-implementation-2026-10-22/gaussian-summary.json).

| Workload | Median current/baseline time | Recorded reduction |
| --- | ---: | ---: |
| Fixed-work flat trees, 2D | 0.6709 | 32.9% |
| Fixed-work flat trees, 16D | 0.6949 | 30.5% |
| Fixed-work flat trees, 64D | 0.7042 | 29.6% |
| Dense fitting, 16D | 0.2312 | 76.9% |
| Dense fitting, 64D | 0.2776 | 72.2% |
| Dense fitting, 128D | 0.2873 | 71.3% |

These standalone consumers use identical harness source, Cargo.lock, stable Rust,
Faer features, release fat LTO and one codegen unit in both builds. Each flat
process executes exactly 1,032,192 new gradients over 16,384 complete depth-six
trees. Its improper constant density isolates mechanical work; it supplies no
inference-quality evidence. Dense fitting times include metric allocation from a
fixed previously accumulated covariance. All 42 pairs favor the new binary.
[Protocol](nuts-implementation-2026-10-22/protocol.json),
[84 process outcomes](nuts-implementation-2026-10-22/timings.jsonl),
[results](nuts-implementation-2026-10-22/timing-summary.json).

**Adaptation remains informative rather than automatic.** The bounded comparison
runs five existing 2D targets, covariance diagonal and Fisher diagonal/rank-cap-2
with periods 80/40, four chains, 512 warmup and 2,048 retained draws, common starts,
target acceptance 0.8, depth 8 and history 160. Only period changes within each
Fisher pair. All 100 chains complete; all 204,800 retained draws, repeats and
divergences remain in the evidence. Five observables per method/target give 125
defined diagnostic records. Python independently recomputes their means from the
raw coordinates. Analytic latent-coordinate mean errors are retained; the largest
observed error divided by reported MCSE is 3.62, without a pass/fail threshold.

| Target | Period 40/80 ESS per call, Fisher diagonal | Period 40/80 ESS per call, rank cap 2 | Diagonal retained divergences, 80 → 40 |
| --- | ---: | ---: | ---: |
| Correlated Gaussian | 0.969 | 1.165 | 0 → 0 |
| Banana | 1.457 | 1.040 | 2 → 3 |
| Logistic | 0.935 | 0.935 | 0 → 0 |
| Funnel | 0.988 | 1.066 | 6 → 2 |
| Rotated Gaussian | 0.867 | 1.002 | 0 → 0 |

ESS uses the minimum bulk ESS across all five declared observables divided by
actual retained calls. The shorter period has mixed effects. The rank-cap-2
configuration helps the rotated Gaussian substantially, but rank two is full rank
here and does not validate scaling to large dimensions. Realized ranks, all
divergences, final geometry, steps, bulk/tail ESS, R-hat, means and MCSE are retained
per case. Existing covariance warmup remains the reference and no default is
changed. [Work](nuts-implementation-2026-10-22/adaptation/work.csv),
[diagnostics](nuts-implementation-2026-10-22/adaptation/diagnostics.csv),
[raw draws](nuts-implementation-2026-10-22/adaptation/draws.csv.gz),
[analysis](nuts-implementation-2026-10-22/adaptation-summary.json).

**Validation and reproduction.** A full workspace run passes 455 tests, zero
failures and one ignored test. Final targeted suites additionally cover the
tightened pool capacity, older draw readers and diagonal/dense checkpoint replay.
Workspace Clippy and rustdoc with warnings denied, formatting, API compatibility,
architecture, maturity/exactness checks and 55 metadata mutation tests pass.
Strict Miri executes the recursive oracle, metric-error/panic recovery and inverse
geometry contracts. Combined SIMD/Faer/OpenBLAS units, allocation, sampling and
warmup checks pass. The full Enzyme CI script and final generated downstream
consumer execute successfully, including the 17-model derivative/sampler cases.

The accepted API remains on its original pinned renderer/compiler: 7,079 rendered
entries across eight crates and five API-neutral variants. There are 55 new unique
API entries and no removals; compatibility retains its original reconstructed
base. Existing maturity profiles and all 14 exactness classifications are unchanged.

[Source capsule](nuts-implementation-2026-10-22/baseline-source.tar.gz),
[incremental numerical patch](nuts-implementation-2026-10-22/numerical-changes.patch),
[saved executables](nuts-implementation-2026-10-22/executables.tar.gz),
[validation logs](nuts-implementation-2026-10-22/validation.tar.gz), and
[receipt](nuts-implementation-2026-10-22/receipt.json) preserve the evidence.
The baseline capsule includes the earlier uncommitted work and vendor notices;
applying the numerical patch produces the measured current crate sources.
The standalone [measurement tool](../../tools/nuts-reference/implementation-profile/README.md),
the adjacent Python scripts and locked manifests reproduce the bounded protocols.
Older reference-review and M16 artifacts remain unchanged.

These results establish lower overhead on the measured native workloads. Backend
costs, adaptation efficiency and posterior geometry still determine complete
workflow performance. A broad PosteriorDB or Enzyme sampler speed comparison has
not been rerun, and the existing research-sampler exactness limits remain.
