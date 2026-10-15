# Baseline hardening after M4

Scope: correctness of existing implementations, not new sampler features. All
pre-existing worktree changes and historical test inventories remain preserved.
No commits or pushes are part of this increment.

## Reproduced low-rank failure

The spectral metric applies `G = S (I + U diag(lambda-1) U^T) S` and samples
momentum with the corresponding inverse square-root correction. Previously,
construction accepted any positive finite eigenvalue with a finite reciprocal.

For d=r=1, S=U=1 and lambda=1e-20, `lambda-1` rounds to -1 in f64. Consequently
`velocity([1])` returned zero despite the nominal positive eigenvalue. Conversely,
at lambda=1e40 the momentum correction rounded to -1, producing zero momentum.
A slightly non-unit basis, `U=1+4e-11`, passed the old 1e-10 Gram tolerance but
could produce a negative inverse mass when lambda=1e-12.

The new constructor regression failed before the fix. Construction now returns
`MetricError::IllConditionedCorrection` when

```
kappa * (rank * max_gram_error + 16 * f64::EPSILON * (dimension + rank)) > 1e-6
kappa = max(1, max(lambda)) / min(1, min(lambda)).
```

Nonfinite estimates also fail. The existing absolute Gram check remains. This is
a conservative explicit precision policy for the correction representation—not a
rigorous floating-point certificate. It deliberately rejects some mathematically
valid SPD matrices rather than silently changing their eigenvalues. A diagonal
representation remains available for diagonal problems. Future broader support
requires a numerically stable factor/action representation and independent oracles,
not merely loosening this threshold.

The guard runs only during construction, introduces no hot-path allocation or new
dependency, and leaves existing operation kernels unchanged. Scalar coordinate
scales and input vectors may still overflow; HMC's finite-result checks remain
necessary. Fisher fitting propagates invalid geometry as a typed error; it does
not publish a partially adapted chain.

## Regression coverage

`alea-math/tests/low_rank.rs` now includes:

- The rejected cancellation and near-orthogonality cases above.
- A deterministic grid of ranks 0–3 and eigenvalues 1e-3–1e3 with distinct coordinate
  scales. Sampling a standard-normal vector must preserve `K(p)=z^T z/2` within
  the stated relative tolerance; determinant conventions are checked independently.
- Shape failures for velocity, momentum sampling and kinetic-energy scratch, with
  unchanged sentinel outputs. Existing dense-oracle and zero-allocation checks stay.

`alea-mcmc/tests/fisher.rs` now compares every foreground window with independently
centered batch scatter. Finite observations causing arithmetic overflow are injected
at initial background reset and later swap boundaries; subsequent valid updates
must agree with the oracle. A low-rank ring test compares its fit with an explicit
last-six-observation batch after wrapping more than twice.

That audit also found `FisherMetricAdapter::fit_low_rank` accepted a scales vector
of a different dimension when its length happened to divide the flat history.
It now rejects the mismatch before delegating to the math fitter.

## CI coverage repair

The workflow previously selected `alea-math/faer`. Cargo feature unification made
the math solver available, but did not activate `cfg(feature="faer")` in MCMC.
Therefore sampler fitting methods and their tests were absent from that CI row.
The workflow now selects `alea-mcmc/faer` (and forwards SIMD/OpenBLAS through the
sampler too), retaining the distribution-specific feature rows.

Reproduce with:

```sh
nix develop .#stable --command bash tools/ci/check.sh
nix develop .#stable --command bash tools/ci/check.sh alea-mcmc/faer
nix develop .#blas --command bash tools/ci/check.sh alea-mcmc/simd,alea-distributions/simd,alea-mcmc/faer,alea-mcmc/openblas
```

The low-rank metric and non-Faer Fisher regressions also run under the existing
strict-provenance/symbolic-alignment Miri script. Native cold fitting is not claimed
to run under Miri. No remote CI execution is claimed.

Local numerical-hardening verification completed successfully:

- Full workspace gate in stable/default, stable/MCMC-Faer, and combined
  SIMD/Faer/OpenBLAS configurations: checks, tests, Clippy with warnings denied,
  rustdoc with warnings denied, formatting and architecture/migration guards.
- Focused native Faer regressions: seven low-rank and eight Fisher tests.
- Targeted strict Miri: five low-rank and four Fisher tests. This was not a rerun
  of the full Miri script or the Enzyme checks.

## Optional capability failure audit — 2026-10-01

HVP and batch wrappers retain their existing allocation-free, caller-owned scratch
contract. This audit adds failure-injection tests, not panic-catching machinery or
a second transactional cache API. It explicitly documents that target panics
propagate, consistently with scalar `evaluate`.

Five additional capability tests cover:

- HVP shape/non-finite input rejection before backend invocation/output writes.
- Original typed backend errors, partial writes, panic propagation and successful
  reuse of HVP scratch after unwinding.
- Independent model errors, invalid log densities and incomplete gradients in the
  same batch, with healthy lanes on both sides, under both physical layouts.
- Batch preflight failures preserving both gradient and lane-status sentinels.
- Reuse after a panic, stale status clearing, and validation of later completed
  lanes even when an earlier lane was left pending.

Crucial distinction: `Complete(Ok(_))` is a backend-written status, not a validated
cache token. If the backend panics, the wrapper never reaches validation; no lane
output may be consumed as validated. If it returns normally, an ordinary failed
lane does not invalidate other validated lanes. A missing status returns a protocol
error after validating the completed lanes. Callers that need a coherent persistent
cache must evaluate into separate scratch and publish only validated results.

These tests exercise only deliberate, recoverable test-double panics under unwind;
they do not claim recovery from aborting panics or arbitrary corrupted backend state.

Verification: all seven capability tests passed natively and under strict-provenance/
symbolic-alignment Miri. The complete stable workspace gate was rerun after the
audit and passed, including tests, Clippy, rustdoc and migration checks. The Faer
and combined-feature gates above precede this test/documentation-only follow-up.

## What remains open

Independent upstream fit/update traces, extreme-scale covariance/fitting accuracy,
complete-macrostep failure-path tests, and predeclared statistical benchmarking remain required
before M4F promotion. Existing warmup/sample tolerances and seeds were not changed
in this increment. No Adam, continuous adaptation, NUTS or engine extraction was added.
