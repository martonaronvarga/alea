# Executed nuts-rs Fisher fit reference

This runner executes `LowRankMassMatrixStrategy::compute_update` from nuts-rs
commit `a762aae513bf7bb5ccb27ffc9a938a923fc837ec` (0.18.3). It appends an observation
probe and a public re-export to a fresh temporary checkout; it does not replace
the fitting, rescaling, SVD, QR, eigensolve, or filtering implementation.
The checkout is removed when the runner exits. No upstream source is vendored.

## Reproduction

From Alea's repository root, with network access for Git/Cargo:

```sh
nix develop .#stable --command python3 tools/fisher-reference/upstream/run.py --check
```

`--write` explicitly regenerates `crates/alea-math/tests/fixtures/fisher-nuts-rs.csv`.
`--lock` resolves a new **reference-only** dependency lockfile and should be used
only for deliberate dependency updates. Normal runs use `cargo --locked`.
The reference disables upstream default features (parallel and relaxed SIMD).
Build products are cached under ignored `target/fisher-upstream/`.

The standalone manifest is a template, not an Alea workspace member: the runner
copies it and the lockfile next to the temporary upstream checkout. Its 74-package
dependency resolution does not modify `crates/Cargo.lock`. The repository's stable
Nix shell pins Rust/Python; the adjacent Cargo lock pins reference dependencies.

## Scope and conventions

### Executed update traces

`run.py --updates --check` executes the pinned strategy's `update_estimators`,
`switch`, and `update` methods and compares the installed transformation. The
probe enables the upstream Gaussian test target outside `cfg(test)`; numerical
implementations remain unchanged. It supplies deterministic collector records,
including an explicit `is_good` false every fifth event, and switches windows at
events 6 and 12. It does not execute the NUTS collector's trajectory-selection rule.

The 28 rows in `alea-mcmc/tests/fixtures/fisher-nuts-updates.csv` record history,
scales, ranks, transformation versions, every inverse-mass basis action and mass
log determinant. The Rust test reconstructs the accepted histories and checks
Alea fits using the emitted upstream scales. This is executed strategy/update
coverage, not equivalence of whole warmup: Alea deliberately retains rejection
repeats and uses different ridge, window, initialization and step-size policies.

### Fit-prefix traces

Eighteen fits use prefixes of 4/8/12 paired observations in dimensions 2/8/16,
ridge 0.001, and cutoffs 1.01/2.0. Inputs are deterministic algebraic sequences,
not stochastic posterior draws. These are **fit-prefix traces**, not an execution
of the NUTS collector, adaptation scheduler, or sampling chain.

The probe emits original positions/scores, upstream diagonal scales, retained
rank, `G = S [I + U diag(lambda-1) U^T] S`, and `log det M = -log det G`.
It only reconstructs these observable matrix actions from the fit result. Alea's
test consumes the emitted scales and checks every basis-vector action, rank,
and mass log determinant against the independent Decimal oracle below, while
characterizing upstream errors separately. It does not compare eigenvector
signs/order, which are not unique. The comparison tolerance is 1e-8 times the geometric diagonal scale
for matrix entries and 1e-8 times `(1 + abs(logdet))` for the determinant, selected
before execution. Regeneration compares floating outputs numerically at 1e-9
times `(1 + abs(reference))`, with exact input/rank checks, not byte identity.

## Conditioning finding and independent adjudication

The first direct comparison failed the preset tolerance. `adjudicate.py` therefore
computes a second, 80-digit dense oracle from **inputs only**, using Denman-Beavers
iteration for `sqrt(C F) F^-1`, followed by Jacobi eigenvalue filtering. It checks
the square-root residual, symmetry, `G F G = C`, orthonormality and spectral
reconstruction before emitting `fisher-nuts-rs-decimal.csv`. No Faer or upstream
computed matrix is used. Tests also compare all six 2D cases against the existing
independent 120-digit closed-form oracle, and check known roots and pivoted inverses.

```sh
nix develop .#stable --command python3 tools/fisher-reference/upstream/adjudicate.py --check
nix develop .#stable --command python3 tools/fisher-reference/upstream/test_adjudicate.py
```

On the recorded reference execution, the maximum entry error relative to the
geometric diagonal scale for `d16_n8_t1.01` is about 2.72e-8 upstream versus
4.47e-9 for Alea. Upstream `d16_n8_t2` has about 1.63e-8 error. Both upstream cases
miss the original 1e-8 threshold and are explicitly marked in the Rust test;
**Alea must still satisfy the unchanged threshold on every case**. All other
upstream matrices satisfy that threshold against Decimal. Direct pairwise errors
can exceed it even when both individual errors pass (e.g. `d16_n12_t1.01`).
This is conditioning-sensitive reference arithmetic, not evidence of complete
upstream equivalence. The test must not be simplified to trusting upstream alone.

The fixture was executed on x86_64 Linux using the stable Nix shell (Rust 1.95.0,
Python 3.13.12). Pinned artifact SHA-256 hashes:

```text
636020d8aa6a0eacd5c15dbceb4a90b1e1712a1f6141f8054d36b607ffe07e64  Cargo.lock
e274b099d82a1fbb75164dc0203d9aeb839fe7b89be91c6428b8eafe672625a4  probe.rs
2f7734d774f06f01484bf2e505b2ab40219f0a8fbc9a068d1d0597a082e455d8  fisher-nuts-rs.csv
61cb05f77a9e18bf96c21129f4063a3f53d001eedffeb19b17487400ec83608f  fisher-nuts-rs-decimal.csv
```

### Deliberate policy differences

- Upstream's low-rank fit computes unregularized diagonal fourth-root variance
  ratios. Alea's warmup diagonal estimator adds a ridge and bounds scales. This
  reference supplies upstream scales directly to Alea's low-rank fitter; it does
  **not** assert that the diagonal adapters are equivalent.
- Upstream estimates an affine translation too. Alea currently fits geometry,
  not a translated target. Both fits center positions and scores; translation
  outputs are outside this comparison.
- Upstream uses SVD plus pivoted QR for the joint subspace. Alea uses twice-MGS
  with an explicit dependence threshold and shared scatter normalization.
- Upstream filters by cutoff without Alea's output rank cap or conditioning
  rejection policy. The comparison permits full ambient rank; Alea's separate
  tests cover caps, extreme scales and typed failures.
- Upstream's draw collector filters some trajectory-index/divergence cases;
  Alea observes actual retained states, including rejection repeats. Initialization,
  window switching, step-size adaptation and freeze schedules are not compared.

No whole-warmup, convergence, or performance equivalence follows from these fits.
The independent Decimal oracle remains a separate check against shared algebra or
backend mistakes. Normal Rust CI consumes the fixture without downloading upstream;
the networked regeneration command is an explicit maintenance check.

Upstream provenance: [fit implementation](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/transform/adapt/low_rank.rs),
[collector](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/transform/adapt/diagonal.rs).
Upstream nuts-rs is MIT licensed; the runner fetches the original license with
the pinned checkout. The local probe is Alea reference tooling, not upstream code.
