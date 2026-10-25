# M14 executed validation and distribution cost

Executed 2026-10-20 on an Intel Core i5-8250U, Linux x86_64, lock-pinned
Rust 1.95.0. Four chains run sequentially with caller-owned seeded SmallRngs.
The [predeclared protocol](../m14-validation-protocol.md) fixes seeds, tolerances,
work budgets and exclusions. Raw [downstream output](m14/downstream.txt),
[example output](m14/examples.txt) and [cost CSV](m14/distribution-cost.csv) are retained.

## Correctness and calibration

All normalized-law value/normalization/support/tail and derivative gates pass.
Scalar RNG means and variances pass their declared bands. MVN, Dirichlet,
categorical/multinomial, LKJ and finite-mixture tests cover their declared measures,
normalizers, scores and generators. Every successful law value/score operation
and all three compiled model targets have zero allocation-count regressions.
Simplex and correlation transforms have explicit one-Jacobian pullback checks.

A separate, independently implemented external normalized law passes reusable
value/measure and optional-score conformance. Empty layout blocks have a regression
for metadata projection, alongside the dimension-reducing simplex check.

The three external model constructions agree with independent normal/logistic
formulas and a separately implemented spectral Wiener density. The elementary
normal/logistic primals pass executed Enzyme reverse gradients with the matching
nightly-2026-06-23 compiler/plugin. Wiener remains opaque with an analytic local
VJP; no derivative through a foreign backend is claimed.

The conjugate normal SBC run uses 64 datasets and 512 independent analytic
posterior draws: bins [14, 22, 15, 13], chi-square 3.125 (<16.3).
The external normal-regression example separately runs covariance/NUTS inference
and diagnostics. Hierarchical SBC uses 32 datasets per model, four chains,
512 single-step covariance warmup transitions and 512 retained NUTS draws per
chain/dataset. All completed states, including repeats, enter the ranks.

| Model | Unconstrained coordinate | Four rank bins | Chi-square | Retained divergences across all datasets |
|---|---|---|---:|---:|
| hierarchical logistic | mu | [7, 9, 6, 10] | 1.25 | 0 |
| hierarchical logistic | log(tau) | [13, 3, 9, 7] | 6.5 | 0 |
| hierarchical logistic | z[0] | [6, 9, 12, 5] | 3.75 | 0 |
| hierarchical logistic | z[1] | [8, 7, 11, 6] | 1.75 | 0 |
| hierarchical DDM | mu | [6, 10, 7, 9] | 1.25 | 0 |
| hierarchical DDM | log(tau) | [7, 11, 4, 10] | 3.75 | 0 |
| hierarchical DDM | z[0] | [6, 14, 7, 5] | 6.25 | 0 |
| hierarchical DDM | z[1] | [7, 8, 10, 7] | 0.75 | 0 |

All eight hierarchical gates pass the unchanged 16.3 threshold, with zero retained
divergences. The two fixed hierarchical diagnostic examples have maximum R-hat
1.00435 and minimum bulk ESS 2,037. Their posterior predictive boundary frequencies
agree with averaged conditional probabilities within the declared 0.05 band.
The standalone three-model executable also runs 2,048 posterior predictions per
model and prints named geometry groups and unconstrained diagnostics.

These are small, weak-data regression suites. They do not establish general
calibration, convergence, efficiency, arbitrary conditioning, or equivalence to
an external engine's adaptation policy. The first five-step warmup trial failed
explicitly on exponential-transform overflow before retained sampling. The
completed protocol uses single-step warmup; the failure was not discarded as
an ordinary rejected draw or silently retried. DDM production prediction samples
boundary events; it does not expose an RT sampler. The SBC-only independent RT
generator uses bounded spectral-tail inversion at fixed alpha/start/nondecision
parameters, not an Euler Brownian approximation.

## Local distribution costs

`m14_profile` executes 200,000 normalized log-probability plus parameter-score
pairs per row, with construction outside timing and outputs consumed through
black_box. This is one local timing observation per row, without uncertainty,
CPU isolation, or comparison claims. Vector cases have dimension two or three;
Binomial has 12 trials and Poisson rate 1.2. It is not a scaling or sampler benchmark.

| Law | ns per value + parameter score |
|---|---:|
| normal | 31.775 |
| student_t | 317.220 |
| lognormal | 79.784 |
| exponential | 35.268 |
| gamma | 161.565 |
| beta | 360.200 |
| bernoulli | 4.655 |
| binomial | 153.963 |
| poisson | 53.054 |
| mvn2 | 78.005 |
| dirichlet3 | 295.441 |
| categorical | 19.116 |
| multinomial | 248.138 |
| lkj2 | 169.375 |
| mixture | 440.268 |

Reproduce correctness and calibration with the commands in the
[modeling guide](../m14-modeling.md). Reproduce the measured loop with:

```console
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml --locked --release -p alea-distributions --example m14_profile
nix develop .#autodiff --command cargo test --manifest-path crates/Cargo.toml --locked --release -p alea-autodiff --features std-autodiff --test m14_models
```

Full workspace tests/doctests, Clippy and rustdoc are separate gates. M13 snapshots,
compatibility adjudication, downstream laws and compiler-negative contracts remain
enforced. No research sampler is promoted and no S3/S4 stability is declared.

M15 preserves the measured downstream lock verbatim in
`m14/downstream-conformance.lock`. Subsequent session and NUTS performance work
adds the runtime-to-core contract, pinned Pulp/normal-generation dependencies and
their transitive features. `m14/reviewed-performance-lock.json` records the exact
new packages, checksums and dependency edges; `tools/test_m14.py` rejects other
resolution changes. The measured workspace lock is also retained with its exact
original digest under `m14/measured-sources/`.

Native Enzyme export subsequently adds primal-source metadata to model assembly
and the normal/Bernoulli-logit helpers. Their actual M14 source bytes are restored
from the sealed M16 archive, checked against the original M14 digests, and retained
under `m14/measured-sources/`. Current modeling and generated-derivative tests
validate the later implementation separately. These historical artifact checks
are not a new calibration or timing measurement of the current implementation.
