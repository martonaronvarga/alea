# M14 validation protocol

Declared before executing the new M14 suites (2026-10-20). Existing Gaussian,
Wiener and sampler fixtures, seeds and tolerances are preserved.

The normalized law API is separate from unnormalized targets. A support zero is
negative infinity; invalid parameters, NaNs, shape errors and backend failures
are errors. Scores require differentiable interior observations. Evaluation is
allocation-free; vector RNG and cold construction may allocate. RNG rejection
work is bounded and failure is explicit; unsupported budgets never fall back to
an approximate draw.

Scalar gates: fixed analytic log-probability cases; numerical normalization at
absolute error 3e-4 (continuous midpoint integration with sufficiently wide finite
bounds, accounting for omitted tails), discrete sums within 1e-10, finite-difference
parameter/event scores within 2e-5 relative/absolute, tails and support boundaries,
invalid parameters and transactional shape checks. RNG tests use 40,000 fixed-seed
draws and a six-standard-error mean band (or a declared absolute 0.03 band for
bounded vector/correlation checks). Allocation counters require zero evaluation
and score allocations over 128 repeated calls. Timings are observations, not a
promotion/performance claim.

Tier 2 gates: independent small-dimensional multivariate-Normal values and scores,
Dirichlet and multinomial normalization cases, categorical frequencies, the
2x2 LKJ normalizer and second moment E[rho^2]=1/(2*eta+1), and higher-dimensional
row support/score checks. Mixtures must preserve normalization and gradients at
large common log offsets, zero weights and disjoint/zero support.

Model gates: compare three downstream-built models (normal regression,
noncentered hierarchical Bernoulli-logistic and hierarchical Wiener4 drift)
against separately written constrained formulas and finite differences through
the transform boundary. Verify shared-parameter gradient accumulation, exactly
one Jacobian per declared parameter, rejected foreign handles, immutable named
coordinate/block metadata, missing VJP writes, failures and no per-evaluation
allocation. Enzyme agreement applies to compiler-compatible elementary normal
and logistic expressions; Wiener is explicitly opaque and uses its independently
validated analytic VJP, never differentiation through the series/FFI boundary.

Inference gates: use the reference HMC/NUTS components, finite covariance warmup,
fixed logical seeds, completed retained draws and diagnostic projections. Prediction
is run separately on constrained posterior/prior draws. DDM predicts the upper
boundary using the analytic first-passage probability; it does not simulate a
finite-step Brownian path or claim an exact RT RNG. Generated quantities may
report deterministic boundary probabilities.

Calibration gates: the conjugate normal vertical uses 64 simulated datasets and
512 posterior draws, with four equal rank bins and Pearson chi-square <16.3.
The hierarchical logistic and Wiener verticals use prior-predictive exchangeability
checks and independently evaluated probability/gradient fixtures; these do not
constitute posterior SBC. A completed M14 claim additionally requires posterior
SBC for both hierarchical verticals under an explicit executed protocol; no
checkbox may silently substitute prior-predictive checks for that requirement.

Hierarchical posterior SBC extension (declared before its execution): 32 datasets
per hierarchical model, two groups with two observations each, mu~Normal(0,0.7),
log(tau)~Normal(-0.7,0.3), z[group]~Normal(0,1). Bernoulli events are generated
from the implied logits. Wiener data use alpha=1.2, tau=0.15, beta=0.5 and the
implied drift. An independent spectral first-passage tail integral with at most
4096 terms, a 64-step bisection and bounded bracket expansion generates RTs;
it does not reuse Alea's Wiener density implementation. Four chains use 512
covariance warmup transitions and 512 retained NUTS draws each. Seeds start at
14100 (logistic) and 14200 (DDM), plus dataset/chain identifiers. For every one of
the four unconstrained coordinates, four equal rank bins must have chi-square
<16.3; ranks use all completed retained states including repeats. Report
all failures and aggregate divergence counts. These bounded weak-data experiments
are regression gates, not general SBC or mixing guarantees. The examples separately
report rank R-hat/bulk ESS, and predictive frequencies are compared with the
independent generative law. No seed selection or failed-dataset deletion.

Warmup policy clarification: initial step .08 and one HMC leapfrog step per
warmup transition, then frozen NUTS depth 7. The first implementation trial used
five-step warmup and failed explicitly on exponential-transform overflow during
step-size adaptation, before producing retained draws. It is preserved as a
limitation of that policy; backend errors were not converted into rejections or
retried. Single-step warmup avoids repeated unstable integration during that
search, with unchanged data, seeds, retained kernel and statistical thresholds.

Additional RNG spread gate (declared before execution): scalar sample variances
from 40,000 seed-14002 draws must be within 8% of the analytic variance. The
Student-t case has seven degrees of freedom, so its fourth moment exists. This
supplements the six-standard-error mean checks and detects a wrong-scale RNG.
