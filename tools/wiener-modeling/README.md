# Wiener modeling benchmark

This executable fits complete two-condition reaction-time models through the
public `alea` API. It measures native model construction, dense covariance
warmup, four-chain NUTS, draw export and posterior diagnostics. It accepts local
data rather than downloading or distributing an empirical dataset.

```sh
nix develop .#stable --command cargo build --release \
  --manifest-path tools/wiener-modeling/Cargo.toml --locked
# Generate a reproducible recovery dataset, or supply empirical observations.
tools/wiener-modeling/target/release/alea-wiener-modeling-benchmark \
  generate /tmp/wiener.csv 7
```

The CSV schema is `subject,condition,rt,response`. Subjects are contiguous
zero-based integers; condition is a centered numerical predictor; reaction times
are positive seconds; response is zero or one and denotes the lower or upper
boundary. With accuracy-coded data, the upper boundary means a correct response,
not a physical rightward response.

A configuration contains:

```json
{
  "name": "two-condition-wiener7",
  "data": "/tmp/wiener.csv",
  "family": 7,
  "hierarchy": false,
  "time_only": false,
  "precision": 0.000001,
  "warmup": 600,
  "draws": 1000,
  "seed": 54,
  "threads": 4,
  "seconds": 1200
}
```

Save it as `/tmp/wiener.json`, then run:

```sh
tools/wiener-modeling/target/release/alea-wiener-modeling-benchmark \
  fit /tmp/wiener.json /tmp/wiener-fit
```

The output directory contains every retained draw and a JSON summary. It records
chain failures, cancellations, numerical rejections, divergences, tree-depth
saturation, target calls, E-BFMI, rank-normalized R-hat, bulk/tail ESS and MCSE.
A failed fit is an outcome, not a fast successful execution. First-draw timing
includes warmup and the first retained transition; it does not isolate warmup
exactly. Compilation and data parsing are outside the internal fit timer; an
external process timer can include startup and input/output.

## Model

All priors below are on unconstrained Normal coordinates. Physical parameters
are deterministic functions of them, so no additional Jacobian is applied.
Writing `L` for the logistic function and `m` for the minimum observed RT:

- `log(alpha) ~ Normal(log(1.5), 0.5)`.
- `tau = m L(q_tau)`, with `q_tau ~ Normal(0, 1)`.
- `q_beta ~ Normal(0, 0.8)`.
- Population drift has a `Normal(0, 1.5)` prior; the centered condition contrast
  has a `Normal(0, 1)` prior.
- Wiener5/7 use `log(s_delta) ~ Normal(log(0.5), 0.7)`.
- Full Wiener7 uses `s_beta = 0.8 L(q_start)`, with
  `q_start ~ Normal(log(1/3), 1)`, and
  `beta = s_beta/2 + (1 - s_beta) L(q_beta)`.
- Wiener7 uses `s_tau = 0.8 m L(q_time)`. Its prior logit centers the time width
  at 0.08 seconds, clipped to 5–80% of its allowed range, with Normal SD one.
- A hierarchy adds `log(sigma_subject) ~ Normal(log(0.6), 0.5)` and independent
  standard-Normal subject effects, with noncentered subject drifts.

`time_only` retains variable nondecision time while fixing trial drift and start
variability to zero. Otherwise family 4, 5 or 7 selects its corresponding density.
The data-dependent support bound on `tau` is explicit; these are benchmark models,
not recommended scientific priors for every experiment.

`evaluate CONFIG POSITIONS OUTPUT` evaluates complete native targets at supplied
raw-coordinate vectors. `predict CONFIG POSITIONS OUTPUT` simulates replicated
choices and reaction times at the original experimental design. The coordinate
order is visible in `draws.csv`. These modes support independent target checks,
precision sensitivity and posterior predictive model assessment.

For reproducible before/after measurements, compile the same executable source
against both pinned Alea revisions, pin dependency versions and CPU affinity,
alternate execution order, retain all outcomes, and compare posterior accuracy
before interpreting ESS per second. Keep raw datasets, run artifacts and research
reports outside Git and release packages.

The optional `target_acceptance` and `initial_step` settings default to 0.9 and
0.05. `step_search: true` enables bounded reasonable-step search with an initial
step supplied by the caller; `maximum_step` defaults to 0.1 for this explicit
search profile. A conservative follow-up can use initial step 0.001 and target
acceptance 0.95. Record such follow-ups separately from the original run, rather
than replacing failed or diagnostically inadequate outcomes.
