# WALNUTS implementation and M7 review

Alea implements within-orbit adaptive leapfrog NUTS (WALNUTS) as an explicit
experimental sampler, following the JMLR construction where it has a proof-level
contract. M7's implementation and evidence work is complete, but its performance
exit gate did not pass, so no default changed.

Primary references are the [JMLR paper](https://www.jmlr.org/papers/v27/25-1452.html)
and the [pinned Walnutpie source](https://github.com/flatironinstitute/walnutpie/tree/5710b0334ba69cf0caa35fc02a2338b61d756953).

## Mathematical contract

For a fixed macro interval `h`, local attempt `j` uses
`m_j = m_min 2^j` leapfrog microsteps of size `h / m_j`. The first attempt whose
Hamiltonian range

```text
max_{0 <= k <= m_j} H(z_k) - min_{0 <= k <= m_j} H(z_k) <= delta
```

is admitted. Alea measures the initial state and every complete microstep, not
only the two macroblock endpoints. Every accepted endpoint lies on the same fixed
macro-time grid, independent of its local refinement count.

The paper's candidate mass contains both `exp(-H)` and the reverse/forward local
selection-probability ratio. Alea exposes deterministic WALNUTS-D and randomized
two-point WALNUTS-R2P. R2P selects the critical level with probability 2/3 and
its next dyadic successor with probability 1/3. Each leaf carries the cumulative
log density and local probability ratios from the origin; endpoint weights are
preserved across both forward and backward tree merges. Zero reverse support
discards the edge as `ReverseSelection`. WALNUTS-D remains the compatibility
default.

Alea discovers the reverse critical level explicitly, including replaying the
selected level rather than assuming exact-arithmetic reversibility at a floating-
point threshold. A bidirectional asymmetric test obtains the exact R2P ratios
1/2 and 2 and verifies that the complete directed log-weight increments cancel.
R2P's possible successor level is included in all shift, exact-f64 and worst-case
work validation.

An optional state-independent uniform macro-interval jitter is drawn once per
orbit and then held fixed over the complete tree, preserving a fixed macro grid
conditional on the auxiliary draw. The paper recommends jitter to break resonance;
the per-orbit form keeps that auxiliary variable especially simple under reversal.

Complete macro endpoints enter an iterative binary-carry tree. Ordinary
multinomial selection is used within a newly built subtree; biased progressive
selection uses `min(1, W_new / W_old)` at the outer doubling. Stable max-shifted
weights retain multiplicity under extreme log-energy gaps. Generalized U-turn
checks use endpoint metric velocities. The fixed metric and macro interval never
adapt during a retained transition.

## API and failure semantics

`Walnuts<T, M>` reuses `PointState`, `PhaseWorkspace` and `EuclideanMetric` but is
not an `Integrator`: local refinement is trajectory construction and changes the
candidate support. `Walnuts::from_hmc` consumes a frozen leapfrog HMC chain after
M4 covariance or experimental M4F warmup without a target evaluation or RNG use.

`WalnutsOptions` validates macro step, tree depth, refinement count, minimum
microsteps, range tolerance, integer shifts, exact f64 microstep conversion and
worst-case counters before sampling. It also validates the local distribution and
relative macro jitter. Work is bounded by depth and refinement
budgets. Concrete target/metric errors remain typed. Numerical failures refine;
exhaustion is a typed transition termination. Model errors and panics never
commit a partial point. Transition records distinguish macro edges, all attempted
microsteps, forward/reverse refinement attempts, selected index, energy range,
tree depth and termination reason.

Library-owned transition storage is `O(dimension × depth)`. Allocation tests
observe no heap allocations for either D or R2P during transitions at depths 1,
5 and 10 and verify linear construction storage. This excludes allocations made
by arbitrary targets.

## Independent references

`tools/walnuts-reference/run.py` builds and executes Walnutpie commit
`5710b0334ba69cf0caa35fc02a2338b61d756953`, repeats the run, and compares a
two-seed Gaussian trace byte for byte. It also verifies the nuts-rs provenance
checkout at `a762aae513bf7bb5ccb27ffc9a938a923fc837ec`.

The checked Walnutpie implementation uses endpoint absolute energy error and its
paper notes that the corresponding proof is expected. Alea deliberately uses the
JMLR paper's stronger full-path Hamiltonian-range condition. Different standard
normal/RNG implementations also preclude a useful exact cross-language Alea
trajectory fixture. Independent scalar leapfrog/range tests therefore validate
Alea's local map; calibrated moments and modern diagnostics validate sampling.

## Statistical and performance evidence

The WALNUTS-D sampling regression uses four chains, 1,024 discarded transitions and 8,192
retained draws on Gaussian, logistic, banana and width-one funnel targets. It
retains repeats and checks analytic or independent BlackJAX moments. The M6
diagnostic regression checks five observables per target with rank/folded R-hat
at most 1.01 and bulk/tail ESS at least 400. All moment and diagnostic gates pass.
R2P separately passes calibrated correlated-Gaussian moments; its two difficult-
target benchmark below supplies its modern diagnostic gate.

The fixed M7 comparison uses the same four seeds for Fisher-warmed multinomial
NUTS and Fisher-warmed WALNUTS, with 1,000 warmup iterations, 4,096 retained
draws, depth 8, and five observables. Time and actual fused target calls include
construction, warmup and retained sampling. WALNUTS uses `4 ×` the frozen HMC
step as macro interval, at most ten dyadic attempts, minimum one microstep and
range tolerance 0.21. No seed, failed transition or diagnostic is removed.

| target/method | max R-hat | min bulk ESS | min tail ESS | total seconds | calls | min bulk ESS/s | min bulk ESS/call |
|---|---:|---:|---:|---:|---:|---:|---:|
| funnel / Fisher NUTS | 1.00242 | 3,800.5 | 5,055.1 | 0.0388 | 106,300 | 98,068 | 0.0358 |
| funnel / Fisher WALNUTS-D | 1.00050 | 5,393.3 | 6,837.6 | 0.0538 | 620,768 | 100,230 | 0.00869 |
| stochastic volatility / Fisher NUTS | 1.00087 | 9,247.9 | 10,557.7 | 2.2115 | 297,344 | 4,182 | 0.0311 |
| stochastic volatility / Fisher WALNUTS-D | 1.00041 | 9,887.3 | 9,839.4 | 4.3156 | 1,672,054 | 2,291 | 0.00591 |

WALNUTS improves the funnel's minimum ESS/s by about 2%, but is about 45% worse
on stochastic volatility and uses 5.6–5.8 times as many target calls. It records
3,094 and 1,795 local-refinement/reverse-support stops respectively. Thus the
declared suite shows neither a suite-wide nor a reproducible production-level
ESS/s improvement. The performance exit gate fails and WALNUTS remains opt-in.

Raw artifact: `docs/benchmarks/m7-fisher-nuts-2026-10-14.csv`, SHA-256
`583e6c8dc39cb01c244d46f238a64ba7c5b623a47bbcebf4dc25432823f65bce`.

The [October 15 R2P completion run](benchmarks/m7-r2p-protocol.md) adds the paper's
randomized profile and ±20% orbit-level jitter without changing the locked result
above. All diagnostics pass. R2P lowers reverse/refinement stops, but uses more
calls; it reaches 69,799 minimum bulk ESS/s versus NUTS's 41,664 on the funnel and
1,398 versus 2,962 on stochastic volatility. The extended evidence therefore also
rejects promotion.

Reproduce with a
new path:

```sh
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml \
  -p alea-runtime --features faer --release --example walnuts_benchmark -- NEW.csv
```

The benchmark is a same-library mechanism comparison, not a universal claim
against Stan, BlackJAX, JAX, nutpie or nuts-rs. Those systems have different
adaptation, compilation and process/JIT cost boundaries.
