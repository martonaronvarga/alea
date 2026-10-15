# M5: fixed-geometry multinomial NUTS

`Nuts<T, M, I>` owns one coherent live point, the existing `PhaseWorkspace`,
chronological tree endpoints and reusable binary-carry slots. Target, Euclidean
metric and sealed palindromic integrator are statically dispatched. No separate
integrator, dynamic callback, target reevaluation at edge starts, or universal
sampler abstraction was introduced.

## Mathematical conventions

Momentum is sampled from N(0,M); velocity is Gp with G=M^-1. Each tree leaf is a
complete integrator macrostep, never an internal stage. Each outer expansion
chooses forward or backward with probability one half and attempts 2^depth edges.
The origin participates in the initial tree. Binary carries build the same
internal subtrees as depth-first recursion while storing O(d*max_depth) elements.

Within a subtree, proposals are selected with probability W_new/(W_old+W_new).
An admitted outer expansion uses min(1,W_new/W_old). The weights are exp(H0-H),
represented by a maximum log weight and a scaled sum, so huge common log offsets
do not erase multiplicity. Merge and selection share a max-shifted calculation;
ordinary ratios use the scaled masses directly, while extreme log gaps retain
log-space ratios to avoid premature underflow. No final independent Metropolis
test is added. Consumed merge inputs exchange owned snapshots rather than copying
their coordinate buffers; their old endpoints are no longer observable.

The generalized turn criterion requires positive dot products between summed
momentum rho and both endpoint velocities. Both cross-subtree checks are also
required: left rho plus the first right momentum, and right rho plus the last
left momentum. Signs are evaluated after positive scaling to avoid dot-product
overflow; an unrepresentable momentum sum is a numerical divergence. A zero dot
product terminates expansion. This is the enhanced Stan convention, not a claim
of identical stopping decisions to BlackJAX's different endpoint treatment.

Reference: [Stan 2.38 multinomial NUTS](https://github.com/stan-dev/stan/blob/v2.38.0/src/stan/mcmc/hmc/nuts/base_nuts.hpp).
Independent executed distributions also come from BlackJAX1.5/JAX0.9.2 and
[pinned nuts-rs](https://github.com/pymc-devs/nuts-rs/tree/a762aae513bf7bb5ccb27ffc9a938a923fc837ec).

## Optional nightly SIMD

Enable `alea-mcmc/simd` in the lock-pinned `nix develop` environment to use
four-lane portable SIMD in turning checks, including identity/diagonal geometry.
The flag also forwards the existing math SIMD feature; Faer/OpenBLAS metric
selection does not disable SIMD turning. Default scalar builds support stable.

SIMD performs finite checks, momentum sums, scale maxima, divisions and products.
Dot products retain scalar summation order rather than horizontal reassociation;
subnormal scales still use division, not potentially overflowing reciprocals.
Models smaller than one full four-lane chunk use the scalar reference. No new
public trait, unsafe code, allocation or change to stopping/error policy is added.
See the [paired SIMD report](benchmarks/nuts-simd-2026-10-13.md) for the measured
scope, numerical edge-case tests and reproducible comparison commands.

## Termination and failure contracts

NUTS uses the one-sided cutoff H-H0>limit, unlike fixed HMC's symmetric absolute
cutoff. A turning/divergent *new subtree* is discarded. Earlier admitted candidates
remain eligible. An outer merged-tree turn is checked after admitting that valid
new subtree. Concrete target/metric errors and panics commit nothing to the live
cache, even if earlier doublings succeeded. RNG/target-side effects are not rolled
back. Dirty scratch is reset at the next transition.

Statistics distinguish U-turn, numerical divergence, and maximum depth. They
include attempted macrosteps (also discarded/failing edges), completed admitted
doublings, admitted state count, selected-edge ordinal, initial and selected energy,
largest finite energy increase, and mean acceptance over all attempted edges.
Finite cutoff divergences retain their min(1,exp(-delta)) contribution; an
unrepresentable edge contributes zero. `accepted` means a non-origin candidate,
not a second acceptance test or a promise that coordinates differ bitwise.

Depth validates count arithmetic, not a practical execution-time budget. Allocation
exhaustion follows Rust's allocator; aggregate vector sizes are checked for address
overflow. Construction allocates O(d*depth); transitions allocate nothing within
the sampler. Arbitrary user targets and metrics may allocate internally.

## Adaptation integration and limits

`Nuts::from_hmc(frozen_chain, depth)` consumes the frozen output of reference
`HmcWarmup` or explicitly experimental `FisherHmcWarmup`, preserving the cache,
metric, step and integrator without evaluation or RNG consumption. Its cutoff
magnitude is preserved but takes NUTS's one-sided meaning. This is fixed-HMC
warmup followed by NUTS, **not** a new NUTS-specific warmup controller. Fisher
remains experimental; no M4F promotion is implied. The existing `MarkovChain`
boundary supports streaming NUTS without a runtime-specific implementation.

## Verification

The stable workspace gate (tests, check, Clippy and rustdoc) passed, as did the
focused combined SIMD/Faer/OpenBLAS NUTS suites. Strict Miri passed the bounded
tree oracle, failure/panic contracts, warmup conversion and initial-momentum
recovery regression. These checks do not constitute a full Enzyme feature-matrix run.

- 768 deterministic trajectories agree with an independently recursive scalar
  oracle on candidates, positions, edge counts, depths, termination, acceptance
  and random consumption. Strict-Miri uses a bounded subset.
- Failure injection at each internal stage of leapfrog/OMF/BCSS checks bit-exact
  live-cache preservation, panic recovery and evaluation counts.
- Gaussian stationary moments cover identity/diagonal/dense/low-rank metrics and
  all three integrators. Allocation tests force all 1023 edges through depth 10;
  construction storage grows linearly, not exponentially.
- Executed Stan/BlackJAX/nuts-rs comparisons use four targets and four chains,
  8192 retained draws per chain. All 80 rank-Rhat/bulk/tail diagnostic gates and
  120 moment gates pass. Largest Rhat 1.001134; minimum bulk ESS 6739.96 and
  tail ESS 7984.63. No retained divergences in the declared workloads.

Protocol, environment pins and reproduction: `tools/nuts-reference/README.md`.
Offline summary fixture: `crates/alea-mcmc/tests/fixtures/nuts-sampling.csv`.
Local raw files: `target/m5-{alea,stan,blackjax,nuts-rs}.csv`; complete diagnostic
results and raw-input hashes: `target/m5-gates.json`. These are distributional
correctness gates, not identical RNG/trajectories, performance comparisons,
general convergence proofs, or evidence for a wider Neal funnel.
