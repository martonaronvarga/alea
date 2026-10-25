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
An admitted outer expansion uses min(1,W_new/W_old). The weights are exp(H0-H). Relative log energies in [-500,500] are exponentiated once at the leaf and accumulated directly; the same value supplies ordinary acceptance. This range leaves enough exponent headroom for every addressable tree. Extreme energies switch to a maximum log weight and scaled sum, so huge common log offsets do not erase multiplicity. Merge and selection share a max-shifted calculation;
ordinary ratios use the scaled masses directly, while extreme log gaps retain
log-space ratios to avoid premature underflow. No final independent Metropolis
test is added. Consumed merge inputs exchange owned snapshots rather than copying
their coordinate buffers; their old endpoints are no longer observable.

The generalized turn criterion requires positive dot products between summed
momentum rho and both endpoint velocities. Both cross-subtree checks are also
required: left rho plus the first right momentum, and right rho plus the last
left momentum. Signs are evaluated after positive scaling to avoid dot-product
overflow; an unrepresentable momentum sum is a numerical divergence. A zero dot
product terminates expansion. An ordinary dot product may supply the sign only
when conservative rounding and underflow bounds certify agreement with the
ordered normalized calculation. Uncertain cases retain that original calculation.
Hölder bounds, sum |v_i*rho_i| <= d*||v||_infinity*||rho||_infinity, permit conservative certification without materializing a temporary sum or scanning its maximum. The three enhanced checks share a six-dot SIMD pass when every sign can be certified; any uncertain sign returns to eager individual checks. Admitted immutable endpoints cache their velocity and momentum scales; the first kick and drift
read the preceding phase directly, avoiding copies of its position, score and
momentum. The certificates preserve the ordered stopping decisions. Linear weight accumulation and kinetic-energy reduction may change last-bit candidate thresholds. This is the enhanced Stan convention, not a claim
of identical stopping decisions to BlackJAX's different endpoint treatment.

Reference: [Stan 2.38 multinomial NUTS](https://github.com/stan-dev/stan/blob/v2.38.0/src/stan/mcmc/hmc/nuts/base_nuts.hpp).
Independent executed distributions also come from BlackJAX1.5/JAX0.9.2 and
[pinned nuts-rs](https://github.com/pymc-devs/nuts-rs/tree/a762aae513bf7bb5ccb27ffc9a938a923fc837ec).

## Trajectory step jitter

`NutsOptions::with_step_size_jitter(relative)` independently draws one uniform
step multiplier in `[1-relative, 1+relative)` per complete trajectory. The step
stays fixed throughout that trajectory; after warmup this is a fixed mixture of
stationary NUTS kernels. The default width is zero and consumes no additional RNG.
`NutsTransition::step_size` and warmup observations report the actual step;
`NutsOptions::step_size()` and the final warmup report retain the base step.
Floating-point extreme steps clamp to positive finite endpoints. Static-HMC
conversion uses the base step and drops NUTS jitter.

## Position turning for constant geometry

`NutsOptions::with_position_turning(true)` explicitly chooses the original NUTS
endpoint-displacement criterion. Both `(q_right-q_left) dot p_left` and its right
endpoint counterpart must be positive. The two cross-subtree intervals are
checked as well. Generalized momentum-sum turning remains the constructor default.

For inverse mass G=L L^T, write q=L z and p=L^-T r. Then kinetic energy is
||r||²/2 and delta(q) dot p = delta(z) dot r. This implements the position test in
mass-whitened coordinates without storing transformed positions. Using velocity
in place of canonical momentum in the physical-coordinate dot would give the
wrong transformation. Position turning avoids maintaining subtree momentum sums;
its trajectories and random-number consumption can differ from the generalized
criterion. See [Hoffman and Gelman, section5](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)
and the pinned nuts-rs transformed Hamiltonian referenced above. Static-HMC
conversion retains the integrator and base step and drops the NUTS turning choice.

## Runtime vectorization and numerical arithmetic

Stable builds use Pulp CPU dispatch for coordinate updates, finite checks,
turning certificates, and high-dimensional identity/inverse-diagonal energy.
Small coordinate vectors keep scalar kernels. Kicks and drifts explicitly use
separate multiplication and addition, preserving the established integrator map.
Finite checks retain the first invalid coordinate; errors and panics never publish
partial state. `PointState::try_update_owned` exchanges candidate coordinate
storage only after a complete checked evaluation.

The energy kernels sum nonnegative half-scaled squared terms in independent SIMD
accumulators. Scaling before summation avoids overflow of an unhalved norm when
the final kinetic energy is representable; shorter accumulation chains also
reduce the usual roundoff bound. Energy rounding can differ from older builds.
For 16 or more coordinates, Euclidean phase starts use Rand's Ziggurat standard
normal generator instead of discarding the second polar variate on every draw.
Smaller starts retain the polar generator. Both sample independent standard
normal coordinates; the larger-vector RNG stream changes. Cross-build bitwise
replay is not promised. RWMH's checkpointed cached polar variate is unchanged.

The optional nightly `alea-mcmc/simd` feature remains supported for portable SIMD
kernels and the existing math backend selection. Ordered uncertain turning
fallbacks still divide by positive scales, including subnormal scales; no
approximate reciprocal or unchecked endpoint is introduced.

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

`NutsWarmup::new(target, position, nuts_options, warmup_options)?.run(rng)` adapts
with NUTS transitions, covariance windows and dual averaging, then consumes all
adaptation state. This remains the reference default.

Calling `.with_gradient_diagonal()?` before `.run(rng)` explicitly selects paired
position/score diagonal adaptation instead, overriding a requested dense metric.
It learns `G_i = sqrt(M2_position_i) / sqrt(M2_score_i)` from centered moments,
including repeated live states. The initial diagonal is
`1 / clamp(abs(score_i), 1, 1e20)`; fitted coordinates are bounded to
`[1e-20, 1e20]`. Invalid or zero-scatter coordinate estimates retain their previous
values. Foreground/background windows grow after the first 30% of warmup; geometry
freezes at 85%. The first fit restarts the bounded step search; freezing performs
a final fixed-geometry search and dual-averaging restart, so the retained step
does not average obsolete geometry. Late metric-adaptation dual averaging uses the symmetric tree statistic
`2*exp(-abs(delta_H))/(1+exp(-abs(delta_H)))` and caps raw steps at pi (within the
caller’s search bounds) before averaging. The final fixed-geometry phase targets
ordinary mean tree acceptance, like covariance warmup. Every adaptive choice ends before
retention; zero warmup preserves the supplied step and identity geometry.

`.with_gradient_final_restart(false)` explicitly keeps continuous dual averaging
and symmetric late acceptance through the final phase, as nuts-rs does. This
switch applies only to the gradient diagonal option. A closer reference workflow
combines that choice with `NutsOptions::with_step_size_jitter(0.1)`; covariance
and zero-jitter defaults remain available for comparison.

This conventional diagonal option is informed by the
[pinned nuts-rs adaptation](https://github.com/pymc-devs/nuts-rs/tree/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src).
Its repeated-state observations, conservative initial score floor, enhanced turning
and absence of retained step jitter differ from that implementation. It is not a
claim of identical whole-workflow trajectories. The private
[`parity` probe](../tools/nuts-reference/parity/README.md) separates retained kernel
overhead from adaptive work; native PosteriorDB measurements also report posterior
diagnostics and actual gradient calls.

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
