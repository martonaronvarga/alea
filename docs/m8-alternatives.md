# M8: ChEES-HMC, GHMC, and multiscale alternatives

M8 adds four opt-in algorithm families without changing the covariance-warmed HMC
or NUTS defaults. The common rules remain: target/cache changes commit atomically,
warmup is separate from retained sampling, backend errors remain typed, and hot
transitions reuse construction-time storage.

## ChEES-HMC

`CheesEnsembleAdapter` implements Algorithm 1 of Hoffman, Radul, and Sountsov's
[ChEES-HMC paper](https://proceedings.mlr.press/v130/hoffman21a.html). For each
lockstep ensemble iteration it uses

```text
ChEES = 1/4 mean[(||q' - mean(q')||² - ||q - mean(q)||²)²]
g_m   = t (||q'_m - mean(q')||² - ||q_m - mean(q)||²)
          <q'_m - mean(q'), v'_m>
```

and averages `g_m` with Metropolis-probability weights. Step size uses dual
averaging against the harmonic-mean acceptance target 0.651. Log trajectory time
uses Adam with `alpha=0.025`, `beta1=0`, and `beta2=0.95`, followed by the paper's
moving-average freeze. Initial and proposal means are separate; the path endpoint
is observed before the Metropolis choice.

Warmup uses one base-two Halton jitter shared by every chain and
`ceil(t / step_size)` complete macrosteps. `alea-runtime::chees::warmup` executes
chains sequentially or in a caller-owned Rayon pool, but always reduces results in
stable input order. The scheduler-determinism regression is bitwise. Successful
warmup consumes the adapter, installs `FrozenCheesParameters`, and retained
sampling uniformly randomizes the integration count without carrying adaptation
state.

The turnkey shared runtime is restricted to `IdentityMetric`, matching the paper's
identity/prewhitened-coordinate derivation. A nonidentity metric must provide a
coherently whitened `CheesObservation` explicitly; raw positions with transformed
velocities are not silently mixed. `CheesCriterion::DifficultDirection` is a
separate, explicitly labeled projection experiment.

## Persistent and delayed-rejection GHMC

`Ghmc` owns momentum as explicit chain state. With persistence `c`, refreshment is

```text
p_refresh = c p + sqrt(1 - c²) eta,  eta ~ N(0, M).
```

The proposal involution is momentum-flip-after-integration. The implementation
then applies the conventional unconditional flip: acceptance retains integrated
momentum, while rejection retains `-p_refresh`. Backend errors publish neither
position nor momentum.

`DrGhmc` implements bounded two-stage deterministic delayed rejection. The second
stage uses a geometrically smaller step and the same configured leapfrog count. Its
acceptance probability includes the reverse first-stage rejection term,

```text
alpha2 = min(1, exp(log pi(y2) - log pi(x))
                * (1 - alpha1(y2)) / (1 - alpha1(x))).
```

The reverse proposal is an actual bounded ghost map, not an assumed symmetric
scalar. Tests pin the scalar ratio, pairwise balance, configured actual/ghost work,
and the persistent rejection convention against the formulation in the
[DR-G-HMC paper and author implementation](https://proceedings.mlr.press/v258/turok25a.html).

## Bounded ATLAS experiment

`probe_local_curvature` is intentionally a curvature/local-step experiment rather
than a partial sampler advertised as ATLAS. It applies the potential Hessian twice
per power iteration, so the estimate tracks the spectral magnitude of an
indefinite Hessian. Restarts and iterations determine an exact HVP budget, and the
safety-scaled local step is clamped to validated bounds. The reported curvature is
a lower spectral estimate, never a certified stability upper bound. A full ATLAS
transition would additionally require its extended-state proposal corrections; see
the [ATLAS paper](https://arxiv.org/abs/2410.21587).

## Isokinetic MAMS and approximate MCLMC

`Mams` is separate from the canonical Hamiltonian path. It uses a unit velocity,
analytic isokinetic score kicks, and accumulates the method-specific kinetic/Jacobian
correction. For score direction `e`, kick scale `delta`, and incoming unit velocity
`u`, the stable correction is evaluated as

```text
(d - 1) * [delta + log1p((e·u - 1) * (1 - exp(-2 delta)) / 2)].
```

Acceptance uses

```text
log alpha = min(0, log pi(q') - log pi(q) - correction),
```

not canonical energy. Reversal/correction cancellation and standard-Gaussian
first/second moments are regression tested against the equations in the
[MAMS paper](https://arxiv.org/abs/2503.01707).

`ApproximateMclmc` omits that Metropolis correction and carries a persistent unit
direction with angular refreshment. Construction requires
`ApproximateTarget::acknowledge_bias()`, and every transition reports the same
marker plus the correction it deliberately discarded. It cannot be confused with
an exact-target kernel through a canonical acceptance shortcut.

## Executed comparison

The reproducible `m8_benchmark` example ran on 2026-10-15 in release mode with
four chains, 500 warmup iterations, 2,048 retained draws, and five observables on
correlated Gaussian, logistic, banana, and funnel targets. Timings and target-call
counts include warmup. The table reports the minimum bulk ESS across observables;
all maximum rank/folded R-hats were below 1.004.

| Target | Method | Minimum bulk ESS | Target calls | Minimum ESS/call |
|---|---:|---:|---:|---:|
| Correlated Gaussian | fixed HMC | 3,630 | 101,980 | 0.0356 |
|  | NUTS | 3,584 | 61,672 | 0.0581 |
|  | ChEES | 4,221 | 19,690 | 0.2143 |
|  | difficult-direction ChEES | 4,193 | 19,613 | 0.2138 |
| Logistic | fixed HMC | 2,049 | 101,980 | 0.0201 |
|  | NUTS | 4,394 | 55,618 | 0.0790 |
|  | ChEES | 3,662 | 14,874 | 0.2462 |
|  | difficult-direction ChEES | 3,698 | 14,954 | 0.2473 |
| Banana | fixed HMC | 2,743 | 101,467 | 0.0270 |
|  | NUTS | 2,536 | 67,311 | 0.0377 |
|  | ChEES | 2,575 | 19,800 | 0.1301 |
|  | difficult-direction ChEES | 2,216 | 15,719 | 0.1410 |
| Funnel | fixed HMC | 3,413 | 101,696 | 0.0336 |
|  | NUTS | 2,222 | 76,124 | 0.0292 |
|  | ChEES | 2,407 | 20,229 | 0.1190 |
|  | difficult-direction ChEES | 1,690 | 15,805 | 0.1069 |

This reproduces the paper's qualitative ChEES advantage in ESS per gradient on
the declared small suite. The difficult-direction variant is mixed and is not a
default. These analytic targets complete in milliseconds, so their ESS/second
numbers are useful only as a local implementation comparison; they are not a
cross-hardware or accelerator claim. Re-run with:

```sh
nix develop -c cargo run --release -p alea-runtime \
  --example m8_benchmark -- /new/path/m8.csv
```

The full workspace test suite, warning-denied Clippy, scheduler replay, hot-path
zero-allocation checks, staged delayed-rejection tests, and MAMS moment gate pass.
M8 remains opt-in; this milestone does not change Alea's default sampler.
