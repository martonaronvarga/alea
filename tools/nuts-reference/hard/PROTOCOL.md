# Higher-dimensional CPU NUTS v1

Predeclared before runs, October 10, 2026. Four serial chains, seeds 701–704,
alternating coordinates ±0.1 (opposite sign in even seeds). 1024 discarded and
2048 retained iterations per chain; depth 9, cutoff 1000, no thinning/filtering.

| Case | Target | Dimension | Step |
|---|---|---:|---:|
| 0 | AR(1) Gaussian, rho=.95 | 32 | .08 |
| 1 | AR(1) Gaussian, rho=.9 | 128 | .10 |
| 2 | Independent banana pairs, bend=.8 | 32 | .12 |
| 3 | Normal-prior logistic, 512 observations | 32 | .02 |
| 4 | Centered funnel, log-scale width=3 | 10 | .04 |

AR innovations are q[0] and (q[i]-rho*q[i-1])/sqrt(1-rho²).
Banana odd coordinates are independent normals plus .8*(even²-1).
Funnel q[0]~N(0,9), other coordinates conditional N(0,exp(q[0])).
Logistic design X[i,j]=((17*i+13*j+7*i*j)%101)/50-1, y[i]=(37*i%101<50),
zero-based indices, prior N(0,I). Dataset generation is outside timed phases.

Alea (optimized scalar) / Stan / BlackJAX use fixed identity geometry; nuts-rs
uses its native diagonal geometry adaptation with fixed step. Label the latter
separately; not a matched adaptive-policy comparison. Rust samplers share the
analytic target cost; JAX and Stan independently differentiate their densities.
Check Rust scores against finite differences and JAX before timing.

Three serial rotated backend orders on one CPU; release/fat-LTO Rust, same
pinned upstream versions as M5. JIT compilation separate, persistent JIT cache
disabled, block_until_ready for each timed JAX phase. No concurrent builds/tests.
Store all coordinates. Rust/JAX phase times exclude CSV output; Stan native times
include CSV output. Record process time separately. Repeat seeds measure timing
variation, not independent statistical ensembles.
GNU time records each process peak RSS, including language runtime and wrappers;
this is not sampler-only heap usage or a sum of concurrent process memory.

For **every coordinate**, report rank Rhat, bulk ESS and tail ESS, plus mean-square
radius; for non-logistic targets also all generative/whitened coordinates.
All diagnostics must be finite, Rhat<=1.01 and bulk/tail ESS>=400 to pass.
Report minimum ESS, maximum Rhat, retained divergences and depth saturation
(511 integration steps, only a lower-bound proxy for depth-limit terminations).
Show failures rather than changing steps/budgets or excluding chains. Check
non-logistic generative means/second moments against 0/1 with 6 batch MCSE +.05;
MCSE is max of batch-128 and between-chain SE. Logistic has no analytic moments.
ESS/time is not reliable evidence of efficiency when convergence gates fail.

No universal ranking, tuned adaptive-NUTS or GPU-throughput claim. These workloads
extend the previous 2D suite, not all possible posterior geometries.
