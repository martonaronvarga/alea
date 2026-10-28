# Wiener numerical methods

Alea evaluates a joint density for reaction time and boundary choice. Reflecting
the lower response changes the drift sign and the relative starting point.
The formulas below use `q` for distance to the selected boundary, `a` for
boundary separation, `t` for decision time, and `u = t / a²`.

## Series tails and scores

For the spectral Gaussian moment

\[
T_p(K,c)=\sum_{n=K+1}^{\infty}n^p e^{-cn^2},\qquad c=\pi^2u/2,
\]

the consecutive ratio is

\[
r_n=\left(1+\frac1n\right)^p e^{-c(2n+1)}.
\]

It decreases with `n`. If `r_(K+1) < 1`, the first omitted term divided by
`1 - r_(K+1)` bounds the tail. A second bound is the integral from `K` to
infinity when `K² >= p / (2c)`. Alea uses the smaller valid candidate. Gaussian
moments through order four suffice for density, time and start derivatives,
including endpoint-aware sine bounds. The identity
`|sin(nπq)| <= n |sin(πq)|` gives a second candidate for the density and time
error near a boundary.

The image tails are two positive lattices with spacing two. For the sequence
`x^p exp(-x² / (2u))`, the ratio at `x` is

\[
\left(1+\frac2x\right)^p e^{-2(x+1)/u}.
\]

It also decreases. An integral candidate is available past `x² >= p u`;
because the lattice includes its first point, this candidate is the first
term plus half the integral from that point. Bounds for the absolute full
image derivatives use the moments

\[
B_0=M_1,\qquad
B_u\leq \frac{M_3}{2u^2}+\frac{3M_1}{2u},\qquad
B_q\leq M_0+\frac{M_2}{u}.
\]

The common Gaussian normalizer is applied separately. The time bound includes
the derivative of `u^(-3/2)`. Log scaling avoids forming an underflowed density.

For finite density `f_hat > B_0`, truncation gives

\[
|\log f-\log\widehat f|\leq-\log(1-B_0/\widehat f),
\]

\[
\left|\frac{d_j}{f}-\frac{\widehat d_j}{\widehat f}\right|
\leq\frac{B_j+|\widehat d_j/\widehat f|B_0}{\widehat f-B_0}.
\]

These bounds motivate the optional common-count evaluator. An error bound for
the density alone is insufficient for the scores. None of these exact-arithmetic
inequalities encloses floating-point rounding or errors in elementary functions.

## Integrating the starting-point interval

With Gaussian drift standard deviation `s`, let `lambda = 1 + s² t`. Each
Wiener5 image has the form

\[
\frac{a}{\sqrt{2\pi t^3\lambda}}(q+2k)
\exp(-Aq^2+B_kq+C_k),
\]

\[
A=\frac{a^2}{2t\lambda},\quad
B_k=\frac{av}{\lambda}-\frac{2a^2k}{t},\quad
C_k=-\frac{v^2t}{2\lambda}-\frac{2a^2k^2}{t}.
\]

Thus the coefficient of `q²` is strictly negative. Completing the square gives
an error-function primitive. Alea evaluates definite moments instead of
subtracting two large primitive values. Write `q = m + h x`,
`x in [-1/2, 1/2]`. The four moments of `exp(-A h² x² + (B_k-2Am)h x)`
through order three supply the integrated density and all seven parameter
derivatives.

For moderate intervals, the evaluator integrates a power series and carries a
Cauchy remainder into each moment's error. The coefficients satisfy
`(n+1)c_(n+1) = B c_n - 2A c_(n-1)`, avoiding repeated polynomial products.
It sums 16, 48 or 80 coefficients according to the exponent radius. On any
circle `|z|=rho>1/2`, the omitted series is bounded uniformly by

\[
\frac{\exp(A\rho^2+|B|\rho)}{1-1/(2\rho)}
\left(\frac1{2\rho}\right)^N.
\]

The contour radius is chosen near the minimum of this bound, within `[1,16]`.
Compensated accumulation combines the integrated monomials. Otherwise it
uses a scaled normal interval probability followed by Gaussian moment
recurrences. Poorly conditioned recurrences return to cubature.

Density and derivative tail envelopes hold uniformly over the entire start
interval. Their construction bounds the drift prefactor at the interval
endpoints, then bounds both omitted image lattices at their nearest possible
distances. Parameter derivatives use polynomial envelopes on the same interval.
The moment remainder is added before converting density and derivative errors
to log-density and score errors.

For zero drift variability, the spectral branch has the simpler uniform average

\[
\operatorname{Im}\left[
 e^{cm}\frac{\sinh(ch/2)}{ch/2}\right],\qquad c=av+i n\pi.
\]

A regular Taylor evaluation handles `ch` near zero. Differentiating the same
expression supplies the scores. A bound on `exp(avq)` over the interval,
together with spectral Gaussian moments, controls the omitted integrated terms.

When these analytic start averages are available throughout an unclipped time
interval, adaptive inference can integrate only over time. The automatic cost
policy selects this reduction for localized starting-point integrands: the
smallest interval curvature `a² h² / (2 t (1+s²t))` must be at least `0.01`.
The coefficient recurrence makes moderate intervals cheaper, while nearly
constant intervals retain the other paths. This policy does not promise minimum latency. Any failed moment or
quadrature check returns to the ordinary two-dimensional evaluator. Fixed
Gauss–Legendre options retain the requested product rule.

## Time variability and cumulative probabilities

Boundary CDFs are subdistributions. The boundary tail is `P(T > t, B = b)`;
it is not `1 - F_b(t)`. The spectral tail integrates each exponentially decaying
density term, dividing by its time decay rate. Its omitted tail is bounded
by the corresponding density envelope divided by the smallest omitted rate.
At short times, integrated image terms use normal probabilities and Mills
ratios. These avoid underflow in normal tails.

With time width `h > 0` and no drift or start variability,

\[
p_7(y)=\frac{F_b(y-\tau_0)-F_b(y-\tau_0-h)}{h}.
\]

Alea chooses a well-separated CDF or boundary-tail difference, tightening the
inner budget before subtraction. Narrow differences retain density cubature.
The fused path differentiates the integrated image/spectral formulas. The width
score is the lower endpoint density divided by interval mass, minus `1/h`.
Independent subtraction checks and tighter inner budgets retain cubature if
probability or width-score cancellation cannot meet the requested budget.
Almost constant start intervals use the bounded moment expansion when available;
time widths below `1e-5` times decision time skip CDF subtraction. These are
measured cost policies, not mathematical accuracy thresholds.

The checked CDF-score API can return a numerical-conditioning error for very
small opposite-boundary probabilities near an absorbing endpoint. The continuous
reference suite retains these cases explicitly. A finite CDF is not a promise
that its log score can be computed at the same tolerance.

The time interval starts at `tau_0`, rather than being centered there. Its
right width score at zero is therefore half the location score. A symmetric
start interval has a zero width score at zero. Direct differentiation under
the fixed unit cube avoids unstable endpoint expressions containing `1/h`.

## Adaptive integration of density and scores

The fused Wiener7 integrator subtracts a common log scale before exponentiating.
It also scales each derivative by the larger of one and its midpoint score
magnitude. Individual absolute and relative cubature tolerances then control
small derivatives independently of large ones. A vector L2 stopping test alone
can hide an inaccurate starting-width derivative behind a large time derivative.

If the midpoint overestimates a score, one bounded retry recalibrates the scales
from the first integral. This matters when the time interval nearly meets zero
support. On clipped time support the width score is exactly `-1 / s_tau`;
adaptive integration uses that identity instead of a redundant derivative
integral. Fixed quadrature retains its selected-rule derivative.
The integral and its error estimates are transformed back together. The final
log-density and score-ratio checks remain authoritative; normalization does not
weaken them. If the scaled integrand overflows or the estimates cannot meet the
budget, the bounded locally scaled fallback remains available. These are
quadrature error estimates, not interval certificates.

## What the error reports establish

Wiener4/5 reports describe analytic series truncation in real arithmetic.
Wiener7 integrates pointwise series envelopes and reports their estimated
integrals separately from estimated cubature errors. A numerical integral of
an upper-bound function is still an estimate unless its integration error is
rigorously enclosed. Gauss rule discrepancies are also estimates.

The independent Python fixtures use Arb/Acb ball arithmetic. They add analytic
remainders to enclosed finite series or continuous integrals and increase
precision until the resulting log-density and score balls are sufficiently
narrow. They certify their own enclosures. Tests compare Rust results with a
separate floating-point allowance. The continuous Wiener7 oracle also integrates
nested time/start variability.
For clipped support it removes a time-zero slab and bounds its full density
and derivative contribution with Gaussian envelopes. The monotonicity condition
`delta / a² <= q_min² / 8` controls every `t^(-r) exp(-a² q² / (2t))` factor
through `r = 7/2`. Complex square roots check analyticity on integration boxes.
A failed reference work or precision budget is explicit.

The pointwise audit compares a supplied binary64 result, treated as an exact
real vector, with these enclosures. Its error bounds include truncation,
quadrature, rounding and elementary-function error together. Certification
uses the lower bound on the permitted relative tolerance, so uncertainty in the
reference cannot produce a false pass. A rejected result is separated from an
unresolved enclosure. This is a certificate for the supplied input/result pair,
not a global proof of all Rust floating-point executions.

## Measured costs

A CPU-pinned comparison on an Intel Core i5-8250U covered 676 parameter
configurations, scalar and SIMD builds, density and fused-score evaluation,
and 32-observation batch paths. Three alternating process pairs retained all
16,656 outcomes. The comparison used source revisions `49b7527` and `7d1f488`,
with the same compiler within each build pair. Every density and fused output
in this grid passed pointwise ball-enclosure comparisons at
`1e-9 * (1 + |reference|)` in both build modes. The old build failed 33 fused
cases per mode: 32 zero-width scores and one rare-tail score. These results
describe that finite workload; they are not a promise for every parameter value.

Median scalar old/new timing ratios were:

| Evaluation | Density | Density and scores |
|---|---:|---:|
| Wiener4/5 | about 1× | about 1× |
| Wiener7, ordinary image start averaging | 1.76× | 1.93× |
| Wiener7, localized image start averaging | 82.6× | 87.9× |
| Wiener7, spectral start averaging | 23.6× | 25.1× |
| Wiener7, time averaging | 8.68× | 2.24× |
| Wiener7, clipped time averaging | 112× | 97.5× |
| Wiener7, localized joint variability | 12.9× | 13.1× |
| Wiener7, broad joint variability | 0.89× | 0.31× |
| Wiener7, clipped joint variability | 0.89× | 0.67× |
| Wiener7, long-time joint variability | 0.90× | 0.32× |

A ratio below one means slower evaluation. General cubature now integrates
direct width derivatives and checks estimated errors in every score, increasing
its cost. Tiny variability widths were roughly 5–12% slower in this grid; rare
tails had mixed results. Batch evaluation was essentially unchanged.

The old implementation returned an incorrect time-width score at zero width.
Consequently, fused start-only timings compare historical execution costs,
rather than two implementations of equally correct complete gradients. SIMD
builds showed the same broad pattern. The largest gains come from eliminating
integration dimensions; scalar calls do not automatically gain SIMD throughput.

## Complete model fits

Kernel timings do not establish inference performance. A complete-fit comparison
used the public `alea` modeling and session APIs on the HDDM Cavanagh dataset
(source commit `19feaa9849619931d306094b48d9803ef9c8d0bd`), retaining all 2,019
unstimulated trials from 14 participants and both responses. Single-participant
Wiener7 fits used all 148 unstimulated trials from participant zero.

The baseline was `49b7527`; the revised integration was `9001222`, with the richer
hierarchical workload in `c4e8392`. Four parallel chains ran on an Intel i5-8250U.
Each pair used identical priors, observations, seeds, tolerances and inference
settings. The study retained 22 fit attempts, including original settings and
separately recorded follow-ups; 12 passed the numerical and diagnostic gate.
Failed fits have no quality-adjusted efficiency score.

| Paired workload | Baseline | Revised | Inference diagnostics |
|---|---:|---:|---|
| Subject-drift Wiener4 hierarchy, 20 parameters | 49.0 s | 49.8 s | Both pass |
| Subject-drift Wiener5 hierarchy, 21 parameters | 44.7 s | 45.8 s | Both pass |
| Subject drift, boundary and time hierarchy, 50 parameters | 126.8 s | 128.3 s | Both pass |
| Time-only Wiener7, 6 parameters | One chain failed | 42.9 s | Revised passes |
| Full Wiener7, 8 parameters | 227.7 s | Warmup failed | Revised fails |
| Simulated Wiener4 hierarchy, 14 parameters | 9.4 s | 9.2 s | Both pass |
| Simulated full Wiener7, 8 parameters | 635.5 s | Warmup failed | Revised fails |

These are descriptive single-seed measurements, not an external-engine ranking.
Budgets differ between workloads: paired settings and all outcomes are retained
in the benchmark artifacts. Qualified Wiener4/5 pairs have identical draws;
small timing differences do not demonstrate acceleration. The 50-parameter
hierarchy has maximum R-hat 1.003 and minimum bulk ESS 1,551 over 6,000 draws.
The revised full Wiener7 path still encounters numerical limits during warmup,
including after covariance-window changes. Its kernel gains therefore do not
justify a general modeling-performance claim.

Numerical convergence also differs from model adequacy. Several observed RT
quantiles fall outside replicated intervals even in the richer empirical
hierarchy. These simplified models do not reproduce the original EEG/DBS
analysis. Inspect predictions and model assumptions before interpreting effects.
The [benchmark executable](../tools/wiener-modeling/README.md) documents the
models, priors, data schema, diagnostics and repeatable commands. Raw datasets
and research run artifacts are kept outside source releases.

## References

- [Navarro and Fuss (2009), density expansions and truncation](https://papers.djnavarro.net/2009_firstpassage.pdf).
- [Blurton, Kesselmeier and Gondan (2012), cumulative first-passage distributions](https://doi.org/10.1016/j.jmp.2012.09.002).
- [Gondan, Blurton and Kesselmeier (2014), improved series evaluation](https://doi.org/10.1016/j.jmp.2014.05.002).
- [Hartmann and Klauer (2021), partial derivatives and error bounds](https://doi.org/10.1016/j.jmp.2021.102550).
- [Python-FLINT documentation](https://fredrikj.net/python-flint/arb.html), including rigorous real and complex ball arithmetic.

## Directional refinement on difficult variability intervals

The locally scaled fallback compares GL15×GL15 with GL7×GL7 and both mixed
rules GL7×GL15 and GL15×GL7. For component `j`, let `H_j` be the high-order
integral and `M_d,j` the mixed rule lowered in direction `d`. The region error
estimate is the larger of the full low-order discrepancy and
`|H_j-M_0,j| + |H_j-M_1,j|`. Refinement chooses the direction with the largest
component discrepancy normalized by `max(|H_j|, H_0)`. This follows derivative
curvature as well as density curvature; the slope at the density peak alone
can choose the wrong direction. These discrepancies remain error estimates.

The work limit counts all four rules. Eight previously failing valid inputs,
including clipped time support and drift standard deviation near 349, are
regression cases at tolerance `1e-6`. An independent composite GL25 calculation
uses geometric decision-time panels to resolve the near-zero boundary layer.
Passing these cases does not establish accuracy at tighter tolerances.

## Large drift variability in the image representation

Let `lambda = 1 + s²t`, `r = aq - vt`, `u = t/a²`, and let `S(u,w)` be
the image sum scaled by its central Gaussian. Combining the central Gaussian
with the analytically marginalized drift gives

\[
\log p=\log a-\tfrac12\log(2\pi)-\tfrac32\log t
-\tfrac12\log\lambda-\frac{r^2}{2t\lambda}+\log S.
\]

Evaluating the two exponentials separately loses precision when drift variance
is large. Their time derivatives can cancel even more severely. With
`E = r²/(2t lambda)`, `h = r/(t lambda)`, `U = S_u/S` and `W = S_w/S`,
the combined derivatives in effective upper-boundary coordinates are

\[
\begin{aligned}
\ell_a &= a^{-1}-qh-2uU/a,\\
\ell_\tau &= 3/(2t)+s^2/(2\lambda)-vh
-E(t^{-1}+s^2/\lambda)-U/a^2,\\
\ell_w &= ah+W,\\
\ell_v &= r/\lambda,\\
\ell_s &= st(2E-1)/\lambda.
\end{aligned}
\]

Reflection applies the usual starting-point and drift signs. The implementation
uses this form for image evaluation when `lambda > 2`, keeping the existing
endpoint rearrangement near either boundary. Twenty-four independent Arb
fixtures cover both responses, times from `1e-8` to `0.03`, and drift standard
deviations from 350 to one million.

The independent Wiener7 oracle retains the same cancellation in its envelopes.
For image distance `d = q + 2k`,

\[
-\frac{a^2d^2}{2t}+\frac{a^2q^2s^2}{2\lambda}
=-\frac{a^2(d^2-q^2)}{2t}-\frac{a^2q^2}{2t\lambda}.
\]

For every noncentral image, `d² >= q²`. The central image is bounded separately.
This avoids an envelope growing as `exp(a²q²s²/2)`. On a clipped time slab,
`t^(-r) exp(-b/(t(1+s²t)))` is increasing if `t(1+s²t) <= b/r`.
The oracle uses the sufficient condition
`t(1+s²t)/a² <= q_min²/8`, including all derivative powers through `r = 7/2`.
These are analytic envelopes; Arb rounds the reference calculations outward.

## Time derivatives as boundary fluxes

Write `T = rt - tau`, `h = s_tau`, `L = max(0, T-h)`, and let `g(t)`
already average over trial starting points and Gaussian drift. Then

\[
I=\frac1h\int_L^T g(t)\,dt,\qquad
\partial_\tau I=\frac{g(L)-g(T)}h,
\]

where `g(0)=0` because the starting interval remains strictly inside the
absorbing boundaries. If `z_max = min(1,T/h)`, integration by parts also gives

\[
\partial_h I=\frac{z_{\max}g(L)-I}{h}.
\]

Consequently, the clipped-width score is exactly `-1/h`. These identities avoid
integrating time derivatives whose large positive and negative contributions
nearly cancel. The adaptive scaled path evaluates the two start-averaged faces
once at one hundredth of the outer tolerance, and propagates their estimated
uncertainty along with the density estimate into the final score-ratio checks.
The face evaluations have their own bounded work; any failure remains explicit.
Fixed quadrature continues to differentiate its chosen finite rule directly.
The eight difficult-input regressions also pass at tolerance `1e-9` with this
reduction. This estimated quadrature contract still excludes floating-point error.

## Paired short-time CDF images near the opposite boundary

Let `J(x)` denote the integrated Gaussian image, `G = phi(x/sqrt(t)) exp(-v²t/2)`,
and `D = exp(vx) Phi(-v sqrt(t)-x/sqrt(t)) - exp(-vx) Phi(v sqrt(t)-x/sqrt(t))`.
For an opposite-boundary distance `w`, pair images about `x = a(2k+1)` with
`h = aw`. Taylor's theorem gives

\[
J(x-h)-J(x+h)=-2hJ'(x)+R,\qquad
|R|\le h^3\sup_{[x-h,x+h]}|J^{(3)}|/3.
\]

The identities

\[
J'=vD-2G/\sqrt t,\quad
J''=v^2J+2xG/t^{3/2},\quad
J'_{v}=D+vxJ,\quad
J'_{t}=(1-x^2/t)G/t^{3/2}
\]

provide density and parameter derivatives without subtracting nearby values.
The separation derivative applies the same symmetric remainder to `x J'(x)`;
its third derivative is `3J^(3)+xJ^(4)`. The starting-point derivative is a
symmetric sum of `J'`, with error at most `a h² sup|J^(3)|`. Time and drift
remainders follow by differentiating `J_t` and `J_v` three times in `x`.
The implementation bounds these derivatives using the left-endpoint values
of positive decreasing `J` and `G`, and the largest polynomial coefficients
on the interval. It adds these remainders to the omitted-image tails before
checking log-density and score errors. Floating-point error remains separate.

All 24 independent continuous CDF-score reference cases now return finite
accurate results, including the two opposite-boundary cases formerly rejected.


The narrow-start branch uses the same integrated Gaussian moments as wider
intervals. It integrates the exponential polynomial exactly and propagates its
uniform remainder through every parameter derivative before accepting a result.
This supplies a controlled small-width evaluation without estimating a fourth
derivative of the complete likelihood or subtracting endpoints of size `1/h`.
Eight terms suffice in the small exponent-radius regime; failure to meet the
error budget still returns to numerical integration. This replaces the earlier
width-only exclusion, which discarded an accurate analytic result solely because
its interval was narrow.

## Cost of componentwise truncation planning

The optional W4/5 truncation-controlled evaluator first finds a common term
count satisfying every requested log-density and score bound in each series.
It chooses the smaller predicted fused cost. The prediction is an affine model
`C(K) = b + mK`, calibrated with CPU-pinned finite-sum measurements at normalized
time 0.15, starts 0.2, 0.3, 0.5 and 0.7, and term counts one through six.
Eleven batches of 10,000 evaluations supplied median costs on an i5-8250U.
Least-squares fits, normalized to three units per spectral term and rounded,
give image/spectral `(b,m)` pairs `(5,6)/(9,3)` for the stable scalar build and
`(5,5)/(7,3)` for the nightly SIMD build. Point evaluation itself uses scalar
finite sums in both builds; vector batch kernels do not use this planner.

Positive slopes make the search pruning valid. After obtaining a certified
candidate of predicted cost `C`, the other branch needs only counts
`K <= floor((C-b_other)/m_other)`, bounded by the 128-term work limit. Counts
with greater predicted cost cannot win. A regression compares this pruned
selection with unpruned searches across both representations.

This replaces the former two-image-terms-to-one-harmonic work heuristic in the
optional evaluator. It minimizes the calibrated estimate among candidates
meeting the truncation bounds, not wall time on every processor. Subnormal
exponentials and fixed-count specializations produce deviations from the affine
model, particularly above six terms. The ordinary fast evaluator retains its
precision convention; a certified truncation request remains an explicit API
choice with additional planning cost.

## Reducing clipped intervals after an ordinary integration failure

Analytic start averaging also applies at every strictly positive decision time
inside a clipped time interval. The rare-path evaluator can therefore integrate
`g(T-hz)` over `0 <= z <= min(1,T/h)` in one dimension, using the same locally
scaled GL7/GL15 refinement. It tries this reduction when the start antiderivative
is available at the upper time face. Every quadrature node must independently
meet the inner moment and series budgets; an unavailable or inaccurate inner
result returns to the two-dimensional fallback. Inner tolerances use at most
five percent of the outer relative tolerance.

No analytic continuation through time zero is needed: the quadrature nodes are
interior, and the boundary density is zero for a strictly interior start interval.
Boundary-flux identities supply the time scores. This is especially helpful for
small start widths far from the selected response: direct integration otherwise
subtracts large positive and negative start-width derivative contributions.
The first failure found by the full Atlasz replay is retained as a regression:
`a=0.7, tau=1e-6, rt=2e-6, beta=0.25, v=30, sd=0.01, sw=1e-8, st=0.2`.
It now returns finite density and scores at tolerance `1e-8`. Quadrature
uncertainties remain estimates, with the same bounded-work failure semantics.

## Endpoint expansion for steep Gaussian intervals

For the normalized start coordinate `z in [-1/2,1/2]`, consider
`exp(-A z²+Bz)` with `A >= 0`. If `L = |B|-A > 0`, its peak lies at an
endpoint. Substituting `z = sign(B)(1/2-y)` gives

\[
-Az^2+Bz=-A/4+|B|/2-Ly-Ay^2,\qquad 0\le y\le1.
\]

Expand only `exp(-Ay²)`, and integrate the resulting polynomial against
`exp(-Ly)` over the positive half-line. Its monomial integrals are
`m!/L^(m+1)`. After `N` terms, Taylor's theorem for a negative real exponent
bounds the remainder by

\[
\frac{A^N}{N!}\int_0^\infty y^{2N+m}e^{-Ly}\,dy
=\frac{A^N(2N+m)!}{N!L^{2N+m+1}}.
\]

For moment `r`, expand `(1/2-y)^r` and use the absolute polynomial coefficients
in the remainder bound. Extending the interval beyond one contributes at most

\[
\frac{e^{-L}}L\sum_{j=0}^r {r\choose j}(3/2)^{r-j}\frac{j!}{L^j}.
\]

The evaluator tries up to 32 terms when `L >= 40` and `A/L² <= 0.0025`, retaining
the result with the smallest explicit remainder. Compensated summation is used
for the alternating expansion. The final likelihood and score checks receive
both the expansion remainder and the removed-tail bound. This remains a
real-arithmetic bound; floating-point error is separate.

This form handles steep intervals without subtracting Gaussian tail
probabilities or using unstable upward moment recurrences. Twelve independent
Acb continuous integrals cover both endpoint signs, curvature up to `1e12`,
and linear coefficients up to `3e12`.

The independent ball reference combines the image Gaussian and drift marginal
before evaluating either density or derivatives. A separate test compares this
form with the original independently enclosed Wiener5 dual series at large
drift variance. Reference quadrature tolerance tightens along with arithmetic
precision: it starts at `goal/4096`, decreases on escalation, and never falls
below the working-precision allowance. This only allocates work. Success still
requires the resulting rigorous output enclosures to satisfy the requested
width; agreement between successive point estimates cannot grant a certificate.

The optional ball reference can also average each finite image analytically
over the start interval. It uses complex Gaussian moments, with bounded
exponential expansions near zero and at a steep endpoint, or error-function
primitives elsewhere. For the endpoint expansion it integrates each monomial
over the finite interval exactly:

\[
\int_0^1 y^m e^{-Ly}\,dy
=\frac{{}_1F_1(m+1;m+2;-L)}{m+1}.
\]

For complex coefficients with `Re(A) >= 0` and `Re(L) > 0`, the remainder is
bounded by `|A|^N (2N+m)! / (N! Re(L)^(2N+m+1))`. Continuous Acb integrals,
including complex decision times, independently check these start averages.
The outer time integral and its omitted time-zero slab remain enclosed.

For a moderate exponent, FLINT evaluates the start moment's power series as a
single polynomial. On the complex circle `|z|=1`, the exponential
`exp(-Az²+Bz)` has modulus at most `exp(|A|+|B|)`. Cauchy's coefficient bound
therefore limits its remainder after degree `N-1`, uniformly on `|z|<=1/2`, to
`exp(|A|+|B|) 2^(1-N)`. This bound is added to each integrated moment. Computing
the polynomial in FLINT avoids repeated Python polynomial products at every
quadrature node without replacing the enclosure check by a heuristic.

For positive time width `h`, the reference obtains time derivatives from the
two continuous boundary densities. If `T = rt - tau`, `L = max(0,T-h)` and
`I = integral_L^T g(t) dt / h`, then

\[
\partial_\tau I=\frac{g(L)-g(T)}h,\qquad
\partial_h I=\frac{g(L)-I}h.
\]

The zero-time density vanishes for the admitted interior start interval.
Ball arithmetic propagates cancellation in these expressions, while independent
saved integrals check the resulting scores. These alternatives help the
reference evaluator; they do not change production accuracy guarantees.

## Start faces and Brownian scaling in the adaptive fallback

For a start interval of width `w`, let `H_-` and `H_+` be the time-averaged
likelihoods at its two ends, and let `I` be the likelihood averaged over both
intervals. Differentiating the continuous uniform average gives

\[
\partial_\beta I=\frac{H_+-H_-}{w},\qquad
\partial_w I=\frac{(H_++H_-)/2-I}{w}.
\]

The rare adaptive fallback can integrate these one-dimensional faces instead
of cancelling large positive and negative start derivatives inside a sharp
time peak. Face tolerances scale with `w`; their errors and the central density
error are propagated through the width division. Region refinement also accounts
for this amplification. Very narrow widths keep the unit-cube or moment
derivatives, and an unavailable face calculation retains the direct integration.

The cumulative probability with Gaussian drift variability satisfies
`F(T,a,v,s) = F(T/a²,1,av,as)`. For a time interval `[L,T]`, with face densities
`g(L)` and `g(T)` and width `h`, differentiation therefore gives

\[
a\,\partial_a I=
\frac{2[Lg(L)-Tg(T)]}{h}
+v\,\partial_v I+s\,\partial_s I.
\]

This identity removes another cancelling integral. It applies to both response
boundaries and to clipped intervals, where `L=0` and `g(L)=0`. The final score
checks include the errors of both faces and both drift derivatives. Six
independent ball fixtures obtain the clipped probability and scores by averaging
spectral CDFs over the start interval, without time quadrature. As elsewhere,
adaptive quadrature discrepancies remain estimates rather than certified bounds.

## Exact time averaging with Gaussian drift variability

Gaussian drift variability does not require an extra numerical integral in the
boundary CDF. For `V ~ Normal(v,s²)`, completing the Gaussian square gives

\[
E[e^{kV}\Phi(A+BV)]
=e^{kv+k^2s^2/2}\Phi\!\left(
\frac{A+B(v+ks^2)}{\sqrt{1+B^2s^2}}\right).
\]

Apply this identity to each integrated image. Write `x=aq`,
`d_j=a|q+2j|`, `lambda=1+s²T`, and `k_sigma=x-sigma*d_j`, with `sigma=+1,-1`.
The Gaussian-drift boundary CDF is

\[
F_5(T)=\sum_{j\in\mathbb Z}\operatorname{sign}(q+2j)
\sum_{\sigma=\pm1}
e^{k_\sigma v+k_\sigma^2s^2/2}
\Phi\!\left(\frac{\sigma(v+k_\sigma s^2)T-d_j}{\sqrt{T\lambda}}\right).
\]

Directly multiplying the exponential and normal tail is poorly conditioned.
For a nonnegative tail argument `z`, both terms share the combined exponent

\[
E_j=-\frac{(aq-vT)^2}{2T\lambda}
-\frac{d_j^2-a^2q^2}{2T}.
\]

Their log values are `E_j - log(2π)/2 + log R(z)`, where `R` is the normal
Mills ratio. The implementation differentiates this expression and uses a
continued fraction for `R'(z)/R(z)` at large `z`, avoiding subtraction of
nearly equal normal-tail ratios. It also simplifies the time derivative of
`z` before evaluation; large drift variance otherwise causes another
cancellation. Signed images use compensated summation.

The omitted images admit explicit derivative bounds. For the first omitted
distance `d`, require `d²-a²q² >= 5T`. Every factor
`t^(-r) exp(-(d²-a²q²)/(2t))`, `r <= 5/2`, then increases throughout
`0<t<=T`. Integrating its endpoint bound controls the CDF tail and all five
parameter derivatives. The same positive lattice moments used by the density
evaluator supply these bounds. This is a real-arithmetic truncation analysis;
it does not bound the production elementary functions' rounding errors.

More explicitly, let `M_p` sum the two omitted lattices of
`d^p exp(-d²/(2u))`, using dimensionless image distances, and put
`P = a exp(q²/(2u)) / sqrt(2πT)` and `r = aq+|v|T`. Sufficient bounds are

\[
\begin{aligned}
B_0&=PM_1,& B_\tau&=PM_1/T,\\
B_a&=P[(1/a+qr/T+aq^2/T)M_1+aM_3/T],\\
B_\beta&=P[M_0+a^2M_2/T+(a^2q+ar)M_1/T],\\
B_v&=PrM_1,& B_s&=Ps(r^2+T)M_1.
\end{aligned}
\]

They deliberately discard the central Gaussian and use `lambda >= 1`, so they
are sufficient envelopes rather than optimal parameter-dependent bounds. The
time derivative is the boundary density, which has the stated endpoint bound.

Two CDF values now remove time quadrature with positive Gaussian drift
variability. The implementation checks interval subtraction and score-error
propagation before accepting the result. Its image implementation is restricted
to `T/a² < 2`; unsupported or poorly conditioned evaluations retain numerical
quadrature. Adaptive evaluation can instead integrate only over the starting
point, evaluating the exact time average at each node. It first retains the
existing Gaussian-start reduction when that supported path is preferred. Fixed
quadrature retains its selected finite-rule semantics.

Seventy-two independent ball fixtures cover clipped and unclipped intervals,
both response boundaries, and drift standard deviations `0.5`, `350` and `1e12`.
The largest variance also checks the numerical fallback: subtracting two CDFs
close to their limiting mass can lose accuracy even when each CDF is accurate.
Cancellation diagnostics reject that shortcut before returning its result.
The independent CDF reference also agrees with the earlier continuous time
integrals. New timing and coverage measurements are needed before attributing
speed or full-support claims to this shortcut.

Nine captured numerical proposals also have independent ball fixtures. One
large-variance case uses the exact CDF reference; the others retain continuous
time integration. Both density and fused scores are checked, since a finite
two-dimensional quadrature result can still miss its requested accuracy.
