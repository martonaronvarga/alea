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

For narrow intervals, the evaluator integrates an exponential Taylor polynomial
and carries the uniform remainder `exp(R) R^20 / 20!` into each moment's error.
Here `R` bounds the absolute exponent throughout the interval. Otherwise it
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
smallest interval curvature `a² h² / (2 t (1+s²t))` must be at least one half.
A measured regression on a broad integrand motivated retaining ordinary
cubature below this threshold. This policy does not promise minimum latency. Any failed moment or
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
Almost constant start intervals (`s_beta <= 1e-6`) use numerical integration;
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

## References

- [Navarro and Fuss (2009), density expansions and truncation](https://papers.djnavarro.net/2009_firstpassage.pdf).
- [Blurton, Kesselmeier and Gondan (2012), cumulative first-passage distributions](https://doi.org/10.1016/j.jmp.2012.09.002).
- [Gondan, Blurton and Kesselmeier (2014), improved series evaluation](https://doi.org/10.1016/j.jmp.2014.05.002).
- [Hartmann and Klauer (2021), partial derivatives and error bounds](https://doi.org/10.1016/j.jmp.2021.102550).
- [Python-FLINT documentation](https://fredrikj.net/python-flint/arb.html), including rigorous real and complex ball arithmetic.
