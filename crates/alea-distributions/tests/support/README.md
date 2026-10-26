# Wiener extreme-support lookup

`wiener_extremes.csv` is a regression table, not a certified interpolation table
or a runtime approximation. Generate it with Python and mpmath 1.4.1 using
`generate_wiener_extremes.py`. It contains 648 joint density/score evaluations at
exact binary64 inputs, for both boundaries. No Stan or Alea evaluator generates
the expected results. Direct image/eigenfunction sums are recomputed at 100 and
160 decimal digits with 24 and 48 terms, requiring 60-digit relative agreement.
This refinement check is empirical numerical evidence, not an interval proof.

The cases include scaled times from 1e-8 to 100, beta from 1e-15 to the immediate
binary64 predecessor of one, drift/SD pairs (0,0), (-30,0), and (-30,5), and starts on
both sides of the endpoint dispatch threshold. Scores are with respect to the
original alpha, tau, beta, delta and s_delta, including lower-boundary signs.

The independent formula uses a log-scaled image sum at scaled times <= 0.25 and
an eigenfunction sum otherwise. mpmath differentiates this formula. Keeping
inputs exact avoids silently replacing beta by 1-(1-beta) in the reference.

`generate_wiener_balls.py` uses python-flint 0.9.0 (Arb) to regenerate
`wiener_balls.csv`. The 288 cases use exact binary64 inputs, rigorous ball
arithmetic and positive geometric remainder bounds for density and both
normalized derivatives. Precision escalates through a bounded schedule until
all log-density and parameter-score enclosures meet `1e-24 * (1 + |midpoint|)`.
Saved radii include conversion of each midpoint to binary64. Ordinary tests
need no Python dependency. These fixtures cover Wiener4/5, not Wiener7
integration, and do not assert that the Rust implementation has enclosed
floating-point error. Regenerate with:

```sh
python -m pip install python-flint==0.9.0
python crates/alea-distributions/tests/support/generate_wiener_balls.py
```

`generate_wiener_cdf_balls.py` generates 400 independent Arb CDF and boundary-tail
log-probability enclosures in `wiener_cdf_balls.csv`, with the same precision and
binary64 conversion rules. It uses integrated image terms or spectral tails,
with geometric remainder bounds, across both boundaries and drift -20 through
20. Run it with python-flint 0.9.0 installed. These cases include the branch
switch, very small CDFs and very small boundary tails.

`generate_wiener7_balls.py` adds 48 continuous start-variability integrals,
enclosing the finite image densities and all seven derivatives with Acb's
rigorous integrator and adding uniform analytic tail envelopes over the whole
start interval. The integrands are entire in the integrated coordinate; no
complex square root or logarithm is applied to that coordinate. The precision
schedule checks the resulting log-density/score balls, not agreement between
successive numerical answers. It covers zero time width, including its correct
one-sided score, and does not cover nested or clipped time integration.

`generate_wiener7_nested_balls.py` adds eight independent nested and clipped
variability integrals. The reusable `wiener7_reference.py` encloses all eight
outputs, bounds omitted images over the integration rectangle, and bounds the
removed positive-time slab when the support is clipped at zero. Complex square
roots propagate the integrator's analyticity flag. A common logarithmic scale
keeps rare-event integrals representable. Work and precision limits return an
explicit failure rather than a certificate.

`generate_wiener_cdf_gradient_balls.py` generates 24 CDF/score enclosures by
integrating the density and its derivatives continuously. It includes extreme
starts and both drift signs. This provides a different calculation from the
production integrated-image CDF formulas.

The optional reference can audit an actual binary64 result pointwise:

```sh
python crates/alea-distributions/tests/support/wiener7_reference.py \
  '{"params":[0.03,1,0,0.5,2,0,0,0.03,false],"candidate":[0,0,0,0,0,0,0,0]}'
python crates/alea-distributions/tests/support/test_wiener_reference.py
```

Supply the actual log-density followed by the seven scores in `candidate`.
The deliberately incorrect example is rejected. Successful comparisons bound
all numerical error at that input, including production rounding, without
assuming that standard-library elementary functions are correctly rounded.
They do not prove a uniform error bound for every production evaluation.
Statuses distinguish a certificate, a rejected candidate, insufficiently narrow
enclosures, a nonfinite candidate, invalid inputs and exhausted reference work.
Ordinary Rust tests consume saved fixtures and need no Python installation.

`generate_wiener7_grid_balls.py` fixes the complete 76-case Wiener7 timing grid
as regression fixtures. It covers tiny and wide start intervals, spectral and
image regimes, tiny and clipped time intervals, simultaneous variability, long
times and rare events. At beta one half it uses the exact boundary/drift
reflection, including both score sign changes, to avoid repeating equivalent
integrals. `wiener7_grid_balls.csv` encloses every expected value; Rust checks
both density-only and fused paths with a declared `1e-9 * (1 + |reference|)`
allowance. This is a finite-domain regression suite, not a universal guarantee.
