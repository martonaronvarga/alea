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
