# Wiener extreme-support lookup

`wiener_extremes.csv` is a regression table, not a certified interpolation table
or a runtime approximation. Generate it with Python and mpmath 1.4.1 using
`generate_wiener_extremes.py`. It contains 432 joint density/score evaluations at
exact binary64 inputs, for both boundaries. No Stan or Alea evaluator generates
the expected results. Direct image/eigenfunction sums are recomputed at 100 and
160 decimal digits with 24 and 48 terms, requiring 60-digit relative agreement.
This refinement check is empirical numerical evidence, not an interval proof.

The cases include scaled times from 1e-8 to 100, beta from 1e-15 to the immediate
binary64 predecessor of one, drift zero or -30, drift SD zero or 5, and starts on
both sides of the endpoint dispatch threshold. Scores are with respect to the
original alpha, tau, beta, delta and s_delta, including lower-boundary signs.

The independent formula uses a log-scaled image sum at scaled times <= 0.25 and
an eigenfunction sum otherwise. mpmath differentiates this formula. Keeping
inputs exact avoids silently replacing beta by 1-(1-beta) in the reference.
