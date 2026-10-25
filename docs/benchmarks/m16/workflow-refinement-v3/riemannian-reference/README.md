# Independent RMHMC numerical investigation

`newton_fixture.py --check` checks the original 80-digit nonlinear root fixture.
`solver_ladder.py --check` regenerates 72 bounded Picard/Newton cases across
four residual tolerances and three iteration budgets, plus eight valid implicit
root combinations. Both are independent Decimal calculations, with no Alea
imports or outputs. Private `riemannian` tests compare actual maps, iteration
counts and failure stages, test singular Newton systems, and verify alternative
roots against the equations. CI checks generator drift.

The [finite-map analysis](../../docs/rmhmc-finite-map-analysis.md) gives the
accepted-support argument for an ideal exact-root map and a counterexample for
finite stopping that passes tolerant reverse replay. The conclusion is negative:
reverse tolerance does not supply a volume certificate. Tight-tolerance bias is
not quantified. `ExactnessPendingRmhmc` remains the appropriate API name.
