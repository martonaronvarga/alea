# Independent ATLAS proposal fixtures

Run `python3 tools/atlas-reference/generate.py --check` from the repository root.
The generator uses a closed harmonic 2×2 leapfrog matrix and explicit auxiliary
variates. It imports no Alea code and reads no Alea outputs. The 576 cases cover
preliminary, delayed and preliminary-failure proposals; unavailable reverse
support, zero rejection mass, Metropolis rejection and per-leg work exhaustion.
The private Rust kernel test compares outcomes, probabilities, positions and
forward/reverse/ghost/HVP counts. Independent scalar log-flux checks use the
paper's Appendix E.2 densities. These are bounded Gaussian construction checks,
not a proof for arbitrary nonlinear targets or author-code trajectory equivalence.
The paper-versus-author density discrepancy is explained in
[the method document](../../docs/native-warmup-and-research-workflows.md).
Hashes and scope are recorded in `provenance.json`. CI checks generator drift.
