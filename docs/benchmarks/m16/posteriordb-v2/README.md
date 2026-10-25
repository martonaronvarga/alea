# Descriptive PosteriorDB benchmark on dusk

20 posteriors,26 configurations,3 repetitions,4 chains.

Outcomes: {"completed": 1267, "failed": 260, "unsupported": 33}.

Open report.html and findings.md; phase-costs.csv and timing-dispersion.csv expose granular costs. All failures and unsupported rows remain. RMHMC is exactness-pending and many FD-third-tensor backend contracts fail before sampling; MCLMC is approximate; ATLAS and Pathfinder are labelled component/initializer workflows. Callback-backed BlackJAX is distinct from native JAX targets. No promotion/quality eligibility gate is computed.

Unpack and verify with: python3 tools/bench/alea-bench pdb unpack BUNDLE NEW_DIRECTORY
