# NUTS performance attribution

This private executable compares the pinned nuts-rs revision with Alea using the
same analytic standard-normal density, identity geometry, fixed step0.3, depth8,
and dimensions2/16/64. The common start is all ones, so nuts-rs's score-based
initialization also installs identity geometry. Its one tuning draw cannot fit a
metric from only two observations. Fixed step adaptation and no jitter hold
geometry and step constant thereafter. Each engine burns another32draws before
timing20,000retained transitions. Seed43001 labels independent library RNG streams.

The target counts actual fused calls. Both loops include obtaining an owned
position and accumulating the first coordinate's moments. No CSV, progress JSON,
diagnostics, build or JIT occurs inside retained timing. Different tree criteria
and candidate/RNG policies remain intentional variables. This diagnoses work per
transition and time per call, rather than claiming identical trajectories.

The normal moments are a bounded smoke check, not a convergence guarantee. Full
posterior accuracy and efficiency measurements remain a separate experiment with
raw draws. Run alternating processes only after builds/tests finish, preserving
every repetition. Report cold process and retained time separately.
