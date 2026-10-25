# Bounded NUTS implementation measurements

This standalone tool adds no production dependency. `flat DIM DRAWS` measures
full depth-six trees with exactly63new gradients per transition. Its improper
constant density is a mechanical work fixture, not an inference target.
Construction and32warm transitions are excluded; allocation-free transitions,
RNG, complete evaluation and counters are timed with nanosecond Instant clocks.
Before/after seeded paths may differ; checked edge counts match.

`fit DIM REPEATS` times dense metric construction from one fixed covariance
estimate; observations are outside timing. It runs unchanged against the archived
baseline and current libraries. Every fit contributes to an output checksum.

With `--features faer`, `adaptation NEW_DIRECTORY` runs a bounded descriptive
window-period comparison. Five existing independent analytic/JAX-tested2D targets
(correlated Gaussian, banana, logistic, width-one funnel, rotated Gaussian),
five methods (covariance diagonal; Fisher diagonal/rank-cap2 with period80/40),
four chains, seeds23001–23004, common start(.25,-.25),512warmup and2048retained,
target acceptance.8, initial step.1, depth8, history160. Every retained raw draw,
actual call count, divergence, depth saturation, step, realized rank and full
frozen inverse mass is recorded. Five independently defined scalar observables
receive modern R-hat/bulk/tail ESS/mean/MCSE; undefined values remain explicit.
Only period differs within each Fisher pair. Comparing covariance with Fisher
also changes estimator and schedule. Rank-cap2 is a full2D fit, not evidence of
large-dimensional low-rank scaling. One run is not a precise walltime estimate.

Build from the root:

```sh
CARGO_TARGET_DIR=target/nuts-profile nix develop .#stable --command \
  cargo build --release --locked --features faer --manifest-path tools/nuts-reference/implementation-profile/Cargo.toml
```

Run timings serially after all builds/tests complete. Preserve every round and
alternate binary order. Process orchestration must use a blocking process wait
and high-resolution clocks, without50-ms timeout polling. These results have no
promotion threshold or automatic default-selection rule.
