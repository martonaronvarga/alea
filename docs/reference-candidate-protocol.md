# Reference candidate validation protocol

Accepted pre-cycle source: `df17c0a0af7ab8aabc7753304e3beacd6d8c7135`.
The proposed release surface is core/math/reference MCMC/runtime. The candidate
remains unpublished; research exactness and default selection are unchanged.
Public Rust API compatibility is checked for all four crates, including their
existing named research exports. No implicit semver exemption is claimed for
those exports. A candidate review is separate from S4/stable publication.

## Independent consumer

`tools/release/fresh.py --workflow` creates a separate project/manifest/lock,
fetches its own dependency graph and compiles only public library APIs. Run the
same workflow from source and all nine actual unpacked `.crate` archives.
`--output NEW_DIRECTORY` retains JSON results, a checkpoint and durable draws.
This is an agent-executed downstream exercise; external-human feedback must be
recorded separately if supplied.

- Gaussian regression: five centered design points, known unit noise, independent
  Normal(0,2) coefficient priors. Four chains, 512 warmup and 2048 retained draws.
  Run covariance/default NUTS and explicit gradient-diagonal/continuous-final-
  tuning/10% jitter/position-turning profiles. Compare both coefficient means and
  variances with independent conjugate formulas. Absolute mean tolerance .05 and
  relative second-moment tolerance .15; R-hat at most1.03 and ESS at least200.
- Positive precision: Gamma(3,2) model compiled through a positive transform.
  Four chains,512warmup/2048draws, covariance/default NUTS. Constrained mean
  tolerance .10 and relative second-moment tolerance .25, with the same diagnostic
  limits. This checks one transform Jacobian through the user workflow.
- Recovery: cancel the fast-profile regression after64 retained draws, write/read
  checkpoint v2, resume through create-new durable streaming without collection.
  All192 resumed draws and the final point/RNG/policy checkpoint must equal the
  uninterrupted256-draw run. Existing output must be preserved.
- Bounded SBC:32 independently generated prior-predictive regression datasets,
  fixed separate generator seed91, one chain each,512covariance warmup/512draws.
  Retain every dataset, truth and coefficient rank. Four bins per coefficient,
  chi-square limit16.27. Single-chain warnings do not certify convergence; this
  bounded calibration sanity check is not broad ecosystem validation.

Seeds, budgets, equations and limits are in `examples/workflow.rs` before candidate
execution. Do not tune seeds, remove datasets or relax limits after observing a
failure. Record and diagnose it instead.

## Portability, compatibility and operational checks

Run native MSRV1.95 and current stable on Linuxx86_64/aarch64, macOSx86_64/aarch64,
and Windowsx86_64. Also run the pinned Faer/SIMD/BLAS/Enzyme matrix, Miri, native
ASan/UBSan, parser/property checks and409,600-step/200-restart operational soak.
CI configuration alone is not execution evidence; bind successful run IDs to
the candidate source revision. Preserve failed attempts and their repairs.

Checkpoint v2 explicitly encodes NUTS jitter/turning. V1 checkpoints are rejected
instead of silently changing the kernel. Earlier draw data remains readable.
Compare the four crates with the accepted pre-cycle source using actual
`cargo-semver-checks`, plus API inventories and the documented binary-schema diff.

## Descriptive performance bands

On native Linuxx86_64 and aarch64, build both accepted-source and candidate probes
before measurement. Use one compiler, generic release opt-level3, identity Gaussian
dimensions2/16/64, step.25, depth8, 10%jitter, position turning, seed43,33burn and
5000draws. Rotate baseline/candidate process order over five paired repetitions.
Timing includes transitions and the draw checksum, excludes builds and warmup,
and has no hard threshold. Preserve work counts/checksums; those should agree
because this cycle does not change the frozen numerical kernel.

Record each host separately, every raw observation, compiler, probe hash, paired
median ratio and observed range. These are initial descriptive regression bands
for these hosts, not a general speed guarantee or a fresh PosteriorDB ranking.

## Candidate seal

After applicable checks pass, retain the exact code revision, API/schema scope,
platform runs, downstream/SBC/soak and semver results, package hashes, two identical
source archives and a signature from the existing maintainer key. Publish no
registry package and make no S4 claim as part of preparing this candidate.
