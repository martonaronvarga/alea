# PSIS reference fixture

`psis.cpp` calls Stan2.38's actual `psis_weights` and tail-shape implementation.
The fixture is `crates/alea-mcmc/tests/fixtures/pathfinder/stan-2.38-psis.txt`.
It contains64/256-element smooth, Pareto-like and mixed log-weight inputs,
unnormalized shifted smoothed weights, tail sizes and Pareto shapes. It is not a
second implementation of PSIS. The tested Rust implementation uses the same
policy with stable excess arithmetic; explicit extensions handle support-zero,
small and tied tails.

The pinned Stan distribution comes from `flake.lock` / `.#cmdStan` (2.38.0).
Compile this small caller with C++17 against `stan/src`, `stan/lib/stan_math`,
its Eigen/Boost/TBB/Sundials includes, `-D_REENTRANT`, and the pinned TBB library;
then redirect its stdout to the fixture. The caller and fixture were executed
locally, including the existing `2e-12` weight/shape regression tolerance.
Upstream code retains Apache-2.0; no Stan implementation is vendored here.
