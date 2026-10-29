# Alea

Probability computation and statistical inference in Rust. Evaluate and sample
distributions, construct models with named constrained parameters, optimize
likelihoods or posterior modes, and run NUTS, HMC, or random-walk Metropolis.

Alea includes hierarchical modeling, batch regression likelihoods, analytic model
gradients, warmup, parallel chains, diagnostics, posterior prediction, streaming
output, and checkpoint recovery. Specialized laws include Wiener diffusion and
directional distributions. Optimization records point estimates and numerical
termination; finite estimator existence and uncertainty need separate assessment.

Use `alea` as the application dependency and import its modules or `alea::prelude`.
The default build requires stable Rust 1.95 or later. Optional features include:

- `distributions`: probability laws and likelihood factors; requires a C compiler.
- `analysis`: named CSV/JSON sampling artifacts for downstream analysis.
- `faer`: matrix operations and Gaussian least-squares assessment.
- `std-autodiff`: compiled Enzyme derivatives with a matching nightly toolchain.

See the [user guide](https://github.com/martonaronvarga/alea/blob/main/docs/README.md)
for complete programs covering modeling, optimization, sampling, and result
export. The [installation guide](https://github.com/martonaronvarga/alea/blob/main/docs/installation.md)
describes Git dependencies, reproducible builds, and optional backends.

## License

GPL-3.0-or-later; see [LICENSE](https://github.com/martonaronvarga/alea/blob/main/LICENSE-GPL) and
[THIRD_PARTY_NOTICES.md](https://github.com/martonaronvarga/alea/blob/main/THIRD_PARTY_NOTICES.md).
