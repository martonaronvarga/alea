# Contributing

Contributions follow the target package's license: MIT OR Apache-2.0 for the six
reusable Rust libraries, GPL-3.0-or-later for ffi/distributions/cli. Preserve all
upstream notices and identify copied or adapted material and its source license.
See [LICENSING.md](LICENSING.md); the project license does not relicense others' work.

Start with a small reproducer and identify the owning crate. Preserve checked
construction, typed model errors, coherent cache rollback and explicit finite
warmup. Research algorithms retain their maturity/exactness labels. See the
[architecture](docs/architecture.md), [maturity policy](docs/maturity-and-promotion.md)
and [release policy](docs/releases.md) before changing public contracts.

Use the lock-pinned development shells. Run focused tests first, then the related
integration/property tests and `nix develop .#stable --command bash tools/ci/check.sh`.
Use the pinned nightly/BLAS/Enzyme lanes only for the corresponding features.
Public API changes require inventory regeneration and explicit compatibility
review; schema changes require versioning and reader tests. Do not regenerate
historical benchmark evidence to make a current change pass.

See [workspace maintenance](docs/workspace.md) for build-cache retention, archived
benchmark restoration and full historical reader checks. Routine CI validates
benchmark contracts and the payload catalog; raw historical replay is explicit.

Report ordinary bugs through the [issue tracker](https://github.com/martonaronvarga/alea/issues)
with revision, compiler, features, platform, seed and minimal input. Report
security-sensitive issues through [SECURITY.md](SECURITY.md). Include relevant
validation and remaining limitations in change descriptions.
