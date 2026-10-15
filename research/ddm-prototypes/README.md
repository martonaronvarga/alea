# Pre-migration DDM prototypes

These six files are byte-preserved from commit
`50b66bb` (`crates/runtime/src/ddm/`). They were **not compiled or exported**
before the architecture migration. They are research inputs, not usable APIs.

- `approx.rs`: grid configuration and linear interpolation implementation, plus
  an unfinished likelihood grid whose `log_pdf` returns NaN.
  Grid/spline utilities are now ported and tested in `alea-math::interpolation`;
  this original remains intact because the likelihood approximation is unfinished.
- `batch.rs`: affine parameter-map sketch and unresolved autodiff integration.
- `cse.rs`: condition encodings and affine parameter maps; unresolved types/methods.
- `ddm.rs`: independent series, CSE likelihoods, transforms and 22 dormant tests.
  Several imports and tests do not compile; the Monte Carlo helper's RNG and
  loose statistical assertions also need review before using it as an oracle.
- `ffi.rs`: C-layout conversion sketch, not a linked external API.
- `model.rs`: observation adapter using superseded, unresolved types.

Keep these sources and tests until their useful behavior has been ported and
validated in `alea-distributions`, `alea-core` or `alea-autodiff`. Do not compile
the duplicate Wiener series into production merely to retain its old path.
The supported latent DDM implementation and its two tests are already migrated
to `crates/alea-distributions/src/latent.rs`.
