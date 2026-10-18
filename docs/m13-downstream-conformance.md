# M13 downstream conformance boundary

Reviewed: 2026-10-18

[`tools/downstream-conformance`](../tools/downstream-conformance) is a separately
locked Cargo package outside Alea's production workspace. It therefore sees the
repository crates through the same public path boundary as an ordinary path
dependency and does not inherit workspace dependencies, private helpers, test-only
visibility, or crate features.

The package provides reusable law assertions and independent implementations for:

- `LogDensityGradient`: fixed dimensions, checked shape atomicity, complete finite
  gradients, known mathematical fixtures, and repeatable semantic results;
- `EuclideanMetric`: shape atomicity and the mass/sampling/velocity/kinetic identity
  `K(Lz) = z^T z / 2`;
- `MarkovChain`: fixed finite positions, coherent success, exact position retention
  on error, and caller-owned randomness;
- `DiagnosticTransition`: deterministic side-effect-free projection, finite bounded
  acceptance probabilities, and finite optional energy; and
- the runtime sink boundary: an external higher-ranked closure consumes borrowed
  `Draw` records and chooses explicitly what to own.

CI formats, tests, lints with warnings denied, and builds rustdoc for this package
using its own lockfile. The suite deliberately excludes sealed integrators: opening
an extension point is contingent on first deriving laws from concrete uses. It also
excludes the S0 particle traits until M12 supplies working vertical slices. Invalid
ownership, adaptation-after-freeze, and exact/approximate composition remain for
the separate compile-fail gate.
