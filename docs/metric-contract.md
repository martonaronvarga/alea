# Momentum mass and metric conventions

Hamiltonian Monte Carlo introduces momentum with a Gaussian distribution. The
momentum mass matrix `M` determines both the momentum distribution and the
relationship between momentum and velocity:

- Momentum has covariance `M`.
- Velocity is `M⁻¹ p`.
- Kinetic energy is `pᵀ M⁻¹ p / 2`.

Confusing mass with inverse mass changes the dynamics. Alea uses distinct type
names for the two representations.

## Diagonal mass

```rust
use alea::math::{buffer::OwnedBuffer, metric::{DiagonalMetric, EuclideanMetric}};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mass = DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| [4.0, 9.0][i]))?;
let mut momentum = [0.0; 2];
mass.sample_momentum(&[1.0, 2.0], &mut momentum)?;
assert_eq!(momentum, [2.0, 6.0]);
let mut velocity = [0.0; 2];
mass.velocity(&momentum, &mut velocity)?;
assert!((velocity[0] - 0.5).abs() < 1e-12);
assert!((velocity[1] - 2.0 / 3.0).abs() < 1e-12);
# Ok(())
# }
```

`sample_momentum` transforms supplied independent standard-normal values. It does
not generate random numbers itself. The diagonal constructor requires positive,
finite masses and does not silently clamp invalid entries.

## Dense mass

`DenseMetric` accepts a lower Cholesky factor `L` such that `M = L Lᵀ`. The full
matrix is stored in **column-major** order, with zeros above the diagonal:

```rust
use alea::math::{buffer::OwnedBuffer, metric::{CholeskyFactor, DenseMetric, EuclideanMetric}};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// L = [[2, 0], [0.5, 3]], stored by column.
let lower = OwnedBuffer::from_fn(4, |i| [2.0, 0.5, 0.0, 3.0][i]);
let metric = DenseMetric::new(CholeskyFactor::new_lower(2, lower)?);
let mut momentum = [0.0; 2];
metric.sample_momentum(&[1.0, 2.0], &mut momentum)?;
assert_eq!(momentum, [2.0, 6.5]);
# Ok(())
# }
```

The constructor checks the factor; it does not factor an arbitrary covariance
matrix. Its diagonal must be strictly positive and finite. The supplied storage
must contain all `dimension × dimension` entries, including the zero upper
triangle. `log_det()` returns the log determinant of the mass, not of its factor.

Core parameter transforms use row-major matrix outputs. Do not pass a transformed
matrix into a metric constructor without checking the required layout and whether
the matrix is a covariance, a mass, or a factor.

## Inverse-mass representations

`InverseDiagonalMetric` and `InverseDenseMetric` store `G = M⁻¹`. They compute
velocity as `G p` while still drawing momentum with covariance `M`. The different
storage convention does not change the Hamiltonian definition.

Warmup owns the conversion from its covariance estimate to the selected frozen
geometry. Applications using session warmup normally do not need to perform that
conversion themselves. See [Sampling and warmup](sampling.md).
