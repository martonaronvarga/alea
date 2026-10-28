# Directional distributions

Directional observations describe an orientation rather than a location in
Euclidean space. A unit vector in three dimensions lies on a sphere; its density
is measured per unit surface area, not per unit volume. `UniformSphere` and
`Bingham` use this measure. `UniformBall` instead uses ordinary volume inside
the unit ball. All three are available with the `distributions` feature.

## An antipodally symmetric model

The Bingham law assigns equal probability density to `x` and `-x`. It is useful
when an axis is observable but its sign is not, for example for orientations of
unoriented fibres. In principal-axis coordinates its density is

\[
p(x\mid\theta)=\frac{\exp(\sum_i\theta_i x_i^2)}{C(\theta)},
\qquad \sum_i x_i^2=1.
\]

Rotate observations into those coordinates when the principal axes differ from
the coordinate axes. Concentrations control relative preference for each axis.
A common shift leaves the distribution unchanged, so fix one concentration when
estimating the others. The implementation subtracts the largest concentration
internally to avoid large positive exponentials.

```rust
use alea::distribution::{Distribution, ParameterScore, Sample};
use alea::distributions::directional::Bingham;
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let law = Bingham::new(vec![-10.0, -2.0, 0.0])?;
    let observation = [0.6, 0.8, 0.0];
    println!("log density = {}", law.log_prob(&observation)?);
    let mut gradient = [0.0; 3];
    law.parameter_score(&observation, &mut gradient)?;
    println!("concentration scores = {gradient:?}");
    println!("mean squared coordinates = {:?}", law.mean_squares());
    let mut rng = SmallRng::seed_from_u64(123);
    println!("sample = {:?}", law.sample(&mut rng)?);
    Ok(())
}
```

Scores are `x_i² - E[X_i²]`. The event score belongs to the sphere's tangent
space: `tangent_score` computes `2 x_i (theta_i - sum_j theta_j x_j²)`.
An unconstrained Euclidean sampler needs a suitable parameterization and its
Jacobian; treating unit vectors as unconstrained real vectors is incorrect.

## Holonomic normalization

Bingham normalization is a useful application of the
[Holonomic Gradient Method](https://arxiv.org/abs/1304.7973). Direct integration
would require a sphere of dimension `d-1` and separate moment integrals.
Instead, Alea continues the normalizer and its derivatives along a radial path.
The following reduction avoids eigenvalue-difference denominators, including
when concentrations are equal.

Let `lambda` be the shifted concentrations, `X` be uniform on the sphere, and

\[
Z(t)=E[\exp(t\sum_i\lambda_i X_i^2)],\quad
 g_i(t)=E[X_i^2\exp(t\sum_j\lambda_j X_j^2)].
\]

The sphere identity gives `sum_i g_i = Z`. Integration by parts under rotations
in the `(i,j)` plane yields
`2t(lambda_i-lambda_j) E[X_i²X_j² exp(tQ)] = g_i-g_j`.
Combining these equations with `sum_j X_j²=1` gives the radial system

\[
g_i'=\lambda_i g_i+\frac{\sum_j g_j-dg_i}{2t}.
\]

For normalized moments `m_i=g_i/Z` and `L=log Z`, it becomes

\[
L'=\mu,\qquad
m_i'=(\lambda_i-\mu)m_i+\frac{1-dm_i}{2t},\qquad
\mu=\sum_i\lambda_i m_i.
\]

At zero, `g_i(0)=1/d`; the apparent singularity is removable. A 20-term Taylor
initialization starts at `t*spread <= 0.05`. If `g_i=sum_k a_(i,k)t^k`, its
coefficients satisfy

\[
S_{k+1}=\frac{\sum_i\lambda_i a_{i,k}}{k+1},\qquad
 a_{i,k+1}=\frac{\lambda_i a_{i,k}+S_{k+1}/2}{k+1+d/2}.
\]

Adaptive fourth-order Runge–Kutta step doubling then continues to `t=1`, with
Richardson correction. The local error scales are `2e-14 + 2e-12*abs(state)`.
These are numerical error estimates, not certified global enclosures. The
constructor checks positive moments, their sum, finite outputs and a bounded
work budget. It supports 2–64 coordinates and concentration spread up to 10,000;
unsupported or unsuccessful evaluations return an error. Equal concentrations
are uniform. In two dimensions a scaled Bessel formula is cheaper, so that case
uses the closed form instead of an ODE.

The final constant and moments are cached. Evaluating an observation or its
parameter score is then linear in dimension and performs no ODE solve. This is
the intended benefit of HGM here; no general speed advantage over other Bingham
implementations has been established.

Sampling uses a normal proposal followed by angular-central-Gaussian rejection.
For `Q=sum_i lambda_i x_i² <= 0`, choose `b` solving
`sum_i 1/(b-2lambda_i)=1`. Normalize independent normals with variances
`1/(b-2lambda_i)`, then accept with log probability

\[
Q+\frac d2\log(1-2Q/b)
-\left[\frac{b-d}{2}+\frac d2\log(d/b)\right].
\]

The bracket is the maximum of the preceding function over `Q<=0`, so the
acceptance probability is at most one. Sampling allows 4096 proposals and
returns an error if that budget is exhausted.

Independent tests compare the normalizer and moments against direct
high-precision spherical integrals, including repeated concentrations and
spread 1,000. They also check the two-dimensional von Mises identity, parameter
finite differences, shift invariance, generated second moments, and the spread
10,000 work boundary. These checks do not constitute a continuous-domain error
certificate.
