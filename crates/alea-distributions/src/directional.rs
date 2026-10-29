//! Probability densities with respect to sphere area or ball volume.
//! Bingham normalization uses holonomic continuation, cached at construction.
use crate::scalar::{checked, standard_normal, uniform};
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, Support,
};
use rand::Rng;
use statrs::function::gamma::ln_gamma;
use std::f64::consts::PI;
fn area(d: usize) -> f64 {
    2.0_f64.ln() + 0.5 * d as f64 * PI.ln() - ln_gamma(0.5 * d as f64)
}
fn sphere(x: &[f64], d: usize) -> Result<bool, E> {
    if x.len() != d {
        return Err(E::Shape);
    }
    if x.iter().any(|v| v.is_nan()) {
        return Err(E::Observation);
    }
    let norm = x.iter().fold(0.0_f64, |n, x| n.hypot(*x));
    Ok(norm.is_finite() && (norm - 1.0).abs() <= 1e-12)
}
fn sphere_draw<R: Rng + ?Sized>(d: usize, rng: &mut R) -> Result<Vec<f64>, E> {
    let mut x = (0..d).map(|_| standard_normal(rng)).collect::<Vec<_>>();
    let norm = x.iter().fold(0.0_f64, |n, x| n.hypot(*x));
    if norm == 0.0 || !norm.is_finite() {
        return Err(E::Numerical);
    }
    for x in &mut x {
        *x /= norm;
    }
    Ok(x)
}
/// Uniform on the unit sphere in R^dimension, with respect to surface area.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct UniformSphere {
    dimension: usize,
    log_area: f64,
}
impl UniformSphere {
    /// # Errors
    /// Requires 2<=dimension<=4096 and a representable area logarithm.
    pub fn new(dimension: usize) -> Result<Self, E> {
        if !(2..=4096).contains(&dimension) {
            return Err(E::Parameter("sphere dimension"));
        }
        let log_area = area(dimension);
        if !log_area.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            dimension,
            log_area,
        })
    }
}
impl Distribution for UniformSphere {
    type Error = alea_core::distribution::DistributionError;
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.dimension)
    }
    fn support(&self) -> Support {
        Support::UnitSphere
    }
    fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
        alea_core::distribution::ReferenceMeasure::SphereSurface {
            ambient_dimension: self.dimension,
        }
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        Ok(if sphere(x, self.dimension)? {
            -self.log_area
        } else {
            f64::NEG_INFINITY
        })
    }
}
impl Sample for UniformSphere {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        sphere_draw(self.dimension, rng)
    }
}
/// Uniform in the closed unit ball in R^dimension, with respect to volume.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct UniformBall {
    dimension: usize,
    log_volume: f64,
}
impl UniformBall {
    /// # Errors
    /// Requires 1<=dimension<=4096 and a representable volume logarithm.
    pub fn new(dimension: usize) -> Result<Self, E> {
        if !(1..=4096).contains(&dimension) {
            return Err(E::Parameter("ball dimension"));
        }
        let log_volume = 0.5 * dimension as f64 * PI.ln() - ln_gamma(0.5 * dimension as f64 + 1.0);
        if !log_volume.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            dimension,
            log_volume,
        })
    }
}
impl Distribution for UniformBall {
    type Error = alea_core::distribution::DistributionError;
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.dimension)
    }
    fn support(&self) -> Support {
        Support::UnitBall
    }
    fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
        alea_core::distribution::ReferenceMeasure::Lebesgue {
            dimension: match self.event_shape() {
                EventShape::Vector(n) => n,
                _ => unreachable!(),
            },
        }
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        if x.len() != self.dimension {
            return Err(E::Shape);
        }
        if x.iter().any(|v| v.is_nan()) {
            return Err(E::Observation);
        }
        Ok(if x.iter().fold(0.0_f64, |n, x| n.hypot(*x)) <= 1.0 {
            -self.log_volume
        } else {
            f64::NEG_INFINITY
        })
    }
}
impl Sample for UniformBall {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let mut x = sphere_draw(self.dimension, rng)?;
        let radius = (uniform(rng).ln() / self.dimension as f64).exp();
        for x in &mut x {
            *x *= radius;
        }
        Ok(x)
    }
}

/// Diagonal Bingham density exp(sum(theta_i*x_i²))/C(theta) on the unit sphere.
/// Antipodal points have equal density. Coordinates are expressed in the principal
/// axes of the law; rotate data into that basis for non-axis-aligned distributions.
/// Score order: concentrations. Adding a common constant does not change the law.
///
/// Construction solves a holonomic ODE for C and its first derivatives. ODE local
/// errors are estimates, not interval certificates. Evaluation reuses the result
/// without solving another ODE. Sampling uses bounded angular-central-Gaussian rejection.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Bingham {
    lambda: Vec<f64>,
    log_c_shifted: f64,
    shift: f64,
    moments: Vec<f64>,
    steps: usize,
    envelope_b: f64,
}
impl Bingham {
    /// # Errors
    /// Requires 2..=64 finite concentrations, spread<=10,000, and successful
    /// error-controlled holonomic continuation within 50,000 attempted steps.
    pub fn new(concentrations: Vec<f64>) -> Result<Self, E> {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!(target: "alea::numerics", "bingham_normalization", dimension = concentrations.len()).entered();
        let outcome = Self::new_inner(concentrations);
        #[cfg(feature = "tracing")]
        match &outcome {
            Ok(result) => {
                tracing::debug!(target: "alea::numerics", schema_version = 1_u64, event = "finished", steps = result.steps, error_contract = "estimated")
            }
            Err(_) => {
                tracing::error!(target: "alea::numerics", schema_version = 1_u64, event = "failed")
            }
        }
        outcome
    }
    fn new_inner(concentrations: Vec<f64>) -> Result<Self, E> {
        let d = concentrations.len();
        if !(2..=64).contains(&d) {
            return Err(E::Shape);
        }
        if concentrations.iter().any(|x| !x.is_finite()) {
            return Err(E::Parameter("concentrations"));
        }
        let shift = concentrations
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lambda = concentrations.iter().map(|x| x - shift).collect::<Vec<_>>();
        let spread = -lambda.iter().copied().fold(0.0_f64, f64::min);
        if spread > 10000.0 {
            return Err(E::Parameter("concentration spread"));
        }
        let (log_c, moments, steps) = holonomic(&lambda)?;
        let mut lo = 1.0;
        let mut hi = d as f64;
        for _ in 0..64 {
            let b = (lo + hi) * 0.5;
            let sum = lambda.iter().map(|l| 1.0 / (b - 2.0 * l)).sum::<f64>();
            if sum > 1.0 {
                lo = b;
            } else {
                hi = b;
            }
        }
        Ok(Self {
            lambda,
            log_c_shifted: area(d) + log_c,
            shift,
            moments,
            steps,
            envelope_b: (lo + hi) * 0.5,
        })
    }
    /// Log normalizing constant relative to sphere area, in the supplied concentration gauge.
    pub fn log_normalizer(&self) -> f64 {
        self.shift + self.log_c_shifted
    }
    /// `E[X_i²]`, obtained from the same holonomic solve as the normalizer.
    pub fn mean_squares(&self) -> &[f64] {
        &self.moments
    }
    /// Number of accepted ODE steps (zero for uniform or two-dimensional laws).
    pub fn normalizer_steps(&self) -> usize {
        self.steps
    }
    /// Riemannian event score in the tangent space of the unit sphere.
    /// # Errors
    /// Rejects non-unit events, shape mismatch and nonrepresentable output.
    pub fn tangent_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        if out.len() != self.lambda.len() {
            return Err(E::Shape);
        }
        if !sphere(x, self.lambda.len())? {
            return Err(E::Observation);
        }
        let q = self
            .lambda
            .iter()
            .zip(x)
            .map(|(l, x)| l * x * x)
            .sum::<f64>();
        for ((o, l), x) in out.iter_mut().zip(&self.lambda).zip(x) {
            *o = 2.0 * (l - q) * x;
        }
        if out.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
}
impl Distribution for Bingham {
    type Error = alea_core::distribution::DistributionError;
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.lambda.len())
    }
    fn support(&self) -> Support {
        Support::UnitSphere
    }
    fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
        alea_core::distribution::ReferenceMeasure::SphereSurface {
            ambient_dimension: self.lambda.len(),
        }
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        if !sphere(x, self.lambda.len())? {
            return Ok(f64::NEG_INFINITY);
        }
        checked(
            self.lambda
                .iter()
                .zip(x)
                .map(|(l, x)| l * x * x)
                .sum::<f64>()
                - self.log_c_shifted,
        )
    }
}
impl ParameterScore for Bingham {
    fn parameter_count(&self) -> usize {
        self.lambda.len()
    }
    fn parameter_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        if out.len() != self.parameter_count() {
            return Err(E::Shape);
        }
        if !sphere(x, self.lambda.len())? {
            return Err(E::Observation);
        }
        for ((o, x), m) in out.iter_mut().zip(x).zip(&self.moments) {
            *o = x * x - m;
        }
        Ok(())
    }
}
impl Sample for Bingham {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let d = self.lambda.len() as f64;
        let b = self.envelope_b;
        let bound = 0.5 * (b - d) + 0.5 * d * (d / b).ln();
        let mut x = vec![0.0; self.lambda.len()];
        for _ in 0..4096 {
            for (x, l) in x.iter_mut().zip(&self.lambda) {
                *x = standard_normal(rng) / (b - 2.0 * l).sqrt();
            }
            let norm = x.iter().fold(0.0_f64, |n, x| n.hypot(*x));
            if norm == 0.0 || !norm.is_finite() {
                return Err(E::Numerical);
            }
            for x in &mut x {
                *x /= norm;
            }
            let q = self
                .lambda
                .iter()
                .zip(&x)
                .map(|(l, x)| l * x * x)
                .sum::<f64>();
            let log_accept = q + 0.5 * d * (-2.0 * q / b).ln_1p() - bound;
            if log_accept > 1e-10 {
                return Err(E::Numerical);
            }
            if uniform(rng).ln() <= log_accept {
                return Ok(x);
            }
        }
        Err(E::SamplingBudget)
    }
}
// Radial Pfaffian system. With g_i=E[X_i² exp(t*lambda.X²)] under uniform sphere,
// g_i'=lambda_i*g_i+(sum(g)-d*g_i)/(2t). Normalizing g gives the moment equations
// below, while d/dt log C = sum(lambda_i*m_i). No divided eigenvalue differences.
fn rhs(t: f64, y: &[f64], lambda: &[f64]) -> Vec<f64> {
    let d = lambda.len();
    let mean = lambda.iter().zip(y).map(|(l, m)| l * m).sum::<f64>();
    let mut out = Vec::with_capacity(d + 1);
    for (i, l) in lambda.iter().enumerate() {
        out.push((l - mean) * y[i] + (1.0 - d as f64 * y[i]) / (2.0 * t));
    }
    out.push(mean);
    out
}
fn rk4(t: f64, h: f64, y: &[f64], lambda: &[f64]) -> Vec<f64> {
    let k1 = rhs(t, y, lambda);
    let temp = |k: &[f64], factor: f64| {
        y.iter()
            .zip(k)
            .map(|(y, k)| y + factor * h * k)
            .collect::<Vec<_>>()
    };
    let k2 = rhs(t + h * 0.5, &temp(&k1, 0.5), lambda);
    let k3 = rhs(t + h * 0.5, &temp(&k2, 0.5), lambda);
    let k4 = rhs(t + h, &temp(&k3, 1.0), lambda);
    (0..y.len())
        .map(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}
fn holonomic(lambda: &[f64]) -> Result<(f64, Vec<f64>, usize), E> {
    let d = lambda.len();
    let spread = -lambda.iter().copied().fold(0.0_f64, f64::min);
    if spread == 0.0 {
        return Ok((0.0, vec![1.0 / d as f64; d], 0));
    }
    // On S¹ a scaled Bessel evaluation is cheaper than continuation.
    if d == 2 {
        let signed_k = 0.5 * (lambda[0] - lambda[1]);
        let k = signed_k.abs();
        let i0 = spec_math::cephes64::i0e(k);
        let ratio = signed_k.signum() * spec_math::cephes64::i1e(k) / i0;
        return Ok((i0.ln(), vec![0.5 * (1.0 + ratio), 0.5 * (1.0 - ratio)], 0));
    }
    let mut t = (0.05 / spread).min(1.0);
    let r = lambda.iter().map(|l| t * l).collect::<Vec<_>>();
    let mut term = vec![1.0 / d as f64; d];
    let mut g = term.clone();
    // Taylor coefficients from the same system; |t*lambda|<=.05 makes 20 terms ample.
    for k in 0..20 {
        let sum = r.iter().zip(&term).map(|(r, a)| r * a).sum::<f64>() / (k + 1) as f64;
        for i in 0..d {
            term[i] = (r[i] * term[i] + 0.5 * sum) / ((k + 1) as f64 + 0.5 * d as f64);
            g[i] += term[i];
        }
    }
    let sum = g.iter().sum::<f64>();
    let mut y = g.iter().map(|g| g / sum).collect::<Vec<_>>();
    y.push(sum.ln());
    let mut h = t * 0.25;
    let mut steps = 0;
    for _ in 0..50000 {
        if t >= 1.0 {
            let moments = y[..d].to_vec();
            let sum = moments.iter().sum::<f64>();
            if moments.iter().any(|m| !m.is_finite() || *m <= 0.0)
                || (sum - 1.0).abs() > 1e-9
                || !y[d].is_finite()
            {
                return Err(E::Numerical);
            }
            return Ok((y[d], moments, steps));
        }
        h = h.min(1.0 - t);
        if t + h == t {
            return Err(E::Numerical);
        }
        let whole = rk4(t, h, &y, lambda);
        let half = rk4(t, h * 0.5, &y, lambda);
        let halves = rk4(t + h * 0.5, h * 0.5, &half, lambda);
        if whole.iter().chain(&halves).any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        let error = (0..=d)
            .map(|i| {
                ((halves[i] - whole[i]) / 15.0).abs()
                    / (2e-14 + 2e-12 * y[i].abs().max(halves[i].abs()))
            })
            .fold(0.0_f64, f64::max);
        if !error.is_finite() {
            return Err(E::Numerical);
        }
        if error <= 1.0 && halves[..d].iter().all(|m| *m > 0.0) {
            for i in 0..=d {
                y[i] = halves[i] + (halves[i] - whole[i]) / 15.0;
            }
            t += h;
            steps += 1;
        }
        let factor = if error == 0.0 {
            2.0
        } else {
            (0.9 * error.powf(-0.2)).clamp(0.1, 2.0)
        };
        h *= factor;
    }
    Err(E::Numerical)
}
