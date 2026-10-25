//! Normalized vector, discrete and correlation-Cholesky laws.
//! Matrix events use full row-major storage. Cholesky densities are with respect
//! to free lower coordinates, not covariance-matrix Lebesgue measure.
use crate::scalar::{checked, gamma_draw, positive, score, standard_normal, uniform};
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, Support,
};
use rand::Rng;
use statrs::function::gamma::{digamma, ln_gamma};
use std::{cell::RefCell, f64::consts::PI};
fn simplex(x: &[f64]) -> Result<bool, E> {
    if x.iter().any(|v| v.is_nan()) {
        return Err(E::Observation);
    }
    Ok(x.iter().all(|v| v.is_finite() && *v > 0.0) && (x.iter().sum::<f64>() - 1.0).abs() <= 1e-12)
}
/// Multivariate Normal parameterized by a checked lower covariance Cholesky factor.
/// Parameter score order: mean, then packed lower Cholesky in row-major order.
#[derive(Debug)]
#[must_use]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    lower: Vec<f64>,
    log_det: f64,
    work: RefCell<Vec<f64>>,
}
impl MultivariateNormal {
    /// # Errors
    /// Rejects empty/mismatched storage, non-finite entries, nonzero upper entries
    /// and non-positive diagonals. No covariance repair is performed.
    pub fn new(mean: Vec<f64>, lower: Vec<f64>) -> Result<Self, E> {
        let d = mean.len();
        if d == 0 || d.checked_mul(d) != Some(lower.len()) {
            return Err(E::Shape);
        }
        if mean.iter().chain(&lower).any(|x| !x.is_finite()) {
            return Err(E::Parameter("mean or Cholesky"));
        }
        let mut log_det = 0.0;
        for i in 0..d {
            positive(lower[i * d + i], "Cholesky diagonal")?;
            log_det += lower[i * d + i].ln();
            for j in i + 1..d {
                if lower[i * d + j] != 0.0 {
                    return Err(E::Parameter("upper Cholesky"));
                }
            }
        }
        if !log_det.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            mean,
            lower,
            log_det,
            work: RefCell::new(vec![0.0; 2 * d]),
        })
    }
    fn whiten(&self, x: &[f64], z: &mut [f64]) -> Result<(), E> {
        let d = self.mean.len();
        if x.len() != d {
            return Err(E::Shape);
        }
        if x.iter().any(|x| !x.is_finite()) {
            return Err(E::Observation);
        }
        for i in 0..d {
            let mut v = x[i] - self.mean[i];
            for (j, z) in z.iter().enumerate().take(i) {
                v -= self.lower[i * d + j] * z;
            }
            z[i] = v / self.lower[i * d + i];
        }
        Ok(())
    }
    fn transpose_solve(&self, z: &[f64], out: &mut [f64]) {
        let d = self.mean.len();
        for i in (0..d).rev() {
            let mut v = z[i];
            for (j, x) in out.iter().enumerate().skip(i + 1) {
                v -= self.lower[j * d + i] * x;
            }
            out[i] = v / self.lower[i * d + i];
        }
    }
    /// Ambient event score in mean coordinates.
    /// # Errors
    /// Rejects shape, invalid events or numerical/workspace failures.
    pub fn event_gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        let d = self.mean.len();
        if out.len() != d {
            return Err(E::Shape);
        }
        let mut work = self.work.try_borrow_mut().map_err(|_| E::Numerical)?;
        let (z, v) = work.split_at_mut(d);
        self.whiten(x, z)?;
        self.transpose_solve(z, v);
        for x in v.iter_mut() {
            *x = -*x;
        }
        score(out, v)
    }
}
impl Distribution for MultivariateNormal {
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.mean.len())
    }
    fn support(&self) -> Support {
        Support::RealVector
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        let mut w = self.work.try_borrow_mut().map_err(|_| E::Numerical)?;
        let z = &mut w[..self.mean.len()];
        self.whiten(x, z)?;
        checked(
            -0.5 * self.mean.len() as f64 * (2.0 * PI).ln()
                - self.log_det
                - 0.5 * z.iter().map(|x| x * x).sum::<f64>(),
        )
    }
}
impl ParameterScore for MultivariateNormal {
    fn parameter_count(&self) -> usize {
        self.mean.len() + self.mean.len() * (self.mean.len() + 1) / 2
    }
    fn parameter_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        let d = self.mean.len();
        if out.len() != self.parameter_count() {
            return Err(E::Shape);
        }
        let mut w = self.work.try_borrow_mut().map_err(|_| E::Numerical)?;
        let (z, v) = w.split_at_mut(d);
        self.whiten(x, z)?;
        self.transpose_solve(z, v);
        out.fill(0.0);
        out[..d].copy_from_slice(v);
        let mut slot = d;
        for (i, velocity) in v.iter().enumerate() {
            for (j, whitened) in z.iter().enumerate().take(i + 1) {
                out[slot] = velocity * whitened
                    - if i == j {
                        self.lower[i * d + i].recip()
                    } else {
                        0.0
                    };
                slot += 1;
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            Err(E::Numerical)
        } else {
            Ok(())
        }
    }
}
impl Sample for MultivariateNormal {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let d = self.mean.len();
        let z = (0..d).map(|_| standard_normal(rng)).collect::<Vec<_>>();
        let mut x = self.mean.clone();
        for (i, x) in x.iter_mut().enumerate() {
            for (j, z) in z.iter().enumerate().take(i + 1) {
                *x += self.lower[i * d + j] * z;
            }
        }
        if x.iter().any(|x| !x.is_finite()) {
            Err(E::Numerical)
        } else {
            Ok(x)
        }
    }
}
/// Dirichlet on a strictly positive simplex, score order: concentration vector.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Dirichlet {
    alpha: Vec<f64>,
    normalizer: f64,
    sum: f64,
}
impl Dirichlet {
    /// # Errors
    /// Rejects dimensions below two, invalid concentrations or unrepresentable normalization.
    pub fn new(alpha: Vec<f64>) -> Result<Self, E> {
        if alpha.len() < 2 {
            return Err(E::Shape);
        }
        for a in &alpha {
            positive(*a, "concentration")?;
        }
        let sum = alpha.iter().sum::<f64>();
        let normalizer = ln_gamma(sum) - alpha.iter().map(|a| ln_gamma(*a)).sum::<f64>();
        if !normalizer.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            alpha,
            normalizer,
            sum,
        })
    }
    /// Ambient score; a simplex transform supplies the tangent pullback.
    /// # Errors
    /// Rejects mismatched or non-interior simplex events.
    pub fn event_gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        if x.len() != self.alpha.len() || out.len() != x.len() {
            return Err(E::Shape);
        }
        if !simplex(x)? {
            return Err(E::Observation);
        }
        for ((v, a), x) in out.iter_mut().zip(&self.alpha).zip(x) {
            *v = (a - 1.0) / x;
        }
        if out.iter().any(|x| !x.is_finite()) {
            Err(E::Numerical)
        } else {
            Ok(())
        }
    }
}
impl Distribution for Dirichlet {
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.alpha.len())
    }
    fn support(&self) -> Support {
        Support::Simplex
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        if x.len() != self.alpha.len() {
            return Err(E::Shape);
        }
        if !simplex(x)? {
            return Ok(f64::NEG_INFINITY);
        }
        checked(
            self.normalizer
                + self
                    .alpha
                    .iter()
                    .zip(x)
                    .map(|(a, x)| (a - 1.0) * x.ln())
                    .sum::<f64>(),
        )
    }
}
impl ParameterScore for Dirichlet {
    fn parameter_count(&self) -> usize {
        self.alpha.len()
    }
    fn parameter_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        if x.len() != self.alpha.len() || out.len() != x.len() {
            return Err(E::Shape);
        }
        if !simplex(x)? {
            return Err(E::Observation);
        }
        for ((v, a), x) in out.iter_mut().zip(&self.alpha).zip(x) {
            *v = digamma(self.sum) - digamma(*a) + x.ln();
        }
        if out.iter().any(|x| !x.is_finite()) {
            Err(E::Numerical)
        } else {
            Ok(())
        }
    }
}
impl Sample for Dirichlet {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let mut x = self
            .alpha
            .iter()
            .map(|a| gamma_draw(*a, rng))
            .collect::<Result<Vec<_>, _>>()?;
        let sum = x.iter().sum::<f64>();
        for x in &mut x {
            *x /= sum;
        }
        if !simplex(&x)? {
            Err(E::Numerical)
        } else {
            Ok(x)
        }
    }
}
/// Categorical with an explicitly normalized probability vector; event is a zero-based index.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Categorical {
    probabilities: Vec<f64>,
}
impl Categorical {
    /// # Errors
    /// Rejects empty, negative, non-finite or non-normalized weights. Zero weights are valid.
    pub fn new(probabilities: Vec<f64>) -> Result<Self, E> {
        if probabilities.is_empty()
            || probabilities.iter().any(|x| !x.is_finite() || *x < 0.0)
            || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(E::Parameter("probabilities"));
        }
        Ok(Self { probabilities })
    }
}
impl Distribution for Categorical {
    type Event = usize;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Finite {
            outcomes: self.probabilities.len(),
        }
    }
    fn log_prob(&self, x: &usize) -> Result<f64, E> {
        Ok(self.probabilities.get(*x).copied().unwrap_or(0.0).ln())
    }
}
impl Sample for Categorical {
    type Draw = usize;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<usize, E> {
        let u = uniform(rng);
        let mut total = 0.0;
        for (i, p) in self.probabilities.iter().enumerate() {
            total += p;
            if u < total {
                return Ok(i);
            }
        }
        self.probabilities
            .iter()
            .rposition(|x| *x > 0.0)
            .ok_or(E::Numerical)
    }
}
impl ParameterScore for Categorical {
    fn parameter_count(&self) -> usize {
        self.probabilities.len()
    }
    fn parameter_score(&self, event: &usize, output: &mut [f64]) -> Result<(), E> {
        if output.len() != self.parameter_count() {
            return Err(E::Shape);
        }
        let p = self
            .probabilities
            .get(*event)
            .filter(|p| **p > 0.0)
            .ok_or(E::Observation)?;
        output.fill(0.0);
        output[*event] = p.recip();
        if output[*event].is_finite() {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
}
/// Multinomial with fixed trials and categorical probabilities.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Multinomial {
    trials: u64,
    categorical: Categorical,
}
impl Multinomial {
    /// # Errors
    /// Rejects trials outside exact integer precision or invalid probabilities.
    pub fn new(trials: u64, p: Vec<f64>) -> Result<Self, E> {
        if trials > 1 << 53 {
            return Err(E::Parameter("trials"));
        }
        Ok(Self {
            trials,
            categorical: Categorical::new(p)?,
        })
    }
}
impl Distribution for Multinomial {
    type Event = [u64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.categorical.probabilities.len())
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, x: &[u64]) -> Result<f64, E> {
        if x.len() != self.categorical.probabilities.len() {
            return Err(E::Shape);
        }
        if x.iter().try_fold(0u64, |a, x| a.checked_add(*x)) != Some(self.trials) {
            return Ok(f64::NEG_INFINITY);
        }
        let mut v = ln_gamma(self.trials as f64 + 1.0);
        for (k, p) in x.iter().zip(&self.categorical.probabilities) {
            v -= ln_gamma(*k as f64 + 1.0);
            if *k != 0 {
                v += *k as f64 * p.ln();
            }
        }
        checked(v)
    }
}
impl Sample for Multinomial {
    type Draw = Vec<u64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<u64>, E> {
        if self.trials > 1_000_000 {
            return Err(E::SamplingBudget);
        }
        let mut x = vec![0; self.categorical.probabilities.len()];
        for _ in 0..self.trials {
            x[self.categorical.sample(rng)?] += 1;
        }
        Ok(x)
    }
}
impl ParameterScore for Multinomial {
    fn parameter_count(&self) -> usize {
        self.categorical.probabilities.len()
    }
    fn parameter_score(&self, event: &[u64], output: &mut [f64]) -> Result<(), E> {
        if output.len() != self.parameter_count() || event.len() != output.len() {
            return Err(E::Shape);
        }
        if event.iter().try_fold(0u64, |sum, n| sum.checked_add(*n)) != Some(self.trials) {
            return Err(E::Observation);
        }
        for ((g, n), p) in output
            .iter_mut()
            .zip(event)
            .zip(&self.categorical.probabilities)
        {
            *g = if *n == 0 { 0.0 } else { *n as f64 / p };
        }
        if output.iter().all(|g| g.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
}
/// Normalized LKJ Cholesky law relative to the free strict-lower coordinates.
/// The positive diagonal is the derived row norm, not an independent coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LkjCholesky {
    dimension: usize,
    eta: f64,
    normalizer: f64,
}
impl LkjCholesky {
    /// # Errors
    /// Rejects dimension below two, storage overflow, invalid eta or normalization.
    pub fn new(dimension: usize, eta: f64) -> Result<Self, E> {
        positive(eta, "eta")?;
        if dimension < 2 || dimension.checked_mul(dimension).is_none() {
            return Err(E::Shape);
        }
        let mut n = 0.0;
        for i in 1..dimension {
            let b = eta + (dimension - i - 1) as f64 * 0.5;
            n += ln_gamma(b + i as f64 * 0.5) - ln_gamma(b) - i as f64 * 0.5 * PI.ln();
        }
        if !n.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            dimension,
            eta,
            normalizer: n,
        })
    }
    fn valid(&self, l: &[f64]) -> Result<bool, E> {
        let d = self.dimension;
        if l.len() != d * d {
            return Err(E::Shape);
        }
        if l.iter().any(|x| x.is_nan()) {
            return Err(E::Observation);
        }
        for i in 0..d {
            if l[i * d + i] <= 0.0 || !l[i * d + i].is_finite() {
                return Ok(false);
            }
            let norm = l[i * d..i * d + i + 1].iter().map(|x| x * x).sum::<f64>();
            if !norm.is_finite()
                || (norm - 1.0).abs() > 1e-12
                || l[i * d + i + 1..(i + 1) * d].iter().any(|x| *x != 0.0)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Ambient full-matrix derivative; a row-norm transform supplies its pullback.
    /// # Errors
    /// Rejects invalid Cholesky support, shape or numerical output.
    pub fn event_gradient(&self, l: &[f64], out: &mut [f64]) -> Result<(), E> {
        if out.len() != l.len() {
            return Err(E::Shape);
        }
        if !self.valid(l)? {
            return Err(E::Observation);
        }
        out.fill(0.0);
        for i in 1..self.dimension {
            out[i * self.dimension + i] = (self.dimension as f64 - i as f64 - 3.0 + 2.0 * self.eta)
                / l[i * self.dimension + i];
        }
        if out.iter().any(|x| !x.is_finite()) {
            Err(E::Numerical)
        } else {
            Ok(())
        }
    }
}
impl Distribution for LkjCholesky {
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        EventShape::Matrix {
            rows: self.dimension,
            columns: self.dimension,
        }
    }
    fn support(&self) -> Support {
        Support::CorrelationCholesky
    }
    fn log_prob(&self, l: &[f64]) -> Result<f64, E> {
        if !self.valid(l)? {
            return Ok(f64::NEG_INFINITY);
        }
        let mut value = self.normalizer;
        for i in 1..self.dimension {
            value += (self.dimension as f64 - i as f64 - 3.0 + 2.0 * self.eta)
                * l[i * self.dimension + i].ln();
        }
        checked(value)
    }
}
impl ParameterScore for LkjCholesky {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, l: &[f64], out: &mut [f64]) -> Result<(), E> {
        if !self.valid(l)? {
            return Err(E::Observation);
        }
        let mut value = 0.0;
        for i in 1..self.dimension {
            let b = self.eta + (self.dimension - i - 1) as f64 * 0.5;
            value +=
                digamma(b + i as f64 * 0.5) - digamma(b) + 2.0 * l[i * self.dimension + i].ln();
        }
        score(out, &[value])
    }
}
impl Sample for LkjCholesky {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let d = self.dimension;
        let mut l = vec![0.0; d * d];
        l[0] = 1.0;
        for i in 1..d {
            let a = gamma_draw(i as f64 * 0.5, rng)?;
            let b = gamma_draw(self.eta + (d - i - 1) as f64 * 0.5, rng)?;
            let r2 = a / (a + b);
            let row = &mut l[i * d..i * d + i];
            for x in row.iter_mut() {
                *x = standard_normal(rng);
            }
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in row {
                *x *= r2.sqrt() / norm;
            }
            l[i * d + i] = (1.0 - r2).sqrt();
        }
        if !self.valid(&l)? {
            Err(E::Numerical)
        } else {
            Ok(l)
        }
    }
}
