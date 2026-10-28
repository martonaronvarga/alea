//! Vector and covariance laws, using full row-major matrix storage.
use super::{Dirichlet, LkjCholesky, Multinomial, MultivariateNormal};
use crate::scalar::{checked, gamma_draw, positive, score, standard_normal};
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, Support,
};
use rand::Rng;
use statrs::function::gamma::{digamma, ln_gamma};
use std::f64::consts::PI;

/// Multivariate Student-t. The supplied lower Cholesky factor describes the
/// scale matrix, NOT covariance; covariance=df/(df-2)*scale when df>2.
/// Score order: df, mean, packed lower Cholesky in row-major order.
#[derive(Debug)]
#[must_use]
pub struct MultivariateStudentT {
    df: f64,
    normal: MultivariateNormal,
    normalizer: f64,
}
impl MultivariateStudentT {
    /// # Errors
    /// Requires positive finite df and the same mean/Cholesky conditions as MultivariateNormal.
    pub fn new(df: f64, mean: Vec<f64>, lower: Vec<f64>) -> Result<Self, E> {
        positive(df, "df")?;
        let normal = MultivariateNormal::new(mean, lower)?;
        let d = normal.mean.len() as f64;
        let normalizer = ln_gamma((df + d) * 0.5)
            - ln_gamma(df * 0.5)
            - 0.5 * d * (df.ln() + PI.ln())
            - normal.log_det;
        if !normalizer.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            df,
            normal,
            normalizer,
        })
    }
    /// Event gradient in ambient vector coordinates.
    /// # Errors
    /// Rejects shape, invalid events and nonrepresentable scores.
    pub fn event_gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        let d = self.normal.mean.len();
        if out.len() != d {
            return Err(E::Shape);
        }
        let mut w = self
            .normal
            .work
            .try_borrow_mut()
            .map_err(|_| E::Numerical)?;
        let (z, v) = w.split_at_mut(d);
        self.normal.whiten(x, z)?;
        self.normal.transpose_solve(z, v);
        let q = z.iter().map(|z| z * z).sum::<f64>();
        let weight = (self.df + d as f64) / (self.df + q);
        for o in v.iter_mut() {
            *o *= -weight;
        }
        score(out, v)
    }
}
impl Distribution for MultivariateStudentT {
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        self.normal.event_shape()
    }
    fn support(&self) -> Support {
        Support::RealVector
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        let d = self.normal.mean.len();
        let mut w = self
            .normal
            .work
            .try_borrow_mut()
            .map_err(|_| E::Numerical)?;
        let z = &mut w[..d];
        self.normal.whiten(x, z)?;
        let norm = z.iter().fold(self.df.sqrt(), |n, z| n.hypot(*z));
        checked(self.normalizer - (self.df + d as f64) * (norm.ln() - 0.5 * self.df.ln()))
    }
}
impl ParameterScore for MultivariateStudentT {
    fn parameter_count(&self) -> usize {
        1 + self.normal.parameter_count()
    }
    fn parameter_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        if out.len() != self.parameter_count() {
            return Err(E::Shape);
        }
        let d = self.normal.mean.len();
        let mut w = self
            .normal
            .work
            .try_borrow_mut()
            .map_err(|_| E::Numerical)?;
        let (z, v) = w.split_at_mut(d);
        self.normal.whiten(x, z)?;
        self.normal.transpose_solve(z, v);
        let q = z.iter().map(|z| z * z).sum::<f64>();
        if !q.is_finite() {
            return Err(E::Numerical);
        }
        let df = self.df;
        let weight = (df + d as f64) / (df + q);
        out[0] = 0.5
            * (digamma((df + d as f64) * 0.5)
                - digamma(df * 0.5)
                - d as f64 / df
                - (q / df).ln_1p()
                + weight * q / df);
        for i in 0..d {
            out[1 + i] = weight * v[i];
        }
        let mut slot = 1 + d;
        for (i, vi) in v.iter().enumerate() {
            for (j, zj) in z.iter().enumerate().take(i + 1) {
                out[slot] = weight * vi * zj
                    - if i == j {
                        self.normal.lower[i * d + i].recip()
                    } else {
                        0.0
                    };
                slot += 1;
            }
        }
        if out.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
}
impl Sample for MultivariateStudentT {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let mut x = self.normal.sample(rng)?;
        let scale = (self.df / (2.0 * gamma_draw(self.df * 0.5, rng)?)).sqrt();
        for (x, m) in x.iter_mut().zip(&self.normal.mean) {
            *x = m + (*x - m) * scale;
        }
        if x.iter().all(|x| x.is_finite()) {
            Ok(x)
        } else {
            Err(E::Numerical)
        }
    }
}

/// Dirichlet-multinomial for a fixed total count. Score order: concentration vector.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DirichletMultinomial {
    trials: u64,
    alpha: Vec<f64>,
    sum: f64,
    normalizer: f64,
}
impl DirichletMultinomial {
    /// # Errors
    /// Requires >=2 positive finite concentrations, trials<=2^53 and representable normalization.
    pub fn new(trials: u64, alpha: Vec<f64>) -> Result<Self, E> {
        let d = Dirichlet::new(alpha)?;
        if trials > 1 << 53 {
            return Err(E::Parameter("trials"));
        }
        let normalizer = ln_gamma(trials as f64 + 1.0) + ln_gamma(d.sum)
            - ln_gamma(d.sum + trials as f64)
            - d.alpha.iter().map(|a| ln_gamma(*a)).sum::<f64>();
        if !normalizer.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            trials,
            alpha: d.alpha,
            sum: d.sum,
            normalizer,
        })
    }
    fn valid(&self, x: &[u64]) -> Result<bool, E> {
        if x.len() != self.alpha.len() {
            return Err(E::Shape);
        }
        Ok(x.iter().try_fold(0u64, |a, k| a.checked_add(*k)) == Some(self.trials))
    }
}
impl Distribution for DirichletMultinomial {
    type Event = [u64];
    fn event_shape(&self) -> EventShape {
        EventShape::Vector(self.alpha.len())
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, x: &[u64]) -> Result<f64, E> {
        if !self.valid(x)? {
            return Ok(f64::NEG_INFINITY);
        }
        checked(
            self.normalizer
                + x.iter()
                    .zip(&self.alpha)
                    .map(|(k, a)| ln_gamma(*k as f64 + a) - ln_gamma(*k as f64 + 1.0))
                    .sum::<f64>(),
        )
    }
}
impl ParameterScore for DirichletMultinomial {
    fn parameter_count(&self) -> usize {
        self.alpha.len()
    }
    fn parameter_score(&self, x: &[u64], out: &mut [f64]) -> Result<(), E> {
        if out.len() != self.alpha.len() {
            return Err(E::Shape);
        }
        if !self.valid(x)? {
            return Err(E::Observation);
        }
        let common = digamma(self.sum) - digamma(self.sum + self.trials as f64);
        for ((g, k), a) in out.iter_mut().zip(x).zip(&self.alpha) {
            *g = common + digamma(a + *k as f64) - digamma(*a);
        }
        if out.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
}
impl Sample for DirichletMultinomial {
    type Draw = Vec<u64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<u64>, E> {
        let p = Dirichlet::new(self.alpha.clone())?.sample(rng)?;
        Multinomial::new(self.trials, p)?.sample(rng)
    }
}

// Matrix helpers deliberately avoid repairing asymmetric or indefinite events.
fn cholesky(x: &[f64], d: usize) -> Result<Option<Vec<f64>>, E> {
    if x.len() != d * d {
        return Err(E::Shape);
    }
    if x.iter().any(|v| v.is_nan()) {
        return Err(E::Observation);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    for i in 0..d {
        for j in 0..i {
            if x[i * d + j] != x[j * d + i] {
                return Ok(None);
            }
        }
    }
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut v = x[i * d + j];
            for k in 0..j {
                v -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if v <= 0.0 {
                    return Ok(None);
                }
                l[i * d + j] = v.sqrt();
            } else {
                l[i * d + j] = v / l[j * d + j];
            }
            if !l[i * d + j].is_finite() {
                return Err(E::Numerical);
            }
        }
    }
    Ok(Some(l))
}
fn lower_inverse(l: &[f64], d: usize) -> Result<Vec<f64>, E> {
    let mut a = vec![0.0; d * d];
    for j in 0..d {
        for i in j..d {
            let mut x = if i == j { 1.0 } else { 0.0 };
            for k in j..i {
                x -= l[i * d + k] * a[k * d + j];
            }
            a[i * d + j] = x / l[i * d + i];
        }
    }
    if a.iter().all(|v| v.is_finite()) {
        Ok(a)
    } else {
        Err(E::Numerical)
    }
}
fn transpose(a: &[f64], d: usize) -> Vec<f64> {
    (0..d * d).map(|i| a[(i % d) * d + i / d]).collect()
}
fn product(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            c[i * d + j] = (0..d).map(|k| a[i * d + k] * b[k * d + j]).sum();
        }
    }
    c
}
fn covariance(l: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let x = (0..d).map(|k| l[i * d + k] * l[j * d + k]).sum();
            c[i * d + j] = x;
            c[j * d + i] = x;
        }
    }
    c
}
fn inverse(l: &[f64], d: usize) -> Result<Vec<f64>, E> {
    let a = lower_inverse(l, d)?;
    Ok(product(&transpose(&a, d), &a, d))
}
fn logdet(l: &[f64], d: usize) -> f64 {
    2.0 * (0..d).map(|i| l[i * d + i].ln()).sum::<f64>()
}
fn multigamma(df: f64, d: usize) -> f64 {
    d as f64 * (d - 1) as f64 * 0.25 * PI.ln()
        + (0..d).map(|i| ln_gamma((df - i as f64) * 0.5)).sum::<f64>()
}
#[derive(Debug, Clone, PartialEq)]
struct MatrixLaw {
    df: f64,
    d: usize,
    lower: Vec<f64>,
    scale: Vec<f64>,
    scale_inverse: Vec<f64>,
    logdet: f64,
    normalizer: f64,
    inverted: bool,
}
impl MatrixLaw {
    fn new(df: f64, dimension: usize, lower: Vec<f64>, inverted: bool) -> Result<Self, E> {
        positive(df, "df")?;
        if dimension == 0 || df <= dimension as f64 - 1.0 {
            return Err(E::Parameter("df must exceed dimension-1"));
        }
        if dimension.checked_mul(dimension) != Some(lower.len()) {
            return Err(E::Shape);
        }
        let normal = MultivariateNormal::new(vec![0.0; dimension], lower)?;
        let lower = normal.lower;
        let d = dimension;
        let determinant = logdet(&lower, d);
        let sign = if inverted { 1.0 } else { -1.0 };
        let normalizer =
            sign * 0.5 * df * determinant - 0.5 * df * d as f64 * 2.0_f64.ln() - multigamma(df, d);
        if !normalizer.is_finite() {
            return Err(E::Numerical);
        }
        let scale = covariance(&lower, d);
        let scale_inverse = inverse(&lower, d)?;
        if scale.iter().chain(&scale_inverse).any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(Self {
            df,
            d,
            lower,
            scale,
            scale_inverse,
            logdet: determinant,
            normalizer,
            inverted,
        })
    }
    fn logp(&self, x: &[f64]) -> Result<f64, E> {
        let Some(l) = cholesky(x, self.d)? else {
            return Ok(f64::NEG_INFINITY);
        };
        let (a, b) = if self.inverted {
            (&l, &self.lower)
        } else {
            (&self.lower, &l)
        };
        let whitened = product(&lower_inverse(a, self.d)?, b, self.d);
        let trace = whitened.iter().map(|v| v * v).sum::<f64>();
        let power = if self.inverted {
            -(self.df + self.d as f64 + 1.0)
        } else {
            self.df - self.d as f64 - 1.0
        };
        checked(self.normalizer + 0.5 * power * logdet(&l, self.d) - 0.5 * trace)
    }
    fn scores(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        let d = self.d;
        if out.len() != 1 + d * (d + 1) / 2 {
            return Err(E::Shape);
        }
        let l = cholesky(x, d)?.ok_or(E::Observation)?;
        let sign = if self.inverted { -1.0 } else { 1.0 };
        out[0] = 0.5
            * (sign * (logdet(&l, d) - self.logdet)
                - d as f64 * 2.0_f64.ln()
                - (0..d)
                    .map(|i| digamma((self.df - i as f64) * 0.5))
                    .sum::<f64>());
        let mut g = if self.inverted {
            let inv = inverse(&l, d)?;
            self.scale_inverse
                .iter()
                .zip(inv)
                .map(|(s, x)| self.df * s - x)
                .collect::<Vec<_>>()
        } else {
            let sxs = product(&product(&self.scale_inverse, x, d), &self.scale_inverse, d);
            sxs.iter()
                .zip(&self.scale_inverse)
                .map(|(x, s)| x - self.df * s)
                .collect()
        };
        g = product(&g, &self.lower, d);
        let mut slot = 1;
        for i in 0..d {
            for j in 0..=i {
                out[slot] = g[i * d + j];
                slot += 1;
            }
        }
        if out.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
    fn gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        let d = self.d;
        if out.len() != d * d {
            return Err(E::Shape);
        }
        let l = cholesky(x, d)?.ok_or(E::Observation)?;
        let inv = inverse(&l, d)?;
        let g = if self.inverted {
            let a = product(&product(&inv, &self.scale, d), &inv, d);
            a.iter()
                .zip(&inv)
                .map(|(a, b)| 0.5 * (a - (self.df + d as f64 + 1.0) * b))
                .collect::<Vec<_>>()
        } else {
            inv.iter()
                .zip(&self.scale_inverse)
                .map(|(a, b)| 0.5 * ((self.df - d as f64 - 1.0) * a - b))
                .collect()
        };
        score(out, &g)
    }
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let d = self.d;
        let mut a = vec![0.0; d * d];
        for i in 0..d {
            a[i * d + i] = (2.0 * gamma_draw((self.df - i as f64) * 0.5, rng)?).sqrt();
            for j in 0..i {
                a[i * d + j] = standard_normal(rng);
            }
        }
        let l = if self.inverted {
            product(&self.lower, &transpose(&lower_inverse(&a, d)?, d), d)
        } else {
            product(&self.lower, &a, d)
        };
        let x = covariance(&l, d);
        if cholesky(&x, d)?.is_some() {
            Ok(x)
        } else {
            Err(E::Numerical)
        }
    }
}
macro_rules! matrix_law {
    ($name:ident,$inverted:literal,$doc:literal) => {
        #[doc=$doc]
        #[derive(Debug, Clone, PartialEq)]
        #[must_use]
        pub struct $name(MatrixLaw);
        impl $name {
            /// # Errors
            /// Requires dimension>0, df>dimension-1 and a finite lower Cholesky scale factor
            /// with positive diagonal. Rejects unrepresentable normalization or inverse.
            pub fn new(df: f64, dimension: usize, scale_cholesky: Vec<f64>) -> Result<Self, E> {
                Ok(Self(MatrixLaw::new(
                    df,
                    dimension,
                    scale_cholesky,
                    $inverted,
                )?))
            }
            /// Full symmetric ambient matrix gradient. An off-diagonal free coordinate
            /// receives the sum of its two mirrored entries.
            /// # Errors
            /// Rejects mismatched shapes, non-SPD events and nonrepresentable scores.
            pub fn event_gradient(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
                self.0.gradient(x, out)
            }
        }
        impl Distribution for $name {
            type Event = [f64];
            fn event_shape(&self) -> EventShape {
                EventShape::Matrix {
                    rows: self.0.d,
                    columns: self.0.d,
                }
            }
            fn support(&self) -> Support {
                Support::PositiveDefinite
            }
            fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
                self.0.logp(x)
            }
        }
        impl ParameterScore for $name {
            fn parameter_count(&self) -> usize {
                1 + self.0.d * (self.0.d + 1) / 2
            }
            fn parameter_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
                self.0.scores(x, out)
            }
        }
        impl Sample for $name {
            type Draw = Vec<f64>;
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
                self.0.draw(rng)
            }
        }
    };
}
matrix_law!(
    Wishart,
    false,
    "Wishart W(df,S) on SPD matrices with E[X]=df*S. Density is with respect to independent symmetric entries, not Cholesky entries. Score order: df, packed lower scale Cholesky. Matrix evaluation uses temporary dense storage."
);
matrix_law!(
    InverseWishart,
    true,
    "Inverse Wishart IW(df,Psi); E[X]=Psi/(df-dimension-1) when defined. Density uses symmetric-matrix measure. Score order: df, packed lower scale Cholesky. Evaluation uses temporary dense storage."
);

/// LKJ density on correlation matrices, with respect to free off-diagonal entries.
/// Score order: concentration eta. Unlike LkjCholesky, eta=1 is a constant density.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Lkj {
    cholesky: LkjCholesky,
}
impl Lkj {
    /// # Errors
    /// Requires dimension>=2 and positive finite eta with finite normalization.
    pub fn new(dimension: usize, eta: f64) -> Result<Self, E> {
        Ok(Self {
            cholesky: LkjCholesky::new(dimension, eta)?,
        })
    }
    fn factor(&self, x: &[f64]) -> Result<Option<Vec<f64>>, E> {
        let d = self.cholesky.dimension;
        if x.len() != d * d {
            return Err(E::Shape);
        }
        if x.iter().any(|v| v.is_nan()) {
            return Err(E::Observation);
        }
        if (0..d).any(|i| x[i * d + i] != 1.0) {
            return Ok(None);
        }
        cholesky(x, d)
    }
}
impl Distribution for Lkj {
    type Event = [f64];
    fn event_shape(&self) -> EventShape {
        self.cholesky.event_shape()
    }
    fn support(&self) -> Support {
        Support::CorrelationMatrix
    }
    fn log_prob(&self, x: &[f64]) -> Result<f64, E> {
        let Some(l) = self.factor(x)? else {
            return Ok(f64::NEG_INFINITY);
        };
        checked(
            self.cholesky.normalizer
                + (self.cholesky.eta - 1.0) * logdet(&l, self.cholesky.dimension),
        )
    }
}
impl ParameterScore for Lkj {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, x: &[f64], out: &mut [f64]) -> Result<(), E> {
        let l = self.factor(x)?.ok_or(E::Observation)?;
        self.cholesky.parameter_score(&l, out)
    }
}
impl Sample for Lkj {
    type Draw = Vec<f64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, E> {
        let l = self.cholesky.sample(rng)?;
        let mut x = covariance(&l, self.cholesky.dimension);
        for i in 0..self.cholesky.dimension {
            x[i * self.cholesky.dimension + i] = 1.0;
        }
        if self.factor(&x)?.is_some() {
            Ok(x)
        } else {
            Err(E::Numerical)
        }
    }
}
