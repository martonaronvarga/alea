//! Normalized scalar laws. Scale/rate conventions and score order are explicit.
//! Evaluation and derivatives allocate nothing. Sampling uses bounded work.
use alea_core::distribution::{
    ContinuousDistribution, Distribution, DistributionError as E, EventShape, ParameterScore,
    Sample, Support,
};
use rand::{Rng, RngExt};
use statrs::function::gamma::{digamma, ln_gamma};
use std::f64::consts::{PI, TAU};

const LN_TAU: f64 = 1.8378770664093453;
pub(crate) fn positive(x: f64, name: &'static str) -> Result<f64, E> {
    if x.is_finite() && x > 0.0 {
        Ok(x)
    } else {
        Err(E::Parameter(name))
    }
}
fn finite(x: f64, name: &'static str) -> Result<f64, E> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(E::Parameter(name))
    }
}
pub(crate) fn checked(x: f64) -> Result<f64, E> {
    if x.is_nan() { Err(E::Numerical) } else { Ok(x) }
}
pub(crate) fn score(output: &mut [f64], values: &[f64]) -> Result<(), E> {
    if output.len() != values.len() {
        return Err(E::Shape);
    }
    if values.iter().any(|x| !x.is_finite()) {
        return Err(E::Numerical);
    }
    output.copy_from_slice(values);
    Ok(())
}
pub(crate) fn uniform<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    // Exactly representable open dyadic grid, avoiding both logarithmic endpoints.
    let bits = rng.random::<u64>() >> 12;
    (bits as f64 + 0.5) * (1.0 / 4503599627370496.0)
}
pub(crate) fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    (-2.0 * uniform(rng).ln()).sqrt() * (TAU * uniform(rng)).cos()
}
pub(crate) fn gamma_draw<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> Result<f64, E> {
    let a = if shape < 1.0 { shape + 1.0 } else { shape };
    let d = a - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    for _ in 0..4096 {
        let z = standard_normal(rng);
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = uniform(rng);
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            let value = d
                * v
                * if shape < 1.0 {
                    (uniform(rng).ln() / shape).exp()
                } else {
                    1.0
                };
            return if value > 0.0 && value.is_finite() {
                Ok(value)
            } else {
                Err(E::Numerical)
            };
        }
    }
    Err(E::SamplingBudget)
}
fn power_log(power: f64, x: f64) -> f64 {
    if power == 0.0 { 0.0 } else { power * x.ln() }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Law {
    Normal,
    Student,
    LogNormal,
    Exponential,
    Gamma,
    Beta,
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scalar {
    law: Law,
    a: f64,
    b: f64,
    c: f64,
}
impl Scalar {
    fn logp(self, x: f64) -> Result<f64, E> {
        if x.is_nan() {
            return Err(E::Observation);
        }
        let Self { law, a, b, c } = self;
        let value = match law {
            Law::Normal => {
                let z = (x - a) / b;
                -0.5 * z * z - b.ln() - 0.5 * LN_TAU
            }
            Law::Student => {
                if x.is_infinite() {
                    return Ok(f64::NEG_INFINITY);
                }
                let z = (x - b) / c;
                ln_gamma((a + 1.0) * 0.5)
                    - ln_gamma(a * 0.5)
                    - 0.5 * (a.ln() + PI.ln())
                    - c.ln()
                    - (a + 1.0) * (z.hypot(a.sqrt()).ln() - 0.5 * a.ln())
            }
            Law::LogNormal => {
                if x <= 0.0 || x == f64::INFINITY {
                    return Ok(f64::NEG_INFINITY);
                }
                let z = (x.ln() - a) / b;
                -0.5 * z * z - b.ln() - 0.5 * LN_TAU - x.ln()
            }
            Law::Exponential => {
                if x < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    a.ln() - a * x
                }
            }
            Law::Gamma => {
                if x < 0.0 || x == f64::INFINITY {
                    return Ok(f64::NEG_INFINITY);
                }
                a * b.ln() - ln_gamma(a) + power_log(a - 1.0, x) - b * x
            }
            Law::Beta => {
                if !(0.0..=1.0).contains(&x) {
                    return Ok(f64::NEG_INFINITY);
                }
                ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
                    + power_log(a - 1.0, x)
                    + power_log(b - 1.0, 1.0 - x)
            }
        };
        checked(value)
    }
    fn scores(self, x: f64) -> Result<(f64, [f64; 3], usize), E> {
        if !x.is_finite() {
            return Err(E::Observation);
        }
        let Self { law, a, b, c } = self;
        let result = match law {
            Law::Normal => {
                let z = (x - a) / b;
                (-z / b, [z / b, (z * z - 1.0) / b, 0.0], 2)
            }
            Law::LogNormal => {
                if x <= 0.0 {
                    return Err(E::Observation);
                }
                let z = (x.ln() - a) / b;
                (-(1.0 + z / b) / x, [z / b, (z * z - 1.0) / b, 0.0], 2)
            }
            Law::Student => {
                let z = (x - b) / c;
                let norm = z.hypot(a.sqrt());
                let w = (a.sqrt() / norm).powi(2);
                let location = (a + 1.0) * (z / norm) / norm / c;
                let nu = 0.5
                    * (digamma((a + 1.0) * 0.5)
                        - digamma(a * 0.5)
                        - a.recip()
                        - 2.0 * (norm.ln() - 0.5 * a.ln())
                        + (a + 1.0) / a * (1.0 - w));
                (
                    -location,
                    [nu, location, ((a + 1.0) * (1.0 - w) - 1.0) / c],
                    3,
                )
            }
            Law::Exponential => {
                if x < 0.0 {
                    return Err(E::Observation);
                }
                (-a, [a.recip() - x, 0.0, 0.0], 1)
            }
            Law::Gamma => {
                if x <= 0.0 {
                    return Err(E::Observation);
                }
                (
                    (a - 1.0) / x - b,
                    [b.ln() - digamma(a) + x.ln(), a / b - x, 0.0],
                    2,
                )
            }
            Law::Beta => {
                if x <= 0.0 || x >= 1.0 {
                    return Err(E::Observation);
                }
                (
                    (a - 1.0) / x - (b - 1.0) / (1.0 - x),
                    [
                        digamma(a + b) - digamma(a) + x.ln(),
                        digamma(a + b) - digamma(b) + (-x).ln_1p(),
                        0.0,
                    ],
                    2,
                )
            }
        };
        Ok(result)
    }
    fn draw<R: Rng + ?Sized>(self, rng: &mut R) -> Result<f64, E> {
        let Self { law, a, b, c } = self;
        let x = match law {
            Law::Normal => a + b * standard_normal(rng),
            Law::LogNormal => (a + b * standard_normal(rng)).exp(),
            Law::Student => {
                b + c * standard_normal(rng) / (2.0 * gamma_draw(a * 0.5, rng)? / a).sqrt()
            }
            Law::Exponential => -uniform(rng).ln() / a,
            Law::Gamma => gamma_draw(a, rng)? / b,
            Law::Beta => {
                let x = gamma_draw(a, rng)?;
                let y = gamma_draw(b, rng)?;
                x / (x + y)
            }
        };
        if !x.is_finite()
            || (matches!(law, Law::LogNormal | Law::Gamma) && x <= 0.0)
            || (law == Law::Beta && !(0.0 < x && x < 1.0))
        {
            Err(E::Numerical)
        } else {
            Ok(x)
        }
    }
}
macro_rules! continuous {
    ($name:ident, $support:ident, $doc:literal) => {
        #[doc=$doc]
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use]
        pub struct $name(Scalar);
        impl Distribution for $name {
            type Event = f64;
            fn event_shape(&self) -> EventShape {
                EventShape::Scalar
            }
            fn support(&self) -> Support {
                Support::$support
            }
            fn log_prob(&self, x: &f64) -> Result<f64, E> {
                self.0.logp(*x)
            }
        }
        impl ParameterScore for $name {
            fn parameter_count(&self) -> usize {
                match self.0.law {
                    Law::Student => 3,
                    Law::Exponential => 1,
                    _ => 2,
                }
            }
            fn parameter_score(&self, x: &f64, out: &mut [f64]) -> Result<(), E> {
                let (_, v, n) = self.0.scores(*x)?;
                score(out, &v[..n])
            }
        }
        impl ContinuousDistribution for $name {
            fn event_score(&self, x: f64) -> Result<f64, E> {
                let (s, _, _) = self.0.scores(x)?;
                if s.is_finite() {
                    Ok(s)
                } else {
                    Err(E::Numerical)
                }
            }
        }
        impl Sample for $name {
            type Draw = f64;
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64, E> {
                self.0.draw(rng)
            }
        }
    };
}
continuous!(
    Normal,
    Real,
    "Normalized Normal; parameter score order: location, positive scale."
);
continuous!(
    StudentT,
    Real,
    "Student-t; score order: degrees of freedom, location, scale."
);
continuous!(
    LogNormal,
    Positive,
    "LogNormal; score order: log location, log scale."
);
continuous!(
    Exponential,
    Positive,
    "Exponential with positive rate; score order: rate."
);
continuous!(
    Gamma,
    Positive,
    "Gamma with shape/rate parameterization; score order: shape, rate."
);
continuous!(
    Beta,
    UnitInterval,
    "Beta; score order: alpha, beta. Boundary density may be infinite."
);
macro_rules! location_scale {
    ($name:ident,$law:ident) => {
        impl $name {
            /// # Errors
            /// Rejects non-finite location and non-positive/non-finite scale.
            pub fn new(location: f64, scale: f64) -> Result<Self, E> {
                Ok(Self(Scalar {
                    law: Law::$law,
                    a: finite(location, "location")?,
                    b: positive(scale, "scale")?,
                    c: 0.0,
                }))
            }
        }
    };
}
location_scale!(Normal, Normal);
location_scale!(LogNormal, LogNormal);
impl StudentT {
    /// # Errors
    /// Rejects invalid degrees of freedom, location or scale.
    pub fn new(df: f64, location: f64, scale: f64) -> Result<Self, E> {
        positive(df, "degrees of freedom")?;
        finite(location, "location")?;
        positive(scale, "scale")?;
        let s = Self(Scalar {
            law: Law::Student,
            a: df,
            b: location,
            c: scale,
        });
        if !s.log_prob(&location)?.is_finite() {
            return Err(E::Numerical);
        }
        Ok(s)
    }
}
impl Exponential {
    /// # Errors
    /// Rejects non-positive/non-finite rate.
    pub fn new(rate: f64) -> Result<Self, E> {
        Ok(Self(Scalar {
            law: Law::Exponential,
            a: positive(rate, "rate")?,
            b: 0.0,
            c: 0.0,
        }))
    }
}
macro_rules! shapes {
    ($name:ident,$law:ident,$first:ident,$second:ident) => {
        impl $name {
            /// # Errors
            /// Rejects non-positive/non-finite parameters or unrepresentable normalization.
            pub fn new($first: f64, $second: f64) -> Result<Self, E> {
                let a = positive($first, stringify!($first))?;
                let b = positive($second, stringify!($second))?;
                let normalizer = if matches!(Law::$law, Law::Gamma) {
                    a * b.ln() - ln_gamma(a)
                } else {
                    ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
                };
                if !normalizer.is_finite() {
                    return Err(E::Numerical);
                }
                Ok(Self(Scalar {
                    law: Law::$law,
                    a,
                    b,
                    c: 0.0,
                }))
            }
        }
    };
}
shapes!(Gamma, Gamma, shape, rate);
shapes!(Beta, Beta, alpha, beta);

/// Bernoulli on boolean events; score order: success probability.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Bernoulli {
    p: f64,
}
fn probability(p: f64) -> Result<f64, E> {
    if p.is_finite() && (0.0..=1.0).contains(&p) {
        Ok(p)
    } else {
        Err(E::Parameter("probability"))
    }
}
impl Bernoulli {
    /// # Errors
    /// Rejects a probability outside the closed unit interval.
    pub fn new(p: f64) -> Result<Self, E> {
        Ok(Self { p: probability(p)? })
    }
    /// Stable normalized log probability parameterized by a finite log odds.
    /// # Errors
    /// Rejects non-finite log odds.
    pub fn logit_log_prob(logit: f64, event: bool) -> Result<f64, E> {
        finite(logit, "log odds")?;
        let z = if event { -logit } else { logit };
        Ok(-z.max(0.0) - (-z.abs()).exp().ln_1p())
    }
}
impl Distribution for Bernoulli {
    type Event = bool;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Finite { outcomes: 2 }
    }
    fn log_prob(&self, x: &bool) -> Result<f64, E> {
        Ok(if *x { self.p.ln() } else { (-self.p).ln_1p() })
    }
}
impl ParameterScore for Bernoulli {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, x: &bool, out: &mut [f64]) -> Result<(), E> {
        score(
            out,
            &[if *x {
                self.p.recip()
            } else {
                -(1.0 - self.p).recip()
            }],
        )
    }
}
impl Sample for Bernoulli {
    type Draw = bool;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<bool, E> {
        Ok(uniform(rng) < self.p)
    }
}
/// Binomial on integer counts; score order: success probability. Trials are fixed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Binomial {
    n: u64,
    p: f64,
}
impl Binomial {
    /// # Errors
    /// Rejects invalid probability or trial counts above the exact f64 integer range.
    pub fn new(n: u64, p: f64) -> Result<Self, E> {
        if n > 1 << 53 {
            return Err(E::Parameter("trials"));
        }
        Ok(Self {
            n,
            p: probability(p)?,
        })
    }
}
impl Distribution for Binomial {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        if *k > self.n {
            return Ok(f64::NEG_INFINITY);
        }
        if self.n == 0 {
            return Ok(0.0);
        }
        if self.p == 0.0 {
            return Ok(if *k == 0 { 0.0 } else { f64::NEG_INFINITY });
        }
        if self.p == 1.0 {
            return Ok(if *k == self.n { 0.0 } else { f64::NEG_INFINITY });
        }
        let n = self.n as f64;
        let k = *k as f64;
        checked(
            ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + power_log(k, self.p)
                + power_log(n - k, 1.0 - self.p),
        )
    }
}
impl ParameterScore for Binomial {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        if *k > self.n {
            return Err(E::Observation);
        }
        score(
            out,
            &[*k as f64 / self.p - (self.n - *k) as f64 / (1.0 - self.p)],
        )
    }
}
impl Sample for Binomial {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        if self.n > 1_000_000 {
            return Err(E::SamplingBudget);
        }
        Ok((0..self.n).filter(|_| uniform(rng) < self.p).count() as u64)
    }
}
/// Poisson on integer counts; score order: positive rate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Poisson {
    rate: f64,
}
impl Poisson {
    /// # Errors
    /// Rejects non-positive/non-finite rates.
    pub fn new(rate: f64) -> Result<Self, E> {
        Ok(Self {
            rate: positive(rate, "rate")?,
        })
    }
}
impl Distribution for Poisson {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        if *k > 1 << 53 {
            return Err(E::Numerical);
        }
        let k = *k as f64;
        checked(k * self.rate.ln() - self.rate - ln_gamma(k + 1.0))
    }
}
impl ParameterScore for Poisson {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        if *k > 1 << 53 {
            return Err(E::Numerical);
        }
        score(out, &[*k as f64 / self.rate - 1.0])
    }
}
impl Sample for Poisson {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        // Sum independent rate<=30 Poissons; bounded exact inversion, no Gaussian approximation.
        if self.rate > 30_000.0 {
            return Err(E::SamplingBudget);
        }
        let chunks = ((self.rate / 30.0).ceil() as usize).max(1);
        let rate = self.rate / chunks as f64;
        let mut total = 0;
        for _ in 0..chunks {
            let mut product = 1.0;
            let threshold = (-rate).exp();
            let mut found = false;
            for k in 0..4096 {
                product *= uniform(rng);
                if product <= threshold {
                    total += k;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(E::SamplingBudget);
            }
        }
        Ok(total)
    }
}
