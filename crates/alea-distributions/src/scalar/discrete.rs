//! Count and finite-population laws. Integer parameters are fixed, not differentiated.
use super::{
    Binomial, Poisson, checked, finite, gamma_draw, positive, probability, score, uniform,
};
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, Support,
};
use rand::Rng;
use statrs::function::gamma::{digamma, ln_gamma};
const EXACT: u64 = 1 << 53;
fn count(n: u64) -> Result<f64, E> {
    if n <= EXACT {
        Ok(n as f64)
    } else {
        Err(E::Numerical)
    }
}
fn choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    if k < 64 {
        (1..=k)
            .map(|j| ((n - k + j) as f64).ln() - (j as f64).ln())
            .sum()
    } else {
        ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
    }
}
fn log_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}
macro_rules! count_law {
    ($name:ident,$n:literal,$doc:literal) => {
        #[doc=$doc]
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use]
        pub struct $name {
            a: f64,
            b: f64,
            n: u64,
        }
        impl Distribution for $name {
            type Event = u64;
            fn event_shape(&self) -> EventShape {
                EventShape::Scalar
            }
            fn support(&self) -> Support {
                Support::Count
            }
            fn log_prob(&self, k: &u64) -> Result<f64, E> {
                self.logp(*k)
            }
        }
        impl ParameterScore for $name {
            fn parameter_count(&self) -> usize {
                $n
            }
            fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
                let s = self.scores(*k)?;
                score(out, &s[..$n])
            }
        }
        impl Sample for $name {
            type Draw = u64;
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
                self.draw(rng)
            }
        }
    };
}
count_law!(
    Geometric,
    1,
    "Failures before the first success, support 0,1,... Score order: success probability in (0,1]."
);
impl Geometric {
    /// # Errors
    /// Requires a probability in (0,1].
    pub fn new(p: f64) -> Result<Self, E> {
        probability(p)?;
        positive(p, "success probability")?;
        Ok(Self { a: p, b: 0.0, n: 0 })
    }
    fn logp(&self, k: u64) -> Result<f64, E> {
        let k = count(k)?;
        Ok(if self.a == 1.0 {
            if k == 0.0 { 0.0 } else { f64::NEG_INFINITY }
        } else {
            self.a.ln() + k * (-self.a).ln_1p()
        })
    }
    fn scores(&self, k: u64) -> Result<[f64; 2], E> {
        let k = count(k)?;
        if self.a == 1.0 {
            return if k == 0.0 {
                Ok([1.0, 0.0])
            } else {
                Err(E::Observation)
            };
        }
        Ok([self.a.recip() - k / (1.0 - self.a), 0.0])
    }
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        if self.a == 1.0 {
            return Ok(0);
        }
        let x = (uniform(rng).ln() / (-self.a).ln_1p()).floor();
        if x.is_finite() && x < EXACT as f64 {
            Ok(x as u64)
        } else {
            Err(E::Numerical)
        }
    }
}
count_law!(
    NegativeBinomial,
    2,
    "Failures before shape successes, allowing real shape. Score order: shape, success probability; mean=shape*(1-p)/p."
);
impl NegativeBinomial {
    /// # Errors
    /// Requires positive shape and success probability in (0,1].
    pub fn new(shape: f64, p: f64) -> Result<Self, E> {
        positive(shape, "shape")?;
        probability(p)?;
        positive(p, "success probability")?;
        Ok(Self {
            a: shape,
            b: p,
            n: 0,
        })
    }
    fn logp(&self, k: u64) -> Result<f64, E> {
        let k = count(k)?;
        if self.b == 1.0 {
            return Ok(if k == 0.0 { 0.0 } else { f64::NEG_INFINITY });
        }
        checked(
            ln_gamma(self.a + k) - ln_gamma(self.a) - ln_gamma(k + 1.0)
                + self.a * self.b.ln()
                + k * (-self.b).ln_1p(),
        )
    }
    fn scores(&self, k: u64) -> Result<[f64; 2], E> {
        let k = count(k)?;
        if self.b == 1.0 {
            return if k == 0.0 {
                Ok([0.0, self.a])
            } else {
                Err(E::Observation)
            };
        }
        Ok([
            digamma(self.a + k) - digamma(self.a) + self.b.ln(),
            self.a / self.b - k / (1.0 - self.b),
        ])
    }
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        if self.b == 1.0 {
            return Ok(0);
        }
        let rate = gamma_draw(self.a, rng)? * ((1.0 - self.b) / self.b);
        Poisson::new(rate)?.sample(rng)
    }
}
/// Negative binomial parameterized by mean and positive inverse dispersion.
/// Score order: mean, inverse dispersion. Variance=mean+mean²/dispersion.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct NegativeBinomialMean {
    mean: f64,
    dispersion: f64,
}
impl NegativeBinomialMean {
    /// # Errors
    /// Requires finite positive mean and dispersion with representable success probability.
    pub fn new(mean: f64, dispersion: f64) -> Result<Self, E> {
        positive(mean, "mean")?;
        positive(dispersion, "dispersion")?;
        let p = 1.0 / (1.0 + mean / dispersion);
        if p == 0.0 || p == 1.0 {
            return Err(E::Numerical);
        }
        Ok(Self { mean, dispersion })
    }
}
impl Distribution for NegativeBinomialMean {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        let k = count(*k)?;
        let r = self.dispersion;
        let logsum = self.mean.ln().max(r.ln()) + (-(self.mean.ln() - r.ln()).abs()).exp().ln_1p();
        checked(
            ln_gamma(r + k) - ln_gamma(r) - ln_gamma(k + 1.0)
                + r * (r.ln() - logsum)
                + k * (self.mean.ln() - logsum),
        )
    }
}
impl ParameterScore for NegativeBinomialMean {
    fn parameter_count(&self) -> usize {
        2
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        let k = count(*k)?;
        let m = self.mean;
        let r = self.dispersion;
        score(
            out,
            &[
                k / m - (r + k) / (r + m),
                digamma(r + k) - digamma(r) + 1.0 - (1.0 + m / r).ln() - (r + k) / (r + m),
            ],
        )
    }
}
impl Sample for NegativeBinomialMean {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        let rate = gamma_draw(self.dispersion, rng)? * (self.mean / self.dispersion);
        Poisson::new(rate)?.sample(rng)
    }
}
count_law!(
    BetaBinomial,
    2,
    "Beta-binomial with fixed trials. Score order: alpha, beta."
);
impl BetaBinomial {
    /// # Errors
    /// Requires exact-range trials and finite positive shapes with finite normalization.
    pub fn new(trials: u64, alpha: f64, beta: f64) -> Result<Self, E> {
        count(trials)?;
        positive(alpha, "alpha")?;
        positive(beta, "beta")?;
        if !log_beta(alpha, beta).is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            a: alpha,
            b: beta,
            n: trials,
        })
    }
    fn logp(&self, k: u64) -> Result<f64, E> {
        if k > self.n {
            return Ok(f64::NEG_INFINITY);
        }
        checked(
            choose(self.n, k) + log_beta(k as f64 + self.a, (self.n - k) as f64 + self.b)
                - log_beta(self.a, self.b),
        )
    }
    fn scores(&self, k: u64) -> Result<[f64; 2], E> {
        if k > self.n {
            return Err(E::Observation);
        }
        let common = digamma(self.a + self.b) - digamma(self.n as f64 + self.a + self.b);
        Ok([
            digamma(k as f64 + self.a) - digamma(self.a) + common,
            digamma((self.n - k) as f64 + self.b) - digamma(self.b) + common,
        ])
    }
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        let a = gamma_draw(self.a, rng)?;
        let b = gamma_draw(self.b, rng)?;
        Binomial::new(self.n, a / (a + b))?.sample(rng)
    }
}
/// Hypergeometric: successes in `draws` without replacement from a finite population.
/// No continuous parameter scores: population, successes and draws are integer parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
}
impl Hypergeometric {
    /// # Errors
    /// Requires population <=2^53 and successes/draws no greater than population.
    pub fn new(population: u64, successes: u64, draws: u64) -> Result<Self, E> {
        count(population)?;
        if successes > population || draws > population {
            return Err(E::Parameter("population counts"));
        }
        Ok(Self {
            population,
            successes,
            draws,
        })
    }
}
impl Distribution for Hypergeometric {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        if *k > self.successes
            || *k > self.draws
            || self.draws - *k > self.population - self.successes
        {
            return Ok(f64::NEG_INFINITY);
        }
        checked(
            choose(self.successes, *k) + choose(self.population - self.successes, self.draws - *k)
                - choose(self.population, self.draws),
        )
    }
}
impl Sample for Hypergeometric {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        if self.draws > 1_000_000 {
            return Err(E::SamplingBudget);
        }
        let mut successes = self.successes;
        let mut total = 0;
        for i in 0..self.draws {
            if uniform(rng) < successes as f64 / (self.population - i) as f64 {
                total += 1;
                successes -= 1;
            }
        }
        Ok(total)
    }
}
/// Uniform on an inclusive signed integer interval. Bounds are fixed parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct DiscreteUniform {
    lower: i64,
    upper: i64,
}
impl DiscreteUniform {
    /// # Errors
    /// Requires lower<=upper; the entire i64 range is allowed.
    pub fn new(lower: i64, upper: i64) -> Result<Self, E> {
        if lower > upper {
            return Err(E::Parameter("integer interval"));
        }
        Ok(Self { lower, upper })
    }
}
impl Distribution for DiscreteUniform {
    type Event = i64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Integer
    }
    fn log_prob(&self, k: &i64) -> Result<f64, E> {
        Ok(if *k < self.lower || *k > self.upper {
            f64::NEG_INFINITY
        } else {
            -((self.upper as i128 - self.lower as i128 + 1) as f64).ln()
        })
    }
}
impl Sample for DiscreteUniform {
    type Draw = i64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<i64, E> {
        use rand::RngExt;
        let width = (self.upper as i128 - self.lower as i128 + 1) as u128;
        if width == 1_u128 << 64 {
            return Ok(rng.random::<u64>() as i64);
        }
        let width = width as u64;
        let threshold = width.wrapping_neg() % width;
        for _ in 0..4096 {
            let draw = rng.random::<u64>();
            if draw >= threshold {
                return Ok((self.lower as i128 + (draw % width) as i128) as i64);
            }
        }
        Err(E::SamplingBudget)
    }
}
/// Zipf on integers 1..=size. Score order: nonnegative exponent. Construction
/// caches its normalized CDF; size is limited to one million entries.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Zipf {
    exponent: f64,
    cdf: Vec<f64>,
    log_z: f64,
    mean_log: f64,
}
impl Zipf {
    /// # Errors
    /// Requires 1<=size<=1,000,000 and a finite nonnegative exponent.
    pub fn new(size: usize, exponent: f64) -> Result<Self, E> {
        if size == 0 || size > 1_000_000 {
            return Err(E::Parameter("size"));
        }
        finite(exponent, "exponent")?;
        if exponent < 0.0 {
            return Err(E::Parameter("exponent"));
        }
        let mut cdf = Vec::with_capacity(size);
        let mut sum = 0.0;
        let mut log_sum = 0.0;
        for k in 1..=size {
            let l = (k as f64).ln();
            let w = (-exponent * l).exp();
            sum += w;
            log_sum += w * l;
            cdf.push(sum);
        }
        for p in &mut cdf {
            *p /= sum;
        }
        cdf[size - 1] = 1.0;
        Ok(Self {
            exponent,
            cdf,
            log_z: sum.ln(),
            mean_log: log_sum / sum,
        })
    }
}
impl Distribution for Zipf {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        Ok(if *k == 0 || *k > self.cdf.len() as u64 {
            f64::NEG_INFINITY
        } else {
            -self.exponent * (*k as f64).ln() - self.log_z
        })
    }
}
impl ParameterScore for Zipf {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        if *k == 0 || *k > self.cdf.len() as u64 {
            return Err(E::Observation);
        }
        score(out, &[self.mean_log - (*k as f64).ln()])
    }
}
impl Sample for Zipf {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        let u = uniform(rng);
        Ok(self.cdf.partition_point(|p| *p <= u) as u64 + 1)
    }
}

/// Zeta (infinite Zipf) on positive integers. Score order: exponent >1.
/// Draws outside the exact binary64 integer range report a numerical error.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Zeta {
    exponent: f64,
    log_z: f64,
    log_z_slope: f64,
}
impl Zeta {
    /// # Errors
    /// Requires finite exponent >1 and representable normalization.
    pub fn new(exponent: f64) -> Result<Self, E> {
        finite(exponent, "exponent")?;
        if exponent <= 1.0 {
            return Err(E::Parameter("exponent"));
        }
        let s = super::jet::Jet::<1>::variable(exponent, 0);
        let mut z = super::jet::Jet::c(0.0);
        for n in 1..32 {
            z = z + (-s * (n as f64).ln()).exp();
        }
        if exponent <= 32.0 {
            let n = 32.0_f64;
            z = z + ((1.0 - s) * n.ln()).exp() / (s - 1.0) + 0.5 * (-s * n.ln()).exp();
            let coefficients = [
                1.0 / 12.0,
                -1.0 / 720.0,
                1.0 / 30240.0,
                -1.0 / 1209600.0,
                1.0 / 47900160.0,
                -691.0 / 1307674368000.0,
                7.0 / 523069747200.0,
                -3617.0 / 10670622842880000.0,
            ];
            let mut rising = s;
            for (i, c) in coefficients.into_iter().enumerate() {
                let k = 2 * i + 1;
                if i > 0 {
                    rising = rising * (s + (k - 2) as f64) * (s + (k - 1) as f64);
                }
                z = z + c * rising * ((-s - k as f64) * n.ln()).exp();
            }
        }
        if !z.v.is_finite() || z.v < 1.0 || !z.d[0].is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self {
            exponent,
            log_z: z.v.ln(),
            log_z_slope: z.d[0] / z.v,
        })
    }
}
impl Distribution for Zeta {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        if *k == 0 {
            return Ok(f64::NEG_INFINITY);
        }
        let k = count(*k)?;
        Ok(-self.exponent * k.ln() - self.log_z)
    }
}
impl ParameterScore for Zeta {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        if *k == 0 {
            return Err(E::Observation);
        }
        score(out, &[-count(*k)?.ln() - self.log_z_slope])
    }
}
impl Sample for Zeta {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        let a = self.exponent - 1.0;
        let b = a * 2.0_f64.ln();
        let log_expm1 = |x: f64| {
            if x > 40.0 {
                x + (-(-x).exp()).ln_1p()
            } else {
                x.exp_m1().ln()
            }
        };
        for _ in 0..4096 {
            let log_x = -uniform(rng).ln() / a;
            if log_x >= (EXACT as f64).ln() {
                return Err(E::Numerical);
            }
            let x = log_x.exp().floor();
            let t = a * x.recip().ln_1p();
            // Devroye's rejection with logarithms to avoid overflow in 2^(s-1).
            if uniform(rng).ln() + x.ln() + log_expm1(t) - log_expm1(b) <= t - b {
                return Ok(x as u64);
            }
        }
        Err(E::SamplingBudget)
    }
}
/// Poisson-binomial: sum of independent Bernoulli trials with unequal probabilities.
/// The constructor caches an exact convolution in binary64; no Gaussian approximation.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PoissonBinomial {
    probabilities: Vec<f64>,
    mass: Vec<f64>,
}
impl PoissonBinomial {
    /// # Errors
    /// Requires at most 4096 finite probabilities in `[0,1]`. Empty trials give a point mass at zero.
    pub fn new(probabilities: Vec<f64>) -> Result<Self, E> {
        if probabilities.len() > 4096 {
            return Err(E::Parameter("trial count"));
        }
        for p in &probabilities {
            probability(*p)?;
        }
        let mut mass = vec![0.0; probabilities.len() + 1];
        mass[0] = 1.0;
        for (i, p) in probabilities.iter().enumerate() {
            for k in (1..=i + 1).rev() {
                mass[k] = mass[k] * (1.0 - p) + mass[k - 1] * p;
            }
            mass[0] *= 1.0 - p;
        }
        Ok(Self {
            probabilities,
            mass,
        })
    }
}
impl Distribution for PoissonBinomial {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        Ok(usize::try_from(*k)
            .ok()
            .and_then(|k| self.mass.get(k))
            .copied()
            .unwrap_or(0.0)
            .ln())
    }
}
impl Sample for PoissonBinomial {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        Ok(self
            .probabilities
            .iter()
            .filter(|p| uniform(rng) < **p)
            .count() as u64)
    }
}

/// Beta-negative-binomial. Failures before `shape` successes, with a beta-distributed
/// success probability. Score order: shape, alpha, beta. Shape need not be integer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BetaNegativeBinomial {
    shape: f64,
    alpha: f64,
    beta: f64,
}
impl BetaNegativeBinomial {
    /// # Errors
    /// Requires positive finite shape and beta concentrations with finite normalization.
    pub fn new(shape: f64, alpha: f64, beta: f64) -> Result<Self, E> {
        positive(shape, "shape")?;
        positive(alpha, "alpha")?;
        positive(beta, "beta")?;
        if !log_beta(alpha, beta).is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self { shape, alpha, beta })
    }
}
impl Distribution for BetaNegativeBinomial {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        let k = count(*k)?;
        let r = self.shape;
        checked(
            ln_gamma(k + r) - ln_gamma(r) - ln_gamma(k + 1.0)
                + log_beta(self.alpha + r, self.beta + k)
                - log_beta(self.alpha, self.beta),
        )
    }
}
impl ParameterScore for BetaNegativeBinomial {
    fn parameter_count(&self) -> usize {
        3
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        let k = count(*k)?;
        let r = self.shape;
        let a = self.alpha;
        let b = self.beta;
        let s = digamma(a + b + r + k);
        score(
            out,
            &[
                digamma(k + r) - digamma(r) + digamma(a + r) - s,
                digamma(a + r) - s - digamma(a) + digamma(a + b),
                digamma(b + k) - s - digamma(b) + digamma(a + b),
            ],
        )
    }
}
impl Sample for BetaNegativeBinomial {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        let a = gamma_draw(self.alpha, rng)?;
        let b = gamma_draw(self.beta, rng)?;
        NegativeBinomial::new(self.shape, a / (a + b))?.sample(rng)
    }
}
