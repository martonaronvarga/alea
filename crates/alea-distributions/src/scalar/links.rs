//! Stable probability laws in regression link coordinates.
use super::{Bernoulli, Binomial, Poisson, finite, score, uniform};
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, Support,
};
use rand::Rng;
use statrs::function::gamma::ln_gamma;
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
/// Bernoulli in log-odds coordinates. Score order: log odds; bool events.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BernoulliLogit {
    logit: f64,
}
impl BernoulliLogit {
    /// # Errors
    /// Requires finite log odds.
    pub fn new(logit: f64) -> Result<Self, E> {
        Ok(Self {
            logit: finite(logit, "log odds")?,
        })
    }
}
impl Distribution for BernoulliLogit {
    type Event = bool;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Finite { outcomes: 2 }
    }
    fn log_prob(&self, x: &bool) -> Result<f64, E> {
        Bernoulli::logit_log_prob(self.logit, *x)
    }
}
impl ParameterScore for BernoulliLogit {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, x: &bool, out: &mut [f64]) -> Result<(), E> {
        score(
            out,
            &[if *x {
                sigmoid(-self.logit)
            } else {
                -sigmoid(self.logit)
            }],
        )
    }
}
impl Sample for BernoulliLogit {
    type Draw = bool;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<bool, E> {
        let u = uniform(rng);
        Ok(u.ln() - (-u).ln_1p() < self.logit)
    }
}
/// Binomial with fixed trials and log odds. Score order: log odds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BinomialLogit {
    trials: u64,
    logit: f64,
}
impl BinomialLogit {
    /// # Errors
    /// Requires trials<=2^53 and finite log odds.
    pub fn new(trials: u64, logit: f64) -> Result<Self, E> {
        if trials > 1 << 53 {
            return Err(E::Parameter("trials"));
        }
        Ok(Self {
            trials,
            logit: finite(logit, "log odds")?,
        })
    }
}
impl Distribution for BinomialLogit {
    type Event = u64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Count
    }
    fn log_prob(&self, k: &u64) -> Result<f64, E> {
        if *k > self.trials {
            return Ok(f64::NEG_INFINITY);
        }
        let n = self.trials as f64;
        let k = *k as f64;
        super::checked(
            ln_gamma(n + 1.0)
                - ln_gamma(k + 1.0)
                - ln_gamma(n - k + 1.0)
                - k * softplus(-self.logit)
                - (n - k) * softplus(self.logit),
        )
    }
}
impl ParameterScore for BinomialLogit {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        if *k > self.trials {
            return Err(E::Observation);
        }
        score(
            out,
            &[*k as f64 * sigmoid(-self.logit) - (self.trials - *k) as f64 * sigmoid(self.logit)],
        )
    }
}
impl Sample for BinomialLogit {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        Binomial::new(self.trials, sigmoid(self.logit))?.sample(rng)
    }
}
/// Poisson in log-rate coordinates. Score order: log rate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PoissonLog {
    log_rate: f64,
}
impl PoissonLog {
    /// # Errors
    /// Requires finite log rate; sampling additionally requires a representable rate.
    pub fn new(log_rate: f64) -> Result<Self, E> {
        Ok(Self {
            log_rate: finite(log_rate, "log rate")?,
        })
    }
}
impl Distribution for PoissonLog {
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
        super::checked(*k as f64 * self.log_rate - self.log_rate.exp() - ln_gamma(*k as f64 + 1.0))
    }
}
impl ParameterScore for PoissonLog {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, k: &u64, out: &mut [f64]) -> Result<(), E> {
        if *k > 1 << 53 {
            return Err(E::Numerical);
        }
        score(out, &[*k as f64 - self.log_rate.exp()])
    }
}
impl Sample for PoissonLog {
    type Draw = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<u64, E> {
        let rate = self.log_rate.exp();
        if rate == 0.0 || !rate.is_finite() {
            return Err(E::Numerical);
        }
        Poisson::new(rate)?.sample(rng)
    }
}
