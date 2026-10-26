//! Bounded first-exit simulation without a time mesh.
//!
//! From the center of (-d,d), propose the one-sided hitting time with drift
//! |v| (inverse Gaussian, or Levy at v=0). In dimensionless time s=t/d²,
//! accept with A(s)=sum_{j>=0} (-1)^j(2j+1)exp(-2j(j+1)/s).
//! For s>1/2 use its eigenfunction transform, with prefactor
//! pi sqrt(2pi) s^(3/2) exp(1/(2s)-pi²s/8)/4 and exponent
//! -j(j+1)pi²s/2. Both alternating sequences decrease from their first term
//! on the respective domains. Partial sums bracket acceptance. The rejection
//! envelope is 1+exp(-2|v|d), so mathematical acceptance is at least 1/2.
//!
//! Exit direction is independent of this time, with upper probability
//! logistic(2vd). Repeat on the largest centered subinterval inside (0,alpha)
//! until hitting an original boundary. Strong Markov composition gives the
//! original diffusion. There is no Euler step or silently truncated trajectory.
//! Floating-point/RNG error is not interval-certified. Unresolved comparisons,
//! nonrepresentable states and bounded-work exhaustion are explicit errors.
use super::*;
use crate::scalar::{standard_normal, uniform};
use rand::Rng;

const MAX_EXITS: usize = 4096;
const MAX_PROPOSALS: usize = 256;
const MAX_TERMS: usize = 64;

fn numerical() -> ProbError {
    ProbError::NumericalError("Wiener simulation exhausted its numerical or work limits".into())
}

fn accept(s: f64, u: f64) -> Result<bool> {
    if !(s.is_finite() && s > 0.) {
        return Err(numerical());
    }
    let (log_pref, rate) = if s <= 0.5 {
        (0., 2. / s)
    } else {
        (
            (PI * (2. * PI).sqrt() / 4.).ln() + 1.5 * s.ln() + 0.5 / s - PI * PI * s / 8.,
            PI * PI * s / 2.,
        )
    };
    // The scaled alternating sum is <= 1. Compare in logs before exponentiating
    // a long-time tail; underflow must never turn a possible acceptance into zero.
    let roundoff = 64. * f64::EPSILON;
    if u.ln() > log_pref + roundoff {
        return Ok(false);
    }
    let target = (u.ln() - log_pref).exp();
    let (mut lower, mut upper, mut sum) = (0., 1., 1.);
    for j in 1..=MAX_TERMS {
        let n = j as f64;
        let term = (2. * n + 1.) * (-n * (n + 1.) * rate).exp();
        if j % 2 == 1 {
            sum -= term;
            lower = sum;
        } else {
            sum += term;
            upper = sum;
        }
        if target < lower - roundoff {
            return Ok(true);
        }
        if target > upper + roundoff {
            return Ok(false);
        }
    }
    Err(numerical())
}

fn centered_time<R: Rng + ?Sized>(eta: f64, rng: &mut R) -> Result<f64> {
    let drift = eta.abs();
    for _ in 0..MAX_PROPOSALS {
        let z = standard_normal(rng);
        let y = z * z;
        // Stable smaller inverse-Gaussian root: no subtraction of near equals.
        let s = if drift == 0. {
            y.recip()
        } else {
            let small = (drift + 0.5 * y + (drift * y + 0.25 * y * y).sqrt()).recip();
            if uniform(rng) < (1. + drift * small).recip() {
                small
            } else {
                (drift * (drift * small)).recip()
            }
        };
        if accept(s, uniform(rng))? {
            return Ok(s);
        }
    }
    Err(numerical())
}

impl Wiener4 {
    /// Draw a joint reaction time and boundary, with unit diffusion variance.
    ///
    /// Uses bounded rejection of one-sided hitting times on centered intervals,
    /// not time discretization. `tau` is added once at the end. Floating-point
    /// accuracy is not a certified distributional error bound.
    ///
    /// # Errors
    /// Rejects invalid parameters, unrepresentable times/states, unresolved
    /// acceptance comparisons, or exhaustion of 4,096 interval exits, 256
    /// proposals per exit, or 64 alternating-series terms per proposal.
    pub fn sample<R: Rng + ?Sized>(
        &self,
        params: &Wiener4Params,
        rng: &mut R,
    ) -> Result<WienerObservation> {
        if !params.valid() {
            return Err(ProbError::InvalidParameters(
                "Wiener simulation requires valid parameters".into(),
            ));
        }
        let mut position = params.alpha * params.beta;
        let mut elapsed = 0.;
        for _ in 0..MAX_EXITS {
            let upper_distance = params.alpha - position;
            let distance = position.min(upper_distance);
            let eta = params.delta * distance;
            if !(distance > 0. && eta.is_finite()) {
                return Err(numerical());
            }
            let s = centered_time(eta, rng)?;
            let increment = distance * (distance * s);
            if !(increment.is_finite() && increment > 0.) {
                return Err(numerical());
            }
            elapsed += increment;
            let upper_probability = if eta >= 0. {
                (1. + (-2. * eta).exp()).recip()
            } else {
                let e = (2. * eta).exp();
                e / (1. + e)
            };
            let upper = uniform(rng) < upper_probability;
            if (upper && upper_distance <= position) || (!upper && position <= upper_distance) {
                let rt = params.tau + elapsed;
                if !(rt.is_finite() && rt > params.tau) {
                    return Err(numerical());
                }
                return Ok(WienerObservation {
                    rt,
                    boundary: if upper {
                        Boundary::Upper
                    } else {
                        Boundary::Lower
                    },
                });
            }
            let next = if upper {
                position + distance
            } else {
                position - distance
            };
            if next == position || !(next > 0. && next < params.alpha) {
                return Err(numerical());
            }
            position = next;
        }
        Err(numerical())
    }
}

impl Wiener5 {
    /// Draw normal drift variability once per observation, then simulate Wiener4.
    /// # Errors
    /// Returns parameter or bounded-simulation errors, including nonfinite drift.
    pub fn sample<R: Rng + ?Sized>(
        &self,
        params: &Wiener5Params,
        rng: &mut R,
    ) -> Result<WienerObservation> {
        if !params.base.valid() || !params.s_delta.is_finite() || params.s_delta < 0. {
            return Err(ProbError::InvalidParameters(
                "Wiener simulation requires valid drift variability".into(),
            ));
        }
        let mut base = params.base;
        if params.s_delta > 0. {
            base.delta += params.s_delta * standard_normal(rng);
        }
        if !base.delta.is_finite() {
            return Err(numerical());
        }
        Wiener4.sample(&base, rng)
    }
}

impl Wiener7 {
    /// Draw independent uniform starting point/nondecision time and normal drift,
    /// once per trial. Uniform widths follow the likelihood's conventions.
    /// # Errors
    /// Rejects invalid variability intervals and returns bounded-simulation errors.
    pub fn sample<R: Rng + ?Sized>(
        &self,
        params: &Wiener7Params,
        rng: &mut R,
    ) -> Result<WienerObservation> {
        if !params.base.base.valid()
            || !params.base.s_delta.is_finite()
            || params.base.s_delta < 0.
            || !params.s_beta.is_finite()
            || params.s_beta < 0.
            || !params.s_tau.is_finite()
            || params.s_tau < 0.
            || params.base.base.beta - 0.5 * params.s_beta <= 0.
            || params.base.base.beta + 0.5 * params.s_beta >= 1.
        {
            return Err(ProbError::InvalidParameters(
                "Wiener simulation requires valid variability intervals".into(),
            ));
        }
        let mut base = params.base;
        if params.s_beta > 0. {
            base.base.beta += params.s_beta * (uniform(rng) - 0.5);
        }
        if params.s_tau > 0. {
            base.base.tau += params.s_tau * uniform(rng);
        }
        if !base.base.valid() {
            return Err(numerical());
        }
        Wiener5.sample(&base, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acceptance_representations_agree_with_direct_image_sum() {
        for s in [0.1, 0.3, 0.499, 0.5, 0.501, 0.7, 1., 2., 5.] {
            let probability: f64 = (0..128)
                .map(|j| {
                    let n = j as f64;
                    let term = (2. * n + 1.) * (-2. * n * (n + 1.) / s).exp();
                    if j % 2 == 0 { term } else { -term }
                })
                .sum();
            for delta in [-1e-8, 1e-8] {
                let u = probability + delta;
                if u > 0. && u < 1. {
                    assert_eq!(accept(s, u).unwrap(), delta < 0.);
                }
            }
        }
        assert!(accept(f64::INFINITY, 0.5).is_err());
        assert!(accept(0., 0.5).is_err());
    }
}
