//! Boundary subdistribution functions from integrated image/spectral series.
//! Truncation tests concern real arithmetic, not floating-point enclosures.
use super::*;

fn log_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        m
    } else {
        m + (-(a - b).abs()).exp().ln_1p()
    }
}
fn log_sub(a: f64, b: f64) -> Option<f64> {
    (a > b).then(|| a + (-(-(a - b)).exp_m1()).ln())
}
fn numerical() -> ProbError {
    ProbError::NumericalError(
        "Wiener CDF series exhausted its work or numerical conditioning limits".into(),
    )
}

fn log_sinhc(x: f64) -> f64 {
    let x = x.abs();
    if x < 0.01 {
        let y = x * x;
        (y / 6. + y * y / 120. + y * y * y / 5040.).ln_1p()
    } else if x < 20. {
        (x.sinh() / x).ln()
    } else {
        x - (2. * x).ln() + (-(-2. * x).exp()).ln_1p()
    }
}
fn log_mass(a: f64, v: f64, w: f64, q: f64) -> Option<f64> {
    let z = a * v;
    if !z.is_finite() || z.abs() > 1e6 {
        return None;
    }
    let mass = w.ln() + z * q + log_sinhc(z * w) - log_sinhc(z);
    (mass.is_finite() && mass < 1e-9).then_some(mass.min(0.))
}

/// Log Mills ratio for a nonnegative argument, avoiding erfc underflow.
fn log_mills(z: f64) -> f64 {
    if z < 10. {
        return (0.5 * libm::erfc(z / std::f64::consts::SQRT_2)).ln()
            + 0.5 * z * z
            + 0.5 * (2. * PI).ln();
    }
    let inv = 1. / (z * z);
    let mut term = 1.;
    let mut sum = 1.;
    for j in 1..=64 {
        let next = -term * (2 * j - 1) as f64 * inv;
        if next.abs() > term.abs() {
            break;
        }
        sum += next;
        term = next;
        if next.abs() < f64::EPSILON * sum.abs() {
            break;
        }
    }
    -z.ln() + sum.ln()
}
fn log_phi(x: f64) -> f64 {
    if x <= -10. {
        -0.5 * x * x - 0.5 * (2. * PI).ln() + log_mills(-x)
    } else {
        (0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)).ln()
    }
}
fn log_hit(x: f64, t: f64, v: f64) -> f64 {
    let root = t.sqrt();
    let z = x / root;
    let drift = v.abs() * root;
    if z >= drift {
        -0.5 * z * z - 0.5 * drift * drift - 0.5 * (2. * PI).ln()
            + log_add(log_mills(z - drift), log_mills(z + drift))
    } else {
        log_add(
            -v.abs() * x + log_phi(drift - z),
            v.abs() * x + log_phi(-drift - z),
        )
    }
}
/// Geometric envelope for the omitted integrated image lattice.
fn log_image_tail(x: f64, t: f64, v: f64, step: f64) -> Option<f64> {
    let vt = v.abs() * t;
    if x <= vt {
        return None;
    }
    let ratio = -step * (2. * x + step) / (2. * t);
    Some(
        -x * x / (2. * t) - 0.5 * v * v * t + 0.5 * t.ln() - 0.5 * (2. * PI).ln()
            + log_add(-(x - vt).ln(), -(x + vt).ln())
            - (-ratio.exp_m1()).ln(),
    )
}

fn short_cdf(t: f64, a: f64, v: f64, q: f64, tolerance: f64) -> Option<f64> {
    let (mut positive, mut negative) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for k in 0..128 {
        positive = log_add(positive, log_hit(a * (q + 2. * k as f64), t, v));
        negative = log_add(negative, log_hit(a * (2. * k as f64 + 2. - q), t, v));
        // Decline cancellation-prone differences; a truncation bound cannot
        // recover information lost while subtracting rounded log probabilities.
        if positive - negative < 1e-5 {
            continue;
        }
        let value = log_sub(positive, negative)?;
        let next = 2. * (k + 1) as f64;
        let Some(p) = log_image_tail(a * (q + next), t, v, 2. * a) else {
            continue;
        };
        let Some(n) = log_image_tail(a * (next + 2. - q), t, v, 2. * a) else {
            continue;
        };
        let relative = (log_add(p, n) - value).exp();
        if relative < 1. && -(-relative).ln_1p() <= tolerance {
            return Some(a * v * q + value);
        }
    }
    None
}

fn long_tail(t: f64, a: f64, v: f64, w: f64, q: f64, tolerance: f64, log_mass: f64) -> Option<f64> {
    let c = 0.5 * PI * PI * t / (a * a);
    if !c.is_finite() || c <= 0. {
        return None;
    }
    let pref = PI.ln() - 2. * a.ln() + a * v * q - 0.5 * v * v * t - c;
    let mut sum = 0.;
    let mut absolute = 0.;
    for k in 1_usize..=128 {
        let n = k as f64;
        let sine = if w < q {
            let s = (n * PI * w).sin();
            if k.is_multiple_of(2) { -s } else { s }
        } else {
            (n * PI * q).sin()
        };
        let rate = 0.5 * (v * v + (n * PI / a).powi(2));
        let term = n * sine * (-(n * n - 1.) * c).exp() / rate;
        sum += term;
        absolute += term.abs();
        if sum <= 0. || absolute / sum > 1e6 {
            continue;
        }
        let value = pref + sum.ln();
        let next_rate = 0.5 * (v * v + ((n + 1.) * PI / a).powi(2));
        let Some(t1) = super::bounds::spectral(1, k, c) else {
            continue;
        };
        let Some(t2) = super::bounds::spectral(2, k, c) else {
            continue;
        };
        let tail = t1.min((PI * w.min(q)).sin().ln() + t2);
        let bound = pref + tail - next_rate.ln();
        let Some(cdf) = log_sub(log_mass, value) else {
            continue;
        };
        let relative = (bound - value.min(cdf)).exp();
        if relative < 1. && -(-relative).ln_1p() <= tolerance {
            return Some(value);
        }
    }
    None
}

impl Wiener4 {
    fn cumulative_log(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
        tail: bool,
    ) -> Result<f64> {
        let options = options.validate()?;
        if !params.valid() || obs.rt.is_nan() {
            return Err(ProbError::InvalidParameters(
                "Wiener CDF requires valid parameters and a non-NaN reaction time".into(),
            ));
        }
        let (w, q, v) = match obs.boundary {
            Boundary::Upper => (params.beta, 1. - params.beta, params.delta),
            Boundary::Lower => (1. - params.beta, params.beta, -params.delta),
        };
        let mass = log_mass(params.alpha, v, w, q).ok_or_else(numerical)?;
        let t = obs.rt - params.tau;
        if t <= 0. {
            return Ok(if tail { mass } else { f64::NEG_INFINITY });
        }
        if t == f64::INFINITY {
            return Ok(if tail { f64::NEG_INFINITY } else { mass });
        }
        let u = t / (params.alpha * params.alpha);
        if u < 0.15
            && let Some(f) = short_cdf(t, params.alpha, v, q, options.precision)
        {
            if !tail {
                return Ok(f.min(mass));
            }
            if f < mass - 0.1 {
                return log_sub(mass, f).ok_or_else(numerical);
            }
        }
        let h =
            long_tail(t, params.alpha, v, w, q, options.precision, mass).ok_or_else(numerical)?;
        if tail {
            Ok(h)
        } else {
            log_sub(mass, h).ok_or_else(numerical)
        }
    }

    /// Log P(T <= rt, B = boundary), including nondecision time.
    /// This is a boundary subdistribution, whose limiting mass can be below one.
    /// Work/cancellation limits return a numerical error; roundoff is not enclosed.
    pub fn try_log_cdf(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<f64> {
        self.cumulative_log(obs, params, options, false)
    }
    /// P(T <= rt, B = boundary). See [`Self::try_log_cdf`].
    pub fn try_cdf(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<f64> {
        self.try_log_cdf(obs, params, options).map(f64::exp)
    }
    /// Log P(T > rt, B = boundary). This is not log(1 - F_boundary(rt)).
    pub fn try_log_survival(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<f64> {
        self.cumulative_log(obs, params, options, true)
    }
}

impl Wiener4 {
    /// Log P(lower_rt < T <= upper_rt, B = boundary).
    ///
    /// Chooses the better separated CDF or boundary-tail difference. Very narrow
    /// differences return a numerical error; no cancellation-prone subtraction
    /// is silently accepted. Nondecision time is included in both endpoints.
    pub fn try_log_interval_prob(
        &self,
        lower_rt: f64,
        upper_rt: f64,
        boundary: Boundary,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<f64> {
        let options = options.validate()?;
        if lower_rt.is_nan() || upper_rt.is_nan() || lower_rt >= upper_rt {
            return Err(ProbError::InvalidParameters(
                "Wiener interval endpoints must be ordered and non-NaN".into(),
            ));
        }
        if upper_rt <= params.tau {
            if !params.valid() {
                return Err(ProbError::InvalidParameters(
                    "Invalid Wiener interval parameters".into(),
                ));
            }
            return Ok(f64::NEG_INFINITY);
        }
        let inner = WienerOptions::new(options.precision * 1e-4);
        let lo = WienerObservation {
            rt: lower_rt,
            boundary,
        };
        let hi = WienerObservation {
            rt: upper_rt,
            boundary,
        };
        let prefer_tail = (lower_rt - params.tau) / (params.alpha * params.alpha) >= 0.15;
        for tail in [prefer_tail, !prefer_tail] {
            let (a, b) = if tail {
                (
                    self.try_log_survival(&lo, params, inner),
                    self.try_log_survival(&hi, params, inner),
                )
            } else {
                (
                    self.try_log_cdf(&hi, params, inner),
                    self.try_log_cdf(&lo, params, inner),
                )
            };
            if let (Ok(a), Ok(b)) = (a, b)
                && a - b > 1e-3
                && let Some(value) = log_sub(a, b)
            {
                return Ok(value);
            }
        }
        Err(numerical())
    }
}
