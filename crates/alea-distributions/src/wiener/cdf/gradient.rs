//! Analytic derivatives of integrated images and spectral boundary tails.
use super::*;

fn coth_minus_reciprocal(x: f64) -> f64 {
    if x.abs() < 0.01 {
        let y = x * x;
        x * (1. / 3. - y / 45. + 2. * y * y / 945. - y * y * y / 4725.)
    } else {
        x.signum() * (1. + 2. / (2. * x.abs()).exp_m1()) - 1. / x
    }
}
fn mass_eval(a: f64, v: f64, w: f64, q: f64, qsign: f64, vsign: f64) -> Option<Wiener4Eval> {
    let z = a * v;
    let dz = q + w * coth_minus_reciprocal(z * w) - coth_minus_reciprocal(z);
    let dq = -1. / w + z - z * coth_minus_reciprocal(z * w);
    Some(Wiener4Eval {
        log_prob: log_mass(a, v, w, q)?,
        grad: Wiener4Grad::from_array([v * dz, 0., qsign * dq, vsign * a * dz]),
    })
}
fn combine(log: f64, raw: [f64; 5]) -> Option<Wiener4Eval> {
    if raw[0] <= 0. || raw.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let grad = Wiener4Grad::from_array(std::array::from_fn(|i| raw[i + 1] / raw[0]));
    Some(Wiener4Eval {
        log_prob: log + raw[0].ln(),
        grad,
    })
}
fn subtract(a: Wiener4Eval, b: Wiener4Eval) -> Option<Wiener4Eval> {
    let lp = log_sub(a.log_prob, b.log_prob)?;
    let wa = (a.log_prob - lp).exp();
    let wb = (b.log_prob - lp).exp();
    let g = std::array::from_fn(|i| wa * a.grad_array()[i] - wb * b.grad_array()[i]);
    g.iter().all(|x| x.is_finite()).then_some(Wiener4Eval {
        log_prob: lp,
        grad: Wiener4Grad::from_array(g),
    })
}
fn subtract_checked(
    a: Wiener4Eval,
    b: Wiener4Eval,
    ea: f64,
    eb: f64,
    eps: f64,
) -> Option<Wiener4Eval> {
    let e = subtract(a, b)?;
    let wa = (a.log_prob - e.log_prob).exp();
    let wb = (b.log_prob - e.log_prob).exp();
    let pa = ea.exp_m1();
    let pb = eb.exp_m1();
    let density = wa * pa + wb * pb;
    if density >= 1. || -(-density).ln_1p() > eps {
        return None;
    }
    for i in 0..4 {
        let ga = a.grad_array()[i];
        let gb = b.grad_array()[i];
        let score = e.grad_array()[i];
        let ba = pa * ga.abs() + ea.exp() * ea * ga.abs().max(1.);
        let bb = pb * gb.abs() + eb.exp() * eb * gb.abs().max(1.);
        let bound = (wa * ba + wb * bb + score.abs() * density) / (1. - density);
        if !bound.is_finite() || bound > eps * score.abs().max(1.) {
            return None;
        }
    }
    Some(e)
}

fn meets(raw: &[f64; 5], errors: &[f64; 5], eps: f64) -> bool {
    let margin = raw[0] - errors[0];
    if margin <= 0. || !margin.is_finite() {
        return false;
    }
    let r = errors[0] / raw[0];
    -(-r).ln_1p() <= eps
        && (0..4).all(|i| {
            let score = raw[i + 1] / raw[0];
            let err = (errors[i + 1] + score.abs() * errors[0]) / margin;
            score.is_finite() && err.is_finite() && err <= eps * score.abs().max(1.)
        })
}
/// J, J_x/J, J_v/J, log G, where G=phi(x/sqrt(t))*exp(-v²t/2).
fn image(x: f64, t: f64, v: f64) -> (f64, f64, f64, f64) {
    let root = t.sqrt();
    let z = x / root;
    let d = v * root;
    let g = -0.5 * z * z - 0.5 * d * d - 0.5 * (2. * PI).ln();
    let (j, r, gj) = if z >= d.abs() {
        let la = log_mills(z - d);
        let lb = log_mills(z + d);
        let sum = log_add(la, lb);
        (g + sum, ((lb - la) / 2.).tanh(), (-sum).exp())
    } else {
        let la = -v * x + log_phi(d - z);
        let lb = v * x + log_phi(-d - z);
        let sum = log_add(la, lb);
        (sum, ((lb - la) / 2.).tanh(), (g - sum).exp())
    };
    (j, v * r - 2. * gj / root, x * r, g)
}
fn log_moment(p: usize, x: f64, step: f64, t: f64) -> Option<f64> {
    let ratio = p as f64 * (step / x).ln_1p() - step * (2. * x + step) / (2. * t);
    (ratio < 0.).then(|| p as f64 * x.ln() - x * x / (2. * t) - (-ratio.exp_m1()).ln())
}
fn short(t: f64, a: f64, v: f64, q: f64, qsign: f64, vsign: f64, eps: f64) -> Option<Wiener4Eval> {
    let scale = image(a * q, t, v).0;
    let mut raw = [0.; 5];
    for k in 0..128 {
        for (distance, sign, dxq) in [
            (q + 2. * k as f64, 1., a),
            (2. * k as f64 + 2. - q, -1., -a),
        ] {
            let x = a * distance;
            let (j, jx, jv, g) = image(x, t, v);
            let weight = sign * (j - scale).exp();
            raw[0] += weight;
            raw[1] += weight * (v * q + jx * distance);
            raw[2] -= sign * (x.ln() + g - 1.5 * t.ln() - scale).exp();
            raw[3] += qsign * weight * (a * v + jx * dxq);
            raw[4] += vsign * weight * (a * q + jv);
        }
        if raw[0] <= 1e-5 {
            continue;
        }
        let next = 2. * (k + 1) as f64;
        let mut tails = [0.; 5];
        let mut valid = true;
        for x in [a * (q + next), a * (next + 2. - q)] {
            if x <= v.abs() * t {
                continue;
            }
            let logj = log_image_tail(x, t, v, 2. * a)?;
            let first_ratio = -2. * a * (2. * x + 2. * a) / (2. * t);
            let weighted_ratio = (2. * a / x).ln_1p() + first_ratio;
            if weighted_ratio >= 0. {
                valid = false;
                continue;
            }
            let xj = (logj + x.ln() + (-first_ratio.exp_m1()).ln()
                - (-weighted_ratio.exp_m1()).ln()
                - scale)
                .exp();
            let jt = (logj - scale).exp();
            let g0 =
                (log_moment(0, x, 2. * a, t)? - 0.5 * v * v * t - 0.5 * (2. * PI).ln() - scale)
                    .exp();
            let g1 =
                (log_moment(1, x, 2. * a, t)? - 0.5 * v * v * t - 0.5 * (2. * PI).ln() - scale)
                    .exp();
            tails[0] += jt;
            tails[1] += v.abs() * q * jt + (v.abs() * xj + 2. * g1 / t.sqrt()) / a;
            tails[2] += g1 / t.powf(1.5);
            tails[3] += 2. * a * v.abs() * jt + 2. * a * g0 / t.sqrt();
            tails[4] += a * q * jt + xj;
        }
        // Both omitted image lattices must be past their monotone regime.
        if a * (q + next) <= v.abs() * t || a * (next + 2. - q) <= v.abs() * t {
            continue;
        }
        if valid && meets(&raw, &tails, eps) {
            return combine(a * v * q + scale, raw);
        }
    }
    None
}
fn spectral(geometry: [f64; 7], eps: f64, mass: Wiener4Eval, tail: bool) -> Option<Wiener4Eval> {
    let [t, a, v, w, q, qsign, vsign] = geometry;
    let c = 0.5 * PI * PI * t / (a * a);
    let pref = PI.ln() - 2. * a.ln() + a * v * q - 0.5 * v * v * t - c;
    let mut raw = [0.; 5];
    let mut absolute = 0.;
    for k in 1_usize..=128 {
        let n = k as f64;
        let omega = PI * n;
        let (sin, cos) = if w < q {
            let (s, c) = (omega * w).sin_cos();
            if k.is_multiple_of(2) {
                (-s, c)
            } else {
                (s, -c)
            }
        } else {
            (omega * q).sin_cos()
        };
        let rate = 0.5 * (v * v + (omega / a).powi(2));
        let weight = n * (-(n * n - 1.) * c).exp() / rate;
        let term = weight * sin;
        absolute += term.abs();
        raw[0] += term;
        raw[1] += term * (-2. / a + v * q + omega * omega / (a * a * a) * (t + 1. / rate));
        raw[2] += rate * term;
        raw[3] += qsign * weight * (a * v * sin + omega * cos);
        raw[4] += vsign * term * (a * q - v * t - v / rate);
        if raw[0] <= 0. || absolute / raw[0] > 1e6 {
            continue;
        }
        let Some(p1) = super::super::bounds::spectral(1, k, c) else {
            continue;
        };
        let Some(p2) = super::super::bounds::spectral(2, k, c) else {
            continue;
        };
        let Some(p3) = super::super::bounds::spectral(3, k, c) else {
            continue;
        };
        let r = 0.5 * (v * v + ((n + 1.) * PI / a).powi(2));
        let (t1, t2, t3) = (p1.exp(), p2.exp(), p3.exp());
        let errors = [
            t1 / r,
            (-2. / a + v * q).abs() * t1 / r + PI * PI / (a * a * a) * (t + 1. / r) * t3 / r,
            t1,
            a * v.abs() * t1 / r + PI * t2 / r,
            (a * q + v.abs() * t + v.abs() / r) * t1 / r,
        ];
        let h = combine(pref, raw)?;
        if tail && meets(&raw, &errors, eps) {
            return Some(h);
        }
        if !tail {
            let Some(f) = subtract(mass, h) else {
                continue;
            };
            let ratio = (pref - f.log_prob).exp();
            let scaled = [1., f.grad.alpha, f.grad.tau, f.grad.beta, f.grad.delta];
            let errors = errors.map(|x| ratio * x);
            if meets(&scaled, &errors, eps) {
                return Some(f);
            }
        }
    }
    None
}

impl Wiener4 {
    fn cumulative_fused(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
        tail: bool,
    ) -> Result<Wiener4Eval> {
        let options = options.validate()?;
        if !params.valid() || obs.rt.is_nan() {
            return Err(ProbError::InvalidParameters(
                "Wiener CDF scores require valid parameters and non-NaN time".into(),
            ));
        }
        let (w, q, v, qs, vs) = match obs.boundary {
            Boundary::Upper => (params.beta, 1. - params.beta, params.delta, -1., 1.),
            Boundary::Lower => (1. - params.beta, params.beta, -params.delta, 1., -1.),
        };
        let mass = mass_eval(params.alpha, v, w, q, qs, vs).ok_or_else(numerical)?;
        let t = obs.rt - params.tau;
        if t <= 0. {
            return if tail {
                Ok(mass)
            } else {
                Err(ProbError::OutOfSupport(
                    "The zero CDF has no finite log score".into(),
                ))
            };
        }
        if t == f64::INFINITY {
            return if tail {
                Err(ProbError::OutOfSupport(
                    "The zero boundary tail has no finite log score".into(),
                ))
            } else {
                Ok(mass)
            };
        }

        if t / (params.alpha * params.alpha) < 0.15
            && let Some(f) = short(
                t,
                params.alpha,
                v,
                q,
                qs,
                vs,
                if tail {
                    0.01 * options.precision
                } else {
                    options.precision
                },
            )
        {
            if !tail {
                return Ok(f);
            }
            if f.log_prob < mass.log_prob - 0.1
                && let Some(h) =
                    subtract_checked(mass, f, 0., 0.01 * options.precision, options.precision)
            {
                return Ok(h);
            }
        }
        spectral(
            [t, params.alpha, v, w, q, qs, vs],
            options.precision,
            mass,
            tail,
        )
        .ok_or_else(numerical)
    }
    /// Boundary log-CDF and analytic parameter scores, in ordinary gradient order.
    /// Series tails control log probability and each score; roundoff is excluded.
    pub fn try_fused_cdf(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<Wiener4Eval> {
        self.cumulative_fused(obs, params, options, false)
    }
    /// Joint boundary log-tail and its analytic parameter scores.
    pub fn try_fused_survival(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<Wiener4Eval> {
        self.cumulative_fused(obs, params, options, true)
    }
    /// Boundary log interval probability and analytic parameter scores.
    pub fn try_fused_interval(
        &self,
        lower_rt: f64,
        upper_rt: f64,
        boundary: Boundary,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<Wiener4Eval> {
        let options = options.validate()?;
        if lower_rt.is_nan() || upper_rt.is_nan() || lower_rt >= upper_rt {
            return Err(ProbError::InvalidParameters(
                "Wiener interval endpoints must be ordered".into(),
            ));
        }
        let hi = WienerObservation {
            rt: upper_rt,
            boundary,
        };
        if lower_rt <= params.tau {
            return self.try_fused_cdf(&hi, params, options);
        }
        let lo = WienerObservation {
            rt: lower_rt,
            boundary,
        };
        let inner = WienerOptions::new(options.precision * 1e-4);
        let preferred = (lower_rt - params.tau) / (params.alpha * params.alpha) >= 0.15;
        for tail in [preferred, !preferred] {
            let (a, b) = if tail {
                (
                    self.try_fused_survival(&lo, params, inner),
                    self.try_fused_survival(&hi, params, inner),
                )
            } else {
                (
                    self.try_fused_cdf(&hi, params, inner),
                    self.try_fused_cdf(&lo, params, inner),
                )
            };
            if let (Ok(a), Ok(b)) = (a, b)
                && a.log_prob - b.log_prob > 1e-3
                && let Some(e) =
                    subtract_checked(a, b, inner.precision, inner.precision, options.precision)
            {
                return Ok(e);
            }
        }
        Err(numerical())
    }
}
