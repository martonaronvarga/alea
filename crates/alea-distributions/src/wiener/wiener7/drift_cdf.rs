//! Integrated images after exact Gaussian drift averaging.
use super::super::{bounds, cdf};
use super::*;

pub(super) struct Cdf {
    pub(super) log_prob: f64,
    pub(super) scores: [f64; 5],
    // Real-arithmetic truncation bounds in log-density and score units.
    pub(super) error: [f64; 6],
}

fn mills_slope(z: f64) -> f64 {
    if z < 10. {
        z - (-cdf::log_mills(z)).exp()
    } else {
        // Laplace's continued fraction for 1/R(z)-z. Unlike subtracting
        // two nearly equal numbers, this retains the O(1/z) difference.
        let mut fraction = 0.;
        for n in (1..=64).rev() {
            fraction = n as f64 / (z + fraction);
        }
        -fraction
    }
}

/// Positive integrated image and its scores, before its lattice sign.
fn image(t: f64, a: f64, q: f64, v: f64, s: f64, j: f64) -> Option<(f64, [f64; 5])> {
    let direction = if j >= 0. { 1. } else { -1. };
    let d = a * (q + 2. * j).abs();
    let lambda = 1. + s * s * t;
    let root = (t * lambda).sqrt();
    let residual = a * q - v * t;
    let central = residual * residual / (2. * t * lambda);
    let energy = -2. * a * a * j * (j + q) / t - central;
    let mut terms = [(0., [0.; 5]); 2];
    for (term, sign) in terms.iter_mut().zip([1., -1.]) {
        let shift = a * ((1. - sign * direction) * q - 2. * sign * direction * j);
        let mean = v + shift * s * s;
        let z = (d - sign * mean * t) / root;
        let dz = [
            (direction * (q + 2. * j) + s * s * t * ((direction - sign) * q + 2. * direction * j))
                / root,
            (d + t * (s * s * a * ((direction + sign) * q + 2. * direction * j) + sign * v))
                / (2. * t * lambda * root),
            a * (direction + (direction - sign) * s * s * t) / root,
            -sign * t / root,
            2. * s * t * a * ((direction - sign) * q + 2. * direction * j) / root
                - z * s * t / lambda,
        ];
        let (log, gradient) = if z >= 0. {
            let de = [
                -4. * a * j * (j + q) / t - q * residual / (t * lambda),
                -(2. * a * a * j * (j + q) / (t * t)
                    + v * residual / (t * lambda)
                    + central * (1. / t + s * s / lambda)),
                -2. * a * a * j / t - a * residual / (t * lambda),
                residual / lambda,
                s * residual * residual / (lambda * lambda),
            ];
            let slope = mills_slope(z);
            (
                energy - 0.5 * (2. * PI).ln() + cdf::log_mills(z),
                std::array::from_fn(|i| de[i] + slope * dz[i]),
            )
        } else {
            let lp = cdf::log_phi(-z);
            let slope = -(-0.5 * z * z - 0.5 * (2. * PI).ln() - lp).exp();
            let dk = a * (1. - sign * direction);
            let dp = [shift / a * mean, 0., dk * mean, shift, shift * shift * s];
            (
                shift * v + 0.5 * shift * shift * s * s + lp,
                std::array::from_fn(|i| dp[i] + slope * dz[i]),
            )
        };
        if !log.is_finite() || gradient.iter().any(|x: &f64| !x.is_finite()) {
            return None;
        }
        *term = (log, gradient);
    }
    let scale = terms[0].0.max(terms[1].0);
    let weights = terms.map(|term| (term.0 - scale).exp());
    let total = weights[0] + weights[1];
    Some((
        scale + total.ln(),
        std::array::from_fn(|i| (weights[0] * terms[0].1[i] + weights[1] * terms[1].1[i]) / total),
    ))
}

pub(super) fn evaluate(t: f64, core: &Wiener7Core, boundary: Boundary, eps: f64) -> Option<Cdf> {
    // The integrated-image tail proof only needs the omitted-distance
    // monotonicity check below, not the density branch crossover. Bound
    // planning work in long time; conditioning checks still gate every result.
    if t <= 0. || !t.is_finite() || t / (core.alpha * core.alpha) >= 512. {
        return None;
    }
    let p = Wiener5Params::from_array([core.alpha, 0., core.beta0, core.delta, core.sv]);
    let obs = WienerObservation { rt: t, boundary };
    let base = Wiener5.core(&obs, &p, eps).ok()?;
    let (a, q, v, s) = (core.alpha, base.base.q_eff, base.base.v_eff, core.sv);
    let u = base.base.t_prime;
    let scale = image(t, a, q, v, s, 0.)?.0;
    let mut sum = [0.; 6];
    let mut correction = [0.; 6];
    let mut absolute = 0.;
    let mut absolute_derivatives = [0.; 5];
    for k in 0..=128 {
        for j in [-(k as f64), k as f64]
            .into_iter()
            .take(if k == 0 { 1 } else { 2 })
        {
            let (log, mut scores) = image(t, a, q, v, s, j)?;
            scores[2] *= -base.base.beta_sign;
            scores[3] *= base.base.delta_sign;
            let weight = (log - scale).exp();
            absolute += weight;
            for (total, score) in absolute_derivatives.iter_mut().zip(scores) {
                *total += weight * score.abs();
            }
            let sign = if j < 0. { -1. } else { 1. };
            for (i, value) in std::iter::once(1.).chain(scores).enumerate() {
                let y = sign * weight * value - correction[i];
                let next = sum[i] + y;
                correction[i] = (next - sum[i]) - y;
                sum[i] = next;
            }
        }
        if sum[0] <= 0. || absolute / sum[0] > 1e6 {
            continue;
        }
        if 64. * f64::EPSILON * absolute > core.opts.req_rel_error * sum[0] {
            return None;
        }
        let next = 2. * (k + 1) as f64;
        if (next - q).powi(2) - q * q < 5. * u {
            continue;
        }
        let factor = a.ln() - 0.5 * (2. * PI * t).ln() + q * q / (2. * u) - scale;
        let mut moments = [0.; 4];
        for (p, moment) in moments.iter_mut().enumerate() {
            *moment = (bounds::add(
                bounds::lattice(p, next + q, 0., u)?,
                bounds::lattice(p, next - q, 0., u)?,
            ) + factor)
                .exp();
        }
        let [m0, m1, m2, m3] = moments;
        let residual = a * q + v.abs() * t;
        let errors = [
            m1,
            (1. / a + q * residual / t + a * q * q / t) * m1 + a * m3 / t,
            m1 / t,
            m0 + a * a * m2 / t + (a * a * q + a * residual) * m1 / t,
            residual * m1,
            s * (residual * residual + t) * m1,
        ];
        let margin = sum[0] - errors[0];
        if margin <= 0. {
            continue;
        }
        let scores = std::array::from_fn(|i| sum[i + 1] / sum[0]);
        // A cancellation diagnostic, not a libm rounding-error certificate.
        // Use the outer score tolerance; the stricter inner budget controls
        // real-arithmetic truncation separately.
        if (0..5).any(|i| {
            64. * f64::EPSILON * (absolute_derivatives[i] + scores[i].abs() * absolute)
                > core.opts.req_rel_error * sum[0] * scores[i].abs().max(1.)
        }) {
            return None;
        }
        let mut error = [0.; 6];
        error[0] = -(-(errors[0] / sum[0])).ln_1p();
        for i in 0..5 {
            error[i + 1] = (errors[i + 1] + scores[i].abs() * errors[0]) / margin;
        }
        if error[0] <= eps
            && (0..5)
                .all(|i| scores[i].is_finite() && error[i + 1] <= eps * scores[i].abs().max(1.))
        {
            return Some(Cdf {
                log_prob: scale + sum[0].ln(),
                scores,
                error,
            });
        }
    }
    None
}
