//! Gaussian-drift spectral start averages via bounded complex power moments.
use super::super::bounds;
use super::gaussian_polynomial::{C, add, complex_moments, mul, scale};
use super::*;

impl Wiener7 {
    #[cfg(test)]
    pub(super) fn analytic_spectral_start(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        self.analytic_spectral_start_components::<true>(core, obs)
    }

    pub(super) fn analytic_spectral_start_components<const SCORES: bool>(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.sw <= 0. || core.st0 != 0. || core.sv <= 0. {
            return None;
        }
        let p = Wiener5Params::from_array([core.alpha, core.tau0, core.beta0, core.delta, core.sv]);
        let base = Wiener5.core(obs, &p, core.eps_series).ok()?;
        let (a, t, u, m, h, v, s, lam) = (
            core.alpha,
            base.base.t,
            base.base.t_prime,
            base.base.q_eff,
            core.sw,
            base.base.v_eff,
            core.sv,
            base.lam,
        );
        if !u.is_finite() || u < 0.15 || !lam.is_finite() {
            return None;
        }
        let b = a * v / lam;
        let d = 0.5 * a * a * s * s / lam;
        let c = 0.5 * PI * PI * u;
        let qlo = m - h / 2.;
        let qhi = m + h / 2.;
        if qlo <= 0. || qhi >= 1. {
            return None;
        }
        let exponent = |q: f64| b * q + d * q * q;
        let factor = exponent(qlo).max(exponent(qhi)) - exponent(m) + c;
        let mut sum = [0.; 8];
        let mut error = [0.; 8];
        let mut correction = [0.; 8];
        let mut absolute = 0.;
        for k in 1_usize..=128 {
            let n = k as f64;
            let omega = n * PI;
            let (moments, remainder) = if SCORES {
                complex_moments::<4>(-d * h * h, (h * (b + 2. * d * m), h * omega))?
            } else {
                complex_moments::<1>(-d * h * h, (h * (b + 2. * d * m), h * omega))?
            };
            let phase = if base.base.w_eff < base.base.q_eff {
                let (s, c) = (omega * base.base.w_eff).sin_cos();
                if k.is_multiple_of(2) {
                    (c, -s)
                } else {
                    (-c, s)
                }
            } else {
                let (s, c) = (omega * m).sin_cos();
                (c, s)
            };
            let component = |moments: [C; 4]| {
                let z = moments.map(|moment| mul(phase, moment));
                let q1 = add(scale(z[0], m), scale(z[1], h));
                let q2 = add(
                    add(scale(z[0], m * m), scale(z[1], 2. * m * h)),
                    scale(z[2], h * h),
                );
                let density = z[0].1;
                if !SCORES {
                    return [density, 0., 0., 0., 0., 0., 0., 0.];
                }
                let alpha = (-2. / a + omega * omega * t / (a * a * a)) * density
                    + (v * q1.1 + a * s * s * q2.1) / lam;
                let tau =
                    0.5 * (s * s / lam + v * v / (lam * lam) + omega * omega / (a * a)) * density
                        + v * a * s * s * q1.1 / (lam * lam)
                        + 0.5 * a * a * s * s * s * s * q2.1 / (lam * lam);
                let beta = -base.base.beta_sign * add(mul((b, omega), z[0]), scale(q1, 2. * d)).1;
                let delta = base.base.delta_sign * (a * q1.1 - v * t * density) / lam;
                let sd = s
                    * ((v * v * t * t / (lam * lam) - t / lam) * density
                        - 2. * v * t * a * q1.1 / (lam * lam)
                        + a * a * q2.1 / (lam * lam));
                let width = add(
                    mul((b, omega), z[1]),
                    scale(add(scale(z[1], m), scale(z[2], h)), 2. * d),
                )
                .1;
                [density, alpha, tau, beta, delta, sd, width, 0.5 * tau]
            };
            let values = component(moments);
            let mut errors = [0.; 8];
            for r in 0..if SCORES { 4 } else { 1 } {
                for unit in [(1., 0.), (0., 1.)] {
                    let mut basis = [(0., 0.); 4];
                    basis[r] = unit;
                    for (e, x) in errors.iter_mut().zip(component(basis)) {
                        *e += x.abs() * remainder[r];
                    }
                }
            }
            let weight = n * (-(n * n - 1.) * c).exp();
            absolute += weight * values[0].abs();
            for i in 0..if SCORES { 8 } else { 1 } {
                let y = weight * values[i] - correction[i];
                let next = sum[i] + y;
                correction[i] = (next - sum[i]) - y;
                sum[i] = next;
                error[i] += weight * errors[i];
            }
            if sum[0] <= 0. || absolute / sum[0] > 1e6 {
                continue;
            }
            if !SCORES {
                let tail = (bounds::spectral(1, k, c)? + factor).exp();
                let bound = error[0] + tail;
                if bound < sum[0] && -(-(bound / sum[0])).ln_1p() <= core.eps_series {
                    let log_prob = PI.ln() - 2. * a.ln() - 0.5 * lam.ln() - 0.5 * v * v * t / lam
                        + exponent(m)
                        - c
                        + sum[0].ln();
                    return log_prob.is_finite().then_some(Wiener7Eval {
                        log_prob,
                        grad: Wiener7Grad::from_array([0.; 7]),
                    });
                }
                continue;
            }
            let tails =
                [1, 2, 3].map(|power| bounds::spectral(power, k, c).map(|x| (x + factor).exp()));
            let [Some(t1), Some(t2), Some(t3)] = tails else {
                continue;
            };
            let beta = (b.abs() + 2. * d * qhi) * t1 + PI * t2;
            let tau = 0.5 * (s * s / lam + ((v.abs() + a * qhi * s * s) / lam).powi(2)) * t1
                + 0.5 * PI * PI / (a * a) * t3;
            let tails = [
                t1,
                (2. / a + (v.abs() * qhi + a * qhi * qhi * s * s) / lam) * t1
                    + PI * PI * t / (a * a * a) * t3,
                tau,
                beta,
                (a * qhi + v.abs() * t) / lam * t1,
                s * ((v.abs() * t + a * qhi).powi(2) / (lam * lam) + t / lam) * t1,
                0.5 * beta,
                0.5 * tau,
            ];
            let errors = std::array::from_fn(|i| error[i] + tails[i]);
            let Some((log_error, score_error)) = super::error::ratios(&sum, &errors) else {
                continue;
            };
            if log_error > core.eps_series
                || score_error
                    .iter()
                    .zip(&sum[1..])
                    .any(|(e, g)| *e > core.eps_series * (g / sum[0]).abs().max(1.))
            {
                continue;
            }
            let log_scale =
                PI.ln() - 2. * a.ln() - 0.5 * lam.ln() - 0.5 * v * v * t / lam + exponent(m) - c;
            let value = Wiener7Eval {
                log_prob: log_scale + sum[0].ln(),
                grad: Wiener7Grad::from_array(std::array::from_fn(|i| sum[i + 1] / sum[0])),
            };
            return (value.log_prob.is_finite()
                && value.grad_array().iter().all(|x| x.is_finite()))
            .then_some(value);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gaussian_drift_spectral_start_matches_continuous_ball_integrals() {
        let mut count = 0;
        for line in include_str!("../../../tests/support/wiener7_spectral_balls.csv")
            .lines()
            .skip(1)
        {
            let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let obs = WienerObservation {
                rt: x[0],
                boundary: if x[5] == 1. {
                    Boundary::Lower
                } else {
                    Boundary::Upper
                },
            };
            let p = Wiener7Params::with_params(1., 0., x[1], x[2], x[4], 0., x[3]).unwrap();
            let core = Wiener7.core(&obs, &p, 1e-10).unwrap();
            let e = Wiener7
                .analytic_spectral_start(&core, &obs)
                .unwrap_or_else(|| panic!("unsupported row {count}: {x:?}"));
            for (i, actual) in std::iter::once(e.log_prob)
                .chain(e.grad_array())
                .enumerate()
            {
                let reference = x[6 + 2 * i];
                assert!(
                    (actual - reference).abs() <= x[7 + 2 * i] + 1e-9 * reference.abs().max(1.),
                    "row {count}, component {i}: {actual} vs {reference}"
                );
            }
            count += 1;
        }
        assert_eq!(count, 288);
    }
}
