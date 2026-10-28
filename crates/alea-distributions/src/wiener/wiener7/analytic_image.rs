//! Gaussian image antiderivatives after analytic Gaussian drift marginalization.
//! Each image is (q+2k) exp(-A q²+B_k q+C_k), A=a²/(2t lambda)>0.
//! Uniform averaging and every parameter derivative reduce to four moments.
use super::super::bounds;
use super::*;

fn gaussian_moments(a: f64, b: f64) -> Option<([f64; 4], [f64; 4], f64)> {
    let radius = a / 4. + b.abs() / 2.;
    if radius <= 4.0 {
        // exp(-a*z²+b*z)=sum c_n*z^n, with
        // (n+1)c_(n+1)=b*c_n-2a*c_(n-1). Integrate coefficients once,
        // rather than repeatedly multiplying powers of the exponent.
        let (terms, errors) = super::gaussian_polynomial::plan(a, b.abs())?;
        let mut previous = 0.;
        let mut coefficient = 1.;
        let mut sums = [0.; 4];
        let mut compensation = [0.; 4];
        for n in 0..terms {
            for (r, sum) in sums.iter_mut().enumerate() {
                let y =
                    coefficient * super::gaussian_polynomial::MONOMIALS[n + r] - compensation[r];
                let next = *sum + y;
                compensation[r] = (next - *sum) - y;
                *sum = next;
            }
            let next = (b * coefficient - 2. * a * previous) / (n + 1) as f64;
            previous = coefficient;
            coefficient = next;
        }
        return Some((sums, errors, 0.));
    }
    if let Some(moments) = super::gaussian_tail::moments(a, b) {
        return Some(moments);
    }
    if a < 0.5 || !a.is_finite() || !b.is_finite() {
        return None;
    }
    let mode = b / (2. * a);
    if mode.abs() > 8. || a > 1e6 {
        return None;
    }
    let root = (2. * a).sqrt();
    let (lo, hi) = (root * (-0.5 - mode), root * (0.5 - mode));
    let log_interval = if lo >= 0. {
        let l = super::super::cdf::log_phi(-lo);
        let h = super::super::cdf::log_phi(-hi);
        super::super::cdf::log_sub(l, h)?
    } else {
        let l = super::super::cdf::log_phi(lo);
        let h = super::super::cdf::log_phi(hi);
        super::super::cdf::log_sub(h, l)?
    };
    let peak = mode.clamp(-0.5, 0.5);
    let scale = -a * peak * peak + b * peak;
    let mut moments = [0.; 4];
    moments[0] = (0.5 * (PI / a).ln() + log_interval + b * b / (4. * a) - scale).exp();
    let (left, right) = (
        (-a / 4. - b / 2. - scale).exp(),
        (-a / 4. + b / 2. - scale).exp(),
    );
    for r in 0..3 {
        let first = b * moments[r];
        let second = if r == 0 {
            0.
        } else {
            r as f64 * moments[r - 1]
        };
        let face = 0.5_f64.powi(r as i32) * right - (-0.5_f64).powi(r as i32) * left;
        let numerator = first + second - face;
        // An exact symmetry can yield zero; otherwise decline ill-conditioned
        // moment recurrences rather than amplifying cancellation.
        if numerator != 0. && (first.abs() + second.abs() + face.abs()) / numerator.abs() > 1e6 {
            return None;
        }
        moments[r + 1] = numerator / (2. * a);
    }
    moments
        .iter()
        .all(|x| x.is_finite())
        .then_some((moments, [0.; 4], scale))
}

struct Sum {
    scale: f64,
    value: [f64; 8],
    error: [f64; 8],
    absolute: f64,
}
impl Sum {
    fn add<const SCORES: bool>(&mut self, log: f64, value: [f64; 8], error: [f64; 8]) {
        if log > self.scale {
            let factor = (self.scale - log).exp();
            for x in self.value.iter_mut().chain(&mut self.error) {
                *x *= factor;
            }
            self.absolute *= factor;
            self.scale = log;
        }
        let factor = (log - self.scale).exp();
        for i in 0..if SCORES { 8 } else { 1 } {
            self.value[i] += factor * value[i];
            self.error[i] += factor * error[i];
        }
        self.absolute += factor * value[0].abs();
    }
}

impl Wiener7 {
    #[cfg(test)]
    pub(super) fn analytic_image_start(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        self.analytic_image_start_components::<true>(core, obs)
    }

    pub(super) fn analytic_image_start_components<const SCORES: bool>(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.sw <= 0. || core.st0 != 0. {
            return None;
        }
        let params =
            Wiener5Params::from_array([core.alpha, core.tau0, core.beta0, core.delta, core.sv]);
        let base = Wiener5.core(obs, &params, core.eps_series).ok()?;
        let (a, t, u, m, h, v, s) = (
            base.base.a,
            base.base.t,
            base.base.t_prime,
            base.base.q_eff,
            core.sw,
            base.base.v_eff,
            core.sv,
        );
        if u >= 0.15 || !u.is_finite() {
            return None;
        }
        let lambda = base.lam;
        let aa = a * a / (2. * t * lambda);
        let log_pref = a.ln() - 0.5 * (2. * PI).ln() - 1.5 * t.ln() - 0.5 * lambda.ln();
        let (qlo, qhi) = (m - h / 2., m + h / 2.);
        if qlo <= 0. || qhi >= 1. {
            return None;
        }
        let n = |q: f64| -v * v * t + 2. * a * v * q + a * a * q * q * s * s;
        let pref_max = log_pref + n(qlo).max(n(qhi)) / (2. * lambda);
        let n_bound = v * v * t + 2. * a * v.abs() * qhi + a * a * qhi * qhi * s * s;
        let pa = 2. / a + (v.abs() * qhi + a * qhi * qhi * s * s) / lambda;
        let pt =
            0.5 * s * s / lambda + v * v / (2. * lambda) + n_bound * s * s / (2. * lambda * lambda);
        let pq = (a * v.abs() + a * a * qhi * s * s) / lambda;
        let pv = (v.abs() * t + a * qhi) / lambda;
        let ps =
            s * t / lambda + a * a * qhi * qhi * s / lambda + n_bound * s * t / (lambda * lambda);
        let mut sum = Sum {
            scale: f64::NEG_INFINITY,
            value: [0.; 8],
            error: [0.; 8],
            absolute: 0.,
        };
        for k in 0..=64 {
            let images = [-(k as f64), k as f64];
            for &image in images.iter().take(if k == 0 { 1 } else { 2 }) {
                let b = a * v / lambda - 2. * a * a * image / t;
                let c = -v * v * t / (2. * lambda) - 2. * a * a * image * image / t;
                let az = aa * h * h;
                let bz = (b - 2. * aa * m) * h;
                let (moments, errors, scale) = gaussian_moments(az, bz)?;
                let component = |j: [f64; 4]| {
                    let q0 = j[0];
                    let q1 = m * j[0] + h * j[1];
                    let q2 = m * m * j[0] + 2. * m * h * j[1] + h * h * j[2];
                    let q3 = m * m * m * j[0]
                        + 3. * m * m * h * j[1]
                        + 3. * m * h * h * j[2]
                        + h * h * h * j[3];
                    let t0 = q1 + 2. * image * q0;
                    if !SCORES {
                        return [t0, 0., 0., 0., 0., 0., 0., 0.];
                    }
                    let t1 = q2 + 2. * image * q1;
                    let t2 = q3 + 2. * image * q2;
                    let derivative = |da: f64, db: f64, dc: f64| -da * t2 + db * t1 + dc * t0;
                    let alpha = t0 / a
                        + derivative(
                            2. * aa / a,
                            v / lambda - 4. * a * image / t,
                            -4. * a * image * image / t,
                        );
                    let dt = (-1.5 / t - 0.5 * s * s / lambda) * t0
                        + derivative(
                            -aa * (1. / t + s * s / lambda),
                            -a * v * s * s / (lambda * lambda) + 2. * a * a * image / (t * t),
                            -v * v / (2. * lambda * lambda) + 2. * a * a * image * image / (t * t),
                        );
                    let beta = -base.base.beta_sign * (q0 - 2. * aa * t1 + b * t0);
                    let delta = base.base.delta_sign * derivative(0., a / lambda, -v * t / lambda);
                    let sd = -s * t / lambda * t0
                        + derivative(
                            -2. * aa * s * t / lambda,
                            -2. * a * v * s * t / (lambda * lambda),
                            v * v * t * t * s / (lambda * lambda),
                        );
                    let width = j[1] + (b - 2. * aa * m) * ((m + 2. * image) * j[1] + h * j[2])
                        - 2. * aa * h * ((m + 2. * image) * j[2] + h * j[3]);
                    [t0, alpha, -dt, beta, delta, sd, width, -0.5 * dt]
                };
                let mut error = [0.; 8];
                for r in 0..if SCORES { 4 } else { 2 } {
                    let mut unit = [0.; 4];
                    unit[r] = 1.;
                    for (e, c) in error.iter_mut().zip(component(unit)) {
                        *e += c.abs() * errors[r];
                    }
                }
                sum.add::<SCORES>(
                    log_pref + c - aa * m * m + b * m + scale,
                    component(moments),
                    error,
                );
            }
            if sum.value[0] <= 0. || sum.absolute / sum.value[0] > 1e6 {
                continue;
            }
            let next = 2. * (k + 1) as f64;
            let logs: [f64; 4] = std::array::from_fn(|p| {
                if !SCORES && p != 1 {
                    return f64::NEG_INFINITY;
                }
                match (
                    bounds::lattice(p, next + qlo, 0., u),
                    bounds::lattice(p, next - qhi, 0., u),
                ) {
                    (Some(a), Some(b)) => bounds::add(a, b),
                    _ => f64::INFINITY,
                }
            });
            let factor = pref_max - sum.scale;
            let density = (logs[1] + factor).exp();
            if !SCORES {
                let error = density + sum.error[0];
                if error < sum.value[0] && -(-(error / sum.value[0])).ln_1p() <= core.eps_series {
                    let log_prob = sum.scale + sum.value[0].ln();
                    return log_prob.is_finite().then_some(Wiener7Eval {
                        log_prob,
                        grad: Wiener7Grad::from_array([0.; 7]),
                    });
                }
                continue;
            }
            let time = (bounds::add(
                logs[3] - (2_f64.ln() + 2. * u.ln()),
                logs[1] + (1.5_f64.ln() - u.ln()),
            ) + factor)
                .exp();
            let start = (bounds::add(logs[0], logs[2] - u.ln()) + factor).exp();
            let beta = pq * density + start;
            let tau = pt * density + time / (a * a);
            let tails = [
                density,
                pa * density + 2. * u / a * time,
                tau,
                beta,
                pv * density,
                ps * density,
                0.5 * beta,
                0.5 * tau,
            ];
            let errors = std::array::from_fn(|i| tails[i] + sum.error[i]);
            let Some((log_error, score_errors)) = super::error::ratios(&sum.value, &errors) else {
                continue;
            };
            if log_error > core.eps_series
                || score_errors
                    .iter()
                    .zip(&sum.value[1..])
                    .any(|(e, d)| *e > core.eps_series * (d / sum.value[0]).abs().max(1.))
            {
                continue;
            }
            let e = Wiener7Eval {
                log_prob: sum.scale + sum.value[0].ln(),
                grad: Wiener7Grad::from_array(std::array::from_fn(|i| {
                    sum.value[i + 1] / sum.value[0]
                })),
            };
            return (e.log_prob.is_finite() && e.grad_array().iter().all(|x| x.is_finite()))
                .then_some(e);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_moments_match_independent_continuous_ball_integrals() {
        let mut count = 0;
        for line in include_str!("../../../tests/support/wiener_polynomial_balls.csv")
            .lines()
            .skip(1)
        {
            let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let (values, errors, scale) = gaussian_moments(x[0], x[1]).unwrap();
            assert_eq!(scale, 0.);
            for i in 0..4 {
                let reference = x[2 + 2 * i];
                assert!(
                    (values[i] - reference).abs()
                        <= errors[i] + x[3 + 2 * i] + 3e-14 * (1. + reference.abs()),
                    "A={}, B={}, moment {i}: {} vs {reference}",
                    x[0],
                    x[1],
                    values[i]
                );
            }
            count += 1;
        }
        assert_eq!(count, 22);
    }
    #[test]
    fn gaussian_moments_have_symmetric_zero_width_limits() {
        let (j, error, _) = gaussian_moments(0., 0.).unwrap();
        assert_eq!(j, [1., 0., 1. / 12., 0.]);
        assert_eq!(error, [0.; 4]);
    }
    #[test]
    fn integrated_images_match_independent_quadrature_and_scores() {
        let mut successes = 0;
        for boundary in [Boundary::Upper, Boundary::Lower] {
            for sd in [0., 0.5, 3.] {
                for drift in [-2., 0., 2.] {
                    for width in [1e-10, 0.1, 0.4] {
                        for t in [0.01, 0.05, 0.1] {
                            let obs = WienerObservation {
                                rt: t + 0.1,
                                boundary,
                            };
                            let p = Wiener7Params::with_params(1., 0.1, 0.5, drift, width, 0., sd)
                                .unwrap();
                            let core = Wiener7.core(&obs, &p, 1e-10).unwrap();
                            let Some(actual) = Wiener7.analytic_image_start(&core, &obs) else {
                                continue;
                            };
                            let reference = Wiener7
                                .try_fused(
                                    &obs,
                                    &p,
                                    WienerOptions::new(1e-12).with_quadrature(
                                        Quadrature::FixedGaussLegendre { order: 25 },
                                    ),
                                )
                                .unwrap();
                            assert!(
                                (actual.log_prob - reference.log_prob).abs() < 1e-8,
                                "{boundary:?},sd={sd},drift={drift},width={width},t={t}: {actual:?} vs {reference:?}"
                            );
                            for (a, b) in
                                actual.grad_array().into_iter().zip(reference.grad_array())
                            {
                                assert!(
                                    (a - b).abs() < 1e-7 * (1. + b.abs()),
                                    "{a} vs {b}: {boundary:?},sd={sd},drift={drift},width={width},t={t}"
                                );
                            }
                            successes += 1;
                        }
                    }
                }
            }
        }
        assert!(successes >= 100, "only {successes} supported points");
    }
    #[test]
    fn benchmark_case_uses_the_image_antiderivative() {
        let obs = WienerObservation {
            rt: 0.05,
            boundary: Boundary::Upper,
        };
        let p = Wiener7Params::with_params(1., 0., 0.5, 2., 0.1, 0., 0.5).unwrap();
        let core = Wiener7.core(&obs, &p, 1e-10).unwrap();
        assert!(Wiener7.analytic_image_start(&core, &obs).is_some());
    }
}
