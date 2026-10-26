//! Exact spectral averaging over uniform start variability when drift SD is zero.
//! The moment generating function is exp(c*m) sinh(c*h/2)/(c*h/2).
//! Differentiating it gives all parameter scores without endpoint subtraction.
use super::*;
type C = (f64, f64);
fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn scale(a: C, x: f64) -> C {
    (a.0 * x, a.1 * x)
}
fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}
fn divide(a: C, b: C) -> C {
    scale(mul(a, (b.0, -b.1)), 1. / (b.0 * b.0 + b.1 * b.1))
}

/// sinh(z)/z and its derivative, with a regular Taylor branch at zero.
fn sinhc(z: C) -> (C, C) {
    if z.0.hypot(z.1) < 0.5 {
        let z2 = mul(z, z);
        let (mut h, mut d) = ((1., 0.), (0., 0.));
        let mut term = (1., 0.);
        let mut derivative = scale(z, 1. / 3.);
        for j in 1..=12 {
            let n = 2. * j as f64;
            term = scale(mul(term, z2), 1. / (n * (n + 1.)));
            h = add(h, term);
            d = add(d, derivative);
            derivative = scale(mul(derivative, z2), 1. / (n * (n + 3.)));
        }
        (h, d)
    } else {
        let (s, c) = z.1.sin_cos();
        let sh = (z.0.sinh() * c, z.0.cosh() * s);
        let ch = (z.0.cosh() * c, z.0.sinh() * s);
        (
            divide(sh, z),
            divide(add(mul(z, ch), scale(sh, -1.)), mul(z, z)),
        )
    }
}

impl Wiener7 {
    /// Conservative supported region; other regimes retain numerical cubature.
    pub(super) fn analytic_start(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.sw <= 0. || core.st0 != 0. {
            return None;
        }
        if core.sv > 0. {
            return self.analytic_image_start(core, obs);
        }
        let params = Wiener4Params::from_array([core.alpha, core.tau0, core.beta0, core.delta]);
        let base = Wiener4.core(obs, &params, core.eps_series).ok()?;
        let (a, t, u, m, h, v) = (
            base.a,
            base.t,
            base.t_prime,
            base.q_eff,
            core.sw,
            base.v_eff,
        );
        let b = a * v;
        if u < 0.15 {
            return self.analytic_image_start(core, obs);
        }
        if !u.is_finite() || (b * h / 2.).abs() > 50. {
            return None;
        }
        let c1 = 0.5 * PI * PI * u;
        let mut sums = [0.; 8];
        let mut absolute = 0.;
        let qmax = (m + h / 2.).min(1.);
        let amplification = (b * h / 2.).abs();
        for k in 1_usize..=128 {
            let n = k as f64;
            let omega = n * PI;
            let c = (b, omega);
            let (sh, ds) = sinhc(scale(c, h / 2.));
            let phase = if base.w_eff < base.q_eff {
                let (s, c) = (omega * base.w_eff).sin_cos();
                if k.is_multiple_of(2) {
                    (c, -s)
                } else {
                    (-c, s)
                }
            } else {
                let (s, c) = (omega * m).sin_cos();
                (c, s)
            };
            let mm = mul(phase, sh);
            let mc = mul(phase, add(scale(sh, m), scale(ds, h / 2.)));
            let mh = mul(phase, mul(ds, scale(c, 0.5)));
            let mq = mul(c, mm);
            let weight = n * (-(n * n - 1.) * c1).exp();
            let density = weight * mm.1;
            absolute += density.abs();
            sums[0] += density;
            sums[1] += weight * ((-2. / a + omega * omega * t / (a * a * a)) * mm.1 + v * mc.1);
            sums[2] += weight * 0.5 * (v * v + omega * omega / (a * a)) * mm.1;
            sums[3] += -base.beta_sign * weight * mq.1;
            sums[4] += base.delta_sign * weight * (a * mc.1 - v * t * mm.1);
            sums[6] += weight * mh.1;
            sums[7] = 0.5 * sums[2];
            if sums[0] <= 0. || absolute / sums[0] > 1e6 {
                continue;
            }
            let t1 = (super::super::bounds::spectral(1, k, c1)? + amplification).exp();
            let t2 = (super::super::bounds::spectral(2, k, c1)? + amplification).exp();
            let t3 = (super::super::bounds::spectral(3, k, c1)? + amplification).exp();
            let time = 0.5 * (v * v * t1 + PI * PI / (a * a) * t3);
            let start = b.abs() * t1 + PI * t2;
            let errors = [
                t1,
                (2. / a + v.abs() * qmax) * t1 + PI * PI * t / (a * a * a) * t3,
                time,
                start,
                (a * qmax + (v * t).abs()) * t1,
                0.,
                0.5 * start,
                0.5 * time,
            ];
            let Some((log_error, score_error)) = super::error::ratios(&sums, &errors) else {
                continue;
            };
            let tolerance = core.eps_series;
            if log_error > tolerance
                || score_error
                    .iter()
                    .zip(&sums[1..])
                    .any(|(e, d)| *e > tolerance * (d / sums[0]).abs().max(1.))
            {
                continue;
            }
            let log_scale = PI.ln() - 2. * a.ln() - 0.5 * v * v * t + b * m - c1;
            let e = Wiener7Eval {
                log_prob: log_scale + sums[0].ln(),
                grad: Wiener7Grad::from_array(std::array::from_fn(|i| sums[i + 1] / sums[0])),
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
    fn sinhc_limit_and_derivative() {
        assert_eq!(sinhc((0., 0.)), ((1., 0.), (0., 0.)));
        for z in [(0.1, 0.2), (0.5, 1.), (-2., 0.3)] {
            let (_, d) = sinhc(z);
            let h = 1e-6;
            let p = sinhc((z.0 + h, z.1)).0;
            let m = sinhc((z.0 - h, z.1)).0;
            assert!((d.0 - (p.0 - m.0) / (2. * h)).abs() < 1e-8);
            assert!((d.1 - (p.1 - m.1) / (2. * h)).abs() < 1e-8);
        }
    }
    #[test]
    fn closed_form_matches_independent_product_quadrature() {
        let mut count = 0;
        for boundary in [Boundary::Upper, Boundary::Lower] {
            for delta in [-8., -0.5, 0., 1., 8.] {
                for (beta, width) in [(0.2, 0.1), (0.5, 0.8), (0.7, 1e-12)] {
                    for t in [0.22, 0.6, 3.] {
                        let obs = WienerObservation {
                            rt: t + 0.1,
                            boundary,
                        };
                        let p = Wiener7Params::with_params(1.1, 0.1, beta, delta, width, 0., 0.)
                            .unwrap();
                        let core = Wiener7.core(&obs, &p, 1e-11).unwrap();
                        let actual = Wiener7
                            .analytic_start(&core, &obs)
                            .expect("supported analytic domain");
                        let reference = Wiener7
                            .try_fused(
                                &obs,
                                &p,
                                WienerOptions::new(1e-12)
                                    .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 }),
                            )
                            .unwrap();
                        assert!(
                            (actual.log_prob - reference.log_prob).abs() < 2e-9,
                            "{actual:?} vs {reference:?}"
                        );
                        for (a, b) in actual.grad_array().into_iter().zip(reference.grad_array()) {
                            assert!((a - b).abs() < 2e-8 * (1. + b.abs()), "{a} vs {b}");
                        }
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 90);
    }
}
