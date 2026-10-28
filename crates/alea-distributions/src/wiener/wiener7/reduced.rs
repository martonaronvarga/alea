//! One-dimensional time integration after analytic uniform-start averaging.
use super::*;

impl Wiener7 {
    /// Positive-time integrand after analytic uniform-start averaging. No
    /// complex analyticity assumption is made at the clipped time-zero face.
    pub(super) fn averaged_time_log_scores(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: f64,
    ) -> Option<(f64, [f64; 7])> {
        Self::averaged_time_components::<true>(core, obs, x)
    }

    pub(super) fn averaged_time_log_density(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: f64,
    ) -> Option<f64> {
        Self::averaged_time_components::<false>(core, obs, x).map(|e| e.0)
    }

    fn averaged_time_components<const SCORES: bool>(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: f64,
    ) -> Option<(f64, [f64; 7])> {
        let t = obs.rt - core.tau0 - core.st0 * x;
        if t <= 0. {
            return Some((f64::NEG_INFINITY, [0.; 7]));
        }
        let mut point = core.clone();
        point.tau0 = 0.;
        point.st0 = 0.;
        point.dim = 1;
        point.xmin = [0.; 2];
        point.xmax = [1.; 2];
        point.eps_series = point.eps_series.min(0.05 * core.opts.req_rel_error);
        let event = WienerObservation {
            rt: t,
            boundary: obs.boundary,
        };
        let value = Wiener7.analytic_start_components::<SCORES>(&point, &event)?;
        let mut scores = value.grad_array();
        scores[6] = x * scores[1];
        Some((value.log_prob, scores))
    }
    /// Try the analytic start average before integrating both coordinates.
    /// Unsupported or poorly conditioned moments retain the other paths.
    pub(super) fn prefer_reduced_time(core: &Wiener7Core, obs: &WienerObservation) -> bool {
        let t = obs.rt - core.tau0;
        let curvature = (core.alpha * core.sw).powi(2) / (2. * t * (1. + core.sv * core.sv * t));
        core.st0 > 0.
            && curvature.is_finite()
            && (curvature >= 0.01 || t / (core.alpha * core.alpha) >= 0.15)
    }

    pub(super) fn reduced_time_integral(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        self.reduced_time_components::<true>(core, obs)
    }

    pub(super) fn reduced_time_density(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<f64> {
        self.reduced_time_components::<false>(core, obs)
            .map(|e| e.log_prob)
    }

    fn reduced_time_components<const SCORES: bool>(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.sw <= 0. || core.st0 <= 0. {
            return None;
        }
        let max_time = obs.rt - core.tau0;
        let min_time = max_time - core.st0;
        // The time-zero face is nonanalytic. Keep the existing clipped-support
        // evaluator rather than extending a holomorphic/moment claim to it.
        if min_time <= 0. {
            return None;
        }
        let mut point = core.clone();
        point.tau0 = 0.;
        point.st0 = 0.;
        point.dim = 1;
        point.xmax = [1.; 2];
        point.eps_series = core.eps_series.min(0.1 * core.opts.req_rel_error);
        let observation = |time| WienerObservation {
            rt: time,
            boundary: obs.boundary,
        };
        // Probe the complete time interval before paying for a reduction attempt.
        self.analytic_start_components::<SCORES>(&point, &observation(min_time))?;
        self.analytic_start_components::<SCORES>(&point, &observation(max_time))?;
        let scale = self
            .analytic_start(&point, &observation((min_time + max_time) / 2.))?
            .log_prob;
        let mut val = [0.; 8];
        let mut err = [0.; 8];
        let integrand = |x: &[f64], fv: &mut [f64]| {
            let Some(e) = self.analytic_start_components::<SCORES>(
                &point,
                &observation(max_time - core.st0 * x[0]),
            ) else {
                fv.fill(f64::NAN);
                return 1;
            };
            let weight = (e.log_prob - scale).exp();
            let mut scores = e.grad_array();
            scores[6] = x[0] * scores[1];
            fv[0] = weight;
            if SCORES {
                for i in 0..7 {
                    fv[i + 1] = weight * scores[i];
                }
            }
            if fv.iter().any(|x| !x.is_finite()) {
                1
            } else {
                0
            }
        };
        hcubature_into(
            if SCORES { 8 } else { 1 },
            Bounds::new(&[0.], &[1.]),
            core.opts,
            &mut val[..if SCORES { 8 } else { 1 }],
            &mut err[..if SCORES { 8 } else { 1 }],
            integrand,
        )
        .ok()?;
        let (log_error, errors) = super::error::ratios(&val, &err)?;
        let mut e = Self::from_integrals(val);
        e.log_prob += scale;
        if !e.log_prob.is_finite()
            || log_error > core.opts.req_rel_error
            || (SCORES
                && errors
                    .iter()
                    .zip(e.grad_array())
                    .any(|(b, g)| *b > core.opts.req_rel_error * g.abs().max(1.)))
        {
            return None;
        }
        Some(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn density_reduction_uses_scalar_cubature_buffers() {
        let obs = WienerObservation {
            rt: 0.01,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(1., 0., 0.5, 2., 0.4, 0.001, 0.5).unwrap();
        let core = Wiener7.core(&obs, &params, 1e-9).unwrap();
        let value = Wiener7
            .reduced_time_density(&core, &obs)
            .expect("supported scalar reduction");
        let reference = Wiener7.reduced_time_integral(&core, &obs).unwrap();
        assert!((value - reference.log_prob).abs() < 1e-9);
    }

    #[test]
    fn one_dimensional_reduction_matches_independent_two_dimensional_rule() {
        let mut successes = 0;
        for boundary in [Boundary::Upper, Boundary::Lower] {
            for sd in [0., 0.5, 3.] {
                for time in [0.03, 0.08, 0.3] {
                    let obs = WienerObservation { rt: time, boundary };
                    let p = Wiener7Params::with_params(1., 0., 0.5, 2., 0.1, 0.005, sd).unwrap();
                    let core = Wiener7.core(&obs, &p, 1e-9).unwrap();
                    let Some(actual) = Wiener7.reduced_time_integral(&core, &obs) else {
                        continue;
                    };
                    let reference = Wiener7
                        .try_fused(
                            &obs,
                            &p,
                            WienerOptions::new(1e-12)
                                .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 }),
                        )
                        .unwrap();
                    assert!((actual.log_prob - reference.log_prob).abs() < 1e-8);
                    for (a, b) in actual.grad_array().into_iter().zip(reference.grad_array()) {
                        assert!((a - b).abs() < 1e-7 * (1. + b.abs()), "{a} vs {b}");
                    }
                    successes += 1;
                }
            }
        }
        assert!(successes >= 10, "only {successes} supported points");
    }
    #[test]
    #[ignore = "release-mode timing experiment, not a CI performance assertion"]
    fn compare_analytic_and_full_adaptive_work() {
        use std::hint::black_box;
        use std::time::Instant;
        for (t, h, time_width) in [
            (0.05, 1e-8, 0.),
            (0.1, 1e-8, 0.),
            (0.05, 0.1, 0.),
            (0.05, 0.1, 0.005),
            (0.01, 0.4, 0.001),
        ] {
            let obs = WienerObservation {
                rt: t,
                boundary: Boundary::Upper,
            };
            let p = Wiener7Params::with_params(1., 0., 0.5, 2., h, time_width, 0.5).unwrap();
            let core = Wiener7.core(&obs, &p, 1e-9).unwrap();
            let full = || {
                let e = Wiener7.integrate_fused_adaptive(&core, &obs);
                if e.log_prob.is_finite()
                    && e.log_prob >= f64::MIN_POSITIVE.ln()
                    && e.grad_array().iter().all(|x| x.is_finite())
                {
                    e
                } else {
                    Wiener7.scaled_adaptive(&core, &obs)
                }
            };
            let reduced = || {
                if time_width == 0. {
                    Wiener7.analytic_start(&core, &obs).unwrap()
                } else {
                    Wiener7.reduced_time_integral(&core, &obs).unwrap()
                }
            };
            let reference = full();
            let alternative = reduced();
            assert!((reference.log_prob - alternative.log_prob).abs() < 1e-7);
            for (name, f) in [
                ("full", &full as &dyn Fn() -> Wiener7Eval),
                ("analytic", &reduced),
            ] {
                let start = Instant::now();
                for _ in 0..200 {
                    black_box(f());
                }
                eprintln!(
                    "t={t},h={h},time_width={time_width},{name},ns_per_call={}",
                    start.elapsed().as_nanos() / 200
                );
            }
        }
    }
}
