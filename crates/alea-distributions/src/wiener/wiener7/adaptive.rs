use super::*;

impl Wiener7 {
    pub(in crate::wiener) fn eval_fused_adaptive(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        if let Some(e) = self.analytic_time(core, obs) {
            return e;
        }
        // Numerical integration is cheaper for almost constant start intervals.
        if core.sw > 1e-6
            && let Some(e) = self.analytic_start(core, obs)
        {
            return e;
        }
        if Self::prefer_reduced_time(core, obs)
            && let Some(e) = self.reduced_time_integral(core, obs)
        {
            return e;
        }
        let e = self.integrate_fused_adaptive(core, obs);
        if e.log_prob.is_finite() && e.grad_array().iter().all(|g| g.is_finite()) {
            e
        } else {
            self.scaled_adaptive(core, obs)
        }
    }

    /// Integrate only the density over the hypercube using adaptive cubature.
    #[allow(dead_code)]
    pub(in crate::wiener) fn eval_density_adaptive(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> f64 {
        // Numerical integration is cheaper for almost constant start intervals.
        if core.sw > 1e-6
            && let Some(e) = self.analytic_start(core, obs)
        {
            return e.log_prob.exp();
        }
        let bounds = Bounds::new(&core.xmin, &core.xmax);
        let mut val = [0.0f64; 1];
        let mut err = [0.0f64; 1];

        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let Some((lp, _)) = Self::point_log_scores(core, obs, x) else {
                return 1;
            };
            fv[0] = lp.exp();
            0
        };
        if hcubature_into(1, bounds, core.opts, &mut val, &mut err, integrand).is_err() {
            return 0.0;
        }
        val[0]
    }

    /// Integrate density and all seven derivatives in one traversal.
    pub(super) fn integrate_fused_adaptive(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        let center: [f64; 2] = std::array::from_fn(|i| 0.5 * (core.xmin[i] + core.xmax[i]));
        let Some((log_scale, center_scores)) = Self::point_log_scores(core, obs, &center) else {
            return fail_eval_7();
        };
        let norms = center_scores.map(|score| score.abs().max(1.0));
        let mut options = core.opts;
        // L2 convergence can hide a small inaccurate derivative behind a large
        // one. Normalize the component budgets and use individual errors, with
        // room for both numerator and denominator uncertainty in a score ratio.
        options.norm = ErrorNorm::Individual;
        options.req_rel_error *= 0.25;
        let volume: f64 = (0..core.dim).map(|i| core.xmax[i] - core.xmin[i]).product();
        options.req_abs_error = options.req_rel_error * volume;
        let mut val = [0.; 8];
        let mut err = [0.; 8];
        let integrand = |x: &[f64], fv: &mut [f64]| {
            let Some((log_prob, scores)) = Self::point_log_scores(core, obs, x) else {
                fv.fill(f64::NAN);
                return 1;
            };
            let weight = (log_prob - log_scale).exp();
            fv[0] = weight;
            for i in 0..7 {
                fv[i + 1] = weight * (scores[i] / norms[i]);
            }
            i32::from(fv.iter().any(|x| !x.is_finite()))
        };
        if hcubature_into(
            8,
            Bounds::new(&core.xmin, &core.xmax),
            options,
            &mut val,
            &mut err,
            integrand,
        )
        .is_err()
        {
            return fail_eval_7();
        }
        for i in 0..7 {
            val[i + 1] *= norms[i];
            err[i + 1] *= norms[i];
        }
        let Some(mut evaluation) =
            Self::checked_adaptive_integrals(val, err, core.opts.req_rel_error)
        else {
            return fail_eval_7();
        };
        evaluation.log_prob += log_scale;
        evaluation
    }

    fn checked_adaptive_integrals(
        val: [f64; 8],
        err: [f64; 8],
        tolerance: f64,
    ) -> Option<Wiener7Eval> {
        let (log_error, score_errors) = super::error::ratios(&val, &err)?;
        let evaluation = Self::from_integrals(val);
        if log_error > tolerance
            || score_errors
                .iter()
                .zip(evaluation.grad_array())
                .any(|(bound, score)| *bound > tolerance * score.abs().max(1.0))
        {
            return None;
        }
        Some(evaluation)
    }
}

#[cfg(test)]
mod empirical_proposals {
    use super::*;
    #[test]
    fn ordinary_empirical_proposals_retain_finite_score_evaluations() {
        for (rt, p) in [
            (
                0.704,
                [
                    2.280343192665146,
                    0.3567360349793536,
                    0.45271564597586883,
                    0.6691910830223944,
                    0.3264538325629062,
                    0.026023180952928007,
                    0.3219602280500041,
                ],
            ),
            (
                0.704,
                [
                    2.48296675017064,
                    0.3491597946156447,
                    0.5205460438528553,
                    0.534316083528779,
                    0.4638369955070195,
                    0.36485501329404224,
                    0.3035676902842804,
                ],
            ),
            (
                0.744,
                [
                    2.3187699090864817,
                    0.42948597427765006,
                    0.5083923294973622,
                    0.2936319691833224,
                    0.5242768141735128,
                    0.3469938025376835,
                    0.3055838476685097,
                ],
            ),
            (
                0.675,
                [
                    2.3187426673143072,
                    0.3752768768780298,
                    0.4873695646912378,
                    -0.07516416990648225,
                    0.11546556159542382,
                    0.4529638265574388,
                    0.2646839176511719,
                ],
            ),
        ] {
            let event = WienerObservation {
                rt,
                boundary: Boundary::Lower,
            };
            let parameters =
                Wiener7Params::with_params(p[0], p[1], p[2], p[3], p[5], p[6], p[4]).unwrap();
            let options = WienerOptions::new(1e-6);
            let result = Wiener7.try_fused(&event, &parameters, options).unwrap();
            let reference = Wiener7
                .try_fused(
                    &event,
                    &parameters,
                    WienerOptions::new(1e-12)
                        .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 }),
                )
                .unwrap();
            for (actual, expected) in std::iter::once(result.log_prob)
                .chain(result.grad_array())
                .zip(std::iter::once(reference.log_prob).chain(reference.grad_array()))
            {
                assert!(
                    (actual - expected).abs() <= 1e-6 * expected.abs().max(1.),
                    "{parameters:?}: {actual} vs {expected}"
                );
            }
        }
    }
}
