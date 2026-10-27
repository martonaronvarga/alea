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
        // The bounded moment expansion also handles almost constant intervals.
        if core.sw > 0.
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
        // The bounded moment expansion also handles almost constant intervals.
        if core.sw > 0.
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
        let mut norms = center_scores.map(|score| score.abs().max(1.0));
        if core.st0 >= obs.rt - core.tau0 {
            norms[6] = core.st0.recip().max(1.0);
        }
        let mut log_scale = log_scale;
        let volume: f64 = (0..core.dim).map(|i| core.xmax[i] - core.xmin[i]).product();
        for attempt in 0..2 {
            let Some((value, error)) =
                Self::normalized_adaptive_integrals(core, obs, log_scale, norms, attempt + 1)
            else {
                return fail_eval_7();
            };
            if let Some(mut evaluation) =
                Self::checked_adaptive_integrals(value, error, core.opts.req_rel_error)
            {
                evaluation.log_prob += log_scale;
                return evaluation;
            }
            // The midpoint can overestimate a score, particularly on clipped
            // support. Recalibrate once from the computed integrals instead of
            // weakening the final ratio budget or repeatedly splitting boxes.
            if value[0] <= 0. || value.iter().any(|x| !x.is_finite()) {
                return fail_eval_7();
            }
            norms = std::array::from_fn(|i| (value[i + 1] / value[0]).abs().max(1.0));
            log_scale += value[0].ln() - volume.ln();
            if !log_scale.is_finite() || norms.iter().any(|x| !x.is_finite()) {
                return fail_eval_7();
            }
        }
        fail_eval_7()
    }

    fn normalized_adaptive_integrals(
        core: &Wiener7Core,
        obs: &WienerObservation,
        log_scale: f64,
        norms: [f64; 7],
        work_multiplier: usize,
    ) -> Option<([f64; 8], [f64; 8])> {
        let mut options = core.opts;
        options.max_eval = options.max_eval.saturating_mul(work_multiplier);
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
                // With clipped time support, all width dependence is the 1/h
                // normalization. Its exact score avoids integrating a redundant
                // cancellation-prone numerator. Fixed rules keep their own VJP.
                let score = if i == 6 && core.st0 >= obs.rt - core.tau0 {
                    -core.st0.recip()
                } else {
                    scores[i]
                };
                fv[i + 1] = weight * (score / norms[i]);
            }
            i32::from(fv.iter().any(|x| !x.is_finite()))
        };
        let status = hcubature_into(
            8,
            Bounds::new(&core.xmin, &core.xmax),
            options,
            &mut val,
            &mut err,
            integrand,
        );
        if status.is_err() {
            return None;
        }
        for i in 0..7 {
            val[i + 1] *= norms[i];
            err[i + 1] *= norms[i];
        }
        Some((val, err))
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
                    2.5617827402056976,
                    0.3980884179105071,
                    0.46827887157650716,
                    0.415936314741221,
                    0.8496832712663716,
                    0.44462996670765725,
                    0.3132678856005957,
                ],
            ),
            (
                0.675,
                [
                    2.7092215216795865,
                    0.34113462875931877,
                    0.4593233880286724,
                    0.16511852804671434,
                    0.39073595979686093,
                    0.46861182956553266,
                    0.3127338429722869,
                ],
            ),
            (
                0.704,
                [
                    2.5789358778334295,
                    0.3700726595378969,
                    0.4715404030069702,
                    0.3766483392727203,
                    0.43404478856973366,
                    0.4197814597726233,
                    0.331465448123216,
                ],
            ),
            (
                0.704,
                [
                    2.3631483212357374,
                    0.4126305678625373,
                    0.4425027031386802,
                    0.6382142920468534,
                    0.3644814966730689,
                    0.40576635202778055,
                    0.31378337174968796,
                ],
            ),
            (
                0.704,
                [
                    2.3254985123722394,
                    0.41074121365474864,
                    0.4761674299360394,
                    0.4484408464571903,
                    0.10646057881334611,
                    0.3966025657516752,
                    0.3302707130314663,
                ],
            ),
            (
                0.704,
                [
                    2.398052699270559,
                    0.3803916734713987,
                    0.4157714820093295,
                    0.8121975854983636,
                    0.4141555857963852,
                    0.3461095632914799,
                    0.32261889897350393,
                ],
            ),
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
            // Independent composite product rule. A single GL25 time panel is
            // insufficient near the clipped time-zero face in these cases.
            let mut parts = Vec::new();
            let panels = 8;
            for i in 0..panels {
                let tau = p[1] + p[6] * i as f64 / panels as f64;
                if tau >= rt {
                    continue;
                }
                let sub = Wiener7Params::with_params(
                    p[0],
                    tau,
                    p[2],
                    p[3],
                    p[5],
                    p[6] / panels as f64,
                    p[4],
                )
                .unwrap();
                let mut value = Wiener7
                    .try_fused(
                        &event,
                        &sub,
                        WienerOptions::new(1e-12)
                            .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 }),
                    )
                    .unwrap();
                value.grad.s_tau = (i as f64 * value.grad.tau + value.grad.s_tau) / panels as f64;
                parts.push(value);
            }
            let scale = parts
                .iter()
                .map(|e| e.log_prob)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = [0.; 8];
            for part in parts {
                let weight = (part.log_prob - scale).exp();
                total[0] += weight;
                for (value, score) in total[1..].iter_mut().zip(part.grad_array()) {
                    *value += weight * score;
                }
            }
            let reference = Wiener7Eval {
                log_prob: scale + (total[0] / panels as f64).ln(),
                grad: Wiener7Grad::from_array(std::array::from_fn(|i| total[i + 1] / total[0])),
            };
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
