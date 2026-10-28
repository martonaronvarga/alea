//! Analytic CDF parameter scores for uniform nondecision-time averaging.
use super::*;
impl Wiener7 {
    fn analytic_drift_time(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        let high = obs.rt - core.tau0;
        let low = high - core.st0;
        let budget = 0.01 * core.eps_series.min(core.opts.req_rel_error);
        let hi = super::drift_cdf::evaluate(high, core, obs.boundary, budget)?;
        let lo = if low > 0. {
            Some(super::drift_cdf::evaluate(low, core, obs.boundary, budget)?)
        } else {
            None
        };
        let log = if let Some(lo) = &lo {
            super::super::cdf::log_sub(hi.log_prob, lo.log_prob)?
        } else {
            hi.log_prob
        };
        let wh = (hi.log_prob - log).exp();
        let wl = lo.as_ref().map_or(0., |lo| (lo.log_prob - log).exp());
        let density_error = 64. * f64::EPSILON * (wh + wl)
            + wh * hi.error[0].exp_m1()
            + lo.as_ref().map_or(0., |lo| wl * lo.error[0].exp_m1());
        if density_error >= 1. || -(-density_error).ln_1p() > core.opts.req_rel_error {
            return None;
        }
        let mut scores = [0.; 7];
        for (i, score) in scores[..5].iter_mut().enumerate() {
            *score = wh * hi.scores[i] - lo.as_ref().map_or(0., |lo| wl * lo.scores[i]);
            let numerator = |cdf: &super::drift_cdf::Cdf| {
                cdf.error[0].exp_m1() * cdf.scores[i].abs() + cdf.error[0].exp() * cdf.error[i + 1]
            };
            let error = (wh * numerator(&hi)
                + lo.as_ref().map_or(0., |lo| wl * numerator(lo))
                + score.abs() * density_error)
                / (1. - density_error);
            if !score.is_finite() || error > core.opts.req_rel_error * score.abs().max(1.) {
                return None;
            }
        }
        let ratio = if low > 0. {
            let p = Wiener5Params::from_array([core.alpha, 0., core.beta0, core.delta, core.sv]);
            let event = WienerObservation {
                rt: low,
                boundary: obs.boundary,
            };
            let (e, r) = Wiener5.try_fused_with_truncation(&event, &p, budget).ok()?;
            let ratio = (e.log_prob - log).exp();
            let error = ratio * (r.log_prob.exp_m1() + density_error) / (1. - density_error);
            (ratio, error)
        } else {
            (0., 0.)
        };
        scores[6] = ratio.0 - core.st0.recip();
        if !scores[6].is_finite()
            || ratio.1 + 64. * f64::EPSILON * (ratio.0 + core.st0.recip())
                > core.opts.req_rel_error * scores[6].abs().max(1.)
        {
            return None;
        }
        Some(Wiener7Eval {
            log_prob: log - core.st0.ln(),
            grad: Wiener7Grad::from_array(scores),
        })
    }

    pub(super) fn analytic_time(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.sw != 0. || core.st0 <= 0. {
            return None;
        }
        if core.st0 < 1e-5 * (obs.rt - core.tau0) {
            return None;
        }
        if core.sv > 0. {
            return self.analytic_drift_time(core, obs);
        }
        let budget = core.eps_series.min(1e-3) * 1e-4;
        let base = Wiener4Params::from_array([core.alpha, core.tau0, core.beta0, core.delta]);
        let interval = Wiener4
            .try_fused_interval(
                obs.rt - core.st0,
                obs.rt,
                obs.boundary,
                &base,
                WienerOptions::new(budget),
            )
            .ok()?;
        let low = obs.rt - core.st0;
        let (ratio, lower_error) = if low <= core.tau0 {
            (0., 0.)
        } else {
            let p = WienerObservation {
                rt: low,
                boundary: obs.boundary,
            };
            let (e, r) = Wiener4.try_fused_with_truncation(&p, &base, budget).ok()?;
            ((e.log_prob - interval.log_prob).exp(), r.log_prob)
        };
        let inv = core.st0.recip();
        let width = ratio - inv;
        // The CDF interval separation controls probability subtraction; check
        // the separate cancellation in the width score before accepting it.
        if !width.is_finite()
            || ratio * (lower_error + budget).exp_m1() + 64. * f64::EPSILON * (ratio + inv)
                > core.eps_series * width.abs().max(1.)
        {
            return None;
        }
        let g = interval.grad;
        Some(Wiener7Eval {
            log_prob: interval.log_prob - core.st0.ln(),
            grad: Wiener7Grad {
                alpha: g.alpha,
                tau: g.tau,
                beta: g.beta,
                delta: g.delta,
                s_delta: 0.,
                s_beta: 0.,
                s_tau: width,
            },
        })
    }
}

#[cfg(test)]
mod long_cdf_tests {
    use super::*;
    #[test]
    fn long_time_cdf_marginalization_matches_ball_integrals() {
        let mut count = 0;
        for line in include_str!("../../../tests/support/wiener7_long_cdf_balls.csv")
            .lines()
            .skip(1)
        {
            let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let obs = WienerObservation {
                rt: x[0],
                boundary: if x[8] == 1. {
                    Boundary::Lower
                } else {
                    Boundary::Upper
                },
            };
            let p = Wiener7Params::with_params(x[1], x[2], x[3], x[4], x[6], x[7], x[5]).unwrap();
            let core = Wiener7.core(&obs, &p, 1e-8).unwrap();
            let value = if x[6] == 0. {
                Wiener7
                    .analytic_time(&core, &obs)
                    .unwrap_or_else(|| panic!("CDF fallback row {count}"))
            } else {
                Wiener7
                    .try_fused(&obs, &p, WienerOptions::new(1e-8))
                    .unwrap_or_else(|e| panic!("row {count}: {e:?}"))
            };
            for (i, actual) in std::iter::once(value.log_prob)
                .chain(value.grad_array())
                .enumerate()
            {
                let reference = x[9 + 2 * i];
                assert!(
                    (actual - reference).abs() <= x[10 + 2 * i] + 1e-8 * reference.abs().max(1.),
                    "row {count}, component {i}: {actual} vs {reference}"
                );
            }
            count += 1;
        }
        assert_eq!(count, 32);
    }
}
