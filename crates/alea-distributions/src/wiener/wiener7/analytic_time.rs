//! Analytic CDF parameter scores for uniform nondecision-time averaging.
use super::*;
impl Wiener7 {
    pub(super) fn analytic_time(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.sw != 0. || core.sv != 0. || core.st0 <= 0. {
            return None;
        }
        if core.st0 < 1e-5 * (obs.rt - core.tau0) {
            return None;
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
