//! Error accounting for optional adaptive Wiener7 diagnostics.
use super::*;

/// Separate estimated contributions to a Wiener7 log density and its scores.
///
/// The pointwise series envelopes are analytic truncation bounds, but their
/// numerical integrals are estimates. Cubature errors are estimates too.
/// Floating-point error is unassessed. None of these fields is a certified
/// end-to-end enclosure. Score order is [`Wiener7Eval::grad_array`].
#[derive(Clone, Copy, Debug)]
pub struct Wiener7ErrorEstimate {
    pub series_log_prob: f64,
    pub series_scores: [f64; 7],
    pub quadrature_log_prob: f64,
    pub quadrature_scores: [f64; 7],
    pub total_log_prob: f64,
    pub total_scores: [f64; 7],
}

pub(super) fn ratios(value: &[f64; 8], error: &[f64; 8]) -> Option<(f64, [f64; 7])> {
    if value[0] <= 0.0
        || value.iter().any(|x| !x.is_finite())
        || error.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return None;
    }
    let relative = error[0] / value[0];
    let margin = value[0] - error[0];
    if !(0. ..1.).contains(&relative) {
        return None;
    }
    let scores = std::array::from_fn(|i| {
        (error[i + 1] + (value[i + 1] / value[0]).abs() * error[0]) / margin
    });
    let log = -(-relative).ln_1p();
    (log.is_finite() && scores.iter().all(|x| x.is_finite())).then_some((log, scores))
}

impl Wiener7 {
    /// Adaptive evaluation with separate series and quadrature error estimates.
    ///
    /// Integrates pointwise derivative-tail envelopes alongside the likelihood.
    /// Estimates are converted to log-density and score units only after a
    /// positive density margin is established. Fixed quadrature has no embedded
    /// error estimator and is rejected. Floating-point error is excluded.
    pub fn try_fused_with_error_estimate(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        options: WienerOptions,
    ) -> Result<(Wiener7Eval, Wiener7ErrorEstimate)> {
        let options = options.validate()?;
        if options.quadrature != Quadrature::Adaptive {
            return Err(ProbError::InvalidParameters(
                "Wiener7 error estimates require adaptive quadrature".into(),
            ));
        }
        let mut core = self.core(obs, params, options.precision)?;
        core.opts.req_rel_error = 0.05 * options.precision;
        let inner_tolerance = options.series_precision().min(0.1 * options.precision);
        let fail = || {
            ProbError::NumericalError("Wiener7 error estimation failed to establish a finite density margin or meet the requested budget".into())
        };
        if core.dim == 0 {
            let (e, r) = Wiener5.try_fused_with_truncation(obs, &params.base, inner_tolerance)?;
            let mut scores = [0.; 7];
            scores[..5].copy_from_slice(&e.grad_array());
            scores[6] = 0.5 * scores[1];
            let mut errors = [0.; 7];
            errors[..5].copy_from_slice(&r.scores);
            errors[6] = 0.5 * errors[1];
            return Ok((
                Wiener7Eval {
                    log_prob: e.log_prob,
                    grad: Wiener7Grad::from_array(scores),
                },
                Wiener7ErrorEstimate {
                    series_log_prob: r.log_prob,
                    series_scores: errors,
                    quadrature_log_prob: 0.,
                    quadrature_scores: [0.; 7],
                    total_log_prob: r.log_prob,
                    total_scores: errors,
                },
            ));
        }
        // A preliminary scaled evaluation locates an effective common scale.
        // It is not used as a numerical reference or as an error estimate.
        let scale = self.try_fused(obs, params, options)?.log_prob;
        let mut val = [0.; 16];
        let mut err = [0.; 16];
        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (_, beta) = Self::map_point(&core, x);
            let ti = usize::from(core.sw > 0.);
            let t = obs.rt - core.tau0 - if core.st0 > 0. { core.st0 * x[ti] } else { 0. };
            if t <= 0. {
                fv.fill(0.);
                return 0;
            }
            let p = Wiener5Params::from_array([core.alpha, 0., beta, core.delta, core.sv]);
            let o = WienerObservation {
                rt: t,
                boundary: obs.boundary,
            };
            let Ok((e, r)) = Wiener5.try_fused_with_truncation(&o, &p, inner_tolerance) else {
                fv.fill(f64::NAN);
                return 1;
            };
            let weight = (e.log_prob - scale).exp();
            let mut scores = [0.; 7];
            scores[..5].copy_from_slice(&e.grad_array());
            let mut tails = [0.; 7];
            tails[..5].copy_from_slice(&r.scores);
            if core.sw > 0. {
                scores[5] = (x[0] - 0.5) * scores[2];
                tails[5] = (x[0] - 0.5).abs() * tails[2];
            }
            let time_weight = if core.st0 > 0. { x[ti] } else { 0.5 };
            scores[6] = time_weight * scores[1];
            tails[6] = time_weight * tails[1];
            fv[0] = weight;
            // Bj/f <= the reported score-error bound. This deliberately
            // overestimates derivative numerator tails before integration.
            fv[8] = weight * (-(-r.log_prob).exp_m1());
            for i in 0..7 {
                fv[i + 1] = weight * scores[i];
                fv[i + 9] = weight * tails[i];
            }
            if fv.iter().any(|x| !x.is_finite()) {
                return 1;
            }
            0
        };
        hcubature_into(
            16,
            Bounds::try_new(&core.xmin, &core.xmax).expect("validated cubature bounds"),
            core.opts,
            &mut val,
            &mut err,
            integrand,
        )
        .map_err(|_| fail())?;
        let value: [f64; 8] = std::array::from_fn(|i| val[i]);
        let series = std::array::from_fn(|i| val[i + 8].abs() + err[i + 8]);
        let quad = std::array::from_fn(|i| err[i]);
        let total = std::array::from_fn(|i| series[i] + quad[i]);
        let (sl, ss) = ratios(&value, &series).ok_or_else(fail)?;
        let (ql, qs) = ratios(&value, &quad).ok_or_else(fail)?;
        let (tl, ts) = ratios(&value, &total).ok_or_else(fail)?;
        let mut e = Self::from_integrals(value);
        e.log_prob += scale;
        if !e.log_prob.is_finite()
            || tl > options.precision
            || ts
                .iter()
                .zip(e.grad_array())
                .any(|(b, g)| *b > options.precision * g.abs().max(1.))
        {
            return Err(ProbError::NumericalError(format!(
                "Wiener7 estimated errors exceed budget: log={tl}, scores={ts:?}, tolerance={}",
                options.precision
            )));
        }
        Ok((
            e,
            Wiener7ErrorEstimate {
                series_log_prob: sl,
                series_scores: ss,
                quadrature_log_prob: ql,
                quadrature_scores: qs,
                total_log_prob: tl,
                total_scores: ts,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ratio_estimates_require_positive_separation() {
        assert!(ratios(&[1.; 8], &[1.; 8]).is_none());
        let (l, s) = ratios(&[2.; 8], &[0.1; 8]).unwrap();
        assert!((l + 0.95_f64.ln()).abs() < 1e-15);
        assert!(s.iter().all(|x| (*x - 0.2 / 1.9).abs() < 1e-15));
    }
}
