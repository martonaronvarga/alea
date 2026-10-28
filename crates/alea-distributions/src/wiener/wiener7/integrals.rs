//! Differentiate the fixed unit-cube mapping before integrating width scores.
use super::*;

impl Wiener7 {
    /// At clipped time support the moving face has zero density, so no boundary
    /// contribution is lost. The right time-width derivative at zero uses x=1/2.
    pub(super) fn point_log_scores(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: &[f64],
    ) -> Option<(f64, [f64; 7])> {
        let (event, params) = Self::point_model(core, obs, x);
        if event.rt <= 0. {
            return Some((f64::NEG_INFINITY, [0.; 7]));
        }
        let e = Wiener5.fused(&event, &params, core.eps_series);
        if !e.log_prob.is_finite() || e.grad_array().iter().any(|v| !v.is_finite()) {
            return None;
        }
        let time_index = usize::from(core.sw > 0.);
        let mut scores = [0.0; 7];
        scores[..5].copy_from_slice(&e.grad_array());
        if core.sw > 0.0 {
            scores[5] = (x[0] - 0.5) * e.grad.beta;
        }
        scores[6] = if core.st0 > 0.0 {
            x[time_index] * e.grad.tau
        } else {
            0.5 * e.grad.tau
        };
        Some((e.log_prob, scores))
    }

    /// Evaluate the density at a trial point without constructing any scores.
    pub(super) fn point_log_density(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: &[f64],
    ) -> Option<f64> {
        let (event, params) = Self::point_model(core, obs, x);
        if event.rt <= 0. {
            return Some(f64::NEG_INFINITY);
        }
        let value = Wiener5.density_from_parts(event.rt, event.boundary, &params, core.eps_series);
        value.is_finite().then_some(value)
    }

    #[inline]
    fn point_model(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: &[f64],
    ) -> (WienerObservation, Wiener5Params) {
        let (_, beta) = Self::map_point(core, x);
        let time_index = usize::from(core.sw > 0.);
        let t = obs.rt
            - core.tau0
            - if core.st0 > 0. {
                core.st0 * x[time_index]
            } else {
                0.
            };
        (
            WienerObservation {
                rt: t,
                boundary: obs.boundary,
            },
            Wiener5Params::with_params_unchecked(core.alpha, 0., beta, core.delta, core.sv),
        )
    }

    pub(super) fn from_integrals(val: [f64; 8]) -> Wiener7Eval {
        if val[0] <= 0.0 || val.iter().any(|v| !v.is_finite()) {
            return fail_eval_7();
        }
        Wiener7Eval {
            log_prob: val[0].ln(),
            grad: Wiener7Grad::from_array(std::array::from_fn(|i| val[i + 1] / val[0])),
        }
    }

    pub(super) fn wiener7_integrand(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: &[f64],
        fv: &mut [f64],
    ) -> i32 {
        let Some((lp, scores)) = Self::point_log_scores(core, obs, x) else {
            fv.fill(f64::NAN);
            return 1;
        };
        let density = lp.exp();
        fv[0] = density;
        for (dst, g) in fv[1..].iter_mut().zip(scores) {
            *dst = density * g;
        }
        0
    }
}
