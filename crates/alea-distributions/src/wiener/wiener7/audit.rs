use super::*;

impl Wiener7 {
    // Reference, flat implementation of the 7 parameter log-density with grads.
    // Superseded by `fused`; kept crate-local for validation while the kernel is being audited.
    #[allow(dead_code)]
    #[inline]
    #[must_use]
    pub(crate) fn fused_ref(
        obs: &WienerObservation,
        params: &Wiener7Params,
        eps: f64,
    ) -> (f64, [f64; 7]) {
        // trivial cases & validity
        let alpha = params.base.base.alpha;
        let tau0 = params.base.base.tau;
        let beta0 = params.base.base.beta;
        let delta = params.base.base.delta;
        let sv = params.base.s_delta;
        let sw = params.s_beta;
        let st0 = params.s_tau;

        if unlikely(
            !params.base.base.valid()
                || !sv.is_finite()
                || sv < 0.0
                || !sw.is_finite()
                || sw < 0.0
                || !st0.is_finite()
                || st0 < 0.0
                || !obs.rt.is_finite()
                || !eps.is_finite()
                || eps <= 0.0,
        ) {
            return (f64::NEG_INFINITY, [f64::NAN; 7]);
        }

        // no inter‑trial variability -> delegate to Wiener5
        if sw <= 0.0 && st0 <= 0.0 {
            let e = Wiener5.fused(obs, &params.base, eps);
            let mut grad = [0.0; 7];
            grad[0] = e.grad.alpha;
            grad[1] = e.grad.tau;
            grad[2] = e.grad.beta;
            grad[3] = e.grad.delta;
            grad[4] = e.grad.s_delta;
            // sw & st0 remain 0
            return (e.log_prob, grad);
        }

        if unlikely(obs.rt <= tau0) {
            return (f64::NEG_INFINITY, [f64::NAN; 7]);
        }
        if unlikely(st0 > 0.0 && (obs.rt - tau0) / st0 <= 0.0) {
            return (f64::NEG_INFINITY, [f64::NAN; 7]);
        }

        // optional: check that the whole w‑interval lies inside (0,1)
        if unlikely(sw > 0.0 && (beta0 - sw / 2.0 <= 0.0 || beta0 + sw / 2.0 >= 1.0)) {
            return (f64::NEG_INFINITY, [f64::NAN; 7]);
        }

        // integration setup
        let dim = usize::from(sw > 0.0) + usize::from(st0 > 0.0);
        let xmin = vec![0.0; dim];
        let mut xmax = vec![1.0; dim];

        if st0 > 0.0 {
            let clip = f64::min(1.0, (obs.rt - tau0) / st0);
            xmax[dim - 1] = clip; // τ‑dimension, either index 0 or 1
        }

        let bounds = Bounds::new(&xmin, &xmax);
        let eps_series = eps;

        // relative error = 0.9 * precision  (Stan’s choice)
        let rel_err = 0.9 * eps;
        let opts = Options {
            max_eval: 6000,
            req_abs_error: 0.0,
            req_rel_error: rel_err,
            norm: ErrorNorm::L2,
        };

        // combined integral of density & gradients
        let mut val = [0.0; 6];
        let mut err = [0.0; 6];

        let integrand = |x: &[f64], fval: &mut [f64]| -> i32 {
            let (tau, beta) = if dim == 1 {
                if sw > 0.0 {
                    (tau0, beta0 + sw * (x[0] - 0.5))
                } else {
                    (tau0 + st0 * x[0], beta0)
                }
            } else {
                let beta = beta0 + sw * (x[0] - 0.5);
                let tau = tau0 + st0 * x[1];
                (tau, beta)
            };

            if obs.rt <= tau || beta <= 0.0 || beta >= 1.0 {
                for v in fval.iter_mut() {
                    *v = 0.0;
                }
                return 0;
            }

            let p5 = Wiener5Params::with_params_unchecked(alpha, tau, beta, delta, sv);
            let fused = Wiener5.fused(obs, &p5, eps_series);
            if fused.log_prob.is_finite() {
                let dens = fused.log_prob.exp();
                fval[0] = dens;
                fval[1] = dens * fused.grad.alpha;
                fval[2] = dens * fused.grad.tau;
                fval[3] = dens * fused.grad.beta;
                fval[4] = dens * fused.grad.delta;
                fval[5] = dens * fused.grad.s_delta;
            } else {
                for v in fval.iter_mut() {
                    *v = 0.0;
                }
            }
            0
        };

        if hcubature_into(6, bounds, opts, &mut val, &mut err, integrand).is_err() {
            return (f64::NEG_INFINITY, [f64::NAN; 7]);
        }

        let total_density: f64 = val[0];
        if unlikely(total_density <= 0.0) {
            return (f64::NEG_INFINITY, [f64::NAN; 7]);
        }
        let log_density = total_density.ln();

        let mut grad = [f64::NAN; 7];
        grad[0] = val[1] / total_density; // alpha
        grad[1] = val[2] / total_density; // tau
        grad[2] = val[3] / total_density; // beta
        grad[3] = val[4] / total_density; // delta
        grad[4] = val[5] / total_density; // s_delta

        // gradient for s_beta
        if sw == 0.0 {
            grad[5] = 0.0;
        } else if st0 == 0.0 {
            // no τ‑variability → pointwise endpoint formula
            let low = beta0 - sw / 2.0;
            let high = beta0 + sw / 2.0;
            let p5_low = Wiener5Params::with_params_unchecked(alpha, tau0, low, delta, sv);
            let p5_high = Wiener5Params::with_params_unchecked(alpha, tau0, high, delta, sv);
            let dens_low = Self::wiener5_density(obs, &p5_low, eps_series);
            let dens_high = Self::wiener5_density(obs, &p5_high, eps_series);
            let derivative = 0.5 * (dens_low + dens_high) / sw;
            grad[5] = derivative / total_density - 1.0 / sw;
        } else {
            // τ‑variability present → integrate over τ
            let xmax_tau = [f64::min(1.0, (obs.rt - tau0) / st0)];
            if xmax_tau[0] <= 0.0 {
                grad[5] = 0.0;
            } else {
                let bounds_tau = Bounds::new(&[0.0], &xmax_tau);
                let mut val_sw = [0.0f64; 1];
                let mut err_sw = [0.0f64; 1];
                let integrand_sw = |x: &[f64], fv: &mut [f64]| -> i32 {
                    let tau = tau0 + st0 * x[0];
                    if obs.rt <= tau {
                        fv[0] = 0.0;
                        return 0;
                    }
                    let low = beta0 - sw / 2.0;
                    let high = beta0 + sw / 2.0;
                    let p5_low = Wiener5Params::with_params_unchecked(alpha, tau, low, delta, sv);
                    let p5_high = Wiener5Params::with_params_unchecked(alpha, tau, high, delta, sv);
                    let d_low = Self::wiener5_density(obs, &p5_low, eps_series);
                    let d_high = Self::wiener5_density(obs, &p5_high, eps_series);
                    fv[0] = 0.5 * (d_low + d_high) / sw;
                    0
                };
                if hcubature_into(1, bounds_tau, opts, &mut val_sw, &mut err_sw, integrand_sw)
                    .is_err()
                {
                    return (f64::NEG_INFINITY, [f64::NAN; 7]);
                }
                let derivative = val_sw[0];
                grad[5] = derivative / total_density - 1.0 / sw;
            }
        }

        // gradient for s_tau
        if st0 == 0.0 {
            grad[6] = 0.0;
        } else {
            let t0plus = tau0 + st0;
            if obs.rt - t0plus <= 0.0 {
                grad[6] = -1.0 / st0;
            } else {
                let f_end: f64;
                if sw == 0.0 {
                    let p5 = Wiener5Params::with_params_unchecked(alpha, t0plus, beta0, delta, sv);
                    f_end = Self::wiener5_density(obs, &p5, eps_series);
                } else {
                    // integrate over ω at τ = t0 + st0
                    let bounds_w = Bounds::new(&[0.0], &[1.0]);
                    let mut val_f = [0.0f64; 1];
                    let mut err_f = [0.0f64; 1];
                    let integrand_f = |x: &[f64], fv: &mut [f64]| -> i32 {
                        let w = beta0 + sw * (x[0] - 0.5);
                        if w <= 0.0 || w >= 1.0 {
                            fv[0] = 0.0;
                            return 0;
                        }
                        let p5 = Wiener5Params::with_params_unchecked(alpha, t0plus, w, delta, sv);
                        fv[0] = Self::wiener5_density(obs, &p5, eps_series);
                        0
                    };
                    if hcubature_into(1, bounds_w, opts, &mut val_f, &mut err_f, integrand_f)
                        .is_err()
                    {
                        return (f64::NEG_INFINITY, [f64::NAN; 7]);
                    }
                    f_end = val_f[0];
                }
                grad[6] = -1.0 / st0 + f_end / (st0 * total_density);
            }
        }
        (log_density, grad)
    }
}
