use super::*;

impl Wiener7 {
    pub(in crate::wiener) fn eval_fused_adaptive(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        let e = self.eval_fused_adaptive_raw(core, obs);
        if e.log_prob.is_finite()
            && e.log_prob >= f64::MIN_POSITIVE.ln()
            && e.grad_array().iter().all(|g| g.is_finite())
        {
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
        let bounds = Bounds::new(&core.xmin, &core.xmax);
        let mut val = [0.0f64; 1];
        let mut err = [0.0f64; 1];

        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (tau, beta) = Self::map_point(core, x);
            if beta <= 0.0 || beta >= 1.0 || tau >= f64::INFINITY {
                fv[0] = 0.0;
                return 0;
            }
            let p5 =
                Wiener5Params::with_params_unchecked(core.alpha, tau, beta, core.delta, core.sv);
            fv[0] = Self::wiener5_density(obs, &p5, core.eps_series);
            0
        };
        if hcubature_into(1, bounds, core.opts, &mut val, &mut err, integrand).is_err() {
            return 0.0;
        }
        val[0]
    }

    /// Integrate density + 5 inner-parameter gradient components with adaptive cubature.
    /// Then compute s_beta and s_tau derivatives separately.
    #[allow(dead_code)]
    fn eval_fused_adaptive_raw(&self, core: &Wiener7Core, obs: &WienerObservation) -> Wiener7Eval {
        let bounds = Bounds::new(&core.xmin, &core.xmax);
        let mut val = [0.0; 6];
        let mut err = [0.0; 6];

        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (tau, beta) = Self::map_point(core, x);
            if beta <= 0.0 || beta >= 1.0 {
                for v in fv.iter_mut() {
                    *v = 0.0;
                }
                return 0;
            }
            let p5 =
                Wiener5Params::with_params_unchecked(core.alpha, tau, beta, core.delta, core.sv);
            let fused = Wiener5.fused(obs, &p5, core.eps_series);
            if fused.log_prob.is_finite() {
                let dens = fused.log_prob.exp();
                fv[0] = dens;
                fv[1] = dens * fused.grad.alpha;
                fv[2] = dens * fused.grad.tau;
                fv[3] = dens * fused.grad.beta;
                fv[4] = dens * fused.grad.delta;
                fv[5] = dens * fused.grad.s_delta;
            } else {
                for v in fv.iter_mut() {
                    *v = 0.0;
                }
            }

            0
        };
        if hcubature_into(6, bounds, core.opts, &mut val, &mut err, integrand).is_err() {
            return fail_eval_7();
        }

        let total_density = val[0];
        if unlikely(total_density <= 0.0) {
            return fail_eval_7();
        }
        let log_density = total_density.ln();

        let mut grad = [f64::NAN; 7];
        grad[0] = val[1] / total_density; // alpha
        grad[1] = val[2] / total_density; // tau
        grad[2] = val[3] / total_density; // beta
        grad[3] = val[4] / total_density; // delta
        grad[4] = val[5] / total_density; // s_delta

        // s_beta gradient
        if core.sw == 0.0 {
            grad[5] = 0.0;
        } else if core.st0 == 0.0 {
            let low = core.beta0 - core.sw / 2.0;
            let high = core.beta0 + core.sw / 2.0;
            let p5_low = Wiener5Params::with_params_unchecked(
                core.alpha, core.tau0, low, core.delta, core.sv,
            );
            let p5_high = Wiener5Params::with_params_unchecked(
                core.alpha, core.tau0, high, core.delta, core.sv,
            );
            let d_low = Self::wiener5_density(obs, &p5_low, core.eps_series);
            let d_high = Self::wiener5_density(obs, &p5_high, core.eps_series);
            grad[5] = 0.5 * (d_low + d_high) / (core.sw * total_density) - 1.0 / core.sw;
        } else {
            let tau_max_idx = usize::from(core.sw > 0.0 && core.st0 > 0.0);
            let tau_max = [core.xmax[tau_max_idx]];
            if tau_max[0] <= 0.0 {
                grad[5] = 0.0;
            } else {
                let mut val_sw = [0.0f64; 1];
                let mut err_sw = [0.0f64; 1];
                let bounds_tau = Bounds::new(&[0.0], &tau_max);
                let integrand_sw = |x: &[f64], fv: &mut [f64]| -> i32 {
                    let tau = core.tau0 + core.st0 * x[0];
                    let low = core.beta0 - core.sw / 2.0;
                    let high = core.beta0 + core.sw / 2.0;
                    let p5_low = Wiener5Params::with_params_unchecked(
                        core.alpha, tau, low, core.delta, core.sv,
                    );
                    let p5_high = Wiener5Params::with_params_unchecked(
                        core.alpha, tau, high, core.delta, core.sv,
                    );
                    let d_low = Self::wiener5_density(obs, &p5_low, core.eps_series);
                    let d_high = Self::wiener5_density(obs, &p5_high, core.eps_series);
                    fv[0] = 0.5 * (d_low + d_high) / core.sw;
                    0
                };
                if hcubature_into(
                    1,
                    bounds_tau,
                    core.opts,
                    &mut val_sw,
                    &mut err_sw,
                    integrand_sw,
                )
                .is_err()
                {
                    return fail_eval_7();
                }
                grad[5] = val_sw[0] / total_density - 1.0 / core.sw;
            }
        }

        // s_tau gradient
        if core.st0 == 0.0 {
            grad[6] = 0.5 * grad[1];
        } else {
            let t0plus = core.tau0 + core.st0;
            if obs.rt - t0plus <= 0.0 {
                grad[6] = -1.0 / core.st0;
            } else {
                let f_end: f64;
                if core.sw == 0.0 {
                    let p5 = Wiener5Params::with_params_unchecked(
                        core.alpha, t0plus, core.beta0, core.delta, core.sv,
                    );
                    f_end = Self::wiener5_density(obs, &p5, core.eps_series);
                } else {
                    let mut val_f = [0.0f64; 1];
                    let mut err_f = [0.0f64; 1];
                    let bounds_w = Bounds::new(&[0.0], &[1.0]);
                    let integrand_f = |x: &[f64], fv: &mut [f64]| -> i32 {
                        let w = core.beta0 + core.sw * (x[0] - 0.5);
                        if w <= 0.0 || w >= 1.0 {
                            fv[0] = 0.0;
                            return 0;
                        }
                        let p5 = Wiener5Params::with_params_unchecked(
                            core.alpha, t0plus, w, core.delta, core.sv,
                        );
                        fv[0] = Self::wiener5_density(obs, &p5, core.eps_series);
                        0
                    };
                    if hcubature_into(1, bounds_w, core.opts, &mut val_f, &mut err_f, integrand_f)
                        .is_err()
                    {
                        return fail_eval_7();
                    }
                    f_end = val_f[0];
                }
                grad[6] = -1.0 / core.st0 + f_end / (core.st0 * total_density);
            }
        }

        Wiener7Eval {
            log_prob: log_density,
            grad: Wiener7Grad::from_array(grad),
        }
    }
}
