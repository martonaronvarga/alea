//! Endpoint sensitivities for the explicitly selected fixed quadrature rule.
use super::*;

impl Wiener7 {
    pub(in crate::wiener) fn eval_fused_from_integrals(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        val: [f64; 6],
        order: usize,
    ) -> Wiener7Eval {
        let total_density = val[0];
        if unlikely(total_density <= 0.0) {
            return fail_eval_7();
        }
        let log_density = total_density.ln();

        let mut grad = [f64::NAN; 7];
        grad[0] = val[1] / total_density;
        grad[1] = val[2] / total_density;
        grad[2] = val[3] / total_density;
        grad[3] = val[4] / total_density;
        grad[4] = val[5] / total_density;

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
            let tau_max = core.xmax[tau_max_idx];
            if tau_max <= 0.0 {
                grad[5] = 0.0;
            } else {
                let sw_integral = Self::fixed_integral_1d(0.0, tau_max, order, |x_tau| {
                    let tau = core.tau0 + core.st0 * x_tau;
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
                    0.5 * (d_low + d_high) / core.sw
                });
                grad[5] = sw_integral / total_density - 1.0 / core.sw;
            }
        }

        if core.st0 == 0.0 {
            grad[6] = 0.0;
        } else {
            let t0plus = core.tau0 + core.st0;
            if obs.rt - t0plus <= 0.0 {
                grad[6] = -1.0 / core.st0;
            } else {
                let f_end = if core.sw == 0.0 {
                    let p5 = Wiener5Params::with_params_unchecked(
                        core.alpha, t0plus, core.beta0, core.delta, core.sv,
                    );
                    Self::wiener5_density(obs, &p5, core.eps_series)
                } else {
                    Self::fixed_integral_1d(0.0, 1.0, order, |x_beta| {
                        let beta = core.beta0 + core.sw * (x_beta - 0.5);
                        if beta <= 0.0 || beta >= 1.0 {
                            return 0.0;
                        }
                        let p5 = Wiener5Params::with_params_unchecked(
                            core.alpha, t0plus, beta, core.delta, core.sv,
                        );
                        Self::wiener5_density(obs, &p5, core.eps_series)
                    })
                };
                grad[6] = -1.0 / core.st0 + f_end / (core.st0 * total_density);
            }
        }

        Wiener7Eval {
            log_prob: log_density,
            grad: Wiener7Grad::from_array(grad),
        }
    }

    // core evaluators
}
