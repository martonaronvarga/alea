use super::*;

#[cold]
#[inline(never)]
fn fail_eval_7() -> Wiener7Eval {
    Wiener7Eval {
        log_prob: f64::NEG_INFINITY,
        grad: NAN_GRAD_7,
    }
}

#[inline]
fn add_scaled_6(dst: &mut [f64; 6], src: &[f64; 6], scale: f64) {
    dst[0] += scale * src[0];
    dst[1] += scale * src[1];
    dst[2] += scale * src[2];
    dst[3] += scale * src[3];
    dst[4] += scale * src[4];
    dst[5] += scale * src[5];
}
use crate::density::LogDensity;

impl Wiener7 {
    /// Build the integration core.
    #[inline]
    pub(super) fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        eps: f64,
    ) -> Result<Wiener7Core> {
        let alpha = params.base.base.alpha;
        let tau0 = params.base.base.tau;
        let beta0 = params.base.base.beta;
        let delta = params.base.base.delta;
        let sv = params.base.s_delta;
        let sw = params.s_beta;
        let st0 = params.s_tau;

        // Basic validity
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
            return Err(ProbError::InvalidParameters(
                "Wiener7 requires valid base parameters, finite non-negative variability terms, finite reaction time, and finite positive precision"
                    .to_string(),
            ));
        }
        if unlikely(obs.rt <= tau0) {
            return Err(ProbError::OutOfSupport(
                "reaction time must exceed non-decision time".to_string(),
            ));
        }
        if unlikely(sw > 0.0 && (beta0 - sw / 2.0 <= 0.0 || beta0 + sw / 2.0 >= 1.0)) {
            return Err(ProbError::OutOfSupport(
                "starting-point variability interval must stay inside (0, 1)".to_string(),
            ));
        }
        if unlikely(st0 > 0.0 && (obs.rt - tau0) / st0 <= 0.0) {
            return Err(ProbError::OutOfSupport(
                "non-decision-time variability leaves no feasible integration interval".to_string(),
            ));
        }

        let dim = usize::from(sw > 0.0) + usize::from(st0 > 0.0);
        let xmin = [0.0; 2];
        let mut xmax = [1.0; 2];

        if st0 > 0.0 {
            xmax[dim - 1] = f64::min(1.0, (obs.rt - tau0) / st0);
        }

        let eps_series = eps;
        let rel_err = 0.9 * eps; // same as Stan (0.9 * precision)
        let opts = Options {
            max_eval: 6000,
            req_abs_error: 0.0,
            req_rel_error: rel_err,
            norm: ErrorNorm::L2,
        };

        Ok(Wiener7Core {
            alpha,
            tau0,
            beta0,
            delta,
            sv,
            sw,
            st0,
            dim,
            xmin,
            xmax,
            eps_series,
            opts,
        })
    }

    // Log‑density only (gradients = NaN in the returned struct).
    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        eps: f64,
    ) -> Wiener7Eval {
        let core = match self.core(obs, params, eps) {
            Ok(c) => c,
            Err(_) => {
                return fail_eval_7();
            }
        };
        let density = self.eval_density_adaptive(&core, obs);
        Wiener7Eval {
            log_prob: if density > 0.0 {
                density.ln()
            } else {
                f64::NEG_INFINITY
            },
            grad: NAN_GRAD_7,
        }
    }

    #[inline]
    pub fn try_log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        options: WienerOptions,
    ) -> Result<Wiener7Eval> {
        let options = options.validate()?;
        let mut core = self.core(obs, params, options.precision)?;
        core.eps_series = options.series_precision();

        let density = if core.sw == 0.0 && core.st0 == 0.0 {
            Wiener5
                .try_log_prob(obs, &params.base, options)?
                .log_prob
                .exp()
        } else {
            match options.quadrature {
                Quadrature::Adaptive => self.eval_density_adaptive(&core, obs),
                Quadrature::FixedGaussLegendre { order } => {
                    self.eval_density_fixed_order(&core, obs, order)
                }
            }
        };

        if likely(density.is_finite() && density > 0.0) {
            Ok(Wiener7Eval {
                log_prob: density.ln(),
                grad: Wiener7Grad::default(),
            })
        } else {
            Err(ProbError::NumericalError(
                "Wiener7 evaluation produced a non-finite density".to_string(),
            ))
        }
    }

    #[inline]
    pub fn try_fused(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        options: WienerOptions,
    ) -> Result<Wiener7Eval> {
        let options = options.validate()?;
        let mut core = self.core(obs, params, options.precision)?;
        core.eps_series = options.series_precision();

        let eval = if core.sw == 0.0 && core.st0 == 0.0 {
            let e = Wiener5.try_fused(obs, &params.base, options)?;
            Wiener7Eval {
                log_prob: e.log_prob,
                grad: Wiener7Grad {
                    alpha: e.grad.alpha,
                    tau: e.grad.tau,
                    beta: e.grad.beta,
                    delta: e.grad.delta,
                    s_delta: e.grad.s_delta,
                    s_beta: 0.0,
                    s_tau: 0.0,
                },
            }
        } else {
            match options.quadrature {
                Quadrature::Adaptive => self.eval_fused_adaptive(&core, obs),
                Quadrature::FixedGaussLegendre { order } => {
                    self.eval_fused_fixed_order(&core, obs, order)
                }
            }
        };

        if likely(eval.log_prob.is_finite()) {
            Ok(eval)
        } else {
            Err(ProbError::NumericalError(
                "Wiener7 evaluation produced a non-finite log density".to_string(),
            ))
        }
    }

    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener7Params, eps: f64) -> Wiener7Eval {
        let core = match self.core(obs, params, eps) {
            Ok(c) => c,
            Err(_) => {
                return fail_eval_7();
            }
        };

        // Short-circuit only the validated exact no-variability case.
        if core.sw == 0.0 && core.st0 == 0.0 {
            let e = Wiener5.fused(obs, &params.base, eps);

            return Wiener7Eval {
                log_prob: e.log_prob,
                grad: Wiener7Grad {
                    alpha: e.grad.alpha,
                    tau: e.grad.tau,
                    beta: e.grad.beta,
                    delta: e.grad.delta,
                    s_delta: e.grad.s_delta,
                    s_beta: 0.0,
                    s_tau: 0.0,
                },
            };
        }

        self.eval_fused_adaptive(&core, obs)
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn eval_fused(&self, core: &Wiener7Core, obs: &WienerObservation) -> Wiener7Eval {
        self.eval_fused_adaptive(core, obs)
    }

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

    #[cfg(test)]
    const FIXED_QUADRATURE_ORDER: usize = 25;

    pub(super) fn fixed_rule(order: usize) -> (&'static [f64], &'static [f64]) {
        match order {
            1 => (&GL_1_NODES, &GL_1_WTS),
            5 => (&GL_5_NODES, &GL_5_WTS),
            7 => (&GL_7_NODES, &GL_7_WTS),
            15 => (&GL_15_NODES, &GL_15_WTS),
            25 => (&GL_25_NODES, &GL_25_WTS),
            _ => (&GL_25_NODES, &GL_25_WTS),
        }
    }

    #[inline]
    pub(super) fn fixed_interval(node: f64, weight: f64, lo: f64, hi: f64) -> (f64, f64) {
        let width = hi - lo;
        (lo + width * node, width * weight)
    }

    fn fixed_integral_1d<F>(lo: f64, hi: f64, order: usize, mut f: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let (nodes, weights) = Self::fixed_rule(order);
        nodes
            .iter()
            .zip(weights.iter())
            .map(|(&node, &weight)| {
                let (x, scaled_weight) = Self::fixed_interval(node, weight, lo, hi);
                scaled_weight * f(x)
            })
            .sum()
    }

    pub(super) fn fixed_integral_6(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> [f64; 6] {
        let mut val = [0.0; 6];
        let mut x = [0.0; 2];

        if core.dim == 1 {
            let (nodes, weights) = Self::fixed_rule(order);
            for (&node, &weight) in nodes.iter().zip(weights.iter()) {
                let (x0, scaled_weight) =
                    Self::fixed_interval(node, weight, core.xmin[0], core.xmax[0]);
                x[0] = x0;
                let mut local = [0.0; 6];
                Self::wiener7_integrand(core, obs, &x[..1], &mut local);
                add_scaled_6(&mut val, &local, scaled_weight);
            }
        } else {
            let (nodes, weights) = Self::fixed_rule(order);
            for (&node0, &weight0) in nodes.iter().zip(weights.iter()) {
                let (x0, scaled_weight0) =
                    Self::fixed_interval(node0, weight0, core.xmin[0], core.xmax[0]);
                x[0] = x0;
                for (&node1, &weight1) in nodes.iter().zip(weights.iter()) {
                    let (x1, scaled_weight1) =
                        Self::fixed_interval(node1, weight1, core.xmin[1], core.xmax[1]);
                    x[1] = x1;
                    let mut local = [0.0; 6];
                    Self::wiener7_integrand(core, obs, &x, &mut local);
                    let scaled_weight = scaled_weight0 * scaled_weight1;
                    for i in 0..6 {
                        val[i] += scaled_weight * local[i];
                    }
                }
            }
        }

        val
    }

    pub(super) fn wiener7_integrand(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: &[f64],
        fv: &mut [f64; 6],
    ) {
        let (tau, beta) = Self::map_point(core, x);
        if beta <= 0.0 || beta >= 1.0 {
            return;
        }

        let p5 = Wiener5Params::with_params_unchecked(core.alpha, tau, beta, core.delta, core.sv);
        let fused = Wiener5.fused(obs, &p5, core.eps_series);
        if fused.log_prob.is_finite() {
            let dens = fused.log_prob.exp();
            fv[0] = dens;
            fv[1] = dens * fused.grad.alpha;
            fv[2] = dens * fused.grad.tau;
            fv[3] = dens * fused.grad.beta;
            fv[4] = dens * fused.grad.delta;
            fv[5] = dens * fused.grad.s_delta;
        }
    }

    /// Integrate only the density over the hypercube using fixed Gauss-Legendre nodes.
    #[cfg(test)]
    pub(super) fn eval_density_fixed(&self, core: &Wiener7Core, obs: &WienerObservation) -> f64 {
        self.fixed_integral_6(core, obs, Self::FIXED_QUADRATURE_ORDER)[0]
    }

    pub(super) fn eval_density_fixed_order(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> f64 {
        self.fixed_integral_6(core, obs, order)[0]
    }

    /// Integrate density + 5 inner-parameter gradient components with fixed Gauss-Legendre nodes.
    /// Then compute s_beta and s_tau derivatives separately.
    pub(super) fn eval_fused_fixed_order(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> Wiener7Eval {
        let val = self.fixed_integral_6(core, obs, order);
        self.eval_fused_from_integrals(core, obs, val, order)
    }

    /// Integrate density + 5 inner-parameter gradient components with fixed Gauss-Legendre nodes.
    /// Then compute s_beta and s_tau derivatives separately.
    #[cfg(test)]
    pub(super) fn eval_fused_fixed(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        self.eval_fused_fixed_order(core, obs, Self::FIXED_QUADRATURE_ORDER)
    }

    pub(super) fn eval_fused_from_integrals(
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
    /// Integrate only the density over the hypercube using adaptive cubature.
    #[allow(dead_code)]
    pub(super) fn eval_density_adaptive(&self, core: &Wiener7Core, obs: &WienerObservation) -> f64 {
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
    pub(super) fn eval_fused_adaptive(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
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
            grad[6] = 0.0;
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

    /// Map integration point `x` (∈ [0,1]^dim) to (tau, beta)
    #[inline]
    pub(super) fn map_point(core: &Wiener7Core, x: &[f64]) -> (f64, f64) {
        if core.dim == 1 {
            if core.sw > 0.0 {
                (core.tau0, core.beta0 + core.sw * (x[0] - 0.5))
            } else {
                (core.tau0 + core.st0 * x[0], core.beta0)
            }
        } else {
            let beta = core.beta0 + core.sw * (x[0] - 0.5);
            let tau = core.tau0 + core.st0 * x[1];
            (tau, beta)
        }
    }

    /// Helper: Wiener5 density (not log)
    pub(super) fn wiener5_density(
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> f64 {
        let lp = Wiener5.log_prob(obs, params, eps).log_prob;
        if lp.is_finite() { lp.exp() } else { 0.0 }
    }

    /// Gauss‑Legendre nodes and weights on `[0,1]` (scaled from `[-1,1]`).
    /// n: number of points.
    pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Legendre nodes/weights on [-1,1] using symmetric Newton method
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let m = n.div_ceil(2);
        for i in 1..=m {
            // Initial guess
            let mut x = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            // Iterate to find root
            for _ in 0..20 {
                let (p1, p2) = Self::legendre_poly(n as u32, x);
                dp = (n as f64) * (x * p1 - p2) / (x * x - 1.0);
                x -= p1 / dp;
                if x.abs() < 1e-15 {
                    break;
                }
            }
            // Scale to [0,1]
            let t = 0.5 * (x + 1.0);
            let w = 1.0 / ((1.0 - x * x) * dp * dp);
            nodes[i - 1] = t;
            weights[i - 1] = w;
            nodes[n - i] = 1.0 - t;
            weights[n - i] = w;
        }
        (nodes, weights)
    }

    /// Evaluate Legendre polynomial of degree n at x.
    fn legendre_poly(n: u32, x: f64) -> (f64, f64) {
        if n == 0 {
            return (1.0, 0.0);
        }
        if n == 1 {
            return (x, 1.0);
        }
        let mut p0 = 1.0;
        let mut p1 = x;
        let mut p = 0.0;
        for k in 1..n {
            let p2 = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
            p0 = p1;
            p1 = p2;
            p = p2;
        }
        // Return P_n(x) and P_{n-1}(x)
        (p, p0)
    }

    // fn eval_fused_fixed(&self, core: &Wiener7Core, obs: &WienerObservation) -> Wiener7Eval {
    //     let a = core.alpha;
    //     let a2 = a * a;
    //     let (v_eff, _w_eff_sign) = if obs.boundary == Boundary::Upper {
    //         (core.delta, 1.0)
    //     } else {
    //         (-core.delta, -1.0)
    //     };

    //     let (n_w, n_tau) = if core.sw != 0.0 && core.st0 != 0.0 {
    //         (5, 5)
    //     } else if core.sw != 0.0 {
    //         (7, 1)
    //     } else if core.st0 != 0.0 {
    //         (1, 7)
    //     } else {
    //         return Wiener7Eval {
    //             log_prob: f64::NEG_INFINITY,
    //             grad: Wiener7Grad::default(),
    //         };
    //     };

    //     fn get_gl(n: usize) -> (&'static [f64], &'static [f64]) {
    //         match n {
    //             1 => (&GL_1_NODES[..], &GL_1_WTS[..]),
    //             5 => (&GL_5_NODES[..], &GL_5_WTS[..]),
    //             7 => (&GL_7_NODES[..], &GL_7_WTS[..]),
    //             15 => (&GL_15_NODES[..], &GL_15_WTS[..]),
    //             _ => unreachable!(),
    //         }
    //     }

    //     let (w_nodes, w_wts) = get_gl(n_w);
    //     let (tau_nodes, tau_wts) = get_gl(n_tau);
    //     let n_w = w_nodes.len();
    //     let n_tau = tau_nodes.len();

    //     // 2. Series cache – correct truncation using true pre‑factor
    //     let mut series_cache = vec![(0.0_f64, 0.0_f64, 0.0_f64); n_w * n_tau]; // (log_series, dlog_dtprime, dlog_dw)

    //     let sv = core.sv;
    //     let sv2 = sv * sv;

    //     for (i, w_node) in w_nodes.iter().enumerate() {
    //         for (j, tau_node) in tau_nodes.iter().enumerate() {
    //             // Map to physical (tau, beta)
    //             let (tau, beta) = if core.sw != 0.0 && core.st0 != 0.0 {
    //                 (
    //                     core.tau0 + core.st0 * tau_node,
    //                     core.beta0 + core.sw * (w_node - 0.5),
    //                 )
    //             } else if core.sw != 0.0 {
    //                 (core.tau0, core.beta0 + core.sw * (w_node - 0.5))
    //             } else {
    //                 (core.tau0 + core.st0 * tau_node, core.beta0)
    //             };
    //             let t = obs.rt - tau;
    //             if t <= 0.0 || beta <= 0.0 || beta >= 1.0 {
    //                 series_cache[i * n_tau + j] = (f64::NEG_INFINITY, 0.0, 0.0);
    //                 continue;
    //             }
    //             let t_prime = t / a2;
    //             let w = beta;
    //             let lam = 1.0 + sv2 * t;
    //             let one_m_w = 1.0 - w;
    //             let v = v_eff;
    //             // Pre‑factor N and pref5 (Wiener5 pre‑factor)
    //             let n = -v * v * t + 2.0 * a * v * one_m_w + a2 * one_m_w * one_m_w * sv2;
    //             let pref5 = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);

    //             let core4 = Wiener4Core {
    //                 t,
    //                 a,
    //                 t_prime,
    //                 w_eff: w,
    //                 v_eff, // not used
    //                 pref: pref5,
    //                 log_eps_eff: (core.eps_series.ln() - pref5).min(-10.0), // eps_series = 1e-12, w. safety cap
    //                 beta_sign: 1.0,
    //                 delta_sign: 1.0,
    //             };
    //             if let Some(se) = Wiener4::eval_fused(&core4) {
    //                 series_cache[i * n_tau + j] = (se.log_series, se.dlog_dtprime, se.dlog_dw);
    //             } else {
    //                 series_cache[i * n_tau + j] = (f64::NEG_INFINITY, 0.0, 0.0);
    //             }
    //         }
    //     }

    //     // 3. Main integration – accumulate density and inner‑parameter gradients
    //     let mut total_density = 0.0;
    //     let mut grad_density = [0.0_f64; 5]; // [dalpha, dtau, dbeta, ddelta, dsv]

    //     for (i, w_node) in w_nodes.iter().enumerate() {
    //         let wgt_w = w_wts[i];
    //         for (j, tau_node) in tau_nodes.iter().enumerate() {
    //             let wgt_tau = tau_wts[j];
    //             let weight = wgt_w * wgt_tau;

    //             let (log_series, dlog_dtprime, dlog_dw) = series_cache[i * n_tau + j];
    //             if !log_series.is_finite() {
    //                 continue;
    //             }

    //             // Compute tau, beta again (could reuse precomputed, but fine)
    //             let (tau, beta) = if core.sw != 0.0 && core.st0 != 0.0 {
    //                 (
    //                     core.tau0 + core.st0 * tau_node,
    //                     core.beta0 + core.sw * (w_node - 0.5),
    //                 )
    //             } else if core.sw != 0.0 {
    //                 (core.tau0, core.beta0 + core.sw * (w_node - 0.5))
    //             } else {
    //                 (core.tau0 + core.st0 * tau_node, core.beta0)
    //             };
    //             let t = obs.rt - tau;
    //             let w = beta;

    //             let lam = 1.0 + sv2 * t;
    //             let lam2 = lam * lam;
    //             let one_m_w = 1.0 - w;
    //             let v = v_eff;

    //             // N and its derivatives (same as Wiener5::fused)
    //             let n = -v * v * t + 2.0 * a * v * one_m_w + a2 * one_m_w * one_m_w * sv2;
    //             let d_n_da = 2.0 * one_m_w * (v + a * one_m_w * sv2);
    //             let d_n_dv = -2.0 * v * t + 2.0 * a * one_m_w;
    //             let d_n_dw = -2.0 * a * v - 2.0 * a2 * one_m_w * sv2;
    //             let d_n_dt = -v * v;
    //             let d_n_dsv = 2.0 * a2 * one_m_w * one_m_w * sv;

    //             let dpref_da = -2.0 / a + d_n_da / (2.0 * lam);
    //             let dpref_dt = -0.5 * sv2 / lam + (d_n_dt * lam - n * sv2) / (2.0 * lam2);
    //             let dpref_dw = d_n_dw / (2.0 * lam);
    //             let dpref_dv = d_n_dv / (2.0 * lam);
    //             let dpref_dsv = -sv * t / lam + (d_n_dsv * lam - n * 2.0 * sv * t) / (2.0 * lam2);

    //             let pref5 = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);
    //             let log_prob = pref5 + log_series;

    //             if !log_prob.is_finite() {
    //                 continue;
    //             }

    //             let dens = log_prob.exp();

    //             // Gradient components (d log density / d param)
    //             let dalpha = dpref_da + dlog_dtprime * (-2.0 * t / (a * a2));
    //             let dtau = -(dpref_dt + dlog_dtprime / a2);
    //             let dbeta = if obs.boundary == Boundary::Upper {
    //                 1.0
    //             } else {
    //                 -1.0
    //             } * (dpref_dw + dlog_dw);
    //             let ddelta = if obs.boundary == Boundary::Upper {
    //                 1.0
    //             } else {
    //                 -1.0
    //             } * dpref_dv;
    //             let dsv = dpref_dsv;

    //             total_density += weight * dens;
    //             grad_density[0] += weight * dens * dalpha;
    //             grad_density[1] += weight * dens * dtau;
    //             grad_density[2] += weight * dens * dbeta;
    //             grad_density[3] += weight * dens * ddelta;
    //             grad_density[4] += weight * dens * dsv;
    //         }
    //     }

    //     if unlikely(total_density <= 0.0) {
    //         return Wiener7Eval {
    //             log_prob: f64::NEG_INFINITY,
    //             grad: Wiener7Grad::default(),
    //         };
    //     }

    //     let log_total = total_density.ln();
    //     let mut grad = [0.0_f64; 7];
    //     grad[0] = grad_density[0] / total_density;
    //     grad[1] = grad_density[1] / total_density;
    //     grad[2] = grad_density[2] / total_density;
    //     grad[3] = grad_density[3] / total_density;
    //     grad[4] = grad_density[4] / total_density;

    //     // 4. s_beta gradient – follows reference formulas, using fixed quadrature
    //     if core.sw == 0.0 {
    //         grad[5] = 0.0;
    //     } else if core.st0 == 0.0 {
    //         // No tau variability → endpoint formula
    //         let low = core.beta0 - core.sw / 2.0;
    //         let high = core.beta0 + core.sw / 2.0;
    //         let t_prime = (obs.rt - core.tau0) / a2;
    //         // Evaluate density at endpoints using correct truncation
    //         let dens_at_w = |w: f64| -> f64 {
    //             let t = obs.rt - core.tau0;
    //             let lam = 1.0 + sv2 * t;
    //             let one_m_w = 1.0 - w;
    //             let n =
    //                 -v_eff * v_eff * t + 2.0 * a * v_eff * one_m_w + a2 * one_m_w * one_m_w * sv2;
    //             let pref5 = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);
    //             let core4 = Wiener4Core {
    //                 t,
    //                 a,
    //                 t_prime,
    //                 w_eff: w,
    //                 v_eff,
    //                 pref: pref5,
    //                 log_eps_eff: (core.eps_series).ln() - pref5,
    //                 beta_sign: 1.0,
    //                 delta_sign: 1.0,
    //             };
    //             if let Some(se) = Wiener4::eval_series(&core4) {
    //                 let log_prob = pref5 + se;
    //                 if log_prob.is_finite() {
    //                     log_prob.exp()
    //                 } else {
    //                     0.0
    //                 }
    //             } else {
    //                 0.0
    //             }
    //         };
    //         let d_low = dens_at_w(low.clamp(0.0, 1.0));
    //         let d_high = dens_at_w(high.clamp(0.0, 1.0));
    //         grad[5] = 0.5 * (d_low + d_high) / (core.sw * total_density) - 1.0 / core.sw;
    //     } else {
    //         // Integrate over tau using 1-D Gauss‑Legendre (no Jacobian factor in integrand, the weight already accounts for interval)
    //         let (tau_sb_nodes, tau_sb_wts): (&[f64], &[f64]) = get_gl(5);
    //         let low = core.beta0 - core.sw / 2.0;
    //         let high = core.beta0 + core.sw / 2.0;
    //         let mut integral = 0.0;
    //         for (tau_n, w_tau) in tau_sb_nodes.iter().zip(tau_sb_wts.iter()) {
    //             let tau = core.tau0 + core.st0 * tau_n;
    //             let t = obs.rt - tau;
    //             if t <= 0.0 {
    //                 continue;
    //             }
    //             let t_prime = t / a2;
    //             // density at low and high for this tau
    //             let dens_at = |w: f64| -> f64 {
    //                 let lam = 1.0 + sv2 * t;
    //                 let one_m_w = 1.0 - w;
    //                 let n = -v_eff * v_eff * t
    //                     + 2.0 * a * v_eff * one_m_w
    //                     + a2 * one_m_w * one_m_w * sv2;
    //                 let pref5 = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);
    //                 let core4 = Wiener4Core {
    //                     t,
    //                     a,
    //                     t_prime,
    //                     w_eff: w,
    //                     v_eff,
    //                     pref: pref5,
    //                     log_eps_eff: (core.eps_series).ln() - pref5,
    //                     beta_sign: 1.0,
    //                     delta_sign: 1.0,
    //                 };
    //                 if let Some(se) = Wiener4::eval_series(&core4) {
    //                     let log_prob = pref5 + se;
    //                     if log_prob.is_finite() {
    //                         log_prob.exp()
    //                     } else {
    //                         0.0
    //                     }
    //                 } else {
    //                     0.0
    //                 }
    //             };
    //             let d_low = dens_at(low.clamp(0.0, 1.0));
    //             let d_high = dens_at(high.clamp(0.0, 1.0));
    //             integral += w_tau * 0.5 * (d_low + d_high) / core.sw;
    //         }
    //         grad[5] = integral / total_density - 1.0 / core.sw;
    //     }

    //     // 5. s_tau gradient – follows reference formulas
    //     if core.st0 == 0.0 {
    //         grad[6] = 0.0;
    //     } else {
    //         let t0plus = core.tau0 + core.st0;
    //         if obs.rt - t0plus <= 0.0 {
    //             grad[6] = -1.0 / core.st0;
    //         } else {
    //             let t_prime = (obs.rt - t0plus) / a2;
    //             let f_end: f64;
    //             if core.sw == 0.0 {
    //                 // No beta variability
    //                 let t = obs.rt - t0plus;
    //                 let lam = 1.0 + sv2 * t;
    //                 let one_m_w = 1.0 - core.beta0;
    //                 let n = -v_eff * v_eff * t
    //                     + 2.0 * a * v_eff * one_m_w
    //                     + a2 * one_m_w * one_m_w * sv2;
    //                 let pref5 = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);
    //                 let core4 = Wiener4Core {
    //                     t,
    //                     a,
    //                     t_prime,
    //                     w_eff: core.beta0,
    //                     v_eff,
    //                     pref: pref5,
    //                     log_eps_eff: (core.eps_series).ln() - pref5,
    //                     beta_sign: 1.0,
    //                     delta_sign: 1.0,
    //                 };
    //                 if let Some(se) = Wiener4::eval_series(&core4) {
    //                     let log_prob = pref5 + se;
    //                     f_end = if log_prob.is_finite() {
    //                         log_prob.exp()
    //                     } else {
    //                         0.0
    //                     };
    //                 } else {
    //                     f_end = 0.0;
    //                 }
    //             } else {
    //                 // Integrate over w using 1-D Gauss‑Legendre
    //                 let (w_st_nodes, w_st_wts): (&[f64], &[f64]) = get_gl(5);
    //                 let t = obs.rt - t0plus;
    //                 let t_prime = t / a2;
    //                 let mut integral = 0.0;
    //                 for (wn, ww) in w_st_nodes.iter().zip(w_st_wts.iter()) {
    //                     let w = core.beta0 + core.sw * (wn - 0.5);
    //                     if w <= 0.0 || w >= 1.0 {
    //                         continue;
    //                     }
    //                     let lam = 1.0 + sv2 * t;
    //                     let one_m_w = 1.0 - w;
    //                     let n = -v_eff * v_eff * t
    //                         + 2.0 * a * v_eff * one_m_w
    //                         + a2 * one_m_w * one_m_w * sv2;
    //                     let pref5 = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);
    //                     let core4 = Wiener4Core {
    //                         t,
    //                         a,
    //                         t_prime,
    //                         w_eff: w,
    //                         v_eff,
    //                         pref: pref5,
    //                         log_eps_eff: (core.eps_series).ln() - pref5,
    //                         beta_sign: 1.0,
    //                         delta_sign: 1.0,
    //                     };
    //                     if let Some(se) = Wiener4::eval_series(&core4) {
    //                         let log_prob = pref5 + se;
    //                         if log_prob.is_finite() {
    //                             integral += ww * log_prob.exp();
    //                         }
    //                     }
    //                 }
    //                 f_end = integral;
    //             }
    //             grad[6] = -1.0 / core.st0 + f_end / (core.st0 * total_density);
    //         }
    //     }

    //     Wiener7Eval {
    //         log_prob: log_total,
    //         grad: Wiener7Grad::from_array(grad),
    //     }
    // }
}

impl Family for Wiener7 {
    type Params = Wiener7Params;
    type Data = Vec<WienerObservation>;

    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        let precision = 1e-4; // match Stan's default precision for derivatives
        data.iter()
            .map(|obs| Wiener7.log_prob(obs, params, precision).log_prob)
            .sum()
    }
}

impl GradLogDensity for Target<Wiener7, Vec<WienerObservation>> {
    type Gradient = [f64; 7];

    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

impl FusedLogDensity for Target<Wiener7, Vec<WienerObservation>> {
    fn log_prob_and_grad<'a>(&'a self, p: &'a Wiener7Params, grad: &mut [f64; 7]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);
        for obs in &self.data {
            let fused = Wiener7.fused(obs, p, 1e-4);
            let (lp, g) = (fused.log_prob, fused.grad);
            total_lp += lp;
            grad[0] += g.alpha;
            grad[1] += g.tau;
            grad[2] += g.beta;
            grad[3] += g.delta;
            grad[4] += g.s_delta;
            grad[5] += g.s_beta;
            grad[6] += g.s_tau;
        }
        total_lp
    }
}

impl LogDensity for Target<Wiener7, WienerObservations> {
    type Point = Wiener7Params;

    fn log_prob(&self, p: &Self::Point) -> f64 {
        let eps = 1e-4;
        self.data
            .upper_rt()
            .iter()
            .map(|&rt| {
                Wiener7
                    .log_prob(
                        &WienerObservation {
                            rt,
                            boundary: Boundary::Upper,
                        },
                        p,
                        eps,
                    )
                    .log_prob
            })
            .chain(self.data.lower_rt().iter().map(|&rt| {
                Wiener7
                    .log_prob(
                        &WienerObservation {
                            rt,
                            boundary: Boundary::Lower,
                        },
                        p,
                        eps,
                    )
                    .log_prob
            }))
            .sum()
    }
}

impl FusedLogDensity for Target<Wiener7, WienerObservations> {
    fn log_prob_and_grad(&self, p: &Wiener7Params, grad: &mut [f64; 7]) -> f64 {
        let eps = 1e-4;
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for &rt in self.data.upper_rt() {
            let fused = Wiener7.fused(
                &WienerObservation {
                    rt,
                    boundary: Boundary::Upper,
                },
                p,
                eps,
            );
            total_lp += fused.log_prob;
            grad[0] += fused.grad.alpha;
            grad[1] += fused.grad.tau;
            grad[2] += fused.grad.beta;
            grad[3] += fused.grad.delta;
            grad[4] += fused.grad.s_delta;
            grad[5] += fused.grad.s_beta;
            grad[6] += fused.grad.s_tau;
        }
        for &rt in self.data.lower_rt() {
            let fused = Wiener7.fused(
                &WienerObservation {
                    rt,
                    boundary: Boundary::Lower,
                },
                p,
                eps,
            );
            total_lp += fused.log_prob;
            grad[0] += fused.grad.alpha;
            grad[1] += fused.grad.tau;
            grad[2] += fused.grad.beta;
            grad[3] += fused.grad.delta;
            grad[4] += fused.grad.s_delta;
            grad[5] += fused.grad.s_beta;
            grad[6] += fused.grad.s_tau;
        }

        total_lp
    }
}

impl GradLogDensity for Target<Wiener7, WienerObservations> {
    type Gradient = [f64; 7];

    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}
