use super::*;

mod adaptive;
#[cfg(test)]
mod audit;
mod endpoints;
mod fixed;
mod scaled;

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
        if core.dim == 0 {
            let e = Wiener5.log_prob(obs, &params.base, eps);
            return Wiener7Eval {
                log_prob: e.log_prob,
                grad: NAN_GRAD_7,
            };
        }
        let density = self.eval_density_adaptive(&core, obs);
        let log_prob = if density.is_normal() && density > 0.0 {
            density.ln()
        } else {
            self.scaled_adaptive(&core, obs).log_prob
        };
        Wiener7Eval {
            log_prob,
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

        if core.dim == 0 {
            return Wiener5
                .try_log_prob(obs, &params.base, options)
                .map(|e| Wiener7Eval {
                    log_prob: e.log_prob,
                    grad: NAN_GRAD_7,
                });
        }
        let density = match options.quadrature {
            Quadrature::Adaptive => self.eval_density_adaptive(&core, obs),
            Quadrature::FixedGaussLegendre { order } => {
                self.eval_density_fixed_order(&core, obs, order)
            }
        };
        let log_prob = if density.is_normal() && density > 0.0 {
            density.ln()
        } else {
            match options.quadrature {
                Quadrature::Adaptive => self.scaled_adaptive(&core, obs).log_prob,
                Quadrature::FixedGaussLegendre { order } => {
                    self.scaled_fixed(&core, obs, order).log_prob
                }
            }
        };
        if log_prob.is_finite() {
            Ok(Wiener7Eval {
                log_prob,
                grad: NAN_GRAD_7,
            })
        } else {
            Err(ProbError::NumericalError(
                "Wiener7 scaled integration did not converge to a finite density".to_string(),
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
                    s_tau: 0.5 * e.grad.tau,
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

        if likely(eval.log_prob.is_finite() && eval.grad_array().iter().all(|g| g.is_finite())) {
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
                    s_tau: 0.5 * e.grad.tau,
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
}

impl WienerModel<Wiener7, Vec<WienerObservation>> {
    pub fn batch_value_gradient<'a>(&'a self, p: &'a Wiener7Params, grad: &mut [f64; 7]) -> f64 {
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

impl WienerModel<Wiener7, WienerObservations> {
    pub fn batch_value_gradient(&self, p: &Wiener7Params, grad: &mut [f64; 7]) -> f64 {
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
