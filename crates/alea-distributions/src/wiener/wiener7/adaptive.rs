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
    fn eval_fused_adaptive_raw(&self, core: &Wiener7Core, obs: &WienerObservation) -> Wiener7Eval {
        let bounds = Bounds::new(&core.xmin, &core.xmax);
        let mut val = [0.; 8];
        let mut err = [0.; 8];
        let integrand = |x: &[f64], fv: &mut [f64]| Self::wiener7_integrand(core, obs, x, fv);
        if hcubature_into(8, bounds, core.opts, &mut val, &mut err, integrand).is_err() {
            return fail_eval_7();
        }
        Self::from_integrals(val)
    }
}
