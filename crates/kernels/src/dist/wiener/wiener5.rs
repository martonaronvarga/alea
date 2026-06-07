use super::*;
use crate::buffer::OwnedBuffer;
use crate::density::LogDensity;

#[cold]
#[inline(never)]
fn fail_eval_5() -> Wiener5Eval {
    FAIL_EVAL_5
}

#[inline]
fn add_wiener5_eval(total_lp: &mut f64, grad: &mut [f64; 5], eval: Wiener5Eval) {
    *total_lp += eval.log_prob;
    grad[0] += eval.grad.alpha;
    grad[1] += eval.grad.tau;
    grad[2] += eval.grad.beta;
    grad[3] += eval.grad.delta;
    grad[4] += eval.grad.s_delta;
}

#[inline]
fn add_wiener5_direct_parts(
    total_lp: &mut f64,
    grad: &mut [f64; 5],
    rt: f64,
    boundary: Boundary,
    params: &Wiener5Params,
    eps: f64,
) {
    add_wiener5_eval(total_lp, grad, Wiener5.fused_from_parts(rt, boundary, params, eps));
}

const W5_SMALL_K2: u8 = 0;
const W5_SMALL_K3: u8 = 1;
const W5_SMALL_OTHER: u8 = 2;
const W5_LARGE_K4: u8 = 3;
const W5_LARGE_K5: u8 = 4;
const W5_LARGE_K6: u8 = 5;
const W5_LARGE_OTHER: u8 = 6;

#[derive(Clone, Copy, Debug)]
struct BucketedWiener5Core {
    core: Wiener5Core,
    bucket: u8,
    k: usize,
}

#[inline]
fn push_wiener5_bucket(
    scratch: &mut OwnedBuffer<BucketedWiener5Core>,
    len: &mut usize,
    core: Wiener5Core,
) {
    let base = core.base;
    let ks_density = Wiener4::k_s(base.t_prime, base.w_eff, base.log_eps_eff);
    let kl_density = Wiener4::k_l(base.t_prime, base.log_eps_eff);
    let (bucket, k) = if unlikely(2 * ks_density <= kl_density) {
        match ks_density {
            2 => (W5_SMALL_K2, 2),
            3 => (W5_SMALL_K3, 3),
            _ => (W5_SMALL_OTHER, ks_density),
        }
    } else {
        let k = kl_density.max(Wiener4::k_l_grad_w(base.t_prime, base.log_eps_eff));
        match k {
            4 => (W5_LARGE_K4, 4),
            5 => (W5_LARGE_K5, 5),
            6 => (W5_LARGE_K6, 6),
            _ => (W5_LARGE_OTHER, k),
        }
    };

    scratch.as_mut_slice()[*len] = BucketedWiener5Core { core, bucket, k };
    *len += 1;
}

#[inline]
fn add_wiener5_scratch_bucket(
    total_lp: &mut f64,
    grad: &mut [f64; 5],
    scratch: &[BucketedWiener5Core],
    bucket: u8,
    branch: SeriesBranch,
    fixed_k: Option<usize>,
) {
    for entry in scratch.iter().filter(|entry| entry.bucket == bucket) {
        let k = fixed_k.unwrap_or(entry.k);
        add_wiener5_eval(
            total_lp,
            grad,
            Wiener5.fused_from_core_with_branch(&entry.core, branch, k),
        );
    }
}

impl Wiener5 {
    #[inline]
    pub(super) fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Result<Wiener5Core> {
        self.core_from_parts(obs.rt, obs.boundary, params, eps)
    }

    #[inline]
    fn core_from_parts(
        &self,
        rt: f64,
        boundary: Boundary,
        params: &Wiener5Params,
        eps: f64,
    ) -> Result<Wiener5Core> {
        let Wiener5Params { base, s_delta } = params;

        if unlikely(
            !base.valid()
                || !s_delta.is_finite()
                || *s_delta < 0.0
                || !rt.is_finite()
                || !eps.is_finite()
                || eps <= 0.0,
        ) {
            return Err(ProbError::InvalidParameters(
                "Wiener5 requires valid base parameters, finite non-negative drift variability, finite reaction time, and finite positive precision"
                    .to_string(),
            ));
        }

        let t = rt - base.tau;
        if unlikely(t <= 0.0) {
            return Err(ProbError::OutOfSupport(
                "reaction time must exceed non-decision time".to_string(),
            ));
        }

        let a = base.alpha;
        let a2 = a * a;
        let t_prime = t / a2;

        let (v_eff, w_eff, beta_sign, delta_sign) = match boundary {
            Boundary::Upper => (base.delta, base.beta, 1.0, 1.0),
            Boundary::Lower => (-base.delta, 1.0 - base.beta, -1.0, -1.0),
        };

        let sv = *s_delta;
        let sv2 = sv * sv;
        let lam = sv2.mul_add(t, 1.0); // 1 + sv²·t   (FMA)

        let one_m_w = 1.0 - w_eff;
        let a_one_m_w = a * one_m_w; // a·(1-w)
        let a_one_m_w_sq = a_one_m_w * a_one_m_w; // a²·(1-w)²

        // numerator inside exp: -v²t + 2av(1-w) + a²(1-w)² sv²
        let num = (-v_eff * v_eff * t)
            .mul_add(1.0, 0.0) // no‑op, keep symmetry
            + 2.0 * v_eff * a_one_m_w
            + a_one_m_w_sq * sv2;

        let lam_inv = lam.recip();
        // -2 ln a  -½ ln lam  +  num/(2 lam)
        let pref = (-2.0_f64).mul_add(a.ln(), -0.5 * lam.ln()) + 0.5 * lam_inv * num;

        let log_eps_eff = (eps.ln() - pref).min(-10.0);

        Ok(Wiener5Core {
            base: Wiener4Core {
                t,
                a,
                t_prime,
                w_eff,
                v_eff,
                pref,
                log_eps_eff,
                beta_sign,
                delta_sign,
            },
            sv,
            sv2,
            lam,
        })
    }

    #[must_use]
    pub fn branch_counts(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Option<WienerBranchCounts> {
        let core = self.core(obs, params, eps).ok()?;
        let base = core.base;
        let k_small_density = Wiener4::k_s(base.t_prime, base.w_eff, base.log_eps_eff);
        let k_large_density = Wiener4::k_l(base.t_prime, base.log_eps_eff);
        let k_small_grad_w = k_small_density;
        let k_large_grad_w = Wiener4::k_l_grad_w(base.t_prime, base.log_eps_eff);
        let branch = if 2 * k_small_density <= k_large_density {
            SeriesBranch::SmallTime
        } else {
            SeriesBranch::LargeTime
        };
        Some(WienerBranchCounts {
            branch,
            k_small_density,
            k_large_density,
            k_small_grad_w,
            k_large_grad_w,
            k_small_used: k_small_density,
            k_large_used: k_large_density.max(k_large_grad_w),
        })
    }

    /// Log‑density only (discards gradient).
    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Wiener5Eval {
        let mut eval = self.fused(obs, params, eps);
        eval.grad = NAN_GRAD_5;
        eval
    }

    #[inline]
    pub fn try_log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        options: WienerOptions,
    ) -> Result<Wiener5Eval> {
        let mut eval = self.try_fused(obs, params, options)?;
        eval.grad = Wiener5Grad::default();
        Ok(eval)
    }

    #[inline]
    pub fn try_fused(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        options: WienerOptions,
    ) -> Result<Wiener5Eval> {
        let options = options.validate()?;
        let eps = options.series_precision();
        let core = self.core(obs, params, eps)?;
        let eval = self.fused_from_core(&core);
        if likely(eval.log_prob.is_finite()) {
            Ok(eval)
        } else {
            Err(ProbError::NumericalError(
                "Wiener5 evaluation produced a non-finite log density".to_string(),
            ))
        }
    }

    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> Wiener5Eval {
        self.fused_from_parts(obs.rt, obs.boundary, params, eps)
    }

    #[inline]
    fn fused_from_parts(
        &self,
        rt: f64,
        boundary: Boundary,
        params: &Wiener5Params,
        eps: f64,
    ) -> Wiener5Eval {
        if params.s_delta == 0.0 {
            let eval = Wiener4.fused_from_parts(rt, boundary, &params.base, eps);
            return Wiener5Eval {
                log_prob: eval.log_prob,
                grad: Wiener5Grad {
                    alpha: eval.grad.alpha,
                    tau: eval.grad.tau,
                    beta: eval.grad.beta,
                    delta: eval.grad.delta,
                    s_delta: 0.0,
                },
            };
        }

        let core = match self.core_from_parts(rt, boundary, params, eps) {
            Ok(core) => core,
            Err(_) => return fail_eval_5(),
        };
        self.fused_from_core(&core)
    }

    #[inline]
    fn fused_from_core(&self, core: &Wiener5Core) -> Wiener5Eval {
        let base = core.base;
        let ks_density = Wiener4::k_s(base.t_prime, base.w_eff, base.log_eps_eff);
        let kl_density = Wiener4::k_l(base.t_prime, base.log_eps_eff);
        let k = if unlikely(2 * ks_density <= kl_density) {
            (SeriesBranch::SmallTime, ks_density)
        } else {
            (
                SeriesBranch::LargeTime,
                kl_density.max(Wiener4::k_l_grad_w(base.t_prime, base.log_eps_eff)),
            )
        };
        self.fused_from_core_with_branch(core, k.0, k.1)
    }

    #[inline]
    fn fused_from_core_with_branch(
        &self,
        core: &Wiener5Core,
        branch: SeriesBranch,
        k: usize,
    ) -> Wiener5Eval {
        let base = core.base;
        let t = base.t;
        let a = base.a;
        let a2 = a * a;
        let t_prime = base.t_prime;
        let w_eff = base.w_eff;
        let v_eff = base.v_eff;
        let beta_sign = base.beta_sign;
        let delta_sign = base.delta_sign;
        let sv = core.sv;
        let sv2 = core.sv2;
        let lam = core.lam;
        let lam_inv = lam.recip();
        let lam_sq_inv = lam_inv * lam_inv; // 1/lam^2

        let one_m_w = 1.0 - w_eff;
        let a_one_m_w = a * one_m_w;
        let a_one_m_w_sq = a_one_m_w * a_one_m_w;

        // N = –v^2 t + 2 a v (1–w) + a^2 (1–w)^2 s^2
        let num = (-v_eff * v_eff * t) + 2.0 * v_eff * a_one_m_w + a_one_m_w_sq * sv2;
        let half_lam_inv = 0.5 * lam_inv;
        let pref = (-2.0_f64).mul_add(a.ln(), -0.5 * lam.ln()) + half_lam_inv * num;

        let dlam_dsv = 2.0 * sv * t; // dlam / dsv
        let dlam_dt = sv2;

        // dN/d*
        let dn_da = 2.0 * v_eff * one_m_w + 2.0 * a * one_m_w * one_m_w * sv2; // also = 2 v (1-w) + 2 a (1-w)^2 sv2
        let dn_dv = -2.0 * v_eff * t + 2.0 * a_one_m_w;
        let dn_dw = -2.0 * a * v_eff - 2.0 * a2 * one_m_w * sv2;
        let dn_dsv = 2.0 * a2 * one_m_w * one_m_w * sv;
        let dn_dt = -v_eff * v_eff;

        // dQ = (lam * dN – N * dlam) / (2 lam^2)   for each parameter
        let dq_da = 0.5 * lam_sq_inv * (lam * dn_da); // dlam/da=0
        let dq_dv = 0.5 * lam_sq_inv * (lam * dn_dv); // dlam/dv=0
        let dq_dw = 0.5 * lam_sq_inv * (lam * dn_dw); // dlam/dw=0
        let dq_dsv = 0.5 * lam_sq_inv * (lam * dn_dsv - num * dlam_dsv);
        let dq_dt = 0.5 * lam_sq_inv * (lam * dn_dt - num * dlam_dt);

        // dP = d(-2 ln a – 0.5 ln lam) + dQ
        let dp_da = -2.0 / a + dq_da; // only -2/a
        let dp_dt = -0.5 * lam_inv * dlam_dt + dq_dt;
        let dp_dw = dq_dw;
        let dp_dv = dq_dv;
        let dp_dsv = -0.5 * lam_inv * dlam_dsv + dq_dsv;

        let eval = match branch {
            SeriesBranch::SmallTime => Wiener4::small_branch_fused(t_prime, w_eff, k),
            SeriesBranch::LargeTime => Wiener4::large_branch_fused(t_prime, w_eff, k),
        };

        let (log_series, dlog_dtprime, dlog_dw) = match eval {
            Some((a, b, c)) => (a, b, c),
            None => return fail_eval_5(),
        };

        let log_prob = pref + log_series;

        // dt_prime/da = -2 t / a^3
        // dt_prime/dt = 1 / a^2
        let dlog_da = dp_da + dlog_dtprime * (-2.0 * t / (a2 * a));
        let dlog_dt = dp_dt + dlog_dtprime / a2; // = dlog / d(rt-tau)
        let dlog_dw_eff = dp_dw + dlog_dw;

        let grad = Wiener5Grad {
            alpha: dlog_da,
            tau: -dlog_dt, // dt/dtau = -1
            beta: beta_sign * dlog_dw_eff,
            delta: delta_sign * dp_dv, // series independent of v
            s_delta: dp_dsv,
        };

        Wiener5Eval { log_prob, grad }
    }
}

impl GradLogDensity for Target<Wiener5, Vec<WienerObservation>> {
    type Gradient = [f64; 5]; // Note: type must match FusedLogDensity array size
    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

impl FusedLogDensity for Target<Wiener5, Vec<WienerObservation>> {
    fn log_prob_and_grad(&self, p: &Wiener5Params, grad: &mut [f64; 5]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for obs in &self.data {
            let fused = Wiener5.fused(obs, p, 1e-6);
            let (lp, g) = (fused.log_prob, fused.grad);
            total_lp += lp;
            grad[0] += g.alpha;
            grad[1] += g.tau;
            grad[2] += g.beta;
            grad[3] += g.delta;
            grad[4] += g.s_delta;
        }

        total_lp
    }
}

impl LogDensity for Target<Wiener5, WienerObservations> {
    type Point = Wiener5Params;

    fn log_prob(&self, p: &Self::Point) -> f64 {
        let eps = 1e-12;
        self.data
            .upper_rt()
            .iter()
            .map(|&rt| {
                Wiener5.fused_from_parts(rt, Boundary::Upper, p, eps).log_prob
            })
            .chain(self.data.lower_rt().iter().map(|&rt| {
                Wiener5.fused_from_parts(rt, Boundary::Lower, p, eps).log_prob
            }))
            .sum()
    }
}

impl FusedLogDensity for Target<Wiener5, WienerObservations> {
    fn log_prob_and_grad(&self, p: &Wiener5Params, grad: &mut [f64; 5]) -> f64 {
        let eps = 1e-12;
        if p.s_delta == 0.0 {
            let mut total_lp = 0.0;
            grad.fill(0.0);
            for &rt in self.data.upper_rt() {
                let fused = Wiener4.fused_from_parts(rt, Boundary::Upper, &p.base, eps);
                total_lp += fused.log_prob;
                grad[0] += fused.grad.alpha;
                grad[1] += fused.grad.tau;
                grad[2] += fused.grad.beta;
                grad[3] += fused.grad.delta;
            }
            for &rt in self.data.lower_rt() {
                let fused = Wiener4.fused_from_parts(rt, Boundary::Lower, &p.base, eps);
                total_lp += fused.log_prob;
                grad[0] += fused.grad.alpha;
                grad[1] += fused.grad.tau;
                grad[2] += fused.grad.beta;
                grad[3] += fused.grad.delta;
            }
            grad[4] = 0.0;
            return total_lp;
        }

        let mut total_lp = 0.0;
        grad.fill(0.0);

        if self.data.batch_strategy() == BatchStrategy::Direct {
            for &rt in self.data.upper_rt() {
                add_wiener5_direct_parts(&mut total_lp, grad, rt, Boundary::Upper, p, eps);
            }
            for &rt in self.data.lower_rt() {
                add_wiener5_direct_parts(&mut total_lp, grad, rt, Boundary::Lower, p, eps);
            }
            return total_lp;
        }

        let mut failed = false;
        let mut scratch = OwnedBuffer::<BucketedWiener5Core>::new(self.data.len());
        let mut scratch_len = 0;

        for (&rt, boundary) in self
            .data
            .upper_rt()
            .iter()
            .map(|rt| (rt, Boundary::Upper))
            .chain(self.data.lower_rt().iter().map(|rt| (rt, Boundary::Lower)))
        {
            match Wiener5.core_from_parts(rt, boundary, p, eps) {
                Ok(core) => push_wiener5_bucket(&mut scratch, &mut scratch_len, core),
                Err(_) => failed = true,
            }
        }
        scratch.truncate(scratch_len);
        let scratch = scratch.as_slice();

        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_SMALL_K2, SeriesBranch::SmallTime, Some(2));
        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_SMALL_K3, SeriesBranch::SmallTime, Some(3));
        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_SMALL_OTHER, SeriesBranch::SmallTime, None);
        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_LARGE_K4, SeriesBranch::LargeTime, Some(4));
        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_LARGE_K5, SeriesBranch::LargeTime, Some(5));
        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_LARGE_K6, SeriesBranch::LargeTime, Some(6));
        add_wiener5_scratch_bucket(&mut total_lp, grad, scratch, W5_LARGE_OTHER, SeriesBranch::LargeTime, None);

        if unlikely(failed) {
            grad.fill(f64::NAN);
            f64::NEG_INFINITY
        } else {
            total_lp
        }
    }
}

impl GradLogDensity for Target<Wiener5, WienerObservations> {
    type Gradient = [f64; 5];

    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

