use super::*;
use crate::buffer::OwnedBuffer;
use crate::density::LogDensity;

#[cold]
#[inline(never)]
fn fail_eval_4() -> Wiener4Eval {
    Wiener4Eval {
        log_prob: f64::NEG_INFINITY,
        grad: NAN_GRAD_4,
    }
}

#[inline]
fn add_wiener4_eval(total_lp: &mut f64, grad: &mut [f64; 4], eval: Wiener4Eval) {
    *total_lp += eval.log_prob;
    grad[0] += eval.grad.alpha;
    grad[1] += eval.grad.tau;
    grad[2] += eval.grad.beta;
    grad[3] += eval.grad.delta;
}

#[inline]
fn add_wiener4_direct_parts(
    total_lp: &mut f64,
    grad: &mut [f64; 4],
    rt: f64,
    boundary: Boundary,
    params: &Wiener4Params,
    eps: f64,
) {
    add_wiener4_eval(total_lp, grad, Wiener4.fused_from_parts(rt, boundary, params, eps));
}

const W4_SMALL_K2: u8 = 0;
const W4_SMALL_K3: u8 = 1;
const W4_SMALL_OTHER: u8 = 2;
const W4_LARGE_K4: u8 = 3;
const W4_LARGE_K5: u8 = 4;
const W4_LARGE_K6: u8 = 5;
const W4_LARGE_OTHER: u8 = 6;

#[derive(Clone, Copy, Debug)]
struct BucketedWiener4Core {
    core: Wiener4Core,
    bucket: u8,
    k: usize,
}

#[inline]
fn push_wiener4_bucket(
    scratch: &mut OwnedBuffer<BucketedWiener4Core>,
    len: &mut usize,
    core: Wiener4Core,
) {
    let ks_density = Wiener4::k_s(core.t_prime, core.w_eff, core.log_eps_eff);
    let kl_density = Wiener4::k_l(core.t_prime, core.log_eps_eff);
    let (bucket, k) = if unlikely(2 * ks_density <= kl_density) {
        match ks_density {
            2 => (W4_SMALL_K2, 2),
            3 => (W4_SMALL_K3, 3),
            _ => (W4_SMALL_OTHER, ks_density),
        }
    } else {
        let k = kl_density.max(Wiener4::k_l_grad_w(core.t_prime, core.log_eps_eff));
        match k {
            4 => (W4_LARGE_K4, 4),
            5 => (W4_LARGE_K5, 5),
            6 => (W4_LARGE_K6, 6),
            _ => (W4_LARGE_OTHER, k),
        }
    };

    scratch.as_mut_slice()[*len] = BucketedWiener4Core { core, bucket, k };
    *len += 1;
}

#[inline]
fn add_wiener4_scratch_bucket(
    total_lp: &mut f64,
    grad: &mut [f64; 4],
    scratch: &[BucketedWiener4Core],
    bucket: u8,
    branch: SeriesBranch,
    fixed_k: Option<usize>,
) {
    #[cfg(feature = "simd")]
    if branch == SeriesBranch::LargeTime
        && matches!(fixed_k, Some(4 | 5 | 6))
    {
        add_wiener4_large_simd_bucket(total_lp, grad, scratch, bucket, fixed_k.unwrap());
        return;
    }

    for entry in scratch.iter().filter(|entry| entry.bucket == bucket) {
        let k = fixed_k.unwrap_or(entry.k);
        add_wiener4_eval(
            total_lp,
            grad,
            Wiener4.fused_from_core_with_branch(&entry.core, branch, k),
        );
    }
}

#[cfg(feature = "simd")]
#[inline]
fn add_wiener4_large_simd_bucket(
    total_lp: &mut f64,
    grad: &mut [f64; 4],
    scratch: &[BucketedWiener4Core],
    bucket: u8,
    k: usize,
) {
    let mut lanes = [None; 4];
    let mut filled = 0;

    for entry in scratch.iter().filter(|entry| entry.bucket == bucket) {
        lanes[filled] = Some(entry.core);
        filled += 1;
        if filled == 4 {
            if !add_wiener4_large_simd4(total_lp, grad, lanes.map(Option::unwrap), k) {
                for core in lanes.map(Option::unwrap) {
                    add_wiener4_eval(
                        total_lp,
                        grad,
                        Wiener4.fused_from_core_with_branch(&core, SeriesBranch::LargeTime, k),
                    );
                }
            }
            lanes = [None; 4];
            filled = 0;
        }
    }

    for lane in lanes.into_iter().flatten() {
        add_wiener4_eval(
            total_lp,
            grad,
            Wiener4.fused_from_core_with_branch(&lane, SeriesBranch::LargeTime, k),
        );
    }
}

#[cfg(feature = "simd")]
#[inline]
fn add_wiener4_large_simd4(
    total_lp: &mut f64,
    grad: &mut [f64; 4],
    cores: [Wiener4Core; 4],
    k: usize,
) -> bool {
    use std::simd::num::SimdFloat;
    use std::simd::{Simd, StdFloat};

    type Vf = Simd<f64, 4>;

    let t_prime = Vf::from_array(cores.map(|c| c.t_prime));
    let w_eff = Vf::from_array(cores.map(|c| c.w_eff));
    let theta = (Vf::splat(1.0) - w_eff) * Vf::splat(std::f64::consts::PI);
    let theta_lanes = theta.to_array();
    let mut sin_j = Vf::from_array(theta_lanes.map(f64::sin));
    let mut cos_j = Vf::from_array(theta_lanes.map(f64::cos));
    let (sin_theta, cos_theta) = (sin_j, cos_j);
    let half_pi2 = 0.5 * std::f64::consts::PI * std::f64::consts::PI;
    let half_pi2_v = Vf::splat(half_pi2);
    let half_pi2_t = t_prime * half_pi2_v;

    let mut raw = Vf::splat(0.0);
    let mut d_t = Vf::splat(0.0);
    let mut d_w = Vf::splat(0.0);

    for j in 1..=k {
        let jf = j as f64;
        let jj = jf * jf;
        let e = (-(Vf::splat(jj - 1.0) * half_pi2_t)).exp();
        raw += Vf::splat(jf) * sin_j * e;
        d_t += Vf::splat(-half_pi2 * jf * (jj - 1.0)) * sin_j * e;
        d_w += Vf::splat(-jj * std::f64::consts::PI) * cos_j * e;

        let next_s = sin_j * cos_theta + cos_j * sin_theta;
        let next_c = cos_j * cos_theta - sin_j * sin_theta;
        sin_j = next_s;
        cos_j = next_c;
    }

    let raw_lanes = raw.to_array();
    if raw_lanes.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return false;
    }

    let pref = Vf::from_array(cores.map(|c| c.pref));
    let a = Vf::from_array(cores.map(|c| c.a));
    let t = Vf::from_array(cores.map(|c| c.t));
    let v = Vf::from_array(cores.map(|c| c.v_eff));
    let one_m_w = Vf::splat(1.0) - w_eff;
    let a2 = a * a;
    let dlog_dtprime = d_t / raw;
    let dlog_dw = d_w / raw;
    let log_series = Vf::splat(super::LN_PI) - half_pi2_t + raw.ln();
    let log_prob = pref + log_series;

    let dpref_da = Vf::splat(-2.0) / a + v * one_m_w;
    let dpref_dt = Vf::splat(-0.5) * v * v;
    let dpref_dw = -a * v;
    let dpref_dv = a * one_m_w - v * t;
    let dlog_dw_eff = dpref_dw + dlog_dw;

    let alpha = dpref_da + dlog_dtprime * (Vf::splat(-2.0) * t / (a2 * a));
    let tau = -(dpref_dt + dlog_dtprime / a2);
    let beta = Vf::from_array(cores.map(|c| c.beta_sign)) * dlog_dw_eff;
    let delta = Vf::from_array(cores.map(|c| c.delta_sign)) * dpref_dv;

    *total_lp += log_prob.reduce_sum();
    grad[0] += alpha.reduce_sum();
    grad[1] += tau.reduce_sum();
    grad[2] += beta.reduce_sum();
    grad[3] += delta.reduce_sum();
    true
}

impl Wiener4 {
    /// Large-time truncation count from the Navarro/Gondan style bound
    /// This is the count for the π-series
    ///
    /// t_prime = (y - tau) / alpha^2
    #[inline]
    pub(crate) fn k_l(t_prime: f64, log_eps: f64) -> usize {
        series::k_l(t_prime, log_eps)
    }

    /// Small-time truncation count.
    #[inline]
    pub(crate) fn k_s(t_prime: f64, w: f64, log_eps: f64) -> usize {
        series::k_s(t_prime, w, log_eps)
    }

    #[inline]
    pub(crate) fn k_s_grad_w(t_prime: f64, w: f64, log_eps: f64) -> usize {
        series::k_s_grad_w(t_prime, w, log_eps)
    }

    #[inline]
    pub(crate) fn k_l_grad_w(t_prime: f64, log_eps: f64) -> usize {
        series::k_l_grad_w(t_prime, log_eps)
    }

    #[inline]
    pub(crate) fn small_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        series::small_time_log_series(t_prime, w, k)
    }

    #[inline]
    pub(crate) fn small_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        series::small_time_series_raw(t_prime, w, k)
    }

    /// d/dw of the scaled raw small-time sum R_s(t', w).
    #[inline]
    pub(crate) fn small_time_dr_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        series::small_time_dr_dw(t_prime, w, k)
    }

    #[inline]
    pub(crate) fn large_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        series::large_time_log_series(t_prime, w, k)
    }

    #[inline]
    pub(super) fn large_time_scaled_accum(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
        series::large_time_scaled_accum(t_prime, w, k)
    }

    #[inline]
    pub(super) fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Result<Wiener4Core> {
        self.core_from_parts(obs.rt, obs.boundary, params, eps)
    }

    #[inline]
    fn core_from_parts(
        &self,
        rt: f64,
        boundary: Boundary,
        params: &Wiener4Params,
        eps: f64,
    ) -> Result<Wiener4Core> {
        if unlikely(!params.valid() || !rt.is_finite() || !eps.is_finite() || eps <= 0.0) {
            return Err(ProbError::InvalidParameters(
                "Wiener4 requires valid parameters, finite reaction time, and finite positive precision"
                    .to_string(),
            ));
        }

        let t = rt - params.tau;
        if unlikely(t <= 0.0) {
            return Err(ProbError::OutOfSupport(
                "reaction time must exceed non-decision time".to_string(),
            ));
        }

        let a = params.alpha;

        let a2 = a * a;
        let inv_a2 = a2.recip();
        let t_prime = t * inv_a2;

        let (v_eff, w_eff, beta_sign, delta_sign) = match boundary {
            Boundary::Upper => (params.delta, params.beta, 1.0, 1.0),
            Boundary::Lower => (-params.delta, 1.0 - params.beta, -1.0, -1.0),
        };

        if unlikely(!(0.0 < w_eff && w_eff < 1.0)) {
            return Err(ProbError::OutOfSupport(
                "effective starting point must be in (0, 1)".to_string(),
            ));
        }

        let log_a2_inv = -2.0 * a.ln();
        let drift_term = a * v_eff * (1.0 - w_eff);
        let diffusion_term = -0.5 * v_eff * v_eff * t;
        let pref = log_a2_inv + drift_term + diffusion_term;
        let log_eps_eff = (eps.ln() - pref).min(-10.0);

        Ok(Wiener4Core {
            t,
            a,
            t_prime,
            w_eff,
            v_eff,
            pref,
            log_eps_eff,
            beta_sign,
            delta_sign,
        })
    }

    #[must_use]
    pub fn branch_counts(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Option<WienerBranchCounts> {
        let core = self.core(obs, params, eps).ok()?;
        let k_small_density = Self::k_s(core.t_prime, core.w_eff, core.log_eps_eff);
        let k_large_density = Self::k_l(core.t_prime, core.log_eps_eff);
        let k_small_grad_w = k_small_density;
        let k_large_grad_w = Self::k_l_grad_w(core.t_prime, core.log_eps_eff);
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

    /// First-passage-time log density for the 4-parameter Wiener model
    ///
    /// Parameters:
    /// * `y`     - random variable (observed reaction time)
    /// * `tau`   – non-decision time (> 0)
    /// * `beta`  – relative starting point (in (0,1))
    /// * `delta` – drift rate
    /// * `alpha` – boundary separation (> 0)
    /// * `eps`   – truncation tolerance
    ///   supplied as `Wiener4Params`
    ///   Returns `f64::NEG_INFINITY` on invalid inputs or numerically degenerate
    ///   series values
    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Wiener4Eval {
        let mut eval = self.fused(obs, params, eps);
        eval.grad = Wiener4Grad::default(); // or NAN – keep existing behaviour
        eval
    }

    #[inline]
    pub fn try_log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<Wiener4Eval> {
        let mut eval = self.try_fused(obs, params, options)?;
        eval.grad = Wiener4Grad::default();
        Ok(eval)
    }

    #[inline]
    pub fn try_fused(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        options: WienerOptions,
    ) -> Result<Wiener4Eval> {
        let options = options.validate()?;
        let eps = options.series_precision();
        let core = self.core(obs, params, eps)?;
        let eval = self.fused_from_core(&core);
        if likely(eval.log_prob.is_finite()) {
            Ok(eval)
        } else {
            Err(ProbError::NumericalError(
                "Wiener4 evaluation produced a non-finite log density".to_string(),
            ))
        }
    }

    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener4Params, eps: f64) -> Wiener4Eval {
        self.fused_from_parts(obs.rt, obs.boundary, params, eps)
    }

    #[inline]
    pub(super) fn fused_from_parts(
        &self,
        rt: f64,
        boundary: Boundary,
        params: &Wiener4Params,
        eps: f64,
    ) -> Wiener4Eval {
        let core = match self.core_from_parts(rt, boundary, params, eps) {
            Ok(c) => c,
            Err(_) => {
                return fail_eval_4();
            }
        };
        self.fused_from_core(&core)
    }

    #[inline]
    pub(super) fn fused_from_core(&self, core: &Wiener4Core) -> Wiener4Eval {
        let ks_density = Self::k_s(core.t_prime, core.w_eff, core.log_eps_eff);
        let kl_density = Self::k_l(core.t_prime, core.log_eps_eff);
        let k = if unlikely(2 * ks_density <= kl_density) {
            (SeriesBranch::SmallTime, ks_density)
        } else {
            (
                SeriesBranch::LargeTime,
                kl_density.max(Self::k_l_grad_w(core.t_prime, core.log_eps_eff)),
            )
        };
        self.fused_from_core_with_branch(core, k.0, k.1)
    }

    #[inline]
    pub(super) fn fused_from_core_with_branch(
        &self,
        core: &Wiener4Core,
        branch: SeriesBranch,
        k: usize,
    ) -> Wiener4Eval {
        let eval: Option<(f64, f64, f64)> = match branch {
            SeriesBranch::SmallTime => Self::small_branch_fused(core.t_prime, core.w_eff, k),
            SeriesBranch::LargeTime => Self::large_branch_fused(core.t_prime, core.w_eff, k),
        };

        let (log_series, dlog_dtprime, dlog_dw) = match eval {
            Some((ls, dt, dw)) => (ls, dt, dw),
            _ => {
                return fail_eval_4();
            }
        };

        let log_prob = core.pref + log_series;

        let a = core.a;
        let t = core.t;
        let a2 = a * a;

        let dpref_da = -2.0 / a + core.v_eff * (1.0 - core.w_eff);
        let dpref_dt = -0.5 * core.v_eff * core.v_eff;
        let dpref_dw = -a * core.v_eff;
        let dpref_dv = a * (1.0 - core.w_eff) - core.v_eff * t;

        let dlog_dw_eff = dpref_dw + dlog_dw;

        let grad = Wiener4Grad {
            alpha: dpref_da + dlog_dtprime * (-2.0 * t / (a2 * a)),
            tau: -(dpref_dt + dlog_dtprime / a2),
            beta: core.beta_sign * dlog_dw_eff,
            delta: core.delta_sign * dpref_dv,
        };

        Wiener4Eval { log_prob, grad }
    }

    #[inline]
    pub(super) fn small_branch_fused(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
        series::small_branch_fused(t_prime, w, k)
    }

    #[inline]
    pub(super) fn large_branch_fused(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
        series::large_branch_fused(t_prime, w, k)
    }
}

impl Family for Wiener4 {
    type Params = Wiener4Params;
    type Data = Vec<WienerObservation>;

    fn log_prob(params: &Wiener4Params, data: &Vec<WienerObservation>) -> f64 {
        let eps = 1e-6;
        data.iter()
            .map(|obs| Wiener4.log_prob(obs, params, eps).log_prob)
            .sum()
    }
}

impl Family for Wiener5 {
    type Params = Wiener5Params;
    type Data = Vec<WienerObservation>; // Dataset batch

    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        let eps = 1e-12;
        data.iter()
            .map(|obs| Wiener5.log_prob(obs, params, eps).log_prob)
            .sum()
    }
}

impl FusedLogDensity for Target<Wiener4, Vec<WienerObservation>> {
    fn log_prob_and_grad<'a>(&'a self, p: &'a Wiener4Params, grad: &mut [f64; 4]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for obs in &self.data {
            let fused = Wiener4.fused(obs, p, 1e-12);
            let (lp, obs_grad) = (fused.log_prob, fused.grad);
            total_lp += lp;
            grad[0] += obs_grad.alpha;
            grad[1] += obs_grad.tau;
            grad[2] += obs_grad.beta;
            grad[3] += obs_grad.delta;
        }
        total_lp
    }
}

impl GradLogDensity for Target<Wiener4, Vec<WienerObservation>> {
    type Gradient = [f64; 4];
    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

impl LogDensity for Target<Wiener4, WienerObservations> {
    type Point = Wiener4Params;

    fn log_prob(&self, p: &Self::Point) -> f64 {
        let eps = 1e-12;
        self.data
            .upper_rt()
            .iter()
            .map(|&rt| {
                Wiener4.fused_from_parts(rt, Boundary::Upper, p, eps).log_prob
            })
            .chain(self.data.lower_rt().iter().map(|&rt| {
                Wiener4.fused_from_parts(rt, Boundary::Lower, p, eps).log_prob
            }))
            .sum()
    }
}

impl FusedLogDensity for Target<Wiener4, WienerObservations> {
    fn log_prob_and_grad(&self, p: &Wiener4Params, grad: &mut [f64; 4]) -> f64 {
        let eps = 1e-12;
        let mut total_lp = 0.0;
        grad.fill(0.0);

        if self.data.batch_strategy() == BatchStrategy::Direct {
            for &rt in self.data.upper_rt() {
                add_wiener4_direct_parts(&mut total_lp, grad, rt, Boundary::Upper, p, eps);
            }
            for &rt in self.data.lower_rt() {
                add_wiener4_direct_parts(&mut total_lp, grad, rt, Boundary::Lower, p, eps);
            }
            return total_lp;
        }

        let mut failed = false;
        let mut scratch = OwnedBuffer::<BucketedWiener4Core>::new(self.data.len());
        let mut scratch_len = 0;

        for (&rt, boundary) in self
            .data
            .upper_rt()
            .iter()
            .map(|rt| (rt, Boundary::Upper))
            .chain(self.data.lower_rt().iter().map(|rt| (rt, Boundary::Lower)))
        {
            match Wiener4.core_from_parts(rt, boundary, p, eps) {
                Ok(core) => push_wiener4_bucket(&mut scratch, &mut scratch_len, core),
                Err(_) => failed = true,
            }
        }
        scratch.truncate(scratch_len);
        let scratch = scratch.as_slice();

        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_SMALL_K2, SeriesBranch::SmallTime, Some(2));
        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_SMALL_K3, SeriesBranch::SmallTime, Some(3));
        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_SMALL_OTHER, SeriesBranch::SmallTime, None);
        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_LARGE_K4, SeriesBranch::LargeTime, Some(4));
        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_LARGE_K5, SeriesBranch::LargeTime, Some(5));
        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_LARGE_K6, SeriesBranch::LargeTime, Some(6));
        add_wiener4_scratch_bucket(&mut total_lp, grad, scratch, W4_LARGE_OTHER, SeriesBranch::LargeTime, None);

        if unlikely(failed) {
            grad.fill(f64::NAN);
            f64::NEG_INFINITY
        } else {
            total_lp
        }
    }
}

impl GradLogDensity for Target<Wiener4, WienerObservations> {
    type Gradient = [f64; 4];

    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

