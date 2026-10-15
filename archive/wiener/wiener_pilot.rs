use crate::density::{FusedLogDensity, GradLogDensity};
use crate::dist::traits::{Family, Parameter, Target};
use crate::error::{ProbError, Result};
use ffi::{hcubature_into, Bounds, ErrorNorm, Options};
use std::f64::consts::{PI, TAU};

const LOG_PI: f64 = PI.ln();
const LN_2: f64 = f64::LN_2;
const PI_SQ: f64 = PI * PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    Upper,
    Lower,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WienerObservation {
    pub rt: f64,
    pub boundary: Boundary,
}

pub struct Wiener4;
pub struct Wiener5;
pub struct Wiener7;

#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener4Params {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
}

impl Wiener4Params {
    pub const DIM: usize = 4;
    pub fn to_array(&self) -> [f64; 4] {
        [self.alpha, self.tau, self.beta, self.delta]
    }
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(alpha: f64, tau: f64, beta: f64, delta: f64) -> Result<Self> {
        if alpha.is_finite()
            && tau.is_finite()
            && beta.is_finite()
            && delta.is_finite()
            && alpha > 0.0
            && tau >= 0.0
            && 0.0 < beta
            && beta < 1.0
        {
            Ok(Self {
                alpha,
                tau,
                beta,
                delta,
            })
        } else {
            Err(ProbError::InvalidParameters("...".into()))
        }
    }
    pub fn with_params_unchecked(alpha: f64, tau: f64, beta: f64, delta: f64) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
        }
    }

    pub fn valid(&self) -> bool {
        self.alpha.is_finite()
            && self.tau.is_finite()
            && self.beta.is_finite()
            && self.delta.is_finite()
            && self.alpha > 0.0
            && self.tau >= 0.0
            && 0.0 < self.beta
            && self.beta < 1.0
    }
}

impl std::ops::Index<usize> for Wiener4Params {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            _ => panic!(),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener4Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            _ => panic!(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener4Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
}

impl Wiener4Grad {
    pub const DIM: usize = 4;
    pub fn to_array(&self) -> [f64; 4] {
        [self.alpha, self.tau, self.beta, self.delta]
    }
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
        }
    }
}
impl Default for Wiener4Grad {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            tau: f64::NAN,
            beta: f64::NAN,
            delta: f64::NAN,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener4Eval {
    pub log_prob: f64,
    pub grad: Wiener4Grad,
}
impl Wiener4Eval {
    pub fn grad_array(&self) -> [f64; 4] {
        self.grad.to_array()
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener5Params {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
}

impl Wiener5Params {
    pub const DIM: usize = 5;
    pub fn to_array(&self) -> [f64; 5] {
        [self.alpha, self.tau, self.beta, self.delta, self.s_delta]
    }
    pub fn from_array(arr: [f64; 5]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
            s_delta: arr[4],
        }
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_params(alpha: f64, tau: f64, beta: f64, delta: f64, s_delta: f64) -> Result<Self> {
        Wiener4Params::with_params(alpha, tau, beta, delta)?;
        if s_delta.is_finite() && s_delta >= 0.0 {
            Ok(Self {
                alpha,
                tau,
                beta,
                delta,
                s_delta,
            })
        } else {
            Err(ProbError::InvalidParameters("…".into()))
        }
    }
    pub fn with_params_unchecked(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_delta: f64,
    ) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
            s_delta,
        }
    }
    pub fn valid(&self) -> bool {
        self.alpha > 0.0
            && self.tau >= 0.0
            && 0.0 < self.beta
            && self.beta < 1.0
            && self.s_delta >= 0.0
    }
}

impl std::ops::Index<usize> for Wiener5Params {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            4 => &self.s_delta,
            _ => panic!(),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener5Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            4 => &mut self.s_delta,
            _ => panic!(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener5Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
}

impl Wiener5Grad {
    pub const DIM: usize = 5;
    pub fn to_array(&self) -> [f64; 5] {
        [self.alpha, self.tau, self.beta, self.delta, self.s_delta]
    }
    pub fn from_array(arr: [f64; 5]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
            s_delta: arr[4],
        }
    }
}
impl Default for Wiener5Grad {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            tau: f64::NAN,
            beta: f64::NAN,
            delta: f64::NAN,
            s_delta: f64::NAN,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener5Eval {
    pub log_prob: f64,
    pub grad: Wiener5Grad,
}
impl Wiener5Eval {
    pub fn grad_array(&self) -> [f64; 5] {
        self.grad.to_array()
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener7Params {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_beta: f64,
    pub s_tau: f64,
    pub s_delta: f64,
}

impl Wiener7Params {
    pub const DIM: usize = 7;
    pub fn to_array(&self) -> [f64; 7] {
        [
            self.alpha,
            self.tau,
            self.beta,
            self.delta,
            self.s_delta,
            self.s_beta,
            self.s_tau,
        ]
    }
    pub fn from_array(arr: [f64; 7]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
            s_delta: arr[4],
            s_beta: arr[5],
            s_tau: arr[6],
        }
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_params(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_beta: f64,
        s_tau: f64,
        s_delta: f64,
    ) -> Result<Self> {
        if alpha > 0.0
            && tau >= 0.0
            && 0.0 < beta
            && beta < 1.0
            && s_delta >= 0.0
            && (0.0..1.0).contains(&s_beta)
            && s_tau >= 0.0
        {
            Ok(Self {
                alpha,
                tau,
                beta,
                delta,
                s_beta,
                s_tau,
                s_delta,
            })
        } else {
            Err(ProbError::InvalidParameters("…".into()))
        }
    }
    pub fn with_params_unchecked(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_beta: f64,
        s_tau: f64,
        s_delta: f64,
    ) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
            s_beta,
            s_tau,
            s_delta,
        }
    }
}

impl std::ops::Index<usize> for Wiener7Params {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            4 => &self.s_delta,
            5 => &self.s_beta,
            6 => &self.s_tau,
            _ => panic!(),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener7Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            4 => &mut self.s_delta,
            5 => &mut self.s_beta,
            6 => &mut self.s_tau,
            _ => panic!(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener7Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
    pub s_beta: f64,
    pub s_tau: f64,
}

impl Wiener7Grad {
    pub const DIM: usize = 7;
    pub fn to_array(&self) -> [f64; 7] {
        [
            self.alpha,
            self.tau,
            self.beta,
            self.delta,
            self.s_delta,
            self.s_beta,
            self.s_tau,
        ]
    }
    pub fn from_array(arr: [f64; 7]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
            s_delta: arr[4],
            s_beta: arr[5],
            s_tau: arr[6],
        }
    }
}
impl Default for Wiener7Grad {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            tau: f64::NAN,
            beta: f64::NAN,
            delta: f64::NAN,
            s_delta: f64::NAN,
            s_beta: f64::NAN,
            s_tau: f64::NAN,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener7Eval {
    pub log_prob: f64,
    pub grad: Wiener7Grad,
}
impl Wiener7Eval {
    pub fn grad_array(&self) -> [f64; 7] {
        self.grad.to_array()
    }
}

#[derive(Copy, Clone, Debug)]
struct SeriesEval {
    log_series: f64,
    dlog_dtprime: f64,
    dlog_dx: f64,
}

fn k_large(t_prime: f64, log_eps: f64) -> usize {
    if !(t_prime.is_finite() && log_eps.is_finite()) || t_prime <= 0.0 || log_eps >= 0.0 {
        return 1;
    }
    let log_x = PI.ln() + t_prime.ln() + log_eps;
    let term1 = if log_x < 0.0 {
        let v = -2.0 * log_x / (PI * PI * t_prime);
        if v > 0.0 {
            v.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };
    let term2 = 1.0 / (PI * t_prime.sqrt());
    term1.max(term2).ceil().max(1.0) as usize
}

fn k_small(t_prime: f64, x: f64, log_eps: f64) -> usize {
    if !(t_prime.is_finite() && x.is_finite() && log_eps.is_finite()) || t_prime <= 0.0 {
        return 0;
    }
    let u_eps = (TAU.ln() + 2.0 * t_prime.ln() + 2.0 * log_eps).min(-1.0);
    let term1 = 0.5 * ((2.0 * t_prime).sqrt() - x);
    let term2 = {
        let inner = -2.0 * u_eps - 2.0;
        if inner > 0.0 {
            let arg = -t_prime * (u_eps - inner.sqrt());
            if arg > 0.0 {
                0.5 * (arg.sqrt() - x)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            f64::NEG_INFINITY
        }
    };
    term1.max(term2).ceil().max(0.0) as usize
}

fn small_time_log_series(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 || x <= 0.0 || x >= 1.0 {
        return None;
    }
    let inv_two_t = 0.5 / t_prime;
    let max_exponent = x * x * inv_two_t;
    let mut sum = x;
    for j in 1..=k {
        let jf = j as f64;
        let xp = x + 2.0 * jf;
        let xm = 2.0 * jf - x;
        let arg_p = (xp * xp) * inv_two_t - max_exponent;
        let arg_m = (xm * xm) * inv_two_t - max_exponent;
        sum += xp * (-arg_p).exp();
        sum -= xm * (-arg_m).exp();
    }
    if sum > 0.0 {
        let log_pref = -0.5 * TAU.ln() - 1.5 * t_prime.ln();
        Some(log_pref - max_exponent + sum.ln())
    } else {
        None
    }
}

fn small_time_raw(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let inv_two_t = 0.5 / t_prime;
    let x_sq = x * x;
    let mut sum = x;
    for j in 1..=k {
        let jf = j as f64;
        let xp = x + 2.0 * jf;
        let xm = 2.0 * jf - x;
        let arg_p = (xp * xp - x_sq) * inv_two_t;
        let arg_m = (xm * xm - x_sq) * inv_two_t;
        sum += xp * (-arg_p).exp();
        sum -= xm * (-arg_m).exp();
    }
    Some(sum)
}

fn small_time_draw_dt(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let inv_two_t = 0.5 / t_prime;
    let tt = t_prime * t_prime;
    let x_sq = x * x;
    let mut sum = 0.0;
    for j in 1..=k {
        let jf = j as f64;
        let xp = x + 2.0 * jf;
        let xm = 2.0 * jf - x;
        let xp2 = xp * xp;
        let xm2 = xm * xm;
        let arg_p = (xp2 - x_sq) * inv_two_t;
        let arg_m = (xm2 - x_sq) * inv_two_t;
        sum += 0.5 * xp * (xp2 - x_sq) * (-arg_p).exp() / tt;
        sum -= 0.5 * xm * (xm2 - x_sq) * (-arg_m).exp() / tt;
    }
    Some(sum)
}

fn small_time_draw_dx(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let inv_two_t = 0.5 / t_prime;
    let x_sq = x * x;
    let mut sum = -1.0;
    for j in 1..=k {
        let jf = j as f64;
        let xp = x + 2.0 * jf;
        let xm = 2.0 * jf - x;
        let arg_p = (xp * xp - x_sq) * inv_two_t;
        let arg_m = (xm * xm - x_sq) * inv_two_t;
        sum += (-arg_p).exp() * (-1.0 + 2.0 * jf * xp / t_prime);
        sum += (-arg_m).exp() * (-1.0 + 2.0 * jf * xm / t_prime);
    }
    Some(sum)
}

fn large_time_log_series(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let pi2_half = 0.5 * PI * PI;
    let max_exponent = pi2_half * t_prime;
    let mut sum = 0.0;
    for j in 1..=k {
        let jf = j as f64;
        let s = (jf * PI * x).sin();
        if s == 0.0 {
            continue;
        }
        let arg = (jf * jf - 1.0) * max_exponent;
        sum += jf * s * (-arg).exp();
    }
    if sum > 0.0 {
        Some(PI.ln() - max_exponent + sum.ln())
    } else {
        None
    }
}

fn large_time_raw(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let pi2_half = 0.5 * PI * PI;
    let mut sum = 0.0;
    for j in 1..=k {
        let jf = j as f64;
        let s = (jf * PI * x).sin();
        if s == 0.0 {
            continue;
        }
        sum += jf * s * (-jf * jf * pi2_half * t_prime).exp();
    }
    Some(sum)
}

fn large_time_draw_dt(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let pi2 = PI * PI;
    let pi2_half = 0.5 * pi2;
    let mut sum = 0.0;
    for j in 1..=k {
        let jf = j as f64;
        let s = (jf * PI * x).sin();
        if s == 0.0 {
            continue;
        }
        sum -= 0.5 * pi2 * jf * jf * jf * s * (-jf * jf * pi2_half * t_prime).exp();
    }
    Some(sum)
}

fn large_time_draw_dx(t_prime: f64, x: f64, k: usize) -> Option<f64> {
    if t_prime <= 0.0 {
        return None;
    }
    let pi2_half = 0.5 * PI * PI;
    let mut sum = 0.0;
    for j in 1..=k {
        let jf = j as f64;
        let c = (jf * PI * x).cos();
        sum += jf * jf * PI * c * (-jf * jf * pi2_half * t_prime).exp();
    }
    Some(sum)
}

fn eval_series(t_prime: f64, x: f64, log_eps_eff: f64) -> Option<SeriesEval> {
    let ks = k_small(t_prime, x, log_eps_eff);
    let kl = k_large(t_prime, log_eps_eff);
    if ks < kl {
        let log_series = small_time_log_series(t_prime, x, ks)?;
        let raw = small_time_raw(t_prime, x, ks)?;
        if raw <= 0.0 {
            return None;
        }
        let d_raw_dt = small_time_draw_dt(t_prime, x, ks)?;
        let d_raw_dx = small_time_draw_dx(t_prime, x, ks)?;
        let dlog_dtprime = -1.5 / t_prime + x * x / (2.0 * t_prime * t_prime) + d_raw_dt / raw;
        let dlog_dx = -x / t_prime + d_raw_dx / raw;
        Some(SeriesEval {
            log_series,
            dlog_dtprime,
            dlog_dx,
        })
    } else {
        let log_series = large_time_log_series(t_prime, x, kl)?;
        let raw = large_time_raw(t_prime, x, kl)?;
        if raw <= 0.0 {
            return None;
        }
        let d_raw_dt = large_time_draw_dt(t_prime, x, kl)?;
        let d_raw_dx = large_time_draw_dx(t_prime, x, kl)?;
        let dlog_dtprime = -0.5 * PI * PI + d_raw_dt / raw;
        let dlog_dx = d_raw_dx / raw;
        Some(SeriesEval {
            log_series,
            dlog_dtprime,
            dlog_dx,
        })
    }
}

// ---------------------------------------------------------------------------
// Wiener4: log_prob and fused (directly use struct fields)
// ---------------------------------------------------------------------------

impl Wiener4 {
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Wiener4Eval {
        if !params.valid() || obs.rt <= params.tau {
            return Wiener4Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener4Grad::default(),
            };
        }
        let t = obs.rt - params.tau;
        let alpha_sq = params.alpha * params.alpha;
        let (drift_eff, start_rel) = match obs.boundary {
            Boundary::Upper => (params.delta, params.beta),
            Boundary::Lower => (-params.delta, 1.0 - params.beta),
        };
        let x = 1.0 - start_rel; // distance to the absorbing boundary
        let pref = -2.0 * params.alpha.ln() + params.alpha * drift_eff * x
            - 0.5 * drift_eff * drift_eff * t;
        let t_prime = t / alpha_sq;
        let log_eps_eff = (eps.ln() - pref).min(-10.0);
        let log_series = eval_series(t_prime, x, log_eps_eff)
            .map(|s| s.log_series)
            .unwrap_or(f64::NEG_INFINITY);
        Wiener4Eval {
            log_prob: pref + log_series,
            grad: Wiener4Grad::default(),
        }
    }

    pub fn fused(&self, obs: &WienerObservation, params: &Wiener4Params, eps: f64) -> Wiener4Eval {
        if !params.valid() || obs.rt <= params.tau {
            return Wiener4Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener4Grad::default(),
            };
        }
        let t = obs.rt - params.tau;
        let alpha_sq = params.alpha * params.alpha;
        let (drift_eff, start_rel) = match obs.boundary {
            Boundary::Upper => (params.delta, params.beta),
            Boundary::Lower => (-params.delta, 1.0 - params.beta),
        };
        let x = 1.0 - start_rel; // distance to absorbing boundary
        let pref = -2.0 * params.alpha.ln() + params.alpha * drift_eff * x
            - 0.5 * drift_eff * drift_eff * t;
        let t_prime = t / alpha_sq;
        let log_eps_eff = (eps.ln() - pref).min(-10.0);
        let series = match eval_series(t_prime, x, log_eps_eff) {
            Some(s) => s,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad::default(),
                }
            }
        };
        let log_prob = pref + series.log_series;

        // derivative of pref w.r.t. alpha, t, x, drift_eff
        let dpref_dalpha = -2.0 / params.alpha + drift_eff * x;
        let dpref_dt = -0.5 * drift_eff * drift_eff;
        let dpref_dx = params.alpha * drift_eff;
        let dpref_dv = params.alpha * x - drift_eff * t;

        let dtprime_dalpha = -2.0 * t / (alpha_sq * params.alpha);
        let dtprime_dt = 1.0 / alpha_sq;

        let dx_dbeta = if obs.boundary == Boundary::Upper {
            -1.0
        } else {
            1.0
        };
        let dv_ddelta = if obs.boundary == Boundary::Upper {
            1.0
        } else {
            -1.0
        };

        let grad = Wiener4Grad {
            alpha: dpref_dalpha + series.dlog_dtprime * dtprime_dalpha,
            tau: -(dpref_dt + series.dlog_dtprime * dtprime_dt),
            beta: dpref_dx * dx_dbeta + series.dlog_dx * dx_dbeta,
            delta: dpref_dv * dv_ddelta,
        };

        Wiener4Eval { log_prob, grad }
    }
}

impl Wiener5 {
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Wiener5Eval {
        let (pref, t_prime, x) = match self.core_values(obs, params) {
            Some(v) => v,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad::default(),
                }
            }
        };
        let log_eps_eff = (eps.ln() - pref).min(-10.0);
        let log_series = eval_series(t_prime, x, log_eps_eff)
            .map(|s| s.log_series)
            .unwrap_or(f64::NEG_INFINITY);
        Wiener5Eval {
            log_prob: pref + log_series,
            grad: Wiener5Grad::default(),
        }
    }

    pub fn fused(&self, obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> Wiener5Eval {
        let (pref, t_prime, x) = match self.core_values(obs, params) {
            Some(v) => v,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad::default(),
                }
            }
        };
        let log_eps_eff = (eps.ln() - pref).min(-10.0);
        let series = match eval_series(t_prime, x, log_eps_eff) {
            Some(s) => s,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad::default(),
                }
            }
        };
        let log_prob = pref + series.log_series;

        let t = obs.rt - params.tau;
        let alpha_sq = params.alpha * params.alpha;
        let drift_eff = match obs.boundary {
            Boundary::Upper => params.delta,
            Boundary::Lower => -params.delta,
        };
        let sv = params.s_delta;
        let sv2 = sv * sv;
        let lam = 1.0 + sv2 * t;
        let lam2 = lam * lam;

        let n = -drift_eff * drift_eff * t
            + 2.0 * params.alpha * drift_eff * x
            + alpha_sq * x * x * sv2;
        let d_n_da = 2.0 * x * (drift_eff + params.alpha * x * sv2);
        let d_n_dv = -2.0 * drift_eff * t + 2.0 * params.alpha * x;
        let d_n_dx = 2.0 * params.alpha * drift_eff + 2.0 * alpha_sq * x * sv2;
        let d_n_dt = -drift_eff * drift_eff;
        let d_n_dsv = 2.0 * alpha_sq * x * x * sv;

        let dpref_dalpha = -2.0 / params.alpha + d_n_da / (2.0 * lam);
        let dpref_dt = -0.5 * sv2 / lam + (d_n_dt * lam - n * sv2) / (2.0 * lam2);
        let dpref_dx = d_n_dx / (2.0 * lam);
        let dpref_dv = d_n_dv / (2.0 * lam);
        let dpref_dsv = -sv * t / lam + (d_n_dsv * lam - n * 2.0 * sv * t) / (2.0 * lam2);

        let dtprime_dalpha = -2.0 * t / (alpha_sq * params.alpha);
        let dtprime_dt = 1.0 / alpha_sq;

        let dx_dbeta = if obs.boundary == Boundary::Upper {
            -1.0
        } else {
            1.0
        };
        let dv_ddelta = if obs.boundary == Boundary::Upper {
            1.0
        } else {
            -1.0
        };

        let grad = Wiener5Grad {
            alpha: dpref_dalpha + series.dlog_dtprime * dtprime_dalpha,
            tau: -(dpref_dt + series.dlog_dtprime * dtprime_dt),
            beta: dpref_dx * dx_dbeta + series.dlog_dx * dx_dbeta,
            delta: dpref_dv * dv_ddelta,
            s_delta: dpref_dsv,
        };

        Wiener5Eval { log_prob, grad }
    }

    // Returns (pref, t_prime, x) where `x` is the distance to the absorbing boundary.
    fn core_values(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
    ) -> Option<(f64, f64, f64)> {
        if !params.valid() || obs.rt <= params.tau {
            return None;
        }
        let t = obs.rt - params.tau;
        let alpha_sq = params.alpha * params.alpha;
        let (v, start_rel) = match obs.boundary {
            Boundary::Upper => (params.delta, params.beta),
            Boundary::Lower => (-params.delta, 1.0 - params.beta),
        };
        let x = 1.0 - start_rel; // distance
        let sv2 = params.s_delta * params.s_delta;
        let lam = 1.0 + sv2 * t;
        let n = -v * v * t + 2.0 * params.alpha * v * x + alpha_sq * x * x * sv2;
        let pref = -2.0 * params.alpha.ln() - 0.5 * lam.ln() + n / (2.0 * lam);
        let t_prime = t / alpha_sq;
        Some((pref, t_prime, x))
    }

    // helpers used by Wiener7: return density + gradient or density only,
    pub(crate) fn fused_scalar(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_delta: f64,
        obs: &WienerObservation,
        eps: f64,
    ) -> Option<(f64, [f64; 5])> {
        let p = Wiener5Params {
            alpha,
            tau,
            beta,
            delta,
            s_delta,
        };
        let e = Wiener5.fused(obs, &p, eps);
        if e.log_prob.is_finite() {
            Some((e.log_prob, e.grad.to_array()))
        } else {
            None
        }
    }

    pub(crate) fn density_only(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_delta: f64,
        obs: &WienerObservation,
        eps: f64,
    ) -> f64 {
        let p = Wiener5Params {
            alpha,
            tau,
            beta,
            delta,
            s_delta,
        };
        Wiener5.log_prob(obs, &p, eps).log_prob.exp()
    }
}

impl Wiener7 {
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        eps: f64,
    ) -> Wiener7Eval {
        if let Some(density) = Self::integrate_density(obs, params, eps) {
            Wiener7Eval {
                log_prob: density.ln(),
                grad: Wiener7Grad::default(),
            }
        } else {
            Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            }
        }
    }

    pub fn fused(&self, obs: &WienerObservation, params: &Wiener7Params, eps: f64) -> Wiener7Eval {
        // no inter-trial variability => delegate to Wiener5
        if params.s_beta == 0.0 && params.s_tau == 0.0 {
            let e = Wiener5.fused(
                obs,
                &Wiener5Params {
                    alpha: params.alpha,
                    tau: params.tau,
                    beta: params.beta,
                    delta: params.delta,
                    s_delta: params.s_delta,
                },
                1e-12,
            );
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

        // validate domain
        if params.alpha <= 0.0
            || params.tau < 0.0
            || params.beta <= 0.0
            || params.beta >= 1.0
            || params.s_delta < 0.0
            || params.s_beta < 0.0
            || params.s_tau < 0.0
            || !obs.rt.is_finite()
            || eps <= 0.0
        {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            };
        }
        if obs.rt <= params.tau {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            };
        }
        if params.s_beta > 0.0
            && (params.beta - params.s_beta / 2.0 <= 0.0
                || params.beta + params.s_beta / 2.0 >= 1.0)
        {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            };
        }
        if params.s_tau > 0.0 && (obs.rt - params.tau) / params.s_tau <= 0.0 {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            };
        }

        // integration bounds and options
        let dim =
            (if params.s_beta != 0.0 { 1 } else { 0 }) + (if params.s_tau != 0.0 { 1 } else { 0 });
        let mut xmin = [0.0; 2];
        let mut xmax = [1.0; 2];
        if params.s_tau != 0.0 {
            xmax[dim - 1] = f64::min(1.0, (obs.rt - params.tau) / params.s_tau);
        }
        let opts = Options {
            max_eval: 6000,
            req_abs_error: 0.0,
            req_rel_error: 0.9 * eps,
            norm: ErrorNorm::L2,
        };
        let eps_inner = 1e-12;

        // helper to map hypercube point to (tau, beta)
        let map = |x: &[f64]| -> (f64, f64) {
            if dim == 1 {
                if params.s_beta != 0.0 {
                    (params.tau, params.beta + params.s_beta * (x[0] - 0.5))
                } else {
                    (params.tau + params.s_tau * x[0], params.beta)
                }
            } else {
                let beta = params.beta + params.s_beta * (x[0] - 0.5);
                let tau = params.tau + params.s_tau * x[1];
                (tau, beta)
            }
        };

        // integrate [density, dens*grad_alpha, dens*grad_tau, dens*grad_beta, dens*grad_delta, dens*grad_s_delta]
        let bounds = Bounds::new(&xmin, &xmax);
        let mut val = [0.0; 6];
        let mut err = [0.0; 6];

        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (tau, beta) = map(x);
            if beta <= 0.0 || beta >= 1.0 {
                for v in fv.iter_mut() {
                    *v = 0.0;
                }
                return 0;
            }
            if let Some((lp, g)) = Wiener5::fused_scalar(
                params.alpha,
                tau,
                beta,
                params.delta,
                params.s_delta,
                obs,
                eps_inner,
            ) {
                let dens = lp.exp();
                fv[0] = dens;
                fv[1] = dens * g[0]; // alpha
                fv[2] = dens * g[1]; // tau
                fv[3] = dens * g[2]; // beta
                fv[4] = dens * g[3]; // delta
                fv[5] = dens * g[4]; // s_delta
            } else {
                for v in fv.iter_mut() {
                    *v = 0.0;
                }
            }
            0
        };

        if hcubature_into(6, bounds, opts, &mut val, &mut err, integrand).is_err() {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            };
        }

        let total_density = val[0];
        if total_density <= 0.0 {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad::default(),
            };
        }
        let log_total = total_density.ln();
        let mut grad = [0.0; 7];
        grad[0] = val[1] / total_density;
        grad[1] = val[2] / total_density;
        grad[2] = val[3] / total_density;
        grad[3] = val[4] / total_density;
        grad[4] = val[5] / total_density;

        // ----- s_beta gradient -----
        grad[5] = if params.s_beta == 0.0 {
            0.0
        } else {
            -1.0 / params.s_beta
                + if params.s_tau == 0.0 {
                    // 1D integral over beta
                    let low = params.beta - params.s_beta / 2.0;
                    let high = params.beta + params.s_beta / 2.0;
                    let d_low = Wiener5::density_only(
                        params.alpha,
                        params.tau,
                        low.clamp(0.0, 1.0),
                        params.delta,
                        params.s_delta,
                        obs,
                        eps_inner,
                    );
                    let d_high = Wiener5::density_only(
                        params.alpha,
                        params.tau,
                        high.clamp(0.0, 1.0),
                        params.delta,
                        params.s_delta,
                        obs,
                        eps_inner,
                    );
                    0.5 * (d_high + d_low) / total_density
                } else {
                    // 2D integral
                    let tau_max = xmax[1];
                    if tau_max <= 0.0 {
                        0.0
                    } else {
                        let mut val_sw = [0.0f64; 1];
                        let mut err_sw = [0.0f64; 1];
                        let bt = Bounds::new(&[0.0], &[tau_max]);
                        let sw_integrand = |x_tau: &[f64], fv: &mut [f64]| -> i32 {
                            let tau_i = params.tau + params.s_tau * x_tau[0];
                            let low = params.beta - params.s_beta / 2.0;
                            let high = params.beta + params.s_beta / 2.0;
                            let d_low = Wiener5::density_only(
                                params.alpha,
                                tau_i,
                                low.clamp(0.0, 1.0),
                                params.delta,
                                params.s_delta,
                                obs,
                                eps_inner,
                            );
                            let d_high = Wiener5::density_only(
                                params.alpha,
                                tau_i,
                                high.clamp(0.0, 1.0),
                                params.delta,
                                params.s_delta,
                                obs,
                                eps_inner,
                            );
                            fv[0] = 0.5 * (d_high + d_low);
                            0
                        };
                        if hcubature_into(1, bt, opts, &mut val_sw, &mut err_sw, sw_integrand)
                            .is_err()
                        {
                            return Wiener7Eval {
                                log_prob: f64::NEG_INFINITY,
                                grad: Wiener7Grad::default(),
                            };
                        }
                        val_sw[0] / total_density
                    }
                }
        };

        // ----- s_tau gradient -----
        grad[6] = if params.s_tau == 0.0 {
            0.0
        } else {
            let t0_max = params.tau + params.s_tau * xmax[dim - 1];
            -total_density / params.s_tau
                + if params.s_beta == 0.0 {
                    Wiener5::density_only(
                        params.alpha,
                        t0_max,
                        params.beta,
                        params.delta,
                        params.s_delta,
                        obs,
                        eps_inner,
                    )
                } else {
                    let mut val_f = [0.0f64; 1];
                    let mut err_f = [0.0f64; 1];
                    let bw = Bounds::new(&[0.0], &[1.0]);
                    let st_integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
                        let b = params.beta + params.s_beta * (x[0] - 0.5);
                        if b <= 0.0 || b >= 1.0 {
                            fv[0] = 0.0;
                            return 0;
                        }
                        fv[0] = Wiener5::density_only(
                            params.alpha,
                            t0_max,
                            b,
                            params.delta,
                            params.s_delta,
                            obs,
                            eps_inner,
                        );
                        0
                    };
                    if hcubature_into(1, bw, opts, &mut val_f, &mut err_f, st_integrand).is_err() {
                        return Wiener7Eval {
                            log_prob: f64::NEG_INFINITY,
                            grad: Wiener7Grad::default(),
                        };
                    }
                    val_f[0]
                }
        } / total_density;

        Wiener7Eval {
            log_prob: log_total,
            grad: Wiener7Grad::from_array(grad),
        }
    }

    fn integrate_density(obs: &WienerObservation, params: &Wiener7Params, eps: f64) -> Option<f64> {
        if params.alpha <= 0.0
            || params.tau < 0.0
            || params.beta <= 0.0
            || params.beta >= 1.0
            || params.s_delta < 0.0
            || params.s_beta < 0.0
            || params.s_tau < 0.0
            || !obs.rt.is_finite()
            || eps <= 0.0
        {
            return None;
        }
        if obs.rt <= params.tau {
            return None;
        }
        if params.s_beta > 0.0
            && (params.beta - params.s_beta / 2.0 <= 0.0
                || params.beta + params.s_beta / 2.0 >= 1.0)
        {
            return None;
        }
        if params.s_tau > 0.0 && (obs.rt - params.tau) / params.s_tau <= 0.0 {
            return None;
        }

        let dim =
            (if params.s_beta != 0.0 { 1 } else { 0 }) + (if params.s_tau != 0.0 { 1 } else { 0 });
        let mut xmin = [0.0; 2];
        let mut xmax = [1.0; 2];
        if params.s_tau != 0.0 {
            xmax[dim - 1] = f64::min(1.0, (obs.rt - params.tau) / params.s_tau);
        }
        let opts = Options {
            max_eval: 6000,
            req_abs_error: 0.0,
            req_rel_error: 0.9 * eps,
            norm: ErrorNorm::L2,
        };
        let eps_inner = 1e-12;

        let bounds = Bounds::new(&xmin, &xmax);
        let mut val = [0.0; 1];
        let mut err = [0.0; 1];
        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (tau, beta) = if dim == 1 {
                if params.s_beta != 0.0 {
                    (params.tau, params.beta + params.s_beta * (x[0] - 0.5))
                } else {
                    (params.tau + params.s_tau * x[0], params.beta)
                }
            } else {
                (
                    params.tau + params.s_tau * x[1],
                    params.beta + params.s_beta * (x[0] - 0.5),
                )
            };
            if beta <= 0.0 || beta >= 1.0 {
                fv[0] = 0.0;
                return 0;
            }
            fv[0] = Wiener5::density_only(
                params.alpha,
                tau,
                beta,
                params.delta,
                params.s_delta,
                obs,
                eps_inner,
            );
            0
        };
        if hcubature_into(1, bounds, opts, &mut val, &mut err, integrand).is_err() {
            None
        } else {
            Some(val[0])
        }
    }
}

impl Family for Wiener4 {
    type Params = Wiener4Params;
    type Data = Vec<WienerObservation>;
    fn log_prob(params: &Wiener4Params, data: &Vec<WienerObservation>) -> f64 {
        data.iter()
            .map(|obs| Wiener4.log_prob(obs, params, 1e-6).log_prob)
            .sum()
    }
}

impl FusedLogDensity for Target<Wiener4, Vec<WienerObservation>> {
    fn log_prob_and_grad<'a>(&'a self, p: &'a Wiener4Params, grad: &mut [f64; 4]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);
        for obs in self.data.iter() {
            let ev = Wiener4.fused(obs, p, 1e-12);
            total_lp += ev.log_prob;
            grad[0] += ev.grad.alpha;
            grad[1] += ev.grad.tau;
            grad[2] += ev.grad.beta;
            grad[3] += ev.grad.delta;
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

impl Family for Wiener5 {
    type Params = Wiener5Params;
    type Data = Vec<WienerObservation>;
    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        data.iter()
            .map(|obs| Wiener5.log_prob(obs, params, 1e-12).log_prob)
            .sum()
    }
}
impl FusedLogDensity for Target<Wiener5, Vec<WienerObservation>> {
    fn log_prob_and_grad(&self, p: &Wiener5Params, grad: &mut [f64; 5]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);
        for obs in self.data.iter() {
            let ev = Wiener5.fused(obs, p, 1e-6);
            total_lp += ev.log_prob;
            grad[0] += ev.grad.alpha;
            grad[1] += ev.grad.tau;
            grad[2] += ev.grad.beta;
            grad[3] += ev.grad.delta;
            grad[4] += ev.grad.s_delta;
        }
        total_lp
    }
}
impl GradLogDensity for Target<Wiener5, Vec<WienerObservation>> {
    type Gradient = [f64; 5];
    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

impl Family for Wiener7 {
    type Params = Wiener7Params;
    type Data = Vec<WienerObservation>;
    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        data.iter()
            .map(|obs| Wiener7.log_prob(obs, params, 1e-4).log_prob)
            .sum()
    }
}
impl FusedLogDensity for Target<Wiener7, Vec<WienerObservation>> {
    fn log_prob_and_grad<'a>(&'a self, p: &'a Wiener7Params, grad: &mut [f64; 7]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);
        for obs in self.data.iter() {
            let ev = Wiener7.fused(obs, p, 1e-4);
            total_lp += ev.log_prob;
            grad[0] += ev.grad.alpha;
            grad[1] += ev.grad.tau;
            grad[2] += ev.grad.beta;
            grad[3] += ev.grad.delta;
            grad[4] += ev.grad.s_delta;
            grad[5] += ev.grad.s_beta;
            grad[6] += ev.grad.s_tau;
        }
        total_lp
    }
}
impl GradLogDensity for Target<Wiener7, Vec<WienerObservation>> {
    type Gradient = [f64; 7];
    fn grad_log_prob(&self, x: &Self::Point, grad: &mut Self::Gradient) {
        self.log_prob_and_grad(x, grad);
    }
}

// Parameter transformations
impl Parameter for Wiener5Params {
    type Constrained = Self;
    type Unconstrained = [f64; 5];
    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.alpha.ln(),
            c.tau.ln(),
            crate::numeric::logit(c.beta),
            c.delta,
            c.s_delta.ln(),
        ]
    }
    fn from_unconstrained(u: &Self::Unconstrained) -> Self::Constrained {
        Wiener5Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            crate::numeric::sigmoid(u[2]),
            u[3],
            u[4].exp(),
        )
    }
    fn log_abs_det_jacobian(u: &Self::Unconstrained) -> f64 {
        u[0] + u[1] + crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2]) + u[4]
    }
}

impl Parameter for Wiener7Params {
    type Constrained = Self;
    type Unconstrained = [f64; 7];
    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.alpha.ln(),
            c.tau.ln(),
            crate::numeric::logit(c.beta),
            c.delta,
            c.s_delta.ln(),
            crate::numeric::logit(c.s_beta),
            c.s_tau.ln(),
        ]
    }
    fn from_unconstrained(u: &Self::Unconstrained) -> Self::Constrained {
        Wiener7Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            crate::numeric::sigmoid(u[2]),
            u[3],
            crate::numeric::sigmoid(u[5]),
            u[6].exp(),
            u[4].exp(),
        )
    }
    fn log_abs_det_jacobian(u: &Self::Unconstrained) -> f64 {
        u[0] + u[1]
            + crate::numeric::log_sigmoid(u[2])
            + crate::numeric::log1m_sigmoid(u[2])
            + u[4]
            + crate::numeric::log_sigmoid(u[5])
            + crate::numeric::log1m_sigmoid(u[5])
            + u[6]
    }
}
