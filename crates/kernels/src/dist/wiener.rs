use crate::density::{FusedLogDensity, GradLogDensity};
use crate::dist::traits::{Family, Parameter, Target};
use crate::error::{ProbError, Result};
use ffi::{Bounds, ErrorNorm, Options, hcubature_into};
use std::f64::consts::{LN_2, PI, TAU};

const LOG_PI: f64 = 1.1447298858494001741434273513531_f64;
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
        if v > 0.0 { v.sqrt() } else { 0.0 }
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
    let mut sum = 1.0;
    for j in 1..=k {
        let jf = j as f64;
        let xp = x + 2.0 * jf;
        let xm = 2.0 * jf - x;
        let arg_p = (xp * xp - x_sq) * inv_two_t;
        let arg_m = (xm * xm - x_sq) * inv_two_t;
        sum += (-arg_p).exp() * (1.0 - 2.0 * jf * xp / t_prime);
        sum += (-arg_m).exp() * (1.0 - 2.0 * jf * xm / t_prime);
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

fn finite_diff_partial<F>(f: F, x: f64, lo: f64, hi: f64) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    let h = 1e-6 * (1.0 + x.abs());
    let can_left = x - h > lo;
    let can_right = x + h < hi;
    if can_left && can_right {
        Some((f(x + h) - f(x - h)) / (2.0 * h))
    } else if can_right {
        Some((f(x + h) - f(x)) / h)
    } else if can_left {
        Some((f(x) - f(x - h)) / h)
    } else {
        None
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
                };
            }
        };
        let log_prob = pref + series.log_series;
        let grad = Wiener4Grad {
            alpha: finite_diff_partial(
                |alpha| {
                    Wiener4
                        .log_prob(
                            obs,
                            &Wiener4Params::with_params(
                                alpha,
                                params.tau,
                                params.beta,
                                params.delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.alpha,
                0.0,
                f64::INFINITY,
            )
            .unwrap_or(f64::NAN),
            tau: finite_diff_partial(
                |tau| {
                    Wiener4
                        .log_prob(
                            obs,
                            &Wiener4Params::with_params(
                                params.alpha,
                                tau,
                                params.beta,
                                params.delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.tau,
                0.0,
                obs.rt,
            )
            .unwrap_or(f64::NAN),
            beta: finite_diff_partial(
                |beta| {
                    Wiener4
                        .log_prob(
                            obs,
                            &Wiener4Params::with_params(
                                params.alpha,
                                params.tau,
                                beta,
                                params.delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.beta,
                0.0,
                1.0,
            )
            .unwrap_or(f64::NAN),
            delta: finite_diff_partial(
                |delta| {
                    Wiener4
                        .log_prob(
                            obs,
                            &Wiener4Params::with_params(
                                params.alpha,
                                params.tau,
                                params.beta,
                                delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.delta,
                f64::NEG_INFINITY,
                f64::INFINITY,
            )
            .unwrap_or(f64::NAN),
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
                };
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
                };
            }
        };
        let log_eps_eff = (eps.ln() - pref).min(-10.0);
        let series = match eval_series(t_prime, x, log_eps_eff) {
            Some(s) => s,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad::default(),
                };
            }
        };
        let log_prob = pref + series.log_series;
        let grad = Wiener5Grad {
            alpha: finite_diff_partial(
                |alpha| {
                    Wiener5
                        .log_prob(
                            obs,
                            &Wiener5Params::with_params(
                                alpha,
                                params.tau,
                                params.beta,
                                params.delta,
                                params.s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.alpha,
                0.0,
                f64::INFINITY,
            )
            .unwrap_or(f64::NAN),
            tau: finite_diff_partial(
                |tau| {
                    Wiener5
                        .log_prob(
                            obs,
                            &Wiener5Params::with_params(
                                params.alpha,
                                tau,
                                params.beta,
                                params.delta,
                                params.s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.tau,
                0.0,
                obs.rt,
            )
            .unwrap_or(f64::NAN),
            beta: finite_diff_partial(
                |beta| {
                    Wiener5
                        .log_prob(
                            obs,
                            &Wiener5Params::with_params(
                                params.alpha,
                                params.tau,
                                beta,
                                params.delta,
                                params.s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.beta,
                0.0,
                1.0,
            )
            .unwrap_or(f64::NAN),
            delta: finite_diff_partial(
                |delta| {
                    Wiener5
                        .log_prob(
                            obs,
                            &Wiener5Params::with_params(
                                params.alpha,
                                params.tau,
                                params.beta,
                                delta,
                                params.s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.delta,
                f64::NEG_INFINITY,
                f64::INFINITY,
            )
            .unwrap_or(f64::NAN),
            s_delta: finite_diff_partial(
                |s_delta| {
                    Wiener5
                        .log_prob(
                            obs,
                            &Wiener5Params::with_params(
                                params.alpha,
                                params.tau,
                                params.beta,
                                params.delta,
                                s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.s_delta,
                0.0,
                f64::INFINITY,
            )
            .unwrap_or(f64::NAN),
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
        let beta_lo = 0.5 * params.s_beta;
        let beta_hi = 1.0 - 0.5 * params.s_beta;
        grad[2] = finite_diff_partial(
            |beta| {
                Wiener7
                    .log_prob(
                        obs,
                        &Wiener7Params::with_params(
                            params.alpha,
                            params.tau,
                            beta,
                            params.delta,
                            params.s_beta,
                            params.s_tau,
                            params.s_delta,
                        )
                        .unwrap(),
                        eps,
                    )
                    .log_prob
            },
            params.beta,
            beta_lo,
            beta_hi,
        )
        .unwrap_or(grad[2]);
        let sb_hi = (2.0 * params.beta.min(1.0 - params.beta) - 1e-12).max(0.0);
        grad[5] = if params.s_beta == 0.0 {
            0.0
        } else {
            finite_diff_partial(
                |s_beta| {
                    Wiener7
                        .log_prob(
                            obs,
                            &Wiener7Params::with_params(
                                params.alpha,
                                params.tau,
                                params.beta,
                                params.delta,
                                s_beta,
                                params.s_tau,
                                params.s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.s_beta,
                0.0,
                sb_hi,
            )
            .unwrap_or(0.0)
        };
        grad[6] = if params.s_tau == 0.0 {
            0.0
        } else {
            finite_diff_partial(
                |s_tau| {
                    Wiener7
                        .log_prob(
                            obs,
                            &Wiener7Params::with_params(
                                params.alpha,
                                params.tau,
                                params.beta,
                                params.delta,
                                params.s_beta,
                                s_tau,
                                params.s_delta,
                            )
                            .unwrap(),
                            eps,
                        )
                        .log_prob
                },
                params.s_tau,
                0.0,
                f64::INFINITY,
            )
            .unwrap_or(0.0)
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, label: &str) {
        let tol = atol.max(rtol * expected.abs());
        let err = (actual - expected).abs();
        assert!(
            err <= tol,
            "{label}: actual={actual:.16e}, expected={expected:.16e}, err={err:.3e}, tol={tol:.3e}"
        );
    }

    fn central_diff<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn wiener7_matches_stan_reference_vector() {
        let y = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.85, 8.9, 9.0, 1.0];
        let a = [2.0, 2.0, 10.0, 4.0, 10.0, 1.0, 3.0, 1.7, 2.4, 11.0, 1.5];
        let v = [2.0, 2.0, 4.0, 3.0, -3.0, 1.0, -1.0, -7.3, -4.9, 4.5, 3.0];
        let w = [0.1, 0.5, 0.8, 0.7, 0.1, 0.9, 0.7, 0.92, 0.9, 0.12, 0.5];
        let t0 = [
            1e-9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1,
        ];
        let sv = [0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.7, 0.0, 0.7, 0.5];
        let sw = [0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.01, 0.0, 0.1, 0.2];
        let st0 = [
            0.0, 0.0, 0.0, 0.007, 0.0, 0.007, 0.007, 0.009, 0.009, 0.009, 0.0,
        ];

        let lp_ref = [
            -4.28564747866615,
            -7.52379235146909,
            -26.1551056209248,
            -22.1939134892089,
            -50.0587553794834,
            -37.2817263586318,
            -10.5428662079438,
            -61.5915905674246,
            -117.238967959795,
            -12.5788594249676,
            -3.1448097740735,
        ];
        let da_ref = [
            3.25018678924105,
            3.59980430191399,
            0.876602303160642,
            1.2215517888504,
            -3.02928674030948,
            67.0322498959921,
            1.95334514374631,
            16.4642201959135,
            5.02038145619773,
            0.688439187670968,
            2.63200041459657,
        ];
        let dt0_ref = [
            3.22509339523307,
            2.91155058614589,
            8.21331631900955,
            4.82948967379739,
            1.50069056428102,
            5.25831601347426,
            1.04831896413742,
            2.67457492096193,
            12.8617364931501,
            1.12047317491985,
            5.68799957241344,
        ];
        let dw_ref = [
            5.67120184517318,
            -3.64396221090076,
            -38.7775057146792,
            -14.1837930137393,
            -34.5869239580708,
            -10.4535345681946,
            0.679597983582904,
            -9.93144540834201,
            2.09117200953597,
            -6.0858540417876,
            -3.74870310978083,
        ];
        let dv_ref = [
            -2.199999998,
            -4.44801714898178,
            -13.6940602985224,
            -13.7593709622169,
            21.5540563802381,
            -5.38233555673517,
            8.88475440789056,
            12.1280680728793,
            43.7785246930371,
            -5.68143495684294,
            -1.57639220567218,
        ];
        let dsv_ref = [
            0.0,
            3.42285198319565,
            0.0,
            0.0,
            91.9551438876654,
            4.70180879974639,
            0.0,
            101.80250964211,
            0.0,
            21.4332628706595,
            0.877556017134384,
        ];
        let dsw_ref = [
            0.0,
            0.0,
            10.1052188867058,
            0.0,
            8.72398,
            0.0,
            -0.122807217815892,
            -0.0506322723373748,
            0.0,
            -0.0704990526706635,
            0.0827817310725268,
        ];
        let dst0_ref = [
            0.0,
            0.0,
            0.0,
            2.42836139121338,
            0.0,
            2.64529825657625,
            0.524800556172613,
            1.34278261179603,
            6.55490874737353,
            0.561295838843035,
            0.0,
        ];

        for i in 0..y.len() {
            let obs = WienerObservation {
                rt: y[i],
                boundary: Boundary::Upper,
            };
            let params = Wiener7Params::with_params(a[i], t0[i], w[i], v[i], sw[i], st0[i], sv[i])
                .expect("valid parameter set");
            let eval = Wiener7.fused(&obs, &params, 1e-8);
            assert!(eval.log_prob.is_finite(), "non-finite log_prob at case {i}");
            let g = eval.grad.to_array();

            assert_close(eval.log_prob, lp_ref[i], 2e-4, 1e-5, "log_prob");
            assert_close(g[0], da_ref[i], 3e-4, 1e-4, "d/da");
            assert_close(g[1], dt0_ref[i], 3e-4, 1e-4, "d/dt0");
            if sw[i] == 0.0 {
                assert_close(g[2], dw_ref[i], 5e-4, 1e-4, "d/dw");
            }
            assert_close(g[3], dv_ref[i], 3e-4, 1e-4, "d/dv");
            assert_close(g[4], dsv_ref[i], 2e-3, 1e-4, "d/dsv");
            assert_close(g[5], dsw_ref[i], 2e-3, 1e-4, "d/dsw");
            assert_close(g[6], dst0_ref[i], 2e-3, 1e-4, "d/dst0");
        }
    }

    #[test]
    fn wiener4_fused_grad_matches_finite_difference() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        for _ in 0..40 {
            let alpha = rng.random_range(0.8..3.0);
            let tau = rng.random_range(0.05..0.4);
            let beta = rng.random_range(0.1..0.9);
            let delta = rng.random_range(-2.0..2.0);
            let rt = tau + rng.random_range(0.1..2.5);
            let boundary = if rng.random::<bool>() {
                Boundary::Upper
            } else {
                Boundary::Lower
            };
            let obs = WienerObservation { rt, boundary };
            let params = Wiener4Params::with_params(alpha, tau, beta, delta).unwrap();
            let eval = Wiener4.fused(&obs, &params, 1e-12);
            assert!(eval.log_prob.is_finite());

            let h = 1e-6;
            let d_alpha = central_diff(
                |x| {
                    Wiener4
                        .log_prob(
                            &obs,
                            &Wiener4Params::with_params(x, tau, beta, delta).unwrap(),
                            1e-12,
                        )
                        .log_prob
                },
                alpha,
                h,
            );
            let d_tau = central_diff(
                |x| {
                    Wiener4
                        .log_prob(
                            &obs,
                            &Wiener4Params::with_params(alpha, x, beta, delta).unwrap(),
                            1e-12,
                        )
                        .log_prob
                },
                tau,
                h,
            );
            let d_beta = central_diff(
                |x| {
                    Wiener4
                        .log_prob(
                            &obs,
                            &Wiener4Params::with_params(alpha, tau, x, delta).unwrap(),
                            1e-12,
                        )
                        .log_prob
                },
                beta,
                h,
            );
            let d_delta = central_diff(
                |x| {
                    Wiener4
                        .log_prob(
                            &obs,
                            &Wiener4Params::with_params(alpha, tau, beta, x).unwrap(),
                            1e-12,
                        )
                        .log_prob
                },
                delta,
                h,
            );

            assert_close(eval.grad.alpha, d_alpha, 1e-4, 5e-4, "grad alpha");
            assert_close(eval.grad.tau, d_tau, 1e-4, 5e-4, "grad tau");
            assert_close(eval.grad.beta, d_beta, 1e-4, 5e-4, "grad beta");
            assert_close(eval.grad.delta, d_delta, 1e-4, 5e-4, "grad delta");
        }
    }

    #[test]
    fn wiener4_total_density_normalizes() {
        let params = Wiener4Params::with_params(1.6, 0.2, 0.45, 0.3).unwrap();
        let t_min = 1e-4;
        let t_max = 8.0;
        let n = 8000usize;
        let dt = (t_max - t_min) / (n as f64);
        let mut integral = 0.0;
        for i in 0..=n {
            let t = t_min + (i as f64) * dt;
            let obs_u = WienerObservation {
                rt: params.tau + t,
                boundary: Boundary::Upper,
            };
            let obs_l = WienerObservation {
                rt: params.tau + t,
                boundary: Boundary::Lower,
            };
            let f = Wiener4.log_prob(&obs_u, &params, 1e-12).log_prob.exp()
                + Wiener4.log_prob(&obs_l, &params, 1e-12).log_prob.exp();
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            integral += weight * f * dt;
        }
        assert!(
            (integral - 1.0).abs() < 5e-3,
            "total mass should be close to 1, got {integral}"
        );
    }

    #[test]
    fn series_switch_is_continuous_at_crossovers() {
        let x_values = [0.15, 0.35, 0.55, 0.75, 0.9];
        let log_eps = -18.0;
        for &x in &x_values {
            let mut prev_small = None;
            for i in 1..5000 {
                let t_prime = 1e-4 + (i as f64) * (4.0 - 1e-4) / 5000.0;
                let ks = k_small(t_prime, x, log_eps);
                let kl = k_large(t_prime, log_eps);
                let pick_small = ks < kl;
                if let Some(prev) = prev_small {
                    if prev != pick_small {
                        let ls = small_time_log_series(t_prime, x, ks).unwrap();
                        let ll = large_time_log_series(t_prime, x, kl).unwrap();
                        assert!(
                            (ls - ll).abs() < 5e-3,
                            "series discontinuity too large at t'={t_prime}, x={x}: small={ls}, large={ll}"
                        );
                    }
                }
                prev_small = Some(pick_small);
            }
        }
    }

    #[test]
    fn randomized_valid_wiener7_outputs_are_finite() {
        let mut rng = StdRng::seed_from_u64(0xBAD5EED);
        for _ in 0..50 {
            let alpha = rng.random_range(0.8..3.0);
            let tau = rng.random_range(0.05..0.5);
            let s_tau = rng.random_range(0.0..0.2);
            let s_beta = rng.random_range(0.0..0.2);
            let beta_margin = 0.5 * s_beta + 0.08;
            let beta = rng.random_range(beta_margin..(1.0 - beta_margin));
            let delta = rng.random_range(-2.0..2.0);
            let s_delta = rng.random_range(0.0..0.6);
            let rt = tau + s_tau + rng.random_range(0.15..3.0);
            let boundary = if rng.random::<bool>() {
                Boundary::Upper
            } else {
                Boundary::Lower
            };
            let obs = WienerObservation { rt, boundary };
            let params =
                Wiener7Params::with_params(alpha, tau, beta, delta, s_beta, s_tau, s_delta)
                    .unwrap();
            let eval = Wiener7.fused(&obs, &params, 1e-7);
            assert!(
                eval.log_prob.is_finite(),
                "non-finite lp for {params:?}, {obs:?}"
            );
            for (j, gj) in eval.grad.to_array().iter().enumerate() {
                assert!(
                    gj.is_finite(),
                    "non-finite grad[{j}] for {params:?}, {obs:?}"
                );
            }
        }
    }
}
