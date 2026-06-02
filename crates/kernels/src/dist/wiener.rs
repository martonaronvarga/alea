use crate::density::{FusedLogDensity, GradLogDensity};
use crate::dist::traits::{Family, Parameter, Target};
use crate::error::{ProbError, Result};
use core::f64::consts::FRAC_1_PI;
use core::option::Option;
use ffi::{hcubature_into, Bounds, ErrorNorm, Options};
use std::f64::consts::{PI, TAU};

const LN_PI: f64 = 1.1447298858494001741434273513530587;

/// 1‑point rule (midpoint)
const GL_1_NODES: [f64; 1] = [0.5];
const GL_1_WTS: [f64; 1] = [1.0];

const GL_5_NODES: [f64; 5] = [
    0.046910077030668074,
    0.2307653449471585,
    0.5,
    0.7692346550528415,
    0.9530899229693319,
];

const GL_5_WTS: [f64; 5] = [
    0.1184634425280945,
    0.23931433524968324,
    0.28444444444444444,
    0.23931433524968324,
    0.1184634425280945,
];

const GL_7_NODES: [f64; 7] = [
    0.025446043828620812,
    0.12923440720030277,
    0.29707742431130146,
    0.5,
    0.7029225756886985,
    0.8707655927996972,
    0.9745539561713792,
];

const GL_7_WTS: [f64; 7] = [
    0.06474248308443484,
    0.13985269574463835,
    0.19091502525255952,
    0.2089795918367347,
    0.19091502525255952,
    0.13985269574463835,
    0.06474248308443484,
];
/// 15‑point Gauss-Legendre on [0,1]
const GL_15_NODES: [f64; 15] = [
    0.006003740989757311,
    0.031363303799647024,
    0.07589670829478634,
    0.13779113431991497,
    0.21451391369573058,
    0.30292432646121825,
    0.3994029530012827,
    0.5,
    0.6005970469987173,
    0.6970756735387817,
    0.7854860863042694,
    0.862208865680085,
    0.9241032917052137,
    0.968636696200353,
    0.9939962590102427,
];

const GL_15_WTS: [f64; 15] = [
    0.015376620998058747,
    0.03518302374405407,
    0.05357961023358602,
    0.06978533896307713,
    0.08313460290849699,
    0.09308050000778105,
    0.09921574266355579,
    0.10128912096278064,
    0.09921574266355579,
    0.09308050000778105,
    0.08313460290849699,
    0.06978533896307713,
    0.05357961023358602,
    0.03518302374405407,
    0.015376620998058747,
];

/// 25‑point Gauss-Legendre on [0,1]
const GL_25_NODES: [f64; 25] = [
    0.002221515104750882,
    0.011668039270241293,
    0.02851271438551284,
    0.05250400106086239,
    0.08327868561958307,
    0.1203703684813211,
    0.16321681576326585,
    0.21116853487938858,
    0.2634986342771425,
    0.31941384709530607,
    0.37806655813950574,
    0.4385676536946448,
    0.5,
    0.5614323463053552,
    0.6219334418604943,
    0.6805861529046939,
    0.7365013657228575,
    0.7888314651206114,
    0.8367831842367341,
    0.8796296315186789,
    0.9167213143804169,
    0.9474959989391376,
    0.9714872856144872,
    0.9883319607297587,
    0.9977784848952491,
];

const GL_25_WTS: [f64; 25] = [
    0.005696899250513088,
    0.013177493307516093,
    0.020469578350653123,
    0.027452347987917656,
    0.034019166906178504,
    0.04007035016750048,
    0.04551413099148185,
    0.0502679745335253,
    0.0542598122371318,
    0.05742912957285579,
    0.059727881767892385,
    0.061121221495155045,
    0.06158802686335772,
    0.061121221495155045,
    0.059727881767892385,
    0.05742912957285579,
    0.0542598122371318,
    0.0502679745335253,
    0.04551413099148185,
    0.04007035016750048,
    0.034019166906178504,
    0.027452347987917656,
    0.020469578350653123,
    0.013177493307516093,
    0.005696899250513088,
];

// TODO
// split to multiple files and organize
// add doctests, document, impl additional traits for parameters

/// Which absorbing boundary terminated evidence accumulation.
///
/// Convention: `Upper` ≡ correct response (drift favours upper boundary
/// when `v > 0`); `Lower` ≡ error response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Process hit the upper boundary at `a`.
    Upper,
    /// Process hit the lower boundary at `0`.
    Lower,
}

/// Wiener type data for scalar interfaces
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WienerObservation {
    pub rt: f64,
    pub boundary: Boundary,
}

/// Canonical Wiener first-passage families in Stan's convention:
/// Wiener4 = standard four parameter (alpha, tau, beta, delta) wiener family
/// Wiener5 = Wiener4 + variance parameter for drift (delta)
/// Wiener7 = Wiener5 + variance parameters for beta and tau
/// Reference:
/// https://mc-stan.org/docs/functions-reference/positive_lower-bounded_distributions.html#wiener-first-passage-time-distribution
pub struct Wiener4;
pub struct Wiener5;
pub struct Wiener7;

/// Canonical Wiener first-passage parameters in Stan's naming:
/// alpha: boundary separation, alpha \in R^+
/// tau: non-decision time, tau \in R^+
/// beta: relative starting point, beta \in (0, 1)
/// delta: drift rate, delta \in R
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener4Params {
    /// Boundary separation (> 0). Larger -> slower, more accurate decisions
    pub alpha: f64,
    /// Non-decision time in seconds (> 0). Minimum possible RT
    pub tau: f64,
    /// Relative starting point `point / a ∈ (0, 1)`. `0.5` = unbiased
    pub beta: f64,
    /// Drift rate. Positive values bias the process toward the upper boundary
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
}

impl std::ops::Index<usize> for Wiener4Params {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            _ => panic!("Wiener4Params index out of range: {}", idx),
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
            _ => panic!("Wiener4Params index out of range: {}", idx),
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
// Default already gives zeros; we can override to set NANs for clarity
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

const NAN_GRAD_4: Wiener4Grad = Wiener4Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
};

impl std::ops::Index<usize> for Wiener4Grad {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            _ => panic!("Wiener4Grad index out of range: {}", idx),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener4Grad {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            _ => panic!("Wiener4Grad index out of range: {}", idx),
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

const FAIL_EVAL_4: Wiener4Eval = Wiener4Eval {
    log_prob: f64::NEG_INFINITY,
    grad: NAN_GRAD_4,
};

#[derive(Copy, Clone, Debug)]
struct Wiener4Core {
    t: f64,
    a: f64,
    t_prime: f64,
    w_eff: f64,
    v_eff: f64,
    pref: f64,
    log_eps_eff: f64,
    beta_sign: f64,
    delta_sign: f64,
}

/// s_delta = standard deviation in drift rate, s_delta \in R^>=0
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener5Params {
    base: Wiener4Params,
    s_delta: f64,
}

impl Wiener5Params {
    pub const DIM: usize = 5;
    pub fn to_array(&self) -> [f64; 5] {
        let b = self.base.to_array();
        [b[0], b[1], b[2], b[3], self.s_delta]
    }
    pub fn from_array(arr: [f64; 5]) -> Self {
        Self {
            base: Wiener4Params::from_array([arr[0], arr[1], arr[2], arr[3]]),
            s_delta: arr[4],
        }
    }
    // Flat accessors
    pub fn alpha(&self) -> f64 {
        self.base.alpha
    }
    pub fn tau(&self) -> f64 {
        self.base.tau
    }
    pub fn beta(&self) -> f64 {
        self.base.beta
    }
    pub fn delta(&self) -> f64 {
        self.base.delta
    }
    pub fn s_delta(&self) -> f64 {
        self.s_delta
    }
}

impl std::ops::Index<usize> for Wiener5Params {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0..=3 => &self.base[idx],
            4 => &self.s_delta,
            _ => panic!("Wiener5Params index out of range: {}", idx),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener5Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0..=3 => &mut self.base[idx],
            4 => &mut self.s_delta,
            _ => panic!("Wiener5Params index out of range: {}", idx),
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

impl std::ops::Index<usize> for Wiener5Grad {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            4 => &self.s_delta,
            _ => panic!("Wiener5Grad index out of range: {}", idx),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener5Grad {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            4 => &mut self.s_delta,
            _ => panic!("Wiener5Grad index out of range: {}", idx),
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

const NAN_GRAD_5: Wiener5Grad = Wiener5Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
    s_delta: f64::NAN,
};

const FAIL_EVAL_5: Wiener5Eval = Wiener5Eval {
    log_prob: f64::NEG_INFINITY,
    grad: NAN_GRAD_5,
};

#[derive(Copy, Clone, Debug)]
struct Wiener5Core {
    base: Wiener4Core,
    sv: f64,
    sv2: f64,
    lam: f64,
}

/// s_beta: standard deviation of beta
/// s_tau: standard deviation of tau
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener7Params {
    base: Wiener5Params,
    s_beta: f64,
    s_tau: f64,
}

impl Wiener7Params {
    pub const DIM: usize = 7;
    pub fn to_array(&self) -> [f64; 7] {
        let b = self.base.to_array(); // [alpha, tau, beta, delta, sv]
        [b[0], b[1], b[2], b[3], b[4], self.s_beta, self.s_tau]
    }
    pub fn from_array(arr: [f64; 7]) -> Self {
        Self {
            base: Wiener5Params::from_array([arr[0], arr[1], arr[2], arr[3], arr[4]]),
            s_beta: arr[5],
            s_tau: arr[6],
        }
    }
    pub fn alpha(&self) -> f64 {
        self.base.base.alpha
    }
    pub fn tau(&self) -> f64 {
        self.base.base.tau
    }
    pub fn beta(&self) -> f64 {
        self.base.base.beta
    }
    pub fn delta(&self) -> f64 {
        self.base.base.delta
    }
    pub fn s_delta(&self) -> f64 {
        self.base.s_delta
    }
    pub fn s_beta(&self) -> f64 {
        self.s_beta
    }
    pub fn s_tau(&self) -> f64 {
        self.s_tau
    }
}

impl std::ops::Index<usize> for Wiener7Params {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0..=4 => &self.base[idx],
            5 => &self.s_beta,
            6 => &self.s_tau,
            _ => panic!("Wiener7Params index out of range: {}", idx),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener7Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0..=4 => &mut self.base[idx],
            5 => &mut self.s_beta,
            6 => &mut self.s_tau,
            _ => panic!("Wiener7Params index out of range: {}", idx),
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
impl std::ops::Index<usize> for Wiener7Grad {
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
            _ => panic!("Wiener7Grad index out of range: {}", idx),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener7Grad {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            4 => &mut self.s_delta,
            5 => &mut self.s_beta,
            6 => &mut self.s_tau,
            _ => panic!("Wiener7Grad index out of range: {}", idx),
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

#[derive(Clone, Debug)]
struct Wiener7Core {
    alpha: f64,
    tau0: f64,
    beta0: f64,
    delta: f64,
    sv: f64,
    sw: f64,
    st0: f64,
    dim: usize,
    xmin: [f64; 2],
    xmax: [f64; 2],
    eps_series: f64,
    opts: Options,
}

#[derive(Copy, Clone, Debug)]
struct SeriesEval {
    log_series: f64,
    dlog_dtprime: f64,
    dlog_dw: f64,
}

impl Wiener4Params {
    /// Default constructor
    /// Calls Default::default()
    /// Returns with `f64::NAN` parameters, invalid for direct use
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructor
    /// Use this to create a correct, validated parameter object
    /// with supplied parameter values
    #[inline]
    pub fn with_params(alpha: f64, tau: f64, beta: f64, delta: f64) -> Result<Self> {
        if alpha.is_finite()
            && tau.is_finite()
            && beta.is_finite()
            && delta.is_finite()
            && alpha > 0.0
            && tau >= 0.0
            && 0_f64 < beta
            && 1_f64 > beta
        {
            Ok(Self {
                alpha,
                tau,
                beta,
                delta,
            })
        } else {
            Err(ProbError::InvalidParameters(
                "Wiener4Params requires alpha > 0, tau >= 0, and beta in (0, 1)".to_string(),
            ))
        }
    }

    /// Constructor
    /// Same as with_params() but without validations
    #[inline]
    pub fn with_params_unchecked(alpha: f64, tau: f64, beta: f64, delta: f64) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
        }
    }
    /// Decision time for a given raw RT. Returns `None` if `rt ≤ tau`.
    #[inline]
    pub fn decision_time(&self, rt: f64) -> Option<f64> {
        let t = rt - self.tau;
        if t > 0.0 && t.is_finite() {
            Some(t)
        } else {
            None
        }
    }
    #[inline]
    pub fn valid(&self) -> bool {
        self.alpha.is_finite()
            && self.tau.is_finite()
            && self.beta.is_finite()
            && self.delta.is_finite()
            && self.alpha > 0.0
            && self.tau >= 0.0
            && 0_f64 < self.beta
            && 1_f64 > self.beta
    }
}

impl Wiener5Params {
    /// Constructs `f64::NAN` by default, use `with_params` or `from_array` to
    /// construct with parameters
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_params(alpha: f64, tau: f64, beta: f64, delta: f64, s_delta: f64) -> Result<Self> {
        let base = Wiener4Params::with_params(alpha, tau, beta, delta)?;

        if s_delta.is_finite() && s_delta >= 0.0 {
            Ok(Self { base, s_delta })
        } else {
            Err(ProbError::InvalidParameters(
                "s_delta must be finite and >= 0.0".to_string(),
            ))
        }
    }

    #[inline]
    pub fn with_params_unchecked(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_delta: f64,
    ) -> Self {
        let base = Wiener4Params::with_params_unchecked(alpha, tau, beta, delta);
        Self { base, s_delta }
    }
}

impl Wiener7Params {
    /// Constructs `f64::NAN` by default, use `with_params` or `from_array` to
    /// construct with parameters
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_params(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_beta: f64,
        s_tau: f64,
        s_delta: f64,
    ) -> Result<Self> {
        let base = Wiener5Params::with_params(alpha, tau, beta, delta, s_delta)?;
        if (0.0..1.0).contains(&s_beta) && (s_tau.is_finite() && s_tau >= 0_f64) {
            Ok(Self {
                base,
                s_beta,
                s_tau,
            })
        } else {
            Err(ProbError::InvalidParameters(
                "s_beta must be in [0, 1) and s_tau must be >= 0".to_string(),
            ))
        }
    }

    #[inline]
    pub fn with_params_unchecked(
        alpha: f64,
        tau: f64,
        beta: f64,
        delta: f64,
        s_beta: f64,
        s_tau: f64,
        s_delta: f64,
    ) -> Self {
        let base = Wiener5Params::with_params_unchecked(alpha, tau, beta, delta, s_delta);
        Self {
            base,
            s_beta,
            s_tau,
        }
    }
}

impl Wiener4 {
    /// Large-time truncation count from the Navarro/Gondan style bound
    /// This is the count for the π-series
    ///
    /// t_prime = (y - tau) / alpha^2
    #[inline]
    pub fn k_l(t_prime: f64, log_eps: f64) -> usize {
        if !t_prime.is_finite() || !log_eps.is_finite() || t_prime <= 0.0 {
            return 1;
        }

        let sqrt_t = t_prime.sqrt();
        let inv_sqrt_t = sqrt_t.recip();

        let k1 = {
            let log_x = LN_PI + t_prime.ln() + log_eps; // ln(pi * t' * eps)
            if log_x < 0.0 {
                (-2.0 * log_x / (PI * PI * t_prime)).sqrt()
            } else {
                0.0
            }
        };

        let k2 = FRAC_1_PI * inv_sqrt_t;

        let k = k1.max(k2).ceil().max(1.0);

        if k.is_finite() && k <= usize::MAX as f64 {
            k as usize
        } else {
            usize::MAX
        }
    }

    /// Small-time truncation count.
    #[inline]
    pub fn k_s(t_prime: f64, w: f64, log_eps: f64) -> usize {
        const LN_TAU: f64 = 1.8378770664093453_f64; // ln(2π)

        if !t_prime.is_finite() || !w.is_finite() || !log_eps.is_finite() || t_prime <= 0.0 {
            return 0;
        }

        let sqrt_2t = (2.0 * t_prime).sqrt();
        let u_eps = (LN_TAU + 2.0 * (t_prime.ln() + log_eps)).min(-1.0);
        let term1 = 0.5 * (sqrt_2t + (1.0 - w));

        let s = (-2.0 * u_eps - 2.0).sqrt();
        let arg = t_prime * (s - u_eps);
        let term2 = 0.5 * (arg.sqrt() + (1.0 - w));

        let k = term1.max(term2).ceil().max(0.0);

        if k.is_finite() && k <= usize::MAX as f64 {
            k as usize
        } else {
            usize::MAX
        }
    }

    #[inline]
    pub fn k_s_grad_w(t_prime: f64, w: f64, log_eps: f64) -> usize {
        const LN_TAU: f64 = 1.8378770664093453;
        const LN_8_OVER_27: f64 = 1.216395324324493; // ln(8/27)

        if !t_prime.is_finite() || !w.is_finite() || !log_eps.is_finite() || t_prime <= 0.0 {
            return 0;
        }
        let sqrt_2t = (2.0 * t_prime).sqrt();
        let one_m_w = 1.0 - w;
        let n1 = 0.5 * (sqrt_2t + one_m_w);

        let u_eps = (LN_TAU + 2.0 * (t_prime.ln() + log_eps)).min(-1.0);
        let s = (-2.0 * u_eps - 2.0).sqrt();
        let arg = t_prime * (s - u_eps);
        let n2 = 0.5 * (arg.sqrt() + one_m_w);

        let k = n1.max(n2).ceil().max(0.0);
        if k.is_finite() && k <= usize::MAX as f64 {
            k as usize
        } else {
            usize::MAX
        }
    }

    #[inline]
    pub fn k_l_grad_w(t_prime: f64, log_eps: f64) -> usize {
        const PI_SQ: f64 = PI * PI;
        const LN_4_OVER_9: f64 = 0.8109302162163288;
        const TWO_LN_PI: f64 = 2.0 * LN_PI; // 2*ln(π)

        if !t_prime.is_finite() || !log_eps.is_finite() || t_prime <= 0.0 {
            return 1;
        }
        let inv_t = t_prime.recip();
        let n1 = (2.0 * inv_t).sqrt() * FRAC_1_PI;

        let u_eps_arg = LN_4_OVER_9 + TWO_LN_PI + 3.0 * t_prime.ln() + 2.0 * log_eps;
        let u_eps = u_eps_arg.min(-1.0);
        let arg = -(u_eps - (-2.0 * u_eps - 2.0).sqrt());
        let n2 = if arg > 0.0 {
            (arg * inv_t).sqrt() * FRAC_1_PI
        } else {
            0.0
        };
        let k = n1.max(n2).ceil().max(1.0);
        if k.is_finite() && k <= usize::MAX as f64 {
            k as usize
        } else {
            usize::MAX
        }
    }

    #[inline]
    pub fn small_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        const LN_TAU: f64 = 1.8378770664093453_f64; // ln(2π)

        if !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0 < w && w < 1.0) {
            return None;
        }

        let one_m_w = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;
        let scale = one_m_w * one_m_w * inv_two_t;
        let log_pref = -0.5 * LN_TAU - 1.5 * t_prime.ln();

        let mut pos = one_m_w;
        let mut neg = 0.0;
        let mut c_pos = 0.0;
        let mut c_neg = 0.0;

        for j in 1..=k {
            let two_j = 2.0 * (j as f64);
            let xp = two_j + one_m_w;
            let xm = two_j - one_m_w;

            let ep = (scale - xp * xp * inv_two_t).exp();
            let en = (scale - xm * xm * inv_two_t).exp();

            // Kahan-compensated accumulation for the positive half
            let y = xp * ep - c_pos;
            let t = pos + y;
            c_pos = (t - pos) - y;
            pos = t;

            // Kahan-compensated accumulation for the negative half
            let y = xm * en - c_neg;
            let t = neg + y;
            c_neg = (t - neg) - y;
            neg = t;
        }

        if !pos.is_finite() || !neg.is_finite() {
            return None;
        }

        if pos <= neg {
            return Some(f64::NEG_INFINITY);
        }

        let ratio = neg / pos;
        Some(log_pref - scale + pos.ln() + (-ratio).ln_1p())
    }

    #[inline]
    pub fn small_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0 < w && w < 1.0) {
            return None;
        }

        let a = 1.0 - w; // a in (0, 1)
        if k == 0 {
            return Some(a);
        }

        let inv_t = t_prime.recip();
        // Kahan compensation for the alternating accumulation
        let mut sum = a;
        let mut c = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let two_j = 2.0 * jf;
            let xp = two_j + a;
            let xm = two_j - a;

            // Exact algebra:
            // xp^2 - a^2 = 4j(j + a)
            // xm^2 - a^2 = 4j(j - a)
            let arg_p = 2.0 * jf * (jf + a) * inv_t;
            let arg_m = 2.0 * jf * (jf - a) * inv_t;

            let term = xp * (-arg_p).exp() - xm * (-arg_m).exp();

            let y = term - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        if sum.is_finite() {
            Some(sum)
        } else {
            None
        }
    }

    #[inline]
    pub fn small_time_dr_dt(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0 < w && w < 1.0) {
            return None;
        }

        if k == 0 {
            return Some(0.0);
        }

        let a = 1.0 - w;
        let inv_t = t_prime.recip();
        // Kahan compensation helps when the alternating contributions nearly cancel.
        let mut sum = 0.0;
        let mut c = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let two_j = 2.0 * jf;
            let xp = two_j + a;
            let xm = two_j - a;

            // Exact algebra:
            // xp^2 - a^2 = 4j(j + a)
            // xm^2 - a^2 = 4j(j - a)
            let arg_p = 2.0 * jf * (jf + a) * inv_t;
            let arg_m = 2.0 * jf * (jf - a) * inv_t;

            // d/dt' of exp(-arg) = exp(-arg) * (arg / t')
            let term =
                xp * (-arg_p).exp() * (arg_p * inv_t) - xm * (-arg_m).exp() * (arg_m * inv_t);

            let y = term - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        if sum.is_finite() {
            Some(sum)
        } else {
            None
        }
    }

    /// d/dw of the scaled raw small-time sum R_s(t', w).
    #[inline]
    pub fn small_time_dr_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0 < w && w < 1.0) {
            return None;
        }

        let a = 1.0 - w;
        let inv_t = 1.0 / t_prime;
        // j = 0 contribution: d/dw of (1 - w) = -1.
        let mut sum = -1.0;
        let mut c = 0.0; // Kahan

        for j in 1..=k {
            let jf = j as f64;
            let two_j = 2.0 * jf;
            let xp = two_j + a;
            let xm = two_j - a;

            // Exact simplifications:
            // xp^2 - a^2 = 4j(j + a)
            // xm^2 - a^2 = 4j(j - a)
            let arg_p = 2.0 * jf * (jf + a) * inv_t;
            let arg_m = 2.0 * jf * (jf - a) * inv_t;

            let ep = (-arg_p).exp();
            let em = (-arg_m).exp();

            let dep_da = ep * (1.0 - xp * 2.0 * jf * inv_t);
            let dem_da = em * (1.0 - xm * 2.0 * jf * inv_t);

            let dsum_da = dep_da + dem_da;
            let dsum_dw = -dsum_da; // da/dw = -1

            let y = dsum_dw - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        if sum.is_finite() {
            Some(sum)
        } else {
            None
        }
    }

    #[inline]
    pub fn large_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        const LN_PI: f64 = 1.144729885849400174143427351353058711_f64;
        const LN_TAU: f64 = 1.837877066409345483560659472811235275_f64; // ln(2π)

        if !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0 < w && w < 1.0) {
            return None;
        }
        if k == 0 {
            return None; // log(0) is not finite
        }

        let base_exp = 0.5 * PI * PI * t_prime;
        let log_pref = LN_PI - 0.5 * LN_TAU - 1.5 * t_prime.ln() - base_exp;

        // Recurrence for sin(jπw), cos(jπw)
        let theta = PI * w;
        let (mut sin_j, mut cos_j) = theta.sin_cos();
        let (sin_theta, cos_theta) = (sin_j, cos_j);

        // Compensated split accumulation: terms with positive and negative sign separately
        let mut pos = 0.0;
        let mut neg = 0.0;
        let mut c_pos = 0.0;
        let mut c_neg = 0.0;

        for j in 1..=k {
            let jf = j as f64;

            // Exact exponent factor relative to j = 1:
            // exp(-(j^2 - 1) * π^2 t' / 2)
            let delta = (jf * jf - 1.0) * base_exp;
            let weight = jf * (-delta).exp();
            let signed = weight * sin_j;

            if signed >= 0.0 {
                let y = signed - c_pos;
                let t = pos + y;
                c_pos = (t - pos) - y;
                pos = t;
            } else {
                let y = -signed - c_neg;
                let t = neg + y;
                c_neg = (t - neg) - y;
                neg = t;
            }

            if j != k {
                let next_sin = sin_j.mul_add(cos_theta, cos_j * sin_theta);
                let next_cos = cos_j.mul_add(cos_theta, -sin_j * sin_theta);
                sin_j = next_sin;
                cos_j = next_cos;
            }
        }

        if !pos.is_finite() || !neg.is_finite() || pos <= neg || pos <= 0.0 {
            return None;
        }

        let ratio = neg / pos;
        if !(0.0..1.0).contains(&ratio) {
            return None;
        }

        Some(log_pref + pos.ln() + (-ratio).ln_1p())
    }

    #[inline]
    fn large_time_scaled_accum(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
        if !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0 < w && w < 1.0) {
            return None;
        }

        if k == 0 {
            return Some((0.0, 0.0, 0.0));
        }

        let theta = PI * (1.0 - w);
        let (mut s_j, mut c_j) = theta.sin_cos();
        let (sin_theta, cos_theta) = (s_j, c_j);

        let half_pi2 = 0.5 * PI * PI;
        let half_pi2_t = half_pi2 * t_prime;

        let mut sum = 0.0;
        let mut d_t = 0.0;
        let mut d_w = 0.0;
        let mut c_sum = 0.0;
        let mut c_dt = 0.0;
        let mut c_dw = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let jj = jf * jf;

            let delta = (jj - 1.0) * half_pi2_t;
            let e = (-delta).exp();

            let term = jf * s_j * e;
            let term_dt = -half_pi2 * jf * (jj - 1.0) * s_j * e;
            let term_dw = -jj * PI * c_j * e;

            let y = term - c_sum;
            let t = sum + y;
            c_sum = (t - sum) - y;
            sum = t;

            let y = term_dt - c_dt;
            let t = d_t + y;
            c_dt = (t - d_t) - y;
            d_t = t;

            let y = term_dw - c_dw;
            let t = d_w + y;
            c_dw = (t - d_w) - y;
            d_w = t;

            if j != k {
                let next_s = s_j.mul_add(cos_theta, c_j * sin_theta);
                let next_c = c_j.mul_add(cos_theta, -s_j * sin_theta);
                s_j = next_s;
                c_j = next_c;
            }
        }

        if sum.is_finite() && d_t.is_finite() && d_w.is_finite() {
            Some((sum, d_t, d_w))
        } else {
            None
        }
    }

    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Option<Wiener4Core> {
        if !params.valid() || !obs.rt.is_finite() || !eps.is_finite() || eps <= 0.0 {
            return None;
        }

        let t = obs.rt - params.tau;
        if t <= 0.0 {
            return None;
        }

        let a = params.alpha;
        if !(a > 0.0 && a.is_finite()) {
            return None;
        }

        let a2 = a * a;
        let inv_a2 = a2.recip();
        let t_prime = t * inv_a2;

        let (v_eff, w_eff, beta_sign, delta_sign) = match obs.boundary {
            Boundary::Upper => (params.delta, params.beta, 1.0, 1.0),
            Boundary::Lower => (-params.delta, 1.0 - params.beta, -1.0, -1.0),
        };

        if !(0.0 < w_eff && w_eff < 1.0) {
            return None;
        }

        let log_a2_inv = -2.0 * a.ln();
        let drift_term = a * v_eff * (1.0 - w_eff);
        let diffusion_term = -0.5 * v_eff * v_eff * t;
        let pref = log_a2_inv + drift_term + diffusion_term;
        let log_eps_eff = (eps.ln() - pref).min(-10.0);

        Some(Wiener4Core {
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
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener4Params, eps: f64) -> Wiener4Eval {
        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: NAN_GRAD_4,
                }
            }
        };

        let ks = Self::k_s(core.t_prime, core.w_eff, core.log_eps_eff);
        let kl = Self::k_l(core.t_prime, core.log_eps_eff);

        let eval: Option<(f64, f64, f64)> = if 2 * ks <= kl {
            Self::small_branch_fused(core.t_prime, core.w_eff, ks)
        } else {
            Self::large_branch_fused(core.t_prime, core.w_eff, kl)
        };

        let (log_series, dlog_dtprime, dlog_dw) = match eval {
            Some((ls, dt, dw)) => (ls, dt, dw),
            _ => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: NAN_GRAD_4,
                }
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
    fn small_branch_fused(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
        let raw = Self::small_time_series_raw(t_prime, w, k)?;
        if raw <= 0.0 {
            return None;
        }

        let d_r_dt = Self::small_time_dr_dt(t_prime, w, k)?;
        let d_r_dw = Self::small_time_dr_dw(t_prime, w, k)?;

        let a = 1.0 - w;
        let inv_t = t_prime.recip();
        let scale = 0.5 * a * a * inv_t;

        let log_series = -0.5 * TAU.ln() - 1.5 * t_prime.ln() - scale + raw.ln();
        let dlog_dtprime = -1.5 * inv_t + scale * inv_t + d_r_dt / raw;
        let dlog_dw = a * inv_t + d_r_dw / raw;

        Some((log_series, dlog_dtprime, dlog_dw))
    }

    #[inline]
    fn large_branch_fused(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
        let (raw, d_r_dt, d_r_dw) = Self::large_time_scaled_accum(t_prime, w, k)?;
        if raw <= 0.0 {
            return None;
        }

        let half_pi2 = 0.5 * PI * PI;
        let log_series = PI.ln() - half_pi2 * t_prime + raw.ln();
        let dlog_dtprime = -half_pi2 + d_r_dt / raw;
        let dlog_dw = d_r_dw / raw;

        Some((log_series, dlog_dtprime, dlog_dw))
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

        for obs in self.data.iter() {
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

impl Wiener5 {
    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Option<Wiener5Core> {
        let Wiener5Params { base, s_delta } = params;
        let WienerObservation { rt, boundary } = obs;

        if !base.valid() || *s_delta < 0.0 || !rt.is_finite() || eps <= 0.0 {
            return None;
        }

        let t = rt - base.tau;
        if t <= 0.0 {
            return None;
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

        Some(Wiener5Core {
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
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> Wiener5Eval {
        let Wiener5Params { base, s_delta } = params;
        let WienerObservation { rt, boundary } = obs;

        if !base.valid() || *s_delta < 0.0 || !rt.is_finite() || eps <= 0.0 {
            return FAIL_EVAL_5;
        }

        let t = rt - base.tau;
        if t <= 0.0 {
            return FAIL_EVAL_5;
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
        let lam = sv2.mul_add(t, 1.0);
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

        let log_eps_eff = (eps.ln() - pref).min(-10.0);
        let ks = Wiener4::k_s(t_prime, w_eff, log_eps_eff);
        let kl = Wiener4::k_l(t_prime, log_eps_eff);

        let eval = if 2 * ks <= kl {
            Wiener4::small_branch_fused(t_prime, w_eff, ks)
        } else {
            Wiener4::large_branch_fused(t_prime, w_eff, kl)
        };

        let (log_series, dlog_dtprime, dlog_dw) = match eval {
            Some((a, b, c)) => (a, b, c),
            None => return FAIL_EVAL_5,
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

        for obs in self.data.iter() {
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

impl Wiener7 {
    /// Build the integration core. Returns `None` for invalid / out‑of‑domain cases.
    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener7Params,
        eps: f64,
    ) -> Option<Wiener7Core> {
        let alpha = params.base.base.alpha;
        let tau0 = params.base.base.tau;
        let beta0 = params.base.base.beta;
        let delta = params.base.base.delta;
        let sv = params.base.s_delta;
        let sw = params.s_beta;
        let st0 = params.s_tau;

        // Basic validity
        if !params.base.base.valid()
            || sv < 0.0
            || sw < 0.0
            || st0 < 0.0
            || !obs.rt.is_finite()
            || eps <= 0.0
        {
            return None;
        }
        if obs.rt <= tau0 {
            return None;
        }
        if sw > 0.0 && (beta0 - sw / 2.0 <= 0.0 || beta0 + sw / 2.0 >= 1.0) {
            return None;
        }
        if st0 > 0.0 && (obs.rt - tau0) / st0 <= 0.0 {
            return None;
        }

        let dim = (if sw != 0.0 { 1 } else { 0 }) + (if st0 != 0.0 { 1 } else { 0 });
        let mut xmin = [0.0; 2];
        let mut xmax = [1.0; 2];

        if st0 != 0.0 {
            xmax[dim - 1] = f64::min(1.0, (obs.rt - tau0) / st0);
        }

        let eps_series = 1e-12; // inner Wiener5 truncation
        let rel_err = 0.9 * eps; // same as Stan (0.9 * precision)
        let opts = Options {
            max_eval: 6000,
            req_abs_error: 0.0,
            req_rel_error: rel_err,
            norm: ErrorNorm::L2,
        };

        Some(Wiener7Core {
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
            Some(c) => c,
            None => {
                return Wiener7Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener7Grad {
                        alpha: f64::NAN,
                        tau: f64::NAN,
                        beta: f64::NAN,
                        delta: f64::NAN,
                        s_delta: f64::NAN,
                        s_beta: f64::NAN,
                        s_tau: f64::NAN,
                    },
                };
            }
        };
        let density = self.eval_density(&core, obs);
        Wiener7Eval {
            log_prob: if density > 0.0 {
                density.ln()
            } else {
                f64::NEG_INFINITY
            },
            grad: Wiener7Grad {
                alpha: f64::NAN,
                tau: f64::NAN,
                beta: f64::NAN,
                delta: f64::NAN,
                s_delta: f64::NAN,
                s_beta: f64::NAN,
                s_tau: f64::NAN,
            },
        }
    }

    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener7Params, eps: f64) -> Wiener7Eval {
        // short‑circuit no variability case
        if params.s_beta == 0.0 && params.s_tau == 0.0 {
            let e = Wiener5.fused(obs, &params.base, 1e-12);

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

        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener7Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener7Grad {
                        alpha: f64::NEG_INFINITY,
                        tau: f64::NEG_INFINITY,
                        beta: f64::NEG_INFINITY,
                        delta: f64::NEG_INFINITY,
                        s_delta: f64::NEG_INFINITY,
                        s_beta: f64::NEG_INFINITY,
                        s_tau: f64::NEG_INFINITY,
                    },
                }
            }
        };
        // self.eval_fused_fixed(&core, obs)
        self.eval_fused(&core, obs)
    }

    // Reference, flat implementation of the 7 parameter log-density with grads
    // superseded by `fused`
    #[inline]
    pub fn _fused_ref(
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

        // no inter‑trial variability -> delegate to Wiener5
        if sw == 0.0 && st0 == 0.0 {
            let e = Wiener5.fused(obs, &params.base, 1e-12);
            let mut grad = [0.0; 7];
            grad[0] = e.grad.alpha;
            grad[1] = e.grad.tau;
            grad[2] = e.grad.beta;
            grad[3] = e.grad.delta;
            grad[4] = e.grad.s_delta;
            // sw & st0 remain 0
            return (e.log_prob, grad);
        }

        if obs.rt <= tau0 {
            return (f64::NEG_INFINITY, [0.0; 7]);
        }
        if st0 > 0.0 && (obs.rt - tau0) / st0 <= 0.0 {
            return (f64::NEG_INFINITY, [0.0; 7]);
        }

        // optional: check that the whole w‑interval lies inside (0,1)
        if sw > 0.0 && (beta0 - sw / 2.0 <= 0.0 || beta0 + sw / 2.0 >= 1.0) {
            return (f64::NEG_INFINITY, [0.0; 7]);
        }

        // integration setup
        let dim = (if sw != 0.0 { 1 } else { 0 }) + (if st0 != 0.0 { 1 } else { 0 });
        let mut xmin = vec![0.0; dim];
        let mut xmax = vec![1.0; dim];

        if st0 != 0.0 {
            let clip = f64::min(1.0, (obs.rt - tau0) / st0);
            xmax[dim - 1] = clip; // τ‑dimension, either index 0 or 1
        }

        let bounds = Bounds::new(&xmin, &xmax);
        let eps_series = 1e-12; // for inner Wiener5 truncation

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
                if sw != 0.0 {
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
            return (f64::NEG_INFINITY, [0.0; 7]);
        }

        let total_density: f64 = val[0];
        if total_density <= 0.0 {
            return (f64::NEG_INFINITY, [0.0; 7]);
        }
        let log_density = total_density.ln();

        let mut grad = [0.0; 7];
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
                    return (f64::NEG_INFINITY, [0.0; 7]);
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
                        return (f64::NEG_INFINITY, [0.0; 7]);
                    }
                    f_end = val_f[0];
                }
                grad[6] = -1.0 / st0 + f_end / (st0 * total_density);
            }
        }
        (log_density, grad)
    }

    // core evaluators
    /// Integrate only the density over the hypercube.
    fn eval_density(&self, core: &Wiener7Core, obs: &WienerObservation) -> f64 {
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

    /// Integrate density + 5 inner‑parameter gradient components.
    /// Then compute s_beta and s_tau derivatives separately.
    fn eval_fused(&self, core: &Wiener7Core, obs: &WienerObservation) -> Wiener7Eval {
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
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad {
                    alpha: f64::NEG_INFINITY,
                    tau: f64::NEG_INFINITY,
                    beta: f64::NEG_INFINITY,
                    delta: f64::NEG_INFINITY,
                    s_delta: f64::NEG_INFINITY,
                    s_beta: f64::NEG_INFINITY,
                    s_tau: f64::NEG_INFINITY,
                },
            };
        }

        let total_density = val[0];
        if total_density <= 0.0 {
            return Wiener7Eval {
                log_prob: f64::NEG_INFINITY,
                grad: Wiener7Grad {
                    alpha: f64::NEG_INFINITY,
                    tau: f64::NEG_INFINITY,
                    beta: f64::NEG_INFINITY,
                    delta: f64::NEG_INFINITY,
                    s_delta: f64::NEG_INFINITY,
                    s_beta: f64::NEG_INFINITY,
                    s_tau: f64::NEG_INFINITY,
                },
            };
        }
        let log_density = total_density.ln();

        let mut grad = [0.0; 7];
        grad[0] = val[1] / total_density; // alpha
        grad[1] = val[2] / total_density; // tau
        grad[2] = val[3] / total_density; // beta
        println!(
            "val[3]: {}, total_density: {}, beta_grad: {}",
            val[3], total_density, grad[2]
        );
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
            let tau_max_idx = if core.sw != 0.0 && core.st0 != 0.0 {
                1
            } else {
                0
            };
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
                    return Wiener7Eval {
                        log_prob: f64::NEG_INFINITY,
                        grad: Wiener7Grad {
                            alpha: f64::NEG_INFINITY,
                            tau: f64::NEG_INFINITY,
                            beta: f64::NEG_INFINITY,
                            delta: f64::NEG_INFINITY,
                            s_delta: f64::NEG_INFINITY,
                            s_beta: f64::NEG_INFINITY,
                            s_tau: f64::NEG_INFINITY,
                        },
                    };
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
                        return Wiener7Eval {
                            log_prob: f64::NEG_INFINITY,
                            grad: Wiener7Grad {
                                alpha: f64::NEG_INFINITY,
                                tau: f64::NEG_INFINITY,
                                beta: f64::NEG_INFINITY,
                                delta: f64::NEG_INFINITY,
                                s_delta: f64::NEG_INFINITY,
                                s_beta: f64::NEG_INFINITY,
                                s_tau: f64::NEG_INFINITY,
                            },
                        };
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
    #[inline(always)]
    fn map_point(core: &Wiener7Core, x: &[f64]) -> (f64, f64) {
        if core.dim == 1 {
            if core.sw != 0.0 {
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
    fn wiener5_density(obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> f64 {
        let lp = Wiener5.log_prob(obs, params, eps).log_prob;
        if lp.is_finite() {
            lp.exp()
        } else {
            0.0
        }
    }

    /// Gauss‑Legendre nodes and weights on [0,1] (scaled from [-1,1]).
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

    //     if total_density <= 0.0 {
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
        for obs in self.data.iter() {
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

impl Parameter for Wiener5Params {
    type Constrained = Self;
    type Unconstrained = [f64; 5];

    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.base.alpha.ln(),
            c.base.tau.ln(),
            crate::numeric::logit(c.base.beta),
            c.base.delta, // Drift is already unconstrained (R)
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

    // Add this to your Parameter trait!
    fn log_abs_det_jacobian(u: &Self::Unconstrained) -> f64 {
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta = crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2]);
        let log_jac_var_delta = u[4];

        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_var_delta
    }
}

impl Parameter for Wiener7Params {
    type Constrained = Self;
    type Unconstrained = [f64; 7];

    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.base.base.alpha.ln(),
            c.base.base.tau.ln(),
            crate::numeric::logit(c.base.base.beta),
            c.base.base.delta, // already unconstrained
            c.base.s_delta.ln(),
            // sw ∈ (0,1) → logit sw
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
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta = crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2]);
        let log_jac_s_delta = u[4];
        let log_jac_sw = crate::numeric::log_sigmoid(u[5]) + crate::numeric::log1m_sigmoid(u[5]);
        let log_jac_st0 = u[6];
        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_s_delta + log_jac_sw + log_jac_st0
    }
}

#[cfg(test)]
mod tests {
    use core::iter::Iterator;

    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};

    // Stan reference values (wiener4_lpdf(y|alpha,tau,beta,delta))
    const STAN_TESTS: &[(&str, WienerObservation, Wiener4Params, f64)] = &[
        (
            "mat/test/prob/wiener/wiener_test/1",
            WienerObservation {
                rt: 1.1,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 2.1,
                tau: 0.3,
                beta: 0.55,
                delta: 0.4,
            },
            -0.892976431870503,
        ),
        (
            "mat/test/prob/wiener/wiener_test/2",
            WienerObservation {
                rt: 2.1,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 4.1,
                tau: 0.6,
                beta: 0.05,
                delta: 0.1,
            },
            -5.28933922584833,
        ),
        (
            "mat/test/prob/wiener/wiener_test/3",
            WienerObservation {
                rt: 1.2,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 10.1,
                tau: 0.35,
                beta: 0.95,
                delta: 0.5,
            },
            -1.36212169454714,
        ),
        (
            "mat/test/prob/wiener/wiener_test/4",
            WienerObservation {
                rt: 50.1,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 4.3,
                tau: 1.05,
                beta: 0.65,
                delta: 0.15,
            },
            -15.3049368722015,
        ),
        (
            "mat/test/prob/wiener/wiener_test/5",
            WienerObservation {
                rt: 1.51,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 1.1,
                tau: 0.9,
                beta: 0.2,
                delta: 10.5,
            },
            -26.4531852275477,
        ),
        (
            "upper_fast",
            WienerObservation {
                rt: 0.5,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 1.5,
                tau: 0.3,
                beta: 0.6,
                delta: -0.5,
            },
            -0.24061277,
        ),
        (
            "upper_slow",
            WienerObservation {
                rt: 1.2,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 2.0,
                tau: 0.2,
                beta: 0.3,
                delta: 0.5,
            },
            -1.1719559,
        ),
        (
            "upper_fast",
            WienerObservation {
                rt: 0.4,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 1.5,
                tau: 0.3,
                beta: 0.4,
                delta: 1.0,
            },
            -0.77042144,
        ),
        (
            "upper_slow",
            WienerObservation {
                rt: 1.5,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 2.0,
                tau: 0.2,
                beta: 0.3,
                delta: 0.5,
            },
            -1.539122,
        ),
        // Edge cases
        (
            "boundary_tau",
            WienerObservation {
                rt: 0.3,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: 1.0,
                tau: 0.3,
                beta: 0.5,
                delta: 0.0,
            },
            f64::NEG_INFINITY,
        ),
        (
            "invalid_alpha",
            WienerObservation {
                rt: 0.5,
                boundary: Boundary::Upper,
            },
            Wiener4Params {
                alpha: -1.0,
                tau: 0.1,
                beta: 0.5,
                delta: 0.0,
            },
            f64::NEG_INFINITY,
        ),
    ];

    #[test]
    fn test_wiener4_log_prob_stan_reference() {
        for (_name, obs, params, expected) in STAN_TESTS {
            let eval = Wiener4.log_prob(obs, params, 1e-8);
            assert_ulps_eq!(eval.log_prob, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_wiener4_gradients_numerical() {
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let params = Wiener4Params::with_params(1.5, 0.2, 0.4, 1.0).unwrap();

        // Finite difference gradients (1e-6 step)
        let h = 1e-8;
        let analytic = Wiener4.fused(&obs, &params, 1e-8).grad;

        // d_alpha
        let p1 = Wiener4Params::with_params_unchecked(
            params.alpha + h,
            params.tau,
            params.beta,
            params.delta,
        );
        let p2 = Wiener4Params::with_params_unchecked(
            params.alpha - h,
            params.tau,
            params.beta,
            params.delta,
        );
        let num_alpha = (Wiener4.log_prob(&obs, &p1, 1e-8).log_prob
            - Wiener4.log_prob(&obs, &p2, 1e-8).log_prob)
            / (2.0 * h);

        assert_relative_eq!(
            analytic.alpha,
            num_alpha,
            epsilon = 1e-4,
            max_relative = 1e-3
        );

        // Similar for tau, beta, delta...
    }

    fn stan_wiener5_data() -> (f64, WienerObservation, Wiener5Params, f64) {
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(1.5, 0.3, 0.55, 0.4, 0.1);
        // Stan log_prob = -0.5239455914
        (1e-10, obs, params, -0.5239455914)
    }

    fn stan_wiener7_data() -> (f64, WienerObservation, Wiener7Params, f64, [f64; 7]) {
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params_unchecked(1.5, 0.3, 0.55, 0.4, 0.05, 0.15, 0.1);
        let log_prob = -0.3338914691;
        let grad = [
            0.1991638664,
            2.500755095,
            -0.3168100084,
            0.5048532259,
            -0.01629080941,
            -0.04512982532,
            1.341071894,
        ];
        (1e-10, obs, params, log_prob, grad)
    }

    #[test]
    fn wiener5_matches_stan() {
        let (eps, obs, params, expected_lp) = stan_wiener5_data();
        let eval = Wiener5.log_prob(&obs, &params, eps);
        println!("Rust Wiener5: {}", eval.log_prob);
        println!("Stan Wiener5: {}", expected_lp);
        assert_relative_eq!(eval.log_prob, expected_lp, epsilon = 1e-8);
    }

    #[test]
    fn wiener5_gradient_matches_stan_numerical() {
        let (eps, obs, params, _expected_lp) = stan_wiener5_data();
        let eval = Wiener5.fused(&obs, &params, eps);

        // Check log_prob consistency
        let lp_only = Wiener5.log_prob(&obs, &params, eps).log_prob;
        assert_relative_eq!(eval.log_prob, lp_only, epsilon = 1e-12);

        // Finite difference verification
        let h = 1e-6;
        let analytic = eval.grad;

        // Test alpha gradient
        let mut p_plus = params;
        let mut p_minus = params;
        p_plus.base.alpha += h;
        p_minus.base.alpha -= h;
        let fd_alpha = (Wiener5.log_prob(&obs, &p_plus, eps).log_prob
            - Wiener5.log_prob(&obs, &p_minus, eps).log_prob)
            / (2.0 * h);
        assert_relative_eq!(analytic.alpha, fd_alpha, epsilon = 1e-4);

        // Test tau gradient
        let mut p_plus = params;
        let mut p_minus = params;
        p_plus.base.tau += h;
        p_minus.base.tau -= h;
        let fd_tau = (Wiener5.log_prob(&obs, &p_plus, eps).log_prob
            - Wiener5.log_prob(&obs, &p_minus, eps).log_prob)
            / (2.0 * h);
        assert_relative_eq!(analytic.tau, fd_tau, epsilon = 1e-4);

        // Test s_delta gradient
        let mut p_plus = params;
        let mut p_minus = params;
        p_plus.s_delta += h;
        p_minus.s_delta -= h;
        let fd_sv = (Wiener5.log_prob(&obs, &p_plus, eps).log_prob
            - Wiener5.log_prob(&obs, &p_minus, eps).log_prob)
            / (2.0 * h);
        assert_relative_eq!(analytic.s_delta, fd_sv, epsilon = 1e-4);
    }

    #[test]
    fn wiener7_matches_stan() {
        let (eps, obs, params, expected_lp, expected_grad) = stan_wiener7_data();
        let eval = Wiener7.fused(&obs, &params, eps);

        println!("Rust Wiener7 log_prob: {}", eval.log_prob);
        println!("Stan Wiener7 log_prob: {}", expected_lp);
        println!("Rust grad: {:?}", eval.grad.to_array());
        println!("Stan grad: {:?}", expected_grad);

        assert_relative_eq!(eval.log_prob, expected_lp, epsilon = 1e-6);

        for (i, g) in eval.grad.to_array().iter().enumerate() {
            assert_relative_eq!(*g, expected_grad[i], epsilon = 1e-4, max_relative = 1e-3);
        }
    }

    #[test]
    fn wiener7_finite_difference_consistency() {
        let (eps, obs, params, _expected_lp, _expected_grad) = stan_wiener7_data();
        let eval = Wiener7.fused(&obs, &params, eps);
        let analytic = eval.grad.to_array();

        // Finite difference for all parameters
        let h = 1e-5;
        let f = |p: &Wiener7Params| Wiener7.log_prob(&obs, p, eps).log_prob;

        let param_names = [
            "alpha", "tau", "beta", "delta", "s_delta", "s_beta", "s_tau",
        ];
        let base_params = params.to_array();

        for i in 0..7 {
            let mut p_plus_arr = base_params;
            let mut p_minus_arr = base_params;
            p_plus_arr[i] += h;
            p_minus_arr[i] -= h;

            let p_plus = Wiener7Params::from_array(p_plus_arr);
            let p_minus = Wiener7Params::from_array(p_minus_arr);

            let fd_grad = (f(&p_plus) - f(&p_minus)) / (2.0 * h);

            println!(
                "{}: analytic={:e}, finite_diff={:e}, diff={:e}",
                param_names[i],
                analytic[i],
                fd_grad,
                (analytic[i] - fd_grad).abs()
            );

            assert_relative_eq!(analytic[i], fd_grad, epsilon = 5e-3, max_relative = 1e-2);
        }
    }

    #[test]
    fn wiener5_log_prob_consistency() {
        let (eps, obs, params, expected_lp) = stan_wiener5_data();

        // Test both boundaries
        let eval_upper = Wiener5.log_prob(&obs, &params, eps);
        assert_relative_eq!(eval_upper.log_prob, expected_lp, epsilon = 1e-8);

        // Test lower boundary
        let obs_lower = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Lower,
        };
        let eval_lower = Wiener5.log_prob(&obs_lower, &params, eps);
        assert!(eval_lower.log_prob.is_finite());
    }

    #[test]
    fn test_truncation_consistency() {
        let params = Wiener4Params {
            alpha: 1.5,
            tau: 0.2,
            beta: 0.4,
            delta: 1.0,
        };
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let core = Wiener4.core(&obs, &params, 1e-8).unwrap();

        let (ks, kl) = (
            Wiener4::k_s(core.t_prime, core.w_eff, core.log_eps_eff),
            Wiener4::k_l(core.t_prime, core.log_eps_eff),
        );
        assert!(ks <= 50, "ks too large: {}", ks); // Reasonable truncation
        assert!(kl <= 20, "kl too large: {}", kl);
    }

    #[test]
    fn test_series_monotonicity() {
        let t_prime = 0.5;
        let w = 0.3;
        let k = 20;

        let small_log = Wiener4::small_time_series_raw(t_prime, w, k).unwrap().ln();
        let large_log = Wiener4::large_time_scaled_accum(t_prime, w, k)
            .unwrap()
            .0
            .ln();

        // Small-time should be more negative for small t'
        assert!(small_log <= large_log + 1e-10);
    }

    #[test]
    fn test_wiener5_validity() {
        let params = Wiener5Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1).unwrap();
        assert!(params.base.valid());

        let invalid = Wiener5Params {
            base: params.base,
            s_delta: -0.1,
        };
        assert!(!invalid.base.valid() || invalid.s_delta < 0.0);
    }

    #[test]
    fn test_wiener7_quadrature() {
        let params = Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap();
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };

        let fused = Wiener7.fused(&obs, &params, 1e-8);
        let (log_pdf, grad) = (fused.log_prob, fused.grad);
        assert!(log_pdf.is_finite());
        for g in grad.to_array().iter() {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_target_fusedlogdensity() {
        let data = vec![
            WienerObservation {
                rt: 0.8,
                boundary: Boundary::Upper,
            },
            WienerObservation {
                rt: 0.6,
                boundary: Boundary::Lower,
            },
        ];
        let params = Wiener4Params::with_params(1.5, 0.2, 0.4, 1.0).unwrap();
        let target = Target::new(Wiener4, data);

        let mut grad = [0.0f64; 4];
        let lp = target.log_prob_and_grad(&params, &mut grad);
        assert!(lp.is_finite());
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn test_parameter_transform() {
        let constrained = Wiener5Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1).unwrap();
        let unconstrained = Wiener5Params::to_unconstrained(&constrained);

        // All unconstrained should be finite
        assert!(unconstrained.iter().all(|u| u.is_finite()));

        let roundtrip = Wiener5Params::from_unconstrained(&unconstrained);
        assert_relative_eq!(
            roundtrip.base.alpha,
            constrained.base.alpha,
            epsilon = 1e-10
        );
        assert_relative_eq!(roundtrip.s_delta, constrained.s_delta, epsilon = 1e-10);

        // Jacobian
        let jac = Wiener5Params::log_abs_det_jacobian(&unconstrained);
        assert!(jac.is_finite());
    }

    #[test]
    #[should_panic(expected = "InvalidParameters")]
    fn test_invalid_params() {
        Wiener4Params::with_params(-1.0, 0.1, 0.5, 0.0).unwrap();
    }

    #[test]
    fn debug_series_derivatives() {
        let t_prime = 0.1; // Small t'
        let w = 0.3;
        let k = 15;

        let log_s = Wiener4::small_time_log_series(t_prime, w, k).unwrap();
        let raw_s = Wiener4::small_time_series_raw(t_prime, w, k).unwrap();

        let missing_pref = -0.5 * std::f64::consts::TAU.ln()
            - 1.5 * t_prime.ln()
            - ((1.0 - w) * (1.0 - w) * 0.5 / t_prime);
        let reconstructed_log_s = missing_pref + raw_s.ln();
        assert_relative_eq!(reconstructed_log_s, log_s, epsilon = 1e-10);
    }

    // Helper: finite‑difference gradient for a scalar parameter via symmetric difference
    fn fd_grad<F: Fn(&Wiener7Params) -> f64>(
        params: &Wiener7Params,
        f: F,
        idx: usize,
        h: f64,
    ) -> f64 {
        let mut p_plus = *params;
        let mut p_minus = *params;
        match idx {
            0 => {
                p_plus.base.base.alpha += h;
                p_minus.base.base.alpha -= h;
            }
            1 => {
                p_plus.base.base.tau += h;
                p_minus.base.base.tau -= h;
            }
            2 => {
                p_plus.base.base.beta += h;
                p_minus.base.base.beta -= h;
            }
            3 => {
                p_plus.base.base.delta += h;
                p_minus.base.base.delta -= h;
            }
            4 => {
                p_plus.base.s_delta += h;
                p_minus.base.s_delta -= h;
            }
            5 => {
                p_plus.s_beta += h;
                p_minus.s_beta -= h;
            }
            6 => {
                p_plus.s_tau += h;
                p_minus.s_tau -= h;
            }
            _ => unreachable!(),
        }
        (f(&p_plus) - f(&p_minus)) / (2.0 * h)
    }

    #[test]
    fn test_wiener7_delegates_to_wiener5_when_no_variability() {
        // sw = 0, st0 = 0 → should match Wiener5 exactly
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let alpha = 1.5;
        let tau = 0.2;
        let beta = 0.4;
        let delta = 1.0;
        let sv = 0.1;
        let w5_params = Wiener5Params::with_params(alpha, tau, beta, delta, sv).unwrap();
        let fused = Wiener5.fused(&obs, &w5_params, 1e-12);
        let (lp5, grad5) = (fused.log_prob, fused.grad);

        let w7_params = Wiener7Params::with_params(alpha, tau, beta, delta, 0.0, 0.0, sv).unwrap();
        let fused = Wiener7.fused(&obs, &w7_params, 1e-4);
        let (lp7, grad7) = (fused.log_prob, fused.grad);
        assert_ulps_eq!(lp7, lp5, epsilon = 1e-6);
        // first five gradients match
        assert_ulps_eq!(grad7[0], grad5.alpha, epsilon = 1e-6);
        assert_ulps_eq!(grad7[1], grad5.tau, epsilon = 1e-6);
        assert_ulps_eq!(grad7[2], grad5.beta, epsilon = 1e-6);
        assert_ulps_eq!(grad7[3], grad5.delta, epsilon = 1e-6);
        assert_ulps_eq!(grad7[4], grad5.s_delta, epsilon = 1e-6);
        // sw, st0 gradients must be NaN or 0 (here 0 because no variability)
        // strict ≈ 0?
        assert!(grad7[5].abs() < 1e-12);
        assert!(grad7[6].abs() < 1e-12);
    }

    #[test]
    fn test_wiener7_gradient_numerical_all_parameters() {
        let obs = WienerObservation {
            rt: 0.9,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(
            1.5, 0.2, 0.4, 0.8, // alpha,tau,beta,delta
            0.1, 0.15, 0.05, // sw, st0, sv
        )
        .unwrap();

        let fused = Wiener7.fused(&obs, &params, 1e-4);
        let grad_anal = fused.grad.to_array();
        let f = |p: &Wiener7Params| Wiener7.log_prob(&obs, p, 1e-4).log_prob;

        let h = 1e-5;
        for (i, item) in grad_anal.iter().enumerate() {
            let num = fd_grad(&params, f, i, h);
            // Allow a somewhat generous tolerance due to adaptive integration noise
            assert_relative_eq!(*item, num, epsilon = 2e-3, max_relative = 5e-3);
        }
    }

    #[test]
    fn test_wiener7_finite_density_nonzero_variability() {
        let obs = WienerObservation {
            rt: 1.2,
            boundary: Boundary::Lower,
        };
        let params = Wiener7Params::with_params(2.0, 0.3, 0.5, -0.2, 0.2, 0.1, 0.3).unwrap();
        let fused = Wiener7.fused(&obs, &params, 1e-4);
        let (lp, grad) = (fused.log_prob, fused.grad);
        assert!(lp.is_finite());
        for g in grad.to_array().iter() {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_wiener7_rt_less_than_t0_returns_neg_inf() {
        let obs = WienerObservation {
            rt: 0.15,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(1.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        let lp = Wiener7.fused(&obs, &params, 1e-4).log_prob;
        assert_eq!(lp, f64::NEG_INFINITY);
    }

    #[test]
    fn test_wiener7_sw_out_of_bounds_returns_neg_inf() {
        // w - sw/2 <= 0  => invalid
        let params = Wiener7Params::with_params(1.0, 0.1, 0.1, 0.0, 0.5, 0.0, 0.0).unwrap(); // w=0.1, sw=0.5 → lower bound = 0.1-0.25 = -0.15 <0
        let obs = WienerObservation {
            rt: 0.5,
            boundary: Boundary::Upper,
        };
        let lp = Wiener7.fused(&obs, &params, 1e-4).log_prob;
        assert_eq!(lp, f64::NEG_INFINITY);
    }

    #[test]
    fn test_wiener7_st0_truncated_interval() {
        // st0 > 0, but (rt - t0)/st0 < 1 → upper limit clipped
        let obs = WienerObservation {
            rt: 0.35,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(1.0, 0.2, 0.5, 0.5, 0.0, 0.0, 0.2).unwrap(); // st0=0.2, rt-t0=0.15, ratio=0.75
        let lp = Wiener7.fused(&obs, &params, 1e-4).log_prob;
        assert!(lp.is_finite()); // integration should succeed
    }

    #[test]
    fn test_wiener7_target_fusedlogdensity_sum_over_batch() {
        let data = vec![
            WienerObservation {
                rt: 0.8,
                boundary: Boundary::Upper,
            },
            WienerObservation {
                rt: 0.6,
                boundary: Boundary::Lower,
            },
        ];
        let params = Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap();
        let target = Target::new(Wiener7, data);
        let mut grad = [0.0; 7];
        let lp = target.log_prob_and_grad(&params, &mut grad);
        assert!(lp.is_finite());
        for g in grad.iter() {
            assert!(g.is_finite());
        }
        // Sum of individual fused calls should match
        let mut sum_lp = 0.0;
        let mut sum_grad = [0.0; 7];
        for obs in target.data.iter() {
            let fused = Wiener7.fused(obs, &params, 1e-4);
            let (lp1, g1) = (fused.log_prob, fused.grad);
            sum_lp += lp1;
            for i in 0..7 {
                sum_grad[i] += g1[i];
            }
        }
        assert_ulps_eq!(lp, sum_lp, epsilon = 1e-10);
        for i in 0..7 {
            assert_ulps_eq!(grad[i], sum_grad[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_wiener7_parameter_roundtrip() {
        let constrained = Wiener7Params::with_params(1.5, 0.2, 0.4, 1.0, 0.1, 0.05, 0.1).unwrap();
        let unconstrained = Wiener7Params::to_unconstrained(&constrained);
        let roundtrip = Wiener7Params::from_unconstrained(&unconstrained);
        assert_relative_eq!(
            roundtrip.base.base.alpha,
            constrained.base.base.alpha,
            epsilon = 1e-10
        );
        assert_relative_eq!(roundtrip.s_beta, constrained.s_beta, epsilon = 1e-10);
        assert_relative_eq!(roundtrip.s_tau, constrained.s_tau, epsilon = 1e-10);
        assert!(Wiener7Params::log_abs_det_jacobian(&unconstrained).is_finite());
    }

    #[test]
    fn test_wiener7_correct_values_from_stan() {
        // Coefficients computed in R with WienR.
        // adapted from Stan tests
        let y_vec = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.85, 8.9, 9.0, 1.0];
        let a_vec = vec![2.0, 2.0, 10.0, 4.0, 10.0, 1.0, 3.0, 1.7, 2.4, 11.0, 1.5];
        let v_vec = vec![2.0, 2.0, 4.0, 3.0, -3.0, 1.0, -1.0, -7.3, -4.9, 4.5, 3.0];
        let w_vec = vec![0.1, 0.5, 0.8, 0.7, 0.1, 0.9, 0.7, 0.92, 0.9, 0.12, 0.5];
        let t0_vec = vec![
            1e-9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1,
        ];
        let sv_vec = vec![0.0, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.7, 0.0, 0.7, 0.5];
        let sw_vec = vec![0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.01, 0.0, 0.1, 0.2];
        let st0_vec = vec![
            0.0, 0.0, 0.0, 0.007, 0.0, 0.007, 0.007, 0.009, 0.009, 0.009, 0.0,
        ];

        let true_dens = vec![
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
        let true_grad_y = vec![
            -3.22509339523307,
            -2.91155058614589,
            -8.21331631900955,
            -4.82948967379739,
            -1.50069056428102,
            -5.25831601347426,
            -1.04831896413742,
            -2.67457492096193,
            -12.8617364931501,
            -1.12047317491985,
            -5.68799957241344,
        ];
        let true_grad_a = vec![
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
        let true_grad_t0 = vec![
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
        let true_grad_w = vec![
            5.67120184517318,
            -3.64396221090076,
            -38.7775057146792,
            -14.1837930137393,
            35.71918681520357,
            -10.4535345681946,
            0.679597983582904,
            -9.93144540834201,
            2.09117200953597,
            -6.0858540417876,
            -3.74870310978083,
        ];
        let true_grad_v = vec![
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
        let true_grad_sv = vec![
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
        let true_grad_sw = vec![
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
        let true_grad_st0 = vec![
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

        let eps = 1e-12;
        let tolerance_log = 1e-6;
        let tolerance_grad = 1e-4;

        for i in 0..y_vec.len() {
            // if i == 4 {
            //     continue;
            // }
            let params = Wiener7Params::with_params_unchecked(
                a_vec[i], t0_vec[i], w_vec[i], v_vec[i], sw_vec[i], st0_vec[i], sv_vec[i],
            );

            // Upper boundary case
            let obs_up = WienerObservation {
                rt: y_vec[i],
                boundary: Boundary::Upper,
            };
            let fused = Wiener7.fused(&obs_up, &params, eps);
            let (lp, grad) = (fused.log_prob, fused.grad);

            assert_relative_eq!(lp, true_dens[i], epsilon = tolerance_log);
            assert_relative_eq!(grad[0], true_grad_a[i], epsilon = tolerance_grad);
            assert_relative_eq!(grad[1], true_grad_t0[i], epsilon = tolerance_grad);
            assert_relative_eq!(grad[2], true_grad_w[i], epsilon = tolerance_grad);
            assert_relative_eq!(grad[3], true_grad_v[i], epsilon = tolerance_grad);
            assert_relative_eq!(grad[4], true_grad_sv[i], epsilon = tolerance_grad);
            assert_relative_eq!(grad[5], true_grad_sw[i], epsilon = tolerance_grad);
            assert_relative_eq!(grad[6], true_grad_st0[i], epsilon = tolerance_grad);
        }
    }

    #[test]
    fn debug_failing_params_truncation() {
        let t_prime = 5.99 / 100.0; // 0.0599
        let w = 0.1;
        let log_eps = (1e-12_f64).ln();
        let ks = Wiener4::k_s(t_prime, w, log_eps);
        let kl = Wiener4::k_l(t_prime, log_eps);
        let ks_gw = Wiener4::k_s_grad_w(t_prime, w, log_eps);
        let kl_gw = Wiener4::k_l_grad_w(t_prime, log_eps);
        println!("ks={}, kl={}, ks_gw={}, kl_gw={}", ks, kl, ks_gw, kl_gw);
        // With these numbers we can check which branch is chosen.
        let use_small = 2 * ks_gw < kl_gw;
        panic!("Use small branch for w-grad: {}", use_small);
    }

    #[test]
    fn debug_small_time_series_failing() {
        let t_prime = 0.0599;
        let w = 0.1;
        let log_eps = (1e-12_f64).ln();
        let k = Wiener4::k_s(t_prime, w, log_eps);
        let raw = Wiener4::small_time_series_raw(t_prime, w, k).unwrap();
        let dr_dt = Wiener4::small_time_dr_dt(t_prime, w, k).unwrap();
        let dr_dw = Wiener4::small_time_dr_dw(t_prime, w, k).unwrap();
        panic!("raw={}, dr_dt={}, dr_dw={}", raw, dr_dt, dr_dw);
    }

    #[test]
    fn debug_wiener5_failing_subcase() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let eval = Wiener5.fused(&obs, &params, 1e-12);
        panic!(
            "Wiener5 log_prob={}, grad.beta={}",
            eval.log_prob, eval.grad.beta
        );
        // If this is wrong, the problem is in Wiener5/Wiener4.
    }

    #[test]
    fn force_large_time_failing() {
        let t_prime = 0.0599;
        let w = 0.1;
        let k = 10; // some large number
        let (raw, dr_dt, dr_dw) = Wiener4::large_time_scaled_accum(t_prime, w, k).unwrap();
        panic!("large raw={}, dr_dw={}", raw, dr_dw);
    }

    #[test]
    fn numerical_check_small_dr_dw() {
        let t_prime = 0.0599;
        let w = 0.1;
        let k = 100; // small k for speed
        let h = 1e-6;
        let r0 = Wiener4::small_time_series_raw(t_prime, w, k).unwrap();
        let r_plus = Wiener4::small_time_series_raw(t_prime, w + h, k).unwrap();
        let num_deriv = (r_plus - r0) / h;
        let analytic = Wiener4::small_time_dr_dw(t_prime, w, k).unwrap();
        println!("Numerical dR/dw = {}, analytic = {}", num_deriv, analytic);
        assert_relative_eq!(analytic, num_deriv, epsilon = 1e-4);
    }

    fn fd_central<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn wiener5_grad_w_matches_fd() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let a = 10.0;
        let t0 = 0.01;
        let w0 = 0.1;
        let v = -3.0;
        let sv = 0.2;
        let eps = 1e-12;

        let analytic = {
            let p = Wiener5Params::with_params_unchecked(a, t0, w0, v, sv);
            Wiener5.fused(&obs, &p, eps).grad.beta
        };

        let numeric = fd_central(
            |w| {
                let p = Wiener5Params::with_params_unchecked(a, t0, w, v, sv);
                Wiener5.fused(&obs, &p, eps).log_prob
            },
            w0,
            1e-6,
        );

        assert!(
            (analytic - numeric).abs() <= 1e-5_f64.max(1e-6 * numeric.abs()),
            "analytic={}, numeric={}",
            analytic,
            numeric
        );
    }

    #[test]
    fn wiener7_grad_beta_matches_fd() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let eps = 1e-12;
        let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.05, 0.2);

        let core0 = Wiener7.core(&obs, &params, eps).unwrap();
        let analytic = Wiener7.eval_fused(&core0, &obs).grad.beta;

        let numeric = fd_central(
            |beta0| {
                let mut core = core0.clone();
                core.beta0 = beta0;
                Wiener7.eval_fused(&core, &obs).log_prob
            },
            core0.beta0,
            1e-6,
        );

        assert!(
            (analytic - numeric).abs() <= 1e-5_f64.max(1e-6 * numeric.abs()),
            "analytic={}, numeric={}",
            analytic,
            numeric
        );
    }

    #[test]
    fn wiener5_reflection_equivalence_upper_vs_lower() {
        let obs_upper = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let obs_lower = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Lower,
        };

        let a = 10.0;
        let t0 = 0.01;
        let w = 0.1;
        let v = -3.0;
        let sv = 0.2;
        let eps = 1e-12;

        let p_upper = Wiener5Params::with_params_unchecked(a, t0, w, v, sv);
        let p_lower_reflected = Wiener5Params::with_params_unchecked(a, t0, 1.0 - w, -v, sv);

        let e1 = Wiener5.fused(&obs_upper, &p_upper, eps);
        let e2 = Wiener5.fused(&obs_lower, &p_lower_reflected, eps);

        assert_relative_eq!(e1.log_prob, e2.log_prob, epsilon = 1e-12);
        assert_relative_eq!(e1.grad.beta, -e2.grad.beta, epsilon = 1e-9);
        assert_relative_eq!(e1.grad.delta, -e2.grad.delta, epsilon = 1e-9);
    }

    #[test]
    fn wiener7_boundary_convention_matches_wiener5() {
        let eps = 1e-12;
        let obs_upper = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params_upper =
            Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);
        let core_upper = Wiener7.core(&obs_upper, &params_upper, eps).unwrap();

        let obs_lower = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Lower,
        };
        let params_lower =
            Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);
        let core_lower = Wiener7.core(&obs_lower, &params_lower, eps).unwrap();

        let e_upper = Wiener7.eval_fused(&core_upper, &obs_upper);
        let e_lower = Wiener7.eval_fused(&core_lower, &obs_lower);

        eprintln!("upper {:?}", e_upper);
        eprintln!("lower {:?}", e_lower);
    }

    #[test]
    fn wiener5_upper_lower_reflection_has_matching_density_and_signed_gradients() {
        let eps = 1e-12;
        let obs_upper = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let obs_lower = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Lower,
        };

        let p_upper = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let p_lower = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.9, 3.0, 0.2);

        let e_upper = Wiener5.fused(&obs_upper, &p_upper, eps);
        let e_lower = Wiener5.fused(&obs_lower, &p_lower, eps);

        assert_relative_eq!(e_upper.log_prob, e_lower.log_prob, epsilon = 1e-12);
        assert_relative_eq!(e_upper.grad.beta, -e_lower.grad.beta, epsilon = 1e-9);
        assert_relative_eq!(e_upper.grad.delta, -e_lower.grad.delta, epsilon = 1e-9);
    }

    #[test]
    fn wiener4_small_and_large_branch_agree_in_overlap_case() {
        let t_prime = 0.06;
        let w = 0.1;
        let log_eps = -12.0_f64.ln();

        let ks = Wiener4::k_s(t_prime, w, log_eps);
        let kl = Wiener4::k_l(t_prime, log_eps);

        // only meaningful if both branches are evaluable in the same neighborhood
        let small = Wiener4::small_branch_fused(t_prime, w, ks).unwrap();
        let large = Wiener4::large_branch_fused(t_prime, w, kl).unwrap();

        assert_relative_eq!(small.0, large.0, epsilon = 1e-10);
        assert_relative_eq!(small.1, large.1, epsilon = 1e-8);
        assert_relative_eq!(small.2, large.2, epsilon = 1e-8);
    }

    #[test]
    fn debug_branch_overlap_manual_k() {
        let t = 0.06;
        let w = 0.1;

        for k in 1..40 {
            let s = Wiener4::small_branch_fused(t, w, k).unwrap();
            let l = Wiener4::large_branch_fused(t, w, k).unwrap();

            println!(
                "k={} small={} large={} diff={}",
                k,
                s.0,
                l.0,
                (s.0 - l.0).abs()
            );
        }
    }

    #[test]
    fn wiener4_small_branch_derivatives_match_fd_at_fixed_k() {
        let t_prime = 0.06;
        let w = 0.1;
        let k = 20;
        let h = 1e-7;

        let ref_val = Wiener4::small_branch_fused(t_prime, w, k).unwrap();

        let fd_dw = fd_central(
            |ww| Wiener4::small_branch_fused(t_prime, ww, k).unwrap().0,
            w,
            h,
        );
        let fd_dt = fd_central(
            |tt| Wiener4::small_branch_fused(tt, w, k).unwrap().0,
            t_prime,
            h,
        );

        assert_relative_eq!(ref_val.2, fd_dw, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(ref_val.1, fd_dt, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    fn wiener4_large_branch_derivatives_match_fd_at_fixed_k() {
        let t_prime = 0.06;
        let w = 0.1;
        let k = 20;
        let h = 1e-7;

        let ref_val = Wiener4::large_branch_fused(t_prime, w, k).unwrap();

        let fd_dw = fd_central(
            |ww| Wiener4::large_branch_fused(t_prime, ww, k).unwrap().0,
            w,
            h,
        );
        let fd_dt = fd_central(
            |tt| Wiener4::large_branch_fused(tt, w, k).unwrap().0,
            t_prime,
            h,
        );

        assert_relative_eq!(ref_val.2, fd_dw, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(ref_val.1, fd_dt, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    fn wiener4_small_branch_converges_with_k() {
        let t_prime = 0.06;
        let w = 0.1;

        let k1 = 20;
        let k2 = 40;
        let a = Wiener4::small_branch_fused(t_prime, w, k1).unwrap();
        let b = Wiener4::small_branch_fused(t_prime, w, k2).unwrap();

        eprintln!("small k{} = {:?}", k1, a);
        eprintln!("small k{} = {:?}", k2, b);

        assert_relative_eq!(a.0, b.0, epsilon = 1e-8, max_relative = 1e-8);
        assert_relative_eq!(a.1, b.1, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(a.2, b.2, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    fn wiener4_large_branch_converges_with_k() {
        let t_prime = 0.06;
        let w = 0.1;

        let k1 = 20;
        let k2 = 40;
        let a = Wiener4::large_branch_fused(t_prime, w, k1).unwrap();
        let b = Wiener4::large_branch_fused(t_prime, w, k2).unwrap();

        eprintln!("large k{} = {:?}", k1, a);
        eprintln!("large k{} = {:?}", k2, b);

        assert_relative_eq!(a.0, b.0, epsilon = 1e-8, max_relative = 1e-8);
        assert_relative_eq!(a.1, b.1, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(a.2, b.2, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    fn wiener4_branch_selection_vs_conservative_truncation() {
        let cases = [
            (0.06, 0.1),
            (0.02, 0.1),
            (0.06, 0.3),
            (0.15, 0.1),
            (0.06, 0.9),
        ];
        let log_eps = -12.0_f64.ln();

        for &(t_prime, w) in &cases {
            let ks = Wiener4::k_s(t_prime, w, log_eps);
            let kl = Wiener4::k_l(t_prime, log_eps);
            let ks_gw = Wiener4::k_s_grad_w(t_prime, w, log_eps);
            let kl_gw = Wiener4::k_l_grad_w(t_prime, log_eps);

            let density_branch = if 2 * ks <= kl { "small" } else { "large" };
            let grad_branch = if 2 * ks_gw <= kl_gw { "small" } else { "large" };

            eprintln!(
                "t'={:.6}, w={:.6}, ks={}, kl={}, ks_gw={}, kl_gw={}, density_branch={}, grad_branch={}",
                t_prime, w, ks, kl, ks_gw, kl_gw, density_branch, grad_branch
            );

            // This does not enforce equality; it only records when the gradient wants a different regime.
            // Those are the points to inspect if the higher-level Wiener5/Wiener7 comparison fails.
        }
    }

    #[test]
    fn wiener4_problem_point_convergence_trace() {
        let t_prime = 0.06;
        let w = 0.1;

        for k in [1usize, 2, 3, 5, 8, 13, 21, 34, 55] {
            let s = Wiener4::small_branch_fused(t_prime, w, k);
            let l = Wiener4::large_branch_fused(t_prime, w, k);
            eprintln!("k={} small={:?} large={:?}", k, s, l);
        }
    }

    fn trapz_1d<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        assert!(
            n >= 2 && n % 2 == 0,
            "use an even n for trapezoidal/Simpson-style resolution"
        );
        let h = (b - a) / (n as f64 - 1.0);
        let mut sum = 0.0;
        for i in 0..n {
            let x = a + (i as f64) * h;
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            sum += w * f(x);
        }
        sum * h
    }

    #[test]
    fn wiener7_i4_hcubature_matches_bruteforce_beta_integration() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

        let eps = 1e-12;
        let core = Wiener7.core(&obs, &params, eps).unwrap();
        let fused = Wiener7.eval_fused(&core, &obs);

        // Only beta varies here because st0 = 0.
        let beta_low = core.beta0 - core.sw / 2.0;
        let beta_high = core.beta0 + core.sw / 2.0;
        let tau = core.tau0;

        let n = 8000usize;

        let dens = trapz_1d(
            |x| {
                let beta = beta_low + (beta_high - beta_low) * x;
                if !(0.0 < beta && beta < 1.0) {
                    return 0.0;
                }
                let p5 = Wiener5Params::with_params_unchecked(
                    core.alpha, tau, beta, core.delta, core.sv,
                );
                Wiener5.fused(&obs, &p5, core.eps_series).log_prob.exp()
            },
            0.0,
            1.0,
            n,
        );

        let grad_beta = trapz_1d(
            |x| {
                let beta = beta_low + (beta_high - beta_low) * x;
                if !(0.0 < beta && beta < 1.0) {
                    return 0.0;
                }
                let p5 = Wiener5Params::with_params_unchecked(
                    core.alpha, tau, beta, core.delta, core.sv,
                );
                let e = Wiener5.fused(&obs, &p5, core.eps_series);
                e.log_prob.exp() * e.grad.beta
            },
            0.0,
            1.0,
            n,
        );

        eprintln!(
            "hcubature lp={}, brute lp={}, hcubature dβ={}, brute dβ={}",
            fused.log_prob,
            dens.ln(),
            fused.grad.beta,
            grad_beta / dens
        );

        assert_relative_eq!(
            fused.log_prob.exp(),
            dens,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            fused.grad.beta,
            grad_beta / dens,
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }

    #[test]
    fn wiener7_i4_reflected_coordinate_is_consistent() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

        let eps = 1e-12;
        let core = Wiener7.core(&obs, &params, eps).unwrap();

        let beta_mid = core.beta0;
        let beta_reflected = 1.0 - beta_mid;

        let p_mid = Wiener5Params::with_params_unchecked(
            core.alpha, core.tau0, beta_mid, core.delta, core.sv,
        );
        let p_ref = Wiener5Params::with_params_unchecked(
            core.alpha,
            core.tau0,
            beta_reflected,
            core.delta,
            core.sv,
        );

        let e_mid = Wiener5.fused(&obs, &p_mid, core.eps_series);
        let e_ref = Wiener5.fused(&obs, &p_ref, core.eps_series);

        eprintln!(
            "mid lp={}, ref lp={}, mid dβ={}, ref dβ={}",
            e_mid.log_prob, e_ref.log_prob, e_mid.grad.beta, e_ref.grad.beta
        );
        panic!()
    }

    #[test]
    fn wiener7_i4_endpoint_sensitivity_beta_bounds() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let mut params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

        let eps = 1e-12;

        let base = Wiener7.eval_fused(&Wiener7.core(&obs, &params, eps).unwrap(), &obs);

        params.base.base.beta = 0.100001;
        let shifted_up = Wiener7.eval_fused(&Wiener7.core(&obs, &params, eps).unwrap(), &obs);

        params.base.base.beta = 0.099999;
        let shifted_down = Wiener7.eval_fused(&Wiener7.core(&obs, &params, eps).unwrap(), &obs);

        eprintln!(
            "base lp={}, up lp={}, down lp={}",
            base.log_prob, shifted_up.log_prob, shifted_down.log_prob
        );

        // This should be smooth unless the support clipping or reflection is biting.
        assert!(base.log_prob.is_finite());
        assert!(shifted_up.log_prob.is_finite());
        assert!(shifted_down.log_prob.is_finite());
    }

    #[test]
    fn wiener7_reference_row_i4_route_trace() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);
        let eps = 1e-12;

        let core = Wiener7.core(&obs, &params, eps).unwrap();
        let fused = Wiener7.eval_fused(&core, &obs);

        let t_prime = (obs.rt - core.tau0) / (core.alpha * core.alpha);
        let ks = Wiener4::k_s(t_prime, core.beta0, core.eps_series.ln());
        let kl = Wiener4::k_l(t_prime, core.eps_series.ln());
        let ks_gw = Wiener4::k_s_grad_w(t_prime, core.beta0, core.eps_series.ln());
        let kl_gw = Wiener4::k_l_grad_w(t_prime, core.eps_series.ln());

        eprintln!(
            "core={:?}\nlog_prob={}\ngrad={:?}\nt_prime={}\nks={}, kl={}, ks_gw={}, kl_gw={}",
            core, fused.log_prob, fused.grad, t_prime, ks, kl, ks_gw, kl_gw
        );
        panic!()
    }

    #[test]
    fn wiener4_selected_k_is_close_to_high_k_on_i4_case() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let eps = 1e-12;

        let core = Wiener5.core(&obs, &params, eps).unwrap();
        let t_prime = core.base.t_prime;
        let w = core.base.w_eff;
        let log_eps = core.base.log_eps_eff;

        let ks = Wiener4::k_s(t_prime, w, log_eps);
        let kl = Wiener4::k_l(t_prime, log_eps);

        eprintln!(
            "selected ks={}, kl={}, t'={}, w={}, log_eps={}",
            ks, kl, t_prime, w, log_eps
        );

        let selected = if 2 * ks <= kl {
            Wiener4::small_branch_fused(t_prime, w, ks).unwrap()
        } else {
            Wiener4::large_branch_fused(t_prime, w, kl).unwrap()
        };

        let high_k = 80usize;
        let high_ref = if 2 * ks <= kl {
            Wiener4::small_branch_fused(t_prime, w, high_k).unwrap()
        } else {
            Wiener4::large_branch_fused(t_prime, w, high_k).unwrap()
        };

        eprintln!("selected={:?}", selected);
        eprintln!("high_k={:?}", high_ref);

        assert_relative_eq!(selected.0, high_ref.0, epsilon = 1e-8, max_relative = 1e-8);
        assert_relative_eq!(selected.1, high_ref.1, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(selected.2, high_ref.2, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    fn wiener5_selected_series_matches_high_k_reference_i4_case() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let eps = 1e-12;

        let fused = Wiener5.fused(&obs, &params, eps);

        let core = Wiener5.core(&obs, &params, eps).unwrap();
        let t_prime = core.base.t_prime;
        let w = core.base.w_eff;
        let log_eps = core.base.log_eps_eff;

        let ks = Wiener4::k_s(t_prime, w, log_eps);
        let kl = Wiener4::k_l(t_prime, log_eps);
        let high_k = 100usize;

        let series_selected = if 2 * ks <= kl {
            Wiener4::small_branch_fused(t_prime, w, ks).unwrap()
        } else {
            Wiener4::large_branch_fused(t_prime, w, kl).unwrap()
        };

        let series_high = if 2 * ks <= kl {
            Wiener4::small_branch_fused(t_prime, w, high_k).unwrap()
        } else {
            Wiener4::large_branch_fused(t_prime, w, high_k).unwrap()
        };

        eprintln!("fused={:?}", fused);
        eprintln!("series_selected={:?}", series_selected);
        eprintln!("series_high={:?}", series_high);

        assert_relative_eq!(
            series_selected.0,
            series_high.0,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            fused.log_prob,
            core.base.pref + series_high.0,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }

    #[test]
    fn wiener5_i4_matches_stan_oracle_values() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let eps = 1e-12;

        let fused = Wiener5.fused(&obs, &params, eps);

        // Stan oracle from your C++ output:
        let lp_ref = -50.058_755_379_483_44_f64;
        let gw_ref = -34.586923958070955_f64;

        eprintln!("fused={:?}", fused);

        assert_relative_eq!(fused.log_prob, lp_ref, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(fused.grad.beta, gw_ref, epsilon = 1e-4, max_relative = 1e-4);
    }

    #[test]
    fn wiener5_oracle_route_trace() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let eps = 1e-12;

        let core = Wiener5.core(&obs, &params, eps).unwrap();
        let ks = Wiener4::k_s(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
        let kl = Wiener4::k_l(core.base.t_prime, core.base.log_eps_eff);
        let ks_gw = Wiener4::k_s_grad_w(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
        let kl_gw = Wiener4::k_l_grad_w(core.base.t_prime, core.base.log_eps_eff);

        eprintln!(
            "core={:?}\nks={}, kl={}, ks_gw={}, kl_gw={}",
            core, ks, kl, ks_gw, kl_gw
        );

        let chosen = if 2 * ks <= kl { "small" } else { "large" };
        let chosen_gw = if 2 * ks_gw <= kl_gw { "small" } else { "large" };
        eprintln!(
            "chosen_density_branch={}, chosen_grad_branch={}",
            chosen, chosen_gw
        );

        assert!(core.base.t_prime.is_finite());
        panic!()
    }

    fn print_case(label: &str, obs: &WienerObservation, params: &Wiener5Params, eps: f64) {
        let core = Wiener5.core(obs, params, eps).expect("core");
        let fused = Wiener5.fused(obs, params, eps);

        let pref_log_norm = -2.0 * core.base.a.ln();
        let pref_drift = core.base.a * core.base.v_eff * (1.0 - core.base.w_eff);
        let pref_diffusion = -0.5 * core.base.v_eff * core.base.v_eff * core.base.t;

        let series = fused.log_prob - core.base.pref;

        let dpref_dw_eff = -core.base.a * core.base.v_eff;
        let dlog_dw_eff = fused.grad.beta / core.base.beta_sign;
        let dseries_dw_eff = dlog_dw_eff - dpref_dw_eff;

        let ks = Wiener4::k_s(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
        let kl = Wiener4::k_l(core.base.t_prime, core.base.log_eps_eff);
        let ks_gw = Wiener4::k_s_grad_w(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
        let kl_gw = Wiener4::k_l_grad_w(core.base.t_prime, core.base.log_eps_eff);

        let selected_density_branch = if 2 * ks <= kl { "small" } else { "large" };
        let selected_grad_branch = if 2 * ks_gw <= kl_gw { "small" } else { "large" };

        eprintln!("=== {} ===", label);
        eprintln!("obs = {:?}", obs);
        eprintln!("params = {:?}", params);
        eprintln!("core.base = {:?}", core.base);
        eprintln!("pref_log_norm   = {}", pref_log_norm);
        eprintln!("pref_drift      = {}", pref_drift);
        eprintln!("pref_diffusion  = {}", pref_diffusion);
        eprintln!("core.base.pref  = {}", core.base.pref);
        eprintln!("log_prob        = {}", fused.log_prob);
        eprintln!("series          = {}", series);
        eprintln!("grad            = {:?}", fused.grad);
        eprintln!("dpref_dw_eff    = {}", dpref_dw_eff);
        eprintln!("dlog_dw_eff     = {}", dlog_dw_eff);
        eprintln!("dseries_dw_eff  = {}", dseries_dw_eff);
        eprintln!(
            "ks={}, kl={}, ks_gw={}, kl_gw={}, density_branch={}, grad_branch={}",
            ks, kl, ks_gw, kl_gw, selected_density_branch, selected_grad_branch
        );
    }

    #[test]
    fn wiener5_i4_route_trace() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);

        print_case("i4_upper", &obs, &params, 1e-12);

        let core = Wiener5.core(&obs, &params, 1e-12).unwrap();
        let ks = Wiener4::k_s(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
        let high_k = 100usize;

        let small_high = Wiener4::small_branch_fused(core.base.t_prime, core.base.w_eff, high_k);
        let large_high = Wiener4::large_branch_fused(core.base.t_prime, core.base.w_eff, high_k);

        eprintln!("high_k_small = {:?}", small_high);
        eprintln!("high_k_large = {:?}", large_high);

        assert!(core.base.t_prime.is_finite());
        assert!(ks > 0);
        panic!()
    }

    #[test]
    fn wiener5_w_grid_route_trace() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let a = 10.0;
        let t0 = 0.01;
        let v = -3.0;
        let sv = 0.2;
        let eps = 1e-12;

        // Small grid around the troublesome region.
        for &w in &[0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90] {
            let params = Wiener5Params::with_params_unchecked(a, t0, w, v, sv);
            let fused = Wiener5.fused(&obs, &params, eps);
            let core = Wiener5.core(&obs, &params, eps).unwrap();

            let ks = Wiener4::k_s(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
            let kl = Wiener4::k_l(core.base.t_prime, core.base.log_eps_eff);
            let ks_gw =
                Wiener4::k_s_grad_w(core.base.t_prime, core.base.w_eff, core.base.log_eps_eff);
            let kl_gw = Wiener4::k_l_grad_w(core.base.t_prime, core.base.log_eps_eff);

            eprintln!(
                "w={:.3} lp={:.17} gw={:.17} pref={:.17} t'={:.17} ks={} kl={} ks_gw={} kl_gw={}",
                w,
                fused.log_prob,
                fused.grad.beta,
                core.base.pref,
                core.base.t_prime,
                ks,
                kl,
                ks_gw,
                kl_gw
            );
        }
        panic!()
    }

    #[test]
    fn wiener5_stan_oracle_row_i4_asserts() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.2);
        let fused = Wiener5.fused(&obs, &params, 1e-12);

        let stan_lp = -50.058_755_379_483_44_f64;
        let stan_gw = -34.586923958070955_f64;

        eprintln!("fused = {:?}", fused);
        eprintln!("stan_lp = {}", stan_lp);
        eprintln!("stan_gw = {}", stan_gw);

        assert_relative_eq!(fused.log_prob, stan_lp, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(
            fused.grad.beta,
            stan_gw,
            epsilon = 1e-4,
            max_relative = 1e-4
        );
        panic!()
    }

    #[test]
    fn wiener7_i4_beta_grad_matches_endpoint_identity() {
        let obs = WienerObservation {
            rt: 6.0,
            boundary: Boundary::Upper,
        };

        let params = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);

        let eps = 1e-12;
        let core = Wiener7.core(&obs, &params, eps).unwrap();
        let fused = Wiener7.eval_fused(&core, &obs);

        assert!(core.sw > 0.0);
        assert_eq!(core.st0, 0.0);

        let low = core.beta0 - core.sw / 2.0;
        let high = core.beta0 + core.sw / 2.0;

        let p_low =
            Wiener5Params::with_params_unchecked(core.alpha, core.tau0, low, core.delta, core.sv);
        let p_high =
            Wiener5Params::with_params_unchecked(core.alpha, core.tau0, high, core.delta, core.sv);

        let f_low = Wiener7::wiener5_density(&obs, &p_low, core.eps_series);
        let f_high = Wiener7::wiener5_density(&obs, &p_high, core.eps_series);

        let density = fused.log_prob.exp();
        let endpoint_grad = (f_high - f_low) / (core.sw * density);

        eprintln!("fused.beta = {}", fused.grad.beta);
        eprintln!("endpoint_grad = {}", endpoint_grad);

        assert_relative_eq!(
            fused.grad.beta,
            endpoint_grad,
            epsilon = 1e-7,
            max_relative = 1e-7
        );
    }
}
