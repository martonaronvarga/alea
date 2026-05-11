use crate::density::{FusedLogDensity, GradLogDensity};
use crate::dist::traits::{Family, Parameter, Target};
use crate::error::{ProbError, Result};
use ffi::{hcubature_into, Bounds, ErrorNorm, Options};
use once_cell::sync::Lazy;
use std::f64::consts::{PI, TAU};

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
/// when `delta > 0`); `Lower` ≡ error response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Process hit the upper boundary at `alpha`.
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
/// alpha: boundary separation, alpha ∈ R^+
/// tau:   non-decision time, tau ∈ R^+
/// beta:  relative starting point (upper-boundary parameterisation), beta ∈ (0,1)
///        beta = 1 is at the upper boundary, beta = 0 is at the lower boundary
/// delta: drift rate, delta ∈ R; positive values bias accumulation toward upper boundary
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener4Params {
    /// Boundary separation (> 0). Larger → slower, more accurate decisions.
    pub alpha: f64,
    /// Non-decision time in seconds (> 0). Minimum possible RT.
    pub tau: f64,
    /// Relative starting point, beta ∈ (0,1). beta = 0.5 is unbiased.
    pub beta: f64,
    /// Drift rate. Positive values bias accumulation toward the upper boundary.
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

/// Internal state for the Wiener4 density computation with boundary-reflected parameters.
///
/// The WienR/Gondan series is derived for the upper boundary only.  For a Lower-boundary
/// observation the density is obtained by the symmetry relation:
///
///   f(t, Lower | α, β, δ) = f(t, Upper | α, 1−β, −δ)
///
/// so we reflect `β → 1−β` and `δ → −δ` before evaluating the series.  The field
/// `boundary_sign = +1` (Upper) or `−1` (Lower) carries the chain-rule factor back from
/// the reflected coordinates to the original parameters `β` and `δ` when computing
/// gradients.
#[derive(Copy, Clone, Debug)]
struct Wiener4Core {
    /// Decision time t = rt − tau  (> 0).
    t: f64,
    /// Boundary separation α.
    a: f64,
    /// Reduced time t′ = t / α².
    t_prime: f64,
    /// Reflected starting point:  β (Upper) or 1−β (Lower).
    w_refl: f64,
    /// Reflected drift:  δ (Upper) or −δ (Lower).
    v_refl: f64,
    /// Log prefactor: −2 ln α + α · v_refl · (1 − w_refl) − v_refl² · t / 2.
    pref: f64,
    /// Adjusted log tolerance for series truncation.
    log_eps_eff: f64,
    /// Chain-rule sign from boundary reflection: +1 (Upper) / −1 (Lower).
    boundary_sign: f64,
}

/// s_delta = standard deviation in drift rate, s_delta ∈ R^≥0
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

#[derive(Copy, Clone, Debug)]
struct Wiener5Core {
    base: Wiener4Core,
    sv: f64,  // = params.s_delta
    sv2: f64, // = sv^2 (cached)
    lam: f64, // = 1 + sv^2 * t (denominator in marginalised pref)
}

/// s_beta: standard deviation of the starting-point prior (beta)
/// s_tau:  standard deviation of the non-decision-time prior (tau)
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

/// Outcome of a single call to `eval_fused`: the log-series value and its
/// partial derivatives w.r.t. the reduced time t′ and reflected starting point w_refl.
#[derive(Copy, Clone, Debug)]
struct SeriesEval {
    log_series: f64,
    dlog_dtprime: f64,
    dlog_dw: f64,
}

// ─── Wiener4Params constructors ────────────────────────────────────────────────

impl Wiener4Params {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validated constructor. Returns `Err` if any constraint is violated.
    #[inline]
    pub fn with_params(alpha: f64, tau: f64, beta: f64, delta: f64) -> Result<Self> {
        if alpha.is_finite()
            && tau.is_finite()
            && beta.is_finite()
            && delta.is_finite()
            && alpha > 0.0
            && tau >= 0.0
            && beta > 0.0
            && beta < 1.0
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

    #[inline]
    pub fn with_params_unchecked(alpha: f64, tau: f64, beta: f64, delta: f64) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
        }
    }

    /// Decision time for a given raw RT.  Returns `None` if `rt ≤ tau`.
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
            && self.beta > 0.0
            && self.beta < 1.0
    }
}

// ─── Wiener5Params constructors ────────────────────────────────────────────────

impl Wiener5Params {
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

// ─── Wiener7Params constructors ────────────────────────────────────────────────

impl Wiener7Params {
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
        if (0.0..1.0).contains(&s_beta) && s_tau.is_finite() && s_tau >= 0.0 {
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

// ─── Wiener4 series machinery ──────────────────────────────────────────────────

impl Wiener4 {
    /// Large-time (π-series) truncation count.
    ///
    /// t_prime = t / α²  (reduced time).
    #[inline]
    pub fn k_l(t_prime: f64, log_eps: f64) -> usize {
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

    /// Small-time (Gaussian) truncation count.
    ///
    /// `w` is the reflected starting point (`w_refl` in [`Wiener4Core`]):
    /// `beta` for the Upper boundary, `1 − beta` for the Lower boundary.
    #[inline]
    pub fn k_s(t_prime: f64, w: f64, log_eps: f64) -> usize {
        if !(t_prime.is_finite() && w.is_finite() && log_eps.is_finite()) || t_prime <= 0.0 {
            return 0;
        }
        // `upper_dist` = distance from the upper boundary in reflected coordinates = 1 − w_refl.
        let upper_dist = 1.0 - w;

        // u_eps = min(−1, ln(2π t′² ε²))
        let u_eps = (TAU.ln() + 2.0 * t_prime.ln() + 2.0 * log_eps).min(-1.0);
        let term1 = 0.5 * ((2.0 * t_prime).sqrt() - upper_dist);

        let term2 = {
            let inner = -2.0 * u_eps - 2.0;
            if inner > 0.0 {
                let arg = -t_prime * (u_eps - inner.sqrt());
                if arg > 0.0 {
                    0.5 * (arg.sqrt() - upper_dist)
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                f64::NEG_INFINITY
            }
        };

        term1.max(term2).ceil().max(0.0) as usize
    }

    /// Log of the small-time Gaussian series S_s(t′, w_refl).
    ///
    /// Uses log-domain summation with the dominant exponent factored out for numerical
    /// stability.  The series is scaled by `exp(−(1−w)²/(2t′))` before summing.
    ///
    /// `w` = `w_refl` (reflected starting point, see [`Wiener4Core`]).
    #[inline]
    pub fn small_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 || w <= 0.0 || w >= 1.0 {
            return None;
        }

        let upper_dist = 1.0 - w; // = 1 − w_refl
        let inv_two_t = 0.5 / t_prime;
        let max_exp = upper_dist * upper_dist * inv_two_t; // dominant exponent (j=0)

        let mut sum = upper_dist; // j = 0 term (scaled by exp(−max_exp) cancels)

        for j in 1..=k {
            let jf = j as f64;
            let xp = upper_dist + 2.0 * jf;
            let xm = 2.0 * jf - upper_dist;
            // Subtract max_exp algebraically to prevent underflow
            sum += xp * (-(xp * xp * inv_two_t - max_exp)).exp();
            sum -= xm * (-(xm * xm * inv_two_t - max_exp)).exp();
        }

        if sum > 0.0 {
            let log_pref = -0.5 * TAU.ln() - 1.5 * t_prime.ln();
            Some(log_pref - max_exp + sum.ln())
        } else {
            None
        }
    }

    /// Scaled raw small-time sum R_s(t′, w) = exp((1−w)²/(2t′)) · S_s(t′, w).
    ///
    /// Used as an intermediate for gradient computation; the pre-factor derivative
    /// must be included separately via the chain rule.
    #[inline]
    pub fn small_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }
        let upper_dist = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;
        let max_exp = upper_dist * upper_dist * inv_two_t;

        let mut sum = upper_dist;

        for j in 1..=k {
            let jf = j as f64;
            let xp = upper_dist + 2.0 * jf;
            let xm = 2.0 * jf - upper_dist;
            sum += xp * (-(xp * xp * inv_two_t - max_exp)).exp();
            sum -= xm * (-(xm * xm * inv_two_t - max_exp)).exp();
        }

        Some(sum)
    }

    /// d/dt′ of the scaled raw small-time sum R_s(t′, w).
    #[inline]
    pub fn small_time_log_dseries_dt(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }
        let upper_dist = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;
        let max_exp = upper_dist * upper_dist * inv_two_t;
        let tt = t_prime * t_prime;

        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let xp = upper_dist + 2.0 * jf;
            let xm = 2.0 * jf - upper_dist;
            let xp2 = xp * xp;
            let xm2 = xm * xm;
            let exp_p = (-(xp2 * inv_two_t - max_exp)).exp();
            let exp_m = (-(xm2 * inv_two_t - max_exp)).exp();
            sum += 0.5 * xp * (xp2 - upper_dist * upper_dist) * exp_p / tt;
            sum -= 0.5 * xm * (xm2 - upper_dist * upper_dist) * exp_m / tt;
        }

        Some(sum)
    }

    /// d/dw of the scaled raw small-time sum R_s(t′, w).
    #[inline]
    pub fn small_time_log_dseries_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }
        let upper_dist = 1.0 - w;
        let inv_two_t = 0.5 / t_prime;
        let max_exp = upper_dist * upper_dist * inv_two_t;

        let mut sum = -1.0; // d/dw of the j=0 term: d(1−w)/dw = −1

        for j in 1..=k {
            let jf = j as f64;
            let xp = upper_dist + 2.0 * jf;
            let xm = 2.0 * jf - upper_dist;
            let xp2 = xp * xp;
            let xm2 = xm * xm;
            let exp_p = (-(xp2 * inv_two_t - max_exp)).exp();
            let exp_m = (-(xm2 * inv_two_t - max_exp)).exp();
            sum += exp_p * (-1.0 + 2.0 * jf * xp / t_prime);
            sum += exp_m * (-1.0 + 2.0 * jf * xm / t_prime);
        }

        Some(sum)
    }

    /// Log of the large-time π-series S_l(t′, w).
    ///
    /// Uses log-domain summation with the dominant exponent (j=1) factored out.
    ///
    /// `w` = `w_refl` (reflected starting point, see [`Wiener4Core`]).
    #[inline]
    pub fn large_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2_t_half = (PI * PI * t_prime) * 0.5;
        let max_exp = pi2_t_half; // dominant term exponent (j=1)

        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let s = (jf * PI * (1.0 - w)).sin();
            if s == 0.0 {
                continue;
            }
            let arg = (jf * jf - 1.0) * max_exp;
            sum += jf * s * (-arg).exp();
        }

        if sum > 0.0 {
            Some(PI.ln() - max_exp + sum.ln())
        } else {
            None
        }
    }

    /// Scaled raw large-time sum R_l(t′, w) = exp(π²t′/2) · S_l(t′, w).
    #[inline]
    pub fn large_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2_half = 0.5 * PI * PI;
        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let s = (jf * PI * (1.0 - w)).sin();
            if s == 0.0 {
                continue;
            }
            sum += jf * s * (-jf * jf * pi2_half * t_prime).exp();
        }

        Some(sum)
    }

    /// d/dt′ of the scaled raw large-time sum R_l(t′, w).
    #[inline]
    pub fn large_time_log_dseries_dt(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2 = PI * PI;
        let pi2_half = 0.5 * pi2;
        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let s = (jf * PI * (1.0 - w)).sin();
            if s == 0.0 {
                continue;
            }
            sum -= 0.5 * pi2 * jf * jf * jf * s * (-jf * jf * pi2_half * t_prime).exp();
        }

        Some(sum)
    }

    /// d/dw of the scaled raw large-time sum R_l(t′, w).
    #[inline]
    pub fn large_time_log_dseries_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
        if !(t_prime.is_finite() && w.is_finite()) || t_prime <= 0.0 {
            return None;
        }

        let pi2_half = 0.5 * PI * PI;
        let mut sum = 0.0;

        for j in 1..=k {
            let jf = j as f64;
            let c = (jf * PI * (1.0 - w)).cos();
            sum -= jf * jf * PI * c * (-jf * jf * pi2_half * t_prime).exp();
        }

        Some(sum)
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /// Build the reflected-coordinate core for a single observation.
    ///
    /// Applies the boundary-symmetry reflection (see [`Wiener4Core`]) and computes
    /// the log prefactor.  Returns `None` for any invalid or degenerate input.
    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Option<Wiener4Core> {
        if !params.valid() || !obs.rt.is_finite() || !eps.is_finite() || eps < 0.0 {
            return None;
        }
        let t = obs.rt - params.tau;
        if t <= 0.0 {
            return None;
        }

        let a = params.alpha;
        let a2 = a * a;
        let t_prime = t / a2;

        // Reflect parameters for the Lower boundary so the series always evaluates
        // at an equivalent Upper-boundary problem.
        let (v_refl, w_refl, boundary_sign) = match obs.boundary {
            Boundary::Upper => (params.delta, params.beta, 1.0_f64),
            Boundary::Lower => (-params.delta, 1.0 - params.beta, -1.0_f64),
        };

        // Canonical log prefactor (WienR / Gondan 2014):
        //   log P = −2 ln α + α · v · (1−w) − ½ v² t + log S(t′, w)
        // where (v, w) are the reflected parameters (v_refl, w_refl).
        let pref = -2.0 * a.ln() + a * v_refl * (1.0 - w_refl) - 0.5 * v_refl * v_refl * t;

        let log_eps_eff = (eps.ln() - pref).min(-10.0);

        Some(Wiener4Core {
            t,
            a,
            t_prime,
            w_refl,
            v_refl,
            pref,
            log_eps_eff,
            boundary_sign,
        })
    }

    #[inline]
    fn series_counts(core: &Wiener4Core) -> (usize, usize) {
        let ks = Self::k_s(core.t_prime, core.w_refl, core.log_eps_eff);
        let kl = Self::k_l(core.t_prime, core.log_eps_eff);
        (ks, kl)
    }

    /// Select and evaluate the series with fewer terms.
    #[inline]
    fn eval_series(core: &Wiener4Core) -> Option<f64> {
        let (ks, kl) = Self::series_counts(core);
        if ks < kl {
            Self::small_time_log_series(core.t_prime, core.w_refl, ks)
        } else {
            Self::large_time_log_series(core.t_prime, core.w_refl, kl)
        }
    }

    /// Jointly evaluate the log-series and its partial derivatives w.r.t. t′ and w_refl.
    #[inline]
    fn eval_fused(core: &Wiener4Core) -> Option<SeriesEval> {
        let (ks, kl) = Self::series_counts(core);

        if ks < kl {
            let log_series = Self::small_time_log_series(core.t_prime, core.w_refl, ks)?;
            let raw = Self::small_time_series_raw(core.t_prime, core.w_refl, ks)?;
            if raw <= 0.0 {
                return None;
            }

            let d_raw_dt = Self::small_time_log_dseries_dt(core.t_prime, core.w_refl, ks)?;
            let d_raw_dw = Self::small_time_log_dseries_dw(core.t_prime, core.w_refl, ks)?;
            let upper_dist = 1.0 - core.w_refl;

            // d log S_s / d t′ = −3/(2t′) + (1−w)²/(2t′²) + (d R_s/dt′) / R_s
            // d log S_s / dw   =  (1−w)/t′ + (d R_s/dw) / R_s
            Some(SeriesEval {
                log_series,
                dlog_dtprime: -1.5 / core.t_prime
                    + upper_dist * upper_dist / (2.0 * core.t_prime * core.t_prime)
                    + d_raw_dt / raw,
                dlog_dw: upper_dist / core.t_prime + d_raw_dw / raw,
            })
        } else {
            let log_series = Self::large_time_log_series(core.t_prime, core.w_refl, kl)?;
            let raw = Self::large_time_series_raw(core.t_prime, core.w_refl, kl)?;
            if raw <= 0.0 {
                return None;
            }

            let d_raw_dt = Self::large_time_log_dseries_dt(core.t_prime, core.w_refl, kl)?;
            let d_raw_dw = Self::large_time_log_dseries_dw(core.t_prime, core.w_refl, kl)?;

            Some(SeriesEval {
                log_series,
                dlog_dtprime: d_raw_dt / raw,
                dlog_dw: d_raw_dw / raw,
            })
        }
    }

    // ─── Public density interface ─────────────────────────────────────────────

    /// Log first-passage density for the 4-parameter Wiener model.
    ///
    /// Returns `f64::NEG_INFINITY` on invalid inputs or numerically degenerate
    /// series.  Gradient fields are `NaN` (not computed; use `fused` for gradients).
    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        eps: f64,
    ) -> Wiener4Eval {
        // Delegate to fused; discard gradient fields.
        self.fused(obs, params, eps)
    }

    /// Jointly compute the log-density and its gradient w.r.t. all four parameters.
    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener4Params, eps: f64) -> Wiener4Eval {
        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad::default(),
                }
            }
        };
        let series = match Self::eval_fused(&core) {
            Some(s) => s,
            None => {
                return Wiener4Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener4Grad::default(),
                }
            }
        };

        let log_prob = core.pref + series.log_series;

        let a = core.a;
        let a2 = a * a;
        let t = core.t;

        // Partial derivatives of the log prefactor w.r.t. reflected variables.
        let d_pref_da = -2.0 / a + core.v_refl * (1.0 - core.w_refl);
        let d_pref_dt = -0.5 * core.v_refl * core.v_refl;
        let d_pref_dw = -a * core.v_refl;
        let d_pref_dv = a * (1.0 - core.w_refl) - core.v_refl * t;

        // Chain rules.  Note t_prime = t / a², so:
        //   ∂/∂α = d_pref_da + dlog_dtprime · (−2t / a³)
        //   ∂/∂τ = −(d_pref_dt + dlog_dtprime / a²)   [sign from ∂t/∂τ = −1]
        //   ∂/∂β = boundary_sign · (d_pref_dw + dlog_dw)
        //   ∂/∂δ = boundary_sign · d_pref_dv
        let grad = Wiener4Grad {
            alpha: d_pref_da + series.dlog_dtprime * (-2.0 * t / (a2 * a)),
            tau: -(d_pref_dt + series.dlog_dtprime / a2),
            beta: core.boundary_sign * (d_pref_dw + series.dlog_dw),
            delta: core.boundary_sign * d_pref_dv,
        };

        Wiener4Eval { log_prob, grad }
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

impl FusedLogDensity for Target<Wiener4, Vec<WienerObservation>> {
    fn log_prob_and_grad<'a>(&'a self, p: &'a Wiener4Params, grad: &mut [f64; 4]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for obs in self.data.iter() {
            let fused = Wiener4.fused(obs, p, 1e-12);
            total_lp += fused.log_prob;
            grad[0] += fused.grad.alpha;
            grad[1] += fused.grad.tau;
            grad[2] += fused.grad.beta;
            grad[3] += fused.grad.delta;
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

// ─── Wiener5 ──────────────────────────────────────────────────────────────────

impl Wiener5 {
    /// Build the Wiener5 core by marginalising over the drift-rate distribution.
    ///
    /// When s_delta > 0 the marginal log prefactor becomes (Gondan 2014, eq. 10):
    ///   pref5 = −2 ln α − ½ ln λ + N / (2λ)
    /// where
    ///   λ = 1 + s_delta² · t
    ///   N = −v² t + 2α v (1−w) + α² (1−w)² s_delta²
    /// and (v, w) = (v_refl, w_refl) are the boundary-reflected parameters.
    #[inline]
    fn core(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Option<Wiener5Core> {
        if !params.base.valid() || params.s_delta < 0.0 || !obs.rt.is_finite() || eps <= 0.0 {
            return None;
        }

        let t = obs.rt - params.base.tau;
        if t <= 0.0 {
            return None;
        }

        let a = params.base.alpha;
        let a2 = a * a;
        let t_prime = t / a2;

        let (v_refl, w_refl, boundary_sign) = match obs.boundary {
            Boundary::Upper => (params.base.delta, params.base.beta, 1.0_f64),
            Boundary::Lower => (-params.base.delta, 1.0 - params.base.beta, -1.0_f64),
        };

        let sv = params.s_delta;
        let sv2 = sv * sv;
        let lam = 1.0 + sv2 * t;
        let upper_dist = 1.0 - w_refl; // = 1 − w_refl

        // Marginalised log prefactor (Wiener5, eq. 10 of Gondan 2014).
        let n = -v_refl * v_refl * t
            + 2.0 * a * v_refl * upper_dist
            + a2 * upper_dist * upper_dist * sv2;
        let pref = -2.0 * a.ln() - 0.5 * lam.ln() + n / (2.0 * lam);

        let log_eps_eff = (eps.ln() - pref).min(-10.0);

        Some(Wiener5Core {
            base: Wiener4Core {
                t,
                a,
                t_prime,
                w_refl,
                v_refl,
                pref,
                log_eps_eff,
                boundary_sign,
            },
            sv,
            sv2,
            lam,
        })
    }

    /// Log first-passage density for the 5-parameter Wiener model.
    ///
    /// Gradient fields are `NaN`; use `fused` to obtain gradients.
    #[inline]
    pub fn log_prob(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        eps: f64,
    ) -> Wiener5Eval {
        // Delegate to fused; discard gradient fields.
        self.fused(obs, params, eps)
    }

    /// Jointly compute the log-density and its gradient w.r.t. all five parameters.
    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> Wiener5Eval {
        let core = match self.core(obs, params, eps) {
            Some(c) => c,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad::default(),
                }
            }
        };
        let series = match Wiener4::eval_fused(&core.base) {
            Some(s) => s,
            None => {
                return Wiener5Eval {
                    log_prob: f64::NEG_INFINITY,
                    grad: Wiener5Grad::default(),
                }
            }
        };

        let b = &core.base;
        let a = b.a;
        let a2 = a * a;
        let t = b.t;
        let w = b.w_refl;
        let v = b.v_refl;
        let sv = core.sv;
        let sv2 = core.sv2;
        let lam = core.lam;
        let lam2 = lam * lam;
        let upper_dist = 1.0 - w; // = 1 − w_refl

        // N = −v² t + 2α v (1−w) + α² (1−w)² sv²   (reuse; already computed in core)
        let n = -v * v * t + 2.0 * a * v * upper_dist + a2 * upper_dist * upper_dist * sv2;

        // Partial derivatives of N.
        let dn_da = 2.0 * upper_dist * (v + a * upper_dist * sv2);
        let dn_dv = -2.0 * v * t + 2.0 * a * upper_dist;
        let dn_dw = -2.0 * a * v - 2.0 * a2 * upper_dist * sv2;
        let dn_dt = -v * v;
        let dn_dsv = 2.0 * a2 * upper_dist * upper_dist * sv;

        // Partial derivatives of pref5 = −2 ln α − ½ ln λ + N/(2λ).
        let dpref_da = -2.0 / a + dn_da / (2.0 * lam);
        let dpref_dt = -0.5 * sv2 / lam + (dn_dt * lam - n * sv2) / (2.0 * lam2);
        let dpref_dw = dn_dw / (2.0 * lam);
        let dpref_dv = dn_dv / (2.0 * lam);
        let dpref_dsv = -sv * t / lam + (dn_dsv * lam - n * 2.0 * sv * t) / (2.0 * lam2);

        let log_prob = b.pref + series.log_series;

        let grad = Wiener5Grad {
            alpha: dpref_da + series.dlog_dtprime * (-2.0 * t / (a * a2)),
            tau: -(dpref_dt + series.dlog_dtprime / a2),
            beta: b.boundary_sign * (dpref_dw + series.dlog_dw),
            delta: b.boundary_sign * dpref_dv,
            s_delta: dpref_dsv,
        };

        Wiener5Eval { log_prob, grad }
    }
}

impl Family for Wiener5 {
    type Params = Wiener5Params;
    type Data = Vec<WienerObservation>;

    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        let eps = 1e-12;
        data.iter()
            .map(|obs| Wiener5.log_prob(obs, params, eps).log_prob)
            .sum()
    }
}

impl FusedLogDensity for Target<Wiener5, Vec<WienerObservation>> {
    fn log_prob_and_grad(&self, p: &Wiener5Params, grad: &mut [f64; 5]) -> f64 {
        let mut total_lp = 0.0;
        grad.fill(0.0);

        for obs in self.data.iter() {
            let fused = Wiener5.fused(obs, p, 1e-6);
            total_lp += fused.log_prob;
            grad[0] += fused.grad.alpha;
            grad[1] += fused.grad.tau;
            grad[2] += fused.grad.beta;
            grad[3] += fused.grad.delta;
            grad[4] += fused.grad.s_delta;
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

// ─── Wiener7 ──────────────────────────────────────────────────────────────────

impl Wiener7 {
    /// Build the integration core.  Returns `None` for invalid or out-of-domain inputs.
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

        // The beta-prior support [beta0 − sw/2, beta0 + sw/2] must lie strictly inside (0,1).
        if sw > 0.0 && (beta0 - sw / 2.0 <= 0.0 || beta0 + sw / 2.0 >= 1.0) {
            return None;
        }
        // At least one non-decision-time sample must give t = rt − tau > 0.
        if st0 > 0.0 && (obs.rt - tau0) / st0 <= 0.0 {
            return None;
        }

        let dim = usize::from(sw != 0.0) + usize::from(st0 != 0.0);
        let mut xmin = [0.0; 2];
        let mut xmax = [1.0; 2];

        if st0 != 0.0 {
            // Clip the upper limit so every tau in [tau0, tau0 + st0 · u] gives rt − tau > 0.
            xmax[dim - 1] = f64::min(1.0, (obs.rt - tau0) / st0);
        }

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
            eps_series: 1e-12,
            opts: Options {
                max_eval: 6000,
                req_abs_error: 0.0,
                req_rel_error: 0.9 * eps, // Stan's choice
                norm: ErrorNorm::L2,
            },
        })
    }

    // ─── Public density interface ────────────────────────────────────────────

    /// Log first-passage density for the 7-parameter Wiener model (no gradients).
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
                    grad: Wiener7Grad::default(),
                }
            }
        };
        let density = self.eval_density(&core, obs);
        Wiener7Eval {
            log_prob: if density > 0.0 {
                density.ln()
            } else {
                f64::NEG_INFINITY
            },
            grad: Wiener7Grad::default(),
        }
    }

    /// Jointly compute the log-density and gradient for the 7-parameter model.
    ///
    /// When `s_beta == 0` and `s_tau == 0` the computation degenerates to Wiener5.
    #[inline]
    pub fn fused(&self, obs: &WienerObservation, params: &Wiener7Params, eps: f64) -> Wiener7Eval {
        // Shortcut: no inter-trial variability in beta or tau → Wiener5.
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
                    grad: Wiener7Grad::default(),
                }
            }
        };
        self.eval_fused(&core, obs)
    }

    // ─── Integration helpers ─────────────────────────────────────────────────

    /// Map a unit-hypercube point `x` to physical (tau, beta) by linear scaling.
    ///
    /// Integration dimension layout:
    ///   - dim == 1, sw != 0 → x[0] maps beta; tau = tau0
    ///   - dim == 1, st0 != 0 → x[0] maps tau; beta = beta0
    ///   - dim == 2            → x[0] maps beta, x[1] maps tau
    #[inline(always)]
    fn map_point(core: &Wiener7Core, x: &[f64]) -> (f64, f64) {
        if core.dim == 1 {
            if core.sw != 0.0 {
                (core.tau0, core.beta0 + core.sw * (x[0] - 0.5))
            } else {
                (core.tau0 + core.st0 * x[0], core.beta0)
            }
        } else {
            (
                core.tau0 + core.st0 * x[1],
                core.beta0 + core.sw * (x[0] - 0.5),
            )
        }
    }

    /// Wiener5 density (not log-density); returns 0.0 if the log-density is non-finite.
    #[inline]
    fn wiener5_density(obs: &WienerObservation, params: &Wiener5Params, eps: f64) -> f64 {
        let lp = Wiener5.log_prob(obs, params, eps).log_prob;
        if lp.is_finite() {
            lp.exp()
        } else {
            0.0
        }
    }

    /// Integrate the density over the (beta, tau) hypercube without computing gradients.
    fn eval_density(&self, core: &Wiener7Core, obs: &WienerObservation) -> f64 {
        let bounds = Bounds::new(&core.xmin, &core.xmax);
        let mut val = [0.0f64; 1];
        let mut err = [0.0f64; 1];

        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (tau, beta) = Self::map_point(core, x);
            if beta <= 0.0 || beta >= 1.0 {
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

    /// Jointly integrate the density and the five inner-parameter gradient components.
    /// Then compute the s_beta and s_tau derivatives separately.
    fn eval_fused(&self, core: &Wiener7Core, obs: &WienerObservation) -> Wiener7Eval {
        let bounds = Bounds::new(&core.xmin, &core.xmax);
        // val[0] = ∫ p5 dx,  val[1..5] = ∫ p5 · ∇_{α,τ,β,δ,sv} log p5 dx
        let mut val = [0.0_f64; 6];
        let mut err = [0.0_f64; 6];

        let integrand = |x: &[f64], fv: &mut [f64]| -> i32 {
            let (tau, beta) = Self::map_point(core, x);
            if beta <= 0.0 || beta >= 1.0 {
                fv.fill(0.0);
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
                fv.fill(0.0);
            }
            0
        };

        if hcubature_into(6, bounds, core.opts, &mut val, &mut err, integrand).is_err() {
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
        let log_density = total_density.ln();

        let mut grad = [0.0_f64; 7];
        grad[0] = val[1] / total_density; // alpha
        grad[1] = val[2] / total_density; // tau

        // ── Gradient of beta0 ────────────────────────────────────────────────
        // Use Leibniz rule: d/dβ0 ∫ p5 dτ dβ / (sw) = (p5(high) − p5(low)) / sw.
        // The factor 1/sw appears from the uniform prior density.
        grad[2] = if core.sw == 0.0 {
            // No beta variability: pointwise derivative still works
            val[3] / total_density
        } else if core.st0 == 0.0 {
            let low = core.beta0 - core.sw / 2.0;
            let high = core.beta0 + core.sw / 2.0;
            let f_low = Self::wiener5_density(
                obs,
                &Wiener5Params::with_params_unchecked(
                    core.alpha,
                    core.tau0,
                    low.clamp(0.0, 1.0),
                    core.delta,
                    core.sv,
                ),
                core.eps_series,
            );
            let f_high = Self::wiener5_density(
                obs,
                &Wiener5Params::with_params_unchecked(
                    core.alpha,
                    core.tau0,
                    high.clamp(0.0, 1.0),
                    core.delta,
                    core.sv,
                ),
                core.eps_series,
            );
            (f_high - f_low) / (core.sw * total_density)
        } else {
            let tau_max_idx = if core.sw != 0.0 && core.st0 != 0.0 {
                1
            } else {
                0
            };
            let tau_max = [core.xmax[tau_max_idx]];
            if tau_max[0] <= 0.0 {
                0.0
            } else {
                let low = core.beta0 - core.sw / 2.0;
                let high = core.beta0 + core.sw / 2.0;
                let bounds_tau = Bounds::new(&[0.0], &tau_max);
                let mut val_tau = [0.0_f64; 2];
                let mut err_tau = [0.0_f64; 2];
                let integrand_tau = |x: &[f64], fv: &mut [f64]| -> i32 {
                    let tau = core.tau0 + core.st0 * x[0];
                    let f_low_tau = Self::wiener5_density(
                        obs,
                        &Wiener5Params::with_params_unchecked(
                            core.alpha,
                            tau,
                            low.clamp(0.0, 1.0),
                            core.delta,
                            core.sv,
                        ),
                        core.eps_series,
                    );
                    let f_high_tau = Self::wiener5_density(
                        obs,
                        &Wiener5Params::with_params_unchecked(
                            core.alpha,
                            tau,
                            high.clamp(0.0, 1.0),
                            core.delta,
                            core.sv,
                        ),
                        core.eps_series,
                    );
                    fv[0] = f_low_tau;
                    fv[1] = f_high_tau;
                    0
                };
                if hcubature_into(
                    2,
                    bounds_tau,
                    core.opts,
                    &mut val_tau,
                    &mut err_tau,
                    integrand_tau,
                )
                .is_err()
                {
                    return Wiener7Eval {
                        log_prob: f64::NEG_INFINITY,
                        grad: Wiener7Grad::default(),
                    };
                }
                (val_tau[1] - val_tau[0]) / (core.sw * total_density)
            }
        };

        grad[3] = val[4] / total_density; // delta
        grad[4] = val[5] / total_density; // s_delta

        // ── Gradient of s_beta ──────────────────────────────────────────────
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
                let mut val_sw = [0.0_f64; 1];
                let mut err_sw = [0.0_f64; 1];
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
                        grad: Wiener7Grad::default(),
                    };
                }
                grad[5] = val_sw[0] / total_density - 1.0 / core.sw;
            }
        }

        // ── Gradient of s_tau ───────────────────────────────────────────────
        if core.st0 == 0.0 {
            grad[6] = 0.0;
        } else {
            let tau_upper = core.tau0 + core.st0;
            if obs.rt - tau_upper <= 0.0 {
                grad[6] = -1.0 / core.st0;
            } else {
                let f_end: f64 = if core.sw == 0.0 {
                    let p5 = Wiener5Params::with_params_unchecked(
                        core.alpha, tau_upper, core.beta0, core.delta, core.sv,
                    );
                    Self::wiener5_density(obs, &p5, core.eps_series)
                } else {
                    let mut val_f = [0.0_f64; 1];
                    let mut err_f = [0.0_f64; 1];
                    let bounds_w = Bounds::new(&[0.0], &[1.0]);
                    let integrand_f = |x: &[f64], fv: &mut [f64]| -> i32 {
                        let beta = core.beta0 + core.sw * (x[0] - 0.5);
                        if beta <= 0.0 || beta >= 1.0 {
                            fv[0] = 0.0;
                            return 0;
                        }
                        let p5 = Wiener5Params::with_params_unchecked(
                            core.alpha, tau_upper, beta, core.delta, core.sv,
                        );
                        fv[0] = Self::wiener5_density(obs, &p5, core.eps_series);
                        0
                    };
                    if hcubature_into(1, bounds_w, core.opts, &mut val_f, &mut err_f, integrand_f)
                        .is_err()
                    {
                        return Wiener7Eval {
                            log_prob: f64::NEG_INFINITY,
                            grad: Wiener7Grad::default(),
                        };
                    }
                    val_f[0]
                };
                grad[6] = -1.0 / core.st0 + f_end / (core.st0 * total_density);
            }
        }

        Wiener7Eval {
            log_prob: log_density,
            grad: Wiener7Grad::from_array(grad),
        }
    }

    // ─── Gauss-Legendre utilities ─────────────────────────────────────────────

    /// Gauss-Legendre nodes and weights on [0,1].
    pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let m = n.div_ceil(2);
        for i in 1..=m {
            let mut x = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..20 {
                let (p1, p2) = Self::legendre_poly(n as u32, x);
                dp = (n as f64) * (x * p1 - p2) / (x * x - 1.0);
                x -= p1 / dp;
                if x.abs() < 1e-15 {
                    break;
                }
            }
            let t = 0.5 * (x + 1.0);
            let w = 1.0 / ((1.0 - x * x) * dp * dp);
            nodes[i - 1] = t;
            weights[i - 1] = w;
            nodes[n - i] = 1.0 - t;
            weights[n - i] = w;
        }
        (nodes, weights)
    }

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
        (p, p0)
    }
}

// ─── Family / FusedLogDensity / GradLogDensity for Wiener7 ───────────────────

impl Family for Wiener7 {
    type Params = Wiener7Params;
    type Data = Vec<WienerObservation>;

    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64 {
        let precision = 1e-4;
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

// ─── Parameter space transformations ─────────────────────────────────────────

impl Parameter for Wiener5Params {
    type Constrained = Self;
    type Unconstrained = [f64; 5];

    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.base.alpha.ln(),
            c.base.tau.ln(),
            crate::numeric::logit(c.base.beta),
            c.base.delta,
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
            c.base.base.alpha.ln(),
            c.base.base.tau.ln(),
            crate::numeric::logit(c.base.base.beta),
            c.base.base.delta,
            c.base.s_delta.ln(),
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
        u[0]  // alpha
            + u[1]  // tau
            + crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2])  // beta
            + u[4]  // s_delta
            + crate::numeric::log_sigmoid(u[5]) + crate::numeric::log1m_sigmoid(u[5])  // s_beta
            + u[6] // s_tau
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        let h = 1e-8;
        let analytic = Wiener4.fused(&obs, &params, 1e-8).grad;

        let lp = |p: &Wiener4Params| Wiener4.log_prob(&obs, p, 1e-8).log_prob;

        macro_rules! fd {
            ($field:ident) => {{
                let mut p1 = params;
                let mut p2 = params;
                p1.$field += h;
                p2.$field -= h;
                (lp(&p1) - lp(&p2)) / (2.0 * h)
            }};
        }

        assert_relative_eq!(
            analytic.alpha,
            fd!(alpha),
            epsilon = 1e-4,
            max_relative = 1e-3
        );
        assert_relative_eq!(analytic.tau, fd!(tau), epsilon = 1e-4, max_relative = 1e-3);
        assert_relative_eq!(
            analytic.beta,
            fd!(beta),
            epsilon = 1e-4,
            max_relative = 1e-3
        );
        assert_relative_eq!(
            analytic.delta,
            fd!(delta),
            epsilon = 1e-4,
            max_relative = 1e-3
        );
    }

    fn stan_wiener5_data() -> (f64, WienerObservation, Wiener5Params, f64) {
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let params = Wiener5Params::with_params_unchecked(1.5, 0.3, 0.55, 0.4, 0.1);
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
        assert_relative_eq!(eval.log_prob, expected_lp, epsilon = 1e-8);
    }

    #[test]
    fn wiener5_gradient_matches_stan_numerical() {
        let (eps, obs, params, _) = stan_wiener5_data();
        let eval = Wiener5.fused(&obs, &params, eps);
        let lp_only = Wiener5.log_prob(&obs, &params, eps).log_prob;
        assert_relative_eq!(eval.log_prob, lp_only, epsilon = 1e-12);

        let h = 1e-6;
        let analytic = eval.grad;
        let lp = |p: &Wiener5Params| Wiener5.log_prob(&obs, p, eps).log_prob;

        macro_rules! fd5 {
            ($field:ident, $sub:ident) => {{
                let mut p1 = params;
                let mut p2 = params;
                p1.$sub.$field += h;
                p2.$sub.$field -= h;
                (lp(&p1) - lp(&p2)) / (2.0 * h)
            }};
            ($field:ident) => {{
                let mut p1 = params;
                let mut p2 = params;
                p1.$field += h;
                p2.$field -= h;
                (lp(&p1) - lp(&p2)) / (2.0 * h)
            }};
        }

        assert_relative_eq!(analytic.alpha, fd5!(alpha, base), epsilon = 1e-4);
        assert_relative_eq!(analytic.tau, fd5!(tau, base), epsilon = 1e-4);
        assert_relative_eq!(analytic.beta, fd5!(beta, base), epsilon = 1e-4);
        assert_relative_eq!(analytic.delta, fd5!(delta, base), epsilon = 1e-4);
        assert_relative_eq!(analytic.s_delta, fd5!(s_delta), epsilon = 1e-4);
    }

    #[test]
    fn wiener7_matches_stan() {
        let (eps, obs, params, expected_lp, expected_grad) = stan_wiener7_data();
        let eval = Wiener7.fused(&obs, &params, eps);
        assert_relative_eq!(eval.log_prob, expected_lp, epsilon = 1e-6);
        for (i, g) in eval.grad.to_array().iter().enumerate() {
            assert_relative_eq!(*g, expected_grad[i], epsilon = 1e-4, max_relative = 1e-3);
        }
    }

    #[test]
    fn wiener7_finite_difference_consistency() {
        let (eps, obs, params, _, _) = stan_wiener7_data();
        let eval = Wiener7.fused(&obs, &params, eps);
        let analytic = eval.grad.to_array();
        let h = 1e-5;
        let f = |p: &Wiener7Params| Wiener7.log_prob(&obs, p, eps).log_prob;
        let param_names = [
            "alpha", "tau", "beta", "delta", "s_delta", "s_beta", "s_tau",
        ];
        let base_params = params.to_array();
        for i in 0..7 {
            let mut pp = base_params;
            let mut pm = base_params;
            pp[i] += h;
            pm[i] -= h;
            let fd =
                (f(&Wiener7Params::from_array(pp)) - f(&Wiener7Params::from_array(pm))) / (2.0 * h);
            assert_relative_eq!(analytic[i], fd, epsilon = 5e-3, max_relative = 1e-2);
        }
    }

    #[test]
    fn wiener5_log_prob_consistency() {
        let (eps, obs, params, expected_lp) = stan_wiener5_data();
        let eval_upper = Wiener5.log_prob(&obs, &params, eps);
        assert_relative_eq!(eval_upper.log_prob, expected_lp, epsilon = 1e-8);
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
        let (ks, kl) = Wiener4::series_counts(&core);
        assert!(ks <= 50, "ks too large: {}", ks);
        assert!(kl <= 20, "kl too large: {}", kl);
    }

    #[test]
    fn test_series_monotonicity() {
        let t_prime = 0.5;
        let w = 0.3;
        let k = 20;
        let small_log = Wiener4::small_time_log_series(t_prime, w, k).unwrap();
        let large_log = Wiener4::large_time_log_series(t_prime, w, k).unwrap();
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
        assert!(fused.log_prob.is_finite());
        assert!(fused.grad.to_array().iter().all(|g| g.is_finite()));
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
        assert!(unconstrained.iter().all(|u| u.is_finite()));
        let roundtrip = Wiener5Params::from_unconstrained(&unconstrained);
        assert_relative_eq!(
            roundtrip.base.alpha,
            constrained.base.alpha,
            epsilon = 1e-10
        );
        assert_relative_eq!(roundtrip.s_delta, constrained.s_delta, epsilon = 1e-10);
        assert!(Wiener5Params::log_abs_det_jacobian(&unconstrained).is_finite());
    }

    #[test]
    #[should_panic(expected = "InvalidParameters")]
    fn test_invalid_params() {
        Wiener4Params::with_params(-1.0, 0.1, 0.5, 0.0).unwrap();
    }

    #[test]
    fn debug_series_derivatives() {
        let t_prime = 0.1;
        let w = 0.3;
        let k = 15;
        let log_s = Wiener4::small_time_log_series(t_prime, w, k).unwrap();
        let raw_s = Wiener4::small_time_series_raw(t_prime, w, k).unwrap();
        let missing_pref =
            -0.5 * TAU.ln() - 1.5 * t_prime.ln() - ((1.0 - w) * (1.0 - w) * 0.5 / t_prime);
        let reconstructed_log_s = missing_pref + raw_s.ln();
        assert_relative_eq!(reconstructed_log_s, log_s, epsilon = 1e-10);
    }

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
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let alpha = 1.5;
        let tau = 0.2;
        let beta = 0.55;
        let delta = 0.4;
        let sv = 0.1;
        let params7 = Wiener7Params::with_params_unchecked(alpha, tau, beta, delta, 0.0, 0.0, sv);
        let params5 = Wiener5Params::with_params_unchecked(alpha, tau, beta, delta, sv);
        let r7 = Wiener7.fused(&obs, &params7, 1e-10);
        let r5 = Wiener5.fused(&obs, &params5, 1e-12);
        assert_relative_eq!(r7.log_prob, r5.log_prob, epsilon = 1e-10);
        assert_relative_eq!(r7.grad.alpha, r5.grad.alpha, epsilon = 1e-10);
        assert_relative_eq!(r7.grad.tau, r5.grad.tau, epsilon = 1e-10);
        assert_relative_eq!(r7.grad.beta, r5.grad.beta, epsilon = 1e-10);
        assert_relative_eq!(r7.grad.delta, r5.grad.delta, epsilon = 1e-10);
        assert_relative_eq!(r7.grad.s_delta, r5.grad.s_delta, epsilon = 1e-10);
        assert_eq!(r7.grad.s_beta, 0.0);
        assert_eq!(r7.grad.s_tau, 0.0);
    }

    #[test]
    fn test_wiener7_finite() {
        let obs = WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(2.0, 0.3, 0.5, -0.2, 0.2, 0.1, 0.3).unwrap();
        let fused = Wiener7.fused(&obs, &params, 1e-4);
        assert!(fused.log_prob.is_finite());
        assert!(fused.grad.to_array().iter().all(|g| g.is_finite()));
    }

    #[test]
    fn test_wiener7_rt_less_than_t0_returns_neg_inf() {
        let obs = WienerObservation {
            rt: 0.15,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(1.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(
            Wiener7.fused(&obs, &params, 1e-4).log_prob,
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_wiener7_sw_out_of_bounds_returns_neg_inf() {
        // beta0=0.1, sw=0.5 → lower bound = 0.1 − 0.25 = −0.15 < 0
        let params = Wiener7Params::with_params(1.0, 0.1, 0.1, 0.0, 0.5, 0.0, 0.0).unwrap();
        let obs = WienerObservation {
            rt: 0.5,
            boundary: Boundary::Upper,
        };
        assert_eq!(
            Wiener7.fused(&obs, &params, 1e-4).log_prob,
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_wiener7_st0_truncated_interval() {
        // st0 > 0 but (rt − t0)/st0 < 1 → upper limit clipped
        let obs = WienerObservation {
            rt: 0.35,
            boundary: Boundary::Upper,
        };
        let params = Wiener7Params::with_params(1.0, 0.2, 0.5, 0.5, 0.0, 0.0, 0.2).unwrap();
        assert!(Wiener7.fused(&obs, &params, 1e-4).log_prob.is_finite());
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
        assert!(grad.iter().all(|g| g.is_finite()));

        let mut sum_lp = 0.0;
        let mut sum_grad = [0.0; 7];
        for obs in target.data.iter() {
            let fused = Wiener7.fused(obs, &params, 1e-4);
            sum_lp += fused.log_prob;
            for i in 0..7 {
                sum_grad[i] += fused.grad[i];
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
        // Reference log-densities and gradients computed with WienR in R;
        // test cases adapted from the Stan test suite.
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
            -34.5869239580708,
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
            let params = Wiener7Params::with_params_unchecked(
                a_vec[i], t0_vec[i], w_vec[i], v_vec[i], sw_vec[i], st0_vec[i], sv_vec[i],
            );
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
}
