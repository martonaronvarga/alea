use crate::buffer::OwnedBuffer;
use crate::error::{ProbError, Result};
use ffi::Options;

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

/// Batch evaluation strategy for `WienerObservations` targets.
///
/// `Direct` is the conservative default. `BranchBuckets` groups observations by
/// series branch and term count; it can help some workloads, but it allocates a
/// scratch buffer per evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BatchStrategy {
    #[default]
    Direct,
    BranchBuckets,
}

/// Structure-of-arrays storage for Wiener reaction-time observations.
///
/// Upper and lower boundary hits are stored separately so batch likelihoods can
/// avoid a branch in the hot loop and use contiguous reaction-time data. Each
/// allocation has 64-byte base alignment; arbitrary subslices may be unaligned.
#[derive(Debug)]
pub struct WienerObservations {
    upper_rt: OwnedBuffer<f64>,
    lower_rt: OwnedBuffer<f64>,
    batch_strategy: BatchStrategy,
}

impl WienerObservations {
    #[must_use]
    pub fn new(upper_rt: OwnedBuffer<f64>, lower_rt: OwnedBuffer<f64>) -> Self {
        Self {
            upper_rt,
            lower_rt,
            batch_strategy: BatchStrategy::default(),
        }
    }

    /// Selects the batch strategy used by `Target<Wiener*, WienerObservations>`.
    #[must_use]
    pub fn with_batch_strategy(mut self, batch_strategy: BatchStrategy) -> Self {
        self.batch_strategy = batch_strategy;
        self
    }

    #[must_use]
    pub fn batch_strategy(&self) -> BatchStrategy {
        self.batch_strategy
    }

    #[must_use]
    pub fn upper_rt(&self) -> &[f64] {
        self.upper_rt.as_slice()
    }

    #[must_use]
    pub fn lower_rt(&self) -> &[f64] {
        self.lower_rt.as_slice()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.upper_rt.len() + self.lower_rt.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FromIterator<WienerObservation> for WienerObservations {
    fn from_iter<T: IntoIterator<Item = WienerObservation>>(iter: T) -> Self {
        let observations: Vec<WienerObservation> = iter.into_iter().collect();
        let upper_len = observations
            .iter()
            .filter(|obs| matches!(obs.boundary, Boundary::Upper))
            .count();
        let lower_len = observations.len() - upper_len;

        let mut upper_rt = OwnedBuffer::new(upper_len);
        let mut lower_rt = OwnedBuffer::new(lower_len);
        let mut upper_idx = 0;
        let mut lower_idx = 0;

        for obs in observations {
            match obs.boundary {
                Boundary::Upper => {
                    upper_rt.as_mut_slice()[upper_idx] = obs.rt;
                    upper_idx += 1;
                }
                Boundary::Lower => {
                    lower_rt.as_mut_slice()[lower_idx] = obs.rt;
                    lower_idx += 1;
                }
            }
        }

        Self::new(upper_rt, lower_rt)
    }
}

impl From<Vec<WienerObservation>> for WienerObservations {
    fn from(observations: Vec<WienerObservation>) -> Self {
        observations.into_iter().collect()
    }
}

/// Canonical Wiener first-passage families in Stan's convention:
/// Wiener4 = standard four parameter (alpha, tau, beta, delta) wiener family
/// Wiener5 = Wiener4 + variance parameter for drift (delta)
/// Wiener7 = Wiener5 + variance parameters for beta and tau
/// Reference:
/// <https://mc-stan.org/docs/functions-reference/positive_lower-bounded_distributions.html#wiener-first-passage-time-distribution>
#[derive(Debug, Clone, Copy)]
pub struct Wiener4;
pub struct Wiener5;
pub struct Wiener7;

/// Series branch selected by the Navarro-Fuss density representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesBranch {
    SmallTime,
    LargeTime,
}

/// Term-count diagnostics for Wiener series branch selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WienerBranchCounts {
    pub branch: SeriesBranch,
    pub k_small_density: usize,
    pub k_large_density: usize,
    pub k_small_grad_w: usize,
    pub k_large_grad_w: usize,
    pub k_small_used: usize,
    pub k_large_used: usize,
}

/// Numerical integration strategy for full Wiener models with variability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quadrature {
    /// Adaptive cubature over the active variability dimensions.
    Adaptive,
    /// Fixed Gauss-Legendre product rule. Supported orders are 1, 5, 7, 15, and 25.
    FixedGaussLegendre { order: usize },
}

/// Numerical options for checked user-facing Wiener evaluations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WienerOptions {
    /// Precision used for adaptive cubature, or as the default series precision.
    pub precision: f64,
    /// Optional precision for inner Wiener series when it should differ from `precision`.
    pub inner_precision: Option<f64>,
    /// Quadrature strategy for Wiener7 start-time/non-decision-time variability.
    pub quadrature: Quadrature,
}

impl Default for WienerOptions {
    fn default() -> Self {
        Self {
            precision: 1e-12,
            inner_precision: None,
            quadrature: Quadrature::Adaptive,
        }
    }
}

impl WienerOptions {
    #[must_use]
    pub fn new(precision: f64) -> Self {
        Self {
            precision,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_inner_precision(mut self, inner_precision: f64) -> Self {
        self.inner_precision = Some(inner_precision);
        self
    }

    #[must_use]
    pub fn with_quadrature(mut self, quadrature: Quadrature) -> Self {
        self.quadrature = quadrature;
        self
    }

    pub(super) fn series_precision(self) -> f64 {
        self.inner_precision.unwrap_or(self.precision)
    }

    pub(super) fn validate(self) -> Result<Self> {
        if !self.precision.is_finite() || self.precision <= 0.0 {
            return Err(ProbError::InvalidParameters(
                "WienerOptions precision must be finite and positive".to_string(),
            ));
        }
        if let Some(inner_precision) = self.inner_precision
            && (!inner_precision.is_finite() || inner_precision <= 0.0)
        {
            return Err(ProbError::InvalidParameters(
                "WienerOptions inner precision must be finite and positive".to_string(),
            ));
        }
        if let Quadrature::FixedGaussLegendre { order } = self.quadrature
            && !matches!(order, 1 | 5 | 7 | 15 | 25)
        {
            return Err(ProbError::InvalidParameters(
                "fixed Gauss-Legendre quadrature order must be one of 1, 5, 7, 15, or 25"
                    .to_string(),
            ));
        }
        Ok(self)
    }
}

/// Canonical Wiener first-passage parameters in Stan's naming:
/// alpha: boundary separation, alpha \in R^+
/// tau: non-decision time, tau \in R^+
/// beta: relative starting point, beta \in (0, 1)
/// delta: drift rate, delta \in R
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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

impl Default for Wiener4Params {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            tau: 0.0,
            beta: 0.5,
            delta: 0.0,
        }
    }
}

impl Wiener4Params {
    pub const DIM: usize = 4;
    #[must_use]
    pub fn to_array(&self) -> [f64; 4] {
        [self.alpha, self.tau, self.beta, self.delta]
    }
    #[must_use]
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
            _ => panic!("Wiener4Params index out of range: {idx}"),
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
            _ => panic!("Wiener4Params index out of range: {idx}"),
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
    #[must_use]
    pub fn to_array(&self) -> [f64; 4] {
        [self.alpha, self.tau, self.beta, self.delta]
    }
    #[must_use]
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

pub(super) const NAN_GRAD_4: Wiener4Grad = Wiener4Grad {
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
            _ => panic!("Wiener4Grad index out of range: {idx}"),
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
            _ => panic!("Wiener4Grad index out of range: {idx}"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener4Eval {
    pub log_prob: f64,
    pub grad: Wiener4Grad,
}

impl Wiener4Eval {
    #[must_use]
    pub fn grad_array(&self) -> [f64; 4] {
        self.grad.to_array()
    }
}

#[derive(Copy, Clone, Debug)]
pub(super) struct Wiener4Core {
    pub(super) t: f64,
    pub(super) a: f64,
    pub(super) t_prime: f64,
    pub(super) w_eff: f64,
    pub(super) v_eff: f64,
    pub(super) pref: f64,
    pub(super) log_eps_eff: f64,
    pub(super) beta_sign: f64,
    pub(super) delta_sign: f64,
}

/// s_delta = standard deviation in drift rate, s_delta \in R^>=0
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener5Params {
    pub(super) base: Wiener4Params,
    pub(super) s_delta: f64,
}

impl Default for Wiener5Params {
    fn default() -> Self {
        Self {
            base: Wiener4Params::default(),
            s_delta: 0.0,
        }
    }
}

impl Wiener5Params {
    pub const DIM: usize = 5;
    #[must_use]
    pub fn to_array(&self) -> [f64; 5] {
        let b = self.base.to_array();
        [b[0], b[1], b[2], b[3], self.s_delta]
    }
    #[must_use]
    pub fn from_array(arr: [f64; 5]) -> Self {
        Self {
            base: Wiener4Params::from_array([arr[0], arr[1], arr[2], arr[3]]),
            s_delta: arr[4],
        }
    }
    // Flat accessors
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.base.alpha
    }
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.base.tau
    }
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.base.beta
    }
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.base.delta
    }
    #[must_use]
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
            _ => panic!("Wiener5Params index out of range: {idx}"),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener5Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0..=3 => &mut self.base[idx],
            4 => &mut self.s_delta,
            _ => panic!("Wiener5Params index out of range: {idx}"),
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
    #[must_use]
    pub fn to_array(&self) -> [f64; 5] {
        [self.alpha, self.tau, self.beta, self.delta, self.s_delta]
    }
    #[must_use]
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
            _ => panic!("Wiener5Grad index out of range: {idx}"),
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
            _ => panic!("Wiener5Grad index out of range: {idx}"),
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Wiener5Eval {
    pub log_prob: f64,
    pub grad: Wiener5Grad,
}

impl Wiener5Eval {
    #[must_use]
    pub fn grad_array(&self) -> [f64; 5] {
        self.grad.to_array()
    }
}

pub(super) const NAN_GRAD_5: Wiener5Grad = Wiener5Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
    s_delta: f64::NAN,
};

pub(super) const FAIL_EVAL_5: Wiener5Eval = Wiener5Eval {
    log_prob: f64::NEG_INFINITY,
    grad: NAN_GRAD_5,
};

#[derive(Copy, Clone, Debug)]
pub(super) struct Wiener5Core {
    pub(super) base: Wiener4Core,
    pub(super) sv: f64,
    pub(super) sv2: f64,
    pub(super) lam: f64,
}

/// s_beta: standard deviation of beta
/// s_tau: standard deviation of tau
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener7Params {
    pub(super) base: Wiener5Params,
    pub(super) s_beta: f64,
    pub(super) s_tau: f64,
}

impl Default for Wiener7Params {
    fn default() -> Self {
        Self {
            base: Wiener5Params::default(),
            s_beta: 0.0,
            s_tau: 0.0,
        }
    }
}

impl Wiener7Params {
    pub const DIM: usize = 7;
    #[must_use]
    pub fn to_array(&self) -> [f64; 7] {
        let b = self.base.to_array(); // [alpha, tau, beta, delta, sv]
        [b[0], b[1], b[2], b[3], b[4], self.s_beta, self.s_tau]
    }
    #[must_use]
    pub fn from_array(arr: [f64; 7]) -> Self {
        Self {
            base: Wiener5Params::from_array([arr[0], arr[1], arr[2], arr[3], arr[4]]),
            s_beta: arr[5],
            s_tau: arr[6],
        }
    }
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.base.base.alpha
    }
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.base.base.tau
    }
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.base.base.beta
    }
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.base.base.delta
    }
    #[must_use]
    pub fn s_delta(&self) -> f64 {
        self.base.s_delta
    }
    #[must_use]
    pub fn s_beta(&self) -> f64 {
        self.s_beta
    }
    #[must_use]
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
            _ => panic!("Wiener7Params index out of range: {idx}"),
        }
    }
}
impl std::ops::IndexMut<usize> for Wiener7Params {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0..=4 => &mut self.base[idx],
            5 => &mut self.s_beta,
            6 => &mut self.s_tau,
            _ => panic!("Wiener7Params index out of range: {idx}"),
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
    #[must_use]
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
    #[must_use]
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

pub(super) const NAN_GRAD_7: Wiener7Grad = Wiener7Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
    s_delta: f64::NAN,
    s_beta: f64::NAN,
    s_tau: f64::NAN,
};
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
            _ => panic!("Wiener7Grad index out of range: {idx}"),
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
            _ => panic!("Wiener7Grad index out of range: {idx}"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener7Eval {
    pub log_prob: f64,
    pub grad: Wiener7Grad,
}

impl Wiener7Eval {
    #[must_use]
    pub fn grad_array(&self) -> [f64; 7] {
        self.grad.to_array()
    }
}

#[derive(Clone, Debug)]
pub(super) struct Wiener7Core {
    pub(super) alpha: f64,
    pub(super) tau0: f64,
    pub(super) beta0: f64,
    pub(super) delta: f64,
    pub(super) sv: f64,
    pub(super) sw: f64,
    pub(super) st0: f64,
    pub(super) dim: usize,
    pub(super) xmin: [f64; 2],
    pub(super) xmax: [f64; 2],
    pub(super) eps_series: f64,
    pub(super) opts: Options,
}

impl Wiener4Params {
    /// Returns the valid neutral default parameter point.
    ///
    /// This is equivalent to [`Default::default`].
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a named-parameter builder from the valid neutral default point.
    #[inline]
    #[must_use]
    pub fn builder() -> Wiener7ParamsBuilder {
        Wiener7ParamsBuilder::default()
    }

    /// Builds a validated parameter object.
    ///
    /// # Errors
    ///
    /// Returns [`ProbError::InvalidParameters`] if `alpha <= 0`, `tau < 0`,
    /// `beta` is outside `(0, 1)`, or any argument is not finite.
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

    /// Builds parameters without validation.
    ///
    /// This intentionally accepts arbitrary finite or non-finite values so callers can
    /// represent unconstrained proposal states. Evaluation routines remain responsible
    /// for validating inputs and returning an out-of-support result for invalid values.
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
    #[must_use]
    pub fn decision_time(&self, rt: f64) -> Option<f64> {
        let t = rt - self.tau;
        if t > 0.0 && t.is_finite() {
            Some(t)
        } else {
            None
        }
    }
    #[inline]
    #[must_use]
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
    /// Returns the valid neutral default parameter point.
    ///
    /// This is equivalent to [`Default::default`].
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a named-parameter builder from the valid neutral default point.
    #[inline]
    #[must_use]
    pub fn builder() -> Wiener7ParamsBuilder {
        Wiener7ParamsBuilder::default()
    }

    /// Builds a validated parameter object.
    ///
    /// # Errors
    ///
    /// Returns [`ProbError::InvalidParameters`] if the four-parameter base is
    /// invalid or `s_delta` is negative or non-finite.
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

    /// Builds parameters without validation.
    ///
    /// This intentionally accepts arbitrary finite or non-finite values so callers can
    /// represent unconstrained proposal states. Evaluation routines remain responsible
    /// for validating inputs and returning an out-of-support result for invalid values.
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
    /// Returns the valid neutral default parameter point.
    ///
    /// This is equivalent to [`Default::default`].
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a named-parameter builder from the valid neutral default point.
    #[inline]
    #[must_use]
    pub fn builder() -> Wiener7ParamsBuilder {
        Wiener7ParamsBuilder::default()
    }

    /// Builds a validated parameter object.
    ///
    /// # Errors
    ///
    /// Returns [`ProbError::InvalidParameters`] if the five-parameter base is
    /// invalid, `s_beta` is outside `[0, 1)`, or `s_tau` is negative or non-finite.
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

    /// Builds parameters without validation.
    ///
    /// This intentionally accepts arbitrary finite or non-finite values so callers can
    /// represent unconstrained proposal states. Evaluation routines remain responsible
    /// for validating inputs and returning an out-of-support result for invalid values.
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

/// Builder for [`Wiener7Params`] using the canonical parameter names.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Wiener7ParamsBuilder {
    params: Wiener7Params,
}

impl Wiener7ParamsBuilder {
    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.params.base.base.alpha = alpha;
        self
    }

    #[must_use]
    pub fn tau(mut self, tau: f64) -> Self {
        self.params.base.base.tau = tau;
        self
    }

    #[must_use]
    pub fn beta(mut self, beta: f64) -> Self {
        self.params.base.base.beta = beta;
        self
    }

    #[must_use]
    pub fn delta(mut self, delta: f64) -> Self {
        self.params.base.base.delta = delta;
        self
    }

    #[must_use]
    pub fn s_delta(mut self, s_delta: f64) -> Self {
        self.params.base.s_delta = s_delta;
        self
    }

    #[must_use]
    pub fn s_beta(mut self, s_beta: f64) -> Self {
        self.params.s_beta = s_beta;
        self
    }

    #[must_use]
    pub fn s_tau(mut self, s_tau: f64) -> Self {
        self.params.s_tau = s_tau;
        self
    }

    /// Builds validated [`Wiener7Params`].
    ///
    /// # Errors
    ///
    /// Returns [`ProbError::InvalidParameters`] if the configured parameter point
    /// fails the same checks as [`Wiener7Params::with_params`].
    pub fn build(self) -> Result<Wiener7Params> {
        Wiener7Params::with_params(
            self.params.alpha(),
            self.params.tau(),
            self.params.beta(),
            self.params.delta(),
            self.params.s_beta(),
            self.params.s_tau(),
            self.params.s_delta(),
        )
    }

    /// Returns the current unchecked parameter point without validation.
    #[must_use]
    pub fn build_unchecked(self) -> Wiener7Params {
        self.params
    }
}
