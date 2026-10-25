use super::*;

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

// Default already gives zeros; we can override to set NANs for clarity

/// s_delta = standard deviation in drift rate, s_delta \in R^>=0
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener5Params {
    pub(in crate::wiener) base: Wiener4Params,
    pub(in crate::wiener) s_delta: f64,
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

/// Full-model variability uses uniform widths: beta ± s_beta/2 and
/// non-decision time in [tau, tau + s_tau]. Drift variability s_delta is
/// the standard deviation of the normal drift distribution.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wiener7Params {
    pub(in crate::wiener) base: Wiener5Params,
    pub(in crate::wiener) s_beta: f64,
    pub(in crate::wiener) s_tau: f64,
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
