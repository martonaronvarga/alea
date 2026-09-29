//! Drift Diffusion Model mathematical primitives.
//!
//! This module implements the Navarro–Fuss (2009) Wiener first-passage time
//! density for the standard DDM and parameter types for the
//! Congruency Sequence Effect (CSE) framework (Luo et al., 2022).
//!
//! # Model
//!
//! Evidence accumulates as a Wiener process:
//!
//!   dX(t) = v dt + σ dW(t),   X(0) = w · a,   X ∈ {0, a}
//!
//! with diffusion coefficient σ fixed at 1 (conventional scaling).
//! A response is made when the process first hits boundary 0 (lower / error)
//! or a (upper / correct).  The observed reaction time is RT = T + t_er, where
//! T is the decision time and t_er is the non-decision time.
//!
//! # References
//!
//! - Ratcliff (1978). *Psychological Review*.
//! - Navarro & Fuss (2009). *Journal of Mathematical Psychology*, 53, 222–230.
//! - Luo et al. (2022). *Psychonomic Bulletin & Review*, 29, 2034–2051.
//! - Stan Wiener documentation

use crate::kernels::numeric::*;
use crate::{density::LogDensity, extension::BatchLogDensity, state_space::ObservationModel};
use std::f64::consts::PI;

// Jacobian
#[inline(always)]
pub fn log_jacobian(theta_a: f64, theta_w: f64, theta_ter: f64) -> f64 {
    // d(exp(theta_a))/d theta_a = exp(theta_a) => log Jacobian = theta_a
    // d(sigmoid(theta_w))/d theta_w = w(1-w)
    theta_a + log_sigmoid(theta_w) + log1m_sigmoid(theta_w) + theta_ter
}
// ─────────────────────────────────────────────────────────────────────────────
// Domain Types
// ─────────────────────────────────────────────────────────────────────────────

/// Which absorbing boundary terminated evidence accumulation.
///
/// Convention: `Upper` ≡ correct response (drift favours upper boundary
/// when `v > 0`); `Lower` ≡ error response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Response {
    /// Process hit the upper boundary at `a`.
    Upper,
    /// Process hit the lower boundary at `0`.
    Lower,
}

/// A single DDM observation.
#[derive(Debug, Clone, Copy)]
pub struct DdmTrial {
    /// Observed reaction time in seconds.
    pub rt: f64,
    /// Which boundary was hit.
    pub response: Response,
}

/// Constrained DDM parameters (the "natural" model space).
///
/// Validity constraints (checked at construction via [`DdmParams::new`]):
/// - `a > 0`
/// - `w ∈ (0, 1)`
/// - `t_er > 0`
/// - `v` is any finite real
#[derive(Debug, Clone, Copy)]
pub struct DdmParams {
    /// Drift rate.  Positive values bias the process toward the upper boundary.
    pub v: f64,
    /// Boundary separation (> 0).  Larger → slower, more accurate decisions.
    pub a: f64,
    /// Relative starting point `z / a ∈ (0, 1)`.  `0.5` = unbiased.
    pub w: f64,
    /// Non-decision time in seconds (> 0).  Minimum possible RT.
    pub t_er: f64,
}

impl DdmParams {
    /// Construct parameters after checking validity.
    ///
    /// Returns `None` if any constraint is violated.
    #[inline]
    pub fn new(v: f64, a: f64, w: f64, t_er: f64) -> Option<Self> {
        if !v.is_finite()
            || !(a.is_finite() && a > 0.0)
            || !(w > 0.0 && w < 1.0)
            || !(t_er.is_finite() && t_er > 0.0)
        {
            return None;
        }
        Some(Self { v, a, w, t_er })
    }

    /// Decision time for a given raw RT.  Returns `None` if `rt ≤ t_er`.
    #[inline]
    pub fn decision_time(&self, rt: f64) -> Option<f64> {
        let t = rt - self.t_er;
        if t > 0.0 && t.is_finite() {
            Some(t)
        } else {
            None
        }
    }
}

/// The four trial types that arise in the Congruency Sequence Effect (CSE):
/// previous-trial congruency × current-trial congruency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrialCondition {
    /// Previous congruent, current congruent.
    CC,
    /// Previous congruent, current incongruent.
    CI,
    /// Previous incongruent, current congruent.
    IC,
    /// Previous incongruent, current incongruent.
    II,
}

impl TrialCondition {
    /// Index in `[CC, CI, IC, II]` order (used for parameter vectors).
    #[inline]
    pub fn index(self) -> usize {
        match self {
            TrialCondition::CC => 0,
            TrialCondition::CI => 1,
            TrialCondition::IC => 2,
            TrialCondition::II => 3,
        }
    }

    /// Return all four conditions in canonical order.
    #[inline]
    pub fn all() -> [TrialCondition; 4] {
        [
            TrialCondition::CC,
            TrialCondition::CI,
            TrialCondition::IC,
            TrialCondition::II,
        ]
    }
}

/// One observation in a CSE experiment.
#[derive(Debug, Clone, Copy)]
pub struct CseTrial {
    /// Trial condition (previous × current congruency).
    pub condition: TrialCondition,
    /// RT and response.
    pub obs: DdmTrial,
}

#[derive(Debug, Clone, Copy)]
pub struct CseParams {
    pub drift: [f64; 4],
    pub a: f64,
    pub w: f64,
    pub t_er: f64,
}

impl CseParams {
    #[inline]
    pub fn new(drift: [f64; 4], a: f64, w: f64, t_er: f64) -> Option<Self> {
        if drift.iter().all(|x| x.is_finite())
            && a.is_finite()
            && a > 0.0
            && w > 0.0
            && w < 1.0
            && t_er.is_finite()
            && t_er > 0.0
        {
            Some(Self { drift, a, w, t_er })
        } else {
            None
        }
    }

    #[inline]
    pub fn for_condition(&self, c: TrialCondition) -> DdmParams {
        DdmParams {
            v: self.drift[c.index()],
            a: self.a,
            w: self.w,
            t_er: self.t_er,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Navarro–Fuss Wiener first-passage time density
// ─────────────────────────────────────────────────────────────────────────────
//
// For the unit-boundary process  dX̃(s) = μ ds + dW̃(s),  X̃(0) = w ∈ (0, 1),
// absorbed at {0, 1}, the density for hitting the upper boundary at normalised
// time s = τ is (Navarro & Fuss, 2009, eqs. 1 & 2):
//
//   Large-time:  f̃(τ; μ, w) = π · Σ_{k≥1} k sin(kπw) exp(−k²π²τ/2)
//                              × exp(μw − μ²τ/2)
//
//   Small-time:  f̃(τ; μ, w) = (2πτ³)^{−1/2} · Σ_{k∈ℤ} (w+2k) exp(−(w+2k)²/(2τ))
//                              × exp(μw − μ²τ/2)
//
// To scale to boundary separation a and starting point z = wa:
//   f(t; v, a, w) = (1/a²) · f̃(t/a²; va, w)
//
// The log-density therefore has the factored form:
//   log f = −2 ln a + (μw − μ²τ/2) + log(series)
// where μ = va, τ = t/a², and "series" is the series sum WITHOUT the
// exp(μw − μ²τ/2) common factor.
//
// For the small-time branch, an additional prefactor (2πτ³)^{−1/2} appears
// inside log(series):
//   log(series_small) = −1/2 ln(2π) − 3/2 ln τ + ln(Σ (w+2k) exp(…))

/// Normalised time threshold below which the small-time series is used.
const TAU_THRESHOLD: f64 = 0.65;
/// Absolute convergence tolerance for series truncation.
const SERIES_TOL: f64 = 1e-12;
/// Maximum series terms (safety cap; should never be reached).
const MAX_TERMS: usize = 200;

/// Large-time series sum:
///   S = π · Σ_{k=1}^{K} k · sin(kπw) · exp(−k²π²τ/2)
///
/// The exp(μw − μ²τ/2) common factor is NOT included here.
/// Caller must verify `tau >= TAU_THRESHOLD`.
fn large_time_sum(tau: f64, w: f64) -> f64 {
    let decay_rate = 0.5 * PI * PI * tau;
    let mut sum = 0.0f64;

    for k in 1..=MAX_TERMS {
        let kf = k as f64;
        // exp factor: monotonically decreasing, drives convergence
        let decay = (-decay_rate * kf * kf).exp();
        // Early-exit: once the envelope is negligible the term is too
        if decay < SERIES_TOL {
            break;
        }
        sum += kf * (PI * kf * w).sin() * decay;
    }

    PI * sum
}

/// Small-time series sum:
///   S = Σ_{k=k_min}^{k_max} (w+2k) · exp(−(w+2k)² / (2τ))
///
/// The exp(μw − μ²τ/2) common factor and the (2πτ³)^{−½} prefactor
/// are NOT included here.
/// Caller must verify `tau < TAU_THRESHOLD`.
fn small_time_sum(tau: f64, w: f64) -> f64 {
    // Include all k where the Gaussian factor exceeds SERIES_TOL.
    // (w+2k)² / (2τ) < −ln(SERIES_TOL)  ⟺  |w+2k| < sqrt(−2τ ln ε)
    let half_range = (-2.0 * tau * SERIES_TOL.ln()).sqrt();
    let k_lo = ((-half_range - w) * 0.5).floor() as i32 - 1;
    let k_hi = ((half_range - w) * 0.5).ceil() as i32 + 1;

    let mut sum = 0.0f64;
    for k in k_lo..=k_hi {
        let kf = k as f64;
        let wk = w + 2.0 * kf;
        sum += wk * (-(wk * wk) / (2.0 * tau)).exp();
    }
    sum
}

/// Log density of the Wiener first-passage time to the **upper** boundary.
///
/// # Arguments
///
/// * `t`  – decision time (RT − t_er), must be > 0
/// * `v`  – drift rate (ℝ)
/// * `a`  – boundary separation (> 0)
/// * `w`  – relative starting point `z/a ∈ (0, 1)`
///
/// Returns `f64::NEG_INFINITY` on invalid inputs or numerically degenerate
/// series values.
pub fn log_wiener_upper(t: f64, v: f64, a: f64, w: f64) -> f64 {
    // Guard invalid inputs
    if !(t.is_finite()
        && t > 0.0
        && a.is_finite()
        && a > 0.0
        && w.is_finite()
        && w > 0.0
        && w < 1.0
        && v.is_finite())
    {
        return f64::NEG_INFINITY;
    }

    let tau = t / (a * a); // normalised time
    let mu = v * a; // normalised drift
    let log_common = mu * w - 0.5 * mu * mu * tau;

    if tau >= TAU_THRESHOLD {
        // Large-time series: reliable for τ ≥ 0.65
        let s = large_time_sum(tau, w);
        if s <= 0.0 || !s.is_finite() {
            return f64::NEG_INFINITY;
        }
        -2.0 * a.ln() + log_common + s.ln()
    } else {
        // Small-time series: reliable for τ < 0.65
        let s = small_time_sum(tau, w);
        if s <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // Prefactor: (2πτ³)^{−½} contributes −½ln(2π) − 3/2 ln τ
        let log_prefactor = -0.5 * (2.0 * PI).ln() - 1.5 * tau.ln();
        -2.0 * a.ln() + log_common + log_prefactor + s.ln()
    }
}

/// Log density of the Wiener first-passage time to the **lower** boundary.
///
/// Uses the symmetry `f_lower(t; v, a, w) = f_upper(t; −v, a, 1−w)`.
#[inline]
pub fn log_wiener_lower(t: f64, v: f64, a: f64, w: f64) -> f64 {
    log_wiener_upper(t, -v, a, 1.0 - w)
}

// ─────────────────────────────────────────────────────────────────────────────
// Trial and dataset log-likelihood
// ─────────────────────────────────────────────────────────────────────────────

/// Log-likelihood contribution of a single trial.
///
/// Returns `NEG_INFINITY` when `trial.rt ≤ params.t_er` or the density
/// evaluates to zero / underflows.
pub fn log_trial_ll(trial: &DdmTrial, params: &DdmParams) -> f64 {
    let t = match params.decision_time(trial.rt) {
        Some(t) => t,
        None => return f64::NEG_INFINITY,
    };
    match trial.response {
        Response::Upper => log_wiener_upper(t, params.v, params.a, params.w),
        Response::Lower => log_wiener_lower(t, params.v, params.a, params.w),
    }
}

/// Summed log-likelihood over a slice of independent trials.
///
/// Returns `NEG_INFINITY` as soon as any individual trial returns it
/// (short-circuit for efficiency).
pub fn log_likelihood(trials: &[DdmTrial], params: &DdmParams) -> f64 {
    let mut ll = 0.0f64;
    for trial in trials {
        let contrib = log_trial_ll(trial, params);
        if contrib == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        ll += contrib;
    }
    ll
}

/// Summed log-likelihood for CSE trials given per-condition drift rates and
/// shared structural parameters.
///
/// `v_per_cond[i]` is the drift for `TrialCondition::all()[i]`.
pub fn log_likelihood_cse(trials: &[CseTrial], params: &CseParams) -> f64 {
    let mut ll = 0.0f64;
    for trial in trials {
        let p = params.for_condition(tr.condition);
        let contrib = log_trial_ll(&trial.obs, &p);
        if contrib == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        ll += contrib;
    }
    ll
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter transformations (constrained ↔ unconstrained)
// ─────────────────────────────────────────────────────────────────────────────
//
// Unconstrained parameterisation for gradient-based / MCMC samplers:
//
//   θ_a   = ln(a)           a     = exp(θ_a)
//   θ_w   = logit(w)        w     = sigmoid(θ_w)
//   θ_ter = ln(t_er)        t_er  = exp(θ_ter)
//
// Jacobian contribution to log-density:
//   log|det J| = ln(a) + ln(w) + ln(1−w) + ln(t_er)
//              = θ_a  + ln σ(θ_w) + ln(1−σ(θ_w)) + θ_ter
//              = θ_a  − softplus(θ_w) − softplus(−θ_w) + θ_ter

/// Decode a 4-element unconstrained vector `[v, ln(a), logit(w), ln(t_er)]`
/// into a [`DdmParams`].  Returns `None` if any decoded value is out of range.
pub fn decode_params(theta: &[f64]) -> Option<DdmParams> {
    if theta.len() < 4 {
        return None;
    }
    DdmParams::new(theta[0], theta[1].exp(), sigmoid(theta[2]), theta[3].exp())
}

/// Decode a 7-element unconstrained CSE vector:
///   `[v_CC, v_CI, v_IC, v_II, ln(a), logit(w), ln(t_er)]`
/// Returns the four drift rates and shared structural parameters.
pub fn decode_cse_params(x: &[f64]) -> Option<CseParams> {
    if theta.len() < 7 || !theta[..7].iter().all(|x| x.is_finite()) {
        return None;
    }
    let drift = [theta[0], theta[1], theta[2], theta[3]];

    CseParams::new(drift, theta[4].exp(), sigmoid(theta[5]), theta[6].exp())
}

#[inline]
pub fn log_unconstrained_jacobian(theta: &[f64]) -> f64 {
    if theta.len() < 4 {
        return f64::NEG_INFINITY;
    }
    log_jacobian(theta[1], theta[2], theta[3])
}

pub struct WienerLikelihood<'a> {
    pub trials: &'a [DdmTrial],
}

impl<'a> WienerLikelihood<'a> {
    #[inline]
    pub fn new(trials: &'a [DdmTrial]) -> Self {
        Self { trials }
    }

    #[inline]
    pub fn log_prob_constrained(&self, params: &DdmParams) -> f64 {
        log_likelihood(self.trials, params)
    }

    #[inline]
    pub fn log_prob_unconstrained(&self, theta: &[f64]) -> f64 {
        let params = match decode_params(theta) {
            Some(p) => p,
            None => return f64::NEG_INFINITY,
        };
        let lp = self.log_prob_constrained(&params);
        if lp == f64::NEG_INFINITY {
            return lp;
        }
        lp + log_unconstrained_jacobian(theta)
    }
}

impl<'a> LogDensity for WienerLikelihood<'a> {
    #[inline]
    fn log_prob(&self, x: &[f64]) -> f64 {
        self.log_prob_unconstrained(x)
    }
}

impl<'a> BatchLogDensity for WienerLikelihood<'a> {
    #[inline]
    fn log_prob_batch(&self, batch: &[&[f64]], out: &mut [f64]) {
        debug_assert_eq!(batch.len(), out.len());
        for (theta, slot) in batch.iter().zip(out.iter_mut()) {
            *slot = self.log_prob_unconstrained(theta);
        }
    }
}

pub struct WienerCseLikelihood<'a> {
    pub trials: &'a [CseTrial],
}

impl<'a> WienerCseLikelihood<'a> {
    #[inline]
    pub fn new(trials: &'a [CseTrial]) -> Self {
        Self { trials }
    }

    #[inline]
    pub fn log_prob_constrained(&self, params: &CseParams) -> f64 {
        log_likelihood_cse(self.trials, params)
    }

    #[inline]
    pub fn log_prob_unconstrained(&self, theta: &[f64]) -> f64 {
        let params = match decode_cse_params(theta) {
            Some(p) => p,
            None => return f64::NEG_INFINITY,
        };
        let lp = self.log_prob_constrained(&params);
        if lp == f64::NEG_INFINITY {
            return lp;
        }
        lp + log_jacobian(theta[4], theta[5], theta[6])
    }
}

impl<'a> LogDensity for WienerCseLikelihood<'a> {
    #[inline]
    fn log_prob(&self, x: &[f64]) -> f64 {
        self.log_prob_unconstrained(x)
    }
}

impl<'a> BatchLogDensity for WienerCseLikelihood<'a> {
    #[inline]
    fn log_prob_batch(&self, batch: &[&[f64]], out: &mut [f64]) {
        debug_assert_eq!(batch.len(), out.len());
        for (theta, slot) in batch.iter().zip(out.iter_mut()) {
            *slot = self.log_prob_unconstrained(theta);
        }
    }
}

// Observation-model adapters for single-trial kernels / SSM composition.

pub struct WienerObservation;

impl ObservationModel<DdmParams, DdmTrial> for WienerObservation {
    #[inline]
    fn log_likelihood(&self, state: &DdmParams, obs: &DdmTrial) -> f64 {
        log_trial_ll(obs, state)
    }
}

pub struct WienerCseObservation;

impl ObservationModel<CseParams, CseTrial> for WienerCseObservation {
    #[inline]
    fn log_likelihood(&self, state: &CseParams, obs: &CseTrial) -> f64 {
        let p = state.for_condition(obs.condition);
        log_trial_ll(&obs.obs, &p)
    }
}

// ============================================================================
// Optional condition map abstraction
// ============================================================================

pub trait DriftByCondition {
    fn drift(&self, condition: TrialCondition) -> f64;
}

impl DriftByCondition for [f64; 4] {
    #[inline]
    fn drift(&self, condition: TrialCondition) -> f64 {
        self[condition.index()]
    }
}

impl DriftByCondition for CseParams {
    #[inline]
    fn drift(&self, condition: TrialCondition) -> f64 {
        self.drift[condition.index()]
    }
}

#[inline]
pub fn log_likelihood_cse_generic<D: DriftByCondition>(
    trials: &[CseTrial],
    drift: &D,
    a: f64,
    w: f64,
    t_er: f64,
) -> f64 {
    let mut ll = 0.0;
    for tr in trials {
        let p = match DdmParams::new(drift.drift(tr.condition), a, w, t_er) {
            Some(p) => p,
            None => return f64::NEG_INFINITY,
        };
        let c = log_trial_ll(&tr.obs, &p);
        if c == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        ll += c;
    }
    ll
}

// ============================================================================
// Small utility templates for model composition
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct DdmLink {
    pub a: f64,
    pub w: f64,
    pub t_er: f64,
}

impl DdmLink {
    #[inline]
    pub fn params(&self, v: f64) -> Option<DdmParams> {
        DdmParams::new(v, self.a, self.w, self.t_er)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CseLink {
    pub a: f64,
    pub w: f64,
    pub t_er: f64,
}

impl CseLink {
    #[inline]
    pub fn params(&self, drift: [f64; 4]) -> Option<CseParams> {
        CseParams::new(drift, self.a, self.w, self.t_er)
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    #[test]
    fn cse_drift_lookup_is_stable() {
        let p = CseParams::new([1.0, 2.0, 3.0, 4.0], 1.5, 0.5, 0.2).unwrap();
        assert_eq!(p.for_condition(TrialCondition::CC).v, 1.0);
        assert_eq!(p.for_condition(TrialCondition::II).v, 4.0);
    }

    #[test]
    fn batch_api_is_pointwise() {
        let trials = [DdmTrial {
            rt: 0.6,
            response: Response::Upper,
        }];
        let model = WienerLikelihood::new(&trials);
        let theta = [1.0, 1.5f64.ln(), logit(0.5), 0.2f64.ln()];
        let mut out = [0.0];
        let batch = [&theta[..]];
        model.log_prob_batch(&batch, &mut out);
        assert!(out[0].is_finite() || out[0].is_sign_negative());
    }

    /// Brute-force Monte Carlo estimate of the upper-boundary density at time t.
    /// Used only in tests to cross-check the series formula.
    fn mc_density_upper(t: f64, v: f64, a: f64, w: f64, n: usize, dt: f64) -> f64 {
        use std::collections::BTreeMap;

        let steps = (t / dt).round() as usize;
        let target_step = steps;
        let mut count = 0usize;
        let mut total = 0usize;

        // Simple Euler-Maruyama, seeded deterministically
        // We use a simple LCG for reproducibility without extra deps.
        let mut seed: u64 = 0xDEAD_BEEF_1234_5678;
        let lcg = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f64) / (u32::MAX as f64)
        };

        for _ in 0..n {
            total += 1;
            let mut x = w * a;
            let mut hit_upper_at: Option<usize> = None;
            for step in 1..=(target_step + target_step / 2) {
                let z = {
                    // Box-Muller
                    let u1 = lcg(&mut seed).max(1e-15);
                    let u2 = lcg(&mut seed);
                    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
                };
                x += v * dt + dt.sqrt() * z;
                if x >= a {
                    hit_upper_at = Some(step);
                    break;
                } else if x <= 0.0 {
                    break;
                }
            }
            // Count trials that hit upper within [t-dt/2, t+dt/2]
            if let Some(hit) = hit_upper_at {
                if hit == target_step {
                    count += 1;
                }
            }
        }

        (count as f64) / (total as f64 * dt)
    }

    // ── parameter invariants ─────────────────────────────────────────────────

    #[test]
    fn ddm_params_new_valid() {
        let p = DdmParams::new(1.0, 1.5, 0.5, 0.3).unwrap();
        assert_eq!(p.v, 1.0);
        assert_eq!(p.a, 1.5);
        assert_eq!(p.w, 0.5);
        assert_eq!(p.t_er, 0.3);
    }

    #[test]
    fn ddm_params_new_rejects_invalid_a() {
        assert!(DdmParams::new(0.0, -1.0, 0.5, 0.3).is_none());
        assert!(DdmParams::new(0.0, 0.0, 0.5, 0.3).is_none());
        assert!(DdmParams::new(0.0, f64::INFINITY, 0.5, 0.3).is_none());
    }

    #[test]
    fn ddm_params_new_rejects_invalid_w() {
        assert!(DdmParams::new(0.0, 1.5, 0.0, 0.3).is_none());
        assert!(DdmParams::new(0.0, 1.5, 1.0, 0.3).is_none());
        assert!(DdmParams::new(0.0, 1.5, -0.1, 0.3).is_none());
        assert!(DdmParams::new(0.0, 1.5, 1.1, 0.3).is_none());
    }

    #[test]
    fn ddm_params_new_rejects_invalid_ter() {
        assert!(DdmParams::new(0.0, 1.5, 0.5, 0.0).is_none());
        assert!(DdmParams::new(0.0, 1.5, 0.5, -0.1).is_none());
    }

    #[test]
    fn decision_time_ok() {
        let p = DdmParams::new(1.0, 1.5, 0.5, 0.3).unwrap();
        assert_eq!(p.decision_time(0.8), Some(0.5));
        assert!(p.decision_time(0.3).is_none()); // rt == t_er
        assert!(p.decision_time(0.2).is_none()); // rt < t_er
    }

    // ── series selection: density must be strictly positive ──────────────────

    #[test]
    fn log_wiener_upper_finite_in_both_regimes() {
        // Large-time regime (tau = t/a² > 0.65)
        let lp_large = log_wiener_upper(2.0, 1.0, 1.5, 0.5);
        assert!(lp_large.is_finite(), "large-time: {lp_large}");
        assert!(lp_large < 0.0, "density < 1 means log < 0");

        // Small-time regime (tau = t/a² < 0.65)
        let lp_small = log_wiener_upper(0.5, 1.0, 1.5, 0.5);
        assert!(lp_small.is_finite(), "small-time: {lp_small}");
    }

    #[test]
    fn log_wiener_upper_invalid_returns_neg_inf() {
        assert_eq!(log_wiener_upper(-0.1, 1.0, 1.5, 0.5), f64::NEG_INFINITY); // t ≤ 0
        assert_eq!(log_wiener_upper(0.5, 1.0, 0.0, 0.5), f64::NEG_INFINITY); // a = 0
        assert_eq!(log_wiener_upper(0.5, 1.0, 1.5, 0.0), f64::NEG_INFINITY); // w = 0
        assert_eq!(log_wiener_upper(0.5, 1.0, 1.5, 1.0), f64::NEG_INFINITY); // w = 1
    }

    #[test]
    fn lower_upper_symmetry() {
        let t = 0.8;
        let v = 1.5;
        let a = 2.0;
        let w = 0.4;
        let lp_lower = log_wiener_lower(t, v, a, w);
        let lp_flipped = log_wiener_upper(t, -v, a, 1.0 - w);
        let diff = (lp_lower - lp_flipped).abs();
        assert!(diff < 1e-12, "symmetry violated: diff = {diff}");
    }

    // ── zero drift: density should be symmetric in RT under w = 0.5 ─────────

    #[test]
    fn zero_drift_symmetry_with_unbiased_start() {
        // With v = 0, w = 0.5: upper and lower densities are identical at same t
        let t = 0.6;
        let a = 1.5;
        let w = 0.5;
        let lp_up = log_wiener_upper(t, 0.0, a, w);
        let lp_low = log_wiener_lower(t, 0.0, a, w);
        let diff = (lp_up - lp_low).abs();
        assert!(diff < 1e-10, "zero-drift symmetry: diff = {diff}");
    }

    // ── large-time and small-time series agree near the threshold ─────────────

    #[test]
    fn series_agree_near_threshold() {
        // At tau ≈ TAU_THRESHOLD, both series should produce nearly identical
        // values (verifies the switch point is not discontinuous).
        let t_large = TAU_THRESHOLD * 1.5 * 1.5 + 0.01; // tau = 0.68, large-time
        let t_small = TAU_THRESHOLD * 1.5 * 1.5 - 0.01; // tau = 0.65, small-time
        let v = 1.0;
        let a = 1.5;
        let w = 0.5;

        let lp_large = log_wiener_upper(t_large, v, a, w);
        let lp_small = log_wiener_upper(t_small, v, a, w);

        // Both should be finite and close to each other (density is smooth)
        assert!(lp_large.is_finite());
        assert!(lp_small.is_finite());
        let diff = (lp_large - lp_small).abs();
        assert!(
            diff < 2.0,
            "expected continuous density near threshold, diff = {diff}"
        );
    }

    // ── density integrates to ≤ 1 (numerical marginalisation) ────────────────
    //
    // The *joint* probability of hitting the upper boundary by any time is
    // ≤ 1.  We verify this by numerical quadrature over a grid.

    #[test]
    fn density_upper_integrates_below_one() {
        let v = 2.0;
        let a = 1.5;
        let w = 0.5;
        let dt = 0.005;
        let t_max = 4.0;
        let mut integral = 0.0f64;
        let mut t = dt;
        while t <= t_max {
            integral += dt * log_wiener_upper(t, v, a, w).exp();
            t += dt;
        }
        assert!(
            integral <= 1.0 + 1e-4,
            "upper integral = {integral} exceeds 1"
        );
        assert!(integral > 0.0);
    }

    // ── upper + lower densities integrate to ≤ 1 ─────────────────────────────

    #[test]
    fn total_density_integrates_below_one() {
        let v = 1.0;
        let a = 1.5;
        let w = 0.5;
        let dt = 0.005;
        let t_max = 6.0;
        let mut integral = 0.0f64;
        let mut t = dt;
        while t <= t_max {
            integral +=
                dt * (log_wiener_upper(t, v, a, w).exp() + log_wiener_lower(t, v, a, w).exp());
            t += dt;
        }
        // Should be close to 1 for long enough t_max
        assert!(integral <= 1.0 + 1e-3, "total integral = {integral}");
        assert!(integral > 0.9, "total integral too low: {integral}");
    }

    // ── trial log-likelihood: invalid RT returns NEG_INFINITY ────────────────

    #[test]
    fn log_trial_ll_invalid_rt() {
        let params = DdmParams::new(1.0, 1.5, 0.5, 0.3).unwrap();
        let bad_trial = DdmTrial {
            rt: 0.2,
            response: Response::Upper,
        };
        assert_eq!(log_trial_ll(&bad_trial, &params), f64::NEG_INFINITY);
    }

    // ── log_likelihood monotone in drift ─────────────────────────────────────
    //
    // A dataset of all-correct (upper) trials with a large positive drift
    // should have higher log-likelihood than the same dataset with a zero drift.

    #[test]
    fn log_likelihood_higher_for_correct_drift() {
        let trials: Vec<DdmTrial> = (0..20)
            .map(|i| DdmTrial {
                rt: 0.4 + 0.02 * (i as f64),
                response: Response::Upper,
            })
            .collect();

        let params_strong = DdmParams::new(3.0, 1.5, 0.5, 0.2).unwrap();
        let params_zero = DdmParams::new(0.0, 1.5, 0.5, 0.2).unwrap();

        let ll_strong = log_likelihood(&trials, &params_strong);
        let ll_zero = log_likelihood(&trials, &params_zero);

        assert!(ll_strong.is_finite());
        assert!(
            ll_strong > ll_zero,
            "strong drift should fit all-upper data better: {ll_strong} > {ll_zero}"
        );
    }

    // ── trial condition index is bijective ───────────────────────────────────

    #[test]
    fn trial_condition_index_bijective() {
        let all = TrialCondition::all();
        let indices: Vec<usize> = all.iter().map(|c| c.index()).collect();
        // All four indices are distinct
        let unique: std::collections::HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), 4);
        // Indices are in range 0..4
        assert!(indices.iter().all(|&i| i < 4));
    }

    // ── CSE log-likelihood: per-condition drift is identified ─────────────────

    #[test]
    fn cse_log_likelihood_condition_specific() {
        // Two conditions: CC (easy, high drift) and II (hard, low drift)
        // All CC trials are upper (correct), all II trials are lower (error)
        let cc_trials: Vec<CseTrial> = (0..10)
            .map(|i| CseTrial {
                condition: TrialCondition::CC,
                obs: DdmTrial {
                    rt: 0.4 + 0.01 * (i as f64),
                    response: Response::Upper,
                },
            })
            .collect();
        let ii_trials: Vec<CseTrial> = (0..10)
            .map(|i| CseTrial {
                condition: TrialCondition::II,
                obs: DdmTrial {
                    rt: 0.5 + 0.01 * (i as f64),
                    response: Response::Lower,
                },
            })
            .collect();

        let trials: Vec<CseTrial> = cc_trials.into_iter().chain(ii_trials).collect();

        // v_CC high (favours upper), v_II negative (favours lower)
        let v_good = [3.0f64, 1.0, 1.0, -2.0];
        let v_bad = [0.0f64, 0.0, 0.0, 0.0];

        let p_good = CseParams::new(v_good, 1.5, 0.5, 0.2)?;
        let p_bad = CseParams::new(v_bad, 1.5, 0.5, 0.2)?;

        let ll_good = log_likelihood_cse(&trials, &p_good);
        let ll_bad = log_likelihood_cse(&trials, &p_bad);

        assert!(ll_good.is_finite());
        assert!(
            ll_good > ll_bad,
            "condition-specific drift should give higher ll: {ll_good} > {ll_bad}"
        );
    }

    // ── parameter transforms are inverses of each other ───────────────────────

    #[test]
    fn sigmoid_logit_roundtrip() {
        for &p in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let roundtrip = sigmoid(logit(p));
            assert!(
                (roundtrip - p).abs() < 1e-12,
                "sigmoid(logit({p})) = {roundtrip}"
            );
        }
    }

    #[test]
    fn decode_params_roundtrip() {
        let v = 1.5;
        let a = 2.0;
        let w = 0.4;
        let t_er = 0.3;
        let x = [v, a.ln(), logit(w), t_er.ln()];
        let p = decode_params(&x).unwrap();
        assert!((p.v - v).abs() < 1e-12);
        assert!((p.a - a).abs() < 1e-12);
        assert!((p.w - w).abs() < 1e-12);
        assert!((p.t_er - t_er).abs() < 1e-12);
    }

    #[test]
    fn decode_cse_params_roundtrip() {
        let drift = [1.5f64, 0.8, 1.2, -0.5];
        let a = 2.0;
        let w = 0.4;
        let t_er = 0.3;
        let theta = [
            drift[0],
            drift[1],
            drift[2],
            drift[3],
            a.ln(),
            logit(w),
            t_er.ln(),
        ];
        let params = decode_cse_params(&theta).unwrap();
        for i in params.drift {
            assert!((params.drift[i] - drift[i]).abs() < 1e-12);
        }
        assert!((params.a - a).abs() < 1e-12);
        assert!((params.w - w).abs() < 1e-12);
        assert!((params.t_er - t_er).abs() < 1e-12);
    }

    // ── log_jacobian: correct for exp-constrained parameters ─────────────────

    #[test]
    fn log_jacobian_correct() {
        let a = 2.0;
        let w = 0.4;
        let t_er = 0.3;
        let theta_a = a.ln();
        let theta_w = logit(w);
        let theta_ter = t_er.ln();

        let expected = a.ln() + w.ln() + (1.0 - w).ln() + t_er.ln();
        let computed = log_jacobian(theta_a, theta_w, theta_ter);
        assert!((computed - expected).abs() < 1e-12);
    }
}
