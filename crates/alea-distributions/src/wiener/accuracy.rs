//! Optional componentwise truncation control, separate from the fast evaluator.
use super::*;

/// Real-arithmetic truncation bounds for a finite Wiener series evaluation.
///
/// Parameter scores follow `grad_array()` order. These are absolute errors.
/// Floating-point rounding and transcendental-function error are **not** included;
/// this report is not an end-to-end numerical certificate.
#[derive(Clone, Copy, Debug)]
pub struct WienerTruncationError<const N: usize> {
    pub log_prob: f64,
    pub scores: [f64; N],
    pub branch: SeriesBranch,
    /// Spectral harmonics or image pairs retained by the selected kernel.
    pub terms: usize,
}

struct Plan {
    branch: SeriesBranch,
    terms: usize,
    error: bounds::RatioErrors,
}

fn candidate(
    core: &Wiener4Core,
    branch: SeriesBranch,
    limit: usize,
    tolerance: f64,
) -> Option<Plan> {
    let u = core.t_prime;
    for k in usize::from(branch == SeriesBranch::LargeTime)..=limit {
        let evaluation = if core.w_eff.min(core.q_eff) < 1e-5 {
            series::endpoint_fused(u, core.w_eff, core.q_eff, k, branch)
        } else {
            match branch {
                SeriesBranch::SmallTime => series::small_branch_fused(u, core.w_eff, k),
                SeriesBranch::LargeTime => series::large_branch_fused(u, core.w_eff, k),
            }
        };
        let Some((log_f, dt, dw)) = evaluation else {
            continue;
        };
        let scale = match branch {
            SeriesBranch::LargeTime => PI.ln() - 0.5 * PI * PI * u,
            SeriesBranch::SmallTime => {
                -0.5 * (2. * PI).ln() - 1.5 * u.ln() - core.q_eff * core.q_eff / (2. * u)
            }
        };
        let Some(tail) = bounds::tails(branch, u, core.w_eff, core.q_eff, k) else {
            continue;
        };
        let Some(error) = bounds::ratio_errors(log_f - scale, dt, dw, tail) else {
            continue;
        };
        let alpha = (2. * u / core.a).abs() * error.time;
        let tau = error.time / core.a / core.a;
        if [error.log, alpha, tau, error.start]
            .iter()
            .all(|x| x.is_finite() && *x <= tolerance)
        {
            return Some(Plan {
                branch,
                terms: k,
                error,
            });
        }
    }
    None
}

fn plan(core: &Wiener4Core, tolerance: f64) -> Result<Plan> {
    if !tolerance.is_finite() || tolerance <= 0. {
        return Err(ProbError::InvalidParameters(
            "Wiener truncation tolerance must be finite and positive".into(),
        ));
    }
    let small = Wiener4::k_s(core.t_prime, core.w_eff, core.log_eps_eff);
    let large = Wiener4::k_l(core.t_prime, core.log_eps_eff);
    let preferred = if small.saturating_mul(2) <= large {
        SeriesBranch::SmallTime
    } else {
        SeriesBranch::LargeTime
    };
    let other = if preferred == SeriesBranch::SmallTime {
        SeriesBranch::LargeTime
    } else {
        SeriesBranch::SmallTime
    };
    let first = candidate(core, preferred, 128, tolerance);
    // Two image exponentials versus one spectral harmonic is an explicit work
    // estimate, not a claim about measured latency or a floating-point proof.
    let cost = |p: &Plan| {
        p.terms
            .saturating_mul(if p.branch == SeriesBranch::SmallTime {
                2
            } else {
                1
            })
    };
    let cap = first.as_ref().map_or(128, |p| {
        cost(p)
            / (if other == SeriesBranch::SmallTime {
                2
            } else {
                1
            })
    });
    let second = candidate(core, other, cap, tolerance);
    match (first, second) {
        (Some(a), Some(b)) => Ok(if cost(&b) < cost(&a) { b } else { a }),
        (Some(p), None) | (None, Some(p)) => Ok(p),
        (None, None) => Err(ProbError::NumericalError(
            "Wiener componentwise truncation budget was not met within 128 terms".into(),
        )),
    }
}

fn report<const N: usize>(core: &Wiener4Core, p: &Plan) -> WienerTruncationError<N> {
    let mut scores = [0.; N];
    scores[0] = (2. * core.t_prime / core.a).abs() * p.error.time;
    scores[1] = p.error.time / core.a / core.a;
    scores[2] = p.error.start;
    WienerTruncationError {
        log_prob: p.error.log,
        scores,
        branch: p.branch,
        terms: p.terms,
    }
}

impl Wiener4 {
    /// Evaluate with a common term count controlling log density and all scores.
    ///
    /// `tolerance` is an absolute real-arithmetic truncation budget for each
    /// returned component. Roundoff is excluded. The ordinary fast path retains
    /// its existing precision convention. Returns a numerical error if neither
    /// representation meets this budget within 128 terms.
    pub fn try_fused_with_truncation(
        &self,
        obs: &WienerObservation,
        params: &Wiener4Params,
        tolerance: f64,
    ) -> Result<(Wiener4Eval, WienerTruncationError<4>)> {
        let core = self.core(obs, params, tolerance)?;
        let p = plan(&core, tolerance)?;
        let e = self.fused_from_core_with_branch(&core, p.branch, p.terms);
        if !e.log_prob.is_finite() || e.grad_array().iter().any(|x| !x.is_finite()) {
            return Err(ProbError::NumericalError(
                "Non-finite truncation-controlled Wiener4 evaluation".into(),
            ));
        }
        Ok((e, report(&core, &p)))
    }
}

impl Wiener5 {
    /// Wiener5 counterpart of [`Wiener4::try_fused_with_truncation`].
    /// Gaussian drift marginalization is analytic; its score has no series tail.
    pub fn try_fused_with_truncation(
        &self,
        obs: &WienerObservation,
        params: &Wiener5Params,
        tolerance: f64,
    ) -> Result<(Wiener5Eval, WienerTruncationError<5>)> {
        let core = self.core(obs, params, tolerance)?;
        let p = plan(&core.base, tolerance)?;
        let e = self.fused_from_core_with_branch(&core, p.branch, p.terms);
        if !e.log_prob.is_finite() || e.grad_array().iter().any(|x| !x.is_finite()) {
            return Err(ProbError::NumericalError(
                "Non-finite truncation-controlled Wiener5 evaluation".into(),
            ));
        }
        Ok((e, report(&core.base, &p)))
    }
}
