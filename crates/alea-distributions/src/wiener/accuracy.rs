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

// Relative fused-work units fitted to CPU-pinned scalar/nightly measurements.
// Both point evaluators use the same scalar finite sums in SIMD builds; vector
// batch kernels do not call this planner. Positive affine costs make pruning
// safe: no larger term count can have a lower predicted cost.
fn branch_cost_parameters(branch: SeriesBranch) -> (usize, usize) {
    match branch {
        SeriesBranch::SmallTime => (5, if cfg!(feature = "simd") { 5 } else { 6 }),
        SeriesBranch::LargeTime => (if cfg!(feature = "simd") { 7 } else { 9 }, 3),
    }
}

fn branch_cost(branch: SeriesBranch, terms: usize) -> usize {
    let (base, slope) = branch_cost_parameters(branch);
    base.saturating_add(slope.saturating_mul(terms))
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
    let cost = |p: &Plan| branch_cost(p.branch, p.terms);
    let cap = first
        .as_ref()
        .map_or(128, |p| {
            let (base, slope) = branch_cost_parameters(other);
            cost(p).saturating_sub(base) / slope
        })
        .min(128);
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

#[cfg(test)]
mod calibration {
    use super::*;
    #[test]
    #[ignore = "CPU-pinned release timing calibration, not a CI assertion"]
    fn fused_branch_costs() {
        use std::{hint::black_box, time::Instant};
        for k in [1, 2, 3, 4, 6, 8, 12, 16] {
            for branch in [SeriesBranch::SmallTime, SeriesBranch::LargeTime] {
                let mut samples = Vec::new();
                for _ in 0..11 {
                    let start = Instant::now();
                    for i in 0..10000 {
                        let w = black_box([0.2, 0.3, 0.5, 0.7][i % 4]);
                        let u = black_box(0.15);
                        black_box(match branch {
                            SeriesBranch::SmallTime => {
                                series::small_branch_fused(u, w, black_box(k))
                            }
                            SeriesBranch::LargeTime => {
                                series::large_branch_fused(u, w, black_box(k))
                            }
                        });
                    }
                    samples.push(start.elapsed().as_nanos() as f64 / 10000.);
                }
                samples.sort_by(f64::total_cmp);
                eprintln!("branch_calibration,{branch:?},{k},{}", samples[5]);
            }
        }
    }
}

#[cfg(test)]
mod planning_tests {
    use super::*;
    #[test]
    fn selected_common_count_minimizes_predicted_fused_cost() {
        for u in [0.001, 0.03, 0.1, 0.15, 0.3, 1., 10.] {
            for beta in [1e-8, 0.1, 0.5, 0.9, 1. - 1e-8] {
                let obs = WienerObservation {
                    rt: u,
                    boundary: Boundary::Upper,
                };
                let params = Wiener4Params::from_array([1., 0., beta, 0.7]);
                let core = Wiener4.core(&obs, &params, 1e-9).unwrap();
                let chosen = plan(&core, 1e-9).unwrap();
                for branch in [SeriesBranch::SmallTime, SeriesBranch::LargeTime] {
                    if let Some(other) = candidate(&core, branch, 128, 1e-9) {
                        assert!(
                            branch_cost(chosen.branch, chosen.terms)
                                <= branch_cost(other.branch, other.terms)
                        );
                    }
                }
            }
        }
    }
}
