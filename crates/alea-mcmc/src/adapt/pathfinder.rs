//! Bounded multi-path Pathfinder initialization.
//!
//! This is an initialization service, not an MCMC transition. Independent
//! L-BFGS paths select local Gaussian approximations by Monte Carlo ELBO and pool
//! their draws with explicitly selected raw-mixture or pathwise PSIS weights.
//! Numerical recovery is opt-in; concrete backend failures retain their type.

use std::{error::Error, num::NonZeroUsize};

use alea_core::target::{EvaluationError, LogDensityGradient, evaluate};
use alea_math::{buffer::OwnedBuffer, random::standard_normal};
use rand::{Rng, RngExt};
use thiserror::Error;

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;
mod psis;

/// Explicitly different weighting conventions for multi-path initialization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImportanceResampling {
    /// Raw ratios against the complete equally weighted Gaussian mixture.
    #[default]
    RawMixture,
    /// Pareto smoothing of each draw's generating-path importance ratio,
    /// following the Stan2.38 multi-path workflow.
    PsisPathwise,
}
/// Whether the requested PSIS calculation fitted a usable nondegenerate tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsisStatus {
    /// Raw importance weighting or a non-importance initializer was selected.
    NotRequested,
    /// A generalized Pareto tail was fitted and smoothed.
    Smoothed,
    /// Fewer than five tail observations; normalized weights are unsmoothed.
    InsufficientTail,
    /// Tail ties or unrepresentable positive excesses prevent a tail fit.
    DegenerateTail,
}
/// A numerical path outcome; concrete backend failures still abort with their type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathfinderPathStatus {
    /// A finite local approximation survived.
    Completed,
    /// No finite local approximation survived the bounded path.
    NoApproximation,
    /// Local optimization/factorization arithmetic or successful target outputs
    /// were nonfinite; concrete backend errors still propagate.
    NumericalFailure,
}
/// Actual work and outcome for one requested path, including failed paths.
#[derive(Debug, Clone, Copy)]
pub struct PathfinderPathReport {
    /// Zero-based path identity.
    pub path: usize,
    /// Numerical completion or failure reason.
    pub status: PathfinderPathStatus,
    /// Accepted optimizer steps on this path, including a later failed path.
    pub optimization_steps: usize,
    /// Fused evaluations attempted on this path, including ELBO trials.
    pub target_evaluations: usize,
}

/// Invalid bounded Pathfinder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid multi-path Pathfinder configuration")]
pub struct PathfinderConfigError;

/// Bounded work and memory policy for multi-path Pathfinder.
#[derive(Debug, Clone, Copy)]
pub struct MultiPathfinderOptions {
    paths: NonZeroUsize,
    max_iterations: NonZeroUsize,
    history: NonZeroUsize,
    elbo_draws: NonZeroUsize,
    draws_per_path: NonZeroUsize,
    max_line_search: NonZeroUsize,
    relative_tolerance: f64,
    resampling: ImportanceResampling,
    recover_numerical: bool,
}

impl MultiPathfinderOptions {
    /// Constructs a policy with L-BFGS history 5, four ELBO draws, four output
    /// draws per path, twenty line-search evaluations and tolerance `1e-8`.
    pub fn new(paths: usize, max_iterations: usize) -> Result<Self, PathfinderConfigError> {
        Ok(Self {
            paths: NonZeroUsize::new(paths).ok_or(PathfinderConfigError)?,
            max_iterations: NonZeroUsize::new(max_iterations).ok_or(PathfinderConfigError)?,
            history: NonZeroUsize::new(5).expect("five is nonzero"),
            elbo_draws: NonZeroUsize::new(4).expect("four is nonzero"),
            draws_per_path: NonZeroUsize::new(4).expect("four is nonzero"),
            max_line_search: NonZeroUsize::new(20).expect("twenty is nonzero"),
            relative_tolerance: 1e-8,
            resampling: ImportanceResampling::RawMixture,
            recover_numerical: false,
        })
    }

    /// Changes the L-BFGS secant history length.
    pub fn with_history(mut self, history: usize) -> Result<Self, PathfinderConfigError> {
        self.history = NonZeroUsize::new(history).ok_or(PathfinderConfigError)?;
        Ok(self)
    }

    /// Changes Monte Carlo draws used for each local ELBO estimate.
    pub fn with_elbo_draws(mut self, draws: usize) -> Result<Self, PathfinderConfigError> {
        self.elbo_draws = NonZeroUsize::new(draws).ok_or(PathfinderConfigError)?;
        Ok(self)
    }

    /// Changes proposal draws retained from each selected path approximation.
    pub fn with_draws_per_path(mut self, draws: usize) -> Result<Self, PathfinderConfigError> {
        self.draws_per_path = NonZeroUsize::new(draws).ok_or(PathfinderConfigError)?;
        Ok(self)
    }

    /// Changes the relative score-norm convergence threshold.
    pub fn with_relative_tolerance(
        mut self,
        tolerance: f64,
    ) -> Result<Self, PathfinderConfigError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PathfinderConfigError);
        }
        self.relative_tolerance = tolerance;
        Ok(self)
    }

    /// Select a declared robust workflow. Numerical candidate failures backtrack,
    /// unsuccessful numerical paths remain recorded, and concrete backend errors
    /// still propagate. Calling the legacy constructor alone preserves its strict
    /// all-paths-successful raw policy for source-bound historical reproduction.
    pub fn with_resampling(mut self, resampling: ImportanceResampling) -> Self {
        self.resampling = resampling;
        self.recover_numerical = true;
        self
    }
}

/// Bounded score-ascent baseline used for full initialization-cost comparisons.
#[derive(Debug, Clone, Copy)]
pub struct ScoreInitializationOptions {
    /// Maximum accepted ascent iterations.
    iterations: usize,
    /// Initial Armijo trial step.
    initial_step: f64,
    /// Maximum target evaluations per line search.
    max_line_search: usize,
}

impl ScoreInitializationOptions {
    /// Validates the baseline work policy.
    pub fn new(
        iterations: usize,
        initial_step: f64,
        max_line_search: usize,
    ) -> Result<Self, PathfinderConfigError> {
        if iterations == 0
            || !initial_step.is_finite()
            || initial_step <= 0.0
            || max_line_search == 0
        {
            return Err(PathfinderConfigError);
        }
        Ok(Self {
            iterations,
            initial_step,
            max_line_search,
        })
    }
}

/// Target or bounded numerical failure during initialization.
#[derive(Debug, Error)]
pub enum PathfinderError<E: Error + 'static> {
    /// Input path storage or options do not match the target.
    #[error("invalid multi-path Pathfinder input shape")]
    Configuration,
    /// A checked target evaluation failed on one optimization path.
    #[error("Pathfinder target evaluation failed on path {path}: {source}")]
    Evaluation {
        /// Zero-based path index.
        path: usize,
        /// Preserved target/backend failure.
        #[source]
        source: EvaluationError<E>,
    },
    /// No finite positive-definite local approximation survived.
    #[error("Pathfinder produced no finite local StandardNormalTarget approximation")]
    NoApproximation,
    /// Counting, factorization, weighting or optimization arithmetic failed.
    #[error("Pathfinder failed numerically")]
    Numerical,
}

/// Exact initialization work performed.
#[derive(Debug, Clone, Copy)]
pub struct PathfinderReport {
    /// Independent optimization paths requested and completed.
    pub paths: usize,
    /// Accepted L-BFGS or score-ascent iterations.
    pub optimization_steps: usize,
    /// Actual fused target/score evaluations, including ELBO and pool draws.
    pub target_evaluations: usize,
    /// Attempted Gaussian pool draws, including unrepresentable coordinates.
    pub proposal_draws: usize,
    /// Raw mixture importance resampling was used (never PSIS by implication).
    pub raw_importance_resampling: bool,
    /// Requested path count, including unsuccessful numerical paths.
    pub requested_paths: usize,
    /// Proposal draws with zero mass or unrepresentable coordinates.
    pub invalid_proposal_draws: usize,
    /// Effective size of normalized importance weights, when used.
    pub importance_ess: Option<f64>,
    /// Fitted Pareto shape; absent when no tail fit was performed.
    pub pareto_shape: Option<f64>,
    /// Tail observations selected for PSIS.
    pub psis_tail_length: usize,
    /// Exact smoothing outcome, including explicitly unsmoothed small/tied tails.
    pub psis_status: PsisStatus,
}

/// One initialization point plus complete work accounting.
#[derive(Debug)]
pub struct PathfinderResult {
    /// Selected point in the target's coordinates.
    pub position: OwnedBuffer,
    /// Bounded initialization work.
    pub report: PathfinderReport,
    /// Per-path numerical outcomes; backend failures remain typed errors.
    pub paths: Vec<PathfinderPathReport>,
}

#[derive(Clone)]
struct Approximation {
    mean: Vec<f64>,
    lower: Vec<f64>,
    log_det: f64,
    elbo: f64,
}

struct LocalApproximationInput<'a> {
    position: &'a [f64],
    score: &'a [f64],
    s_history: &'a [Vec<f64>],
    y_history: &'a [Vec<f64>],
    elbo_draws: usize,
    path: usize,
    recover_numerical: bool,
}

/// Runs independent bounded paths and the declared raw/PSIS resampling policy.
///
/// `initial_positions` is path-major and must contain exactly
/// `options.paths * target.dimension()` finite coordinates.
pub fn multi_pathfinder<T, R>(
    target: &T,
    initial_positions: &[f64],
    options: MultiPathfinderOptions,
    rng: &mut R,
) -> Result<PathfinderResult, PathfinderError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    R: Rng + ?Sized,
{
    let dimension = target.dimension();
    if dimension == 0
        || dimension.checked_mul(options.paths.get()) != Some(initial_positions.len())
        || initial_positions.iter().any(|value| !value.is_finite())
    {
        return Err(PathfinderError::Configuration);
    }
    let mut evaluations = 0_usize;
    let mut optimization_steps = 0_usize;
    let mut approximations = Vec::with_capacity(options.paths.get());
    let mut paths = Vec::with_capacity(options.paths.get());
    for path in 0..options.paths.get() {
        let before = evaluations;
        let before_steps = optimization_steps;
        let start = path * dimension;
        match single_path(
            target,
            &initial_positions[start..start + dimension],
            options,
            rng,
            path,
            &mut evaluations,
            &mut optimization_steps,
        ) {
            Ok((approximation, _)) => {
                approximations.push(approximation);
                paths.push(PathfinderPathReport {
                    path,
                    status: PathfinderPathStatus::Completed,
                    optimization_steps: optimization_steps - before_steps,
                    target_evaluations: evaluations - before,
                });
            }
            Err(error)
                if options.recover_numerical
                    && matches!(
                        error,
                        PathfinderError::NoApproximation
                            | PathfinderError::Numerical
                            | PathfinderError::Evaluation {
                                source: EvaluationError::NonFiniteLogDensity
                                    | EvaluationError::NonFiniteGradient { .. },
                                ..
                            }
                    ) =>
            {
                let status = if matches!(error, PathfinderError::NoApproximation) {
                    PathfinderPathStatus::NoApproximation
                } else {
                    PathfinderPathStatus::NumericalFailure
                };
                paths.push(PathfinderPathReport {
                    path,
                    status,
                    optimization_steps: optimization_steps - before_steps,
                    target_evaluations: evaluations - before,
                });
            }
            Err(error) => return Err(error),
        }
    }
    if approximations.is_empty() {
        return Err(PathfinderError::NoApproximation);
    }
    let pool_size = approximations
        .len()
        .checked_mul(options.draws_per_path.get())
        .ok_or(PathfinderError::Numerical)?;
    let mut pool = Vec::with_capacity(pool_size);
    let mut weights = Vec::with_capacity(pool_size);
    let mut gradient = vec![0.; dimension];
    let mut invalid_proposal_draws = 0;
    for approximation in &approximations {
        for _ in 0..options.draws_per_path.get() {
            let (draw, logq) = match draw_gaussian(approximation, rng) {
                Ok(value) => value,
                Err(PathfinderError::Numerical) if options.recover_numerical => {
                    invalid_proposal_draws += 1;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let logp = if options.recover_numerical {
                proposal_evaluate(target, &draw, &mut gradient, 0, &mut evaluations)?
                    .unwrap_or(f64::NEG_INFINITY)
            } else {
                checked_evaluate(target, &draw, &mut gradient, 0, &mut evaluations)?
            };
            let denominator = match options.resampling {
                ImportanceResampling::RawMixture => {
                    let terms = approximations
                        .iter()
                        .map(|component| log_gaussian(component, &draw))
                        .collect::<Result<Vec<_>, _>>()?;
                    log_sum_exp(&terms)? - (approximations.len() as f64).ln()
                }
                ImportanceResampling::PsisPathwise => logq,
            };
            let weight = logp - denominator;
            if !weight.is_finite() {
                if options.recover_numerical && weight == f64::NEG_INFINITY {
                    invalid_proposal_draws += 1;
                } else {
                    return Err(PathfinderError::Numerical);
                }
            }
            pool.push(draw);
            weights.push(weight);
        }
    }
    let (pareto_shape, psis_tail_length, psis_status, importance_ess) = match options.resampling {
        ImportanceResampling::RawMixture => {
            let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !maximum.is_finite() {
                return Err(PathfinderError::NoApproximation);
            }
            let sum = weights.iter().map(|w| (w - maximum).exp()).sum::<f64>();
            let square = weights
                .iter()
                .map(|w| (2. * (w - maximum)).exp())
                .sum::<f64>();
            (None, 0, PsisStatus::NotRequested, sum * sum / square)
        }
        ImportanceResampling::PsisPathwise => {
            let smoothed = psis::smooth(&weights).map_err(|_| PathfinderError::Numerical)?;
            weights = smoothed.logs;
            (smoothed.shape, smoothed.tail, smoothed.status, smoothed.ess)
        }
    };
    let selected = sample_log_weights(&weights, rng)?;
    Ok(PathfinderResult {
        position: OwnedBuffer::from_fn(dimension, |i| pool[selected][i]),
        report: PathfinderReport {
            paths: approximations.len(),
            optimization_steps,
            target_evaluations: evaluations,
            proposal_draws: pool_size,
            raw_importance_resampling: options.resampling == ImportanceResampling::RawMixture,
            requested_paths: options.paths.get(),
            invalid_proposal_draws,
            importance_ess: Some(importance_ess),
            pareto_shape,
            psis_tail_length,
            psis_status,
        },
        paths,
    })
}

fn single_path<T, R>(
    target: &T,
    initial: &[f64],
    options: MultiPathfinderOptions,
    rng: &mut R,
    path: usize,
    evaluations: &mut usize,
    optimization_steps: &mut usize,
) -> Result<(Approximation, usize), PathfinderError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    R: Rng + ?Sized,
{
    let dimension = initial.len();
    let mut position = initial.to_vec();
    let mut score = vec![0.0; dimension];
    let mut logp = if options.recover_numerical {
        proposal_evaluate(target, &position, &mut score, path, evaluations)?
            .ok_or(PathfinderError::NoApproximation)?
    } else {
        checked_evaluate(target, &position, &mut score, path, evaluations)?
    };
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(options.history.get());
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(options.history.get());
    let best = local_approximation(
        target,
        LocalApproximationInput {
            position: &position,
            score: &score,
            s_history: &s_history,
            y_history: &y_history,
            elbo_draws: options.elbo_draws.get(),
            path,
            recover_numerical: options.recover_numerical,
        },
        rng,
        evaluations,
    );
    let mut best = match best {
        Ok(value) => Some(value),
        Err(PathfinderError::NoApproximation) if options.recover_numerical => None,
        Err(error) => return Err(error),
    };
    let mut steps = 0;
    let mut direction = vec![0.0; dimension];
    let mut candidate = vec![0.0; dimension];
    let mut candidate_score = vec![0.0; dimension];
    for _ in 0..options.max_iterations.get() {
        let score_norm = score.iter().map(|value| value * value).sum::<f64>().sqrt();
        if score_norm <= options.relative_tolerance * (1.0 + logp.abs()) {
            break;
        }
        lbfgs_apply(&score, &s_history, &y_history, &mut direction)?;
        let mut slope = score
            .iter()
            .zip(&direction)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        if !slope.is_finite() || slope <= 0.0 {
            direction.copy_from_slice(&score);
            slope = score.iter().map(|value| value * value).sum();
        }
        // Standard first-iteration trust scaling is needed before any L-BFGS
        // curvature exists: twenty backtracks from1 cannot resolve a1e11 score.
        // The legacy strict workflow keeps its original trial sequence.
        let mut step = if options.recover_numerical && steps == 0 {
            let maximum = direction.iter().fold(0_f64, |m, x| m.max(x.abs()));
            if maximum > 1. { maximum.recip() } else { 1. }
        } else {
            1.
        };
        let mut accepted = None;
        for _ in 0..options.max_line_search.get() {
            for ((value, &current), &axis) in candidate.iter_mut().zip(&position).zip(&direction) {
                *value = current + step * axis;
            }
            if candidate.iter().all(|value| value.is_finite()) {
                let candidate_logp = if options.recover_numerical {
                    proposal_evaluate(target, &candidate, &mut candidate_score, path, evaluations)?
                } else {
                    Some(checked_evaluate(
                        target,
                        &candidate,
                        &mut candidate_score,
                        path,
                        evaluations,
                    )?)
                };
                let improvement = if options.recover_numerical {
                    score
                        .iter()
                        .zip(&direction)
                        .map(|(g, v)| g * (step * v))
                        .sum::<f64>()
                } else {
                    step * slope
                };
                if let Some(candidate_logp) = candidate_logp
                    && candidate_logp >= logp + 1e-4 * improvement
                {
                    accepted = Some(candidate_logp);
                    break;
                }
            }
            step *= 0.5;
        }
        let Some(candidate_logp) = accepted else {
            break;
        };
        let s = candidate
            .iter()
            .zip(&position)
            .map(|(new, old)| new - old)
            .collect::<Vec<_>>();
        // L-BFGS minimizes U=-logp, so y = grad U_new-grad U_old.
        let y = candidate_score
            .iter()
            .zip(&score)
            .map(|(new, old)| old - new)
            .collect::<Vec<_>>();
        let curvature = s.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>();
        let scale = s.iter().map(|v| v * v).sum::<f64>().sqrt()
            * y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if curvature.is_finite() && curvature > f64::EPSILON.sqrt() * scale {
            if s_history.len() == options.history.get() {
                s_history.remove(0);
                y_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
        }
        position.copy_from_slice(&candidate);
        score.copy_from_slice(&candidate_score);
        logp = candidate_logp;
        steps += 1;
        *optimization_steps = optimization_steps
            .checked_add(1)
            .ok_or(PathfinderError::Numerical)?;
        match local_approximation(
            target,
            LocalApproximationInput {
                position: &position,
                score: &score,
                s_history: &s_history,
                y_history: &y_history,
                elbo_draws: options.elbo_draws.get(),
                path,
                recover_numerical: options.recover_numerical,
            },
            rng,
            evaluations,
        ) {
            Ok(approximation)
                if best
                    .as_ref()
                    .is_none_or(|old| approximation.elbo > old.elbo) =>
            {
                best = Some(approximation)
            }
            Ok(_) | Err(PathfinderError::NoApproximation) => {}
            Err(error) => return Err(error),
        }
    }
    Ok((best.ok_or(PathfinderError::NoApproximation)?, steps))
}

fn lbfgs_apply<E: Error + 'static>(
    vector: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    output: &mut [f64],
) -> Result<(), PathfinderError<E>> {
    alea_math::quasi_newton::lbfgs_inverse_product(vector, s_history, y_history, output)
        .map_err(|_| PathfinderError::Numerical)
}

fn local_approximation<T, R>(
    target: &T,
    input: LocalApproximationInput<'_>,
    rng: &mut R,
    evaluations: &mut usize,
) -> Result<Approximation, PathfinderError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    R: Rng + ?Sized,
{
    let LocalApproximationInput {
        position,
        score,
        s_history,
        y_history,
        elbo_draws,
        path,
        recover_numerical,
    } = input;
    let dimension = position.len();
    let mut covariance = vec![0.0; dimension * dimension];
    let mut axis = vec![0.0; dimension];
    let mut action = vec![0.0; dimension];
    for column in 0..dimension {
        axis.fill(0.0);
        axis[column] = 1.0;
        lbfgs_apply::<T::Error>(&axis, s_history, y_history, &mut action)?;
        for row in 0..dimension {
            covariance[row * dimension + column] = action[row];
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            let value =
                0.5 * (covariance[row * dimension + column] + covariance[column * dimension + row]);
            covariance[row * dimension + column] = value;
            covariance[column * dimension + row] = value;
        }
    }
    let lower = cholesky(&covariance, dimension)?;
    let log_det = 2.0
        * (0..dimension)
            .map(|i| lower[i * dimension + i].ln())
            .sum::<f64>();
    let mut shift = vec![0.0; dimension];
    lbfgs_apply::<T::Error>(score, s_history, y_history, &mut shift)?;
    let mean = position
        .iter()
        .zip(shift)
        .map(|(&value, delta)| value + delta)
        .collect::<Vec<_>>();
    if mean.iter().any(|value| !value.is_finite()) || !log_det.is_finite() {
        return Err(PathfinderError::Numerical);
    }
    let mut approximation = Approximation {
        mean,
        lower,
        log_det,
        elbo: f64::NEG_INFINITY,
    };
    let mut gradient = vec![0.0; dimension];
    let mut total = 0.0;
    for _ in 0..elbo_draws {
        let (draw, logq) = draw_gaussian(&approximation, rng)?;
        let logp = if recover_numerical {
            proposal_evaluate(target, &draw, &mut gradient, path, evaluations)?
                .ok_or(PathfinderError::NoApproximation)?
        } else {
            checked_evaluate(target, &draw, &mut gradient, path, evaluations)?
        };
        total += logp - logq;
    }
    approximation.elbo = total / elbo_draws as f64;
    if !approximation.elbo.is_finite() {
        return Err(PathfinderError::Numerical);
    }
    Ok(approximation)
}

fn cholesky<E: Error + 'static>(
    matrix: &[f64],
    dimension: usize,
) -> Result<Vec<f64>, PathfinderError<E>> {
    let mut lower = vec![0.0; matrix.len()];
    for row in 0..dimension {
        for column in 0..=row {
            let mut residual = matrix[row * dimension + column];
            for k in 0..column {
                residual -= lower[row * dimension + k] * lower[column * dimension + k];
            }
            if row == column {
                if !residual.is_finite() || residual <= 0.0 {
                    return Err(PathfinderError::NoApproximation);
                }
                lower[row * dimension + column] = residual.sqrt();
            } else {
                lower[row * dimension + column] = residual / lower[column * dimension + column];
            }
        }
    }
    Ok(lower)
}

fn draw_gaussian<E: Error + 'static, R: Rng + ?Sized>(
    approximation: &Approximation,
    rng: &mut R,
) -> Result<(Vec<f64>, f64), PathfinderError<E>> {
    let dimension = approximation.mean.len();
    let normal = (0..dimension)
        .map(|_| standard_normal(rng))
        .collect::<Vec<_>>();
    let mut draw = approximation.mean.clone();
    for (row, value) in draw.iter_mut().enumerate() {
        for (column, &z) in normal.iter().enumerate().take(row + 1) {
            *value += approximation.lower[row * dimension + column] * z;
        }
    }
    let squared = normal.iter().map(|value| value * value).sum::<f64>();
    let logq = -0.5 * (dimension as f64 * LOG_TWO_PI + approximation.log_det + squared);
    if draw.iter().any(|value| !value.is_finite()) || !logq.is_finite() {
        return Err(PathfinderError::Numerical);
    }
    Ok((draw, logq))
}

fn log_gaussian<E: Error + 'static>(
    approximation: &Approximation,
    point: &[f64],
) -> Result<f64, PathfinderError<E>> {
    let dimension = point.len();
    let mut standardized = vec![0.0; dimension];
    for row in 0..dimension {
        let mut residual = point[row] - approximation.mean[row];
        for (column, &value) in standardized.iter().enumerate().take(row) {
            residual -= approximation.lower[row * dimension + column] * value;
        }
        standardized[row] = residual / approximation.lower[row * dimension + row];
    }
    let squared = standardized.iter().map(|value| value * value).sum::<f64>();
    let logq = -0.5 * (dimension as f64 * LOG_TWO_PI + approximation.log_det + squared);
    if logq.is_finite() {
        Ok(logq)
    } else {
        Err(PathfinderError::Numerical)
    }
}

// A proposal can have genuine support zero without constituting a valid cached
// point. Preserve all other model/shape/output failures as typed errors.
fn proposal_evaluate<T: LogDensityGradient + ?Sized>(
    target: &T,
    position: &[f64],
    gradient: &mut [f64],
    path: usize,
    evaluations: &mut usize,
) -> Result<Option<f64>, PathfinderError<T::Error>> {
    let failure = |source| PathfinderError::Evaluation { path, source };
    if position.len() != target.dimension() || gradient.len() != target.dimension() {
        return Err(failure(EvaluationError::Dimension(
            alea_core::target::DimensionError {
                expected: target.dimension(),
                position: position.len(),
                gradient: gradient.len(),
            },
        )));
    }
    if position.iter().any(|x| !x.is_finite()) {
        return Ok(None);
    }
    gradient.fill(f64::NAN);
    *evaluations = evaluations
        .checked_add(1)
        .ok_or(PathfinderError::Numerical)?;
    let logp = target
        .logp_grad(position, gradient)
        .map_err(|e| failure(EvaluationError::Model(e)))?;
    if logp == f64::NEG_INFINITY {
        return Ok(None);
    }
    if !logp.is_finite() {
        return Err(failure(EvaluationError::NonFiniteLogDensity));
    }
    if let Some(index) = gradient.iter().position(|x| !x.is_finite()) {
        return Err(failure(EvaluationError::NonFiniteGradient { index }));
    }
    Ok(Some(logp))
}

fn checked_evaluate<T: LogDensityGradient + ?Sized>(
    target: &T,
    position: &[f64],
    gradient: &mut [f64],
    path: usize,
    evaluations: &mut usize,
) -> Result<f64, PathfinderError<T::Error>> {
    *evaluations = evaluations
        .checked_add(1)
        .ok_or(PathfinderError::Numerical)?;
    evaluate(target, position, gradient)
        .map_err(|source| PathfinderError::Evaluation { path, source })
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn log_sum_exp<E: Error + 'static>(values: &[f64]) -> Result<f64, PathfinderError<E>> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(PathfinderError::Numerical);
    }
    let sum = values
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f64>();
    let result = maximum + sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(PathfinderError::Numerical)
    }
}

fn sample_log_weights<E: Error + 'static, R: Rng + ?Sized>(
    weights: &[f64],
    rng: &mut R,
) -> Result<usize, PathfinderError<E>> {
    let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total = weights
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f64>();
    if !maximum.is_finite() || !total.is_finite() || total <= 0.0 {
        return Err(PathfinderError::Numerical);
    }
    let threshold = rng.random::<f64>() * total;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += (weight - maximum).exp();
        if threshold < cumulative {
            return Ok(index);
        }
    }
    Ok(weights.len() - 1)
}

/// Runs the bounded score-ascent baseline with Armijo backtracking.
pub fn score_initialize<T: LogDensityGradient + ?Sized>(
    target: &T,
    initial: &[f64],
    options: ScoreInitializationOptions,
) -> Result<PathfinderResult, PathfinderError<T::Error>> {
    if initial.len() != target.dimension() || initial.iter().any(|value| !value.is_finite()) {
        return Err(PathfinderError::Configuration);
    }
    let mut evaluations = 0;
    let mut position = initial.to_vec();
    let mut score = vec![0.0; initial.len()];
    let mut candidate = position.clone();
    let mut candidate_score = score.clone();
    let mut logp = checked_evaluate(target, &position, &mut score, 0, &mut evaluations)?;
    let mut completed = 0;
    for _ in 0..options.iterations {
        let slope = dot(&score, &score);
        if !slope.is_finite() || slope == 0.0 {
            break;
        }
        let mut step = options.initial_step;
        let mut accepted = None;
        for _ in 0..options.max_line_search {
            for ((value, &current), &gradient) in candidate.iter_mut().zip(&position).zip(&score) {
                *value = current + step * gradient;
            }
            let value = checked_evaluate(
                target,
                &candidate,
                &mut candidate_score,
                0,
                &mut evaluations,
            )?;
            if value >= logp + 1e-4 * step * slope {
                accepted = Some(value);
                break;
            }
            step *= 0.5;
        }
        let Some(value) = accepted else { break };
        position.copy_from_slice(&candidate);
        score.copy_from_slice(&candidate_score);
        logp = value;
        completed += 1;
    }
    Ok(PathfinderResult {
        position: OwnedBuffer::from_fn(position.len(), |i| position[i]),
        report: PathfinderReport {
            paths: 1,
            optimization_steps: completed,
            target_evaluations: evaluations,
            proposal_draws: 0,
            raw_importance_resampling: false,
            requested_paths: 1,
            invalid_proposal_draws: 0,
            importance_ess: None,
            pareto_shape: None,
            psis_tail_length: 0,
            psis_status: PsisStatus::NotRequested,
        },
        paths: vec![PathfinderPathReport {
            path: 0,
            status: PathfinderPathStatus::Completed,
            optimization_steps: completed,
            target_evaluations: evaluations,
        }],
    })
}
