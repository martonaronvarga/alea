//! Bounded multi-path Pathfinder initialization.
//!
//! This is an initialization service, not an MCMC transition. Independent
//! L-BFGS paths select local Gaussian approximations by Monte Carlo ELBO and pool
//! their draws with raw mixture importance resampling. Pareto smoothing is not
//! silently implied; the report names the exact raw-resampling policy.

use std::{error::Error, num::NonZeroUsize};

use alea_core::target::{EvaluationError, LogDensityGradient, evaluate};
use alea_math::{buffer::OwnedBuffer, random::standard_normal};
use rand::{Rng, RngExt};
use thiserror::Error;

const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;

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
    #[error("Pathfinder produced no finite local Gaussian approximation")]
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
    /// Gaussian proposal draws entering the final pool.
    pub proposal_draws: usize,
    /// Raw mixture importance resampling was used (never PSIS by implication).
    pub raw_importance_resampling: bool,
}

/// One initialization point plus complete work accounting.
#[derive(Debug)]
pub struct PathfinderResult {
    /// Selected point in the target's coordinates.
    pub position: OwnedBuffer,
    /// Bounded initialization work.
    pub report: PathfinderReport,
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
}

/// Runs independent bounded paths and raw mixture importance resampling.
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
    for path in 0..options.paths.get() {
        let start = path * dimension;
        let (approximation, path_steps) = single_path(
            target,
            &initial_positions[start..start + dimension],
            options,
            rng,
            path,
            &mut evaluations,
        )?;
        optimization_steps = optimization_steps
            .checked_add(path_steps)
            .ok_or(PathfinderError::Numerical)?;
        approximations.push(approximation);
    }

    let pool_size = options
        .paths
        .get()
        .checked_mul(options.draws_per_path.get())
        .ok_or(PathfinderError::Numerical)?;
    let mut pool = Vec::with_capacity(pool_size);
    let mut weights = Vec::with_capacity(pool_size);
    let mut gradient = vec![0.0; dimension];
    for approximation in &approximations {
        for _ in 0..options.draws_per_path.get() {
            let (draw, _) = draw_gaussian(approximation, rng)?;
            let logp = checked_evaluate(target, &draw, &mut gradient, 0, &mut evaluations)?;
            let mut mixture_terms = Vec::with_capacity(approximations.len());
            for component in &approximations {
                mixture_terms.push(log_gaussian(component, &draw)?);
            }
            let log_mixture = log_sum_exp(&mixture_terms)? - (approximations.len() as f64).ln();
            let log_weight = logp - log_mixture;
            if !log_weight.is_finite() {
                return Err(PathfinderError::Numerical);
            }
            pool.push(draw);
            weights.push(log_weight);
        }
    }
    let selected = sample_log_weights(&weights, rng)?;
    Ok(PathfinderResult {
        position: OwnedBuffer::from_fn(dimension, |i| pool[selected][i]),
        report: PathfinderReport {
            paths: options.paths.get(),
            optimization_steps,
            target_evaluations: evaluations,
            proposal_draws: pool_size,
            raw_importance_resampling: true,
        },
    })
}

fn single_path<T, R>(
    target: &T,
    initial: &[f64],
    options: MultiPathfinderOptions,
    rng: &mut R,
    path: usize,
    evaluations: &mut usize,
) -> Result<(Approximation, usize), PathfinderError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    R: Rng + ?Sized,
{
    let dimension = initial.len();
    let mut position = initial.to_vec();
    let mut score = vec![0.0; dimension];
    let mut logp = checked_evaluate(target, &position, &mut score, path, evaluations)?;
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(options.history.get());
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(options.history.get());
    let mut best = local_approximation(
        target,
        LocalApproximationInput {
            position: &position,
            score: &score,
            s_history: &s_history,
            y_history: &y_history,
            elbo_draws: options.elbo_draws.get(),
            path,
        },
        rng,
        evaluations,
    )?;
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
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..options.max_line_search.get() {
            for ((value, &current), &axis) in candidate.iter_mut().zip(&position).zip(&direction) {
                *value = current + step * axis;
            }
            if candidate.iter().all(|value| value.is_finite()) {
                let candidate_logp =
                    checked_evaluate(target, &candidate, &mut candidate_score, path, evaluations)?;
                if candidate_logp >= logp + 1e-4 * step * slope {
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
        match local_approximation(
            target,
            LocalApproximationInput {
                position: &position,
                score: &score,
                s_history: &s_history,
                y_history: &y_history,
                elbo_draws: options.elbo_draws.get(),
                path,
            },
            rng,
            evaluations,
        ) {
            Ok(approximation) if approximation.elbo > best.elbo => best = approximation,
            Ok(_) | Err(PathfinderError::NoApproximation) => {}
            Err(error) => return Err(error),
        }
    }
    Ok((best, steps))
}

fn lbfgs_apply<E: Error + 'static>(
    vector: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    output: &mut [f64],
) -> Result<(), PathfinderError<E>> {
    output.copy_from_slice(vector);
    let mut alphas = vec![0.0; s_history.len()];
    for index in (0..s_history.len()).rev() {
        let sy = dot(&s_history[index], &y_history[index]);
        if !sy.is_finite() || sy <= 0.0 {
            return Err(PathfinderError::Numerical);
        }
        alphas[index] = dot(&s_history[index], output) / sy;
        for (value, &y) in output.iter_mut().zip(&y_history[index]) {
            *value -= alphas[index] * y;
        }
    }
    let gamma = if let (Some(s), Some(y)) = (s_history.last(), y_history.last()) {
        dot(s, y) / dot(y, y)
    } else {
        1.0
    };
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(PathfinderError::Numerical);
    }
    for value in output.iter_mut() {
        *value *= gamma;
    }
    for index in 0..s_history.len() {
        let sy = dot(&s_history[index], &y_history[index]);
        let beta = dot(&y_history[index], output) / sy;
        for (value, &s) in output.iter_mut().zip(&s_history[index]) {
            *value += s * (alphas[index] - beta);
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(PathfinderError::Numerical);
    }
    Ok(())
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
        let logp = checked_evaluate(target, &draw, &mut gradient, path, evaluations)?;
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
        },
    })
}
