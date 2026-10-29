//! Bounded local maximization of explicitly declared scalar objectives.
//!
//! For MLE use a likelihood-only `ComposedObjective`; for declared-measure MAP
//! compose likelihood and prior without a parameter-chart Jacobian. A transformed
//! sampling target generally has a different mode. This driver does not infer
//! statistical meaning, establish existence/global optimality or estimate covariance.
#![deny(missing_docs)]
use alea_core::target::{DimensionError, EvaluationError, LogDensityGradient};
use alea_core::{
    model::{assembly::Coordinate, prediction::NamedOutput},
    transform::TransformError,
};
use alea_math::quasi_newton::lbfgs_inverse_product;
use std::error::Error;
use thiserror::Error;

/// Caller-declared statistical interpretation; not verified by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveKind {
    /// Likelihood without priors or coordinate Jacobians.
    Likelihood,
    /// Posterior density in the explicitly declared parameter measure.
    Posterior,
    /// Another differentiable scalar objective.
    Other,
}
/// Objective identity and measure, retained with every outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveDescription {
    /// Intended statistical interpretation.
    pub kind: ObjectiveKind,
    /// Measure with respect to which the original objective is defined.
    pub measure: String,
    /// Model identity, preferably a content digest.
    pub model: String,
    /// Data identity, preferably a content digest.
    pub data: String,
    /// Source/build identity; labels alone do not verify provenance.
    pub source: String,
}
/// Invalid bounds, tolerances, scales, objective metadata or initial coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid optimization configuration, objective metadata or initial coordinates")]
pub struct InvalidOptimization;

/// Local inverse-BFGS maximization policy with Armijo backtracking.
#[derive(Debug, Clone)]
pub struct LbfgsOptions {
    iterations: usize,
    evaluations: usize,
    history: usize,
    backtracks: usize,
    score_tolerance: f64,
    scales: Option<Vec<f64>>,
}
impl LbfgsOptions {
    /// Maximum accepted steps.
    pub fn iteration_limit(&self) -> usize {
        self.iterations
    }
    /// Maximum actual fused calls.
    pub fn evaluation_limit(&self) -> usize {
        self.evaluations
    }
    /// Maximum stored secant pairs.
    pub fn history_limit(&self) -> usize {
        self.history
    }
    /// Maximum line-search trials per iteration.
    pub fn line_search_limit(&self) -> usize {
        self.backtracks
    }
    /// Absolute scaled-score stopping tolerance.
    pub fn score_tolerance(&self) -> f64 {
        self.score_tolerance
    }
    /// Declared free-coordinate scales; `None` means unit scales.
    pub fn scales(&self) -> Option<&[f64]> {
        self.scales.as_deref()
    }

    /// Bounds accepted steps and fused evaluations (including the initial call).
    /// Defaults: history 10, 40 search trials, absolute scaled-score tolerance 1e-8.
    /// # Errors
    /// Rejects zero limits.
    pub fn new(iterations: usize, evaluations: usize) -> Result<Self, InvalidOptimization> {
        if iterations == 0 || evaluations == 0 {
            return Err(InvalidOptimization);
        }
        Ok(Self {
            iterations,
            evaluations,
            history: 10,
            backtracks: 40,
            score_tolerance: 1e-8,
            scales: None,
        })
    }
    /// Sets the maximum number of retained positive-curvature secant pairs.
    /// # Errors
    /// Rejects zero history.
    pub fn with_history(mut self, history: usize) -> Result<Self, InvalidOptimization> {
        if history == 0 {
            return Err(InvalidOptimization);
        }
        self.history = history;
        Ok(self)
    }
    /// Sets the maximum trials per line search, including its first trial.
    /// # Errors
    /// Rejects zero trials.
    pub fn with_line_search(mut self, trials: usize) -> Result<Self, InvalidOptimization> {
        if trials == 0 {
            return Err(InvalidOptimization);
        }
        self.backtracks = trials;
        Ok(self)
    }
    /// Requires `max_i |scale_i * score_i| <= tolerance`; not relative to log value.
    /// A small score does not prove a finite/global estimator or identify a maximum.
    /// # Errors
    /// Rejects nonfinite or nonpositive tolerances.
    pub fn with_score_tolerance(mut self, tolerance: f64) -> Result<Self, InvalidOptimization> {
        if !tolerance.is_finite() || tolerance <= 0. {
            return Err(InvalidOptimization);
        }
        self.score_tolerance = tolerance;
        Ok(self)
    }
    /// Optimizes in coordinates `z_i = q_i / scale_i`, defaulting to unit scales.
    /// # Errors
    /// Rejects nonfinite/nonpositive scales; dimensional agreement is checked at run time.
    pub fn with_scales(mut self, scales: Vec<f64>) -> Result<Self, InvalidOptimization> {
        if scales.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(InvalidOptimization);
        }
        self.scales = Some(scales);
        Ok(self)
    }
}
/// A completely evaluated candidate in the supplied objective's free coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationCandidate {
    /// Free coordinates; convert with the objective's parameter layout for named estimates.
    pub position: Vec<f64>,
    /// Original objective value, without a sign reversal or added Jacobian.
    pub value: f64,
    /// Gradient in free coordinates, not a constrained-coordinate score.
    pub gradient: Vec<f64>,
}
/// Why the bounded local search stopped. Only `ScoreTolerance` meets a tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationTermination {
    /// The declared absolute scaled-score condition was met.
    ScoreTolerance,
    /// Accepted-step budget exhausted.
    IterationLimit,
    /// Fused evaluation budget exhausted.
    EvaluationLimit,
    /// Cancellation requested between evaluations; user calls are not interrupted.
    Cancelled,
    /// No strictly improving, representable Armijo step was found.
    LineSearchFailed,
    /// A finite ascent direction could not be represented.
    NumericalDirection,
    /// A checked evaluation or backend failed; inspect the accompanying typed error.
    EvaluationFailed,
}
/// Point-estimation outcome, distinct from draws and Pathfinder approximations.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Dimension of the free-coordinate objective, even when no candidate exists.
    pub dimension: usize,
    /// Complete numerical policy, including scales and stopping budgets.
    pub options: LbfgsOptions,
    /// Explicit objective identity/measure, not inferred from a trait name.
    pub objective: ObjectiveDescription,
    /// Last complete accepted candidate; absent before a successful initial evaluation.
    pub candidate: Option<OptimizationCandidate>,
    /// Exact stopping condition; no global-optimum claim.
    pub termination: OptimizationTermination,
    /// Accepted steps, excluding the initial evaluation.
    pub iterations: usize,
    /// Actual backend fused calls, including failed and rejected evaluations.
    pub evaluations: usize,
    /// Nonfinite trial positions or negative-infinity trial objectives backtracked over.
    pub domain_trials: usize,
    /// Secant pairs omitted because finite positive curvature was not established.
    pub skipped_updates: usize,
}
/// Original checked backend error plus the uncorrupted last candidate and counters.
#[derive(Debug, Error)]
#[error("optimization evaluation failed: {source}")]
pub struct OptimizationError<E: Error + 'static> {
    /// Checked error retaining the model's concrete error type.
    #[source]
    pub source: EvaluationError<E>,
    /// All completed progress, including failed evaluation accounting.
    pub result: Box<OptimizationResult>,
}
/// Invalid configuration is separate from target/backend failure.
#[derive(Debug, Error)]
pub enum MaximizeError<E: Error + 'static> {
    /// Invalid bounds, metadata, input dimensions or scales, before model invocation.
    #[error(transparent)]
    Invalid(#[from] InvalidOptimization),
    /// A backend/checked evaluation failure with retained progress.
    #[error(transparent)]
    Evaluation(#[from] Box<OptimizationError<E>>),
}

/// Maximizes the supplied scalar objective by bounded L-BFGS ascent.
/// # Errors
/// Returns invalid configuration or a typed evaluation error with retained progress.
/// A trial returning negative infinity is a domain rejection; typed backend errors,
/// NaN, positive infinity and incomplete gradients propagate rather than backtrack.
pub fn maximize<T: LogDensityGradient + ?Sized>(
    target: &T,
    initial: &[f64],
    objective: ObjectiveDescription,
    options: LbfgsOptions,
) -> Result<OptimizationResult, MaximizeError<T::Error>> {
    maximize_with_cancel(target, initial, objective, options, || false)
}
/// Maximizes with cooperative cancellation before initialization and between trials.
/// # Errors
/// Same admission/evaluation contract as [`maximize`]. Callback/model panics propagate;
/// no partial trial is published, and no backend call is forcibly interrupted.
pub fn maximize_with_cancel<T: LogDensityGradient + ?Sized>(
    target: &T,
    initial: &[f64],
    objective: ObjectiveDescription,
    options: LbfgsOptions,
    cancelled: impl FnMut() -> bool,
) -> Result<OptimizationResult, MaximizeError<T::Error>> {
    #[cfg(feature = "tracing")]
    let _span = tracing::info_span!(target: "alea::optimization", "maximize", input_dimension = initial.len()).entered();
    #[cfg(feature = "tracing")]
    tracing::info!(target: "alea::optimization", schema_version = 1_u64, event = "started", iteration_limit = options.iterations, evaluation_limit = options.evaluations);
    let outcome = maximize_with_cancel_inner(target, initial, objective, options, cancelled);
    #[cfg(feature = "tracing")]
    match &outcome {
        Ok(result) => {
            tracing::info!(target: "alea::optimization", schema_version = 1_u64, event = "finished", termination = ?result.termination, iterations = result.iterations, evaluations = result.evaluations, domain_trials = result.domain_trials, skipped_updates = result.skipped_updates)
        }
        Err(MaximizeError::Invalid(_)) => {
            tracing::error!(target: "alea::optimization", schema_version = 1_u64, event = "failed", failure = "configuration")
        }
        Err(MaximizeError::Evaluation(error)) => {
            tracing::error!(target: "alea::optimization", schema_version = 1_u64, event = "failed", failure = "evaluation", iterations = error.result.iterations, evaluations = error.result.evaluations)
        }
    }
    outcome
}
fn maximize_with_cancel_inner<T: LogDensityGradient + ?Sized>(
    target: &T,
    initial: &[f64],
    objective: ObjectiveDescription,
    options: LbfgsOptions,
    mut cancelled: impl FnMut() -> bool,
) -> Result<OptimizationResult, MaximizeError<T::Error>> {
    let n = target.dimension();
    if n != initial.len()
        || initial.iter().any(|x| !x.is_finite())
        || [
            &objective.measure,
            &objective.model,
            &objective.data,
            &objective.source,
        ]
        .iter()
        .any(|x| x.is_empty())
        || options
            .history
            .checked_mul(n)
            .and_then(|size| size.checked_mul(2 * size_of::<f64>()))
            .is_none()
    {
        return Err(InvalidOptimization.into());
    }
    let scales = options.scales.clone().unwrap_or_else(|| vec![1.; n]);
    if scales.len() != n {
        return Err(InvalidOptimization.into());
    }
    let mut result = OptimizationResult {
        dimension: n,
        options: options.clone(),
        objective,
        candidate: None,
        termination: OptimizationTermination::Cancelled,
        iterations: 0,
        evaluations: 0,
        domain_trials: 0,
        skipped_updates: 0,
    };
    if cancelled() {
        return Ok(result);
    }
    let mut gradient = vec![f64::NAN; n];
    let value = match trial(target, initial, &mut gradient, n, &mut result.evaluations) {
        Ok(Some(value)) => value,
        Ok(None) => return Err(failed(result, EvaluationError::NonFiniteLogDensity)),
        Err(error) => return Err(failed(result, error)),
    };
    result.candidate = Some(OptimizationCandidate {
        position: initial.to_vec(),
        value,
        gradient,
    });
    let mut s_history = Vec::new();
    let mut y_history = Vec::new();
    loop {
        let current = result
            .candidate
            .as_ref()
            .expect("initial candidate admitted");
        let score: Vec<_> = current
            .gradient
            .iter()
            .zip(&scales)
            .map(|(g, s)| g * s)
            .collect();
        let norm = score.iter().fold(0_f64, |m, x| m.max(x.abs()));
        if norm.is_finite() && norm <= options.score_tolerance {
            result.termination = OptimizationTermination::ScoreTolerance;
            break;
        }
        if cancelled() {
            result.termination = OptimizationTermination::Cancelled;
            break;
        }
        if result.iterations == options.iterations {
            result.termination = OptimizationTermination::IterationLimit;
            break;
        }
        if result.evaluations == options.evaluations {
            result.termination = OptimizationTermination::EvaluationLimit;
            break;
        }
        let mut direction = vec![0.; n];
        if lbfgs_inverse_product(&score, &s_history, &y_history, &mut direction).is_err()
            || dot(&score, &direction) <= 0.
            || !dot(&score, &direction).is_finite()
        {
            direction.copy_from_slice(&score);
            s_history.clear();
            y_history.clear();
        }
        let slope = dot(&score, &direction);
        if !slope.is_finite() || slope <= 0. {
            result.termination = OptimizationTermination::NumericalDirection;
            break;
        }
        let maximum = direction.iter().fold(0_f64, |m, x| m.max(x.abs()));
        let mut step = if result.iterations == 0 && maximum > 1. {
            maximum.recip()
        } else {
            1.
        };
        let mut accepted = None;
        for _ in 0..options.backtracks {
            if cancelled() {
                result.termination = OptimizationTermination::Cancelled;
                return Ok(result);
            }
            if result.evaluations == options.evaluations {
                result.termination = OptimizationTermination::EvaluationLimit;
                return Ok(result);
            }
            let position: Vec<_> = current
                .position
                .iter()
                .zip(&direction)
                .zip(&scales)
                .map(|((q, d), s)| q + step * d * s)
                .collect();
            if position == current.position {
                break;
            }
            if position.iter().all(|x| x.is_finite()) {
                let mut gradient = vec![f64::NAN; n];
                let value =
                    match trial(target, &position, &mut gradient, n, &mut result.evaluations) {
                        Ok(value) => value,
                        Err(error) => return Err(failed(result, error)),
                    };
                if let Some(value) = value {
                    let improvement = score
                        .iter()
                        .zip(&direction)
                        .map(|(g, d)| g * (step * d))
                        .sum::<f64>();
                    if value > current.value && value >= current.value + 1e-4 * improvement {
                        accepted = Some(OptimizationCandidate {
                            position,
                            value,
                            gradient,
                        });
                        break;
                    }
                } else {
                    result.domain_trials += 1;
                }
            } else {
                result.domain_trials += 1;
            }
            step *= 0.5;
        }
        let Some(candidate) = accepted else {
            result.termination = OptimizationTermination::LineSearchFailed;
            break;
        };
        let s: Vec<_> = candidate
            .position
            .iter()
            .zip(&current.position)
            .zip(&scales)
            .map(|((a, b), scale)| (a - b) / scale)
            .collect();
        // Curvature is for the minimized negative objective in z coordinates.
        let y: Vec<_> = current
            .gradient
            .iter()
            .zip(&candidate.gradient)
            .zip(&scales)
            .map(|((old, new), scale)| (old - new) * scale)
            .collect();
        let curvature = dot(&s, &y);
        let scale = dot(&s, &s).sqrt() * dot(&y, &y).sqrt();
        if curvature.is_finite() && scale.is_finite() && curvature > f64::EPSILON.sqrt() * scale {
            if s_history.len() == options.history {
                s_history.remove(0);
                y_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
        } else {
            result.skipped_updates += 1;
        }
        result.candidate = Some(candidate);
        result.iterations += 1;
        #[cfg(feature = "tracing")]
        tracing::debug!(target: "alea::optimization", schema_version = 1_u64, event = "iteration_completed", iterations = result.iterations, evaluations = result.evaluations, domain_trials = result.domain_trials, skipped_updates = result.skipped_updates);
    }
    Ok(result)
}
fn failed<E: Error + 'static>(
    mut result: OptimizationResult,
    source: EvaluationError<E>,
) -> MaximizeError<E> {
    result.termination = OptimizationTermination::EvaluationFailed;
    Box::new(OptimizationError {
        source,
        result: Box::new(result),
    })
    .into()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn trial<T: LogDensityGradient + ?Sized>(
    target: &T,
    q: &[f64],
    g: &mut [f64],
    n: usize,
    evaluations: &mut usize,
) -> Result<Option<f64>, EvaluationError<T::Error>> {
    if target.dimension() != n {
        return Err(EvaluationError::TargetDimensionChanged {
            expected: n,
            actual: target.dimension(),
        });
    }
    if q.len() != n || g.len() != n {
        return Err(DimensionError {
            expected: n,
            position: q.len(),
            gradient: g.len(),
        }
        .into());
    }
    *evaluations += 1;
    let value = target.logp_grad(q, g).map_err(EvaluationError::Model)?;
    if value == f64::NEG_INFINITY {
        return Ok(None);
    }
    if !value.is_finite() {
        return Err(EvaluationError::NonFiniteLogDensity);
    }
    if let Some(index) = g.iter().position(|x| !x.is_finite()) {
        return Err(EvaluationError::NonFiniteGradient { index });
    }
    Ok(Some(value))
}

/// Named, constrained point estimates together with their original numerical outcome.
/// Metadata is supplied explicitly: matching dimensions do not prove model identity.
/// Construction never turns a score condition into an existence/global-optimum claim.
#[derive(Debug, Clone)]
pub struct NamedFit {
    parameters: NamedOutput<()>,
    result: OptimizationResult,
    estimates: Option<Vec<f64>>,
}
/// Projection failure retaining the unmodified optimization result.
#[derive(Debug, Error)]
#[error("cannot name optimization result: {source}")]
pub struct FitProjectionError {
    /// Invalid dimensions, candidate values or a parameter-chart failure.
    #[source]
    pub source: TransformError,
    /// Original outcome, including its candidate and failure/termination evidence.
    pub result: Box<OptimizationResult>,
}
impl NamedFit {
    /// Attaches parameter metadata and constrains the candidate without a Jacobian.
    /// An outcome without a candidate retains `None` estimates. Free-coordinate
    /// gradients remain in the original report, not relabeled as constrained scores.
    /// # Errors
    /// Rejects mismatched dimensions, nonfinite candidate fields and failed transforms.
    /// Failure retains the original numerical result without publishing partial estimates.
    pub fn new(
        parameters: NamedOutput<()>,
        result: OptimizationResult,
    ) -> Result<Self, FitProjectionError> {
        let project = || -> Result<Option<Vec<f64>>, TransformError> {
            let n = parameters.layout().unconstrained_dimension();
            if n != result.dimension {
                return Err(TransformError::Dimension {
                    expected: n,
                    actual: result.dimension,
                });
            }
            let Some(candidate) = &result.candidate else {
                return Ok(None);
            };
            if candidate.position.len() != n || candidate.gradient.len() != n {
                return Err(TransformError::InvalidDimension);
            }
            if !candidate.value.is_finite() || candidate.gradient.iter().any(|x| !x.is_finite()) {
                return Err(TransformError::NonFiniteOutput { index: 0 });
            }
            let mut values = vec![0.; parameters.layout().constrained_dimension()];
            let mut scratch = vec![0.; parameters.layout().scratch_dimension()];
            parameters
                .layout()
                .constrain_values(&candidate.position, &mut values, &mut scratch)?;
            Ok(Some(values))
        };
        match project() {
            Ok(estimates) => Ok(Self {
                parameters,
                result,
                estimates,
            }),
            Err(source) => Err(FitProjectionError {
                source,
                result: Box::new(result),
            }),
        }
    }
    /// Names, transform blocks and constrained-coordinate labels.
    pub fn parameters(&self) -> &NamedOutput<()> {
        &self.parameters
    }
    /// Original numerical result, with free-coordinate gradients and termination.
    pub fn result(&self) -> &OptimizationResult {
        &self.result
    }
    /// Constrained estimates in parameter metadata order, if a candidate exists.
    pub fn estimates(&self) -> Option<&[f64]> {
        self.estimates.as_deref()
    }
    /// A constrained estimate by its exact parameter name.
    pub fn estimate(&self, name: &str) -> Option<f64> {
        let index = self
            .parameters
            .coordinates()
            .iter()
            .position(|c: &Coordinate| c.name == name)?;
        self.estimates.as_ref().map(|v| v[index])
    }
    /// Recovers the original numerical result without parameter-chart reinterpretation.
    pub fn into_result(self) -> OptimizationResult {
        self.result
    }
}
