//! Cold, standalone point-estimation wire format; never a sampling checkpoint.
use super::{AnalysisError, Provenance, schema};
use crate::optimize::{LineSearchFailure, NamedFit, ObjectiveKind, OptimizationTermination};
use serde_json::{Value, json};
use std::io::Write;

/// Point-estimation JSON version, independent of the sampling-table schema.
pub const FIT_SCHEMA_VERSION: u32 = 1;

/// Writes one complete named point-estimation report, without chain/draw axes.
///
/// Model/data/source identities must match the fit's objective declaration.
/// `encode_assessment` explicitly encodes caller-owned evidence (use
/// `|_| Ok(serde_json::Value::Null)` for an unassessed fit). Its JSON is preserved
/// without interpreting its statistical claims. An absent candidate stays null;
/// failed or cancelled outcomes are not promoted to successful estimates.
/// Gradients remain free-coordinate scores; physical estimates omit Jacobians.
/// No covariance, standard errors, backend error message or convergence claim is
/// inferred. Include separately retained backend error details in the assessment
/// if needed. The artifact does not reconstruct a model or resume optimization.
///
/// Validation and assessment encoding finish before touching the writer, including
/// on callback error or panic. Writes/flushes may fail partially: discard that
/// output and do not append/retry on the same stream. Atomic file installation
/// and source/data identity verification remain the application's responsibility.
///
/// # Errors
/// Rejects inconsistent provenance, policy dimensions, counters or stopping
/// evidence. Propagates assessment, JSON, write and flush errors unchanged.
pub fn write_fit<W: Write, A>(
    mut writer: W,
    fit: &NamedFit<A>,
    provenance: &Provenance,
    encode_assessment: impl FnOnce(&A) -> Result<Value, AnalysisError>,
) -> Result<(), AnalysisError> {
    schema::validate_provenance(provenance)?;
    let r = fit.result();
    let o = &r.options;
    let objective = &r.objective;
    if objective.measure.is_empty()
        || objective.model != provenance.model
        || objective.data != provenance.data
        || objective.source != provenance.source
    {
        return Err(AnalysisError::Invalid(
            "fit and provenance identities differ",
        ));
    }
    if o.scales().is_some_and(|s| s.len() != r.dimension)
        || r.iterations > o.iteration_limit()
        || r.evaluations > o.evaluation_limit()
        || r.skipped_updates > r.iterations
        || r.evaluations.saturating_sub(r.iterations) < usize::from(r.candidate.is_some())
        || (r.termination == OptimizationTermination::LineSearchFailed)
            != r.line_search_failure.is_some()
    {
        return Err(AnalysisError::Invalid(
            "inconsistent fit policy or counters",
        ));
    }
    if r.candidate.is_none()
        && (r.iterations != 0
            || r.domain_trials != 0
            || r.evaluations
                > usize::from(r.termination == OptimizationTermination::EvaluationFailed)
            || !matches!(
                r.termination,
                OptimizationTermination::Cancelled | OptimizationTermination::EvaluationFailed
            ))
    {
        return Err(AnalysisError::Invalid(
            "fit stopping evidence requires a candidate",
        ));
    }
    let (termination, valid_stop) = match r.termination {
        OptimizationTermination::ScoreTolerance => (
            "score_tolerance",
            r.candidate.as_ref().is_some_and(|c| {
                c.gradient.iter().enumerate().all(|(i, g)| {
                    let score = g * o.scales().map_or(1., |s| s[i]);
                    score.is_finite() && score.abs() <= o.score_tolerance()
                })
            }),
        ),
        OptimizationTermination::IterationLimit => {
            ("iteration_limit", r.iterations == o.iteration_limit())
        }
        OptimizationTermination::EvaluationLimit => {
            ("evaluation_limit", r.evaluations == o.evaluation_limit())
        }
        OptimizationTermination::Cancelled => ("cancelled", true),
        OptimizationTermination::LineSearchFailed => ("line_search_failed", true),
        OptimizationTermination::NumericalDirection => ("numerical_direction", true),
        OptimizationTermination::EvaluationFailed => ("evaluation_failed", true),
    };
    if !valid_stop {
        return Err(AnalysisError::Invalid("inconsistent fit stopping evidence"));
    }
    let kind = match objective.kind {
        ObjectiveKind::Likelihood => "likelihood",
        ObjectiveKind::Posterior => "posterior",
        ObjectiveKind::Other => "other",
    };
    let search_failure = r.line_search_failure.map(|reason| match reason {
        LineSearchFailure::BacktrackLimit => "backtrack_limit",
        LineSearchFailure::NoRepresentableStep => "no_representable_step",
    });
    let parameters: Vec<_> = fit.parameters().coordinates().iter().map(|p|
        json!({"name":p.name,"dimension":p.dimension,"coordinate":p.coordinate,"block":p.block})
    ).collect();
    let candidate = r.candidate.as_ref().map(|c| {
        json!({
            "free_position":c.position,"value":c.value,"free_gradient":c.gradient,
            "constrained_estimates":fit.estimates(),
        })
    });
    let record = json!({
        "format":"alea-fit","schema_version":FIT_SCHEMA_VERSION,"kind":"point_estimate",
        "provenance":schema::provenance(provenance),
        "objective":{"kind":kind,"measure":objective.measure},
        "parameters":parameters,"unconstrained_dimension":r.dimension,
        "numerical":{"algorithm":"lbfgs_armijo","termination":termination,"line_search_failure":search_failure,
            "options":{"iteration_limit":o.iteration_limit(),"evaluation_limit":o.evaluation_limit(),
                "history_limit":o.history_limit(),"line_search_limit":o.line_search_limit(),
                "score_tolerance":o.score_tolerance(),"scales":o.scales()},
            "iterations":r.iterations,"evaluations":r.evaluations,"domain_trials":r.domain_trials,"skipped_updates":r.skipped_updates},
        "candidate":candidate,"assessment":encode_assessment(fit.assessment())?,
    });
    // The bounded result is fully encoded before writes, unlike streamed draws.
    let mut bytes = serde_json::to_vec(&record)?;
    bytes.push(b'\n');
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}
