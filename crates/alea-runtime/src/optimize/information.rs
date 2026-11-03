//! Explicit observed-likelihood information; no finite-difference fallback.
#![deny(missing_docs)]
use super::{NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination};
use alea_core::{
    capability::{HvpError, PotentialHessianVector, evaluate_hvp},
    model::prediction::NamedOutput,
    target::{EvaluationError, evaluate},
    transform::TransformError,
};
use alea_math::{
    information::{InformationError, InformationMatrix, InformationOptions},
    matrix::{RowMajor, SquareMatrixRef},
};
use std::error::Error;

/// Failure to obtain an explicitly qualified numerical information report.
#[derive(Debug, thiserror::Error)]
pub enum ObservedInformationError<E: Error + 'static> {
    /// Requires a nonempty likelihood fit meeting its declared score tolerance.
    #[error("observed information requires a stationary likelihood fit with matching dimensions")]
    InvalidFit,
    /// Fresh evaluation of the supplied objective failed.
    #[error(transparent)]
    Evaluation(#[from] EvaluationError<E>),
    /// Fresh objective does not reproduce the stored candidate's value and score.
    #[error("information objective does not reproduce the fitted candidate")]
    ObjectiveMismatch,
    /// Original checked curvature failure at a specific basis-vector column.
    #[error("information HVP column {column} failed: {source}")]
    Hvp {
        /// Basis column being evaluated.
        column: usize,
        /// Original checked derivative failure.
        #[source]
        source: HvpError<E>,
    },
    /// Original parameter-chart derivative failure.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Matrix symmetry, conditioning, positivity or representability failure.
    #[error(transparent)]
    Matrix(#[from] InformationError),
}
/// Scoped observed-likelihood curvature, not confidence intervals or calibration.
/// Contains the original objective declaration and coordinate meaning. Positive
/// numerical curvature and a small score do not prove regularity, identifiability,
/// existence of an MLE, global optimality or valid repeated-sampling coverage.
#[derive(Debug, Clone)]
pub struct ObservedInformation {
    objective: ObjectiveDescription,
    position: Vec<f64>,
    factor: InformationMatrix,
    free: Vec<f64>,
    constrained: NamedOutput<Vec<f64>>,
}
impl ObservedInformation {
    /// Original likelihood identity/measure declaration; content is not verified.
    pub fn objective(&self) -> &ObjectiveDescription {
        &self.objective
    }
    /// Free position at which the curvature was evaluated.
    pub fn position(&self) -> &[f64] {
        &self.position
    }
    /// Full-rank factor and numerical conditioning policy in free coordinates.
    pub fn factor(&self) -> &InformationMatrix {
        &self.factor
    }
    /// Covariance in original free coordinates, not L-BFGS-scaled coordinates.
    pub fn free_covariance(&self) -> SquareMatrixRef<'_, RowMajor> {
        SquareMatrixRef::new(self.factor.dimension(), &self.free)
            .expect("validated covariance shape")
    }
    /// Named physical-coordinate delta covariance, stored as a full row-major
    /// matrix in parameter metadata order. It may be singular on a simplex.
    pub fn constrained_covariance(&self) -> &NamedOutput<Vec<f64>> {
        &self.constrained
    }
}
/// Evaluates explicit potential HVPs of the fit's **no-Jacobian likelihood
/// objective**, checks its numerical information and applies the analytic chart
/// Jacobian. Uses one checked fused evaluation and d checked basis-vector HVPs.
/// No finite differences, sampling-density correction or quasi-Newton history.
///
/// The caller must supply the same mathematical objective, not a transformed
/// sampling target. An exact fresh value/score replay detects many mismatches,
/// but cannot establish identity of arbitrary callbacks or Hessians. Numerical
/// stationarity/conditioning is not a statistical regularity certificate.
/// Projection is the first-order delta method; chart curvature is already in
/// the free HVP. No claim of exact nonlinear uncertainty propagation is made.
///
/// # Errors
/// Rejects non-likelihood/nonstationary/dimensionally inconsistent fits, a
/// non-replaying objective, bad derivatives, non-SPD/ill-conditioned information
/// or unrepresentable projection. Errors/panics retain the original fit unchanged;
/// no partial covariance is returned. Original backend sources stay typed.
pub fn observed_information<T: PotentialHessianVector + ?Sized, A>(
    target: &T,
    fit: &NamedFit<A>,
    options: InformationOptions,
) -> Result<ObservedInformation, ObservedInformationError<T::Error>> {
    options.validate()?;
    let r = fit.result();
    let d = r.dimension;
    let valid_size = |rows: usize, cols: usize| {
        rows.checked_mul(cols)
            .is_some_and(|n| n <= isize::MAX as usize / size_of::<f64>())
    };
    let m = fit.parameters().coordinates().len();
    let c = r
        .candidate
        .as_ref()
        .ok_or(ObservedInformationError::InvalidFit)?;
    if r.objective.kind != ObjectiveKind::Likelihood
        || r.termination != OptimizationTermination::ScoreTolerance
        || [
            &r.objective.measure,
            &r.objective.model,
            &r.objective.data,
            &r.objective.source,
        ]
        .iter()
        .any(|s| s.is_empty())
        || d == 0
        || target.dimension() != d
        || !valid_size(d, d)
        || !valid_size(m, d)
        || !valid_size(m, m)
        || r.options.scales().is_some_and(|s| s.len() != d)
        || c.gradient.iter().enumerate().any(|(i, g)| {
            let score = g * r.options.scales().map_or(1., |s| s[i]);
            !score.is_finite() || score.abs() > r.options.score_tolerance()
        })
    {
        return Err(ObservedInformationError::InvalidFit);
    }
    // Exact replay is an operational identity check, not an approximate score test.
    let mut gradient = vec![0.; d];
    let value = evaluate(target, &c.position, &mut gradient)?;
    if target.dimension() != d {
        return Err(ObservedInformationError::InvalidFit);
    }
    if value.to_bits() != c.value.to_bits()
        || gradient
            .iter()
            .zip(&c.gradient)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(ObservedInformationError::ObjectiveMismatch);
    }
    let mut hessian = vec![0.; d * d];
    let mut basis = vec![0.; d];
    let mut column = vec![0.; d];
    for j in 0..d {
        if target.dimension() != d {
            return Err(ObservedInformationError::InvalidFit);
        }
        basis[j] = 1.;
        evaluate_hvp(target, &c.position, &basis, &mut column)
            .map_err(|source| ObservedInformationError::Hvp { column: j, source })?;
        if target.dimension() != d {
            return Err(ObservedInformationError::InvalidFit);
        }
        for i in 0..d {
            hessian[i * d + j] = column[i];
        }
        basis[j] = 0.;
    }
    let matrix = SquareMatrixRef::new(d, &hessian).map_err(InformationError::from)?;
    let factor = InformationMatrix::new(matrix, options)?;
    let free = factor.covariance()?;
    let layout = fit.parameters().layout();
    let mut jacobian = vec![0.; m * d];
    let mut physical_basis = vec![0.; m];
    let mut scratch = vec![0.; layout.scratch_dimension()];
    for i in 0..m {
        physical_basis[i] = 1.;
        layout.pullback_composition(
            &c.position,
            &physical_basis,
            &mut jacobian[i * d..(i + 1) * d],
            &mut scratch,
        )?;
        physical_basis[i] = 0.;
    }
    let covariance = factor.project(m, &jacobian)?;
    let constrained = NamedOutput::new(
        fit.parameters().coordinates().to_vec().into(),
        layout.clone(),
        covariance,
    )?;
    Ok(ObservedInformation {
        objective: r.objective.clone(),
        position: c.position.clone(),
        factor,
        free,
        constrained,
    })
}
