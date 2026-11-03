use super::RegressionData;
use alea_math::{
    least_squares::{self, LeastSquares, LeastSquaresError},
    numeric::{CompensatedSum, euclidean_norm},
};
use statrs::distribution::{ContinuousCDF, StudentsT, StudentsTError};
use thiserror::Error;

/// Explicit numerical and marginal confidence-interval policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianIntervalOptions {
    /// Two-sided marginal confidence level strictly between zero and one.
    pub confidence_level: f64,
    /// Relative singular-value rank cutoff in (0,1), dependent on design units.
    pub relative_rank_tolerance: f64,
    /// Relative adjusted-response norm cutoff in (0,1) for vanishing residuals.
    pub relative_residual_tolerance: f64,
    /// Allowed relative Student-t survival-probability replay error, in (0,1).
    /// Backend replay is a numerical check, not an independent error enclosure.
    pub quantile_relative_tolerance: f64,
}
/// Failure to obtain full-rank, representable Gaussian coefficient intervals.
#[derive(Debug, Error)]
pub enum GaussianIntervalError {
    /// Invalid confidence level or numerical policy.
    #[error("invalid Gaussian interval policy")]
    Policy,
    /// No coefficients, positive residual degrees of freedom or admitted scale.
    #[error("Gaussian coefficient intervals require full-rank nonvanishing residual evidence")]
    Unqualified,
    /// Original checked SVD/shape/rank/covariance failure.
    #[error("Gaussian interval least squares failed: {0}")]
    LeastSquares(#[from] LeastSquaresError),
    /// Original Student-t construction failure.
    #[error("Gaussian interval Student-t construction failed: {0}")]
    Quantile(#[from] StudentsTError),
    /// Nonrepresentable variance, quantile, width, or distinct finite endpoints.
    #[error("Gaussian interval calculation is not representable or fails quantile replay")]
    Numerical,
}
/// Marginal Student-t coefficient intervals for independent homoskedastic
/// Gaussian errors, a full-rank fixed design and fixed known offsets. Coefficients
/// are unconstrained; no weights, priors or penalties. Calling this method declares
/// those assumptions; data alone cannot establish them. Not simultaneous bands,
/// prediction intervals, sigma intervals, or intervals for an arbitrary NamedFit.
#[derive(Debug, Clone)]
pub struct GaussianCoefficientIntervals {
    /// Original SVD/rank/residual evidence; scale_candidate still uses n, not n-p.
    pub assessment: GaussianAssessment,
    /// Explicit supplied numerical/confidence policy.
    pub options: GaussianIntervalOptions,
    /// RSS/(n-p), separately from the likelihood's ML residual variance.
    pub residual_variance: f64,
    /// Full row-major coefficient covariance, s^2 times inverse design Gram.
    /// Computed from the SVD, not by forming or inverting normal equations.
    pub covariance: Vec<f64>,
    /// Positive finite coefficient standard errors, in supplied column order.
    pub standard_errors: Vec<f64>,
    /// Positive Student-t critical value with n-p residual degrees of freedom.
    pub critical_value: f64,
    /// Finite [lower, upper] pairs in supplied coefficient-column order.
    pub bounds: Vec<[f64; 2]>,
}
/// Numerical Gaussian estimator warning, not a symbolic existence certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianIssue {
    /// Some coefficient directions were discarded under the declared SVD cutoff.
    NumericalRankDeficiency,
    /// Residual norm is at/below its declared relative-data threshold. This includes
    /// exact and near-perfect fits; no positive residual-scale estimate is qualified.
    VanishingResidual,
    /// Positive residual norm divided by sqrt(n) is not representable positively.
    ScaleNotRepresentable,
}
/// Least-squares evidence and residual-scale assessment for unconstrained Gaussian ML.
/// Numerical rank and vanishing-residual cutoffs are caller choices, not rigorous
/// roundoff enclosures. Offset-adjusted responses are used; no priors are included.
#[derive(Debug, Clone)]
pub struct GaussianAssessment {
    /// Truncated-SVD coefficients and numerical rank, not normal-equation output.
    pub least_squares: LeastSquares,
    /// Absolute residual-norm cutoff in response units.
    pub residual_cutoff: f64,
    /// Declared relative response-norm tolerance.
    pub relative_residual_tolerance: f64,
    /// Residual norm/sqrt(n), only when above the cutoff and representable.
    /// This uses n, not residual degrees of freedom. It is a candidate from the
    /// retained singular subspace, not a finite-MLE certificate.
    pub scale_candidate: Option<f64>,
    /// All applicable warnings, including simultaneous rank and residual problems.
    pub issues: Vec<GaussianIssue>,
}
impl RegressionData<f64> {
    /// Computes numerical rank, least-squares coefficients and Gaussian ML scale.
    /// Requires the `faer` feature. No covariance or n-p residual variance is fabricated.
    /// # Errors
    /// Rejects tolerances outside (0,1), nonfinite adjusted responses and numerical
    /// factorization/residual failures. Coefficient constraints are not inspected.
    pub fn gaussian_assessment(
        &self,
        relative_rank_tolerance: f64,
        relative_residual_tolerance: f64,
    ) -> Result<GaussianAssessment, LeastSquaresError> {
        self.gaussian_fit(relative_rank_tolerance, relative_residual_tolerance, None)
            .map(|(assessment, _)| assessment)
    }
    /// Fits least squares and returns two-sided marginal Student-t intervals.
    /// Uses one thin SVD and RSS/(n-p) for coefficient covariance, never optimizer
    /// history or a density Jacobian. Intercept columns count in p. The fixed-design
    /// Gaussian assumptions on [`GaussianCoefficientIntervals`] are caller declarations.
    /// Numerical rank/residual cutoffs and backend roundoff limit the qualification;
    /// this is not an exact-rank, full-range or empirical-coverage certificate.
    /// # Errors
    /// Rejects invalid policy, no coefficients/n-p, deficient numerical rank,
    /// vanishing residuals, quantile replay failure and nonrepresentable covariance,
    /// residual variance or distinct finite interval endpoints. No partial report,
    /// ridge, rank-truncated covariance or silent parameter clipping is returned.
    pub fn gaussian_coefficient_intervals(
        &self,
        options: GaussianIntervalOptions,
    ) -> Result<GaussianCoefficientIntervals, GaussianIntervalError> {
        let valid = |x: f64| x.is_finite() && x > 0. && x < 1.;
        if ![
            options.confidence_level,
            options.relative_rank_tolerance,
            options.relative_residual_tolerance,
            options.quantile_relative_tolerance,
        ]
        .into_iter()
        .all(valid)
        {
            return Err(GaussianIntervalError::Policy);
        }
        let p = self.columns();
        let df = self.rows().checked_sub(p).filter(|&df| df > 0);
        let df = df
            .filter(|_| p > 0)
            .ok_or(GaussianIntervalError::Unqualified)?;
        let (assessment, covariance) = self.gaussian_fit(
            options.relative_rank_tolerance,
            options.relative_residual_tolerance,
            Some(df),
        )?;
        if !assessment.issues.is_empty() || assessment.scale_candidate.is_none() {
            return Err(GaussianIntervalError::Unqualified);
        }
        // The math boundary checks exact integer conversion of this divisor.
        let law = StudentsT::new(0., 1., df as f64)?;
        // Lower-tail evaluation avoids rounding (1 + confidence)/2 to one.
        let tail = (1. - options.confidence_level) * 0.5;
        let critical = -law.inverse_cdf(tail);
        let replay_error = (law.sf(critical) - tail) / tail;
        if !critical.is_finite()
            || critical <= 0.
            || !replay_error.is_finite()
            || replay_error.abs() > options.quantile_relative_tolerance
        {
            return Err(GaussianIntervalError::Numerical);
        }
        let residual_scale = assessment.least_squares.residual_norm / (df as f64).sqrt();
        let residual_variance = residual_scale * residual_scale;
        if !residual_variance.is_finite() || residual_variance <= 0. {
            return Err(GaussianIntervalError::Numerical);
        }
        let mut standard_errors = Vec::with_capacity(p);
        let mut bounds = Vec::with_capacity(p);
        for (i, &coefficient) in assessment.least_squares.coefficients.iter().enumerate() {
            let se = covariance[i * p + i].sqrt();
            let width = critical * se;
            let lower = coefficient - width;
            let upper = coefficient + width;
            if !se.is_finite()
                || se <= 0.
                || !lower.is_finite()
                || !upper.is_finite()
                || lower >= coefficient
                || upper <= coefficient
            {
                return Err(GaussianIntervalError::Numerical);
            }
            standard_errors.push(se);
            bounds.push([lower, upper]);
        }
        Ok(GaussianCoefficientIntervals {
            assessment,
            options,
            residual_variance,
            covariance,
            standard_errors,
            critical_value: critical,
            bounds,
        })
    }
    fn gaussian_fit(
        &self,
        relative_rank_tolerance: f64,
        relative_residual_tolerance: f64,
        divisor: Option<usize>,
    ) -> Result<(GaussianAssessment, Vec<f64>), LeastSquaresError> {
        if !relative_residual_tolerance.is_finite()
            || relative_residual_tolerance <= 0.
            || relative_residual_tolerance >= 1.
        {
            return Err(LeastSquaresError::Shape);
        }
        let response: Vec<_> = self
            .observations()
            .iter()
            .zip(self.offsets())
            .map(|(y, o)| {
                let mut sum = CompensatedSum::default();
                sum.add(*y);
                sum.add(-o);
                sum.total()
            })
            .collect();
        let (fit, covariance) = match divisor {
            Some(divisor) => least_squares::solve_with_residual_covariance(
                self.rows(),
                self.columns(),
                self.design(),
                &response,
                relative_rank_tolerance,
                divisor,
            )?,
            None => (
                least_squares::solve(
                    self.rows(),
                    self.columns(),
                    self.design(),
                    &response,
                    relative_rank_tolerance,
                )?,
                Vec::new(),
            ),
        };
        let response_norm = euclidean_norm(&response);
        let cutoff = relative_residual_tolerance * response_norm;
        if !cutoff.is_finite() {
            return Err(LeastSquaresError::Numerical);
        }
        let mut issues = Vec::new();
        if fit.numerical_rank < self.columns() {
            issues.push(GaussianIssue::NumericalRankDeficiency);
        }
        let sigma = fit.residual_norm / (self.rows() as f64).sqrt();
        let scale_candidate = if fit.residual_norm <= cutoff {
            issues.push(GaussianIssue::VanishingResidual);
            None
        } else if !sigma.is_finite() || sigma <= 0. {
            issues.push(GaussianIssue::ScaleNotRepresentable);
            None
        } else {
            Some(sigma)
        };
        Ok((
            GaussianAssessment {
                least_squares: fit,
                residual_cutoff: cutoff,
                relative_residual_tolerance,
                scale_candidate,
                issues,
            },
            covariance,
        ))
    }
}
