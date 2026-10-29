use super::RegressionData;
use alea_math::{
    least_squares::{self, LeastSquares, LeastSquaresError},
    numeric::{CompensatedSum, euclidean_norm},
};
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
        let fit = least_squares::solve(
            self.rows(),
            self.columns(),
            self.design(),
            &response,
            relative_rank_tolerance,
        )?;
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
        Ok(GaussianAssessment {
            least_squares: fit,
            residual_cutoff: cutoff,
            relative_residual_tolerance,
            scale_candidate,
            issues,
        })
    }
}
