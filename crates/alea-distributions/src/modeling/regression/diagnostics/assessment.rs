//! Named fixed-design estimator evidence. No optimizer or runtime dependency.
use super::{ConeSearchOptions, MleExistence, RegressionData, RegressionDataError};
use thiserror::Error;

/// Invalid evidence request. No coefficient columns or observations are dropped.
#[derive(Debug, Error)]
pub enum RegressionAssessmentError {
    /// Supply one nonempty, unique coefficient name per design column, in order.
    #[error("coefficient names must be nonempty, unique and match design columns")]
    CoefficientNames,
    /// The family-specific data or exact-search inputs were invalid.
    #[error(transparent)]
    Data(#[from] RegressionDataError),
    /// Optional numerical rank or Gaussian least-squares assessment failed.
    #[cfg(feature = "faer")]
    #[error(transparent)]
    Numerical(#[from] alea_math::least_squares::LeastSquaresError),
}

/// Family-specific evidence, independent of whether optimization converged.
#[derive(Debug, Clone)]
pub enum RegressionEvidence {
    /// Numerical least-squares and residual-scale evidence, not an exact certificate.
    #[cfg(feature = "faer")]
    Gaussian(super::GaussianAssessment),
    /// Exact finite-attainment decision or explicit unresolved search budget.
    BernoulliLogit(MleExistence),
    /// Exact finite-attainment decision or explicit unresolved search budget.
    PoissonLog(MleExistence),
}

/// Numerical design-rank evidence under a declared relative singular-value cutoff.
/// This is not exact algebraic rank, a condition-number guarantee or covariance.
#[derive(Debug, Clone)]
pub struct DesignRank {
    /// Number of rows in the assessed design.
    pub rows: usize,
    /// Number of coefficient columns, including any caller-supplied intercept.
    pub columns: usize,
    /// Singular values retained under the declared cutoff.
    pub numerical_rank: usize,
    /// Relative cutoff applied to singular values of the scaled design.
    pub relative_tolerance: f64,
}

/// Evidence for independent, unconstrained fixed-design coefficients, fixed finite
/// offsets, and no priors, penalties or weights. Names declare the design-column
/// order; the caller must match that order and these assumptions to the fitted
/// objective. No inspection of an arbitrary expression graph is performed.
#[derive(Debug, Clone)]
pub struct RegressionAssessment {
    names: Vec<String>,
    evidence: RegressionEvidence,
    search: Option<ConeSearchOptions>,
    rank: Option<DesignRank>,
}
impl RegressionAssessment {
    /// Explicit coefficient names in design-column order.
    pub fn coefficient_names(&self) -> &[String] {
        &self.names
    }
    /// Gaussian numerical assessment or GLM finite-MLE evidence.
    pub fn evidence(&self) -> &RegressionEvidence {
        &self.evidence
    }
    /// Declared exact-search budget, absent for Gaussian numerical assessment.
    pub fn search_options(&self) -> Option<ConeSearchOptions> {
        self.search
    }
    /// Optional numerical rank. Absence means not assessed, not full rank.
    pub fn design_rank(&self) -> Option<&DesignRank> {
        self.rank.as_ref()
    }
}
fn names(columns: usize, supplied: &[&str]) -> Result<Vec<String>, RegressionAssessmentError> {
    if supplied.len() != columns
        || supplied
            .iter()
            .enumerate()
            .any(|(i, n)| n.is_empty() || supplied[..i].contains(n))
    {
        return Err(RegressionAssessmentError::CoefficientNames);
    }
    Ok(supplied.iter().map(|n| (*n).to_owned()).collect())
}
impl RegressionData<bool> {
    /// Assesses finite MLE attainment with explicitly named coefficient columns.
    /// A finite maximum need not have unique coefficients or be found by an optimizer.
    /// # Errors
    /// Rejects invalid coefficient names and propagates exact-search input errors.
    pub fn assess_bernoulli(
        &self,
        coefficients: &[&str],
        options: ConeSearchOptions,
    ) -> Result<RegressionAssessment, RegressionAssessmentError> {
        let names = names(self.columns(), coefficients)?;
        Ok(RegressionAssessment {
            names,
            evidence: RegressionEvidence::BernoulliLogit(self.bernoulli_mle_existence(options)?),
            search: Some(options),
            rank: None,
        })
    }
    /// Also computes numerical design rank through the optional SVD backend.
    /// # Errors
    /// Propagates name, exact-search, tolerance and factorization errors.
    #[cfg(feature = "faer")]
    pub fn assess_bernoulli_with_rank(
        &self,
        coefficients: &[&str],
        options: ConeSearchOptions,
        relative_rank_tolerance: f64,
    ) -> Result<RegressionAssessment, RegressionAssessmentError> {
        let mut assessment = self.assess_bernoulli(coefficients, options)?;
        assessment.rank = Some(design_rank(self, relative_rank_tolerance)?);
        Ok(assessment)
    }
}
impl RegressionData<u64> {
    /// Assesses finite MLE attainment with explicitly named coefficient columns.
    /// Offsets are log exposures, not exposures; priors and constraints are excluded.
    /// # Errors
    /// Rejects invalid names/counts and propagates exact-search input errors.
    pub fn assess_poisson(
        &self,
        coefficients: &[&str],
        options: ConeSearchOptions,
    ) -> Result<RegressionAssessment, RegressionAssessmentError> {
        let names = names(self.columns(), coefficients)?;
        Ok(RegressionAssessment {
            names,
            evidence: RegressionEvidence::PoissonLog(self.poisson_mle_existence(options)?),
            search: Some(options),
            rank: None,
        })
    }
    /// Also computes numerical design rank through the optional SVD backend.
    /// # Errors
    /// Propagates name, exact-search, tolerance and factorization errors.
    #[cfg(feature = "faer")]
    pub fn assess_poisson_with_rank(
        &self,
        coefficients: &[&str],
        options: ConeSearchOptions,
        relative_rank_tolerance: f64,
    ) -> Result<RegressionAssessment, RegressionAssessmentError> {
        let mut assessment = self.assess_poisson(coefficients, options)?;
        assessment.rank = Some(design_rank(self, relative_rank_tolerance)?);
        Ok(assessment)
    }
}
#[cfg(feature = "faer")]
fn design_rank<Y>(
    data: &RegressionData<Y>,
    tolerance: f64,
) -> Result<DesignRank, alea_math::least_squares::LeastSquaresError> {
    // Reuse the checked scaled SVD. Zero responses produce no statistical estimate.
    let fit = alea_math::least_squares::solve(
        data.rows(),
        data.columns(),
        data.design(),
        &vec![0.; data.rows()],
        tolerance,
    )?;
    Ok(DesignRank {
        rows: data.rows(),
        columns: data.columns(),
        numerical_rank: fit.numerical_rank,
        relative_tolerance: tolerance,
    })
}
#[cfg(feature = "faer")]
impl RegressionData<f64> {
    /// Named Gaussian least-squares and ML residual-scale evidence. Uses n for
    /// the ML scale, not n-rank; no uncertainty or exact rank claim is attached.
    /// # Errors
    /// Rejects invalid names/tolerances and propagates least-squares failures.
    pub fn assess_gaussian(
        &self,
        coefficients: &[&str],
        relative_rank_tolerance: f64,
        relative_residual_tolerance: f64,
    ) -> Result<RegressionAssessment, RegressionAssessmentError> {
        let names = names(self.columns(), coefficients)?;
        let evidence =
            self.gaussian_assessment(relative_rank_tolerance, relative_residual_tolerance)?;
        let rank = DesignRank {
            rows: self.rows(),
            columns: self.columns(),
            numerical_rank: evidence.least_squares.numerical_rank,
            relative_tolerance: relative_rank_tolerance,
        };
        Ok(RegressionAssessment {
            names,
            evidence: RegressionEvidence::Gaussian(evidence),
            rank: Some(rank),
            search: None,
        })
    }
}
