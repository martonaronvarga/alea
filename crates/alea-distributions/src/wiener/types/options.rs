use super::*;

/// Canonical Wiener first-passage families in Stan's convention:
/// Wiener4 = standard four parameter (alpha, tau, beta, delta) wiener family
/// Wiener5 = Wiener4 + normal drift standard deviation (s_delta)
/// Wiener7 = Wiener5 + uniform variability widths for beta and tau
/// Reference:
/// <https://mc-stan.org/docs/functions-reference/positive_lower-bounded_distributions.html#wiener-first-passage-time-distribution>
#[derive(Debug, Clone, Copy)]
pub struct Wiener4;
pub struct Wiener5;
pub struct Wiener7;

/// Series branch selected by the Navarro-Fuss density representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesBranch {
    SmallTime,
    LargeTime,
}

/// Term-count diagnostics for Wiener series branch selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WienerBranchCounts {
    pub branch: SeriesBranch,
    pub k_small_density: usize,
    pub k_large_density: usize,
    pub k_small_grad_w: usize,
    pub k_large_grad_w: usize,
    pub k_small_used: usize,
    pub k_large_used: usize,
}

/// Numerical integration strategy for full Wiener models with variability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quadrature {
    /// Adaptive cubature over the active variability dimensions.
    Adaptive,
    /// Fixed Gauss-Legendre product rule. Supported orders are 1, 5, 7, 15, and 25.
    FixedGaussLegendre { order: usize },
}

/// Numerical options for checked user-facing Wiener evaluations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WienerOptions {
    /// Precision used for adaptive cubature, or as the default series precision.
    pub precision: f64,
    /// Optional precision for inner Wiener series when it should differ from `precision`.
    pub inner_precision: Option<f64>,
    /// Quadrature strategy for Wiener7 start-time/non-decision-time variability.
    pub quadrature: Quadrature,
}

impl Default for WienerOptions {
    fn default() -> Self {
        Self {
            precision: 1e-12,
            inner_precision: None,
            quadrature: Quadrature::Adaptive,
        }
    }
}

impl WienerOptions {
    #[must_use]
    pub fn new(precision: f64) -> Self {
        Self {
            precision,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_inner_precision(mut self, inner_precision: f64) -> Self {
        self.inner_precision = Some(inner_precision);
        self
    }

    #[must_use]
    pub fn with_quadrature(mut self, quadrature: Quadrature) -> Self {
        self.quadrature = quadrature;
        self
    }

    pub(in crate::wiener) fn series_precision(self) -> f64 {
        self.inner_precision.unwrap_or(self.precision)
    }

    pub(in crate::wiener) fn validate(self) -> Result<Self> {
        if !self.precision.is_finite() || self.precision <= 0.0 {
            return Err(ProbError::InvalidParameters(
                "WienerOptions precision must be finite and positive".to_string(),
            ));
        }
        if let Some(inner_precision) = self.inner_precision
            && (!inner_precision.is_finite() || inner_precision <= 0.0)
        {
            return Err(ProbError::InvalidParameters(
                "WienerOptions inner precision must be finite and positive".to_string(),
            ));
        }
        if let Quadrature::FixedGaussLegendre { order } = self.quadrature
            && !matches!(order, 1 | 5 | 7 | 15 | 25)
        {
            return Err(ProbError::InvalidParameters(
                "fixed Gauss-Legendre quadrature order must be one of 1, 5, 7, 15, or 25"
                    .to_string(),
            ));
        }
        Ok(self)
    }
}
