//! Validated grid spacing and clamped linear interpolation, migrated from the
//! dormant DDM approximation sketch. No likelihood approximation is implied.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InterpolationError {
    #[error("interpolation needs at least two matching knots and values")]
    Shape,
    #[error("knots must be finite and strictly increasing")]
    Knots,
    #[error("interpolation values must be finite")]
    Values,
    #[error("grid spacing is not finite and positive")]
    Spacing,
    #[error("interpolation query must not be NaN")]
    Query,
}

/// Uniform-grid specification; does not allocate or materialize the grid.
#[derive(Debug, Clone, Copy)]
pub struct UniformGrid {
    min: f64,
    max: f64,
    points: usize,
    spacing: f64,
}
impl UniformGrid {
    /// Validates bounds, count, and representable positive spacing.
    ///
    /// # Errors
    /// Rejects non-finite/unordered bounds, fewer than two points, or spacing
    /// that overflows/underflows. Rounded materialized knots need independent
    /// validation; this specification does not promise every knot is distinct.
    pub fn new(min: f64, max: f64, points: usize) -> Result<Self, InterpolationError> {
        if points < 2 {
            return Err(InterpolationError::Shape);
        }
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(InterpolationError::Knots);
        }
        let count = (points - 1) as f64;
        let span = max - min;
        let spacing = if span.is_finite() {
            span / count
        } else {
            max / count - min / count
        };
        if !spacing.is_finite() || spacing <= 0.0 {
            return Err(InterpolationError::Spacing);
        }
        Ok(Self {
            min,
            max,
            points,
            spacing,
        })
    }
    pub fn min(self) -> f64 {
        self.min
    }
    pub fn max(self) -> f64 {
        self.max
    }
    pub fn points(self) -> usize {
        self.points
    }
    pub fn spacing(self) -> f64 {
        self.spacing
    }
}

/// Owns validated knots and values. Evaluations allocate nothing and use binary
/// search, clamping outside the domain to the nearest endpoint value.
#[derive(Debug, Clone)]
pub struct LinearSpline {
    knots: Vec<f64>,
    values: Vec<f64>,
}
impl LinearSpline {
    /// Takes ownership without copying or allocating.
    ///
    /// # Errors
    /// Rejects mismatched/short arrays, non-finite entries or unordered knots.
    pub fn new(knots: Vec<f64>, values: Vec<f64>) -> Result<Self, InterpolationError> {
        if knots.len() < 2 || knots.len() != values.len() {
            return Err(InterpolationError::Shape);
        }
        if knots.iter().any(|x| !x.is_finite()) || knots.windows(2).any(|w| w[0] >= w[1]) {
            return Err(InterpolationError::Knots);
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(InterpolationError::Values);
        }
        Ok(Self { knots, values })
    }
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Evaluates the piecewise linear curve; infinities clamp like any other
    /// out-of-domain query. Exact knots return their stored values.
    ///
    /// # Errors
    /// Rejects NaN queries. Finite extreme knots/values avoid subtraction overflow.
    pub fn evaluate(&self, x: f64) -> Result<f64, InterpolationError> {
        if x.is_nan() {
            return Err(InterpolationError::Query);
        }
        if x <= self.knots[0] {
            return Ok(self.values[0]);
        }
        let last = self.knots.len() - 1;
        if x >= self.knots[last] {
            return Ok(self.values[last]);
        }
        let hi = self.knots.partition_point(|knot| *knot <= x);
        let lo = hi - 1;
        // Equality here is intentional: preserve the value stored at a knot.
        if x == self.knots[lo] {
            return Ok(self.values[lo]);
        }
        let span = self.knots[hi] - self.knots[lo];
        let weight = if span.is_finite() {
            (x - self.knots[lo]) / span
        } else {
            (0.5 * x - 0.5 * self.knots[lo]) / (0.5 * self.knots[hi] - 0.5 * self.knots[lo])
        };
        let a = self.values[lo];
        let b = self.values[hi];
        let value = if a.is_sign_negative() == b.is_sign_negative() {
            a + weight * (b - a)
        } else {
            (1.0 - weight) * a + weight * b
        };
        Ok(value.clamp(a.min(b), a.max(b)))
    }
}
