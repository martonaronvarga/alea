//! Explicit likelihood-preserving coefficient coordinates for fixed designs.
use super::{RegressionData, RegressionDataError, finite};
use alea_math::numeric::CompensatedSum;
use std::sync::Arc;

/// Reversible coefficient coordinates for a design with one explicit unit intercept.
/// For a non-intercept column, `z_j=(x_j-center_j)/scale_j`,
/// `theta_j=scale_j*beta_j`, and `theta_intercept=beta_intercept+sum(center_j*beta_j)`.
/// Thus both designs define the same linear predictor in real arithmetic.
/// This map does not transform priors, constraints or density measures for you.
#[derive(Debug, Clone)]
pub struct CoefficientStandardization {
    intercept: usize,
    centers: Vec<f64>,
    scales: Vec<f64>,
}
impl CoefficientStandardization {
    /// Index of the caller-supplied column containing exactly one in every row.
    pub fn intercept_column(&self) -> usize {
        self.intercept
    }
    /// Population means; zero at the intercept column.
    pub fn centers(&self) -> &[f64] {
        &self.centers
    }
    /// Population standard deviations; one for the intercept and constant columns.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }
    /// Maps original coefficients to centered/scaled coordinates, in column order.
    /// # Errors
    /// Rejects wrong dimensions, nonfinite inputs or unrepresentable coefficients.
    pub fn to_standardized(&self, coefficients: &[f64]) -> Result<Vec<f64>, RegressionDataError> {
        self.check(coefficients)?;
        let mut result: Vec<_> = coefficients
            .iter()
            .zip(&self.scales)
            .map(|(b, s)| b * s)
            .collect();
        let mut intercept = CompensatedSum::default();
        intercept.add(coefficients[self.intercept]);
        for (j, &b) in coefficients.iter().enumerate() {
            if j != self.intercept {
                intercept.add_product(self.centers[j], b);
            }
        }
        result[self.intercept] = intercept.total();
        finite(&result, "standardized coefficient")?;
        Ok(result)
    }
    /// Maps centered/scaled coefficients back to the original design's column order.
    /// Floating-point rounding can limit the accuracy of large cancelling coefficients;
    /// evaluate predictions using the same accurate linear-predictor primitive.
    /// # Errors
    /// Rejects wrong dimensions, nonfinite inputs or unrepresentable coefficients.
    pub fn to_original(&self, coefficients: &[f64]) -> Result<Vec<f64>, RegressionDataError> {
        self.check(coefficients)?;
        let mut result: Vec<_> = coefficients
            .iter()
            .zip(&self.scales)
            .map(|(b, s)| b / s)
            .collect();
        finite(&result, "original coefficient")?;
        let mut intercept = CompensatedSum::default();
        intercept.add(coefficients[self.intercept]);
        for (j, &b) in result.iter().enumerate() {
            if j != self.intercept {
                intercept.add_product(-self.centers[j], b);
            }
        }
        result[self.intercept] = intercept.total();
        finite(&result, "original coefficient")?;
        Ok(result)
    }
    fn check(&self, coefficients: &[f64]) -> Result<(), RegressionDataError> {
        if coefficients.len() != self.scales.len() {
            return Err(RegressionDataError::Shape);
        }
        finite(coefficients, "coefficient")
    }
}

/// Centered/scaled design and its explicit coefficient map. Observations and
/// fixed offsets are shared with the original data. No columns are discarded.
#[derive(Debug, Clone)]
pub struct StandardizedRegressionData<Y> {
    data: RegressionData<Y>,
    map: CoefficientStandardization,
}
impl<Y> StandardizedRegressionData<Y> {
    /// Data to use with coefficients in the standardized coordinates.
    pub fn data(&self) -> &RegressionData<Y> {
        &self.data
    }
    /// Coefficient conversion; this is not a Bayesian density adapter.
    pub fn coefficient_map(&self) -> &CoefficientStandardization {
        &self.map
    }
    /// Recovers the transformed data and coefficient map without copying observations.
    pub fn into_parts(self) -> (RegressionData<Y>, CoefficientStandardization) {
        (self.data, self.map)
    }
}
impl<Y> RegressionData<Y> {
    /// Centers non-intercept columns and divides by population standard deviation.
    /// The selected intercept column must consist of exact ones. Constant columns
    /// become zero, with scale one; no rank truncation or ridge penalty is applied.
    /// This is explicit coordinate preconditioning for likelihood fitting, not an
    /// automatic transformation of priors, constraints or optimizer stopping rules.
    /// # Errors
    /// Rejects a missing/non-unit intercept or nonrepresentable centering/scaling arithmetic.
    pub fn standardize(
        &self,
        intercept_column: usize,
    ) -> Result<StandardizedRegressionData<Y>, RegressionDataError> {
        if intercept_column >= self.columns()
            || self
                .design()
                .chunks_exact(self.columns())
                .any(|r| r[intercept_column] != 1.)
        {
            return Err(RegressionDataError::Intercept {
                column: intercept_column,
            });
        }
        let (rows, columns) = (self.rows(), self.columns());
        let mut centers = vec![0.; columns];
        let mut scales = vec![1.; columns];
        let mut design = self.design().to_vec();
        for j in 0..columns {
            if j == intercept_column {
                continue;
            }
            let first = self.design()[j];
            if self
                .design()
                .chunks_exact(columns)
                .all(|row| row[j] == first)
            {
                centers[j] = first;
                for i in 0..rows {
                    design[i * columns + j] = 0.;
                }
                continue;
            }
            let mut mean = CompensatedSum::default();
            for i in 0..rows {
                mean.add(self.design()[i * columns + j] / rows as f64);
            }
            centers[j] = mean.total();
            let mut maximum = 0_f64;
            for i in 0..rows {
                let index = i * columns + j;
                design[index] -= centers[j];
                maximum = maximum.max(design[index].abs());
            }
            if maximum > 0. {
                let mut squared = CompensatedSum::default();
                for i in 0..rows {
                    let x = design[i * columns + j] / maximum;
                    squared.add_product(x, x);
                }
                scales[j] = maximum * (squared.total() / rows as f64).sqrt();
                if !scales[j].is_finite() || scales[j] <= 0. {
                    return Err(RegressionDataError::NonFinite {
                        field: "standardization scale",
                        index: j,
                    });
                }
                for i in 0..rows {
                    design[i * columns + j] /= scales[j];
                }
            }
        }
        finite(&centers, "standardization mean")?;
        finite(&design, "standardized predictor")?;
        Ok(StandardizedRegressionData {
            data: RegressionData {
                columns,
                design: design.into(),
                observations: Arc::clone(&self.observations),
                offsets: Arc::clone(&self.offsets),
            },
            map: CoefficientStandardization {
                intercept: intercept_column,
                centers,
                scales,
            },
        })
    }
}
