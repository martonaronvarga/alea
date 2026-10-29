//! Estimator existence checks for unconstrained fixed-design likelihoods.
//! Priors, coefficient constraints and nonlinear/shared coefficient expressions
//! change the problem and are not inspected here. Finite offsets do not alter
//! recession directions. A failed direction check never establishes overlap.
#![deny(missing_docs)]
use super::{RegressionData, RegressionDataError, finite};
use alea_math::exact::dot_sign;
use std::cmp::Ordering;

/// Likelihood improvement along an unbounded coefficient ray.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecessionKind {
    /// Every signed Bernoulli margin strictly improves.
    CompleteSeparation,
    /// At least one Bernoulli margin improves and the rest stay fixed or improve.
    QuasiCompleteSeparation,
    /// Zero-count rates decrease while all positive-count predictors stay fixed.
    PoissonBoundary,
}
/// Verified strict likelihood improvement along beta+t*direction, t>=0.
/// Exact dyadic dot signs certify the stored data/direction, not pre-conversion
/// real-world measurements. Fields are read-only to retain that relationship.
#[derive(Debug, Clone)]
pub struct RecessionWitness {
    kind: RecessionKind,
    direction: Vec<f64>,
    improving_rows: usize,
    flat_rows: usize,
}
impl RecessionWitness {
    /// Which likelihood geometry was established.
    pub fn kind(&self) -> RecessionKind {
        self.kind
    }
    /// Checked coefficient direction in supplied column order.
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }
    /// Rows with strictly improving likelihood.
    pub fn improving_rows(&self) -> usize {
        self.improving_rows
    }
    /// Rows unchanged along this direction.
    pub fn flat_rows(&self) -> usize {
        self.flat_rows
    }
}
/// A direction is either a witness or inconclusive about existence.
#[derive(Debug, Clone)]
pub enum DirectionAssessment {
    /// No finite likelihood maximizer exists under the module's assumptions.
    Witness(RecessionWitness),
    /// This direction is not a strict improving ray. Other directions may be;
    /// this result is not evidence of overlap or a finite maximizer.
    NotWitness {
        /// Rows violating the required sign/equality conditions.
        violating_rows: usize,
        /// Rows improving under the required sign condition.
        improving_rows: usize,
        /// Rows unchanged.
        flat_rows: usize,
    },
}
impl RegressionData<bool> {
    /// Checks signed margins (2*y-1)*X*d >=0 with at least one strict margin.
    /// # Errors
    /// Rejects wrong dimension/nonfinite direction before computing exact signs.
    pub fn bernoulli_recession(
        &self,
        direction: &[f64],
    ) -> Result<DirectionAssessment, RegressionDataError> {
        assess(
            self,
            direction,
            |row, sign| {
                if self.observations()[row] {
                    sign
                } else {
                    sign.reverse()
                }
            },
            false,
        )
    }
    /// Screens only +/- individual columns. None is unresolved, not overlap.
    /// # Errors
    /// Propagates invalid direction/data shape errors.
    pub fn bernoulli_coordinate_recession(
        &self,
    ) -> Result<Option<RecessionWitness>, RegressionDataError> {
        coordinate_screen(self.columns(), |d| self.bernoulli_recession(d))
    }
}
impl RegressionData<u64> {
    /// Checks X*d=0 on positive counts and X*d<=0 on zero counts, strictly
    /// negative for at least one zero. It does not infer existence from all-zero
    /// responses alone: a direction must actually be feasible for the design.
    /// # Errors
    /// Rejects wrong dimension/nonfinite direction before computing exact signs.
    pub fn poisson_recession(
        &self,
        direction: &[f64],
    ) -> Result<DirectionAssessment, RegressionDataError> {
        assess(
            self,
            direction,
            |row, sign| {
                if self.observations()[row] > 0 {
                    if sign == Ordering::Equal {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                } else {
                    sign.reverse()
                }
            },
            true,
        )
    }
    /// Screens only +/- individual columns. None does not establish existence.
    /// # Errors
    /// Propagates invalid direction/data shape errors.
    pub fn poisson_coordinate_recession(
        &self,
    ) -> Result<Option<RecessionWitness>, RegressionDataError> {
        coordinate_screen(self.columns(), |d| self.poisson_recession(d))
    }
}
fn assess<Y>(
    data: &RegressionData<Y>,
    direction: &[f64],
    mut signed: impl FnMut(usize, Ordering) -> Ordering,
    poisson: bool,
) -> Result<DirectionAssessment, RegressionDataError> {
    if direction.len() != data.columns() {
        return Err(RegressionDataError::Shape);
    }
    finite(direction, "direction")?;
    let (mut improving, mut flat, mut violating) = (0, 0, 0);
    for row in 0..data.rows() {
        let sign = dot_sign(
            &data.design()[row * data.columns()..(row + 1) * data.columns()],
            direction,
        )
        .expect("validated finite equal-length data and direction");
        match signed(row, sign) {
            Ordering::Greater => improving += 1,
            Ordering::Equal => flat += 1,
            Ordering::Less => violating += 1,
        }
    }
    if violating == 0 && improving > 0 {
        Ok(DirectionAssessment::Witness(RecessionWitness {
            kind: if poisson {
                RecessionKind::PoissonBoundary
            } else if flat == 0 {
                RecessionKind::CompleteSeparation
            } else {
                RecessionKind::QuasiCompleteSeparation
            },
            direction: direction.to_vec(),
            improving_rows: improving,
            flat_rows: flat,
        }))
    } else {
        Ok(DirectionAssessment::NotWitness {
            violating_rows: violating,
            improving_rows: improving,
            flat_rows: flat,
        })
    }
}
fn coordinate_screen(
    columns: usize,
    mut check: impl FnMut(&[f64]) -> Result<DirectionAssessment, RegressionDataError>,
) -> Result<Option<RecessionWitness>, RegressionDataError> {
    let mut direction = vec![0.; columns];
    for j in 0..columns {
        for sign in [1., -1.] {
            direction[j] = sign;
            if let DirectionAssessment::Witness(w) = check(&direction)? {
                return Ok(Some(w));
            }
        }
        direction[j] = 0.;
    }
    Ok(None)
}
#[cfg(feature = "faer")]
mod gaussian;
#[cfg(feature = "faer")]
pub use gaussian::{GaussianAssessment, GaussianIssue};
