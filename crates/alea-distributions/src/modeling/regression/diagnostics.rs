//! Estimator existence checks for unconstrained fixed-design likelihoods.
//! Priors, coefficient constraints and nonlinear/shared coefficient expressions
//! change the problem and are not inspected here. Finite offsets do not alter
//! recession directions. A failed direction check never establishes overlap.
#![deny(missing_docs)]
use super::{RegressionData, RegressionDataError, finite};
pub use alea_math::exact::ConeSearchOptions;
use alea_math::exact::{ConeSearch, IntegerDirection, dot_sign, strict_cone_direction};
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
/// Verified improving ray with exact integer coordinates from general search.
/// The kind describes this ray, not whether another completely separating ray
/// exists. Recheck against the original data when retaining a saved direction.
#[derive(Debug, Clone)]
pub struct ExactRecessionWitness {
    kind: RecessionKind,
    direction: IntegerDirection,
    improving_rows: usize,
    flat_rows: usize,
}
impl ExactRecessionWitness {
    /// Likelihood improvement along the returned ray.
    pub fn kind(&self) -> RecessionKind {
        self.kind
    }
    /// Exact coordinates; rounding them to f64 can destroy equality constraints.
    pub fn direction(&self) -> &IntegerDirection {
        &self.direction
    }
    /// Rows with strictly improving likelihood.
    pub fn improving_rows(&self) -> usize {
        self.improving_rows
    }
    /// Rows whose likelihood is unchanged along the ray.
    pub fn flat_rows(&self) -> usize {
        self.flat_rows
    }
}
/// Existence of an unconstrained fixed-design Bernoulli-logit or Poisson-log MLE.
/// Applies to the stored finite design and fixed offsets, with independent free
/// coefficients and no priors, penalties, weights, or nonlinear coefficient maps.
#[derive(Debug, Clone)]
pub enum MleExistence {
    /// Exact elimination rules out every strict improving recession direction.
    /// A finite maximum is attained; coefficients may be nonunique through the
    /// design nullspace. This does not compute estimates or standard errors.
    FiniteMaximum,
    /// An exact ray proves that no finite maximum is attained.
    NoFiniteMaximum(ExactRecessionWitness),
    /// The exact search reached a declared resource limit; existence is unknown.
    Unresolved,
}
impl RegressionData<bool> {
    /// Searches all coefficient directions for complete or quasi separation.
    /// Finite existence is proved only after exact infeasibility of the entire
    /// strict recession system; exceeding the budget remains unresolved.
    /// # Errors
    /// Propagates malformed matrix errors (validated data normally exclude them).
    pub fn bernoulli_mle_existence(
        &self,
        options: ConeSearchOptions,
    ) -> Result<MleExistence, RegressionDataError> {
        if self.columns() == 0 {
            return Ok(MleExistence::FiniteMaximum);
        }
        if self.columns() > options.max_columns || self.rows() >= options.max_rows {
            return Ok(MleExistence::Unresolved);
        }
        let mut signed = self.design().to_vec();
        for (row, y) in self.observations().iter().enumerate() {
            if !y {
                for x in &mut signed[row * self.columns()..(row + 1) * self.columns()] {
                    *x = -*x;
                }
            }
        }
        search(self.columns(), &signed, &[], options, false)
    }
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
    /// Searches the full Poisson recession cone, including all positive-count
    /// equality constraints. All-zero responses are assessed using the design.
    /// # Errors
    /// Rejects counts not exactly representable by the associated likelihood,
    /// and propagates malformed matrix errors.
    pub fn poisson_mle_existence(
        &self,
        options: ConeSearchOptions,
    ) -> Result<MleExistence, RegressionDataError> {
        let mut required = 1_usize;
        for (index, &y) in self.observations().iter().enumerate() {
            if y > (1_u64 << 53) {
                return Err(RegressionDataError::CountPrecision { index });
            }
            required = required.saturating_add(if y == 0 { 1 } else { 2 });
        }
        if self.columns() == 0 {
            return Ok(MleExistence::FiniteMaximum);
        }
        if self.columns() > options.max_columns || required > options.max_rows {
            return Ok(MleExistence::Unresolved);
        }
        let (mut zero, mut positive) = (Vec::new(), Vec::new());
        for (row, &y) in self.observations().iter().enumerate() {
            let x = &self.design()[row * self.columns()..(row + 1) * self.columns()];
            if y == 0 {
                zero.extend(x.iter().map(|v| -v));
            } else {
                positive.extend_from_slice(x);
            }
        }
        search(self.columns(), &zero, &positive, options, true)
    }
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
fn search(
    columns: usize,
    inequalities: &[f64],
    equalities: &[f64],
    options: ConeSearchOptions,
    poisson: bool,
) -> Result<MleExistence, RegressionDataError> {
    match strict_cone_direction(columns, inequalities, equalities, options)
        .map_err(|_| RegressionDataError::Shape)?
    {
        ConeSearch::NoImprovingDirection => Ok(MleExistence::FiniteMaximum),
        ConeSearch::BudgetExceeded => Ok(MleExistence::Unresolved),
        ConeSearch::Direction(direction) => {
            let mut improving = 0;
            // A strict cone direction is impossible in zero dimensions.
            let mut flat = equalities.len() / columns;
            for row in inequalities.chunks_exact(columns) {
                match direction.dot_sign(row).expect("finite validated cone row") {
                    Ordering::Greater => improving += 1,
                    Ordering::Equal => flat += 1,
                    Ordering::Less => {
                        unreachable!("exact cone reconstruction violated an inequality")
                    }
                }
            }
            assert!(improving > 0, "exact cone ray must strictly improve");
            assert!(
                equalities
                    .chunks_exact(columns)
                    .all(|row| direction.dot_sign(row) == Ok(Ordering::Equal))
            );
            Ok(MleExistence::NoFiniteMaximum(ExactRecessionWitness {
                kind: if poisson {
                    RecessionKind::PoissonBoundary
                } else if flat == 0 {
                    RecessionKind::CompleteSeparation
                } else {
                    RecessionKind::QuasiCompleteSeparation
                },
                direction,
                improving_rows: improving,
                flat_rows: flat,
            }))
        }
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
