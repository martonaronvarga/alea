//! Bounded Fourier–Motzkin elimination over exact integer inequalities.
use super::{ExactDotError, dyadic};
use num_bigint::{BigInt, Sign};
use std::{cmp::Ordering, collections::BTreeSet};

/// Resource limits for an allocating, exact cone feasibility calculation.
/// Exceeding any limit yields an unresolved result, never infeasibility.
#[derive(Debug, Clone, Copy)]
pub struct ConeSearchOptions {
    /// Maximum input dimension.
    pub max_columns: usize,
    /// Maximum total input and generated inequalities, including duplicates.
    pub max_rows: usize,
    /// Maximum bits per normalized row entry or reconstructed integer.
    /// Temporary products can use twice this many bits, plus summation carry.
    pub max_coefficient_bits: u64,
}
impl Default for ConeSearchOptions {
    fn default() -> Self {
        Self {
            max_columns: 32,
            max_rows: 16_384,
            max_coefficient_bits: 8192,
        }
    }
}
/// An exact integer ray; multiplying it by a positive number preserves its signs.
#[derive(Debug, Clone)]
pub struct IntegerDirection(Vec<BigInt>);
impl IntegerDirection {
    /// Integer coordinates in the original column order. No f64 rounding is used.
    pub fn coefficients(&self) -> &[BigInt] {
        &self.0
    }

    /// Exact sign of a stored binary64 row dotted with this ray.
    /// # Errors
    /// Rejects wrong dimensions and nonfinite row values.
    pub fn dot_sign(&self, row: &[f64]) -> Result<Ordering, ExactDotError> {
        if row.len() != self.0.len() {
            return Err(ExactDotError::Shape);
        }
        let row = integer_row(row)?;
        Ok(dot(&row, &self.0).cmp(&BigInt::from(0)))
    }
}
/// Outcome of a bounded exact search for A*d>=0, E*d=0, and sum(A*d)>0.
#[derive(Debug, Clone)]
pub enum ConeSearch {
    /// A feasible integer ray, including at least one strict inequality.
    Direction(IntegerDirection),
    /// Exact elimination proves that no strict improving ray exists.
    NoImprovingDirection,
    /// A declared resource limit was reached. Feasibility remains unresolved.
    BudgetExceeded,
}

/// Searches a homogeneous polyhedral cone for a strictly improving direction.
/// `inequalities` and `equalities` are flat row-major matrices with `columns`
/// columns. All inequality rows must have nonnegative dot products and at least
/// one must be positive; equality rows must have zero dot products.
///
/// Every finite input is converted exactly to an integer on scale 2^-1074.
/// Positive row scaling removes common factors. Homogeneity makes strict
/// improvement equivalent to sum(A*d)>=1 for these rescaled rows. Exact
/// Fourier–Motzkin elimination proves feasibility or infeasibility; rational
/// back substitution is scaled to an integer ray. No tolerance or floating-point
/// linear solver is involved. Elimination may grow exponentially; limits bound
/// stored rows, dimension, and integer sizes. This is a cold diagnostic.
/// # Errors
/// Rejects malformed matrix storage or nonfinite input.
pub fn strict_cone_direction(
    columns: usize,
    inequalities: &[f64],
    equalities: &[f64],
    options: ConeSearchOptions,
) -> Result<ConeSearch, ExactDotError> {
    if (columns == 0 && (!inequalities.is_empty() || !equalities.is_empty()))
        || (columns != 0
            && (!inequalities.len().is_multiple_of(columns)
                || !equalities.len().is_multiple_of(columns)))
    {
        return Err(ExactDotError::Shape);
    }
    if inequalities
        .iter()
        .chain(equalities)
        .any(|x| !x.is_finite())
    {
        return Err(ExactDotError::NonFinite);
    }
    if columns == 0 || inequalities.is_empty() {
        return Ok(ConeSearch::NoImprovingDirection);
    }
    if columns > options.max_columns {
        return Ok(ConeSearch::BudgetExceeded);
    }
    let Some(input_rows) = (equalities.len() / columns)
        .checked_mul(2)
        .and_then(|n| n.checked_add(inequalities.len() / columns))
        .and_then(|n| n.checked_add(1))
    else {
        return Ok(ConeSearch::BudgetExceeded);
    };
    if input_rows > options.max_rows {
        return Ok(ConeSearch::BudgetExceeded);
    }
    Ok(
        solve(columns, inequalities, equalities, options, input_rows)
            .unwrap_or(ConeSearch::BudgetExceeded),
    )
}

fn integer_row(row: &[f64]) -> Result<Vec<BigInt>, ExactDotError> {
    row.iter()
        .map(|&x| {
            if !x.is_finite() {
                return Err(ExactDotError::NonFinite);
            }
            let (mantissa, exponent) = dyadic(x);
            let n = BigInt::from(mantissa) << (exponent + 1074) as usize;
            Ok(if x.is_sign_negative() { -n } else { n })
        })
        .collect()
}
fn dot(a: &[BigInt], b: &[BigInt]) -> BigInt {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while b.sign() != Sign::NoSign {
        let r = a % &b;
        a = b;
        b = r;
    }
    if a.sign() == Sign::Minus { -a } else { a }
}
fn normalize(row: &mut [BigInt]) {
    let divisor = row.iter().cloned().fold(BigInt::from(0), gcd);
    if divisor.sign() != Sign::NoSign {
        for x in row {
            *x /= &divisor;
        }
    }
}
fn admit(mut row: Vec<BigInt>, rows: &mut BTreeSet<Vec<BigInt>>, bits: u64) -> Option<()> {
    // Remove common integer factors without changing an inequality's sign.
    normalize(&mut row);
    if row.iter().any(|x| x.bits() > bits) {
        return None;
    }
    rows.insert(row);
    Some(())
}
fn solve(
    p: usize,
    a: &[f64],
    e: &[f64],
    options: ConeSearchOptions,
    mut used: usize,
) -> Option<ConeSearch> {
    let zero = BigInt::from(0);
    let mut rows = BTreeSet::new();
    let mut strict = vec![zero.clone(); p + 1];
    for row in a.chunks_exact(p) {
        let mut r = integer_row(row).ok()?;
        normalize(&mut r);
        for (sum, x) in strict.iter_mut().zip(&r) {
            *sum += x;
        }
        // Bound running sums as well as normalized rows.
        if strict
            .iter()
            .any(|x| x.bits() > options.max_coefficient_bits)
        {
            return None;
        }
        r.push(zero.clone());
        admit(r, &mut rows, options.max_coefficient_bits)?;
    }
    strict[p] = BigInt::from(1);
    admit(strict, &mut rows, options.max_coefficient_bits)?;
    for row in e.chunks_exact(p) {
        let mut r = integer_row(row).ok()?;
        r.push(zero.clone());
        admit(r.clone(), &mut rows, options.max_coefficient_bits)?;
        admit(
            r.into_iter().map(|x| -x).collect(),
            &mut rows,
            options.max_coefficient_bits,
        )?;
    }
    let mut stages = Vec::new();
    for n in (1..=p).rev() {
        let current: Vec<_> = rows.into_iter().collect();
        let mut next = BTreeSet::new();
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        for r in &current {
            match r[n - 1].sign() {
                Sign::Plus => positive.push(r),
                Sign::Minus => negative.push(r),
                Sign::NoSign => {
                    let mut r = r.clone();
                    r.remove(n - 1);
                    admit(r, &mut next, options.max_coefficient_bits)?;
                }
            }
        }
        for lo in &positive {
            for hi in &negative {
                used = used.checked_add(1)?;
                if used > options.max_rows {
                    return None;
                }
                // (-hi[k])*lo + lo[k]*hi eliminates variable k using
                // only positive multiples, preserving the inequality.
                let mut r = Vec::with_capacity(n);
                for j in (0..=n).filter(|&j| j != n - 1) {
                    r.push(&lo[n - 1] * &hi[j] - &hi[n - 1] * &lo[j]);
                }
                admit(r, &mut next, options.max_coefficient_bits)?;
            }
        }
        // A zero left side with positive right side is an exact contradiction.
        if next
            .iter()
            .any(|r| r[..n - 1].iter().all(|x| x == &zero) && r[n - 1] > zero)
        {
            return Some(ConeSearch::NoImprovingDirection);
        }
        stages.push(current);
        rows = next;
    }
    let mut numerators = vec![zero.clone(); p];
    let mut denominator = BigInt::from(1);
    for (k, stage) in stages.iter().rev().enumerate() {
        let mut lower: Option<(BigInt, BigInt)> = None;
        let mut upper: Option<(BigInt, BigInt)> = None;
        for r in stage {
            if r[k] == zero {
                continue;
            }
            let mut b = &r[k + 1] * &denominator - dot(&r[..k], &numerators[..k]);
            if b.bits() > options.max_coefficient_bits {
                return None;
            }
            let a = if r[k].sign() == Sign::Minus {
                b = -b;
                -&r[k]
            } else {
                r[k].clone()
            };
            let bound = if r[k].sign() == Sign::Plus {
                &mut lower
            } else {
                &mut upper
            };
            let replace = bound.as_ref().is_none_or(|(old, divisor)| {
                let cmp = (&b * divisor).cmp(&(old * &a));
                if r[k].sign() == Sign::Plus {
                    cmp == Ordering::Greater
                } else {
                    cmp == Ordering::Less
                }
            });
            if replace {
                *bound = Some((b, a));
            }
        }
        let chosen = lower
            .filter(|(b, _)| b > &zero)
            .or_else(|| upper.filter(|(b, _)| b < &zero));
        if let Some((b, a)) = chosen {
            if denominator.bits() + a.bits() > options.max_coefficient_bits
                || numerators[..k]
                    .iter()
                    .any(|x| x.bits() + a.bits() > options.max_coefficient_bits)
            {
                return None;
            }
            denominator *= &a;
            for x in &mut numerators[..k] {
                *x *= &a;
            }
            numerators[k] = b;
        }
    }
    let divisor = numerators.iter().cloned().fold(zero, gcd);
    if divisor.sign() != Sign::NoSign {
        for x in &mut numerators {
            *x /= &divisor;
        }
    }
    Some(ConeSearch::Direction(IntegerDirection(numerators)))
}
