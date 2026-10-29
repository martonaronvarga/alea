//! Exact signs of finite binary64 dot products using dyadic integer arithmetic.
#![deny(missing_docs)]
use num_bigint::{BigInt, Sign};
use std::cmp::Ordering;
use thiserror::Error;
mod cone;
pub use cone::{ConeSearch, ConeSearchOptions, IntegerDirection, strict_cone_direction};
/// Invalid operands for an exact dot sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ExactDotError {
    /// Operand dimensions differ.
    #[error("exact dot operand dimensions differ")]
    Shape,
    /// Infinity and NaN are not finite dyadic rationals.
    #[error("nonfinite exact dot operand")]
    NonFinite,
}
/// Returns the exact sign of the real dot product of the stored binary64 values.
/// No floating-point products are formed: cancellation, underflow and overflow
/// in a rounded dot do not change this result. Empty operands have sign Equal.
/// This is a cold, allocating predicate, not a hot-path summation replacement.
/// It certifies the stored values, not decimals before their conversion to f64.
/// # Errors
/// Rejects dimension mismatch and nonfinite operands.
pub fn dot_sign(left: &[f64], right: &[f64]) -> Result<Ordering, ExactDotError> {
    if left.len() != right.len() {
        return Err(ExactDotError::Shape);
    }
    if crate::numeric::first_nonfinite(left).is_some()
        || crate::numeric::first_nonfinite(right).is_some()
    {
        return Err(ExactDotError::NonFinite);
    }
    let mut sum = BigInt::from(0_u8);
    for (&a, &b) in left.iter().zip(right) {
        let (ma, ea) = dyadic(a);
        let (mb, eb) = dyadic(b);
        if ma == 0 || mb == 0 {
            continue;
        }
        // Every product is an integer multiple of 2^-2148. The largest
        // mantissa product needs 106 bits; the nonnegative shift is <=4090.
        let term = BigInt::from(u128::from(ma) * u128::from(mb)) << (ea + eb + 2148) as usize;
        if (a.to_bits() ^ b.to_bits()) >> 63 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
    }
    Ok(match sum.sign() {
        Sign::Minus => Ordering::Less,
        Sign::NoSign => Ordering::Equal,
        Sign::Plus => Ordering::Greater,
    })
}
fn dyadic(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1_u64 << 52), exponent - 1075)
    }
}
