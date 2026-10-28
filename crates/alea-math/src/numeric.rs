//! Scalar numerical functions and explicit nonfinite-value detection.
//!
//! Elementary functions follow their floating-point domains: they propagate NaN
//! and expose boundary infinities rather than converting errors into support
//! zeros or clamping parameters. Model and target boundaries perform validation.
#![deny(missing_docs)]

/// Logistic function, evaluated without overflowing the exponential.
///
/// Infinite inputs give the limiting probabilities zero and one; NaN propagates.
/// Extreme finite inputs can also round to an endpoint.
#[inline(always)]
pub fn sigmoid(theta: f64) -> f64 {
    if theta >= 0.0 {
        let e = (-theta).exp();
        1.0 / (1.0 + e)
    } else {
        let e = theta.exp();
        e / (1.0 + e)
    }
}

/// Log odds of a probability in the open unit interval.
///
/// Zero and one give negative and positive infinity. Values outside the closed
/// unit interval, including infinite inputs, give NaN. This is an elementary
/// function, not a checked probability constructor.
#[inline(always)]
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// Evaluates `ln(1 + exp(x))` with tail branches that avoid overflow.
///
/// Negative infinity gives zero, positive infinity gives infinity, and NaN
/// propagates. `ln_1p` preserves the small positive tail without an asymptotic cutoff.
#[inline(always)]
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Log logistic probability without first rounding the probability to an endpoint.
///
/// Negative infinity gives negative infinity, positive infinity gives zero,
/// and NaN propagates.
#[inline(always)]
pub fn log_sigmoid(x: f64) -> f64 {
    -softplus(-x)
}

/// Log complementary logistic probability without subtracting a rounded probability.
///
/// Negative infinity gives zero, positive infinity gives negative infinity,
/// and NaN propagates.
#[inline(always)]
pub fn log1m_sigmoid(x: f64) -> f64 {
    -softplus(x)
}
/// Index of the first NaN or infinite entry, including arbitrary slice tails.
/// Finite blocks are checked together; the first failing block is then searched
/// in coordinate order. This preserves error indexing without a branch per value.
#[inline]
pub fn first_nonfinite(values: &[f64]) -> Option<usize> {
    if values.len() < 8 {
        return values.iter().position(|x| !x.is_finite());
    }
    if values.len() >= 16 {
        return pulp::Arch::new().dispatch(FirstNonfinite(values));
    }
    let (blocks, tail) = values.as_chunks::<8>();
    for (i, v) in blocks.iter().enumerate() {
        if !(v[0].is_finite()
            & v[1].is_finite()
            & v[2].is_finite()
            & v[3].is_finite()
            & v[4].is_finite()
            & v[5].is_finite()
            & v[6].is_finite()
            & v[7].is_finite())
        {
            return v.iter().position(|x| !x.is_finite()).map(|j| 8 * i + j);
        }
    }
    tail.iter()
        .position(|x| !x.is_finite())
        .map(|j| 8 * blocks.len() + j)
}

struct FirstNonfinite<'a>(&'a [f64]);

impl pulp::WithSimd for FirstNonfinite<'_> {
    type Output = Option<usize>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let (head, tail) = S::as_simd_f64s(self.0);
        let lanes = size_of::<S::f64s>() / size_of::<f64>();
        let infinity = simd.splat_f64s(f64::INFINITY);
        let mut valid = [simd.less_than_f64s(simd.splat_f64s(0.0), infinity); 4];
        let (blocks, rest) = head.as_chunks::<4>();
        for block in blocks {
            for lane in 0..4 {
                valid[lane] = simd.and_m64s(
                    valid[lane],
                    simd.less_than_f64s(simd.abs_f64s(block[lane]), infinity),
                );
            }
        }
        for v in rest {
            valid[0] = simd.and_m64s(valid[0], simd.less_than_f64s(simd.abs_f64s(*v), infinity));
        }
        let all = simd.and_m64s(
            simd.and_m64s(valid[0], valid[1]),
            simd.and_m64s(valid[2], valid[3]),
        );
        if simd.first_true_m64s(simd.not_m64s(all)) < lanes {
            return self.0.iter().position(|v| !v.is_finite());
        }
        tail.iter()
            .position(|v| !v.is_finite())
            .map(|i| head.len() * lanes + i)
    }
}

#[cfg(test)]
mod finite_tests {
    use super::*;
    #[test]
    fn logistic_primitives_preserve_small_log_probabilities() {
        // Independently rounded high-precision values of log(1+exp(x)).
        let fixtures: [(f64, f64); 6] = [
            (-40.0, 4.248354255291589e-18),
            (-30.0, 9.357622968839737e-14),
            (-20.0, 2.061153620314381e-9),
            (-1.0, 0.31326168751822286),
            (0.0, std::f64::consts::LN_2),
            (1.0, 1.3132616875182228),
        ];
        for (x, expected) in fixtures {
            let actual = softplus(x);
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 2,
                "x={x}: {actual} != {expected}"
            );
            assert_eq!(log_sigmoid(-x), -actual);
            assert_eq!(log1m_sigmoid(x), -actual);
        }
        assert_eq!(softplus(f64::NEG_INFINITY), 0.0);
        assert_eq!(softplus(f64::INFINITY), f64::INFINITY);
        assert_eq!(softplus(f64::MAX), f64::MAX);
        assert!(softplus(f64::NAN).is_nan());
    }
    #[test]
    fn finite_check_preserves_first_error_across_lanes_tails_and_alignment() {
        // Keep every error position and alignment through scalar, chunk and
        // dispatch boundaries under the interpreter. Native tests exhaust the
        // wider grid; its cubic repeated scanning is prohibitively slow in Miri.
        let max_len = if cfg!(miri) { 33 } else { 129 };
        for n in 0..=max_len {
            for offset in 0..4 {
                let mut values = vec![f64::MAX; n + offset];
                assert_eq!(first_nonfinite(&values[offset..]), None);
                for i in 0..n {
                    for bad in [f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                        values[offset + i] = bad;
                        if i + 1 < n {
                            values[offset + i + 1] = f64::NAN;
                        }
                        assert_eq!(
                            first_nonfinite(&values[offset..]),
                            Some(i),
                            "n={n} offset={offset} i={i}"
                        );
                        values[offset + i] = f64::MAX;
                        if i + 1 < n {
                            values[offset + i + 1] = f64::MAX;
                        }
                    }
                }
            }
        }
    }
}
