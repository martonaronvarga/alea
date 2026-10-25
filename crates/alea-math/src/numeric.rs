#[inline(always)]
pub fn positive_finite(x: f64) -> f64 {
    x.max(f64::MIN_POSITIVE)
}

#[inline(always)]
pub fn finite_or_neg_inf(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::NEG_INFINITY }
}

#[inline(always)]
pub fn log_accept_ratio(current: f64, proposal: f64) -> f64 {
    if proposal.is_finite() {
        proposal - current
    } else {
        f64::NEG_INFINITY
    }
}

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

#[inline(always)]
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[inline(always)]
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
pub fn log_sigmoid(x: f64) -> f64 {
    -softplus(-x)
}

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
