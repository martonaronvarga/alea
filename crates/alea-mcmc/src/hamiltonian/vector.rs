//! Runtime-vectorized coordinate operations. Keep multiplication and addition
//! separate: FMA would change the integrator's established floating-point map.

pub(super) fn admit<const POSITION: bool>(
    p: &[f64],
    v: &[f64],
    q: &[f64],
) -> Option<Option<[f64; 2]>> {
    if p.len() < 4 {
        return v.iter().all(|v| v.is_finite()).then_some(None);
    }
    let norms = if p.len() >= 16 {
        pulp::Arch::new().dispatch(Admit::<POSITION> { p, v, q })
    } else {
        pulp::WithSimd::with_simd(Admit::<POSITION> { p, v, q }, pulp::Scalar)
    };
    norms.map(Some)
}

struct Admit<'a, const POSITION: bool> {
    p: &'a [f64],
    v: &'a [f64],
    q: &'a [f64],
}
impl<const POSITION: bool> pulp::WithSimd for Admit<'_, POSITION> {
    type Output = Option<[f64; 2]>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let a = if POSITION { self.p } else { self.v };
        let b = if POSITION { self.q } else { self.p };
        let (a, at) = S::as_simd_f64s(a);
        let (b, bt) = S::as_simd_f64s(b);
        let (v, vt) = S::as_simd_f64s(self.v);
        let infinity = simd.splat_f64s(f64::INFINITY);
        let mut valid = simd.less_than_f64s(simd.splat_f64s(0.0), infinity);
        let (mut am, mut bm) = (simd.splat_f64s(0.0), simd.splat_f64s(0.0));
        for ((a, b), v) in a.iter().zip(b).zip(v) {
            am = simd.max_f64s(am, simd.abs_f64s(*a));
            bm = simd.max_f64s(bm, simd.abs_f64s(*b));
            valid = simd.and_m64s(valid, simd.less_than_f64s(simd.abs_f64s(*v), infinity));
        }
        let lanes = size_of::<S::f64s>() / size_of::<f64>();
        if simd.first_true_m64s(simd.not_m64s(valid)) < lanes {
            return None;
        }
        let (mut am, mut bm) = (simd.reduce_max_f64s(am), simd.reduce_max_f64s(bm));
        for ((a, b), v) in at.iter().zip(bt).zip(vt) {
            if !v.is_finite() {
                return None;
            }
            am = am.max(a.abs());
            bm = bm.max(b.abs());
        }
        Some([am, bm])
    }
}

#[inline]
pub(super) fn add_scaled(out: &mut [f64], source: &[f64], scale: f64) -> Option<usize> {
    if out.len() < 16 {
        for (out, x) in out.iter_mut().zip(source) {
            *out += scale * x;
        }
        out.iter().position(|p| !p.is_finite())
    } else {
        pulp::Arch::new().dispatch(AddScaled { out, source, scale })
    }
}

#[inline]
pub(super) fn add_scaled_into<const CHECK: bool>(
    out: &mut [f64],
    base: &[f64],
    source: &[f64],
    scale: f64,
) -> Option<usize> {
    if out.len() < 16 {
        for ((out, base), x) in out.iter_mut().zip(base).zip(source) {
            *out = base + scale * x;
        }
        if CHECK {
            out.iter().position(|p| !p.is_finite())
        } else {
            None
        }
    } else {
        pulp::Arch::new().dispatch(AddScaledInto::<CHECK> {
            out,
            base,
            source,
            scale,
        })
    }
}

struct AddScaled<'a> {
    out: &'a mut [f64],
    source: &'a [f64],
    scale: f64,
}
impl pulp::WithSimd for AddScaled<'_> {
    type Output = Option<usize>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let all = self.out;
        let mut valid = simd.less_than_f64s(simd.splat_f64s(0.0), simd.splat_f64s(f64::INFINITY));
        let infinity = simd.splat_f64s(f64::INFINITY);
        let (out, ot) = S::as_mut_simd_f64s(all);
        let (source, st) = S::as_simd_f64s(self.source);
        let scale = simd.splat_f64s(self.scale);
        for (out, source) in out.iter_mut().zip(source) {
            *out = simd.add_f64s(*out, simd.mul_f64s(scale, *source));
            valid = simd.and_m64s(valid, simd.less_than_f64s(simd.abs_f64s(*out), infinity));
        }
        for (out, source) in ot.iter_mut().zip(st) {
            *out += self.scale * source;
        }
        let lanes = size_of::<S::f64s>() / size_of::<f64>();
        if simd.first_true_m64s(simd.not_m64s(valid)) < lanes {
            all.iter().position(|p| !p.is_finite())
        } else {
            ot.iter()
                .position(|p| !p.is_finite())
                .map(|i| out.len() * lanes + i)
        }
    }
}

struct AddScaledInto<'a, const CHECK: bool> {
    out: &'a mut [f64],
    base: &'a [f64],
    source: &'a [f64],
    scale: f64,
}
impl<const CHECK: bool> pulp::WithSimd for AddScaledInto<'_, CHECK> {
    type Output = Option<usize>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let all = self.out;
        let mut valid = simd.less_than_f64s(simd.splat_f64s(0.0), simd.splat_f64s(f64::INFINITY));
        let infinity = simd.splat_f64s(f64::INFINITY);
        let (out, ot) = S::as_mut_simd_f64s(all);
        let (base, bt) = S::as_simd_f64s(self.base);
        let (source, st) = S::as_simd_f64s(self.source);
        let scale = simd.splat_f64s(self.scale);
        for ((out, base), source) in out.iter_mut().zip(base).zip(source) {
            *out = simd.add_f64s(*base, simd.mul_f64s(scale, *source));
            if CHECK {
                valid = simd.and_m64s(valid, simd.less_than_f64s(simd.abs_f64s(*out), infinity));
            }
        }
        for ((out, base), source) in ot.iter_mut().zip(bt).zip(st) {
            *out = base + self.scale * source;
        }
        let lanes = size_of::<S::f64s>() / size_of::<f64>();
        if CHECK && simd.first_true_m64s(simd.not_m64s(valid)) < lanes {
            all.iter().position(|p| !p.is_finite())
        } else if CHECK {
            ot.iter()
                .position(|p| !p.is_finite())
                .map(|i| out.len() * lanes + i)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_cache_admission_checks_every_velocity_lane_and_tail() {
        for n in 1..=81 {
            let p = vec![2.0; n];
            let q = vec![-3.0; n];
            let mut v = vec![4.0; n];
            assert_eq!(
                admit::<false>(&p, &v, &q),
                Some((n >= 4).then_some([4.0, 2.0]))
            );
            assert_eq!(
                admit::<true>(&p, &v, &q),
                Some((n >= 4).then_some([2.0, 3.0]))
            );
            for i in 0..n {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    v[i] = bad;
                    assert_eq!(admit::<false>(&p, &v, &q), None, "n{n} i{i}");
                    assert_eq!(admit::<true>(&p, &v, &q), None, "position n{n} i{i}");
                }
                v[i] = 4.0;
            }
        }
    }
    #[test]
    fn vector_updates_keep_scalar_rounding_first_error_and_arbitrary_tails() {
        for n in 0..=97 {
            for offset in 0..4 {
                for scale in [0.0, -0.0, 1e-300, -0.25, f64::MAX] {
                    let source: Vec<_> =
                        (0..n + offset).map(|i| (i as f64 - 30.0) * 0.375).collect();
                    let base: Vec<_> = (0..n + offset).map(|i| (i as f64 - 20.0) * 0.125).collect();
                    let expected: Vec<_> = base[offset..]
                        .iter()
                        .zip(&source[offset..])
                        .map(|(b, x)| b + scale * x)
                        .collect();
                    let invalid = expected.iter().position(|x| !x.is_finite());
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    let mut out = base.clone();
                    assert_eq!(
                        add_scaled(&mut out[offset..], &source[offset..], scale),
                        invalid
                    );
                    assert_eq!(bits(&out[offset..]), bits(&expected));
                    out.fill(123.0);
                    assert_eq!(
                        add_scaled_into::<true>(
                            &mut out[offset..],
                            &base[offset..],
                            &source[offset..],
                            scale
                        ),
                        invalid
                    );
                    assert_eq!(bits(&out[offset..]), bits(&expected));
                    assert_eq!(
                        add_scaled_into::<false>(
                            &mut out[offset..],
                            &base[offset..],
                            &source[offset..],
                            scale
                        ),
                        None
                    );
                    assert_eq!(bits(&out[offset..]), bits(&expected));
                }
            }
        }
    }
    #[test]
    fn vector_updates_do_not_contract_multiply_and_add() {
        for n in [2, 16, 17, 64] {
            let eps = 2.0_f64.powi(-27);
            let source = vec![1.0 - eps; n];
            let mut out = vec![-1.0; n];
            assert_eq!(add_scaled(&mut out, &source, 1.0 + eps), None);
            assert!(out.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
            assert_eq!(
                add_scaled_into::<true>(&mut out, &vec![-1.0; n], &source, 1.0 + eps),
                None
            );
            assert!(out.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
        }
    }
}
