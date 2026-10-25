//! Sign-only turning arithmetic. Scaling and ordered sums are intentional:
//! reassociating a nearly cancelling dot product can change the sampled tree.

#[cfg(not(feature = "simd"))]
pub(super) use scalar as continues;
#[cfg(feature = "simd")]
pub(super) use vectorized as continues;

pub(super) fn scalar(
    rho: &mut [f64],
    a: &[f64],
    b: &[f64],
    left: &[f64],
    right: &[f64],
) -> Option<bool> {
    let mut scale: f64 = 0.0;
    let mut ls: f64 = 0.0;
    let mut rs: f64 = 0.0;
    for ((((r, a), b), l), v) in rho.iter_mut().zip(a).zip(b).zip(left).zip(right) {
        *r = a + b;
        if !r.is_finite() || !l.is_finite() || !v.is_finite() {
            return None;
        }
        scale = scale.max(r.abs());
        ls = ls.max(l.abs());
        rs = rs.max(v.abs());
    }
    if scale == 0.0 || ls == 0.0 || rs == 0.0 {
        return Some(false);
    }
    if let Some(continues) = certified_continues(rho, left, right, scale, ls, rs) {
        return Some(continues);
    }
    // Share the normalized momentum sum without changing either dot product's
    // ordered additions or replacing division by a reciprocal (scales may be
    // subnormal). Each coordinate needs three divisions instead of four.
    let mut left_dot = -0.0;
    let mut right_dot = -0.0;
    for ((l, r), p) in left.iter().zip(right).zip(rho.iter()) {
        let p = p / scale;
        left_dot += (l / ls) * p;
        right_dot += (r / rs) * p;
    }
    Some(left_dot > 0.0 && right_dot > 0.0)
}

// Most turns are far from cancellation. Certify the two signs without division;
// retain the original normalized, ordered predicate for every uncertain case.
// With n*eps < 1/64, rounding of each division/product and ordered sum is bounded
// by less than 8*n*eps times the absolute product sum. The factor32 covers both
// dot evaluations, rounding in the bound, and a conservative normal-range floor
// for all subnormal/underflow errors in the normalized computation. An overflow
// in a product, sum or bound cannot certify a sign and falls back as well.
#[inline]
fn certified_continues(
    rho: &[f64],
    left: &[f64],
    right: &[f64],
    scale: f64,
    ls: f64,
    rs: f64,
) -> Option<bool> {
    if rho.len() < 4 {
        return None;
    }
    // Independent lanes avoid a coordinate-long addition dependency. Certification
    // permits this association; uncertain signs still use ordered normalization.
    let (mut ld, mut rd, mut la, mut ra) = ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
    let (pc, pt) = rho.as_chunks::<4>();
    let (lc, lt) = left.as_chunks::<4>();
    let (rc, rt) = right.as_chunks::<4>();
    for ((p, l), r) in pc.iter().zip(lc).zip(rc) {
        for lane in 0..4 {
            let lp = l[lane] * p[lane];
            let rp = r[lane] * p[lane];
            ld[lane] += lp;
            rd[lane] += rp;
            la[lane] += lp.abs();
            ra[lane] += rp.abs();
        }
    }
    let (mut ldot, mut rdot, mut labs, mut rabs) = (
        ld.into_iter().sum::<f64>(),
        rd.into_iter().sum::<f64>(),
        la.into_iter().sum::<f64>(),
        ra.into_iter().sum::<f64>(),
    );
    for ((p, l), r) in pt.iter().zip(lt).zip(rt) {
        let lp = l * p;
        let rp = r * p;
        ldot += lp;
        rdot += rp;
        labs += lp.abs();
        rabs += rp.abs();
    }
    certify(rho.len(), ldot, rdot, labs, rabs, [scale, ls, rs])
}

#[inline]
fn certify(n: usize, ldot: f64, rdot: f64, labs: f64, rabs: f64, scales: [f64; 3]) -> Option<bool> {
    let [scale, ls, rs] = scales;
    let n = n as f64;
    if n * f64::EPSILON >= 1.0 / 64.0 {
        return None;
    }
    let factor = 32.0 * n;
    let bound = |absolute: f64, endpoint_scale: f64| {
        let product_scale = endpoint_scale * scale;
        factor * f64::EPSILON * absolute + factor * f64::MIN_POSITIVE * product_scale.max(1.0)
    };
    let lb = bound(labs, ls);
    let rb = bound(rabs, rs);
    if ldot.is_finite() && rdot.is_finite() && ldot.abs() > lb && rdot.abs() > rb {
        Some(ldot > 0.0 && rdot > 0.0)
    } else {
        None
    }
}

/// A supplied infinity-norm upper bound avoids materializing the sum and
/// scanning its maximum. Failed certification always returns to the original
/// checked, normalized arithmetic; even an infinite upper bound is harmless.
pub(super) fn bounded<const DIFFERENCE: bool>(
    a: &[f64],
    b: &[f64],
    left: &[f64],
    right: &[f64],
    scales: [f64; 3],
) -> Option<bool> {
    if a.len() >= 16 {
        let (ldot, rdot) = pulp::Arch::new().dispatch(Bounded::<DIFFERENCE> { a, b, left, right });
        let [scale, ls, rs] = scales;
        return certify(
            a.len(),
            ldot,
            rdot,
            a.len() as f64 * (scale * ls),
            a.len() as f64 * (scale * rs),
            scales,
        );
    }
    let (mut ld, mut rd) = ([0.0; 4], [0.0; 4]);
    let (ac, at) = a.as_chunks::<4>();
    let (bc, bt) = b.as_chunks::<4>();
    let (lc, lt) = left.as_chunks::<4>();
    let (rc, rt) = right.as_chunks::<4>();
    for (((a, b), l), r) in ac.iter().zip(bc).zip(lc).zip(rc) {
        for lane in 0..4 {
            let p = if DIFFERENCE {
                b[lane] - a[lane]
            } else {
                a[lane] + b[lane]
            };
            ld[lane] += l[lane] * p;
            rd[lane] += r[lane] * p;
        }
    }
    let (mut ldot, mut rdot) = (ld.into_iter().sum::<f64>(), rd.into_iter().sum::<f64>());
    for (((a, b), l), r) in at.iter().zip(bt).zip(lt).zip(rt) {
        let p = if DIFFERENCE { b - a } else { a + b };
        ldot += l * p;
        rdot += r * p;
    }
    let [scale, ls, rs] = scales;
    certify(
        a.len(),
        ldot,
        rdot,
        a.len() as f64 * (scale * ls),
        a.len() as f64 * (scale * rs),
        scales,
    )
}

struct Bounded<'a, const DIFFERENCE: bool> {
    a: &'a [f64],
    b: &'a [f64],
    left: &'a [f64],
    right: &'a [f64],
}

pub(super) fn bounded_three<const POSITION: bool>(
    x: [&[f64]; 4],
    v: [&[f64]; 4],
    scales: [[f64; 3]; 3],
) -> Option<bool> {
    let dots = pulp::Arch::new().dispatch(Three::<POSITION> { x, v });
    let mut valid = true;
    for (dots, scales) in dots.chunks_exact(2).zip(scales) {
        let [scale, ls, rs] = scales;
        valid &= certify(
            x[0].len(),
            dots[0],
            dots[1],
            x[0].len() as f64 * (scale * ls),
            x[0].len() as f64 * (scale * rs),
            scales,
        )?;
    }
    Some(valid)
}
struct Three<'a, const POSITION: bool> {
    x: [&'a [f64]; 4],
    v: [&'a [f64]; 4],
}
impl<const POSITION: bool> pulp::WithSimd for Three<'_, POSITION> {
    type Output = [f64; 6];
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let [(a, at), (b, bt), (c, ct), (d, dt)] = self.x.map(S::as_simd_f64s);
        let [(l, lt), (m, mt), (n, nt), (r, rt)] = self.v.map(S::as_simd_f64s);
        let mut sums = [simd.splat_f64s(0.0); 6];
        for (((((((a, b), c), d), l), m), n), r) in
            a.iter().zip(b).zip(c).zip(d).zip(l).zip(m).zip(n).zip(r)
        {
            let (x, y, z) = if POSITION {
                (
                    simd.sub_f64s(*d, *a),
                    simd.sub_f64s(*c, *a),
                    simd.sub_f64s(*d, *b),
                )
            } else {
                (
                    simd.add_f64s(*a, *b),
                    simd.add_f64s(*a, *c),
                    simd.add_f64s(*b, *d),
                )
            };
            for (s, (x, v)) in
                sums.iter_mut()
                    .zip([(x, *l), (x, *r), (y, *l), (y, *n), (z, *m), (z, *r)])
            {
                *s = simd.add_f64s(*s, simd.mul_f64s(x, v));
            }
        }
        let mut sums = sums.map(|s| simd.reduce_sum_f64s(s));
        for (((((((a, b), c), d), l), m), n), r) in at
            .iter()
            .zip(bt)
            .zip(ct)
            .zip(dt)
            .zip(lt)
            .zip(mt)
            .zip(nt)
            .zip(rt)
        {
            let (x, y, z) = if POSITION {
                (d - a, c - a, d - b)
            } else {
                (a + b, a + c, b + d)
            };
            for (s, (x, v)) in
                sums.iter_mut()
                    .zip([(x, *l), (x, *r), (y, *l), (y, *n), (z, *m), (z, *r)])
            {
                *s += x * v;
            }
        }
        sums
    }
}
impl<const DIFFERENCE: bool> pulp::WithSimd for Bounded<'_, DIFFERENCE> {
    type Output = (f64, f64);
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let (a, at) = S::as_simd_f64s(self.a);
        let (b, bt) = S::as_simd_f64s(self.b);
        let (l, lt) = S::as_simd_f64s(self.left);
        let (r, rt) = S::as_simd_f64s(self.right);
        let (mut ld, mut rd) = (simd.splat_f64s(0.0), simd.splat_f64s(0.0));
        for (((a, b), l), r) in a.iter().zip(b).zip(l).zip(r) {
            let p = if DIFFERENCE {
                simd.sub_f64s(*b, *a)
            } else {
                simd.add_f64s(*a, *b)
            };
            ld = simd.add_f64s(ld, simd.mul_f64s(*l, p));
            rd = simd.add_f64s(rd, simd.mul_f64s(*r, p));
        }
        let (mut ld, mut rd) = (simd.reduce_sum_f64s(ld), simd.reduce_sum_f64s(rd));
        for (((a, b), l), r) in at.iter().zip(bt).zip(lt).zip(rt) {
            let p = if DIFFERENCE { b - a } else { a + b };
            ld += l * p;
            rd += r * p;
        }
        (ld, rd)
    }
}

/// Endpoint scales were computed once after successful finite velocity admission.
pub(super) fn cached(
    rho: &mut [f64],
    a: &[f64],
    b: &[f64],
    left: &[f64],
    right: &[f64],
    ls: f64,
    rs: f64,
) -> Option<bool> {
    cached_combine::<false>(rho, a, b, left, right, ls, rs)
}

pub(super) fn cached_position(
    rho: &mut [f64],
    left_q: &[f64],
    right_q: &[f64],
    left_p: &[f64],
    right_p: &[f64],
    ls: f64,
    rs: f64,
) -> Option<bool> {
    cached_combine::<true>(rho, left_q, right_q, left_p, right_p, ls, rs)
}

fn cached_combine<const DIFFERENCE: bool>(
    rho: &mut [f64],
    a: &[f64],
    b: &[f64],
    left: &[f64],
    right: &[f64],
    ls: f64,
    rs: f64,
) -> Option<bool> {
    // Fuse sum admission, scale discovery and sign certification. The immutable
    // endpoint caches have already admitted every velocity coordinate as finite.
    let (mut ld, mut rd, mut scales) = ([0.0; 4], [0.0; 4], [0.0_f64; 4]);
    let (pc, pt) = rho.as_chunks_mut::<4>();
    let (ac, at) = a.as_chunks::<4>();
    let (bc, bt) = b.as_chunks::<4>();
    let (lc, lt) = left.as_chunks::<4>();
    let (rc, rt) = right.as_chunks::<4>();
    for ((((p, a), b), l), r) in pc.iter_mut().zip(ac).zip(bc).zip(lc).zip(rc) {
        for lane in 0..4 {
            p[lane] = if DIFFERENCE {
                b[lane] - a[lane]
            } else {
                a[lane] + b[lane]
            };
            scales[lane] = scales[lane].max(p[lane].abs());
            let lp = l[lane] * p[lane];
            let rp = r[lane] * p[lane];
            ld[lane] += lp;
            rd[lane] += rp;
        }
    }
    let (mut ldot, mut rdot) = (ld.into_iter().sum::<f64>(), rd.into_iter().sum::<f64>());
    let mut scale = scales.into_iter().fold(0.0_f64, f64::max);
    for ((((p, a), b), l), r) in pt.iter_mut().zip(at).zip(bt).zip(lt).zip(rt) {
        *p = if DIFFERENCE { b - a } else { a + b };
        scale = scale.max(p.abs());
        let lp = l * *p;
        let rp = r * *p;
        ldot += lp;
        rdot += rp;
    }
    // Hölder's inequality bounds the absolute product sum by n times the
    // product of the infinity norms. This avoids two coordinate-wise absolute
    // sums; the conservative certificate and ordered fallback are unchanged.
    let labs = rho.len() as f64 * (scale * ls);
    let rabs = rho.len() as f64 * (scale * rs);
    if let Some(result) = certify(rho.len(), ldot, rdot, labs, rabs, [scale, ls, rs]) {
        return Some(result);
    }
    // Nonfinite sums poison at least one raw dot. Validate before the ordered
    // fallback because max deliberately ignores NaNs.
    if rho.iter().any(|p| !p.is_finite()) {
        return None;
    }
    if scale == 0.0 || ls == 0.0 || rs == 0.0 {
        return Some(false);
    }
    let (mut ldot, mut rdot) = (-0.0, -0.0);
    for ((p, l), r) in rho.iter().zip(left).zip(right) {
        let p = p / scale;
        ldot += (l / ls) * p;
        rdot += (r / rs) * p;
    }
    Some(ldot > 0.0 && rdot > 0.0)
}

/// Position turning in whitened coordinates is displacement dotted with the
/// canonical momentum in the original coordinates, not with M^-1 p.
pub(super) fn position(
    delta: &mut [f64],
    left_q: &[f64],
    right_q: &[f64],
    left_p: &[f64],
    right_p: &[f64],
) -> Option<bool> {
    let (mut ld, mut rd, mut scales, mut ls, mut rs) =
        ([0.0; 4], [0.0; 4], [0.0_f64; 4], [0.0_f64; 4], [0.0_f64; 4]);
    let (dc, dt) = delta.as_chunks_mut::<4>();
    let (lc, lt) = left_q.as_chunks::<4>();
    let (rc, rt) = right_q.as_chunks::<4>();
    let (lp, lpt) = left_p.as_chunks::<4>();
    let (rp, rpt) = right_p.as_chunks::<4>();
    for ((((d, lq), rq), l), r) in dc.iter_mut().zip(lc).zip(rc).zip(lp).zip(rp) {
        for lane in 0..4 {
            d[lane] = rq[lane] - lq[lane];
            scales[lane] = scales[lane].max(d[lane].abs());
            ls[lane] = ls[lane].max(l[lane].abs());
            rs[lane] = rs[lane].max(r[lane].abs());
            ld[lane] += l[lane] * d[lane];
            rd[lane] += r[lane] * d[lane];
        }
    }
    let (mut ldot, mut rdot) = (ld.into_iter().sum::<f64>(), rd.into_iter().sum::<f64>());
    let (mut scale, mut ls, mut rs) = (
        scales.into_iter().fold(0.0_f64, f64::max),
        ls.into_iter().fold(0.0_f64, f64::max),
        rs.into_iter().fold(0.0_f64, f64::max),
    );
    for ((((d, lq), rq), l), r) in dt.iter_mut().zip(lt).zip(rt).zip(lpt).zip(rpt) {
        *d = rq - lq;
        scale = scale.max(d.abs());
        ls = ls.max(l.abs());
        rs = rs.max(r.abs());
        ldot += l * *d;
        rdot += r * *d;
    }
    let labs = delta.len() as f64 * (scale * ls);
    let rabs = delta.len() as f64 * (scale * rs);
    if let Some(result) = certify(delta.len(), ldot, rdot, labs, rabs, [scale, ls, rs]) {
        return Some(result);
    }
    if delta.iter().any(|d| !d.is_finite()) {
        return None;
    }
    if scale == 0.0 || ls == 0.0 || rs == 0.0 {
        return Some(false);
    }
    let (mut ldot, mut rdot) = (-0.0, -0.0);
    for ((d, l), r) in delta.iter().zip(left_p).zip(right_p) {
        let d = d / scale;
        ldot += (l / ls) * d;
        rdot += (r / rs) * d;
    }
    Some(ldot > 0.0 && rdot > 0.0)
}

#[cfg(test)]
mod scalar_tests {
    use super::{certified_continues, scalar};

    #[test]
    fn combined_certificates_match_all_three_independent_checked_predicates() {
        let mut bits = 53117_u64;
        let mut next = || {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            let exponent = [0, 1, 512, 900, 1023, 1100, 1536, 1900, 2046][(bits as usize >> 3) % 9];
            f64::from_bits((bits & (((1_u64 << 52) - 1) | (1_u64 << 63))) | exponent << 52)
        };
        let max = |v: &[f64]| v.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
        let mut certified = [0; 2];
        for case in 0..10000 {
            let n = 16 + case % 66;
            let x: [Vec<f64>; 4] = std::array::from_fn(|_| {
                (0..n)
                    .map(|_| {
                        let x = next();
                        if case % 2 == 0 {
                            x
                        } else {
                            ((x.to_bits() % 65536) as f64 - 32768.0) / 4096.0
                        }
                    })
                    .collect()
            });
            let v: [Vec<f64>; 4] = std::array::from_fn(|_| {
                (0..n)
                    .map(|_| {
                        let x = next();
                        if case % 2 == 0 {
                            x
                        } else {
                            ((x.to_bits() % 65536) as f64 - 32768.0) / 4096.0
                        }
                    })
                    .collect()
            });
            let xs = x.each_ref().map(|x| &x[..]);
            let vs = v.each_ref().map(|v| &v[..]);
            let norms = x.each_ref().map(|x| max(x));
            let vn = v.each_ref().map(|v| max(v));
            let expected = [
                scalar(&mut vec![0.0; n], xs[0], xs[1], vs[0], vs[3]),
                scalar(&mut vec![0.0; n], xs[0], xs[2], vs[0], vs[2]),
                scalar(&mut vec![0.0; n], xs[1], xs[3], vs[1], vs[3]),
            ];
            if let Some(actual) = super::bounded_three::<false>(
                xs,
                vs,
                [
                    [2.0 * norms[0].max(norms[1]), vn[0], vn[3]],
                    [2.0 * norms[0].max(norms[2]), vn[0], vn[2]],
                    [2.0 * norms[1].max(norms[3]), vn[1], vn[3]],
                ],
            ) {
                assert!(expected.iter().all(Option::is_some));
                certified[0] += 1;
                assert_eq!(
                    actual,
                    expected.iter().all(|v| *v == Some(true)),
                    "generalized case{case}"
                );
            }
            let expected = [
                super::position(&mut vec![0.0; n], xs[0], xs[3], vs[0], vs[3]),
                super::position(&mut vec![0.0; n], xs[0], xs[2], vs[0], vs[2]),
                super::position(&mut vec![0.0; n], xs[1], xs[3], vs[1], vs[3]),
            ];
            if let Some(actual) = super::bounded_three::<true>(
                xs,
                vs,
                [
                    [2.0 * norms[0].max(norms[3]), vn[0], vn[3]],
                    [2.0 * norms[0].max(norms[2]), vn[0], vn[2]],
                    [2.0 * norms[1].max(norms[3]), vn[1], vn[3]],
                ],
            ) {
                assert!(expected.iter().all(Option::is_some));
                certified[1] += 1;
                assert_eq!(
                    actual,
                    expected.iter().all(|v| *v == Some(true)),
                    "position case{case}"
                );
            }
        }
        assert!(
            certified.iter().all(|n| *n > 100),
            "nonvacuous fast paths: {certified:?}"
        );
    }

    #[test]
    fn uncertain_cancellation_and_overflow_keep_the_normalized_predicate() {
        assert_eq!(
            certified_continues(&[1.0, 1.0], &[1.0, -1.0], &[1.0, 1.0], 1.0, 1.0, 1.0),
            None
        );
        assert_eq!(
            certified_continues(
                &[f64::MAX, 0.0, 0.0, 0.0],
                &[f64::MAX, 0.0, 0.0, 0.0],
                &[1.0, 0.0, 0.0, 0.0],
                f64::MAX,
                f64::MAX,
                1.0
            ),
            None
        );
        // Normalized products would underflow despite representable raw dots.
        assert_eq!(
            certified_continues(
                &[1e100, 1e-200, 0.0, 0.0],
                &[0.0, 1e50, 1e100, 0.0],
                &[0.0, 1e50, 1e100, 0.0],
                1e100,
                1e100,
                1e100
            ),
            None
        );
        assert_eq!(
            certified_continues(
                &[2.0, 3.0, 0.0, 0.0],
                &[4.0, 5.0, 0.0, 0.0],
                &[6.0, 7.0, 0.0, 0.0],
                3.0,
                5.0,
                7.0
            ),
            Some(true)
        );
    }

    #[test]
    fn certified_signs_match_normalized_order_across_exponent_and_cancellation_grids() {
        let mut bits = 43001_u64;
        let mut next = || {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            let exponents = [0, 1, 8, 128, 512, 900, 1023, 1100, 1536, 1900, 2046];
            let exponent = exponents[(bits as usize >> 3) % exponents.len()];
            f64::from_bits((bits & (((1_u64 << 52) - 1) | (1_u64 << 63))) | (exponent << 52))
        };
        for case in 0..20_000 {
            let n = 1 + case % 65;
            let vectors: [Vec<f64>; 4] = std::array::from_fn(|_| (0..n).map(|_| next()).collect());
            let [a, b, left, right] = &vectors;
            let rho: Vec<_> = a.iter().zip(b).map(|(a, b)| a + b).collect();
            let expected = if rho.iter().any(|r| !r.is_finite()) {
                None
            } else {
                let max = |v: &[f64]| v.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
                let (scale, ls, rs) = (max(&rho), max(left), max(right));
                if scale == 0.0 || ls == 0.0 || rs == 0.0 {
                    Some(false)
                } else {
                    let (mut ldot, mut rdot) = (-0.0, -0.0);
                    for ((p, l), r) in rho.iter().zip(left).zip(right) {
                        let p = p / scale;
                        ldot += (l / ls) * p;
                        rdot += (r / rs) * p;
                    }
                    Some(ldot > 0.0 && rdot > 0.0)
                }
            };
            assert_eq!(
                scalar(&mut vec![0.0; n], a, b, left, right),
                expected,
                "case {case}"
            );
            let max = |v: &[f64]| v.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
            assert_eq!(
                super::cached(&mut vec![0.0; n], a, b, left, right, max(left), max(right)),
                expected,
                "cached case {case}"
            );
            let bound = 2.0 * max(a).max(max(b));
            let bounded =
                super::bounded::<false>(a, b, left, right, [bound, max(left), max(right)])
                    .map(Some)
                    .unwrap_or_else(|| {
                        super::cached(&mut vec![0.0; n], a, b, left, right, max(left), max(right))
                    });
            assert_eq!(bounded, expected, "bounded case {case}");
            let delta: Vec<_> = b.iter().zip(a).map(|(b, a)| b - a).collect();
            let positional = if delta.iter().any(|d| !d.is_finite()) {
                None
            } else {
                let (ds, ls, rs) = (max(&delta), max(left), max(right));
                let (mut ld, mut rd) = (-0.0, -0.0);
                if ds != 0.0 && ls != 0.0 && rs != 0.0 {
                    for ((d, l), r) in delta.iter().zip(left).zip(right) {
                        let d = d / ds;
                        ld += (l / ls) * d;
                        rd += (r / rs) * d;
                    }
                }
                Some(ld > 0.0 && rd > 0.0)
            };
            assert_eq!(
                super::position(&mut vec![0.0; n], a, b, left, right),
                positional,
                "position case {case}"
            );
            let bounded_position =
                super::bounded::<true>(a, b, left, right, [bound, max(left), max(right)])
                    .map(Some)
                    .unwrap_or_else(|| {
                        super::cached_position(
                            &mut vec![0.0; n],
                            a,
                            b,
                            left,
                            right,
                            max(left),
                            max(right),
                        )
                    });
            assert_eq!(bounded_position, positional, "bounded position case {case}");
        }
    }

    #[test]
    fn position_turning_uses_canonical_momentum_under_dense_whitening() {
        // q=L*z and p=L^-T*r for L=[[2,0],[3,1]]. Then Δq·p=Δz·r;
        // using velocity in physical coordinates would violate this identity.
        for x in -3..=3 {
            for y in -3..=3 {
                for p in -3..=3 {
                    for r in -3..=3 {
                        let (x, y, p, r) = (x as f64, y as f64, p as f64, r as f64);
                        let q = [2.0 * x, 3.0 * x + y];
                        let momentum = [0.5 * (p - 3.0 * r), r];
                        let expected = x * p + y * r > 0.0;
                        assert_eq!(
                            super::position(&mut [0.0; 2], &[0.0; 2], &q, &momentum, &momentum),
                            Some(expected)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shared_scaling_preserves_the_original_ordered_stopping_decision() {
        for n in 0..=130 {
            for magnitude in [f64::from_bits(1), 1e-200, 1.0, 1e200, f64::MAX / 32.0] {
                for shift in 0..12 {
                    let values = |offset| -> Vec<f64> {
                        (0..n)
                            .map(|i| (((i * 37 + offset) % 15) as f64 - 7.0) * magnitude)
                            .collect()
                    };
                    let (a, b, left, right) = (values(shift), values(3), values(7), values(11));
                    let rho: Vec<_> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
                    let maximum = |v: &[f64]| v.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
                    let scale = maximum(&rho);
                    let sign = |v: &[f64]| {
                        let vs = maximum(v);
                        vs != 0.0
                            && v.iter()
                                .zip(&rho)
                                .map(|(v, r)| (v / vs) * (r / scale))
                                .sum::<f64>()
                                > 0.0
                    };
                    let expected = scale != 0.0 && sign(&left) && sign(&right);
                    assert_eq!(
                        scalar(&mut vec![0.0; n], &a, &b, &left, &right),
                        Some(expected),
                        "n={n}, magnitude={magnitude}, shift={shift}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "simd")]
pub(super) fn vectorized(
    rho: &mut [f64],
    a: &[f64],
    b: &[f64],
    left: &[f64],
    right: &[f64],
) -> Option<bool> {
    use std::simd::{Simd, num::SimdFloat};
    type Vf = Simd<f64, 4>;
    // There is no full SIMD chunk in tiny models; retain the scalar path.
    if rho.len() < 4 {
        return scalar(rho, a, b, left, right);
    }
    debug_assert!(
        [a.len(), b.len(), left.len(), right.len()]
            .iter()
            .all(|&n| n == rho.len())
    );
    let (chunks, tail) = rho.as_chunks_mut::<4>();
    let (ac, at) = a.as_chunks::<4>();
    let (bc, bt) = b.as_chunks::<4>();
    let (lc, lt) = left.as_chunks::<4>();
    let (rc, rt) = right.as_chunks::<4>();
    let mut scales = [Vf::splat(0.0); 3];
    for ((((r, a), b), l), v) in chunks.iter_mut().zip(ac).zip(bc).zip(lc).zip(rc) {
        let sum = Vf::from_array(*a) + Vf::from_array(*b);
        let l = Vf::from_array(*l);
        let v = Vf::from_array(*v);
        if !(sum.is_finite() & l.is_finite() & v.is_finite()).all() {
            return None;
        }
        *r = sum.to_array();
        scales[0] = scales[0].simd_max(sum.abs());
        scales[1] = scales[1].simd_max(l.abs());
        scales[2] = scales[2].simd_max(v.abs());
    }
    let [mut scale, mut ls, mut rs] = scales.map(|v| v.reduce_max());
    for ((((r, a), b), l), v) in tail.iter_mut().zip(at).zip(bt).zip(lt).zip(rt) {
        *r = a + b;
        if !r.is_finite() || !l.is_finite() || !v.is_finite() {
            return None;
        }
        scale = scale.max(r.abs());
        ls = ls.max(l.abs());
        rs = rs.max(v.abs());
    }
    if scale == 0.0 || ls == 0.0 || rs == 0.0 {
        return Some(false);
    }
    if let Some(continues) = certified_continues(rho, left, right, scale, ls, rs) {
        return Some(continues);
    }
    Some(
        scale != 0.0
            && ls != 0.0
            && rs != 0.0
            && ordered_scaled_dot(left, rho, ls, scale) > 0.0
            && ordered_scaled_dot(right, rho, rs, scale) > 0.0,
    )
}

#[cfg(feature = "simd")]
fn ordered_scaled_dot(v: &[f64], rho: &[f64], vs: f64, scale: f64) -> f64 {
    use std::simd::Simd;
    type Vf = Simd<f64, 4>;
    let (vc, vt) = v.as_chunks::<4>();
    let (rc, rt) = rho.as_chunks::<4>();
    let mut sum = -0.0;
    for (v, r) in vc.iter().zip(rc) {
        // Divide, rather than multiply by a reciprocal: the scales may be
        // subnormal. No FMA and no horizontal reassociation of the dot product.
        let products =
            (Vf::from_array(*v) / Vf::splat(vs)) * (Vf::from_array(*r) / Vf::splat(scale));
        for product in products.to_array() {
            sum += product;
        }
    }
    for (v, r) in vt.iter().zip(rt) {
        sum += (v / vs) * (r / scale);
    }
    sum
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar_across_lengths_alignments_and_scales() {
        for n in 0..=130 {
            for offset in 0..4 {
                for magnitude in [f64::from_bits(1), 1e-200, 1.0, 1e200, f64::MAX / 16.0] {
                    let values = |shift: usize| -> Vec<f64> {
                        (0..n + offset)
                            .map(|i| ((i * 37 + shift) % 15) as f64 - 7.0)
                            .map(|v| v * magnitude)
                            .collect()
                    };
                    let (a, b, l, r) = (values(0), values(3), values(7), values(11));
                    let mut expected = vec![0.; n + offset];
                    let mut actual = expected.clone();
                    assert_eq!(
                        scalar(
                            &mut expected[offset..],
                            &a[offset..],
                            &b[offset..],
                            &l[offset..],
                            &r[offset..]
                        ),
                        vectorized(
                            &mut actual[offset..],
                            &a[offset..],
                            &b[offset..],
                            &l[offset..],
                            &r[offset..]
                        ),
                        "length={n} offset={offset} magnitude={magnitude}"
                    );
                    assert_eq!(
                        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn simd_rejects_nonfinite_lanes_and_overflow_including_tails() {
        for n in 1..=17 {
            for i in 0..n {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
                    for which in 0..4 {
                        let mut inputs = [vec![1.; n], vec![1.; n], vec![1.; n], vec![1.; n]];
                        inputs[which][i] = bad;
                        if bad == f64::MAX && which < 2 {
                            inputs[1 - which][i] = f64::MAX;
                        }
                        let [a, b, l, r] = inputs;
                        assert_eq!(
                            scalar(&mut vec![0.; n], &a, &b, &l, &r),
                            vectorized(&mut vec![0.; n], &a, &b, &l, &r)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn simd_dot_preserves_order_at_cancellation_and_subnormal_scales() {
        for n in 1..=33 {
            for scale in [f64::from_bits(1), 1e-200, 1.0, 1e200] {
                let v = vec![scale; n];
                let rho: Vec<_> = (0..n)
                    .map(|i| [1.0, f64::EPSILON / 2.0, -1.0, f64::EPSILON / 4.0][i % 4])
                    .collect();
                let expected: f64 = v
                    .iter()
                    .zip(&rho)
                    .map(|(v, r)| (v / scale) * (r / 1.0))
                    .sum();
                assert_eq!(
                    expected.to_bits(),
                    ordered_scaled_dot(&v, &rho, scale, 1.0).to_bits()
                );
            }
        }
        let mut rho = [0.; 9];
        assert_eq!(
            vectorized(&mut rho, &[0.; 9], &[0.; 9], &[1.; 9], &[1.; 9]),
            Some(false)
        );
        assert_eq!(
            vectorized(&mut rho, &[1.; 9], &[0.; 9], &[0.; 9], &[1.; 9]),
            Some(false)
        );
    }
}
