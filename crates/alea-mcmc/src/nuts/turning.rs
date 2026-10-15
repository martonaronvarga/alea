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
    let sign = |v: &[f64], vs: f64| {
        vs != 0.0
            && v.iter()
                .zip(rho.iter())
                .map(|(v, r)| (v / vs) * (r / scale))
                .sum::<f64>()
                > 0.0
    };
    Some(scale != 0.0 && sign(left, ls) && sign(right, rs))
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
