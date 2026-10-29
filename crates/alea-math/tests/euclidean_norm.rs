use alea_math::numeric::euclidean_norm;
#[test]
fn large_small_and_subnormal_vectors_do_not_overflow_intermediate_squares() {
    for scale in [1., 1e300, 1e-300] {
        assert!((euclidean_norm(&[3. * scale, 4. * scale]) / (5. * scale) - 1.).abs() < 1e-15);
    }
    let tiny = f64::from_bits(1);
    assert_eq!(euclidean_norm(&[tiny; 4]), 2. * tiny);
    assert_eq!(euclidean_norm(&[]), 0.);
    assert_eq!(euclidean_norm(&[0., -0.]), 0.);
    assert!(euclidean_norm(&[f64::NAN, f64::INFINITY]).is_nan());
    assert_eq!(euclidean_norm(&[f64::INFINITY]), f64::INFINITY);
}
