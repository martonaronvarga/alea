use alea_math::numeric::CompensatedSum;
#[test]
fn cancellation_and_small_terms_are_retained_without_certification() {
    let mut sum = CompensatedSum::default();
    for x in [1e16, 1., -1e16] {
        sum.add(x);
    }
    assert_eq!(sum.total(), 1.);
    let mut sum = CompensatedSum::default();
    sum.add(1.);
    for _ in 0..10000 {
        sum.add(1e-16);
    }
    assert_eq!(sum.total(), 1. + 1e-12);
    sum.add(f64::INFINITY);
    assert!(!sum.total().is_finite());
}

#[test]
fn product_residuals_survive_cancellation_against_exact_dyadic_reference() {
    let a = 1. + 2_f64.powi(-27);
    let b = 1. - 2_f64.powi(-27);
    let mut sum = CompensatedSum::default();
    sum.add_product(a, b);
    sum.add(-1.);
    assert_eq!(sum.total(), -2_f64.powi(-54));
    // A rounded product sum cannot recover this residual.
    let mut rounded = CompensatedSum::default();
    rounded.add(a * b);
    rounded.add(-1.);
    assert_eq!(rounded.total(), 0.);
    let mut sum = CompensatedSum::default();
    sum.add_product(f64::MIN_POSITIVE, 0.5);
    assert_eq!(sum.total(), f64::MIN_POSITIVE * 0.5);
    for pair in [(f64::INFINITY, 0.), (f64::MAX, 2.), (f64::NAN, 1.)] {
        let mut sum = CompensatedSum::default();
        sum.add_product(pair.0, pair.1);
        assert!(!sum.total().is_finite());
    }
}
