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
