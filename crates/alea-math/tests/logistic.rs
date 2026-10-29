use alea_math::numeric::{sigmoid, softplus, softplus_and_sigmoid};

#[test]
fn fused_softplus_preserves_scalar_results_and_representable_tail_derivatives() {
    let check = |x: f64| {
        let (value, slope) = softplus_and_sigmoid(x);
        assert_eq!(value.to_bits(), softplus(x).to_bits(), "value x={x}");
        assert_eq!(slope.to_bits(), sigmoid(x).to_bits(), "slope x={x}");
    };
    for i in -8000..=8000 {
        check(i as f64 * 0.1);
    }
    for x in [
        0.,
        -0.,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        check(x);
    }
    let (value, slope) = softplus_and_sigmoid(f64::NAN);
    assert!(value.is_nan() && slope.is_nan());
    // Independent special points, including a tail lost by 1-sigmoid(40).
    assert_eq!(softplus_and_sigmoid(0.), (std::f64::consts::LN_2, 0.5));
    assert_eq!(sigmoid(40.), 1.);
    assert!((softplus_and_sigmoid(-40.).1 / 4.248_354_255_291_589e-18 - 1.).abs() < 1e-15);
    assert!(softplus_and_sigmoid(-740.).1 > 0.);
}
