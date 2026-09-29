use alea_math::interpolation::{InterpolationError, LinearSpline, UniformGrid};

#[test]
fn grid_spacing_retains_original_formula_and_rejects_unrepresentable_inputs() {
    let grid = UniformGrid::new(-1.0, 2.0, 7).unwrap();
    assert_eq!(
        (grid.min(), grid.max(), grid.points(), grid.spacing()),
        (-1.0, 2.0, 7, 0.5)
    );
    for count in [0, 1] {
        assert!(UniformGrid::new(0.0, 1.0, count).is_err());
    }
    for (min, max) in [
        (1.0, 0.0),
        (1.0, 1.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
    ] {
        assert!(UniformGrid::new(min, max, 3).is_err());
    }
    assert!(UniformGrid::new(-f64::MAX, f64::MAX, 2).is_err());
    assert_eq!(
        UniformGrid::new(-f64::MAX, f64::MAX, 3).unwrap().spacing(),
        f64::MAX
    );
    assert!(UniformGrid::new(0.0, f64::from_bits(1), 3).is_err());
}

#[test]
fn spline_preserves_knots_clamping_and_piecewise_values() {
    let spline = LinearSpline::new(vec![-1.0, 0.0, 2.0], vec![3.0, 1.0, 5.0]).unwrap();
    for (x, expected) in [
        (f64::NEG_INFINITY, 3.0),
        (-1.0, 3.0),
        (-0.5, 2.0),
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 5.0),
        (f64::INFINITY, 5.0),
    ] {
        assert_eq!(spline.evaluate(x).unwrap(), expected);
    }
    assert_eq!(spline.evaluate(f64::NAN), Err(InterpolationError::Query));
    for (knots, values) in [
        (vec![], vec![]),
        (vec![0.0, 0.0], vec![1.0, 2.0]),
        (vec![1.0, 0.0], vec![1.0, 2.0]),
        (vec![0.0, f64::INFINITY], vec![1.0, 2.0]),
        (vec![0.0, 1.0], vec![f64::NAN, 2.0]),
        (vec![0.0, 1.0], vec![1.0]),
    ] {
        assert!(LinearSpline::new(knots, values).is_err());
    }
}

#[test]
fn spline_avoids_overflow_for_extreme_finite_knots_and_values() {
    let spline = LinearSpline::new(vec![-f64::MAX, f64::MAX], vec![-f64::MAX, f64::MAX]).unwrap();
    for x in [-f64::MAX, -1e300, 0.0, 1e300, f64::MAX] {
        let y = spline.evaluate(x).unwrap();
        assert!(y.is_finite());
        assert!((y / f64::MAX - x / f64::MAX).abs() < 1e-15);
    }
    let tiny = LinearSpline::new(vec![0.0, f64::from_bits(2)], vec![0.0, 1.0]).unwrap();
    assert_eq!(tiny.evaluate(f64::from_bits(1)).unwrap(), 0.5);
}

#[test]
fn spline_matches_the_original_scalar_formula_on_irregular_grids() {
    let knots = [-2.0, -1.7, -0.1, 0.0, 0.9, 2.0, 4.0];
    let values = [1.0, -5.0, 3.0, 0.0, 7.0, -2.0, 4.0];
    let spline = LinearSpline::new(knots.to_vec(), values.to_vec()).unwrap();
    for i in 0..knots.len() - 1 {
        for j in 0..=100 {
            let x = knots[i] + (knots[i + 1] - knots[i]) * j as f64 / 100.0;
            let reference = values[i]
                + (x - knots[i]) * (values[i + 1] - values[i]) / (knots[i + 1] - knots[i]);
            assert!((spline.evaluate(x).unwrap() - reference).abs() < 1e-12);
        }
    }
}

#[test]
#[cfg(not(miri))]
fn spline_reuses_input_allocations_and_evaluation_allocates_nothing() {
    let knots = vec![0.0, 1.0, 2.0];
    let values = vec![1.0, -2.0, 4.0];
    let (k, v) = (knots.as_ptr(), values.as_ptr());
    let counts = allocation_counter::measure(|| {
        let spline = LinearSpline::new(knots, values).unwrap();
        assert_eq!(spline.knots().as_ptr(), k);
        assert_eq!(spline.values().as_ptr(), v);
        for i in 0..100 {
            std::hint::black_box(spline.evaluate(i as f64 / 50.0).unwrap());
        }
    });
    assert_eq!(counts.count_total, 0);
}
