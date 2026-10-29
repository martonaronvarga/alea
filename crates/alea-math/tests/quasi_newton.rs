use alea_math::quasi_newton::{LbfgsError, lbfgs_inverse_product};

#[test]
fn two_loop_matches_independent_dense_inverse_bfgs_updates() {
    let s = [
        vec![1., 2., -1.],
        vec![0.3, -0.2, 0.7],
        vec![-0.4, 0.8, 0.1],
    ];
    let y: Vec<Vec<_>> = s
        .iter()
        .map(|s| vec![2. * s[0] + 0.2 * s[1], 0.2 * s[0] + 3. * s[1], 5. * s[2]])
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    for count in 0..=s.len() {
        let gamma = if count == 0 {
            1.
        } else {
            dot(&s[count - 1], &y[count - 1]) / dot(&y[count - 1], &y[count - 1])
        };
        let mut h = [[0.; 3]; 3];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = gamma;
        }
        for (s, y) in s[..count].iter().zip(&y) {
            let rho = dot(s, y).recip();
            let mut next = [[0.; 3]; 3];
            for (i, row) in next.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = rho * s[i] * s[j];
                    for (k, hrow) in h.iter().enumerate() {
                        for (l, hkl) in hrow.iter().enumerate() {
                            *v += (f64::from(i == k) - rho * s[i] * y[k])
                                * hkl
                                * (f64::from(l == j) - rho * y[l] * s[j]);
                        }
                    }
                }
            }
            h = next;
        }
        let v = [0.5, -1.7, 2.3];
        let mut actual = [0.; 3];
        lbfgs_inverse_product(&v, &s[..count], &y[..count], &mut actual).unwrap();
        let mut reused = [0.; 3];
        alea_math::quasi_newton::lbfgs_inverse_product_with_scratch(
            &v,
            &s[..count],
            &y[..count],
            &mut reused,
            &mut [f64::NAN; 3],
        )
        .unwrap();
        assert_eq!(actual.map(f64::to_bits), reused.map(f64::to_bits));
        for (a, row) in actual.iter().zip(h) {
            assert!((a - dot(&row, &v)).abs() < 2e-14);
        }
    }
}
#[test]
fn malformed_history_rejects_shapes_without_writes_and_bad_curvature() {
    let mut output = [17.; 2];
    assert_eq!(
        lbfgs_inverse_product(&[1., 2.], &[vec![1.]], &[vec![1.]], &mut output),
        Err(LbfgsError::Dimension)
    );
    assert_eq!(output, [17.; 2]);
    assert_eq!(
        lbfgs_inverse_product(&[1., 2.], &[vec![1., 0.]], &[vec![-1., 0.]], &mut output),
        Err(LbfgsError::Numerical)
    );
    assert_eq!(
        lbfgs_inverse_product(&[f64::NAN, 2.], &[], &[], &mut output),
        Err(LbfgsError::Numerical)
    );
}

#[test]
fn caller_scratch_preserves_bits_and_eliminates_action_allocations() {
    use alea_math::quasi_newton::lbfgs_inverse_product_with_scratch;
    let s = [vec![1., 2.], vec![0.5, -0.3]];
    let y = [vec![2., 6.], vec![1., -0.9]];
    let mut expected = [0.; 2];
    let baseline = allocation_counter::measure(|| {
        lbfgs_inverse_product(&[0.7, -2.], &s, &y, &mut expected).unwrap();
    });
    assert_eq!(baseline.count_total, 1);
    let mut output = [0.; 2];
    let mut scratch = [f64::NAN; 3];
    let measured = allocation_counter::measure(|| {
        for _ in 0..100 {
            lbfgs_inverse_product_with_scratch(&[0.7, -2.], &s, &y, &mut output, &mut scratch)
                .unwrap();
        }
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(output.map(f64::to_bits), expected.map(f64::to_bits));
    assert!(scratch[2].is_nan());
    output.fill(17.);
    assert_eq!(
        lbfgs_inverse_product_with_scratch(&[0.7, -2.], &s, &y, &mut output, &mut scratch[..1],),
        Err(LbfgsError::Dimension)
    );
    assert_eq!(output, [17.; 2]);
}
