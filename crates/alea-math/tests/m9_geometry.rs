use alea_math::{
    affine::{AffineMapError, LowRankAffineMap},
    buffer::OwnedBuffer,
    metric::{
        BlockDiagonalMetric, BlockMetric, CholeskyFactor, DenseMetric, EuclideanMetric,
        SparseCholeskyMetric,
    },
};

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}

#[test]
fn affine_rejects_numerically_unsupported_spectral_correction() {
    let error = LowRankAffineMap::new(
        buffer(&[0.0, 0.0]),
        buffer(&[1.0, 1.0]),
        buffer(&[1.0, 0.0]),
        buffer(&[1e200]),
    )
    .unwrap_err();
    assert_eq!(error, AffineMapError::IllConditionedCorrection);
}

fn close(left: &[f64], right: &[f64]) {
    for (&left, &right) in left.iter().zip(right) {
        assert!((left - right).abs() < 2e-12 * left.abs().max(right.abs()).max(1.0));
    }
}

#[test]
fn low_rank_affine_roundtrip_pullback_and_determinant() {
    let root = 0.5_f64.sqrt();
    let map = LowRankAffineMap::new(
        buffer(&[1.0, -2.0, 0.5]),
        buffer(&[2.0, 3.0, 0.5]),
        buffer(&[root, root, 0.0, 0.0, 0.0, 1.0]),
        buffer(&[4.0, 0.25]),
    )
    .unwrap();
    let latent = [0.3, -1.2, 2.0];
    let mut target = [0.0; 3];
    map.forward(&latent, &mut target).unwrap();
    let mut recovered = [0.0; 3];
    map.inverse(&target, &mut recovered).unwrap();
    close(&latent, &recovered);

    let score = [0.7, -0.2, 1.3];
    let mut pulled = [0.0; 3];
    map.pullback(&score, &mut pulled).unwrap();
    let h = 1e-6;
    for coordinate in 0..3 {
        let mut plus = latent;
        let mut minus = latent;
        plus[coordinate] += h;
        minus[coordinate] -= h;
        let mut xp = [0.0; 3];
        let mut xm = [0.0; 3];
        map.forward(&plus, &mut xp).unwrap();
        map.forward(&minus, &mut xm).unwrap();
        let numerical = xp
            .iter()
            .zip(xm)
            .zip(score)
            .map(|((&p, m), g)| (p - m) * g / (2.0 * h))
            .sum::<f64>();
        assert!((pulled[coordinate] - numerical).abs() < 2e-9);
    }
    let expected = (2.0_f64 * 3.0 * 0.5).ln() + 0.5 * (4.0_f64 * 0.25).ln();
    assert!((map.log_abs_det() - expected).abs() < 1e-14);
}

#[test]
fn block_and_sparse_actions_match_dense_factor() {
    // L = [[2,0,0], [.5,3,0], [0,.25,4]] in column-major storage.
    let dense_factor = buffer(&[2.0, 0.5, 0.0, 0.0, 3.0, 0.25, 0.0, 0.0, 4.0]);
    let dense = DenseMetric::new(CholeskyFactor::new_lower(3, dense_factor).unwrap());
    let sparse = SparseCholeskyMetric::new(
        3,
        vec![0, 2, 4, 5].into_boxed_slice(),
        vec![0, 1, 1, 2, 2].into_boxed_slice(),
        buffer(&[2.0, 0.5, 3.0, 0.25, 4.0]),
    )
    .unwrap();
    let input = [0.4, -1.3, 2.1];
    let mut expected = [0.0; 3];
    let mut actual = [0.0; 3];
    dense.sample_momentum(&input, &mut expected).unwrap();
    sparse.sample_momentum(&input, &mut actual).unwrap();
    close(&expected, &actual);
    dense.velocity(&input, &mut expected).unwrap();
    sparse.velocity(&input, &mut actual).unwrap();
    close(&expected, &actual);
    assert!((dense.log_det() - sparse.log_det()).abs() < 1e-14);

    let block = BlockDiagonalMetric::new(vec![
        BlockMetric::dense(CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 3.0])).unwrap()),
        BlockMetric::diagonal(buffer(&[16.0])).unwrap(),
    ])
    .unwrap();
    block.sample_momentum(&input, &mut actual).unwrap();
    close(&expected_momentum(&input), &actual);
    block.velocity(&input, &mut actual).unwrap();
    close(&expected_velocity(&input), &actual);
}

fn expected_momentum(input: &[f64; 3]) -> [f64; 3] {
    [
        2.0 * input[0],
        0.5 * input[0] + 3.0 * input[1],
        4.0 * input[2],
    ]
}

fn expected_velocity(input: &[f64; 3]) -> [f64; 3] {
    let forward_first = input[0] / 2.0;
    let forward_second = (input[1] - 0.5 * forward_first) / 3.0;
    let second = forward_second / 3.0;
    [
        (forward_first - 0.5 * second) / 2.0,
        second,
        input[2] / 16.0,
    ]
}

#[test]
fn structured_actions_allocate_nothing() {
    let metric = SparseCholeskyMetric::new(
        3,
        vec![0, 2, 4, 5].into_boxed_slice(),
        vec![0, 1, 1, 2, 2].into_boxed_slice(),
        buffer(&[2.0, 0.5, 3.0, 0.25, 4.0]),
    )
    .unwrap();
    let input = [0.4, -1.3, 2.1];
    let mut output = [0.0; 3];
    let measured = allocation_counter::measure(|| {
        for _ in 0..32 {
            metric.velocity(&input, &mut output).unwrap();
            metric.sample_momentum(&input, &mut output).unwrap();
        }
    });
    assert_eq!(measured.count_total, 0);
}
