//! Backend-independent metric contracts. Run with default, simd, faer, and openblas.
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric, MetricError,
};

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}

#[test]
fn diagonal_rejects_invalid_masses_without_coercion() {
    for value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            DiagonalMetric::new(buffer(&[1.0, value])).unwrap_err(),
            MetricError::InvalidDiagonal { index: 1 }
        );
    }
    // The constructor must not silently regularize even subnormal positive mass.
    let mass = f64::from_bits(1);
    let metric = DiagonalMetric::new(buffer(&[mass])).unwrap();
    let mut output = [0.0];
    metric.velocity(&[mass], &mut output).unwrap();
    assert_eq!(output, [1.0]);
    metric.sample_momentum(&[1.0], &mut output).unwrap();
    assert_eq!(output, [mass.sqrt()]);
}

#[test]
fn factor_rejects_shape_overflow_and_wrong_storage() {
    assert_eq!(
        CholeskyFactor::new_lower(usize::MAX, OwnedBuffer::new(0)).unwrap_err(),
        MetricError::DimensionOverflow { dim: usize::MAX }
    );
    for actual in [0, 3, 5] {
        assert_eq!(
            CholeskyFactor::new_lower(2, OwnedBuffer::new(actual)).unwrap_err(),
            MetricError::StorageLength {
                expected: 4,
                actual
            }
        );
    }
    // On 64-bit, the element count fits usize but the byte count exceeds isize.
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        CholeskyFactor::new_lower(1 << 31, OwnedBuffer::new(0)).unwrap_err(),
        MetricError::DimensionOverflow { dim: 1 << 31 }
    );
}

#[test]
fn factor_rejects_invalid_diagonal_and_both_nonfinite_triangles() {
    for value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            CholeskyFactor::new_lower(2, buffer(&[1.0, 0.5, 0.0, value])).unwrap_err(),
            MetricError::InvalidDiagonal { index: 1 }
        );
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            CholeskyFactor::new_lower(2, buffer(&[1.0, value, 0.0, 1.0])).unwrap_err(),
            MetricError::NonFiniteFactor { row: 1, column: 0 }
        );
        assert_eq!(
            CholeskyFactor::new_lower(2, buffer(&[1.0, 0.5, value, 1.0])).unwrap_err(),
            MetricError::NonFiniteFactor { row: 0, column: 1 }
        );
    }
    assert_eq!(
        CholeskyFactor::new_lower(2, buffer(&[1.0, 0.5, 0.25, 1.0])).unwrap_err(),
        MetricError::NonZeroUpperTriangle { row: 0, column: 1 }
    );
    assert!(CholeskyFactor::new_lower(2, buffer(&[1.0, 0.5, -0.0, 1.0])).is_ok());
}

#[test]
fn constructors_preserve_aligned_allocations() {
    let storage = buffer(&[2.0, 0.5, 0.0, 3.0]);
    let original = storage.as_ptr();
    let factor = CholeskyFactor::new_lower(2, storage).unwrap();
    assert_eq!(factor.as_slice().as_ptr(), original);
    assert_eq!(original as usize % 64, 0);
    assert_eq!(factor.as_slice(), &[2.0, 0.5, 0.0, 3.0]);
}

fn check_dimensions(metric: &dyn EuclideanMetric) {
    for source_len in [0, 1, 2, 3] {
        for destination_len in [0, 1, 2, 3] {
            if source_len == 2 && destination_len == 2 {
                continue;
            }
            let input = vec![1.0; source_len];
            let mut output = vec![123.0; destination_len];
            let expected = MetricError::VectorLength {
                expected: 2,
                source_len,
                destination_len,
            };
            assert_eq!(metric.sample_momentum(&input, &mut output), Err(expected));
            assert_eq!(output, vec![123.0; destination_len]);
            assert_eq!(metric.velocity(&input, &mut output), Err(expected));
            assert_eq!(output, vec![123.0; destination_len]);
            assert_eq!(metric.kinetic_energy(&input, &mut output), Err(expected));
            assert_eq!(output, vec![123.0; destination_len]);
        }
    }
}

#[test]
fn every_metric_rejects_dimensions_before_mutation() {
    check_dimensions(&IdentityMetric::new(2));
    check_dimensions(&DiagonalMetric::new(buffer(&[4.0, 9.0])).unwrap());
    check_dimensions(&DenseMetric::new(
        CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 3.0])).unwrap(),
    ));
}

#[test]
fn empty_metrics_are_consistent_without_backend_calls() {
    let metrics: [&dyn EuclideanMetric; 3] = [
        &IdentityMetric::new(0),
        &DiagonalMetric::new(OwnedBuffer::new(0)).unwrap(),
        &DenseMetric::new(CholeskyFactor::new_lower(0, OwnedBuffer::new(0)).unwrap()),
    ];
    for metric in metrics {
        assert_eq!(metric.dimension(), 0);
        assert_eq!(metric.log_det(), 0.0);
        metric.velocity(&[], &mut []).unwrap();
        metric.sample_momentum(&[], &mut []).unwrap();
        let mut output = [17.0];
        assert!(metric.velocity(&[], &mut output).is_err());
        assert_eq!(output, [17.0]);
    }
}

#[test]
fn mass_actions_and_log_determinants_have_the_documented_convention() {
    let diagonal = DiagonalMetric::new(buffer(&[4.0, 9.0])).unwrap();
    let mut output = [0.0; 2];
    diagonal.sample_momentum(&[1.0, 2.0], &mut output).unwrap();
    assert_eq!(output, [2.0, 6.0]);
    diagonal.velocity(&[4.0, 18.0], &mut output).unwrap();
    assert_eq!(output, [1.0, 2.0]);
    assert!((diagonal.log_det() - 36.0_f64.ln()).abs() < 1e-14);

    // L=[[2,0],[0.5,3]], M=[[4,1],[1,9.25]], det(M)=36.
    let dense =
        DenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 3.0])).unwrap());
    dense.sample_momentum(&[1.0, 2.0], &mut output).unwrap();
    assert_eq!(output, [2.0, 6.5]);
    dense.velocity(&[6.0, 19.5], &mut output).unwrap();
    for (actual, expected) in output.iter().zip([1.0, 2.0]) {
        assert!((actual - expected).abs() < 1e-14);
    }
    assert!((dense.log_det() - 36.0_f64.ln()).abs() < 1e-14);
}

#[test]
fn dense_actions_match_independent_mass_reference_across_block_boundaries() {
    for n in [1, 2, 7, 8, 9, 31, 32, 33, 65] {
        let lower = OwnedBuffer::from_fn(n * n, |k| {
            let (i, j) = (k % n, k / n);
            if i == j {
                1.0 + i as f64 / n as f64
            } else if i > j {
                0.02 * ((i * 13 + j * 7) as f64).sin()
            } else {
                0.0
            }
        });
        let metric = DenseMetric::new(CholeskyFactor::new_lower(n, lower).unwrap());
        let l = metric.factor().as_slice();
        let x: Vec<_> = (0..n).map(|i| (i as f64 + 0.4).cos()).collect();
        let mut mx = vec![0.0; n];
        let mut lx = vec![0.0; n];
        // Compute Mx = L(L^T x) independently of the production triangular
        // solves, without an O(n^3) materialization of M in the Miri suite.
        let transpose_x: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| l[i + j * n] * x[i]).sum())
            .collect();
        for i in 0..n {
            for j in 0..n {
                lx[i] += l[i + j * n] * x[j];
                mx[i] += l[i + j * n] * transpose_x[j];
            }
        }
        // Deliberately offset the output to exercise unaligned SIMD/BLAS slices.
        let mut storage = OwnedBuffer::new(n + 1);
        storage[0] = 123.0;
        metric.sample_momentum(&x, &mut storage[1..]).unwrap();
        for (actual, expected) in storage[1..].iter().zip(lx) {
            assert!((actual - expected).abs() < 1e-12);
        }
        metric.velocity(&mx, &mut storage[1..]).unwrap();
        for (actual, expected) in storage[1..].iter().zip(x) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(storage[0], 123.0);
    }
}
