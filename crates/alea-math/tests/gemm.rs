use alea_math::gemm::{MatrixKernel, try_gemm};

fn kernels() -> Vec<MatrixKernel> {
    vec![
        MatrixKernel::Compensated,
        #[cfg(any(feature = "faer", feature = "openblas"))]
        MatrixKernel::Auto,
        #[cfg(feature = "faer")]
        MatrixKernel::Faer,
        #[cfg(feature = "openblas")]
        MatrixKernel::Blas,
    ]
}

#[test]
fn checked_gemm_transposes_accumulation_and_empty_products() {
    for kernel in kernels() {
        for ta in [false, true] {
            for tb in [false, true] {
                let (m, k, n) = (3, 5, 2);
                let mut a = vec![0.; m * k];
                let mut b = vec![0.; k * n];
                for i in 0..m {
                    for j in 0..k {
                        a[if ta { j * m + i } else { i * k + j }] = (i * k + j) as f64 / 8.;
                    }
                }
                for j in 0..k {
                    for c in 0..n {
                        b[if tb { c * k + j } else { j * n + c }] = (j + c + 1) as f64 / 4.;
                    }
                }
                for add in [false, true] {
                    let mut out = vec![7.; m * n];
                    let done = try_gemm(kernel, &a, &b, &mut out, (m, k, n), (ta, tb), add);
                    if kernel == MatrixKernel::Compensated {
                        assert!(!done);
                        assert_eq!(out, vec![7.; m * n]);
                        continue;
                    }
                    assert!(done);
                    for i in 0..m {
                        for c in 0..n {
                            let expected = (0..k)
                                .map(|j| (i * k + j) as f64 / 8. * (j + c + 1) as f64 / 4.)
                                .sum::<f64>()
                                + if add { 7. } else { 0. };
                            assert!((out[i * n + c] - expected).abs() < 1e-12);
                        }
                    }
                }
            }
        }
        let mut out = [9.; 6];
        assert!(!try_gemm(
            kernel,
            &[],
            &[],
            &mut out,
            (usize::MAX, 2, 3),
            (false, false),
            false
        ));
        assert_eq!(out, [9.; 6]);
        assert!(!try_gemm(
            kernel,
            &[1.],
            &[],
            &mut out,
            (3, 0, 2),
            (false, false),
            false
        ));
        assert_eq!(out, [9.; 6]);
        if kernel != MatrixKernel::Compensated {
            assert!(try_gemm(
                kernel,
                &[],
                &[],
                &mut out,
                (3, 0, 2),
                (true, true),
                true
            ));
            assert_eq!(out, [9.; 6]);
            assert!(try_gemm(
                kernel,
                &[],
                &[],
                &mut out,
                (3, 0, 2),
                (true, true),
                false
            ));
            assert_eq!(out, [0.; 6]);
        }
    }
}
