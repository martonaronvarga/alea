use alea_core::model::{
    assembly::{AssemblyError as E, ModelBuilder},
    tensor::{BinaryOp as B, Comparison, Shape, TensorOperation, UnaryOp as U},
};
use alea_core::target::LogDensityGradient;

fn shape(d: &[usize]) -> Shape {
    Shape::new(d.to_vec()).unwrap()
}
#[test]
fn tensor_target_budget_rejects_before_allocation_and_counts_control_flow_frames() {
    for steps in [0, 1, 7, 64] {
        let program = recurrence(steps);
        let bytes = program.workspace_bytes();
        assert!(bytes > 0);
        let measured = allocation_counter::measure(|| {
            let error = program.workspace_with_limit(bytes - 1).err().unwrap();
            assert_eq!(error.required, bytes);
            assert_eq!(error.limit, bytes - 1);
        });
        assert_eq!(measured.count_total, 0);

        assert_eq!(
            allocation_counter::measure(|| {
                assert!(program.check_workspace_limit(bytes - 1).is_err());
                program.check_workspace_limit(bytes).unwrap();
            })
            .count_total,
            0
        );
        let mut work = program.workspace_with_limit(bytes).unwrap();
        let mut retained = None;
        let allocation = allocation_counter::measure(|| {
            retained = Some(program.workspace());
        });
        assert!(allocation.bytes_current >= 0 && allocation.bytes_current as usize <= bytes);
        drop(retained);
        let mut out = [0.];
        // Same plan/data semantics as the unrestricted API; budget is admission
        // only, never a different computation or silently truncated scan.
        let x = vec![0.25; program.dimension()];
        program.evaluate(&x, &mut out, &mut work).unwrap();
        let mut expected = [0.];
        program
            .evaluate(&x, &mut expected, &mut program.workspace())
            .unwrap();
        assert_eq!(out.map(f64::to_bits), expected.map(f64::to_bits));
    }
    assert!(recurrence(64).workspace_bytes() > recurrence(7).workspace_bytes());
}
fn check_vjp(program: &alea_core::model::tensor::BoundProgram, x: &[f64], seed: &[f64]) {
    let mut w = program.workspace();
    let mut out = vec![0.; seed.len()];
    let mut g = vec![0.; x.len()];
    program
        .value_and_vjp(x, seed, &mut out, &mut g, &mut w)
        .unwrap();
    for i in 0..x.len() {
        let mut plus = x.to_vec();
        let mut minus = x.to_vec();
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        let mut a = vec![0.; seed.len()];
        let mut b = vec![0.; seed.len()];
        program.evaluate(&plus, &mut a, &mut w).unwrap();
        program.evaluate(&minus, &mut b, &mut w).unwrap();
        let fd = a
            .iter()
            .zip(b)
            .zip(seed)
            .map(|((&a, b), &s)| (a - b) * s)
            .sum::<f64>()
            / 2e-5;
        assert!(
            (fd - g[i]).abs() < 2e-7 * (1. + fd.abs()),
            "coordinate {i}: {} vs {fd}",
            g[i]
        );
    }
    let allocations = allocation_counter::measure(|| {
        for _ in 0..16 {
            program
                .value_and_vjp(x, seed, &mut out, &mut g, &mut w)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn broadcast_rank_three_gather_and_shared_reshape_have_correct_reverse_maps() {
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", shape(&[2, 1, 3])).unwrap();
    let y = b.input("y", shape(&[1, 4, 1])).unwrap();
    let product = b.binary(B::Multiply, x, y).unwrap();
    let selected = b
        .gather(product, shape(&[2, 2]), vec![0, 3, 0, 23])
        .unwrap();
    let flat = b.reshape(selected, shape(&[4])).unwrap();
    let square = b.binary(B::Multiply, flat, flat).unwrap();
    let out = b.sum(square).unwrap();
    let program = b.compile(out).unwrap();
    assert_eq!(program.node_count(), 7);
    let p = program.bind(&[]).unwrap();
    let input = [1., 2., 3., 4., 5., 6., 0.5, 1., 1.5, 2.];
    let mut result = [0.];
    p.evaluate(&input, &mut result, &mut p.workspace()).unwrap();
    assert_eq!(result, [0.25 + 1. + 0.25 + 144.]);
    check_vjp(&p, &input, &[1.]);
}

#[test]
fn regular_broadcast_matches_explicit_gathers_without_reordering_scores() {
    for op in [B::Add, B::Subtract, B::Multiply, B::Divide] {
        for n in [2, 683] {
            let build = |expand: bool| {
                let mut b = ModelBuilder::new().region();
                let mut a = b.input("a", shape(&[n, 1, 3])).unwrap();
                let mut r = b.input("r", shape(&[1, 2, 1])).unwrap();
                if expand {
                    a = b
                        .gather(
                            a,
                            shape(&[n, 2, 3]),
                            (0..n * 6).map(|j| j / 6 * 3 + j % 3).collect(),
                        )
                        .unwrap();
                    r = b
                        .gather(
                            r,
                            shape(&[n, 2, 3]),
                            (0..n * 6).map(|j| j / 3 % 2).collect(),
                        )
                        .unwrap();
                }
                let result = b.binary(op, a, r).unwrap();
                // Long case also exercises the native forward fusion schedule.
                let neg = b.unary(U::Neg, result).unwrap();
                let square = b.binary(B::Multiply, neg, neg).unwrap();
                let out = b.binary(B::Add, result, square).unwrap();
                b.compile(out).unwrap().bind(&[]).unwrap()
            };
            let regular = build(false);
            let gathered = build(true);
            let mut x: Vec<_> = (0..n * 3).map(|j| (j % 7) as f64 / 8. - 0.25).collect();
            x.extend([0.5, 1.5]);
            let seed: Vec<_> = (0..n * 6)
                .map(|j| [1e16, 1., -1e16, 0., -0.5][j % 5])
                .collect();
            let run = |p: &alea_core::model::tensor::BoundProgram| {
                let mut out = vec![0.; seed.len()];
                let mut g = vec![0.; x.len()];
                p.value_and_vjp(&x, &seed, &mut out, &mut g, &mut p.workspace())
                    .unwrap();
                (
                    out.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
                    g.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
                )
            };
            assert_eq!(run(&regular), run(&gathered), "{op:?}, n={n}");
            if n == 2 {
                check_vjp(&regular, &x, &vec![0.25; n * 6]);
            }
        }
    }
}

#[test]
fn broadcast_source_is_compact_and_rebinding_preserves_error_staging() {
    let build = |n| {
        let mut b = ModelBuilder::new().region();
        let a = b.input("a", shape(&[2, 1])).unwrap();
        let r = b.data("r", shape(&[n])).unwrap();
        let out = b.binary(B::Divide, a, r).unwrap();
        b.compile(out).unwrap()
    };
    let small = build(16)
        .bind(&[("r", &[1.; 16])])
        .unwrap()
        .rust_function()
        .unwrap();
    let large = build(4096)
        .bind(&[("r", &[1.; 4096])])
        .unwrap()
        .rust_function()
        .unwrap();
    // Data literals still scale with data; regular maps must not add literals.
    assert!(small.contains("j%16"));
    assert!(large.contains("j%4096"));
    assert!(!large.contains("[usize;"));
    let plan = build(3);
    let good = plan.bind(&[("r", &[1., 2., 4.])]).unwrap();
    let bad = plan.bind(&[("r", &[1., 0., 4.])]).unwrap();
    let mut w = good.workspace();
    let mut out = [41.; 6];
    let mut g = [41.; 2];
    assert!(
        bad.value_and_vjp(&[1., 2.], &[1.; 6], &mut out, &mut g, &mut w)
            .is_err()
    );
    assert_eq!(out, [41.; 6]);
    assert_eq!(g, [41.; 2]);
    good.value_and_vjp(&[1., 2.], &[1.; 6], &mut out, &mut g, &mut w)
        .unwrap();
    assert_eq!(out, [1., 0.5, 0.25, 2., 1., 0.5]);
    assert_eq!(g, [1.75, 1.75]);
}

#[test]
fn empty_broadcasts_do_not_address_inputs_or_adjoint_ranges() {
    for reverse in [false, true] {
        let mut b = ModelBuilder::new().region();
        let empty = b.input("empty", shape(&[0, 3])).unwrap();
        let full = b.constant(shape(&[3]), vec![1., 2., 3.]).unwrap();
        let (a, r) = if reverse {
            (full, empty)
        } else {
            (empty, full)
        };
        let out = b.binary(B::Add, a, r).unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        p.value_and_vjp(&[], &[], &mut [], &mut [], &mut p.workspace())
            .unwrap();
    }
}

#[test]
fn matrix_block_vjp_matches_finite_differences_and_exact_product() {
    let mut b = ModelBuilder::new().region();
    let a = b.input("a", shape(&[2, 3])).unwrap();
    let c = b.input("b", shape(&[3, 2])).unwrap();
    let out = b.matmul(a, c).unwrap();
    let plan = b.compile(out).unwrap();
    assert_eq!(plan.node_count(), 3);
    assert!(plan.nodes().any(|n| n.operation == TensorOperation::Matmul));
    let program = plan.bind(&[]).unwrap();
    let x = [1., 2., 3., 4., 5., 6., 1., 0., 0., 1., 1., 1.];
    let mut out = [0.; 4];
    program
        .evaluate(&x, &mut out, &mut program.workspace())
        .unwrap();
    assert_eq!(out, [4., 5., 10., 11.]);
    check_vjp(&program, &x, &[0.5, -0.2, 0.3, 1.]);
}

#[test]
fn matrix_kernel_policies_preserve_products_vjps_aliases_and_error_recovery() {
    use alea_core::model::tensor::{MatrixProvider, MatrixRequest};
    for kernel in [
        MatrixRequest::default(),
        #[cfg(any(feature = "faer", feature = "openblas"))]
        MatrixRequest::standard(MatrixProvider::Auto),
        #[cfg(feature = "faer")]
        MatrixRequest::standard(MatrixProvider::Faer),
        #[cfg(feature = "openblas")]
        MatrixRequest::standard(MatrixProvider::Blas),
    ] {
        let mut b = ModelBuilder::new().region();
        let a = b.input("a", shape(&[2, 3])).unwrap();
        let r = b.data("r", shape(&[3, 2])).unwrap();
        let out = b.matmul_with_request(a, r, kernel).unwrap();
        // A direct path contributes existing adjoints before matrix reversal.
        let extra = b.sum(a).unwrap();
        let sum = b.sum(out).unwrap();
        let total = b.binary(B::Add, extra, sum).unwrap();
        let p = b
            .compile(total)
            .unwrap()
            .bind(&[("r", &[1., 2., 3., 4., 5., 6.])])
            .unwrap();
        check_vjp(&p, &[1., 2., 3., 4., 5., 6.], &[1.]);
        let changed = p.rebind("r", &[2., 1., 4., 3., 6., 5.]).unwrap();
        check_vjp(&changed, &[1., 2., 3., 4., 5., 6.], &[1.]);
        let mut out = [77.];
        let mut g = [88.; 6];
        let mut w = p.workspace();
        assert!(
            p.value_and_vjp(&[f64::MAX; 6], &[1.], &mut out, &mut g, &mut w)
                .is_err()
        );
        assert_eq!(out, [77.]);
        assert_eq!(g, [88.; 6]);
        p.value_and_vjp(&[1.; 6], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        assert_eq!(g, [4., 8., 12., 4., 8., 12.]);

        for shared in [false, true] {
            let mut b = ModelBuilder::new().region();
            let a = b.input("a", shape(&[2, 2])).unwrap();
            let r = if shared {
                b.reshape(a, shape(&[2, 2])).unwrap()
            } else {
                b.input("r", shape(&[2, 2])).unwrap()
            };
            let out = b.matmul_with_request(a, r, kernel).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            let x = if shared {
                vec![1., 2., 3., 4.]
            } else {
                vec![1., 2., 3., 4., 5., 6., 7., 8.]
            };
            check_vjp(&p, &x, &[1., 2., 3., 4.]);
            check_vjp(&p, &x, &[0., 1., 0., 2.]);
        }
    }
}

#[test]
fn matrix_kernel_rectangular_and_empty_products_match_reference() {
    use alea_core::model::tensor::{MatrixProvider, MatrixRequest};
    for (m, k, n) in [
        (17, 33, 19),
        (65, 9, 1),
        (1, 11, 7),
        (3, 0, 4),
        (0, 3, 2),
        (2, 3, 0),
    ] {
        let x: Vec<_> = (0..m * k + k * n)
            .map(|i| ((i % 29) as f64 - 14.).sin() / 3.)
            .collect();
        let seed: Vec<_> = (0..m * n).map(|i| (i % 7 + 1) as f64 / 13.).collect();
        let mut reference: Option<(Vec<f64>, Vec<f64>)> = None;
        for kernel in [
            MatrixRequest::default(),
            #[cfg(feature = "faer")]
            MatrixRequest::standard(MatrixProvider::Faer),
            #[cfg(feature = "openblas")]
            MatrixRequest::standard(MatrixProvider::Blas),
        ] {
            let mut b = ModelBuilder::new().region();
            let a = b.input("a", shape(&[m, k])).unwrap();
            let r = b.input("r", shape(&[k, n])).unwrap();
            let out = b.matmul_with_request(a, r, kernel).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            let mut out = vec![0.; m * n];
            let mut g = vec![0.; x.len()];
            p.value_and_vjp(&x, &seed, &mut out, &mut g, &mut p.workspace())
                .unwrap();
            if let Some((values, scores)) = &reference {
                for (got, want) in out.iter().zip(values).chain(g.iter().zip(scores)) {
                    assert!((got - want).abs() < 1e-11);
                }
            } else {
                reference = Some((out, g));
            }
        }
    }
}
#[test]
fn matrix_vector_specialization_preserves_ordered_compensated_bits_and_aliases() {
    use alea_math::numeric::CompensatedSum;
    for rows in [0, 1, 7, 33] {
        for cols in [0, 1, 3, 8, 17] {
            let mut b = ModelBuilder::new().region();
            let a = b.input("a", shape(&[rows, cols])).unwrap();
            let x = b.input("x", shape(&[cols, 1])).unwrap();
            let out = b.matmul(a, x).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            let values: Vec<_> = (0..rows * cols + cols)
                .map(|i| match i % 5 {
                    0 => 1e16,
                    1 => 1.,
                    2 => -1e16,
                    3 => 0.125,
                    _ => -0.25,
                })
                .collect();
            let seed: Vec<_> = (0..rows)
                .map(|i| if i % 3 == 0 { 0. } else { i as f64 / 8. })
                .collect();
            let mut actual = vec![0.; rows];
            let mut g = vec![0.; values.len()];
            p.value_and_vjp(&values, &seed, &mut actual, &mut g, &mut p.workspace())
                .unwrap();
            let mut expected = vec![0.; rows];
            let mut expected_g = vec![0.; values.len()];
            for row in 0..rows {
                let mut sum = CompensatedSum::default();
                for j in 0..cols {
                    sum.add_product(values[row * cols + j], values[rows * cols + j]);
                    if seed[row] != 0. {
                        expected_g[row * cols + j] += seed[row] * values[rows * cols + j];
                        expected_g[rows * cols + j] += seed[row] * values[row * cols + j];
                    }
                }
                expected[row] = sum.total();
            }
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_g.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    let mut b = ModelBuilder::new().region();
    let a = b.input("a", shape(&[1, 3])).unwrap();
    let x = b.reshape(a, shape(&[3, 1])).unwrap();
    let out = b.matmul(a, x).unwrap();
    let p = b.compile(out).unwrap().bind(&[]).unwrap();
    check_vjp(&p, &[0.2, -0.5, 0.7], &[0.3]);
}
fn log_region(negative: bool) -> alea_core::model::tensor::BoundProgram {
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let x = if negative {
        b.unary(U::Neg, x).unwrap()
    } else {
        x
    };
    let out = b.unary(U::Log, x).unwrap();
    b.compile(out).unwrap().bind(&[]).unwrap()
}
#[test]
fn condition_is_lazy_in_both_passes_and_failures_do_not_publish_or_poison_recovery() {
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
    let out = b
        .cond(
            x,
            zero,
            Comparison::Greater,
            &[x],
            log_region(false),
            log_region(true),
        )
        .unwrap();
    let plan = b.compile(out).unwrap();
    assert_eq!(plan.node_count(), 3);
    let program = plan.bind(&[]).unwrap();
    let mut work = program.workspace();
    let mut out = [91.];
    let mut g = [92.];
    assert!(
        program
            .value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut work)
            .is_err()
    );
    assert_eq!(out, [91.]);
    assert_eq!(g, [92.]);
    for value in [2., -2., 1e-100, -1e-100] {
        program
            .value_and_vjp(&[value], &[1.], &mut out, &mut g, &mut work)
            .unwrap();
        assert_eq!(out[0], value.abs().ln());
        assert_eq!(g[0], 1. / value);
    }
    check_vjp(&program, &[2.], &[1.]);
    check_vjp(&program, &[-2.], &[1.]);
    assert!(program.rust_function().unwrap().contains("else {no"));
}

#[test]
fn selected_intermediate_overflow_cannot_be_hidden_by_a_finite_final_expression_or_zero_seed() {
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let huge = b.unary(U::Exp, x).unwrap();
    let negative = b.unary(U::Neg, huge).unwrap();
    let result = b.unary(U::Exp, negative).unwrap();
    let p = b.compile(result).unwrap().bind(&[]).unwrap();
    let mut w = p.workspace();
    for seed in [0., 1.] {
        let (mut out, mut g) = ([91.], [92.]);
        // Unchecked arithmetic would misleadingly end at exp(-infinity) = 0.
        assert!(matches!(
            p.evaluate(&[1000.], &mut out, &mut w),
            Err(E::Numerical)
        ));
        assert!(matches!(
            p.value_and_vjp(&[1000.], &[seed], &mut out, &mut g, &mut w),
            Err(E::Numerical)
        ));
        assert_eq!((out, g), ([91.], [92.]));
        p.value_and_vjp(&[0.], &[seed], &mut out, &mut g, &mut w)
            .unwrap();
        assert_eq!(out, [(-1_f64).exp()]);
        assert_eq!(g, [-seed * (-1_f64).exp()]);
    }
}

#[test]
fn repeated_gathers_obey_seed_linearity_without_differentiating_inactive_singular_data() {
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", shape(&[3])).unwrap();
    let gathered = b.gather(x, shape(&[4]), vec![2, 0, 2, 1]).unwrap();
    let square = b.binary(B::Multiply, gathered, gathered).unwrap();
    let data = b.data("fixed", shape(&[4])).unwrap();
    let root = b.unary(U::Sqrt, data).unwrap();
    let out = b.binary(B::Add, square, root).unwrap();
    let p = b
        .compile(out)
        .unwrap()
        .bind(&[("fixed", &[0.; 4])])
        .unwrap();
    let mut w = p.workspace();
    let x = [0.7, -1.2, 2.3];
    let u = [0., -0.7, 1.3, 0.2];
    let v = [0.4, 0.2, -0.3, 0.];
    let combined = std::array::from_fn::<_, 4, _>(|i| -0.6 * u[i] + 1.7 * v[i]);
    let mut gradients = [[0.; 3]; 3];
    for (seed, gradient) in [u, v, combined].iter().zip(&mut gradients) {
        let mut out = [99.; 4];
        p.value_and_vjp(&x, seed, &mut out, gradient, &mut w)
            .unwrap();
        assert_eq!(out, [x[2] * x[2], x[0] * x[0], x[2] * x[2], x[1] * x[1]]);
        let expected = [
            2. * x[0] * seed[1],
            2. * x[1] * seed[3],
            2. * x[2] * (seed[0] + seed[2]),
        ];
        for (actual, expected) in gradient.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-14 * (1. + expected.abs()));
        }
        check_vjp(&p, &x, seed);
    }
    for ((u, v), combined) in gradients[0].iter().zip(&gradients[1]).zip(&gradients[2]) {
        let expected = -0.6 * u + 1.7 * v;
        assert!((combined - expected).abs() < 2e-14 * (1. + expected.abs()));
    }
    // Native built-in VJPs skip a zero seed at a singular point; an opaque
    // supplied local score must instead be admitted even with a zero seed.
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let root = b.unary(U::Sqrt, x).unwrap();
    let p = b.compile(root).unwrap().bind(&[]).unwrap();
    let mut w = p.workspace();
    let (mut out, mut g) = ([91.], [92.]);
    p.value_and_vjp(&[0.], &[0.], &mut out, &mut g, &mut w)
        .unwrap();
    assert_eq!((out, g), ([0.], [0.]));
    assert!(matches!(
        p.value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut w),
        Err(E::Numerical)
    ));
    assert_eq!((out, g), ([0.], [0.]));
    p.value_and_vjp(&[4.], &[1.], &mut out, &mut g, &mut w)
        .unwrap();
    assert_eq!((out, g), ([2.], [0.25]));
}
fn recurrence(steps: usize) -> alea_core::model::tensor::BoundProgram {
    let mut body = ModelBuilder::new().region();
    let c = body.input("carry", shape(&[2])).unwrap();
    let row = body.input("row", shape(&[2])).unwrap();
    let factor = body.input("factor", Shape::scalar()).unwrap();
    let scaled = body.binary(B::Multiply, c, factor).unwrap();
    let next = body.binary(B::Add, scaled, row).unwrap();
    let body = body.compile(next).unwrap().bind(&[]).unwrap();
    let mut b = ModelBuilder::new().region();
    let initial = b.input("initial", shape(&[2])).unwrap();
    let factor = b.input("factor", Shape::scalar()).unwrap();
    let xs = b.data("observations", shape(&[steps, 2])).unwrap();
    let result = b.scan(initial, xs, &[factor], body).unwrap();
    let result = b.sum(result).unwrap();
    let program = b.compile(result).unwrap();
    assert_eq!(program.node_count(), 5);
    let data: Vec<_> = (0..2 * steps).map(|i| i as f64 / 10.).collect();
    program.bind(&[("observations", &data)]).unwrap()
}
#[test]
fn structured_scan_has_constant_graph_size_correct_capture_gradients_and_zero_trip_identity() {
    for steps in [0, 1, 7, 100] {
        let program = recurrence(steps);
        let x = [0.1, -0.2, 0.8];
        let mut expected = [x[0], x[1]];
        for i in 0..steps {
            for (j, value) in expected.iter_mut().enumerate() {
                *value = *value * x[2] + (2 * i + j) as f64 / 10.;
            }
        }
        let mut out = [0.];
        program
            .evaluate(&x, &mut out, &mut program.workspace())
            .unwrap();
        assert!((out[0] - expected.iter().sum::<f64>()).abs() < 1e-12);
        check_vjp(&program, &x, &[1.]);
        let source = program.rust_function().unwrap();
        assert_eq!(
            source.contains(&format!("for step in 0..{steps}")),
            steps != 0
        );
    }
}

#[test]
fn checkpointed_scan_preserves_nonlinear_vjp_rebinding_errors_and_allocation_budget() {
    // An odd trip count exercises the partial final segment. The retained body
    // tape exceeds the planner's memory threshold; immutable xs are NOT claimed
    // to disappear from the enclosing workspace.
    for steps in [257, 1025] {
        let width = 16;
        let mut body = ModelBuilder::new().region();
        let carry = body.input("carry", shape(&[width])).unwrap();
        let row = body.input("row", shape(&[width])).unwrap();
        let gain = body.input("gain", Shape::scalar()).unwrap();
        let decay = body.constant(Shape::scalar(), vec![0.8]).unwrap();
        let a = body.binary(B::Multiply, carry, decay).unwrap();
        let b = body.binary(B::Multiply, row, gain).unwrap();
        let mut next = body.binary(B::Add, a, b).unwrap();
        // Preserve a sizeable pure body, including inactive and active blocks.
        for _ in 0..8 {
            next = body.unary(U::Neg, next).unwrap();
        }
        let next = body.unary(U::Tanh, next).unwrap();
        let body = body.compile(next).unwrap().bind(&[]).unwrap();
        let retained_body_bytes = steps * body.workspace_bytes();
        let mut builder = ModelBuilder::new().region();
        let initial = builder.input("initial", shape(&[width])).unwrap();
        let gain = builder.input("gain", Shape::scalar()).unwrap();
        let rows = builder.data("rows", shape(&[steps, width])).unwrap();
        let end = builder
            .checkpointed_scan(initial, rows, &[gain], body.clone())
            .unwrap();
        let plan = builder.compile(end).unwrap();
        let data: Vec<_> = (0..steps * width).map(|i| (i % 13) as f64 / 16.).collect();
        let bound = plan.bind(&[("rows", &data)]).unwrap();
        let mut plain = ModelBuilder::new().region();
        let initial = plain.input("initial", shape(&[width])).unwrap();
        let gain = plain.input("gain", Shape::scalar()).unwrap();
        let rows = plain.data("rows", shape(&[steps, width])).unwrap();
        let end = plain.scan(initial, rows, &[gain], body).unwrap();
        let plain = plain
            .compile(end)
            .unwrap()
            .bind(&[("rows", &data)])
            .unwrap();
        assert_eq!(
            bound.rust_function().unwrap(),
            plain.rust_function().unwrap()
        );
        eprintln!(
            "checkpoint scan steps={steps}: native budget {} -> {} bytes",
            plain.workspace_bytes(),
            bound.workspace_bytes()
        );
        assert!(bound.workspace_bytes() < retained_body_bytes / 2);
        let mut work = bound.workspace_with_limit(bound.workspace_bytes()).unwrap();
        let mut retained = None;
        let allocations = allocation_counter::measure(|| retained = Some(bound.workspace()));
        assert!(allocations.bytes_current as usize <= bound.workspace_bytes());
        drop(retained);
        let x: Vec<_> = (0..=width).map(|i| 0.02 * (i + 1) as f64).collect();
        let seed: Vec<_> = (0..width).map(|i| (i as f64 - 7.) / 8.).collect();
        let mut out = vec![0.; width];
        let mut g = vec![0.; x.len()];
        bound
            .value_and_vjp(&x, &seed, &mut out, &mut g, &mut work)
            .unwrap();
        let (mut full_out, mut full_g) = (out.clone(), g.clone());
        plain
            .value_and_vjp(
                &x,
                &seed,
                &mut full_out,
                &mut full_g,
                &mut plain.workspace(),
            )
            .unwrap();
        assert_eq!((&out, &g), (&full_out, &full_g));
        // Independent scalar recurrence and analytic reverse, outside all timed
        // work. Store every carry here as a reference to the bounded native tape.
        let mut states = vec![x[..width].to_vec()];
        for row in data.chunks_exact(width) {
            states.push(
                states
                    .last()
                    .unwrap()
                    .iter()
                    .zip(row)
                    .map(|(&c, &r)| (c * 0.8 + r * x[width]).tanh())
                    .collect(),
            );
        }
        assert_eq!(&out, states.last().unwrap());
        let mut expected = seed.clone();
        let mut expected_gain = 0.;
        for step in (0..steps).rev() {
            let mut local_gain = 0.;
            for j in 0..width {
                let v = states[step + 1][j];
                let score = expected[j] * (1. - v * v);
                local_gain += score * data[step * width + j];
                expected[j] = score * 0.8;
            }
            expected_gain += local_gain;
        }
        expected.push(expected_gain);
        for (&a, &b) in g.iter().zip(&expected) {
            assert!((a - b).abs() <= 2e-12 * (1. + b.abs()), "{a} vs {b}");
        }
        check_vjp(&bound, &x, &seed);
        let before = (out.clone(), g.clone());
        let bad_data = vec![f64::MAX; data.len()];
        let bad = plan.bind(&[("rows", &bad_data)]).unwrap();
        let mut bad_x = x.clone();
        bad_x[width] = 2.;
        assert!(matches!(
            bad.value_and_vjp(&bad_x, &seed, &mut out, &mut g, &mut work),
            Err(E::Numerical)
        ));
        assert_eq!((&out, &g), (&before.0, &before.1));
        drop(bad);
        bound
            .value_and_vjp(&x, &seed, &mut out, &mut g, &mut work)
            .unwrap();
        assert_eq!((out.clone(), g.clone()), before);
        let zero_seed = vec![0.; width];
        let allocations = allocation_counter::measure(|| {
            bound
                .value_and_vjp(&x, &zero_seed, &mut out, &mut g, &mut work)
                .unwrap();
        });
        assert_eq!(allocations.count_total, 0);
        assert!(g.iter().all(|&v| v == 0.));
    }
}

#[test]
fn checkpoint_replay_keeps_first_iteration_activity_and_singular_gradient_errors() {
    let steps = 1025;
    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let _ = body.input("row", Shape::scalar()).unwrap();
    let gain = body.input("gain", Shape::scalar()).unwrap();
    let root = body.unary(U::Sqrt, carry).unwrap();
    let mut next = body.binary(B::Add, root, gain).unwrap();
    for _ in 0..16 {
        next = body.unary(U::Neg, next).unwrap();
    }
    let body = body.compile(next).unwrap().bind(&[]).unwrap();
    let full = steps * body.workspace_bytes();
    let mut b = ModelBuilder::new().region();
    let gain = b.input("gain", Shape::scalar()).unwrap();
    let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
    let rows = b.data("rows", shape(&[steps])).unwrap();
    let out = b.checkpointed_scan(zero, rows, &[gain], body).unwrap();
    let p = b
        .compile(out)
        .unwrap()
        .bind(&[("rows", &vec![0.; steps])])
        .unwrap();
    assert!(p.workspace_bytes() < full / 2);
    let mut w = p.workspace();
    let mut out = [0.];
    let mut g = [0.];
    p.value_and_vjp(&[0.5], &[1.], &mut out, &mut g, &mut w)
        .unwrap();
    let (mut c, mut d) = (0_f64, 0_f64);
    for step in 0..steps {
        d = if step == 0 {
            1.
        } else {
            d * 0.5 / c.sqrt() + 1.
        };
        c = c.sqrt() + 0.5;
    }
    assert_eq!(out, [c]);
    assert!((g[0] - d).abs() < 1e-12);
    let before = (out, g);
    assert!(matches!(
        p.value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut w),
        Err(E::Numerical)
    ));
    assert_eq!((out, g), before);
    p.value_and_vjp(&[0.5], &[1.], &mut out, &mut g, &mut w)
        .unwrap();
    assert_eq!((out, g), before);
}
#[test]
fn cached_bindings_and_sparse_adjoints_refresh_after_failure_and_snapshot_drop() {
    for n in [0, 1, 33, 129] {
        let mut t = ModelBuilder::new().region();
        let x = t.input("x", Shape::scalar()).unwrap();
        let weights = t.data("weights", shape(&[n])).unwrap();
        let product = t.binary(B::Multiply, x, weights).unwrap();
        let logs = t.unary(U::Log, product).unwrap();
        let total = t.sum(logs).unwrap();
        let plan = t.compile(total).unwrap();
        let initial = plan.bind(&[("weights", &vec![1.; n])]).unwrap();
        let mut work = initial.workspace();
        drop(initial); // Cached Arc identity must remain owned, never a dangling address.
        for count in 1..8 {
            let values: Vec<_> = (0..n).map(|i| 0.5 + (i % 7 + count) as f64 / 8.).collect();
            let binding = plan.bind(&[("weights", &values)]).unwrap();
            let mut out = [91.];
            let mut g = [92.];
            binding
                .value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut work)
                .unwrap();
            let expected: f64 = values.iter().map(|v| (2. * v).ln()).sum();
            assert!((out[0] - expected).abs() < 1e-12 * (1. + expected.abs()));
            assert!((g[0] - n as f64 / 2.).abs() < 1e-12);
            let bad = binding.rebind("weights", &vec![-1.; n]).unwrap();
            out = [91.];
            g = [92.];
            if n > 0 {
                assert!(
                    bad.value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut work)
                        .is_err()
                );
                assert_eq!(out, [91.]);
                assert_eq!(g, [92.]);
            }
            for seed in [0., 1., -1., 1.] {
                binding
                    .value_and_vjp(&[2.], &[seed], &mut out, &mut g, &mut work)
                    .unwrap();
                assert!((g[0] - seed * n as f64 / 2.).abs() < 1e-12);
            }
            let allocation = allocation_counter::measure(|| {
                binding
                    .value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut work)
                    .unwrap();
            });
            assert_eq!(allocation.count_total, 0);
        }
    }
}

#[test]
fn binding_snapshots_share_plan_and_rebinding_requires_consuming_the_compiled_model() {
    let mut t = ModelBuilder::new().region();
    let x = t.input("x", shape(&[3])).unwrap();
    let data = t.data("weights", shape(&[3])).unwrap();
    let out = t.binary(B::Multiply, x, data).unwrap();
    let out = t.sum(out).unwrap();
    let plan = t.compile(out).unwrap();
    let old = plan.bind(&[("weights", &[1., 2., 3.])]).unwrap();
    let new = old.rebind("weights", &[3., 2., 1.]).unwrap();
    let mut w = old.workspace();
    let mut result = [0.];
    old.evaluate(&[1., 2., 3.], &mut result, &mut w).unwrap();
    assert_eq!(result, [14.]);
    new.evaluate(&[1., 2., 3.], &mut result, &mut w).unwrap();
    assert_eq!(result, [10.]);
    old.evaluate(&[1., 2., 3.], &mut result, &mut w).unwrap();
    assert_eq!(result, [14.]);
    assert!(old.rebind("weights", &[1.]).is_err());
    assert!(old.rebind("weights", &[f64::NAN; 3]).is_err());
    let mut b = ModelBuilder::new();
    let inputs = [
        b.real("a").unwrap().expr(),
        b.real("b").unwrap().expr(),
        b.real("c").unwrap().expr(),
    ];
    let factor = b.apply_region(&inputs, old).unwrap();
    b.potential("objective", factor).unwrap();
    let model = b.compile().unwrap();
    let source = model.rust_primal().unwrap();
    let error = model
        .binding()
        .rebind_region(factor, "weights", &[0.])
        .err()
        .unwrap();
    assert!(matches!(error, E::Shape));
    assert_eq!(model.rust_primal().unwrap(), source);
    let model = model
        .binding()
        .rebind_region(factor, "weights", &[3., 2., 1.])
        .unwrap()
        .compile()
        .unwrap()
        .into_native_model();
    assert_ne!(model.rust_primal().unwrap(), source);
    let target = model.into_target().unwrap();
    let mut g = [0.; 3];
    assert_eq!(target.logp_grad(&[1., 2., 3.], &mut g).unwrap(), 10.);
    assert_eq!(g, [3., 2., 1.]);
}
#[test]
fn admission_checks_owners_shapes_empty_tensors_and_dead_operations() {
    assert!(Shape::new(vec![usize::MAX, 2]).is_err());
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", shape(&[0, 3])).unwrap();
    let zero = b.sum(x).unwrap();
    let unused = b.constant(Shape::scalar(), vec![-1.]).unwrap();
    let _ = b.unary(U::Log, unused).unwrap();
    let plan = b.compile(zero).unwrap();
    assert!(plan.nodes().any(|n| !n.active));
    let program = plan.bind(&[]).unwrap();
    let mut out = [1.];
    program
        .evaluate(&[], &mut out, &mut program.workspace())
        .unwrap();
    assert_eq!(out, [0.]);
    let mut model = ModelBuilder::new();
    let scalar = model.constant(1.).unwrap();
    let mut b = model.region();
    let local = b.input("x", shape(&[2])).unwrap();
    let mut other = model.region();
    let foreign = other.input("x", shape(&[2])).unwrap();
    assert_eq!(b.value_shape(local).unwrap(), &shape(&[2]));
    assert!(matches!(b.value_shape(scalar), Err(E::ForeignHandle)));
    assert!(matches!(model.factor(local), Err(E::ForeignHandle)));
    assert!(matches!(
        b.binary(B::Add, local, foreign),
        Err(E::ForeignHandle)
    ));
    assert!(b.reshape(local, shape(&[3])).is_err());
    assert!(b.gather(local, shape(&[1]), vec![2]).is_err());
    let nonscalar = other.compile(foreign).unwrap().bind(&[]).unwrap();
    assert!(matches!(
        model.apply_region(&[scalar, scalar], nonscalar),
        Err(E::Shape)
    ));
    let result = b.sum(local).unwrap();
    let program = b.compile(result).unwrap().bind(&[]).unwrap();
    assert!(matches!(
        model.apply_region(&[scalar], program.clone()),
        Err(E::Shape)
    ));
    assert!(matches!(
        model.apply_region(&[scalar, local], program.clone()),
        Err(E::ForeignHandle)
    ));
    let applied = model
        .apply_region(&[scalar, scalar], program.clone())
        .unwrap();
    model.factor(applied).unwrap();
    let mut out = [19.];
    let mut g = [20.; 2];
    let mut w = program.workspace();
    assert!(
        program
            .value_and_vjp(&[1., 2.], &[f64::NAN], &mut out, &mut g, &mut w)
            .is_err()
    );
    assert_eq!(out, [19.]);
    assert_eq!(g, [20.; 2]);
}

#[test]
fn elementary_block_pullbacks_cover_all_operations_and_scalar_broadcast_reduction() {
    for op in [
        U::Neg,
        U::Exp,
        U::Log,
        U::Log1p,
        U::Expm1,
        U::Tanh,
        U::Sin,
        U::Cos,
        U::Sqrt,
    ] {
        let mut b = ModelBuilder::new().region();
        let x = b.input("x", shape(&[3])).unwrap();
        let result = b.unary(op, x).unwrap();
        let p = b.compile(result).unwrap().bind(&[]).unwrap();
        check_vjp(&p, &[0.2, 0.7, 1.4], &[0.3, -0.2, 0.9]);
    }
    for op in [B::Add, B::Subtract, B::Multiply, B::Divide] {
        let mut b = ModelBuilder::new().region();
        let x = b.input("x", shape(&[3])).unwrap();
        let y = b.input("y", Shape::scalar()).unwrap();
        let result = b.binary(op, x, y).unwrap();
        let p = b.compile(result).unwrap().bind(&[]).unwrap();
        check_vjp(&p, &[0.2, -0.7, 1.4, 0.8], &[0.3, -0.2, 0.9]);
    }
}

#[test]
fn empty_matrix_does_no_outer_loop_work_and_empty_scan_metadata_overflow_is_rejected() {
    let mut b = ModelBuilder::new().region();
    let a = b.input("a", shape(&[usize::MAX, 0])).unwrap();
    let c = b.input("b", shape(&[0, 0])).unwrap();
    let result = b.matmul(a, c).unwrap();
    let p = b.compile(result).unwrap().bind(&[]).unwrap();
    p.value_and_vjp(&[], &[], &mut [], &mut [], &mut p.workspace())
        .unwrap();
    assert!(!p.rust_function().unwrap().contains("for row"));

    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", shape(&[0])).unwrap();
    let _ = body.input("row", shape(&[0])).unwrap();
    let body = body.compile(carry).unwrap().bind(&[]).unwrap();
    let mut outer = ModelBuilder::new().region();
    let carry = outer.input("carry", shape(&[0])).unwrap();
    let xs = outer.data("xs", shape(&[usize::MAX, 0])).unwrap();
    assert!(matches!(outer.scan(carry, xs, &[], body), Err(E::Shape)));
}

#[test]
fn non_differentiable_data_and_predicates_do_not_acquire_spurious_singular_derivatives() {
    let mut branch = ModelBuilder::new().region();
    let data = branch.input("data", Shape::scalar()).unwrap();
    let root = branch.unary(U::Sqrt, data).unwrap();
    let branch = branch.compile(root).unwrap().bind(&[]).unwrap();
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let data = b.data("data", Shape::scalar()).unwrap();
    let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
    let chosen = b
        .cond(
            x,
            zero,
            Comparison::Greater,
            &[data],
            branch.clone(),
            branch,
        )
        .unwrap();
    let result = b.binary(B::Add, x, chosen).unwrap();
    let p = b.compile(result).unwrap().bind(&[("data", &[0.])]).unwrap();
    check_vjp(&p, &[1.], &[1.]);
    check_vjp(&p, &[-1.], &[1.]);

    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let row = body.input("row", Shape::scalar()).unwrap();
    let root = body.unary(U::Sqrt, carry).unwrap();
    let next = body.binary(B::Add, root, row).unwrap();
    let body = body.compile(next).unwrap().bind(&[]).unwrap();
    let mut b = ModelBuilder::new().region();
    let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
    let xs = b.input("xs", shape(&[3])).unwrap();
    let result = b.scan(zero, xs, &[], body).unwrap();
    let p = b.compile(result).unwrap().bind(&[]).unwrap();
    check_vjp(&p, &[1., 2., 3.], &[1.]);
}

#[test]
fn specialized_binary_blocks_preserve_scalar_bits_for_aliases_broadcasts_and_inactive_data() {
    let values = [1e-100, -1e100, 0.3, -0.7];
    let other = [-1e100, 1e-100, 1.3, 0.8];
    let seeds = [0.3, 0., -0.7, 1.];
    for op in [B::Add, B::Subtract, B::Multiply, B::Divide] {
        for case in 0..6 {
            let mut b = ModelBuilder::new().region();
            let (a, r, q, amap, rmap) = match case {
                0 => (
                    b.input("a", shape(&[4])).unwrap(),
                    b.input("r", shape(&[4])).unwrap(),
                    [values, other].concat(),
                    Some((0, false)),
                    Some((4, false)),
                ),
                1 => (
                    b.input("a", shape(&[4])).unwrap(),
                    b.constant(Shape::scalar(), vec![0.8]).unwrap(),
                    values.to_vec(),
                    Some((0, false)),
                    None,
                ),
                2 => (
                    b.constant(Shape::scalar(), vec![0.8]).unwrap(),
                    b.input("r", shape(&[4])).unwrap(),
                    other.to_vec(),
                    None,
                    Some((0, false)),
                ),
                3 => (
                    b.input("a", Shape::scalar()).unwrap(),
                    b.constant(shape(&[4]), other.to_vec()).unwrap(),
                    vec![0.8],
                    Some((0, true)),
                    None,
                ),
                4 => (
                    b.constant(shape(&[4]), values.to_vec()).unwrap(),
                    b.input("r", Shape::scalar()).unwrap(),
                    vec![0.8],
                    None,
                    Some((0, true)),
                ),
                _ => {
                    let a = b.input("a", shape(&[4])).unwrap();
                    let r = b.reshape(a, shape(&[4])).unwrap();
                    (a, r, values.to_vec(), Some((0, false)), Some((0, false)))
                }
            };
            let out = b.binary(op, a, r).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            let mut actual = [0.; 4];
            let mut g = vec![0.; q.len()];
            p.value_and_vjp(&q, &seeds, &mut actual, &mut g, &mut p.workspace())
                .unwrap();
            let mut expected_g = vec![0.; q.len()];
            for j in 0..4 {
                let av = match case {
                    2 | 3 => 0.8,
                    _ => values[j],
                };
                let rv = match case {
                    1 | 4 => 0.8,
                    5 => values[j],
                    _ => other[j],
                };
                let (value, da, dr) = match op {
                    B::Add => (av + rv, 1., 1.),
                    B::Subtract => (av - rv, 1., -1.),
                    B::Multiply => (av * rv, rv, av),
                    B::Divide => (av / rv, 1. / rv, -(av / rv) / rv),
                    _ => unreachable!(),
                };
                assert_eq!(
                    actual[j].to_bits(),
                    value.to_bits(),
                    "{op:?}, case {case}, row {j}"
                );
                if seeds[j] != 0. {
                    if let Some((start, scalar)) = amap {
                        expected_g[start + if scalar { 0 } else { j }] += seeds[j] * da;
                    }
                    if let Some((start, scalar)) = rmap {
                        expected_g[start + if scalar { 0 } else { j }] += seeds[j] * dr;
                    }
                }
            }
            assert_eq!(
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{op:?}, case {case}"
            );
        }
    }
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", shape(&[4])).unwrap();
    let alias = b.reshape(x, shape(&[4])).unwrap();
    let square = b.binary(B::Multiply, x, alias).unwrap();
    let out = b.binary(B::Add, square, x).unwrap();
    let p = b.compile(out).unwrap().bind(&[]).unwrap();
    let q = [0.1, 0.2, -0.1, -0.2];
    let mut g = [0.; 4];
    p.value_and_vjp(&q, &[1.; 4], &mut [0.; 4], &mut g, &mut p.workspace())
        .unwrap();
    for (g, x) in g.into_iter().zip(q) {
        assert_eq!(g.to_bits(), ((1. + x) + x).to_bits());
    }
}

#[test]
fn model_block_edges_share_forward_residuals_and_accumulate_consumer_seeds() {
    use alea_core::model::{
        ConstrainedModel,
        assembly::OperationKind,
        tensor::{BlockRules, BlockType, ElementType},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct Calls {
        forward: AtomicUsize,
        reverse: AtomicUsize,
    }

    let calls = Arc::new(Calls::default());
    let mut model = ModelBuilder::new();
    let x = model.real("x").unwrap().expr();
    let mut region = model.region();
    let input = region.input("x", Shape::scalar()).unwrap();
    let count = region
        .count_data(Shape::scalar(), vec![(1 << 53) + 1])
        .unwrap();
    let enabled = region.boolean_data(Shape::scalar(), vec![true]).unwrap();
    let index = region.indices(Shape::scalar(), vec![1], 2).unwrap();
    let output = region
        .register_block(
            &[input, count, enabled, index],
            Arc::clone(&calls),
            BlockRules {
                inputs: [
                    ElementType::Real,
                    ElementType::Count,
                    ElementType::Bool,
                    ElementType::Index,
                ]
                .map(|dtype| BlockType {
                    dtype,
                    shape: Shape::scalar(),
                })
                .into(),
                output: shape(&[2]),
                workspace: |_| (0_u64, false),
                heap_bytes: 0,
                forward: |calls, inputs, out, state| {
                    calls.forward.fetch_add(1, Ordering::Relaxed);
                    state.0 = inputs.counts(1)?[0];
                    state.1 = inputs.booleans(2)?[0];
                    let i = inputs.indices(3)?[0];
                    let x = inputs.real(0)?[0];
                    if x < 0. {
                        return Err(E::Numerical);
                    }
                    out[1 - i] = 2. * x;
                    out[i] = 3. * x;
                    Ok(())
                },
                vjp: Some(|calls, _, _, seed, scores, state| {
                    calls.reverse.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(*state, ((1 << 53) + 1, true));
                    assert_eq!(scores.len(), 1);
                    scores[0] = 2. * seed[0] + 3. * seed[1];
                    Ok(())
                }),
            },
        )
        .unwrap();
    let bound = region.compile(output).unwrap().bind(&[]).unwrap();
    let block = model.call_region(&[x], bound).unwrap();
    assert_eq!(model.value_shape(block).unwrap(), shape(&[2]));
    assert!(matches!(model.factor(block), Err(E::Shape)));
    let left = model.sum_value(block).unwrap();
    let right = model.sum_value(block).unwrap();
    let root = model.add(left, right).unwrap();
    model.potential("native block", root).unwrap();
    let square = model.multiply(x, x).unwrap();
    let prior = model.multiply(-1., square).unwrap();
    model.prior("scalar prior", prior).unwrap();
    let model = model.compile().unwrap();
    let consumers = model
        .graph()
        .filter(|node| node.operation == OperationKind::Region && node.inputs == [block])
        .count();
    assert_eq!(consumers, 2);
    assert!(!model.source_capabilities().supports_rust_primal());
    assert!(matches!(model.rust_primal(), Err(E::MissingPrimal { .. })));

    let mut gradient = [42.];
    assert!(matches!(
        model.logp_grad(&[-1.], &mut gradient),
        Err(E::Numerical)
    ));
    assert_eq!(gradient, [42.]);
    let measured = allocation_counter::measure(|| {
        assert_eq!(model.logp_grad(&[2.], &mut gradient).unwrap(), 16.);
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(gradient, [6.]);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 2);
    assert_eq!(calls.reverse.load(Ordering::Relaxed), 1);
    let prior = model.select_prior().unwrap();
    assert!(prior.source_capabilities().supports_rust_primal());
    assert_eq!(prior.logp_grad(&[-1.], &mut gradient).unwrap(), -1.);
    assert_eq!(gradient, [2.]);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 2);
}

#[test]
fn model_matrix_edges_preserve_parameter_chart_and_rebind_named_data() {
    use alea_core::model::ConstrainedModel;

    let mut model = ModelBuilder::new();
    let beta = model.real_vector("beta", &["a", "b"]).unwrap();
    let beta = model.vector_value(&beta, shape(&[2, 1])).unwrap();
    let design = {
        let mut region = model.region();
        let x = region.data("X", shape(&[2, 2])).unwrap();
        region
            .compile(x)
            .unwrap()
            .bind(&[("X", &[1., 2., 3., 4.])])
            .unwrap()
    };
    let design = model.call_region(&[], design).unwrap();
    let eta = model.matmul(design, beta).unwrap();
    let eta = model.reshape(eta, shape(&[2])).unwrap();
    let root = model.sum_value(eta).unwrap();
    model.factor(root).unwrap();
    let model = model.compile().unwrap();
    let mut g = [0.; 2];
    assert_eq!(model.logp_grad(&[2., 3.], &mut g).unwrap(), 26.);
    assert_eq!(g, [4., 6.]);
    let err = model
        .binding()
        .rebind_region(design, "X", &[0.])
        .err()
        .unwrap();
    assert!(matches!(err, E::Shape));
    let model = model
        .binding()
        .rebind_region(design, "X", &[1., 0., 0., 1.])
        .unwrap()
        .compile()
        .unwrap()
        .into_native_model();
    assert_eq!(model.logp_grad(&[2., 3.], &mut g).unwrap(), 5.);
    assert_eq!(g, [1., 1.]);
}

#[test]
fn executable_broadcast_grid_matches_dense_coordinates_across_contexts_and_immutable_rebinds() {
    // A dense Cartesian oracle, independent of Map and its run/index renderer.
    let oracle = |rank, n, width, inverse, q: [f64; 2], data: &[f64]| {
        let lanes = if rank == 3 { 3 } else { 1 };
        let (mut value, mut score) = (0., [0.; 2]);
        for i in 0..n {
            for &r in data.iter().take(width) {
                for k in 0..lanes {
                    let slot = (i * lanes + k) % 2;
                    let z = if inverse { r / q[slot] } else { q[slot] / r };
                    value += z * z;
                    score[slot] += if inverse {
                        -2. * z * z / q[slot]
                    } else {
                        2. * z / r
                    };
                }
            }
        }
        (value, score)
    };
    for rank in [2, 3] {
        for (n, width) in [(0, 3), (1, 0), (1, 1), (3, 3), (17, 3)] {
            for inverse in [false, true] {
                let data: Vec<_> = (0..width).map(|j| [1., 2., 4.][j % 3]).collect();
                let mut model = ModelBuilder::new();
                let parameters = model.real_vector("q", &["a", "b"]).unwrap();
                let q = model.vector_value(&parameters, shape(&[2])).unwrap();
                let mut region = model.region();
                let input = region.input("q", shape(&[2])).unwrap();
                let lanes = if rank == 3 { 3 } else { 1 };
                let left_shape = if rank == 3 {
                    shape(&[n, 1, 3])
                } else {
                    shape(&[n, 1])
                };
                let right_shape = if rank == 3 {
                    shape(&[1, width, 1])
                } else {
                    shape(&[1, width])
                };
                let left = region
                    .gather(input, left_shape, (0..n * lanes).map(|i| i % 2).collect())
                    .unwrap();
                let right = region.data("r", right_shape.clone()).unwrap();
                let value = if inverse {
                    region.binary(B::Divide, right, left)
                } else {
                    region.binary(B::Divide, left, right)
                }
                .unwrap();
                let alias = region
                    .reshape(value, region.value_shape(value).unwrap().clone())
                    .unwrap();
                let square = region.binary(B::Multiply, value, alias).unwrap();
                let sum = region.sum(square).unwrap();
                let program = region.compile(sum).unwrap().bind(&[("r", &data)]).unwrap();
                let root = model.call_region(&[q], program).unwrap();
                model.factor(root).unwrap();
                let definition = model.finish().unwrap();
                let binding = definition.bind();
                let executable = binding.compile().unwrap();
                let (mut a, mut b) = (executable.context(), executable.context());
                for (point, context) in [([0.5, 1.5], &mut a), ([-0.5, -1.5], &mut b)] {
                    let expected = oracle(rank, n, width, inverse, point, &data);
                    let mut g = [19.; 2];
                    let value = executable.evaluate(context, &point, &mut g).unwrap();
                    assert!((value - expected.0).abs() < 1e-12 * (1. + expected.0.abs()));
                    for (x, y) in g.into_iter().zip(expected.1) {
                        assert!((x - y).abs() < 1e-12 * (1. + y.abs()));
                    }
                }
                // Whole model calls remain native-only; do not infer source AD
                // from this successful native map/call path.
                assert!(matches!(
                    executable.rust_primal(),
                    Err(E::MissingPrimal { .. })
                ));
                assert!(matches!(
                    binding.rebind_region(root, "r", &vec![1.; width + 1]),
                    Err(E::Shape)
                ));
                if width > 0 {
                    assert!(
                        binding
                            .rebind_region(root, "r", &vec![f64::NAN; width])
                            .is_err()
                    );
                    let changed: Vec<_> = data.iter().map(|r| 2. * r).collect();
                    let replacement = binding
                        .rebind_region(root, "r", &changed)
                        .unwrap()
                        .compile()
                        .unwrap();
                    let mut g = [19.; 2];
                    assert!(matches!(
                        replacement.evaluate(&mut a, &[0.5, 1.5], &mut g),
                        Err(E::ContextMismatch)
                    ));
                    assert_eq!(g, [19.; 2]);
                    let mut context = replacement.context();
                    let expected = oracle(rank, n, width, inverse, [0.5, 1.5], &changed);
                    let value = replacement
                        .evaluate(&mut context, &[0.5, 1.5], &mut g)
                        .unwrap();
                    assert!((value - expected.0).abs() < 1e-12 * (1. + expected.0.abs()));
                    for (x, y) in g.into_iter().zip(expected.1) {
                        assert!((x - y).abs() < 1e-12 * (1. + y.abs()));
                    }
                    if !inverse && n > 0 {
                        let mut bad = data.clone();
                        bad[0] = 0.;
                        let invalid = binding
                            .rebind_region(root, "r", &bad)
                            .unwrap()
                            .compile()
                            .unwrap();
                        let mut bad_context = invalid.context();
                        let mut g = [19.; 2];
                        assert!(matches!(
                            invalid.evaluate(&mut bad_context, &[0.5, 1.5], &mut g),
                            Err(E::Numerical)
                        ));
                        assert_eq!(g, [19.; 2]);
                    }
                }
                let mut g = [19.; 2];
                let expected = oracle(rank, n, width, inverse, [0.5, 1.5], &data);
                assert!(
                    (executable.evaluate(&mut a, &[0.5, 1.5], &mut g).unwrap() - expected.0).abs()
                        < 1e-12 * (1. + expected.0.abs())
                );
            }
        }
    }
}

#[test]
fn executable_lazy_broadcast_admits_selected_intermediates_even_for_empty_or_zero_weight_results() {
    for empty in [false, true] {
        for zero_weight in [false, true] {
            let branch = |negative| {
                let mut b = ModelBuilder::new().region();
                let x = b.input("x", Shape::scalar()).unwrap();
                let positive = if negative {
                    b.unary(U::Neg, x).unwrap()
                } else {
                    x
                };
                let log = b.unary(U::Log, positive).unwrap();
                let rows = if empty {
                    b.constant(shape(&[0, 3]), vec![]).unwrap()
                } else {
                    b.constant(shape(&[1, 3]), vec![1., 2., 3.]).unwrap()
                };
                let values = b.binary(B::Add, log, rows).unwrap();
                let out = b.sum(values).unwrap();
                b.compile(out).unwrap().bind(&[]).unwrap()
            };
            let mut model = ModelBuilder::new();
            let x = model.real("x").unwrap().expr();
            let mut region = model.region();
            let input = region.input("x", Shape::scalar()).unwrap();
            let zero = region.constant(Shape::scalar(), vec![0.]).unwrap();
            let value = region
                .cond(
                    input,
                    zero,
                    Comparison::Less,
                    &[input],
                    branch(true),
                    branch(false),
                )
                .unwrap();
            let region = region.compile(value).unwrap().bind(&[]).unwrap();
            let value = model.call_region(&[x], region).unwrap();
            let value = if zero_weight {
                model.multiply(value, 0.).unwrap()
            } else {
                value
            };
            model.factor(value).unwrap();
            let executable = model.finish().unwrap().bind().compile().unwrap();
            let mut context = executable.context();
            for point in [-0.5_f64, 0.5] {
                let mut g = [19.];
                let expected = if empty || zero_weight {
                    0.
                } else {
                    3. * point.abs().ln() + 6.
                };
                assert!(
                    (executable.evaluate(&mut context, &[point], &mut g).unwrap() - expected).abs()
                        < 1e-12
                );
                assert_eq!(g, [if empty || zero_weight { 0. } else { 3. / point }]);
                let mut g = [19.];
                assert!(matches!(
                    executable.evaluate(&mut context, &[0.], &mut g),
                    Err(E::Numerical)
                ));
                assert_eq!(g, [19.]);
            }
            assert!(
                executable
                    .evaluate(&mut context, &[0.5], &mut [19.])
                    .is_ok()
            );
        }
    }
}

#[test]
#[allow(clippy::single_element_loop)] // Optional providers extend this matrix.
fn matrix_source_capability_preserves_the_requested_provider_recursively() {
    use alea_core::model::ConstrainedModel;
    use alea_core::model::tensor::{BoundProgram, MatrixProvider, MatrixRequest};
    for kernel in [
        MatrixRequest::default(),
        #[cfg(any(feature = "faer", feature = "openblas"))]
        MatrixRequest::standard(MatrixProvider::Auto),
        #[cfg(feature = "faer")]
        MatrixRequest::standard(MatrixProvider::Faer),
        #[cfg(feature = "openblas")]
        MatrixRequest::standard(MatrixProvider::Blas),
    ] {
        let provider = kernel != MatrixRequest::default();
        let build = |scan_body| {
            let mut b = ModelBuilder::new().region();
            let x = b.input("x", Shape::scalar()).unwrap();
            if scan_body {
                let _ = b.input("row", Shape::scalar()).unwrap();
            }
            let a = b.reshape(x, shape(&[1, 1])).unwrap();
            let product = b.matmul_with_request(a, a, kernel).unwrap();
            let result = b.sum(product).unwrap();
            b.compile(result).unwrap().bind(&[]).unwrap()
        };
        let check =
            |p: BoundProgram, dimension: usize, expected: f64, derivative: f64, provider: bool| {
                let mut out = [19.];
                let mut g = vec![19.; dimension];
                p.value_and_vjp(
                    &vec![2.; dimension],
                    &[1.],
                    &mut out,
                    &mut g,
                    &mut p.workspace(),
                )
                .unwrap();
                assert_eq!(out, [expected]);
                assert_eq!(g[0], derivative);
                assert_eq!(p.rust_function().unwrap().contains("try_gemm"), provider);
                let mut model = ModelBuilder::new();
                let inputs: Vec<_> = (0..dimension)
                    .map(|i| model.real(&format!("x{i}")).unwrap().expr())
                    .collect();
                let root = model.apply_region(&inputs, p).unwrap();
                model.factor(root).unwrap();
                let model = model.compile().unwrap();
                let capabilities = model.source_capabilities();
                assert!(capabilities.supports_rust_primal());
                assert_eq!(capabilities.unqualified_curvature.is_empty(), !provider);
                assert!(model.rust_primal().is_ok());
                use alea_core::model::assembly::{
                    CurvatureRoute, DerivativeOrder, SourceMode, SourceRequest,
                };
                for mode in [SourceMode::Density, SourceMode::Objective] {
                    for order in [
                        DerivativeOrder::Value,
                        DerivativeOrder::First,
                        DerivativeOrder::Hvp,
                        DerivativeOrder::Third,
                    ] {
                        for curvature in [
                            CurvatureRoute::Unavailable,
                            CurvatureRoute::OriginalCompilerReverse,
                        ] {
                            let request = SourceRequest {
                                mode,
                                order,
                                curvature,
                            };
                            let blocked =
                                matches!(order, DerivativeOrder::Hvp | DerivativeOrder::Third)
                                    && (provider || curvature == CurvatureRoute::Unavailable);
                            match capabilities.resolve(request) {
                                Err(E::UnqualifiedDerivative { order: rejected }) if blocked => {
                                    assert_eq!(rejected, order);
                                }
                                Ok(resolved) if !blocked => assert_eq!(resolved, request),
                                result => panic!("{kernel:?}/{request:?}: {result:?}"),
                            }
                        }
                    }
                }
                if provider {
                    assert!(matches!(
                        model.source_first_order(SourceMode::Density),
                        Ok(Some(_))
                    ));
                }
                assert_eq!(
                    model.logp_grad(&vec![2.; dimension], &mut g).unwrap(),
                    expected
                );
            };
        check(build(false), 1, 4., 4., provider);
        // Nested regions preserve provider identity and curvature rejection.
        let mut outer = ModelBuilder::new().region();
        let x = outer.input("x", Shape::scalar()).unwrap();
        let zero = outer.constant(Shape::scalar(), vec![0.]).unwrap();
        let root = outer
            .cond(x, zero, Comparison::Less, &[x], build(false), build(false))
            .unwrap();
        check(
            outer.compile(root).unwrap().bind(&[]).unwrap(),
            1,
            4.,
            4.,
            provider,
        );
        for steps in [0, 1] {
            for checkpoint in [false, true] {
                let mut outer = ModelBuilder::new().region();
                let x = outer.input("x", Shape::scalar()).unwrap();
                let rows = outer.constant(shape(&[steps]), vec![0.; steps]).unwrap();
                let result = if checkpoint {
                    outer.checkpointed_scan(x, rows, &[], build(true)).unwrap()
                } else {
                    outer.scan(x, rows, &[], build(true)).unwrap()
                };
                check(
                    outer.compile(result).unwrap().bind(&[]).unwrap(),
                    1,
                    if steps == 0 { 2. } else { 4. },
                    if steps == 0 { 1. } else { 4. },
                    provider && steps != 0,
                );
            }
        }
        // An unreachable matrix node does not remove the selected root's source.
        let mut b = ModelBuilder::new().region();
        let x = b.input("x", Shape::scalar()).unwrap();
        let a = b.reshape(x, shape(&[1, 1])).unwrap();
        let _ = b.matmul_with_request(a, a, kernel).unwrap();
        check(b.compile(x).unwrap().bind(&[]).unwrap(), 1, 2., 1., false);
    }
}

#[test]
fn matrix_requests_reject_before_node_publication_and_report_original_preferences() {
    use alea_core::model::tensor::{
        MatrixArithmetic, MatrixCpu, MatrixPolicyError, MatrixProvider, MatrixRequest,
    };
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", shape(&[1, 1])).unwrap();
    let invalid = MatrixRequest {
        arithmetic: MatrixArithmetic::Compensated,
        provider: MatrixProvider::Faer,
        ..MatrixRequest::default()
    };
    assert!(
        matches!(b.matmul_with_request(x,x,invalid), Err(E::MatrixPolicy(MatrixPolicyError::Incompatible(r))) if r==invalid)
    );
    let first = b.matmul(x, x).unwrap();
    let request = MatrixRequest {
        cpu: MatrixCpu::Scalar,
        ..MatrixRequest::default()
    };
    let mut b = b.with_matrix_request(request).unwrap();
    let second = b.matmul(x, first).unwrap();
    let p = b.compile(second).unwrap().bind(&[]).unwrap();
    let reports = p.matrix_policies();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0].requested(), MatrixRequest::default());
    assert_eq!(reports[1].requested(), request);
    assert!(
        reports
            .iter()
            .all(|r| r.provider() == MatrixProvider::Reference)
    );
    let mut out = [99.];
    p.evaluate(&[2.], &mut out, &mut p.workspace()).unwrap();
    assert_eq!(out, [8.]);
}
