use alea_core::model::{
    assembly::{AssemblyError as E, ModelBuilder},
    tensor::{
        BinaryOp as B, Comparison, Shape, TensorBuilder, TensorFactor, TensorOperation,
        UnaryOp as U,
    },
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
        let mut builder = ModelBuilder::new();
        let inputs: Vec<_> = (0..program.dimension())
            .map(|i| builder.real(&format!("x{i}")).unwrap().expr())
            .collect();
        let factor = TensorFactor::new(program.clone(), &inputs).unwrap();
        assert_eq!(
            allocation_counter::measure(|| {
                assert!(factor.check_workspace_limit(bytes - 1).is_err());
                factor.check_workspace_limit(bytes).unwrap();
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
    let mut b = TensorBuilder::new();
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
fn matrix_block_vjp_matches_finite_differences_and_exact_product() {
    let mut b = TensorBuilder::new();
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
fn matrix_vector_specialization_preserves_ordered_compensated_bits_and_aliases() {
    use alea_math::numeric::CompensatedSum;
    for rows in [0, 1, 7, 33] {
        for cols in [0, 1, 3, 8, 17] {
            let mut b = TensorBuilder::new();
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
    let mut b = TensorBuilder::new();
    let a = b.input("a", shape(&[1, 3])).unwrap();
    let x = b.reshape(a, shape(&[3, 1])).unwrap();
    let out = b.matmul(a, x).unwrap();
    let p = b.compile(out).unwrap().bind(&[]).unwrap();
    check_vjp(&p, &[0.2, -0.5, 0.7], &[0.3]);
}
fn log_region(negative: bool) -> alea_core::model::tensor::BoundProgram {
    let mut b = TensorBuilder::new();
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
    let mut b = TensorBuilder::new();
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
    assert!(program.rust_function().contains("else {no"));
}
fn recurrence(steps: usize) -> alea_core::model::tensor::BoundProgram {
    let mut body = TensorBuilder::new();
    let c = body.input("carry", shape(&[2])).unwrap();
    let row = body.input("row", shape(&[2])).unwrap();
    let factor = body.input("factor", Shape::scalar()).unwrap();
    let scaled = body.binary(B::Multiply, c, factor).unwrap();
    let next = body.binary(B::Add, scaled, row).unwrap();
    let body = body.compile(next).unwrap().bind(&[]).unwrap();
    let mut b = TensorBuilder::new();
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
        let source = program.rust_function();
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
        let mut body = TensorBuilder::new();
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
        let mut builder = TensorBuilder::new();
        let initial = builder.input("initial", shape(&[width])).unwrap();
        let gain = builder.input("gain", Shape::scalar()).unwrap();
        let rows = builder.data("rows", shape(&[steps, width])).unwrap();
        let end = builder
            .checkpointed_scan(initial, rows, &[gain], body.clone())
            .unwrap();
        let plan = builder.compile(end).unwrap();
        let data: Vec<_> = (0..steps * width).map(|i| (i % 13) as f64 / 16.).collect();
        let bound = plan.bind(&[("rows", &data)]).unwrap();
        let mut plain = TensorBuilder::new();
        let initial = plain.input("initial", shape(&[width])).unwrap();
        let gain = plain.input("gain", Shape::scalar()).unwrap();
        let rows = plain.data("rows", shape(&[steps, width])).unwrap();
        let end = plain.scan(initial, rows, &[gain], body).unwrap();
        let plain = plain
            .compile(end)
            .unwrap()
            .bind(&[("rows", &data)])
            .unwrap();
        assert_eq!(bound.rust_function(), plain.rust_function());
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
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    body.input("row", Shape::scalar()).unwrap();
    let gain = body.input("gain", Shape::scalar()).unwrap();
    let root = body.unary(U::Sqrt, carry).unwrap();
    let mut next = body.binary(B::Add, root, gain).unwrap();
    for _ in 0..16 {
        next = body.unary(U::Neg, next).unwrap();
    }
    let body = body.compile(next).unwrap().bind(&[]).unwrap();
    let full = steps * body.workspace_bytes();
    let mut b = TensorBuilder::new();
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
        let mut t = TensorBuilder::new();
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
    let mut t = TensorBuilder::new();
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
    let factor = b
        .primitive(TensorFactor::new(old, &inputs).unwrap())
        .unwrap();
    b.potential("objective", factor).unwrap();
    let model = b.compile().unwrap();
    let source = model.rust_primal().unwrap();
    let error = model
        .rebind::<TensorFactor>(factor, |p| p.with_data("weights", &[0.]))
        .err()
        .unwrap();
    assert!(matches!(error.source, E::Shape));
    assert_eq!(error.model.rust_primal().unwrap(), source);
    let model = error
        .model
        .rebind::<TensorFactor>(factor, |p| p.with_data("weights", &[3., 2., 1.]))
        .unwrap();
    assert_ne!(model.rust_primal().unwrap(), source);
    let target = model.into_target().unwrap();
    let mut g = [0.; 3];
    assert_eq!(target.logp_grad(&[1., 2., 3.], &mut g).unwrap(), 10.);
    assert_eq!(g, [3., 2., 1.]);
}
#[test]
fn admission_checks_owners_shapes_empty_tensors_and_dead_operations() {
    assert!(Shape::new(vec![usize::MAX, 2]).is_err());
    let mut b = TensorBuilder::new();
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
    let mut b = TensorBuilder::new();
    let local = b.input("x", shape(&[2])).unwrap();
    let mut other = TensorBuilder::new();
    let foreign = other.input("x", shape(&[2])).unwrap();
    assert!(matches!(
        b.binary(B::Add, local, foreign),
        Err(E::ForeignHandle)
    ));
    assert!(b.reshape(local, shape(&[3])).is_err());
    assert!(b.gather(local, shape(&[1]), vec![2]).is_err());
    let result = b.sum(local).unwrap();
    let program = b.compile(result).unwrap().bind(&[]).unwrap();
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
        let mut b = TensorBuilder::new();
        let x = b.input("x", shape(&[3])).unwrap();
        let result = b.unary(op, x).unwrap();
        let p = b.compile(result).unwrap().bind(&[]).unwrap();
        check_vjp(&p, &[0.2, 0.7, 1.4], &[0.3, -0.2, 0.9]);
    }
    for op in [B::Add, B::Subtract, B::Multiply, B::Divide] {
        let mut b = TensorBuilder::new();
        let x = b.input("x", shape(&[3])).unwrap();
        let y = b.input("y", Shape::scalar()).unwrap();
        let result = b.binary(op, x, y).unwrap();
        let p = b.compile(result).unwrap().bind(&[]).unwrap();
        check_vjp(&p, &[0.2, -0.7, 1.4, 0.8], &[0.3, -0.2, 0.9]);
    }
}

#[test]
fn empty_matrix_does_no_outer_loop_work_and_empty_scan_metadata_overflow_is_rejected() {
    let mut b = TensorBuilder::new();
    let a = b.input("a", shape(&[usize::MAX, 0])).unwrap();
    let c = b.input("b", shape(&[0, 0])).unwrap();
    let result = b.matmul(a, c).unwrap();
    let p = b.compile(result).unwrap().bind(&[]).unwrap();
    p.value_and_vjp(&[], &[], &mut [], &mut [], &mut p.workspace())
        .unwrap();
    assert!(!p.rust_function().contains("for row"));

    let mut body = TensorBuilder::new();
    let carry = body.input("carry", shape(&[0])).unwrap();
    body.input("row", shape(&[0])).unwrap();
    let body = body.compile(carry).unwrap().bind(&[]).unwrap();
    let mut outer = TensorBuilder::new();
    let carry = outer.input("carry", shape(&[0])).unwrap();
    let xs = outer.data("xs", shape(&[usize::MAX, 0])).unwrap();
    assert!(matches!(outer.scan(carry, xs, &[], body), Err(E::Shape)));
}

#[test]
fn non_differentiable_data_and_predicates_do_not_acquire_spurious_singular_derivatives() {
    let mut branch = TensorBuilder::new();
    let data = branch.input("data", Shape::scalar()).unwrap();
    let root = branch.unary(U::Sqrt, data).unwrap();
    let branch = branch.compile(root).unwrap().bind(&[]).unwrap();
    let mut b = TensorBuilder::new();
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

    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let row = body.input("row", Shape::scalar()).unwrap();
    let root = body.unary(U::Sqrt, carry).unwrap();
    let next = body.binary(B::Add, root, row).unwrap();
    let body = body.compile(next).unwrap().bind(&[]).unwrap();
    let mut b = TensorBuilder::new();
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
            let mut b = TensorBuilder::new();
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
    let mut b = TensorBuilder::new();
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
