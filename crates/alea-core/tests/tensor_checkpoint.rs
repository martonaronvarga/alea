//! Checkpoint replay through lazy regions, including row-input pullbacks.
use alea_core::capability::{PotentialHessianVector, evaluate_hvp};
use alea_core::model::assembly::ModelBuilder;
use alea_core::model::tensor::{BinaryOp, BoundProgram, Comparison, Shape, UnaryOp};

fn shape(axes: &[usize]) -> Shape {
    Shape::new(axes.to_vec()).unwrap()
}

fn branch(negative: bool) -> BoundProgram {
    let mut b = ModelBuilder::new().region();
    let carry = b.input("carry", shape(&[16])).unwrap();
    let row = b.input("row", shape(&[16])).unwrap();
    let gain = b.input("gain", Shape::scalar()).unwrap();
    let value = if negative {
        b.unary(UnaryOp::Neg, carry).unwrap()
    } else {
        carry
    };
    let log = b.unary(UnaryOp::Log, value).unwrap();
    let scaled = b.binary(BinaryOp::Multiply, row, gain).unwrap();
    let out = b.binary(BinaryOp::Add, log, scaled).unwrap();
    b.compile(out).unwrap().bind(&[]).unwrap()
}

fn recurrence(steps: usize, checkpointed: bool, scalar: bool) -> BoundProgram {
    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", shape(&[16])).unwrap();
    let row = body.input("row", shape(&[16])).unwrap();
    let gain = body.input("gain", Shape::scalar()).unwrap();
    let first = body.gather(carry, Shape::scalar(), vec![0]).unwrap();
    let zero = body.constant(Shape::scalar(), vec![0.]).unwrap();
    let out = body
        .cond(
            first,
            zero,
            Comparison::Less,
            &[carry, row, gain],
            branch(true),
            branch(false),
        )
        .unwrap();
    let body = body.compile(out).unwrap().bind(&[]).unwrap();
    let mut b = ModelBuilder::new().region();
    let initial = b.input("initial", shape(&[16])).unwrap();
    let gain = b.input("gain", Shape::scalar()).unwrap();
    let rows = b.input("rows", shape(&[steps, 16])).unwrap();
    let out = if checkpointed {
        b.checkpointed_scan(initial, rows, &[gain], body).unwrap()
    } else {
        b.scan(initial, rows, &[gain], body).unwrap()
    };
    let out = if scalar {
        b.gather(out, Shape::scalar(), vec![0]).unwrap()
    } else {
        out
    };
    b.compile(out).unwrap().bind(&[]).unwrap()
}

#[test]
fn segment_replay_preserves_lazy_branch_paths_row_gradients_zero_trips_and_reuse() {
    for steps in [0, 1, 65, 1025] {
        let full = recurrence(steps, false, false);
        let compact = recurrence(steps, true, false);
        assert_eq!(
            full.rust_function().unwrap(),
            compact.rust_function().unwrap()
        );
        if steps == 1025 {
            assert!(compact.workspace_bytes() < full.workspace_bytes() / 4);
        }
        let mut w = compact.workspace();
        let mut full_w = full.workspace();
        let mut x = vec![0.125; full.dimension()];
        x[16] = 0.5;
        let mut out = [0.; 16];
        let mut expected = out;
        let mut gradient = vec![0.; x.len()];
        let mut expected_gradient = gradient.clone();
        let mut full_jvp = full.jvp(&[true, true, true]).unwrap();
        let mut compact_jvp = compact.jvp(&[true, true, true]).unwrap();
        let mut full_direction = [0.; 16];
        let mut compact_direction = [0.; 16];
        let initial_direction = [0.019; 16];
        let gain_direction = [-0.025];
        let row_direction = vec![0.031; steps * 16];
        let directions = [
            Some(&initial_direction[..]),
            Some(&gain_direction[..]),
            Some(&row_direction[..]),
        ];

        for initial in [0.2, -0.2, 0.8, -0.8] {
            x[..16].fill(initial);
            let mut seed = [0.; 16];
            seed[0] = 0.5;
            compact
                .value_and_vjp(&x, &seed, &mut out, &mut gradient, &mut w)
                .unwrap();
            full.value_and_vjp(
                &x,
                &seed,
                &mut expected,
                &mut expected_gradient,
                &mut full_w,
            )
            .unwrap();
            assert_eq!(out.map(f64::to_bits), expected.map(f64::to_bits));
            assert_eq!(
                gradient.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected_gradient
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            let allocation = allocation_counter::measure(|| {
                compact
                    .value_and_vjp(&x, &seed, &mut out, &mut gradient, &mut w)
                    .unwrap();
            });
            assert_eq!(allocation.count_total, 0);
            full_jvp
                .value_and_jvp(&x, &directions, &mut expected, &mut full_direction)
                .unwrap();
            let allocation = allocation_counter::measure(|| {
                compact_jvp
                    .value_and_jvp(&x, &directions, &mut out, &mut compact_direction)
                    .unwrap();
            });
            assert_eq!(allocation.count_total, 0);
            assert_eq!(out.map(f64::to_bits), expected.map(f64::to_bits));
            assert_eq!(
                compact_direction.map(f64::to_bits),
                full_direction.map(f64::to_bits)
            );
            let (mut y, mut dy) = (initial, initial_direction[0]);
            for _ in 0..steps {
                dy = dy / y + 0.5 * 0.031 + 0.125 * gain_direction[0];
                y = y.abs().ln() + 0.5 * 0.125;
            }
            assert!((out[0] - y).abs() <= 2e-13 * (1. + y.abs()));
            assert!((compact_direction[0] - dy).abs() <= 2e-10 * (1. + dy.abs()));
            if steps != 0 {
                let saved = (out, compact_direction);
                // Flip one carry lane while the predicate follows the first lane.
                x[1] = -initial;
                assert!(
                    compact_jvp
                        .value_and_jvp(&x, &directions, &mut out, &mut compact_direction)
                        .is_err()
                );
                assert_eq!((out, compact_direction), saved);
                x[1] = initial;
                compact_jvp
                    .value_and_jvp(&x, &directions, &mut out, &mut compact_direction)
                    .unwrap();
                assert_eq!((out, compact_direction), saved);
            }
        }
    }
}

#[test]
fn retained_and_replayed_scans_preserve_finite_primal_singular_vjp_admission_and_recovery() {
    for steps in [1, 65, 1025] {
        for checkpointed in [false, true] {
            let mut body = ModelBuilder::new().region();
            let carry = body.input("carry", Shape::scalar()).unwrap();
            let row = body.input("row", Shape::scalar()).unwrap();
            let root = body.unary(UnaryOp::Sqrt, carry).unwrap();
            let mut next = body.binary(BinaryOp::Add, root, row).unwrap();
            // Keep enough retained body state to exercise actual segmented
            // replay, not just a checkpoint request declined for a small tape.
            for _ in 0..16 {
                next = body.unary(UnaryOp::Neg, next).unwrap();
            }
            let body = body.compile(next).unwrap().bind(&[]).unwrap();
            let retained = steps * body.workspace_bytes();
            let mut b = ModelBuilder::new().region();
            let initial = b.input("initial", Shape::scalar()).unwrap();
            let rows = b.constant(shape(&[steps]), vec![0.; steps]).unwrap();
            let result = if checkpointed {
                b.checkpointed_scan(initial, rows, &[], body).unwrap()
            } else {
                b.scan(initial, rows, &[], body).unwrap()
            };
            let p = b.compile(result).unwrap().bind(&[]).unwrap();
            if checkpointed && steps == 1025 {
                assert!(p.workspace_bytes() < retained / 2);
            }
            let mut w = p.workspace();
            let (mut out, mut g) = ([91.], [92.]);
            p.evaluate(&[0.], &mut out, &mut w).unwrap();
            assert_eq!(out, [0.]);
            p.value_and_vjp(&[0.], &[0.], &mut out, &mut g, &mut w)
                .unwrap();
            assert_eq!((out, g), ([0.], [0.]));
            out[0] = 91.;
            g[0] = 92.;
            assert!(matches!(
                p.value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut w),
                Err(alea_core::model::assembly::AssemblyError::Numerical)
            ));
            assert_eq!((out, g), ([91.], [92.]));
            p.value_and_vjp(&[4.], &[1.], &mut out, &mut g, &mut w)
                .unwrap();
            let (mut value, mut gradient) = (4_f64, 1_f64);
            for _ in 0..steps {
                value = value.sqrt();
                gradient *= 0.5 / value;
            }
            assert_eq!(out, [value]);
            assert!((g[0] - gradient).abs() <= 1e-13 * gradient.abs());

            let mut jvp = p.jvp(&[true]).unwrap();
            jvp.value_and_jvp(&[0.], &[Some(&[0.])], &mut out, &mut g)
                .unwrap();
            assert_eq!((out, g), ([0.], [0.]));
            (out, g) = ([91.], [92.]);
            assert!(matches!(
                jvp.value_and_jvp(&[0.], &[Some(&[1.])], &mut out, &mut g),
                Err(alea_core::model::assembly::AssemblyError::Numerical)
            ));
            assert_eq!((out, g), ([91.], [92.]));
            jvp.value_and_jvp(&[4.], &[Some(&[1.])], &mut out, &mut g)
                .unwrap();
            assert_eq!(out, [value]);
            assert!((g[0] - gradient).abs() <= 1e-13 * gradient.abs());
        }
    }
}

#[test]
fn scan_jvp_reuses_carry_scratch_when_direction_becomes_active_and_captures_repeat() {
    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let row = body.input("row", Shape::scalar()).unwrap();
    let left = body.input("left", Shape::scalar()).unwrap();
    let right = body.input("right", Shape::scalar()).unwrap();
    let half = body.constant(Shape::scalar(), vec![0.5]).unwrap();
    let carry = body.binary(BinaryOp::Multiply, carry, half).unwrap();
    let scaled = body.binary(BinaryOp::Multiply, row, left).unwrap();
    let sum = body.binary(BinaryOp::Add, carry, scaled).unwrap();
    let sum = body.binary(BinaryOp::Add, sum, right).unwrap();
    let body = body.compile(sum).unwrap().bind(&[]).unwrap();
    for steps in [0, 1, 4] {
        let mut b = ModelBuilder::new().region();
        let init = b.input("init", Shape::scalar()).unwrap();
        let rows = b.input("rows", shape(&[steps])).unwrap();
        let capture = b.input("capture", Shape::scalar()).unwrap();
        let out = b
            .scan(init, rows, &[capture, capture], body.clone())
            .unwrap();
        let bound = b.compile(out).unwrap().bind(&[]).unwrap();
        let x: Vec<_> = [2.]
            .into_iter()
            .chain((1..=steps).map(|i| i as f64))
            .chain([0.75])
            .collect();
        let (mut expected, mut tangent) = (2., 0.);
        for &row in &x[1..1 + steps] {
            expected = expected / 2. + (row + 1.) * 0.75;
            tangent = tangent / 2. + (row + 1.) * 0.25;
        }
        let (mut value, mut direction) = ([17.], [19.]);
        bound
            .jvp(&[false, false, true])
            .unwrap()
            .value_and_jvp(&x, &[None, None, Some(&[0.25])], &mut value, &mut direction)
            .unwrap();
        assert_eq!((value, direction), ([expected], [tangent]));
        let mut gradient = vec![0.; x.len()];
        bound
            .value_and_vjp(&x, &[1.], &mut value, &mut gradient, &mut bound.workspace())
            .unwrap();
        assert_eq!(gradient[x.len() - 1] * 0.25, tangent);
        assert_eq!(gradient[0], 0.5_f64.powi(steps as i32));
        let mut v = vec![0.125; x.len()];
        v[x.len() - 1] = 0.25;
        let mut hvp = vec![19.; x.len()];
        evaluate_hvp(&bound.potential_hvp_target().unwrap(), &x, &v, &mut hvp).unwrap();
        let mut expected_capture = 0.;
        for row in 0..steps {
            let coefficient = 0.5_f64.powi((steps - row - 1) as i32);
            assert_eq!(hvp[row + 1], -coefficient * v[x.len() - 1]);
            expected_capture -= coefficient * v[row + 1];
        }
        assert_eq!(hvp[0], 0.);
        assert_eq!(hvp[x.len() - 1], expected_capture);
    }
}

#[test]
fn nested_scan_curvature_matches_a_dense_recurrence_across_checkpoints() {
    for steps in [0, 1, 7, 1025] {
        let full = recurrence(steps, false, true);
        let compact = recurrence(steps, true, true);
        if steps == 1025 {
            assert!(compact.workspace_bytes() < full.workspace_bytes());
        }
        let retained = full.potential_hvp_target().unwrap();
        let replayed = compact.potential_hvp_target().unwrap();
        let (mut x, mut v) = (vec![0.125; full.dimension()], vec![0.031; full.dimension()]);
        x[..16].fill(-1.4);
        v[..16].fill(0.019);
        x[16] = 0.5;
        v[16] = -0.025;
        for (step, row) in x[17..].chunks_exact_mut(16).enumerate() {
            row.fill(if step % 2 == 0 { 3. } else { -3. });
        }
        let (mut reference, mut output) = (vec![0.; x.len()], vec![0.; x.len()]);
        evaluate_hvp(&retained, &x, &v, &mut reference).unwrap();
        evaluate_hvp(&replayed, &x, &v, &mut output).unwrap();
        let allocs = allocation_counter::measure(|| {
            replayed.potential_hvp(&x, &v, &mut output).unwrap();
        });
        assert_eq!(allocs.count_total, 0);
        assert!(
            reference
                .iter()
                .zip(&output)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        // Independent recurrence for lane zero. Differentiate each component
        // of its forward gradient using d(log|c|)' = -dc/c².
        let mut carry = x[0];
        let mut dc = v[0];
        let (mut g, mut dg) = (vec![0.; x.len()], vec![0.; x.len()]);
        g[0] = 1.;
        for step in 0..steps {
            let row = 17 + 16 * step;
            for (grad, dgrad) in g.iter_mut().zip(&mut dg) {
                *dgrad = *dgrad / carry - (*grad / carry) * (dc / carry);
                *grad /= carry;
            }
            g[16] += x[row];
            g[row] += x[16];
            dg[16] += v[row];
            dg[row] += v[16];
            dc = dc / carry + v[16] * x[row] + x[16] * v[row];
            carry = carry.abs().ln() + x[16] * x[row];
        }
        for (actual, expected) in output.iter().zip(dg) {
            assert!(
                (actual + expected).abs() < 2e-11 * (1. + expected.abs()),
                "{actual} vs {}",
                -expected
            );
        }
        if steps != 0 {
            let saved = output.clone();
            x[1] = -x[0];
            assert!(replayed.potential_hvp(&x, &v, &mut output).is_err());
            assert_eq!(output, saved);
            x[1] = x[0];
            replayed.potential_hvp(&x, &v, &mut output).unwrap();
            assert_eq!(output, saved);
        }
    }
}
