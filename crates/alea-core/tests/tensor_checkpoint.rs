//! Checkpoint replay through lazy regions, including row-input pullbacks.
use alea_core::model::tensor::{BinaryOp, BoundProgram, Comparison, Shape, TensorBuilder, UnaryOp};

fn shape(axes: &[usize]) -> Shape {
    Shape::new(axes.to_vec()).unwrap()
}

fn branch(negative: bool) -> BoundProgram {
    let mut b = TensorBuilder::new();
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

fn recurrence(steps: usize, checkpointed: bool) -> BoundProgram {
    let mut body = TensorBuilder::new();
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
    let mut b = TensorBuilder::new();
    let initial = b.input("initial", shape(&[16])).unwrap();
    let gain = b.input("gain", Shape::scalar()).unwrap();
    let rows = b.input("rows", shape(&[steps, 16])).unwrap();
    let out = if checkpointed {
        b.checkpointed_scan(initial, rows, &[gain], body).unwrap()
    } else {
        b.scan(initial, rows, &[gain], body).unwrap()
    };
    b.compile(out).unwrap().bind(&[]).unwrap()
}

#[test]
fn segment_replay_preserves_lazy_branch_paths_row_gradients_zero_trips_and_reuse() {
    for steps in [0, 1, 65, 1025] {
        let full = recurrence(steps, false);
        let compact = recurrence(steps, true);
        assert_eq!(full.rust_function(), compact.rust_function());
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
        }
    }
}

#[test]
fn retained_and_replayed_scans_preserve_finite_primal_singular_vjp_admission_and_recovery() {
    for steps in [1, 65, 1025] {
        for checkpointed in [false, true] {
            let mut body = TensorBuilder::new();
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
            let mut b = TensorBuilder::new();
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
        }
    }
}
