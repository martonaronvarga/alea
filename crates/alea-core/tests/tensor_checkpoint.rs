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
