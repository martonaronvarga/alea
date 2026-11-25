//! Bounded partial evaluation of existing deterministic builtin calls. Failures
//! stay at their original admission node; callbacks/control/provider calls are
//! never evaluated here. The canonical graph remains the execution fallback.
use super::*;

// Bound automatic retained f64 payloads, not immutable input data or allocator
// bookkeeping. Declining preparation never declines an otherwise valid binding.
const MAX_FIXED_BYTES: usize = 8 * 1024 * 1024;
pub(super) enum Fixed {
    Value(Arc<[f64]>),
    Numerical,
}
pub(super) fn materialize(plan: &Plan, data: &[Arc<[f64]>]) -> Vec<Option<Fixed>> {
    materialize_with_limit(plan, data, MAX_FIXED_BYTES)
}
fn materialize_with_limit(plan: &Plan, data: &[Arc<[f64]>], limit: usize) -> Vec<Option<Fixed>> {
    let Some(last) = plan.facts.last_binding_fixed() else {
        return Vec::new();
    };
    if limit == 0 {
        return Vec::new();
    }
    let mut known: Vec<Option<Arc<[f64]>>> = vec![None; last + 1];
    let mut fixed = Vec::new();
    let mut bytes = 0;
    for (i, node) in plan.nodes[..=last].iter().enumerate() {
        if !plan.active[i] {
            continue;
        }
        match &node.op {
            Op::Data(j) => {
                known[i] = Some(Arc::clone(&data[*j]));
                continue;
            }
            Op::Constant(value) => {
                known[i] = Some(Arc::clone(value));
                continue;
            }
            Op::Alias(v) => {
                known[i] = known[v.node].clone();
                continue;
            }
            _ => {}
        }
        if !plan.facts.binding_fixed(i) {
            continue;
        }
        let needed = node.shape.elements * size_of::<f64>();
        if needed > limit - bytes || node.inputs().iter().any(|v| known[v.node].is_none()) {
            continue;
        }
        let value = |v: Value| known[v.node].as_deref().expect("checked fixed dependency");
        let mut out = vec![0.; node.shape.elements];
        match &node.op {
            Op::Unary(op, v) => kernels::unary_forward(*op, value(*v), &mut out),
            Op::Binary(b) => kernels::forward(b, value(b.a), value(b.b), &mut out),
            Op::Sum(v) => out[0] = kernels::sum_forward(value(*v)),
            Op::Gather(v, indices) => kernels::gather_forward(value(*v), indices, &mut out),
            Op::Matmul {
                a,
                b,
                m,
                k,
                n,
                kernel,
            } => {
                if !out.is_empty() {
                    kernels::matmul_forward(value(*a), value(*b), &mut out, (*m, *k, *n), *kernel);
                }
            }
            _ => unreachable!("checked binding builtin"),
        }
        fixed.resize_with(i + 1, || None);
        if alea_math::numeric::first_nonfinite(&out).is_some() {
            fixed[i] = Some(Fixed::Numerical);
        } else {
            bytes += needed;
            let out: Arc<[f64]> = Arc::from(out);
            known[i] = Some(Arc::clone(&out));
            fixed[i] = Some(Fixed::Value(out));
        }
    }
    fixed
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (BoundProgram, Value, Value, Value) {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![2, 1]).unwrap()).unwrap();
        let data = b.data("data", Shape::new(vec![2, 2]).unwrap()).unwrap();
        let neg = b.unary(UnaryOp::Neg, data).unwrap();
        let transpose = b
            .gather(neg, Shape::new(vec![2, 2]).unwrap(), vec![0, 2, 1, 3])
            .unwrap();
        let matrix = b.matmul(transpose, data).unwrap();
        let y = b.matmul(matrix, x).unwrap();
        let square = b.binary(BinaryOp::Multiply, y, y).unwrap();
        let out = b.sum(square).unwrap();
        (
            b.compile(out)
                .unwrap()
                .bind(&[("data", &[1., 2., 3., 4.])])
                .unwrap(),
            matrix,
            y,
            neg,
        )
    }
    fn numbers(p: &BoundProgram, x: &[f64], w: &mut TensorWorkspace) -> ([f64; 1], [f64; 2]) {
        let (mut out, mut g) = ([0.], [0.; 2]);
        p.value_and_vjp(x, &[1.], &mut out, &mut g, w).unwrap();
        (out, g)
    }
    #[test]
    fn fixed_design_transforms_match_uncached_bits_refresh_and_frozen_source() {
        let (p, matrix, dynamic, _) = fixture();
        assert!(
            matches!(p.fixed(matrix.node), Some(Fixed::Value(v)) if **v == [-10., -14., -14., -20.])
        );
        assert!(p.fixed(dynamic.node).is_none());
        let mut original = p.clone();
        // Independent scheduling control of the same canonical operations,
        // without exposing a permanent uncached API or copying their formulas.
        original.snapshot = None;
        let (mut cached, mut plain) = (p.workspace(), original.workspace());
        for x in [[0.5, -1.], [0., 0.], [-2., 0.3], [1e-40, -2e-40]] {
            let (a, ga) = numbers(&p, &x, &mut cached);
            let (b, gb) = numbers(&original, &x, &mut plain);
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
            assert_eq!(ga.map(f64::to_bits), gb.map(f64::to_bits));
        }
        assert_eq!(
            numbers(&p, &[0.5, -1.], &mut cached),
            ([250.], [-544., -772.])
        );
        assert_eq!(
            allocation_counter::measure(|| {
                for _ in 0..16 {
                    numbers(&p, &[0.5, -1.], &mut cached);
                }
            })
            .count_total,
            0
        );
        let source = p.rust_function().unwrap();
        assert!(source.contains(&format!("n{}[0] = -10.0_f64;", matrix.node)));
        let next = p.rebind("data", &[2., 0., 1., 3.]).unwrap();
        assert_eq!(
            numbers(&next, &[0.5, -1.], &mut cached),
            ([56.5], [-50., -138.])
        );
        assert_ne!(source, next.rust_function().unwrap());
        assert_eq!(source, p.rust_function().unwrap());
        assert_eq!(
            numbers(&p, &[0.5, -1.], &mut cached),
            ([250.], [-544., -772.])
        );
        let (mut out, mut g) = ([17.], [19., 23.]);
        assert!(matches!(
            p.value_and_vjp(&[f64::MAX; 2], &[1.], &mut out, &mut g, &mut cached),
            Err(E::Numerical)
        ));
        assert_eq!((out, g), ([17.], [19., 23.]));
        assert_eq!(
            numbers(&p, &[0.5, -1.], &mut cached),
            ([250.], [-544., -772.])
        );
        p.evaluate(&[0.5, -1.], &mut out, &mut cached).unwrap();
        assert!(matches!(
            p.rebind("data", &[f64::NAN; 4]),
            Err(E::Numerical)
        ));
        p.reverse_point(&[1.], &mut g, &mut cached).unwrap();
        assert_eq!(g, [-544., -772.]);
        p.evaluate(&[0.5, -1.], &mut out, &mut cached).unwrap();
        assert!(matches!(
            next.reverse_point(&[1.], &mut g, &mut cached),
            Err(E::Shape)
        ));
    }
    #[test]
    fn bounded_materialization_declines_without_changing_admission_or_allocating_parameters() {
        let (p, _, _, neg) = fixture();
        assert!(materialize_with_limit(&p.plan, &p.data, 0).is_empty());
        let fixed = materialize_with_limit(&p.plan, &p.data, 32);
        assert!(matches!(&fixed[neg.node], Some(Fixed::Value(v)) if v.len() == 4));
        let bytes: usize = fixed
            .iter()
            .flatten()
            .map(|f| match f {
                Fixed::Value(v) => v.len() * 8,
                Fixed::Numerical => 0,
            })
            .sum();
        assert_eq!(bytes, 32);
        let mut b = RegionBuilder::new();
        // Parameter storage is never allocated by binding, even beside fixed work.
        let _unused = b
            .input("huge", Shape::new(vec![usize::MAX / 1024]).unwrap())
            .unwrap();
        let data = b.data("data", Shape::scalar()).unwrap();
        let out = b.unary(UnaryOp::Neg, data).unwrap();
        let p = b.compile(out).unwrap().bind(&[("data", &[2.])]).unwrap();
        assert!(matches!(p.fixed(out.node), Some(Fixed::Value(v)) if **v == [-2.]));
    }
    #[test]
    fn deferred_builtin_errors_do_not_execute_callbacks_or_factories_at_binding() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Calls {
            work: AtomicUsize,
            forward: AtomicUsize,
            fail: AtomicBool,
        }
        let calls = Arc::new(Calls {
            work: AtomicUsize::new(0),
            forward: AtomicUsize::new(0),
            fail: AtomicBool::new(true),
        });
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let opaque = b
            .register_block(
                &[x],
                Arc::clone(&calls),
                BlockRules {
                    inputs: vec![BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar(),
                    }],
                    output: Shape::scalar(),
                    heap_bytes: 0,
                    preparation: None,
                    workspace: |c| {
                        c.work.fetch_add(1, Ordering::Relaxed);
                    },
                    forward: |c, args, out, _| {
                        c.forward.fetch_add(1, Ordering::Relaxed);
                        if c.fail.load(Ordering::Relaxed) {
                            return Err(E::Name);
                        }
                        out[0] = args.real(0)?[0];
                        Ok(())
                    },
                    vjp: Some(|_, _, _, seed, g, _| {
                        g[0] = seed[0];
                        Ok(())
                    }),
                },
            )
            .unwrap();
        let data = b.data("bad", Shape::scalar()).unwrap();
        let log = b.unary(UnaryOp::Log, data).unwrap();
        let out = b.binary(BinaryOp::Add, opaque, log).unwrap();
        let p = b.compile(out).unwrap().bind(&[("bad", &[-1.])]).unwrap();
        assert!(matches!(p.fixed(log.node), Some(Fixed::Numerical)));
        assert!(p.fixed(opaque.node).is_none());
        assert_eq!(calls.forward.load(Ordering::Relaxed), 0);
        assert_eq!(calls.work.load(Ordering::Relaxed), 0);
        let (mut w, mut value, mut g) = (p.workspace(), [17.], [19.]);
        assert!(matches!(
            p.value_and_vjp(&[1.], &[0.], &mut value, &mut g, &mut w),
            Err(E::Name)
        ));
        calls.fail.store(false, Ordering::Relaxed);
        assert!(matches!(
            p.value_and_vjp(&[1.], &[0.], &mut value, &mut g, &mut w),
            Err(E::Numerical)
        ));
        assert_eq!((value, g), ([17.], [19.]));
        let next = p.rebind("bad", &[1.]).unwrap();
        next.value_and_vjp(&[1.], &[1.], &mut value, &mut g, &mut w)
            .unwrap();
        assert_eq!((value, g), ([1.], [1.]));
    }
    fn bad_body(scan: bool) -> BoundProgram {
        let mut b = RegionBuilder::new();
        let _x = b.input("carry", Shape::scalar()).unwrap();
        if scan {
            let _row = b.input("row", Shape::scalar()).unwrap();
        }
        let data = b.data("bad", Shape::scalar()).unwrap();
        let out = b.unary(UnaryOp::Log, data).unwrap();
        b.compile(out).unwrap().bind(&[("bad", &[-1.])]).unwrap()
    }
    #[test]
    fn prepared_errors_remain_lazy_in_branches_and_zero_trip_loops() {
        let mut good = RegionBuilder::new();
        let x = good.input("x", Shape::scalar()).unwrap();
        let good = good.compile(x).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
        let out = b
            .cond(x, zero, Comparison::Greater, &[x], bad_body(false), good)
            .unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let (mut w, mut out, mut g) = (p.workspace(), [17.], [19.]);
        p.value_and_vjp(&[-2.], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        assert_eq!((out, g), ([-2.], [1.]));
        assert!(matches!(
            p.value_and_vjp(&[2.], &[0.], &mut out, &mut g, &mut w),
            Err(E::Numerical)
        ));
        assert_eq!((out, g), ([-2.], [1.]));
        p.value_and_vjp(&[-2.], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        for steps in [0, 1] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::scalar()).unwrap();
            let xs = b
                .constant(Shape::new(vec![steps]).unwrap(), vec![0.; steps])
                .unwrap();
            let out = b.scan(x, xs, &[], bad_body(true)).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            let result = p.value_and_vjp(&[2.], &[1.], &mut [0.], &mut [0.], &mut p.workspace());
            assert_eq!(result.is_ok(), steps == 0);
        }
    }
    #[test]
    fn empty_fixed_matrix_domains_do_not_iterate_huge_outer_extents() {
        let mut b = RegionBuilder::new();
        let left = b
            .data("left", Shape::new(vec![usize::MAX, 0]).unwrap())
            .unwrap();
        let right = b.data("right", Shape::new(vec![0, 0]).unwrap()).unwrap();
        let matrix = b.matmul(left, right).unwrap();
        let out = b.sum(matrix).unwrap();
        let p = b
            .compile(out)
            .unwrap()
            .bind(&[("left", &[]), ("right", &[])])
            .unwrap();
        assert!(matches!(p.fixed(matrix.node), Some(Fixed::Value(v)) if v.is_empty()));
        let mut value = [17.];
        p.evaluate(&[], &mut value, &mut p.workspace()).unwrap();
        assert_eq!(value, [0.]);
    }
    #[cfg(feature = "faer")]
    #[test]
    fn accelerated_fixed_data_calls_keep_provider_execution_at_the_original_boundary() {
        let mut b = RegionBuilder::new();
        let a = b.data("a", Shape::new(vec![1, 1]).unwrap()).unwrap();
        let product = b
            .matmul_with_request(a, a, MatrixRequest::standard(MatrixProvider::Faer))
            .unwrap();
        let out = b.sum(product).unwrap();
        let p = b.compile(out).unwrap().bind(&[("a", &[2.])]).unwrap();
        assert!(p.fixed(product.node).is_none());
        assert!(p.fixed(out.node).is_none());
        assert!(!p.plan.facts.binding_fixed(product.node));
        assert!(p.requires_explicit_source());
    }
}
