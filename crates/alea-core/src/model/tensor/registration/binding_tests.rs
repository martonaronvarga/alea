//! Production binding preparation, including the former scaffold's negatives.
use super::*;

fn rules(input: BlockType) -> BlockRules<(), ()> {
    BlockRules {
        inputs: vec![input],
        output: Shape::scalar(),
        workspace: |_| (),
        heap_bytes: 0,
        preparation: Some(BlockPreparation::new(vec![0], |_, args| {
            Ok(args.real(0)?[0])
        })),
        forward: |_, inputs, out, _| {
            out[0] = *inputs.prepared::<f64>()?;
            Ok(())
        },
        vjp: None,
    }
}
#[test]
fn binding_rules_reject_dynamic_dependencies_bad_arguments_and_shapes_before_mutation() {
    let mut b = RegionBuilder::new();
    let x = b.input("x", Shape::scalar()).unwrap();
    let data = b.data("data", Shape::scalar()).unwrap();
    let derived = b.unary(UnaryOp::Neg, data).unwrap();
    for (input, argument) in [(x, 0), (derived, 0), (data, 1)] {
        let before = b.nodes.len();
        let mut r = rules(BlockType::of(b.node(input).unwrap()));
        r.preparation = Some(BlockPreparation::new(
            vec![argument],
            |_, _| -> Result<(), E> { panic!("invalid preparation executed") },
        ));
        assert!(matches!(b.register_block(&[input], (), r), Err(E::Shape)));
        assert_eq!(b.nodes.len(), before);
    }
    let mut foreign = RegionBuilder::new();
    let foreign = foreign.constant(Shape::scalar(), vec![2.]).unwrap();
    let before = b.nodes.len();
    assert!(matches!(
        b.register_block(
            &[foreign],
            (),
            rules(BlockType {
                dtype: ElementType::Real,
                shape: Shape::scalar()
            })
        ),
        Err(E::ForeignHandle)
    ));
    for ty in [
        BlockType {
            dtype: ElementType::Bool,
            shape: Shape::scalar(),
        },
        BlockType {
            dtype: ElementType::Real,
            shape: Shape::new(vec![2]).unwrap(),
        },
    ] {
        assert!(matches!(
            b.register_block(&[data], (), rules(ty)),
            Err(E::Shape)
        ));
    }
    assert_eq!(b.nodes.len(), before);
    let constant = b.constant(Shape::scalar(), vec![2.]).unwrap();
    let mut r = rules(BlockType::of(b.node(constant).unwrap()));
    r.preparation = Some(BlockPreparation::new(vec![0], |_, args| {
        assert_eq!(args.real(0)?, &[2.]);
        assert!(matches!(args.real(1), Err(E::Shape)));
        Ok(())
    }));
    r.forward = |_, args, out, _| {
        args.prepared::<()>()?;
        assert!(matches!(args.prepared::<f64>(), Err(E::Shape)));
        out[0] = args.real(0)?[0];
        Ok(())
    };
    let out = b.register_block(&[constant], (), r).unwrap();
    let mut unused = rules(BlockType::of(b.node(constant).unwrap()));
    unused.preparation = Some(BlockPreparation::new(vec![0], |_, _| -> Result<(), E> {
        panic!("pruned recipe executed")
    }));
    let _unreachable = b.register_block(&[constant], (), unused).unwrap();
    let p = b.compile(out).unwrap().bind(&[("data", &[1.])]).unwrap();
    let (mut out, mut g) = ([17.], [19.]);
    p.value_and_vjp(&[1.], &[1.], &mut out, &mut g, &mut p.workspace())
        .unwrap();
    assert_eq!((out, g), ([2.], [0.]));
    assert!(matches!(p.rust_function(), Err(E::MissingPrimal { .. })));
}

#[test]
fn binding_preparation_reads_discrete_leaves_without_float_conversion() {
    struct Payload {
        flag: bool,
        count: u64,
        index: usize,
    }
    let mut b = RegionBuilder::new();
    let flag = b.boolean_data(Shape::scalar(), vec![true]).unwrap();
    let count = b
        .count_data(Shape::scalar(), vec![9_007_199_254_740_993])
        .unwrap();
    let index = b
        .typed_data(Shape::scalar(), Data::Index(vec![usize::MAX].into()))
        .unwrap();
    let inputs = [flag, count, index];
    let r = BlockRules {
        inputs: inputs
            .iter()
            .map(|&i| BlockType::of(b.node(i).unwrap()))
            .collect(),
        output: Shape::scalar(),
        workspace: |_| (),
        heap_bytes: 0,
        preparation: Some(BlockPreparation::new(vec![2, 0, 1], |_: &(), args| {
            assert!(matches!(args.real(0), Err(E::Shape)));
            assert!(matches!(args.counts(1), Err(E::Shape)));
            Ok(Payload {
                flag: args.booleans(1)?[0],
                count: args.counts(2)?[0],
                index: args.indices(0)?[0],
            })
        })),
        forward: |_, args, out, _| {
            let p = args.prepared::<Payload>()?;
            assert_eq!(p.count, 9_007_199_254_740_993);
            assert_eq!(p.index, usize::MAX);
            out[0] = f64::from(p.flag);
            Ok(())
        },
        vjp: None,
    };
    let out = b.register_block(&inputs, (), r).unwrap();
    let p = b.compile(out).unwrap().bind(&[]).unwrap();
    let mut value = [19.];
    p.value_and_vjp(&[], &[1.], &mut value, &mut [], &mut p.workspace())
        .unwrap();
    assert_eq!(value, [1.]);
}

#[test]
fn constant_only_preparations_still_require_the_exact_admitted_snapshot() {
    let mut b = RegionBuilder::new();
    let x = b.input("x", Shape::scalar()).unwrap();
    let constant = b.constant(Shape::scalar(), vec![2.]).unwrap();
    let out = b
        .register_block(
            &[x, constant],
            (),
            BlockRules {
                inputs: vec![
                    BlockType::of(b.node(x).unwrap()),
                    BlockType::of(b.node(constant).unwrap()),
                ],
                output: Shape::scalar(),
                workspace: |_| (),
                heap_bytes: 0,
                preparation: Some(BlockPreparation::new(vec![1], |_: &(), args| {
                    Ok(args.real(0)?[0])
                })),
                forward: |_, args, out, _| {
                    out[0] = args.real(0)?[0] * args.prepared::<f64>()?;
                    Ok(())
                },
                vjp: Some(|_, args, _, seed, out, _| {
                    out[0] = seed[0] * args.prepared::<f64>()?;
                    out[1] = 0.;
                    Ok(())
                }),
            },
        )
        .unwrap();
    let definition = b.compile(out).unwrap();
    let old = definition.bind(&[]).unwrap();
    let new = definition.bind(&[]).unwrap();
    let mut w = old.workspace();
    old.evaluate(&[3.], &mut [0.], &mut w).unwrap();
    let mut scores = [19.];
    assert!(matches!(
        new.reverse_point(&[1.], &mut scores, &mut w),
        Err(E::Shape)
    ));
    assert_eq!(scores, [19.]);
    old.value_and_vjp(&[3.], &[1.], &mut [0.], &mut scores, &mut w)
        .unwrap();
    assert_eq!(scores, [2.]);
}

#[test]
fn binding_never_evaluates_guarded_forward_callbacks_or_reprepares_child_snapshots() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Calls {
        preparation: AtomicUsize,
        forward: AtomicUsize,
    }
    let calls = Arc::new(Calls {
        preparation: AtomicUsize::new(0),
        forward: AtomicUsize::new(0),
    });
    let mut bad = RegionBuilder::new();
    let x = bad.input("x", Shape::scalar()).unwrap();
    let out = bad
        .register_block(
            &[x],
            Arc::clone(&calls),
            BlockRules {
                inputs: vec![BlockType::of(bad.node(x).unwrap())],
                output: Shape::scalar(),
                workspace: |_| (),
                heap_bytes: 0,
                preparation: Some(BlockPreparation::new(vec![], |c: &Arc<Calls>, _| {
                    c.preparation.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })),
                forward: |c, _, _, _| {
                    c.forward.fetch_add(1, Ordering::Relaxed);
                    Err(E::Name)
                },
                vjp: None,
            },
        )
        .unwrap();
    let bad = bad.compile(out).unwrap().bind(&[]).unwrap();
    let mut good = RegionBuilder::new();
    let x = good.input("x", Shape::scalar()).unwrap();
    let good = good.compile(x).unwrap().bind(&[]).unwrap();
    let mut b = RegionBuilder::new();
    let x = b.input("x", Shape::scalar()).unwrap();
    let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
    let out = b
        .cond(x, zero, Comparison::Greater, &[x], bad, good)
        .unwrap();
    let p = b.compile(out).unwrap().bind(&[]).unwrap();
    assert_eq!(calls.preparation.load(Ordering::Relaxed), 1);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 0);
    let (mut w, mut value, mut scores) = (p.workspace(), [17.], [19.]);
    p.value_and_vjp(&[-2.], &[1.], &mut value, &mut scores, &mut w)
        .unwrap();
    assert_eq!((value, scores), ([-2.], [1.]));
    assert_eq!(calls.forward.load(Ordering::Relaxed), 0);
    assert!(matches!(
        p.value_and_vjp(&[2.], &[0.], &mut value, &mut scores, &mut w),
        Err(E::Name)
    ));
    assert_eq!((value, scores), ([-2.], [1.]));
    assert_eq!(calls.preparation.load(Ordering::Relaxed), 1);
    assert_eq!(calls.forward.load(Ordering::Relaxed), 1);
}

#[test]
fn later_preparation_failure_does_not_publish_partial_payloads_or_invalidate_old_point() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let prepares = Arc::new(AtomicUsize::new(0));
    let mut b = RegionBuilder::new();
    let x = b.input("x", Shape::scalar()).unwrap();
    let data = b.data("data", Shape::scalar()).unwrap();
    let ty = BlockType {
        dtype: ElementType::Real,
        shape: Shape::scalar(),
    };
    let mut out = x;
    for first in [true, false] {
        let prepare: fn(&Arc<AtomicUsize>, BindingInputs<'_>) -> Result<f64, E> = if first {
            |n, args| {
                n.fetch_add(1, Ordering::Relaxed);
                Ok(args.real(0)?[0] * 2.)
            }
        } else {
            |n, args| {
                n.fetch_add(1, Ordering::Relaxed);
                let value = args.real(0)?[0];
                if value < 0. {
                    return Err(E::Name);
                }
                Ok(value + 1.)
            }
        };
        out = b
            .register_block(
                &[out, data],
                Arc::clone(&prepares),
                BlockRules {
                    inputs: vec![ty.clone(), ty.clone()],
                    output: Shape::scalar(),
                    preparation: Some(BlockPreparation::new(vec![1], prepare)),
                    workspace: |_| (),
                    heap_bytes: 0,
                    forward: |_, args, out, _| {
                        out[0] = args.real(0)?[0] * args.prepared::<f64>()?;
                        Ok(())
                    },
                    vjp: Some(|_, args, _, seed, g, _| {
                        g[0] = seed[0] * args.prepared::<f64>()?;
                        g[1] = 0.;
                        Ok(())
                    }),
                },
            )
            .unwrap();
    }
    let old = b.compile(out).unwrap().bind(&[("data", &[2.])]).unwrap();
    let mut w = old.workspace();
    let (mut value, mut g) = ([0.], [0.]);
    old.evaluate(&[3.], &mut value, &mut w).unwrap();
    assert_eq!(value, [36.]);
    assert!(matches!(old.rebind("data", &[-1.]), Err(E::Name)));
    assert_eq!(prepares.load(Ordering::Relaxed), 4);
    old.reverse_point(&[1.], &mut g, &mut w).unwrap();
    assert_eq!(g, [12.]);
    let new = old.rebind("data", &[3.]).unwrap();
    new.value_and_vjp(&[3.], &[1.], &mut value, &mut g, &mut w)
        .unwrap();
    assert_eq!((value, g), ([72.], [24.]));
    old.value_and_vjp(&[3.], &[1.], &mut value, &mut g, &mut w)
        .unwrap();
    assert_eq!((value, g), ([36.], [12.]));
    assert_eq!(prepares.load(Ordering::Relaxed), 6);
}
