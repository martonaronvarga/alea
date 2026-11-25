//! Production ownership checks replacing the private lifecycle target adapter.
use alea_core::{
    model::assembly::*,
    target::{EvaluationWorkspace, PointState},
};
use alea_math::buffer::OwnedBuffer;
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Default)]
struct Controls {
    workspaces: AtomicUsize,
    evaluations: AtomicUsize,
    failure: AtomicUsize,
    seen: AtomicUsize,
}
struct Workspace {
    id: usize,
    last: Cell<f64>,
}
#[derive(Clone)]
struct Rule {
    weight: f64,
    width: usize,
    controls: Arc<Controls>,
}
impl OperationDescriptor for Rule {
    // Send but deliberately NOT Sync or Clone. Only contexts own this state.
    type Workspace = Workspace;
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::<Self>::new(
            PrimitiveSignature {
                name: "lifecycle quadratic",
                arguments: vec![PrimitiveArgument {
                    name: "x",
                    shape: if self.width == 1 {
                        ArgumentShape::Scalar
                    } else {
                        ArgumentShape::Vector(self.width)
                    },
                }],
                probability: None,
            },
            |rule| Workspace {
                id: rule.controls.workspaces.fetch_add(1, Ordering::Relaxed),
                last: Cell::new(f64::NAN),
            },
            |rule, x, g, work| {
                rule.controls.evaluations.fetch_add(1, Ordering::Relaxed);
                rule.controls
                    .seen
                    .fetch_or(1_usize << work.id, Ordering::Relaxed);
                work.last.set(x[0]);
                match rule.controls.failure.load(Ordering::Relaxed) {
                    1 => return Err(AssemblyError::Name),
                    2 => {
                        g[0] = f64::NAN;
                        return Ok(0.);
                    }
                    3 => panic!("deliberate evaluator unwind"),
                    4 => {
                        return Err(AssemblyError::Factor(Box::new(std::io::Error::other(
                            "typed callback",
                        ))));
                    }
                    _ => {}
                }
                g[0] = 2. * rule.weight * x[0];
                Ok(rule.weight * x[0] * x[0])
            },
        );
        rules.source = Some(SourceRule {
            body: SourceBody::Generated(|rule| format!("{:?}_f64*x[0]*x[0]", rule.weight)),
            gradient: None,
            workspace: 0,
        });
        rules
    }
}
fn definition() -> (Arc<ModelDefinition>, Value, Arc<Controls>) {
    let controls = Arc::new(Controls::default());
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let f = b
        .primitive(
            &[x],
            Rule {
                weight: 2.,
                width: 1,
                controls: Arc::clone(&controls),
            },
        )
        .unwrap();
    b.factor(f).unwrap();
    (b.finish().unwrap(), f, controls)
}

#[test]
fn snapshots_share_one_definition_and_recompile_without_stale_native_or_frozen_source_data() {
    let (definition, root, controls) = definition();
    let original = definition.bind();
    let same = definition.bind();
    assert_ne!(original.identity(), same.identity());
    assert_eq!(original.clone().identity(), original.identity());
    let changed = original
        .rebind::<Rule>(root, |r| {
            Ok(Rule {
                weight: 3.,
                ..r.clone()
            })
        })
        .unwrap();
    let identical = original.rebind::<Rule>(root, |r| Ok(r.clone())).unwrap();
    assert_ne!(changed.identity(), original.identity());
    assert_ne!(identical.identity(), original.identity());
    assert!(Arc::ptr_eq(original.definition(), changed.definition()));
    let (old, new, other) = (
        original.compile().unwrap(),
        changed.compile().unwrap(),
        original.compile().unwrap(),
    );
    assert_ne!(old.identity(), other.identity());
    assert_eq!(controls.workspaces.load(Ordering::Relaxed), 0);
    assert!(old.rust_primal().unwrap().contains("2.0_f64*x[0]*x[0]"));
    assert!(new.rust_primal().unwrap().contains("3.0_f64*x[0]*x[0]"));
    let mut context = old.context();
    let mut g = [19.];
    assert_eq!(old.evaluate(&mut context, &[3.], &mut g).unwrap(), 18.);
    assert_eq!(g, [12.]);
    for executable in [&new, &other] {
        assert!(matches!(
            executable.evaluate(&mut context, &[3.], &mut g),
            Err(AssemblyError::ContextMismatch)
        ));
        assert_eq!(g, [12.]);
    }
    let mut fresh = new.context();
    assert_eq!(new.evaluate(&mut fresh, &[3.], &mut g).unwrap(), 27.);
    assert_eq!(g, [18.]);
    assert_eq!(old.evaluate(&mut context, &[3.], &mut g).unwrap(), 18.);
    assert_eq!(g, [12.]);
    let (foreign, _, _) = self::definition();
    assert!(matches!(
        foreign.compile(&original),
        Err(AssemblyError::ForeignHandle)
    ));
}

#[test]
fn independent_non_sync_workspaces_share_only_an_immutable_executable_and_publish_once() {
    let (definition, _, controls) = definition();
    let executable = Arc::new(definition.bind().compile().unwrap());
    fn requires_sync(_: &impl Sync) {}
    requires_sync(&*executable);
    let (mut a, mut b) = (executable.context(), executable.context());
    let ra = executable.prepare(&mut a, &[2.]).unwrap();
    let rb = executable.prepare(&mut b, &[3.]).unwrap();
    let mut g = [19.];
    assert!(matches!(
        executable.publish(&mut b, &ra, &mut g),
        Err(AssemblyError::StaleEvaluation)
    ));
    assert_eq!(g, [19.]);
    assert_eq!(executable.publish(&mut b, &rb, &mut g).unwrap(), 18.);
    assert_eq!(g, [12.]);
    assert_eq!(executable.publish(&mut a, &ra, &mut g).unwrap(), 8.);
    assert_eq!(g, [8.]);
    assert_eq!(controls.evaluations.load(Ordering::Relaxed), 2); // Publication never re-evaluates.
    assert!(executable.publish(&mut a, &ra, &mut g).is_err());
    let old = executable.prepare(&mut a, &[2.]).unwrap();
    let current = executable.prepare(&mut a, &[2.]).unwrap();
    assert!(executable.publish(&mut a, &old, &mut g).is_err());
    assert_eq!(executable.publish(&mut a, &current, &mut g).unwrap(), 8.);
    std::thread::scope(|scope| {
        for x in [2., 3.] {
            let executable = Arc::clone(&executable);
            scope.spawn(move || {
                let mut context = executable.context();
                let mut g = [19.];
                use alea_core::target::LogDensityGradient;
                let target = executable.target().unwrap();
                assert_eq!(target.logp_grad(&[x], &mut g).unwrap(), 2. * x * x);
                assert_eq!(g, [4. * x]);
                for _ in 0..16 {
                    assert_eq!(
                        executable.evaluate(&mut context, &[x], &mut g).unwrap(),
                        2. * x * x
                    );
                    assert_eq!(g, [4. * x]);
                }
            });
        }
    });
    assert_eq!(controls.workspaces.load(Ordering::Relaxed), 6);
    assert_eq!(controls.seen.load(Ordering::Relaxed), 0b111111);
}

#[test]
fn failed_rebind_and_unwind_leave_old_snapshot_and_context_usable_without_allocating_new_workspaces()
 {
    let (definition, root, controls) = definition();
    let binding = definition.bind();
    let executable = binding.compile().unwrap();
    let mut context = executable.context();
    for mode in 0..4 {
        let result = catch_unwind(AssertUnwindSafe(|| {
            binding.rebind::<Rule>(root, |r| match mode {
                0 => Err(AssemblyError::Name),
                1 => Err(AssemblyError::Factor(Box::new(std::io::Error::other(
                    "binding failure",
                )))),
                2 => Ok(Rule {
                    width: 2,
                    ..r.clone()
                }),
                _ => panic!("deliberate binding unwind"),
            })
        }));
        match mode {
            0 => assert!(matches!(result, Ok(Err(AssemblyError::Name)))),
            1 => match result.unwrap() {
                Err(AssemblyError::Factor(error)) => assert!(error.is::<std::io::Error>()),
                _ => panic!("lost typed cause"),
            },
            2 => assert!(matches!(result, Ok(Err(AssemblyError::Shape)))),
            _ => assert!(result.is_err()),
        }
        let mut g = [19.];
        assert_eq!(
            executable.evaluate(&mut context, &[2.], &mut g).unwrap(),
            8.
        );
        assert_eq!(g, [8.]);
        assert_eq!(controls.workspaces.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn failed_evaluations_shapes_and_unwind_invalidate_preparations_and_recover_transactionally() {
    let (definition, _, controls) = definition();
    let executable = definition.bind().compile().unwrap();
    let mut context = executable.context();
    for mode in 1..=4 {
        let old = executable.prepare(&mut context, &[2.]).unwrap();
        controls.failure.store(mode, Ordering::Relaxed);
        let result = catch_unwind(AssertUnwindSafe(|| executable.prepare(&mut context, &[3.])));
        match mode {
            1 => assert!(matches!(result, Ok(Err(AssemblyError::Name)))),
            2 => assert!(matches!(result, Ok(Err(AssemblyError::Numerical)))),
            3 => assert!(result.is_err()),
            _ => match result.unwrap() {
                Err(AssemblyError::Factor(error)) => assert!(error.is::<std::io::Error>()),
                _ => panic!("lost typed cause"),
            },
        }
        let mut g = [19.];
        assert!(executable.publish(&mut context, &old, &mut g).is_err());
        assert_eq!(g, [19.]);
        controls.failure.store(0, Ordering::Relaxed);
        assert_eq!(
            executable.evaluate(&mut context, &[2.], &mut g).unwrap(),
            8.
        );
        assert_eq!(g, [8.]);
    }
    for q in [&[][..], &[f64::NAN][..]] {
        let old = executable.prepare(&mut context, &[2.]).unwrap();
        let before = controls.evaluations.load(Ordering::Relaxed);
        assert!(executable.prepare(&mut context, q).is_err());
        assert_eq!(controls.evaluations.load(Ordering::Relaxed), before);
        assert!(executable.publish(&mut context, &old, &mut [19.]).is_err());
    }
    let old = executable.prepare(&mut context, &[2.]).unwrap();
    let before = controls.evaluations.load(Ordering::Relaxed);
    assert!(matches!(
        executable.evaluate(&mut context, &[2.], &mut []),
        Err(AssemblyError::Shape)
    ));
    assert_eq!(controls.evaluations.load(Ordering::Relaxed), before);
    assert!(executable.publish(&mut context, &old, &mut [19.]).is_err());
    let old = executable.prepare(&mut context, &[2.]).unwrap();
    assert!(matches!(
        executable.publish(&mut context, &old, &mut []),
        Err(AssemblyError::Shape)
    ));
    assert!(executable.publish(&mut context, &old, &mut [19.]).is_err());
    assert_eq!(
        executable
            .evaluate(&mut context, &[3.], &mut [19.])
            .unwrap(),
        18.
    );
}

#[test]
fn owned_target_coherent_points_and_explicit_contexts_allocate_nothing_per_evaluation() {
    let (definition, _, controls) = definition();
    let executable = definition.bind().compile().unwrap();
    let mut context = executable.context();
    let mut g = [19.];
    assert_eq!(
        allocation_counter::measure(|| {
            for x in [2., 3., -2., 0.] {
                executable.evaluate(&mut context, &[x], &mut g).unwrap();
            }
        })
        .count_total,
        0
    );
    let target = executable.into_target().unwrap();
    drop(definition);
    let mut point = PointState::new(&target, OwnedBuffer::from_fn(1, |_| 2.)).unwrap();
    let mut work = EvaluationWorkspace::new(1);
    assert_eq!((point.log_density(), point.gradient()), (8., &[8.][..]));
    assert_eq!(
        allocation_counter::measure(|| {
            for _ in 0..16 {
                point.try_update(&[3.], &mut work).unwrap();
            }
        })
        .count_total,
        0
    );
    let before = (
        point.position().to_vec(),
        point.gradient().to_vec(),
        point.log_density(),
    );
    for mode in 1..=4 {
        controls.failure.store(mode, Ordering::Relaxed);
        let result = catch_unwind(AssertUnwindSafe(|| point.try_update(&[2.], &mut work)));
        assert!(match result {
            Ok(result) => result.is_err(),
            Err(_) => mode == 3,
        });
        assert_eq!(
            (point.position(), point.gradient(), point.log_density()),
            (&before.0[..], &before.1[..], before.2)
        );
        controls.failure.store(0, Ordering::Relaxed);
        point.try_update(&[3.], &mut work).unwrap();
    }
}

#[test]
fn native_block_reverse_failures_invalidate_tokens_even_for_zero_final_weight() {
    use alea_core::model::tensor::{BlockRules, BlockType, ElementType, Shape};
    for zero in [false, true] {
        let controls = Arc::new(Controls::default());
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let value = b
            .register_block(
                &[x],
                Arc::clone(&controls),
                BlockRules {
                    preparation: None,
                    inputs: vec![BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar(),
                    }],
                    output: Shape::scalar(),
                    workspace: |_: &Arc<Controls>| Cell::new(0.),
                    heap_bytes: 0,
                    forward: |_, args, out, _| {
                        out[0] = 2. * args.real(0)?[0].powi(2);
                        Ok(())
                    },
                    vjp: Some(|controls, args, _, seed, g, work| {
                        work.set(args.real(0)?[0]);
                        match controls.failure.load(Ordering::Relaxed) {
                            1 => return Err(AssemblyError::Name),
                            2 => {
                                g[0] = f64::NAN;
                                return Ok(());
                            }
                            3 => panic!("deliberate block reverse unwind"),
                            _ => {}
                        }
                        g[0] = 4. * args.real(0)?[0] * seed[0];
                        Ok(())
                    }),
                },
            )
            .unwrap();
        let value = if zero {
            b.multiply(value, 0.).unwrap()
        } else {
            value
        };
        b.factor(value).unwrap();
        let definition = b.finish().unwrap();
        let executable = definition.bind().compile().unwrap();
        let mut context = executable.context();
        let scale = if zero { 0. } else { 1. };
        for mode in 1..=3 {
            let old = executable.prepare(&mut context, &[2.]).unwrap();
            controls.failure.store(mode, Ordering::Relaxed);
            let result = catch_unwind(AssertUnwindSafe(|| executable.prepare(&mut context, &[3.])));
            match mode {
                1 => assert!(matches!(result, Ok(Err(AssemblyError::Name)))),
                2 => assert!(matches!(result, Ok(Err(AssemblyError::Numerical)))),
                _ => assert!(result.is_err()),
            }
            let mut g = [19.];
            assert!(executable.publish(&mut context, &old, &mut g).is_err());
            assert_eq!(g, [19.]);
            controls.failure.store(0, Ordering::Relaxed);
            assert_eq!(
                executable.evaluate(&mut context, &[3.], &mut g).unwrap(),
                18. * scale
            );
            assert_eq!(g, [12. * scale]);
        }
    }
}

#[test]
fn production_scalar_target_keeps_empty_coordinates_and_rejects_vector_one_factors() {
    use alea_core::{model::tensor::Shape, target::LogDensityGradient, transform::Transform};
    let mut b = ModelBuilder::new();
    let x = b
        .vector::<Simplex>(
            "x",
            "simplex",
            &["only"],
            "x",
            Transform::simplex(1).unwrap(),
        )
        .unwrap();
    b.factor(x.expressions()[0]).unwrap();
    let target = b.compile().unwrap().into_target().unwrap();
    assert_eq!(target.logp_grad(&[], &mut []).unwrap(), 1.);
    let mut b = ModelBuilder::new();
    let one = b
        .constant_block(Shape::new(vec![1]).unwrap(), vec![1.])
        .unwrap();
    assert!(matches!(b.factor(one), Err(AssemblyError::Shape)));
}

#[test]
fn shared_owned_adapters_preserve_density_objective_and_exactly_one_chart_boundary() {
    use alea_core::target::LogDensityGradient;
    let mut b = ModelBuilder::new();
    let x = b.positive("x").unwrap().expr();
    let square = b.multiply(x, x).unwrap();
    let factor = b.multiply(-0.5, square).unwrap();
    b.factor(factor).unwrap();
    let executable = Arc::new(b.finish().unwrap().bind().compile().unwrap());
    let density = executable.target().unwrap();
    let objective = executable.native_model().into_objective().unwrap();
    for q in [-0.5_f64, 0., 0.7] {
        let mut g = [19.];
        let e = q.exp();
        assert!((density.logp_grad(&[q], &mut g).unwrap() - (-0.5 * e * e + q)).abs() < 1e-12);
        assert!((g[0] - (-e * e + 1.)).abs() < 1e-12);
        assert!((objective.logp_grad(&[q], &mut g).unwrap() + 0.5 * e * e).abs() < 1e-12);
        assert!((g[0] + e * e).abs() < 1e-12);
    }
}

mod block_binding {
    use super::*;
    use alea_core::model::tensor::{
        BlockPreparation, BlockRules, BlockType, ElementType, Shape, TensorProgram,
    };
    #[derive(Default)]
    struct Counters {
        preparations: AtomicUsize,
        workspaces: AtomicUsize,
        prepare_mode: AtomicUsize,
        eval_mode: AtomicUsize,
        payload: AtomicUsize,
        workspace: AtomicUsize,
    }
    struct Packed {
        weight: f64,
    }
    // Send, deliberately not Sync/Clone: every context owns this scratch.
    struct Scratch {
        id: usize,
        calls: Cell<usize>,
    }
    #[derive(Debug)]
    struct PrepareFailure;
    impl std::fmt::Display for PrepareFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("preparation failed")
        }
    }
    impl std::error::Error for PrepareFailure {}
    fn definition(c: &Arc<Counters>) -> TensorProgram {
        let mut b = ModelBuilder::new().region();
        let x = b.input("x", Shape::scalar()).unwrap();
        let data = b.data("data", Shape::new(vec![2]).unwrap()).unwrap();
        let out = b
            .register_block(
                &[x, data],
                Arc::clone(c),
                BlockRules {
                    inputs: vec![
                        BlockType {
                            dtype: ElementType::Real,
                            shape: Shape::scalar(),
                        },
                        BlockType {
                            dtype: ElementType::Real,
                            shape: Shape::new(vec![2]).unwrap(),
                        },
                    ],
                    output: Shape::scalar(),
                    heap_bytes: 0,
                    preparation: Some(BlockPreparation::new(vec![1], |c: &Arc<Counters>, args| {
                        c.preparations.fetch_add(1, Ordering::Relaxed);
                        match c.prepare_mode.load(Ordering::Relaxed) {
                            1 => return Err(AssemblyError::Factor(Box::new(PrepareFailure))),
                            2 => panic!("deliberate preparation unwind"),
                            3 => {
                                args.counts(0)?;
                            }
                            _ => {}
                        }
                        let data = args.real(0)?;
                        let weight = data[0] + data[1];
                        if !weight.is_finite() {
                            return Err(AssemblyError::Numerical);
                        }
                        Ok(Packed { weight })
                    })),
                    workspace: |c| Scratch {
                        id: c.workspaces.fetch_add(1, Ordering::Relaxed),
                        calls: Cell::new(0),
                    },
                    forward: |c, args, out, scratch| {
                        if c.eval_mode.load(Ordering::Relaxed) == 1 {
                            args.prepared::<String>()?;
                        }
                        let p = args.prepared::<Packed>()?;
                        c.payload
                            .store(p as *const Packed as usize, Ordering::Relaxed);
                        c.workspace.store(scratch.id, Ordering::Relaxed);
                        scratch.calls.set(scratch.calls.get() + 1);
                        out[0] = p.weight * args.real(0)?[0].powi(2);
                        Ok(())
                    },
                    vjp: Some(|_, args, _, seed, scores, _| {
                        scores.fill(0.);
                        scores[0] =
                            2. * args.prepared::<Packed>()?.weight * args.real(0)?[0] * seed[0];
                        Ok(())
                    }),
                },
            )
            .unwrap();
        b.compile(out).unwrap()
    }
    #[test]
    fn prepared_blocks_share_snapshots_but_keep_execution_scratch_private() {
        let c = Arc::new(Counters::default());
        let definition = definition(&c);
        assert_eq!(c.preparations.load(Ordering::Relaxed), 0);
        let p = definition.bind(&[("data", &[3., 4.])]).unwrap();
        assert_eq!(c.preparations.load(Ordering::Relaxed), 1);
        let cloned = p.clone();
        let (mut a, mut b) = (p.workspace(), cloned.workspace());
        assert_eq!(c.workspaces.load(Ordering::Relaxed), 2);
        let (mut out, mut g) = ([0.], [0.]);
        p.value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut a)
            .unwrap();
        assert_eq!((out, g), ([28.], [28.]));
        let old_ptr = c.payload.load(Ordering::Relaxed);
        let first = c.workspace.load(Ordering::Relaxed);
        cloned
            .value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut b)
            .unwrap();
        assert_eq!(old_ptr, c.payload.load(Ordering::Relaxed));
        assert_ne!(first, c.workspace.load(Ordering::Relaxed));
        assert_eq!(
            allocation_counter::measure(|| {
                for _ in 0..16 {
                    p.value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut a)
                        .unwrap();
                }
            })
            .count_total,
            0
        );
        assert_eq!(c.preparations.load(Ordering::Relaxed), 1);
        let new = p.rebind("data", &[3., 5.]).unwrap();
        assert_eq!(c.preparations.load(Ordering::Relaxed), 2);
        new.value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut a)
            .unwrap();
        assert_eq!((out, g), ([32.], [32.]));
        assert_ne!(old_ptr, c.payload.load(Ordering::Relaxed));
        p.value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut b)
            .unwrap();
        assert_eq!((out, g), ([28.], [28.]));
        assert_eq!(old_ptr, c.payload.load(Ordering::Relaxed));
        assert_eq!(c.workspaces.load(Ordering::Relaxed), 2);
        assert!(matches!(
            p.rust_function(),
            Err(AssemblyError::MissingPrimal { .. })
        ));
    }
    #[test]
    fn preparation_errors_unwinds_and_payload_type_errors_preserve_old_bindings() {
        let c = Arc::new(Counters::default());
        let p = definition(&c).bind(&[("data", &[3., 4.])]).unwrap();
        let (mut w, mut out, mut g) = (p.workspace(), [17.], [19.]);
        for mode in 1..=3 {
            c.prepare_mode.store(mode, Ordering::Relaxed);
            let result = catch_unwind(AssertUnwindSafe(|| p.rebind("data", &[5., 6.])));
            match (mode, result) {
                (1, Ok(Err(AssemblyError::Factor(e)))) => {
                    assert!(e.downcast_ref::<PrepareFailure>().is_some())
                }
                (2, Err(_)) | (3, Ok(Err(AssemblyError::Shape))) => {}
                _ => panic!("wrong preparation error boundary"),
            }
            p.value_and_vjp(&[2.], &[1.], &mut out, &mut g, &mut w)
                .unwrap();
            assert_eq!((out, g), ([28.], [28.]));
        }
        c.prepare_mode.store(0, Ordering::Relaxed);
        let before = c.preparations.load(Ordering::Relaxed);
        assert!(matches!(p.rebind("data", &[1.]), Err(AssemblyError::Shape)));
        assert!(matches!(
            p.rebind("data", &[1., f64::NAN]),
            Err(AssemblyError::Numerical)
        ));
        assert!(matches!(
            p.rebind("missing", &[1., 2.]),
            Err(AssemblyError::Name)
        ));
        assert_eq!(c.preparations.load(Ordering::Relaxed), before);
        c.eval_mode.store(1, Ordering::Relaxed);
        assert!(matches!(
            p.value_and_vjp(&[2.], &[0.], &mut out, &mut g, &mut w),
            Err(AssemblyError::Shape)
        ));
        assert_eq!((out, g), ([28.], [28.]));
        c.eval_mode.store(0, Ordering::Relaxed);
        p.value_and_vjp(&[3.], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        assert_eq!((out, g), ([63.], [42.]));
        assert_eq!(c.preparations.load(Ordering::Relaxed), before);
    }
    #[test]
    fn model_rebinding_prepares_transactionally_and_old_contexts_remain_usable() {
        let c = Arc::new(Counters::default());
        let region = definition(&c).bind(&[("data", &[3., 4.])]).unwrap();
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let value = b.call_region(&[x], region).unwrap();
        b.factor(value).unwrap();
        let definition = b.finish().unwrap();
        let binding = definition.bind();
        let old = binding.compile().unwrap();
        let (mut context, mut g) = (old.context(), [0.]);
        assert_eq!(old.evaluate(&mut context, &[2.], &mut g).unwrap(), 28.);
        c.prepare_mode.store(1, Ordering::Relaxed);
        assert!(
            matches!(binding.rebind_region(value, "data", &[5., 6.]), Err(AssemblyError::Factor(e)) if e.downcast_ref::<PrepareFailure>().is_some())
        );
        assert_eq!(old.evaluate(&mut context, &[2.], &mut g).unwrap(), 28.);
        c.prepare_mode.store(0, Ordering::Relaxed);
        let fresh = binding
            .rebind_region(value, "data", &[5., 6.])
            .unwrap()
            .compile()
            .unwrap();
        assert!(matches!(
            fresh.evaluate(&mut context, &[2.], &mut g),
            Err(AssemblyError::ContextMismatch)
        ));
        assert_eq!(g, [28.]);
        assert_eq!(
            fresh.evaluate(&mut fresh.context(), &[2.], &mut g).unwrap(),
            44.
        );
        assert_eq!(g, [44.]);
        assert_eq!(old.evaluate(&mut context, &[2.], &mut g).unwrap(), 28.);
    }
}

#[test]
fn typed_block_preparation_uses_exact_edges_and_resets_each_evaluation() {
    use alea_core::model::tensor::{BlockRules, BlockType, ElementType, Shape};

    #[derive(Clone, Copy)]
    struct Prepared(f64);
    static PREPARATIONS: AtomicUsize = AtomicUsize::new(0);
    fn prepare(x: f64) -> Prepared {
        PREPARATIONS.fetch_add(1, Ordering::Relaxed);
        Prepared(x * x)
    }

    let failure = Arc::new(AtomicUsize::new(0));
    let mut builder = ModelBuilder::new();
    let q = builder.real("q").unwrap().expr();
    let mut region = builder.region();
    let data = region.data("scale", Shape::scalar()).unwrap();
    let data = region
        .compile(data)
        .unwrap()
        .bind(&[("scale", &[2.])])
        .unwrap();
    let scale = builder.call_region(&[], data).unwrap();
    for (i, input) in [q, scale, scale].into_iter().enumerate() {
        let rules = || BlockRules::<(Arc<AtomicUsize>, usize), ()> {
            inputs: vec![BlockType {
                dtype: ElementType::Real,
                shape: Shape::scalar(),
            }],
            output: Shape::scalar(),
            workspace: |_| (),
            heap_bytes: 0,
            preparation: None,
            forward: |(failure, i), input, out, _| {
                out[0] = input.evaluation_prepared::<Prepared>()?.0;
                if *i == 2 {
                    match failure.load(Ordering::Relaxed) {
                        1 => return Err(AssemblyError::Name),
                        2 => panic!("deliberate block unwind"),
                        _ => {}
                    }
                }
                Ok(())
            },
            vjp: Some(|_, inputs, _, seed, scores, _| {
                scores[0] = 2. * inputs.real(0)?[0] * seed[0];
                Ok(())
            }),
        };
        if i == 0 {
            let erased_recipe: fn(f64) -> Prepared = prepare;
            assert!(matches!(
                builder.register_block_with_preparation(
                    &[input],
                    (Arc::clone(&failure), i),
                    rules(),
                    0,
                    erased_recipe,
                ),
                Err(AssemblyError::Shape)
            ));
        }
        let value = builder
            .register_block_with_preparation(
                &[input],
                (Arc::clone(&failure), i),
                rules(),
                0,
                prepare,
            )
            .unwrap();
        builder.factor(value).unwrap();
    }
    let binding = builder.finish().unwrap().bind();
    let executable = binding.compile().unwrap();
    let mut first = executable.context();
    let mut second = executable.context();
    let mut g = [19.];
    for (calls, use_second, point, expected) in
        [(2, false, 2., 12.), (4, false, 2., 12.), (6, true, 3., 17.)]
    {
        let context = if use_second { &mut second } else { &mut first };
        assert_eq!(
            executable.evaluate(context, &[point], &mut g).unwrap(),
            expected
        );
        assert_eq!(g, [2. * point]);
        assert_eq!(PREPARATIONS.load(Ordering::Relaxed), calls);
    }
    failure.store(1, Ordering::Relaxed);
    g = [19.];
    assert!(matches!(
        executable.evaluate(&mut first, &[2.], &mut g),
        Err(AssemblyError::Name)
    ));
    assert_eq!(g, [19.]);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 8);
    failure.store(2, Ordering::Relaxed);
    assert!(
        catch_unwind(AssertUnwindSafe(|| executable.evaluate(
            &mut first,
            &[2.],
            &mut g
        )))
        .is_err()
    );
    assert_eq!(g, [19.]);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 10);
    failure.store(0, Ordering::Relaxed);
    assert_eq!(executable.evaluate(&mut first, &[2.], &mut g).unwrap(), 12.);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 12);

    let changed = binding
        .rebind_region(scale, "scale", &[5.])
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(
        changed
            .evaluate(&mut changed.context(), &[2.], &mut g)
            .unwrap(),
        54.
    );
    assert_eq!(executable.evaluate(&mut first, &[2.], &mut g).unwrap(), 12.);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 16);
    let measured = allocation_counter::measure(|| {
        assert_eq!(executable.evaluate(&mut first, &[2.], &mut g).unwrap(), 12.);
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 18);
}
