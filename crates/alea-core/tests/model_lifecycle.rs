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
