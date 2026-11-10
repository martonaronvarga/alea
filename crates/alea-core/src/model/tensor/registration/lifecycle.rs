//! R0.2b private lifecycle proof. TensorProgram is the definition, BoundProgram
//! the immutable data snapshot, and the existing tensor executor does all work.
//! These names are not a proposed public API or a second model representation.
use super::*;
use crate::target::{LogDensityGradient, TargetSpace};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExecutableId(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContextId(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    definition: u64,
    binding: BindingId,
    executable: ExecutableId,
}
struct Executable {
    identity: Identity,
    binding: BoundProgram,
}
struct Context {
    identity: Identity,
    id: ContextId,
    epoch: u64,
    ready: Option<u64>,
    workspace: TensorWorkspace,
}
// This token is checked against the owning context's current successful forward.
// It neither owns residual arrays nor permits direct access to mutable scratch.
struct Residual {
    identity: Identity,
    context: ContextId,
    epoch: u64,
}
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum LifecycleError {
    #[error("evaluation context belongs to a different executable")]
    Context,
    #[error("residual is stale or belongs to a different evaluation context")]
    Residual,
    #[error("evaluation epoch space exhausted")]
    Epoch,
}
fn failure(error: LifecycleError) -> E {
    E::Factor(Box::new(error))
}
impl Context {
    fn begin(&mut self) -> Result<(), E> {
        // Invalidate before fallible work or callbacks, including unwinding.
        self.ready = None;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| failure(LifecycleError::Epoch))?;
        Ok(())
    }
}
impl BoundProgram {
    fn compile_native(self) -> Executable {
        Executable {
            identity: Identity {
                definition: self.plan.owner,
                binding: self.snapshot.id,
                executable: ExecutableId(identity()),
            },
            binding: self,
        }
    }
}
impl TensorProgram {
    pub(super) fn native_target(&self, data: &[(&str, &[f64])]) -> Result<OwnedTarget, E> {
        self.bind(data)?.compile_native().into_target()
    }
}
impl Executable {
    fn context(&self) -> Context {
        Context {
            identity: self.identity,
            id: ContextId(identity()),
            epoch: 0,
            ready: None,
            workspace: self.binding.workspace(),
        }
    }
    fn check(&self, context: &Context) -> Result<(), E> {
        if context.identity != self.identity {
            return Err(failure(LifecycleError::Context));
        }
        Ok(())
    }
    fn linearize(&self, context: &mut Context, x: &[f64]) -> Result<Residual, E> {
        self.check(context)?;
        context.begin()?;
        self.binding.prepare_point(x, &mut context.workspace)?;
        context.ready = Some(context.epoch);
        Ok(Residual {
            identity: self.identity,
            context: context.id,
            epoch: context.epoch,
        })
    }
    fn pullback(
        &self,
        context: &mut Context,
        residual: &Residual,
        seed: &[f64],
        out: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), E> {
        self.check(context)?;
        if residual.identity != self.identity
            || residual.context != context.id
            || context.ready != Some(residual.epoch)
        {
            return Err(failure(LifecycleError::Residual));
        }
        // A native VJP may modify its local scratch or fail. Do not promise a
        // reusable linearization without a residual/scratch separation contract.
        context.ready = None;
        self.binding
            .pullback_point(seed, out, g, &mut context.workspace)
    }
    fn value_vjp(
        &self,
        context: &mut Context,
        x: &[f64],
        seed: &[f64],
        out: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), E> {
        self.check(context)?;
        context.begin()?;
        self.binding
            .value_and_vjp(x, seed, out, g, &mut context.workspace)
    }
    fn into_target(self) -> Result<OwnedTarget, E> {
        if self.binding.output_shape() != &Shape::scalar() {
            return Err(E::Shape);
        }
        let context = RefCell::new(self.context());
        Ok(OwnedTarget {
            executable: self,
            context,
        })
    }
}
// Only this convenience adapter uses interior mutability for today's &self
// target trait. Executable evaluation receives exclusive scratch. No Sync bound
// is added to TargetSpace/LogDensityGradient or to other user targets.
pub(super) struct OwnedTarget {
    executable: Executable,
    context: RefCell<Context>,
}
impl TargetSpace for OwnedTarget {
    type Error = E;
    fn dimension(&self) -> usize {
        self.executable.binding.dimension()
    }
}
impl LogDensityGradient for OwnedTarget {
    fn is_numerical_error(&self, error: &E) -> bool {
        matches!(error, E::Numerical)
    }
    fn logp_grad(&self, x: &[f64], g: &mut [f64]) -> Result<f64, E> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| E::WorkspaceBusy)?;
        let mut out = [0.];
        self.executable
            .value_vjp(&mut context, x, &[1.], &mut out, g)?;
        Ok(out[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{EvaluationWorkspace, PointState};
    use alea_math::buffer::OwnedBuffer;
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct Controls {
        preparations: AtomicUsize,
        forwards: AtomicUsize,
        reverses: AtomicUsize,
        binding_failure: AtomicUsize,
        forward_failure: AtomicUsize,
        reverse_failure: AtomicUsize,
    }
    // A deterministic polynomial, not another probability-law implementation.
    // Fixed-data preparation forms a scale; forward saves the current x in
    // context-local scratch and VJP uses that residual once without replay.
    fn definition() -> (TensorProgram, Arc<Controls>) {
        let control = Arc::new(Controls::default());
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let weights = b.data("weights", Shape::new(vec![2]).unwrap()).unwrap();
        let yes = b
            .typed_data(Shape::scalar(), Data::Bool(vec![true].into()))
            .unwrap();
        let count = b
            .typed_data(Shape::scalar(), Data::Count(vec![u64::MAX].into()))
            .unwrap();
        let index = b
            .typed_data(Shape::scalar(), Data::Index(vec![usize::MAX - 1].into()))
            .unwrap();
        let inputs = [x, weights, yes, count, index];
        let signature = Signature {
            inputs: inputs
                .iter()
                .map(|v| BlockType::of(b.node(*v).unwrap()))
                .collect(),
            output: BlockType {
                dtype: ElementType::Real,
                shape: Shape::scalar(),
            },
            scratch: 1,
        };
        let (p, f, r) = (
            Arc::clone(&control),
            Arc::clone(&control),
            Arc::clone(&control),
        );
        let y = b.register(signature, &inputs, || Ok(Kernel {
            binding: Some(BindingRule {
                arguments: vec![1, 2, 3, 4].into_boxed_slice(), elements: 1,
                prepare: Box::new(move |args, out| {
                    p.preparations.fetch_add(1, Ordering::Relaxed);
                    match p.binding_failure.load(Ordering::Relaxed) {
                        1 => return Err(E::Name), 2 => return Ok(()),
                        3 => { out[0] = f64::INFINITY; return Ok(()); }
                        4 => panic!("binding preparation"), _ => {}
                    }
                    let BindingValue::Real(weights) = args.get(0)? else { return Err(E::Shape) };
                    assert!(matches!(args.get(1)?, BindingValue::Bool([true])));
                    assert!(matches!(args.get(2)?, BindingValue::Count([u64::MAX])));
                    assert!(matches!(args.get(3)?, BindingValue::Index([v]) if *v == usize::MAX - 1));
                    assert!(matches!(args.get(4), Err(E::Shape)));
                    out[0] = weights[0] + weights[1];
                    Ok(())
                }),
            }),
            forward: Box::new(move |args, out, scratch| {
                f.forwards.fetch_add(1, Ordering::Relaxed);
                scratch[0] = f64::NAN;
                match f.forward_failure.load(Ordering::Relaxed) {
                    1 => return Err(E::Name), 2 => return Ok(()),
                    3 => panic!("point preparation"), _ => {}
                }
                scratch[0] = args.real(0)?[0];
                out[0] = args.prepared[0] * scratch[0] * scratch[0];
                Ok(())
            }),
            vjp: Some(Box::new(move |args, _, seed, g, scratch| {
                r.reverses.fetch_add(1, Ordering::Relaxed);
                match r.reverse_failure.load(Ordering::Relaxed) {
                    1 => return Err(E::Name), 2 => return Ok(()),
                    3 => panic!("point pullback"), _ => {}
                }
                g.fill(0.);
                g[0] = 2. * args.prepared[0] * scratch[0] * seed[0];
                scratch[0] = f64::NAN; // Forces the one-use residual contract.
                Ok(())
            })),
        })).unwrap();
        (b.compile(y).unwrap(), control)
    }
    fn bind(p: &TensorProgram, weights: &[f64]) -> BoundProgram {
        p.bind(&[("weights", weights)]).unwrap()
    }
    fn answer(e: &Executable, c: &mut Context, x: f64, scale: f64) {
        let (mut out, mut g) = ([17.], [19.]);
        e.value_vjp(c, &[x], &[1.], &mut out, &mut g).unwrap();
        assert_eq!(out, [scale * x * x]);
        assert_eq!(g, [2. * scale * x]);
    }
    fn is_lifecycle(e: &E, expected: LifecycleError) -> bool {
        matches!(e, E::Factor(source) if source.downcast_ref::<LifecycleError>() == Some(&expected))
    }

    #[test]
    fn binding_rules_reject_dynamic_dependencies_bad_arguments_and_extent_before_graph_mutation() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let data = b.data("data", Shape::scalar()).unwrap();
        let derived_data = b.unary(UnaryOp::Neg, data).unwrap();
        for (input, argument, elements) in [
            (x, 0, 1),
            (derived_data, 0, 1),
            (data, 1, 1),
            (data, 0, usize::MAX),
        ] {
            let nodes = b.nodes.len();
            let result = b.register(
                Signature {
                    inputs: vec![BlockType::of(b.node(input).unwrap())],
                    output: BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar(),
                    },
                    scratch: 0,
                },
                &[input],
                || {
                    Ok(Kernel {
                        forward: Box::new(|_, out, _| {
                            out[0] = 0.;
                            Ok(())
                        }),
                        vjp: None,
                        binding: Some(BindingRule {
                            arguments: vec![argument].into_boxed_slice(),
                            elements,
                            prepare: Box::new(|_, _| {
                                panic!("invalid binding preparation executed")
                            }),
                        }),
                    })
                },
            );
            assert!(matches!(result, Err(E::Shape)));
            assert_eq!(b.nodes.len(), nodes);
        }
        // A direct immutable constant is permitted, and its preparation may
        // itself be empty; neither requires a parameter or derivative buffer.
        let constant = b.constant(Shape::scalar(), vec![2.]).unwrap();
        let output = b
            .register(
                Signature {
                    inputs: vec![BlockType::of(b.node(constant).unwrap())],
                    output: BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar(),
                    },
                    scratch: 0,
                },
                &[constant],
                || {
                    Ok(Kernel {
                        binding: Some(BindingRule {
                            arguments: vec![0].into_boxed_slice(),
                            elements: 0,
                            prepare: Box::new(|args, out| {
                                assert!(out.is_empty());
                                assert!(matches!(args.get(0)?, BindingValue::Real([2.])));
                                Ok(())
                            }),
                        }),
                        forward: Box::new(|args, out, _| {
                            assert!(args.prepared.is_empty());
                            out[0] = args.real(0)?[0];
                            Ok(())
                        }),
                        vjp: None,
                    })
                },
            )
            .unwrap();
        let e = b
            .compile(output)
            .unwrap()
            .bind(&[("data", &[1.])])
            .unwrap()
            .compile_native();
        let (mut out, mut g) = ([17.], [19.]);
        e.value_vjp(&mut e.context(), &[1.], &[1.], &mut out, &mut g)
            .unwrap();
        assert_eq!((out, g), ([2.], [0.])); // Unused x stays inactive.
    }

    #[test]
    fn snapshots_preparation_and_executable_identities_are_distinct_and_transactional() {
        let (p, ctl) = definition();
        let binding = bind(&p, &[2., 3.]);
        let clone = binding.clone();
        assert_eq!(binding.snapshot.id, clone.snapshot.id);
        let e = binding.clone().compile_native();
        let other = clone.compile_native();
        assert_eq!(e.identity.binding, other.identity.binding);
        assert_ne!(e.identity.executable, other.identity.executable);
        let mut c = e.context();
        for invalid in [
            [("weights", &[1.][..])],
            [("unknown", &[1., 2.][..])],
            [("weights", &[f64::NAN, 2.][..])],
        ] {
            assert!(p.bind(&invalid).is_err());
            assert!(binding.rebind(invalid[0].0, invalid[0].1).is_err());
        }
        assert_eq!(ctl.preparations.load(Ordering::Relaxed), 1);
        for mode in 1..=4 {
            ctl.binding_failure.store(mode, Ordering::Relaxed);
            let result = catch_unwind(AssertUnwindSafe(|| binding.rebind("weights", &[3., 4.])));
            match mode {
                1 => assert!(matches!(result, Ok(Err(E::Name)))),
                2 | 3 => assert!(matches!(result, Ok(Err(E::Numerical)))),
                4 => assert!(result.is_err()),
                _ => unreachable!(),
            }
            answer(&e, &mut c, 2., 5.);
        }
        ctl.binding_failure.store(0, Ordering::Relaxed);
        let same = binding.rebind("weights", &[2., 3.]).unwrap();
        let changed = binding.rebind("weights", &[3., 4.]).unwrap();
        assert_ne!(same.snapshot.id, binding.snapshot.id);
        assert_ne!(changed.snapshot.id, same.snapshot.id);
        assert!(Arc::ptr_eq(&p.plan, &changed.plan));
        assert_eq!(ctl.preparations.load(Ordering::Relaxed), 7);
        let changed = changed.compile_native();
        answer(&changed, &mut changed.context(), 2., 7.);
        answer(&e, &mut c, 2., 5.);
        let mut out = [17.];
        let mut g = [19.];
        assert!(is_lifecycle(
            &other
                .value_vjp(&mut c, &[2.], &[1.], &mut out, &mut g)
                .unwrap_err(),
            LifecycleError::Context
        ));
        assert!(is_lifecycle(
            &changed
                .value_vjp(&mut c, &[2.], &[1.], &mut out, &mut g)
                .unwrap_err(),
            LifecycleError::Context
        ));
        assert_eq!((out, g), ([17.], [19.]));
        let old_point = e.linearize(&mut c, &[2.]).unwrap();
        let mut changed_context = changed.context();
        let changed_point = changed.linearize(&mut changed_context, &[2.]).unwrap();
        assert!(is_lifecycle(
            &changed
                .pullback(&mut changed_context, &old_point, &[1.], &mut out, &mut g)
                .unwrap_err(),
            LifecycleError::Residual
        ));
        changed
            .pullback(
                &mut changed_context,
                &changed_point,
                &[1.],
                &mut out,
                &mut g,
            )
            .unwrap();
        assert_eq!((out, g), ([28.], [28.]));
        e.pullback(&mut c, &old_point, &[1.], &mut out, &mut g)
            .unwrap();
        assert_eq!((out, g), ([20.], [20.]));
        // Existing raw same-plan workspaces can refresh another binding; the
        // stricter executable context instead requires an explicit new context.
        let (foreign, _) = definition();
        let foreign = bind(&foreign, &[2., 3.]).compile_native();
        assert_ne!(foreign.identity.definition, e.identity.definition);
        assert!(is_lifecycle(
            &foreign
                .value_vjp(&mut c, &[2.], &[1.], &mut out, &mut g)
                .unwrap_err(),
            LifecycleError::Context
        ));
    }

    #[test]
    fn contexts_share_only_immutable_plans_and_residuals_reject_other_points_and_contexts() {
        let (p, ctl) = definition();
        let e = Arc::new(bind(&p, &[2., 3.]).compile_native());
        let (mut a, mut b) = (e.context(), e.context());
        assert_ne!(a.id, b.id);
        let ra = e.linearize(&mut a, &[2.]).unwrap();
        let rb = e.linearize(&mut b, &[3.]).unwrap();
        let (mut out, mut g) = ([17.], [19.]);
        assert!(is_lifecycle(
            &e.pullback(&mut b, &ra, &[1.], &mut out, &mut g)
                .unwrap_err(),
            LifecycleError::Residual
        ));
        assert_eq!((out, g), ([17.], [19.]));
        e.pullback(&mut b, &rb, &[1.], &mut out, &mut g).unwrap();
        assert_eq!((out, g), ([45.], [30.]));
        e.pullback(&mut a, &ra, &[1.], &mut out, &mut g).unwrap();
        assert_eq!((out, g), ([20.], [20.]));
        assert_eq!(ctl.forwards.load(Ordering::Relaxed), 2); // VJP did not replay.
        assert!(e.pullback(&mut a, &ra, &[1.], &mut out, &mut g).is_err());
        let old = e.linearize(&mut a, &[2.]).unwrap();
        let current = e.linearize(&mut a, &[2.]).unwrap(); // Equal coordinates still new epoch.
        assert!(e.pullback(&mut a, &old, &[1.], &mut out, &mut g).is_err());
        e.pullback(&mut a, &current, &[-0.5], &mut out, &mut g)
            .unwrap();
        assert_eq!((out, g), ([20.], [-10.]));
        std::thread::scope(|scope| {
            for x in [2., 3.] {
                let e = Arc::clone(&e);
                scope.spawn(move || {
                    let mut c = e.context();
                    for _ in 0..10 {
                        answer(&e, &mut c, x, 5.);
                    }
                });
            }
        });
        assert_eq!(ctl.preparations.load(Ordering::Relaxed), 1);
        assert_eq!(Arc::strong_count(&e), 1);
    }

    #[test]
    fn failed_forward_pullback_and_unwind_invalidate_residuals_and_recover() {
        let (p, ctl) = definition();
        let e = bind(&p, &[2., 3.]).compile_native();
        let mut c = e.context();
        for mode in 1..=3 {
            let old = e.linearize(&mut c, &[2.]).unwrap();
            ctl.forward_failure.store(mode, Ordering::Relaxed);
            let result = catch_unwind(AssertUnwindSafe(|| e.linearize(&mut c, &[3.])));
            match mode {
                1 => assert!(matches!(result, Ok(Err(E::Name)))),
                2 => assert!(matches!(result, Ok(Err(E::Numerical)))),
                3 => assert!(result.is_err()),
                _ => unreachable!(),
            }
            let (mut out, mut g) = ([17.], [19.]);
            assert!(e.pullback(&mut c, &old, &[1.], &mut out, &mut g).is_err());
            assert_eq!((out, g), ([17.], [19.]));
            ctl.forward_failure.store(0, Ordering::Relaxed);
            answer(&e, &mut c, 2., 5.);
        }
        for seed in [0., 1.] {
            for mode in 1..=3 {
                let token = e.linearize(&mut c, &[2.]).unwrap();
                ctl.reverse_failure.store(mode, Ordering::Relaxed);
                let (mut out, mut g) = ([17.], [19.]);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    e.pullback(&mut c, &token, &[seed], &mut out, &mut g)
                }));
                match mode {
                    1 => assert!(matches!(result, Ok(Err(E::Name)))),
                    2 => assert!(matches!(result, Ok(Err(E::Numerical)))),
                    3 => assert!(result.is_err()),
                    _ => unreachable!(),
                }
                assert_eq!((out, g), ([17.], [19.]));
                ctl.reverse_failure.store(0, Ordering::Relaxed);
                assert!(e.pullback(&mut c, &token, &[1.], &mut out, &mut g).is_err());
                answer(&e, &mut c, 2., 5.);
            }
        }
        for x in [&[][..], &[f64::NAN][..]] {
            let token = e.linearize(&mut c, &[2.]).unwrap();
            let before = ctl.forwards.load(Ordering::Relaxed);
            assert!(e.linearize(&mut c, x).is_err());
            assert_eq!(ctl.forwards.load(Ordering::Relaxed), before);
            assert!(
                e.pullback(&mut c, &token, &[1.], &mut [17.], &mut [19.])
                    .is_err()
            );
        }
        let token = e.linearize(&mut c, &[2.]).unwrap();
        let (mut out, mut g) = ([17.], [19.]);
        let before = ctl.reverses.load(Ordering::Relaxed);
        assert!(matches!(
            e.pullback(&mut c, &token, &[f64::NAN], &mut out, &mut g),
            Err(E::Numerical)
        ));
        assert_eq!(ctl.reverses.load(Ordering::Relaxed), before);
        assert_eq!((out, g), ([17.], [19.]));
        assert!(e.pullback(&mut c, &token, &[1.], &mut out, &mut g).is_err());
        c.epoch = u64::MAX;
        assert!(is_lifecycle(
            &e.linearize(&mut c, &[2.]).err().unwrap(),
            LifecycleError::Epoch
        ));
        assert!(c.ready.is_none());
    }

    #[test]
    fn owned_non_sync_target_uses_existing_coherent_points_and_allocates_nothing_per_call() {
        let (p, ctl) = definition();
        let target = p.native_target(&[("weights", &[2., 3.])]).unwrap();
        drop(p); // Owned adapter retains its definition, binding and prepared data.
        let mut point = PointState::new(&target, OwnedBuffer::from_fn(1, |_| 2.)).unwrap();
        let mut work = EvaluationWorkspace::new(1);
        assert_eq!((point.log_density(), point.gradient()), (20., &[20.][..]));
        let allocations = allocation_counter::measure(|| {
            for _ in 0..32 {
                point.try_update(&[3.], &mut work).unwrap();
            }
        });
        assert_eq!((allocations.count_total, allocations.bytes_total), (0, 0));
        assert_eq!(ctl.preparations.load(Ordering::Relaxed), 1);
        let before = (
            point.position().to_vec(),
            point.gradient().to_vec(),
            point.log_density(),
        );
        for mode in 1..=3 {
            ctl.forward_failure.store(mode, Ordering::Relaxed);
            match catch_unwind(AssertUnwindSafe(|| point.try_update(&[2.], &mut work))) {
                Ok(result) => assert!(result.is_err()),
                Err(_) => assert_eq!(mode, 3),
            }
            assert_eq!(
                (point.position(), point.gradient(), point.log_density()),
                (&before.0[..], &before.1[..], before.2)
            );
            ctl.forward_failure.store(0, Ordering::Relaxed);
            point.try_update(&[3.], &mut work).unwrap();
        }
        let lock = target.context.borrow_mut();
        let mut gradient = [19.];
        assert!(matches!(
            target.logp_grad(&[2.], &mut gradient),
            Err(E::WorkspaceBusy)
        ));
        assert_eq!(gradient, [19.]);
        drop(lock);
        assert_eq!(target.logp_grad(&[2.], &mut gradient).unwrap(), 20.);
        assert!(target.is_numerical_error(&E::Numerical));
        assert!(!target.is_numerical_error(&failure(LifecycleError::Residual)));
        assert!(!target.is_numerical_error(&E::WorkspaceBusy));
    }

    #[test]
    fn scalar_adapter_rejects_vector_one_and_accepts_empty_coordinate_space() {
        let mut b = RegionBuilder::new();
        let x = b.input("empty", Shape::new(vec![0]).unwrap()).unwrap();
        let sum = b.sum(x).unwrap();
        let p = b.compile(sum).unwrap();
        let target = p.native_target(&[]).unwrap();
        assert_eq!(target.dimension(), 0);
        assert_eq!(target.logp_grad(&[], &mut []).unwrap(), 0.);
        let mut b = RegionBuilder::new();
        let x = b.input("one", Shape::new(vec![1]).unwrap()).unwrap();
        let p = b.compile(x).unwrap();
        assert!(matches!(p.native_target(&[]), Err(E::Shape)));
        let e = p.bind(&[]).unwrap().compile_native();
        let mut context = e.context();
        let token = e.linearize(&mut context, &[2.]).unwrap();
        let mut out = [17.];
        let mut g = [19.];
        assert!(matches!(
            e.pullback(&mut context, &token, &[], &mut out, &mut g),
            Err(E::Shape)
        ));
        assert_eq!((out, g), ([17.], [19.]));
        e.value_vjp(&mut context, &[2.], &[1.], &mut out, &mut g)
            .unwrap();
        assert_eq!((out, g), ([2.], [1.]));
    }
}
