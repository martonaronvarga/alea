use alea_core::model::assembly::{
    ArgumentShape, AssemblyError as E, BatchRule, Expr, ModelBuilder, Primitive, PrimitiveArgument,
    PrimitiveSignature, ScalarPreparation,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

static PREPARATIONS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn fixed_argument_packets_preserve_aliases_score_admission_and_failure_recovery() {
    use alea_core::model::ConstrainedModel;
    for arity in 0..=6 {
        let mut builder = ModelBuilder::new();
        let x = builder.real("x").unwrap().expr();
        let mode = Arc::new(AtomicUsize::new(0));
        let control = Arc::clone(&mode);
        let root = builder
            .custom(&vec![x; arity], move |x, g| {
                match control.load(Ordering::Relaxed) {
                    1 => return Err(E::Shape),
                    2 if !g.is_empty() => return Ok(0.), // Missing writes must remain errors.
                    3 => panic!("callback control"),
                    _ => {}
                }
                let mut value = 0.;
                for (i, (&x, g)) in x.iter().zip(g).enumerate() {
                    *g = (i + 1) as f64;
                    value += x * *g;
                }
                Ok(value)
            })
            .unwrap();
        builder.likelihood("factor", root).unwrap();
        let model = builder.compile().unwrap().select_likelihood().unwrap();
        for failure in [1, 2, 3] {
            if failure == 2 && arity == 0 {
                continue;
            }
            mode.store(failure, Ordering::Relaxed);
            let mut gradient = [17.];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.logp_grad(&[0.5], &mut gradient)
            }));
            match failure {
                1 => assert!(matches!(result, Ok(Err(E::Shape)))),
                2 => assert!(matches!(result, Ok(Err(E::Numerical)))),
                _ => assert!(result.is_err()),
            }
            assert_eq!(gradient, [17.]);
            mode.store(0, Ordering::Relaxed);
            let expected = (arity * (arity + 1) / 2) as f64;
            assert_eq!(
                model.logp_grad(&[0.5], &mut gradient).unwrap(),
                0.5 * expected
            );
            assert_eq!(gradient, [expected]);
        }
    }
}
struct Prepared {
    input: [Expr; 1],
    bias: f64,
}
impl Primitive for Prepared {
    type Workspace = ();
    fn inputs(&self) -> &[Expr] {
        &self.input
    }
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "prepared",
            arguments: vec![PrimitiveArgument {
                name: "x",
                shape: ArgumentShape::Scalar,
            }],
            probability: None,
        }
    }
    fn workspace(&self) {}
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|this| format!("x[0]+{:?}_f64", this.bias))
    }
    fn source_gradient_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|this| format!("g[0]=1.;x[0]+{:?}_f64", this.bias))
    }
    fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
        g[0] = 1.;
        Ok(x[0] + self.bias)
    }
    fn scalar_preparation(&self) -> Option<ScalarPreparation<Self>> {
        Some(ScalarPreparation {
            argument: 0,
            prepare: |x| {
                PREPARATIONS.fetch_add(1, Ordering::Relaxed);
                x
            },
            evaluate: |this, x, g, w, prepared| {
                assert_eq!(prepared.to_bits(), x[0].to_bits());
                this.evaluate(x, g, w)
            },
        })
    }
}

#[test]
fn shared_preparation_refreshes_after_input_changes_selection_and_rebinding() {
    use alea_core::model::ConstrainedModel;
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let a = b
        .primitive(Prepared {
            input: [x],
            bias: 0.,
        })
        .unwrap();
    let c = b
        .primitive(Prepared {
            input: [x],
            bias: 1.,
        })
        .unwrap();
    b.prior("a", a).unwrap();
    b.likelihood("c", c).unwrap();
    let model = b.compile().unwrap();
    PREPARATIONS.store(0, Ordering::Relaxed);
    for x in [0., 0., -0., 1., 1.] {
        let mut g = [17.];
        assert_eq!(model.logp_grad(&[x], &mut g).unwrap(), 2. * x + 1.);
        assert_eq!(g, [2.]);
    }
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 3);
    assert!(
        model
            .rust_first_order_gradient(true)
            .unwrap()
            .unwrap()
            .0
            .contains("g[0]=1.;x[0]+0.0_f64")
    );
    let model = model
        .rebind::<Prepared>(a, |p| {
            Ok(Prepared {
                input: p.input,
                bias: 2.,
            })
        })
        .unwrap();
    let mut g = [17.];
    assert_eq!(model.logp_grad(&[1.], &mut g).unwrap(), 5.);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 4);
    let source = model.rust_first_order_gradient(true).unwrap().unwrap().0;
    assert!(source.contains("g[0]=1.;x[0]+2.0_f64"));
    assert!(!source.contains("g[0]=1.;x[0]+0.0_f64"));
    let selected = model.select_likelihood().unwrap();
    assert_eq!(selected.logp_grad(&[1.], &mut g).unwrap(), 2.);
    assert_eq!(g, [1.]);

    let mut builder = ModelBuilder::new();
    let _unused = builder.real("unused").unwrap();
    let constant = builder.constant(2.).unwrap();
    for bias in [0., 1.] {
        let factor = builder
            .primitive(Prepared {
                input: [constant],
                bias,
            })
            .unwrap();
        builder.factor(factor).unwrap();
    }
    let model = builder.compile().unwrap();
    // Compilation does not run preparation (even an excluded pure rule may
    // panic). Constant preparation remains lazy, then reusable across forwards.
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 4);
    for input in [0., 1., -0.] {
        assert_eq!(model.logp_grad(&[input], &mut g).unwrap(), 5.);
        assert_eq!(g, [0.]);
        assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 5);
    }
}

struct Buffered {
    input: [Expr; 1],
    slots: usize,
}
impl Primitive for Buffered {
    type Workspace = ();
    fn inputs(&self) -> &[Expr] {
        &self.input
    }
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "buffered",
            arguments: vec![PrimitiveArgument {
                name: "x",
                shape: ArgumentShape::Scalar,
            }],
            probability: None,
        }
    }
    fn workspace(&self) {}
    fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
        g[0] = 1.;
        Ok(x[0])
    }
    fn source_workspace(&self) -> usize {
        self.slots
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|_| "scratch[0]=x[0]; scratch[0]".into())
    }
}

#[test]
fn source_workspace_is_checked_combined_selected_and_rebound_without_evaluating() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    assert!(matches!(
        b.primitive(Buffered {
            input: [x],
            slots: usize::MAX
        }),
        Err(E::Shape)
    ));
    let a = b
        .primitive(Buffered {
            input: [x],
            slots: 2,
        })
        .unwrap();
    let c = b
        .primitive(Buffered {
            input: [x],
            slots: 3,
        })
        .unwrap();
    b.prior("prior", a).unwrap();
    b.likelihood("likelihood", c).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.graph().count(), 3);
    // One argument buffer plus each primitive's independent scratch range.
    let (body, slots) = model.rust_primal_with_workspace().unwrap();
    assert_eq!(slots, 7);
    assert!(body.contains("scratch[0..3]"));
    assert!(body.contains("scratch[3..7]"));
    assert!(!body.contains("let mut scratch"));
    assert!(
        model
            .rust_primal()
            .unwrap()
            .contains("let mut scratch = [0.0_f64; 7]")
    );
    assert_eq!(model.rust_objective_primal_with_workspace().unwrap().1, 7);
    let model = model
        .rebind::<Buffered>(a, |p| {
            Ok(Buffered {
                input: p.input,
                slots: 4,
            })
        })
        .unwrap();
    assert_eq!(model.rust_primal_with_workspace().unwrap().1, 9);
    // Both individually legal buffers together exceed the addressable byte size.
    let max = isize::MAX as usize / size_of::<f64>();
    let model = model
        .rebind::<Buffered>(a, |p| {
            Ok(Buffered {
                input: p.input,
                slots: max - 2,
            })
        })
        .unwrap();
    assert!(matches!(model.rust_primal_with_workspace(), Err(E::Shape)));
    assert_eq!(
        model
            .select_likelihood()
            .unwrap()
            .rust_primal_with_workspace()
            .unwrap()
            .1,
        4
    );
}

struct Rule {
    inputs: [Expr; 1],
    exports: Arc<AtomicUsize>,
    units: usize,
}
impl Primitive for Rule {
    type Workspace = ();
    fn inputs(&self) -> &[Expr] {
        &self.inputs
    }
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "rule",
            arguments: vec![PrimitiveArgument {
                name: "x",
                shape: ArgumentShape::Scalar,
            }],
            probability: None,
        }
    }
    fn workspace(&self) {}
    fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
        g[0] = 1.;
        Ok(x[0])
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|p| {
            p.exports.fetch_add(1, Ordering::Relaxed);
            "x[0]".into()
        })
    }
    fn batch_rule(&self) -> Option<BatchRule<Self>> {
        Some(BatchRule {
            units: self.units,
            evaluate: |_, x, out, _| {
                out[0] = x[0] * 0.25;
                match x[0] {
                    13. => return Err(E::Shape),
                    14. => panic!("row callback panic"),
                    15. => return Ok(()), // incomplete successful write must fail
                    16. => {
                        out[1] = f64::INFINITY;
                        return Ok(());
                    }
                    _ => out[1] = x[0] * 0.75,
                }
                Ok(())
            },
        })
    }
}
#[test]
fn generated_sources_are_lazy_and_batch_failures_preserve_outputs_and_typed_causes() {
    let exports = Arc::new(AtomicUsize::new(0));
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    assert!(matches!(
        b.primitive(Rule {
            inputs: [x],
            exports: Arc::clone(&exports),
            units: usize::MAX
        }),
        Err(E::Shape)
    ));
    let f = b
        .primitive(Rule {
            inputs: [x],
            exports: Arc::clone(&exports),
            units: 2,
        })
        .unwrap();
    b.likelihood("rows", f).unwrap();
    let model = b.compile().unwrap();
    assert!(model.source_capabilities().missing_primals.is_empty());
    assert_eq!(exports.load(Ordering::Relaxed), 0);
    let source = model.rust_primal().unwrap();
    assert_eq!(exports.load(Ordering::Relaxed), 1);
    let failed = model
        .rebind::<Rule>(f, |p| {
            Ok(Rule {
                inputs: p.inputs,
                exports: Arc::clone(&p.exports),
                units: 3,
            })
        })
        .err()
        .unwrap();
    assert!(matches!(failed.source, E::Shape));
    let model = *failed.model;
    let model = model
        .rebind::<Rule>(f, |p| {
            Ok(Rule {
                inputs: p.inputs,
                exports: Arc::clone(&p.exports),
                units: 2,
            })
        })
        .unwrap();
    assert_eq!(exports.load(Ordering::Relaxed), 1);
    assert_eq!(model.rust_primal().unwrap(), source);
    let report = model.batch_likelihood().unwrap();
    assert_eq!(
        report
            .units()
            .iter()
            .map(|u| (u.factor.as_str(), u.row))
            .collect::<Vec<_>>(),
        [("rows", 0), ("rows", 1)]
    );
    let mut rows = [91., 92.];
    for q in [13., 14., 15., 16.] {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            report.evaluate(&[q], &mut rows)
        }));
        match q {
            13. => assert!(matches!(outcome.unwrap(), Err(E::Shape))),
            14. => assert!(outcome.is_err()),
            _ => assert!(matches!(outcome.unwrap(), Err(E::Numerical))),
        }
        assert_eq!(rows, [91., 92.]);
        assert_eq!(report.evaluate(&[4.], &mut rows).unwrap(), 4.);
        assert_eq!(rows, [1., 3.]);
        rows = [91., 92.];
    }
    assert!(report.evaluate(&[f64::NAN], &mut rows).is_err());
    assert_eq!(rows, [91., 92.]);
    assert_eq!(
        allocation_counter::measure(|| {
            report.evaluate(&[4.], &mut rows).unwrap();
        })
        .count_total,
        0
    );
}
