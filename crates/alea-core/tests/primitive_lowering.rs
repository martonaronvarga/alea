use alea_core::model::assembly::{
    ArgumentShape, AssemblyError as E, BatchRule, Expr, ModelBuilder, Primitive, PrimitiveArgument,
    PrimitiveSignature,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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
