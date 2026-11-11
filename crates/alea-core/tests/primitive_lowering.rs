use alea_core::model::assembly::{
    ArgumentShape, AssemblyError as E, BatchRule, ModelBuilder, PrimitiveArgument,
    PrimitiveSignature, ScalarPreparation,
};
use alea_core::model::assembly::{NativeRules, OperationDescriptor, SourceBody, SourceRule};
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
    bias: f64,
}
impl Prepared {
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
    /// Native kernel workspace.
    pub fn workspace(&self) {}
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|this| format!("x[0]+{:?}_f64", this.bias))
    }
    fn source_gradient_rule(
        &self,
    ) -> Option<fn(&Self) -> alea_core::model::assembly::SourceGradient> {
        Some(|this| format!("g[0]=1.;x[0]+{:?}_f64", this.bias).into())
    }
    /// Native kernel evaluate.
    pub fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
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
impl OperationDescriptor for Prepared {
    type Workspace = ();
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        rules.preparation = self.scalar_preparation();
        let body = self.source_rule().map(SourceBody::Generated);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: self.source_gradient_rule(),
            workspace: 0,
        });
        rules
    }
}

#[test]
fn shared_preparation_refreshes_after_input_changes_selection_and_rebinding() {
    use alea_core::model::ConstrainedModel;
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let a = b.primitive(&[x], Prepared { bias: 0. }).unwrap();
    let c = b.primitive(&[x], Prepared { bias: 1. }).unwrap();
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
            .source_first_order(alea_core::model::assembly::SourceMode::Density)
            .unwrap()
            .unwrap()
            .program
            .source
            .contains("g[0]=1.;x[0]+0.0_f64")
    );
    let model = model
        .binding()
        .rebind::<Prepared>(a, |_| Ok(Prepared { bias: 2. }))
        .unwrap()
        .compile()
        .unwrap()
        .into_native_model();
    let mut g = [17.];
    assert_eq!(model.logp_grad(&[1.], &mut g).unwrap(), 5.);
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 4);
    let source = model
        .source_first_order(alea_core::model::assembly::SourceMode::Density)
        .unwrap()
        .unwrap()
        .program
        .source;
    assert!(source.contains("g[0]=1.;x[0]+2.0_f64"));
    assert!(!source.contains("g[0]=1.;x[0]+0.0_f64"));
    let selected = model.select_likelihood().unwrap();
    assert_eq!(selected.logp_grad(&[1.], &mut g).unwrap(), 2.);
    assert_eq!(g, [1.]);

    let mut builder = ModelBuilder::new();
    let _unused = builder.real("unused").unwrap();
    let constant = builder.constant(2.).unwrap();
    for bias in [0., 1.] {
        let factor = builder.primitive(&[constant], Prepared { bias }).unwrap();
        builder.factor(factor).unwrap();
    }
    let model = builder.compile().unwrap();
    // Compilation does not run preparation (even an excluded pure rule may
    // panic). Constant preparation remains lazy, then reusable across forwards.
    // Selecting another executable constructs a fresh context and preparation cache.
    assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 5);
    for input in [0., 1., -0.] {
        assert_eq!(model.logp_grad(&[input], &mut g).unwrap(), 5.);
        assert_eq!(g, [0.]);
        assert_eq!(PREPARATIONS.load(Ordering::Relaxed), 6);
    }
}

struct Buffered {
    slots: usize,
}
impl Buffered {
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
    /// Native kernel workspace.
    pub fn workspace(&self) {}
    /// Native kernel evaluate.
    pub fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
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
impl OperationDescriptor for Buffered {
    type Workspace = ();
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        let body = self.source_rule().map(SourceBody::Generated);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: None,
            workspace: self.source_workspace(),
        });
        rules
    }
}

#[test]
fn source_workspace_is_checked_combined_selected_and_rebound_without_evaluating() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    assert!(matches!(
        b.primitive(&[x], Buffered { slots: usize::MAX }),
        Err(E::Shape)
    ));
    let a = b.primitive(&[x], Buffered { slots: 2 }).unwrap();
    let c = b.primitive(&[x], Buffered { slots: 3 }).unwrap();
    b.prior("prior", a).unwrap();
    b.likelihood("likelihood", c).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.graph().count(), 3);
    // One argument buffer plus each primitive's independent scratch range.
    let primal = model
        .source_primal(alea_core::model::assembly::SourceMode::Density)
        .unwrap();
    let body = primal.source;
    let slots = primal.workspace.elements;
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
    assert_eq!(
        model
            .source_primal(alea_core::model::assembly::SourceMode::Objective)
            .unwrap()
            .workspace
            .elements,
        7
    );
    let model = model
        .binding()
        .rebind::<Buffered>(a, |_| Ok(Buffered { slots: 4 }))
        .unwrap()
        .compile()
        .unwrap()
        .into_native_model();
    assert_eq!(
        model
            .source_primal(alea_core::model::assembly::SourceMode::Density)
            .unwrap()
            .workspace
            .elements,
        9
    );
    // Both individually legal buffers together exceed the addressable byte size.
    let max = isize::MAX as usize / size_of::<f64>();
    let model = model
        .binding()
        .rebind::<Buffered>(a, |_| Ok(Buffered { slots: max - 2 }))
        .unwrap()
        .compile()
        .unwrap()
        .into_native_model();
    assert!(matches!(
        model.source_primal(alea_core::model::assembly::SourceMode::Density),
        Err(E::Shape)
    ));
    assert_eq!(
        model
            .select_likelihood()
            .unwrap()
            .source_primal(alea_core::model::assembly::SourceMode::Density)
            .unwrap()
            .workspace
            .elements,
        4
    );
}

struct Rule {
    exports: Arc<AtomicUsize>,
    workspaces: Arc<AtomicUsize>,
    units: usize,
}
impl Rule {
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
    /// Native kernel workspace.
    pub fn workspace(&self) {
        self.workspaces.fetch_add(1, Ordering::Relaxed);
    }
    /// Native kernel evaluate.
    pub fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
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
impl OperationDescriptor for Rule {
    type Workspace = ();
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        rules.batch = self.batch_rule();
        let body = self.source_rule().map(SourceBody::Generated);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: None,
            workspace: 0,
        });
        rules
    }
}
#[test]
fn generated_sources_are_lazy_and_batch_failures_preserve_outputs_and_typed_causes() {
    let exports = Arc::new(AtomicUsize::new(0));
    let workspaces = Arc::new(AtomicUsize::new(0));
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    assert!(matches!(
        b.primitive(
            &[x],
            Rule {
                exports: Arc::clone(&exports),
                workspaces: Arc::clone(&workspaces),
                units: usize::MAX
            }
        ),
        Err(E::Shape)
    ));
    assert_eq!(workspaces.load(Ordering::Relaxed), 0);
    let f = b
        .primitive(
            &[x],
            Rule {
                exports: Arc::clone(&exports),
                workspaces: Arc::clone(&workspaces),
                units: 2,
            },
        )
        .unwrap();
    b.likelihood("rows", f).unwrap();
    let model = b.compile().unwrap();
    assert!(model.source_capabilities().missing_primals.is_empty());
    assert_eq!(exports.load(Ordering::Relaxed), 0);
    assert_eq!(workspaces.load(Ordering::Relaxed), 2);
    let source = model.rust_primal().unwrap();
    assert_eq!(exports.load(Ordering::Relaxed), 1);
    let failed = model
        .binding()
        .rebind::<Rule>(f, |p| {
            Ok(Rule {
                exports: Arc::clone(&p.exports),
                workspaces: Arc::clone(&p.workspaces),
                units: 3,
            })
        })
        .err()
        .unwrap();
    assert!(matches!(failed, E::Shape));
    assert_eq!(workspaces.load(Ordering::Relaxed), 2);
    let model = model
        .binding()
        .rebind::<Rule>(f, |p| {
            Ok(Rule {
                exports: Arc::clone(&p.exports),
                workspaces: Arc::clone(&p.workspaces),
                units: 2,
            })
        })
        .unwrap()
        .compile()
        .unwrap()
        .into_native_model();
    assert_eq!(workspaces.load(Ordering::Relaxed), 4);
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
