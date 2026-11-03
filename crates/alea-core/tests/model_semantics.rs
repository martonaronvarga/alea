use alea_core::model::{
    ConstrainedModel,
    assembly::{
        ArgumentShape, AssemblyError, Expr, ModelBuilder, OperationKind, Primitive,
        PrimitiveArgument, PrimitiveSignature,
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Square {
    inputs: [Expr; 1],
    width: usize,
    workspaces: Arc<AtomicUsize>,
}
impl Primitive for Square {
    type Workspace = usize;
    fn inputs(&self) -> &[Expr] {
        &self.inputs
    }
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "square",
            arguments: vec![PrimitiveArgument {
                name: "x",
                shape: ArgumentShape::Vector(self.width),
            }],
            probability: None,
        }
    }
    fn workspace(&self) -> usize {
        self.workspaces.fetch_add(1, Ordering::Relaxed);
        0
    }
    fn evaluate(&self, x: &[f64], g: &mut [f64], calls: &mut usize) -> Result<f64, AssemblyError> {
        *calls += 1; // Scratch may change, but must not change mathematical results.
        if x[0] == 15. {
            return Ok(0.); // Incomplete derivative, rejected by admission.
        }
        g[0] = 2. * x[0];
        if x[0] == 13. {
            return Err(AssemblyError::Numerical);
        }
        if x[0] == 14. {
            panic!("primitive evaluation panic");
        }
        Ok(x[0] * x[0])
    }
}

#[test]
fn registration_validates_before_workspace_creation_and_retains_typed_identity() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let workspaces = Arc::new(AtomicUsize::new(0));
    let square = |input, width| Square {
        inputs: [input],
        width,
        workspaces: Arc::clone(&workspaces),
    };
    assert!(matches!(
        b.primitive(square(x, 2)),
        Err(AssemblyError::Shape)
    ));
    let mut other = ModelBuilder::new();
    let foreign = other.real("x").unwrap().expr();
    assert!(matches!(
        b.primitive(square(foreign, 1)),
        Err(AssemblyError::ForeignHandle)
    ));
    assert_eq!(workspaces.load(Ordering::Relaxed), 0);
    let f = b.primitive(square(x, 1)).unwrap();
    b.likelihood("term", f).unwrap();
    b.factor(f).unwrap(); // Multiplicity is part of the density, not deduplicated.
    let model = b.compile().unwrap();
    assert_eq!(workspaces.load(Ordering::Relaxed), 1);
    assert_eq!(model.expression_count(), 2);
    let nodes: Vec<_> = model.graph().collect();
    assert_eq!(nodes[0].operation, OperationKind::Parameter(0));
    assert_eq!(nodes[1].operation, OperationKind::Primitive);
    assert_eq!(nodes[1].inputs, [x]);
    assert_eq!(nodes[1].expression, f);
    assert_eq!(nodes[1].factors.len(), 2);
    assert_eq!(nodes[1].factors[0].name(), Some("term"));
    let metadata = nodes[1].primitive.unwrap();
    assert!(metadata.descriptor::<Square>().is_some());
    assert!(metadata.signature().probability.is_none());
    assert_eq!(model.source_capabilities().missing_primals, [f]);
    let mut g = [0.];
    assert_eq!(model.logp_grad(&[2.], &mut g).unwrap(), 8.);
    assert_eq!(g, [8.]);
}

#[test]
fn primitive_failures_and_unwind_leave_live_output_untouched_and_scratch_reusable() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let f = b
        .primitive(Square {
            inputs: [x],
            width: 1,
            workspaces: Arc::default(),
        })
        .unwrap();
    b.factor(f).unwrap();
    let model = b.compile().unwrap();
    for invalid in [13., 14., 15.] {
        let mut gradient = [99.];
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.logp_grad(&[invalid], &mut gradient)
        }));
        if invalid == 14. {
            assert!(outcome.is_err());
        } else {
            assert!(matches!(outcome.unwrap(), Err(AssemblyError::Numerical)));
        }
        assert_eq!(gradient, [99.]);
        assert_eq!(model.logp_grad(&[2.], &mut gradient).unwrap(), 4.);
        assert_eq!(gradient, [4.]);
    }
    let mut gradient = [0.];
    let allocations = allocation_counter::measure(|| {
        for _ in 0..32 {
            std::hint::black_box(model.logp_grad(&[2.], &mut gradient).unwrap());
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn scopes_compose_reusable_functions_and_restore_names_after_errors_and_unwinding() {
    fn component(b: &mut ModelBuilder) -> Result<Expr, AssemblyError> {
        let p = b.real("mean")?;
        b.prior("prior", p.expr())?;
        Ok(p.expr())
    }
    let mut b = ModelBuilder::new();
    assert!(b.scope("").is_err());
    assert!(b.scope("a::b").is_err());
    let a = b.with_scope("a", component).unwrap();
    let c = {
        let mut outer = b.scope("outer").unwrap();
        component(&mut outer.scope("inner").unwrap()).unwrap()
    };
    assert_ne!(a, c);
    assert!(component(&mut b.scope("a").unwrap()).is_err());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut scope = b.scope("panic").unwrap();
        let _ = scope.real("kept").unwrap();
        panic!("scope unwind");
    }));
    assert!(panic.is_err());
    let _ = component(&mut b).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(
        model
            .coordinates()
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["a::mean", "outer::inner::mean", "panic::kept", "mean"]
    );
    assert_eq!(
        model
            .factors()
            .iter()
            .map(|f| f.name().unwrap())
            .collect::<Vec<_>>(),
        ["a::prior", "outer::inner::prior", "prior"]
    );
}

#[test]
fn scoped_vector_defaults_do_not_merge_unrelated_dimensions_or_geometry_blocks() {
    let mut b = ModelBuilder::new();
    for group in ["a", "b"] {
        let mut scope = b.scope(group).unwrap();
        let beta = scope.real_vector("beta", &["intercept", "slope"]).unwrap();
        scope.likelihood("data", beta.expressions()[0]).unwrap();
    }
    let model = b.compile().unwrap();
    let names = model.coordinates();
    for (i, group) in ["a", "b"].iter().enumerate() {
        assert_eq!(names[2 * i].name, format!("{group}::beta[intercept]"));
        assert_eq!(
            names[2 * i].dimension.as_deref(),
            Some(format!("{group}::beta").as_str())
        );
        assert_eq!(names[2 * i].block, names[2 * i].dimension);
    }
}

#[test]
fn malformed_signatures_reject_names_and_shape_overflow_before_lowering() {
    struct Bad {
        inputs: [Expr; 1],
        signature: PrimitiveSignature,
    }
    impl Primitive for Bad {
        type Workspace = ();
        fn inputs(&self) -> &[Expr] {
            &self.inputs
        }
        fn signature(&self) -> PrimitiveSignature {
            self.signature.clone()
        }
        fn workspace(&self) {
            panic!("invalid metadata must reject before workspace construction")
        }
        fn evaluate(&self, _: &[f64], _: &mut [f64], _: &mut ()) -> Result<f64, AssemblyError> {
            panic!("invalid metadata must never execute")
        }
    }
    let valid = PrimitiveSignature {
        name: "test",
        arguments: vec![PrimitiveArgument {
            name: "x",
            shape: ArgumentShape::Scalar,
        }],
        probability: None,
    };
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    b.factor(x).unwrap();
    let mut cases = vec![];
    let mut s = valid.clone();
    s.name = "";
    cases.push(s);
    let mut s = valid.clone();
    s.arguments[0].name = "";
    cases.push(s);
    let mut s = valid.clone();
    s.arguments.push(s.arguments[0].clone());
    cases.push(s);
    let mut s = valid.clone();
    s.arguments[0].shape = ArgumentShape::Matrix {
        rows: usize::MAX,
        columns: 2,
    };
    cases.push(s);
    let mut s = valid;
    s.probability = Some(alea_core::model::assembly::ProbabilitySemantics {
        event: alea_core::distribution::EventShape::Scalar,
        batch: vec![usize::MAX, 2],
        support: alea_core::distribution::Support::Real,
        measure: alea_core::distribution::ReferenceMeasure::Lebesgue { dimension: 1 },
    });
    cases.push(s);
    for signature in cases {
        assert!(
            b.primitive(Bad {
                inputs: [x],
                signature
            })
            .is_err()
        );
    }
    assert_eq!(b.compile().unwrap().expression_count(), 1);
}

#[test]
fn inspection_preserves_all_scalar_dependencies_constants_and_repeated_inputs() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let two = b.constant(2.).unwrap();
    let sum = b.add(x, x).unwrap();
    let product = b.multiply(sum, two).unwrap();
    let exponential = b.exp(product).unwrap();
    let log = b.log(exponential).unwrap();
    let opaque = b
        .custom(&[log], |x, g| {
            g[0] = 1.;
            Ok(x[0])
        })
        .unwrap();
    b.factor(opaque).unwrap();
    let model = b.compile().unwrap();
    let nodes: Vec<_> = model.graph().collect();
    assert_eq!(nodes[1].constant, Some(2.));
    assert_eq!(nodes[2].inputs, [x, x]);
    assert_eq!(nodes[3].inputs, [sum, two]);
    assert_eq!(nodes[4].inputs, [product]);
    assert_eq!(nodes[5].inputs, [exponential]);
    assert_eq!(nodes[6].inputs, [log]);
    assert_eq!(nodes[6].operation, OperationKind::Opaque);
    assert!(nodes.iter().all(|n| n.active));
    assert_eq!(model.source_capabilities().missing_primals, [opaque]);
}
