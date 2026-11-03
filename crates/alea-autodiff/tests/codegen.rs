use alea_autodiff::codegen::enzyme_source;
use alea_core::{
    model::assembly::{AssemblyError, ModelBuilder},
    transform::Transform,
};

#[test]
fn a_zero_dimensional_chart_cannot_generate_a_sampler_target() {
    let mut b = ModelBuilder::new();
    let x = b
        .vector::<alea_core::model::assembly::Simplex>(
            "x",
            "simplex",
            &["only"],
            "x",
            Transform::simplex(1).unwrap(),
        )
        .unwrap();
    b.factor(x.expressions()[0]).unwrap();
    assert!(matches!(
        enzyme_source(&b.compile().unwrap()),
        Err(AssemblyError::Empty)
    ));
}

#[test]
fn checkpoint_higher_derivatives_keep_the_qualified_compiler_reverse() {
    use alea_core::model::tensor::{Shape, TensorBuilder, TensorFactor};
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    body.input("row", Shape::scalar()).unwrap();
    let body = body.compile(carry).unwrap().bind(&[]).unwrap();
    let mut t = TensorBuilder::new();
    let input = t.input("input", Shape::scalar()).unwrap();
    let rows = t.constant(Shape::new(vec![0]).unwrap(), vec![]).unwrap();
    let output = t.checkpointed_scan(input, rows, &[], body).unwrap();
    let binding = t.compile(output).unwrap().bind(&[]).unwrap();
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let term = b
        .primitive(TensorFactor::new(binding, &[x]).unwrap())
        .unwrap();
    b.factor(term).unwrap();
    let source = enzyme_source(&b.compile().unwrap()).unwrap();
    assert!(source.contains("SOURCE_CHECKPOINTED: bool = true"));
    assert!(source.contains("__higher_gradient(q, g, work, reverse, 1.0)"));
    assert!(source.contains("g.iter().any(|v|!v.is_finite()) || a.iter().any(|v|!v.is_finite())"));
    assert!(!source.contains("g.iter().chain(a.iter())"));
    assert!(source.contains("value.is_finite() && score.iter().all(|v| v.is_finite())"));
    assert_eq!(
        source
            .matches("!__valid(q) || !__first_order_valid(q)")
            .count(),
        2
    );
}
