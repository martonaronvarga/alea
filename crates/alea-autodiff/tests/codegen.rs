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
fn native_first_order_rules_do_not_claim_checkpointing_or_replace_higher_ad() {
    use alea_distributions::modeling::{
        normal,
        regression::{RegressionData, bernoulli_logit_regression},
    };
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap().expr();
    let sigma = b.positive("sigma").unwrap().expr();
    let y = b.constant(0.5).unwrap();
    let term = normal(&mut b, y, mu, sigma).unwrap();
    b.factor(term).unwrap();
    let batch = bernoulli_logit_regression(
        &mut b,
        &[mu],
        RegressionData::new(1, vec![1.; 4], vec![false, true, false, true]).unwrap(),
    )
    .unwrap();
    b.factor(batch).unwrap();
    let model = b.compile().unwrap();
    assert!(model.rust_checkpoint_gradient(true).unwrap().is_none());
    let source = enzyme_source(&model).unwrap();
    assert!(source.contains("SOURCE_EXPLICIT_FIRST_ORDER: bool = true"));
    assert!(source.contains("SOURCE_CHECKPOINTED: bool = false"));
    assert!(source.contains("ParameterScore::log_prob_event_parameter_score"));
    assert!(source.contains("kernel.evaluate(x,g,workspace)"));
    assert!(source.contains("__higher_gradient(q, g, work, reverse, 1.0)"));
    // Scores are admitted/staged on the forward visit, not recomputed in reverse.
    assert_eq!(source.matches("let value=__native").count(), 2);
    assert!(source.contains("scores.iter().any(|s|!s.is_finite())"));
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
