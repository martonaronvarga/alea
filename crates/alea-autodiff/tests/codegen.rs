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
    use alea_core::model::tensor::Shape;
    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let _ = body.input("row", Shape::scalar()).unwrap();
    let body = body.compile(carry).unwrap().bind(&[]).unwrap();
    let mut t = ModelBuilder::new().region();
    let input = t.input("input", Shape::scalar()).unwrap();
    let rows = t.constant(Shape::new(vec![0]).unwrap(), vec![]).unwrap();
    let output = t.checkpointed_scan(input, rows, &[], body).unwrap();
    let binding = t.compile(output).unwrap().bind(&[]).unwrap();
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let term = b.apply_region(&[x], binding).unwrap();
    b.factor(term).unwrap();
    let source = enzyme_source(&b.compile().unwrap()).unwrap();
    assert!(source.contains("SOURCE_CHECKPOINTED: bool = true"));
    assert!(source.contains("SOURCE_TENSOR_SCORE_VALIDATION: bool = false"));
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

#[test]
fn compiler_tensor_scores_have_separate_checked_admission_without_changing_value_or_route() {
    use alea_core::model::tensor::{BinaryOp, Shape, UnaryOp};
    let mut t = ModelBuilder::new().region();
    let x = t.input("x", Shape::scalar()).unwrap();
    let square = t.binary(BinaryOp::Multiply, x, x).unwrap();
    let root = t.unary(UnaryOp::Sqrt, square).unwrap();
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let term = b
        .apply_region(&[x], t.compile(root).unwrap().bind(&[]).unwrap())
        .unwrap();
    b.factor(term).unwrap();
    let model = b.compile().unwrap();
    assert!(model.rust_first_order_gradient(true).unwrap().is_none());
    let (guard, slots) = model.rust_tensor_score_validation(true).unwrap().unwrap();
    assert!(slots > 0);
    assert!(guard.contains("fn __tensor"));
    for source in [
        enzyme_source(&model).unwrap(),
        alea_autodiff::codegen::enzyme_objective_source(&model).unwrap(),
    ] {
        assert!(source.contains("SOURCE_EXPLICIT_FIRST_ORDER: bool = false"));
        assert!(source.contains("SOURCE_TENSOR_SCORE_VALIDATION: bool = true"));
        assert!(source.contains("mod __tensor_admission"));
        assert!(source.contains("!__valid(q) || !__score_valid(q)"));
        assert!(source.contains("if !__score_valid(q) { return false; }"));
        // Value-only uses only rounded chart/primal admission.
        assert!(source.contains("!__valid(q) { f64::NAN } else { __log_density(q) }"));
    }
    let mut plain = ModelBuilder::new();
    let x = plain.real("x").unwrap().expr();
    plain.factor(x).unwrap();
    let plain = plain.compile().unwrap();
    assert!(plain.rust_tensor_score_validation(true).unwrap().is_none());
    assert!(
        enzyme_source(&plain)
            .unwrap()
            .contains("SOURCE_TENSOR_SCORE_VALIDATION: bool = false")
    );
}
