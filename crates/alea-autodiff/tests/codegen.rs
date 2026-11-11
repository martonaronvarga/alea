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
    assert!(
        !model
            .source_first_order(alea_core::model::assembly::SourceMode::Density)
            .unwrap()
            .unwrap()
            .checkpointed
    );
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
    assert!(
        model
            .source_first_order(alea_core::model::assembly::SourceMode::Density)
            .unwrap()
            .is_none()
    );
    let admission = model
        .source_score_admission(alea_core::model::assembly::SourceMode::Density)
        .unwrap()
        .unwrap();
    let slots = admission.workspace.elements;
    let guard = admission.source;
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
    assert!(
        plain
            .source_score_admission(alea_core::model::assembly::SourceMode::Density)
            .unwrap()
            .is_none()
    );
    assert!(
        enzyme_source(&plain)
            .unwrap()
            .contains("SOURCE_TENSOR_SCORE_VALIDATION: bool = false")
    );
}

#[test]
fn requests_emit_only_the_selected_entries_and_require_explicit_curvature() {
    use alea_core::model::assembly::{
        CurvatureRoute, DerivativeOrder as O, SourceMode, SourceRequest,
    };
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let term = b.multiply(x, x).unwrap();
    b.factor(term).unwrap();
    let model = b.compile().unwrap();
    for mode in [SourceMode::Density, SourceMode::Objective] {
        for order in [O::Value, O::First, O::Hvp, O::Third] {
            let mut request = SourceRequest {
                mode,
                order,
                curvature: CurvatureRoute::Unavailable,
            };
            if matches!(order, O::Hvp | O::Third) {
                assert!(
                    matches!(alea_autodiff::codegen::source(&model, request), Err(AssemblyError::UnqualifiedDerivative { order: rejected }) if rejected == order)
                );
                request.curvature = CurvatureRoute::OriginalCompilerReverse;
            }
            assert_eq!(
                model.source_capabilities().resolve(request).unwrap(),
                request
            );
            let source = alea_autodiff::codegen::source(&model, request).unwrap();
            assert_eq!(source.contains("autodiff_reverse"), order != O::Value);
            assert_eq!(
                source.contains("pub fn target()") || source.contains("pub fn objective()"),
                order != O::Value
            );
            assert_eq!(
                source.contains("__gradient_jvp"),
                matches!(order, O::Hvp | O::Third)
            );
            assert_eq!(source.contains("__hvp_jvp"), order == O::Third);
            assert_eq!(
                source.contains("PotentialHessianDerivative"),
                order == O::Third
            );
            assert_eq!(source.contains("PotentialHessianVector"), order == O::Hvp);
            if order == O::First {
                assert!(source.contains("impl ::alea::target::LogDensityGradient"));
            }
        }
    }
}

#[test]
fn native_vjp_does_not_qualify_any_source_order() {
    use alea_core::model::assembly::{
        CurvatureRoute, DerivativeOrder as O, SourceMode, SourceRequest,
    };
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let term = b
        .custom(&[x], |x, g| {
            g[0] = 1.;
            Ok(x[0])
        })
        .unwrap();
    b.factor(term).unwrap();
    let model = b.compile().unwrap();
    for mode in [SourceMode::Density, SourceMode::Objective] {
        for order in [O::Value, O::First, O::Hvp, O::Third] {
            let request = SourceRequest {
                mode,
                order,
                curvature: CurvatureRoute::OriginalCompilerReverse,
            };
            assert!(matches!(
                alea_autodiff::codegen::source(&model, request),
                Err(AssemblyError::MissingPrimal { .. })
            ));
        }
    }
}

#[test]
fn trusted_first_order_source_has_no_root_curvature_reverse() {
    use alea_core::model::assembly::{CurvatureRoute, DerivativeOrder, SourceMode, SourceRequest};
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap().expr();
    let sigma = b.positive("sigma").unwrap().expr();
    let y = b.constant(0.5).unwrap();
    let term = alea_distributions::modeling::normal(&mut b, y, mu, sigma).unwrap();
    b.factor(term).unwrap();
    let model = b.compile().unwrap();
    for mode in [SourceMode::Density, SourceMode::Objective] {
        let source = alea_autodiff::codegen::source(
            &model,
            SourceRequest {
                mode,
                order: DerivativeOrder::First,
                curvature: CurvatureRoute::Unavailable,
            },
        )
        .unwrap();
        assert!(source.contains("SOURCE_EXPLICIT_FIRST_ORDER: bool = true"));
        assert!(!source.contains("__higher_gradient"));
        assert!(!source.contains("autodiff_forward"));
        assert!(!source.contains("__hvp"));
        assert!(!source.contains("PotentialHessian"));
    }
}

#[test]
fn leaf_marker_text_is_preserved_across_every_entry_and_admission_route() {
    use alea_core::model::assembly::{
        CurvatureRoute, DerivativeOrder as O, SourceMode, SourceRequest,
    };
    const LEAF: &str = r#"let receipt = "__WORKSPACE__ __PUBLIC_TARGET__ __VALIDATION__ __TENSOR_SCORE_VALIDATION__ __CURVATURE_GRADIENT__ __FIRST_ORDER_SOURCE__"; let _=receipt; x[0]*x[0]"#;
    for trusted in [false, true] {
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let term = b
            .custom_with_primal(&[x], LEAF, |x, g| {
                g[0] = 2. * x[0];
                Ok(x[0] * x[0])
            })
            .unwrap();
        b.factor(term).unwrap();
        if trusted {
            let one = b.constant(1.).unwrap();
            let term = alea_distributions::modeling::normal(&mut b, x, 0., one).unwrap();
            b.factor(term).unwrap();
        }
        let model = b.compile().unwrap();
        for mode in [SourceMode::Density, SourceMode::Objective] {
            for order in [O::Value, O::First, O::Hvp, O::Third] {
                let source = alea_autodiff::codegen::source(
                    &model,
                    SourceRequest {
                        mode,
                        order,
                        curvature: CurvatureRoute::OriginalCompilerReverse,
                    },
                )
                .unwrap();
                assert!(source.contains(LEAF), "leaf changed for {mode:?}/{order:?}");
                assert_eq!(
                    source.contains("SOURCE_EXPLICIT_FIRST_ORDER: bool = true"),
                    trusted && order != O::Value
                );
                assert_eq!(
                    source.contains("pub(super) fn __hvp("),
                    matches!(order, O::Hvp | O::Third)
                );
            }
        }
    }
}
