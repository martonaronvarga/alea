use alea_core::model::assembly::{AssemblyError, ModelBuilder};

#[test]
fn source_rejects_opaque_nodes_without_silently_using_their_vjp() {
    use alea_core::target::LogDensityGradient;
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let f = b
        .custom(&[x], |x, g| {
            g[0] = 2. * x[0];
            Ok(x[0] * x[0])
        })
        .unwrap();
    b.factor(f).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.source_capabilities().missing_primals, [f]);
    assert!(!model.source_capabilities().supports_rust_primal());
    assert!(matches!(
        model.rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    assert!(matches!(
        model.rust_objective_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    let native = model.into_target().unwrap();
    let mut g = [99.];
    assert_eq!(native.logp_grad(&[3.], &mut g).unwrap(), 9.);
    assert_eq!(g, [6.]);
    assert!(native.logp_grad(&[f64::NAN], &mut g).is_err());
    assert_eq!(g, [6.]);
    assert_eq!(native.logp_grad(&[-2.], &mut g).unwrap(), 4.);
    assert_eq!(g, [-4.]);
}

#[test]
fn selected_out_missing_source_and_invalid_scores_do_not_poison_an_aliased_factor() {
    use alea_core::target::LogDensityGradient;
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let a = b
        .custom_with_primal(&[x, x], "x[0]*x[1]*x[0]", |x, g| {
            g[0] = 2. * x[0] * x[1];
            g[1] = x[0] * x[0];
            Ok(x[0] * x[1] * x[0])
        })
        .unwrap();
    b.likelihood("A", a).unwrap();
    let missing = b
        .custom(&[x], |_, g| {
            g[0] = f64::NAN;
            Ok(1.) // Finite primal is not sufficient admission.
        })
        .unwrap();
    b.prior("B", missing).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.source_capabilities().missing_primals, [missing]);
    assert!(matches!(
        model.rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    let native = model.into_target().unwrap();
    let mut gradient = [41.];
    assert!(native.logp_grad(&[0.5], &mut gradient).is_err());
    assert_eq!(gradient, [41.]);
    let selected = native.into_model().select_likelihood().unwrap();
    assert!(selected.source_capabilities().supports_rust_primal());
    assert!(selected.rust_primal().is_ok());
    assert!(selected.rust_first_order_gradient(true).unwrap().is_none());
    let target = selected.into_target().unwrap();
    for x in [-2., 0., 0.5, 2.] {
        // Dyadic inputs make these independent polynomial answers exact.
        assert_eq!(target.logp_grad(&[x], &mut gradient).unwrap(), x * x * x);
        assert_eq!(gradient, [3. * x * x]);
    }
    let before = gradient;
    assert!(target.logp_grad(&[f64::NAN], &mut gradient).is_err());
    assert_eq!(gradient, before);
    assert_eq!(target.logp_grad(&[0.5], &mut gradient).unwrap(), 0.125);
    assert_eq!(gradient, [0.75]);
}

#[test]
fn emission_is_deterministic_preserves_shared_factors_and_ignores_metadata_as_code() {
    fn source() -> String {
        let mut b = ModelBuilder::new();
        let x = b
            .positive("not a Rust identifier\"; panic!()")
            .unwrap()
            .expr();
        let f = b.multiply(x, x).unwrap();
        b.factor(f).unwrap();
        b.factor(f).unwrap();
        b.compile().unwrap().rust_primal().unwrap()
    }
    let source1 = source();
    assert_eq!(source1, source());
    assert_eq!(source1.matches("jacobian +=").count(), 1);
    assert_eq!(source1.matches("total += n1;").count(), 2);
    assert!(!source1.contains("panic!"));
}

#[test]
fn objective_source_has_no_density_jacobian_or_unused_log_products() {
    use alea_core::model::assembly::DynamicDomain;
    use alea_core::transform::Transform;
    for transform in [
        Transform::positive(2),
        Transform::interval(2, -2., 3.),
        Transform::simplex(3),
        Transform::unit_vector(3),
        Transform::ordered(2),
        Transform::positive_ordered(2),
        Transform::covariance(3),
        Transform::correlation(3),
        Transform::cholesky_covariance(3),
        Transform::cholesky_correlation(3),
    ] {
        let mut b = ModelBuilder::new();
        let transform = transform.unwrap();
        let names: Vec<_> = (0..transform.constrained_dimension())
            .map(|i| i.to_string())
            .collect();
        let names: Vec<_> = names.iter().map(String::as_str).collect();
        let p = b
            .vector::<DynamicDomain>("x", "coordinate", &names, "x", transform)
            .unwrap();
        b.factor(p.expressions()[0]).unwrap();
        let model = b.compile().unwrap();
        let source = model.rust_objective_primal().unwrap();
        assert!(!source.contains("jac"));
        assert!(!source.contains("lp0"));
        assert!(source.ends_with("total\n"));
        if transform.domain() == alea_core::transform::TransformDomain::PositiveDefinite
            || transform.domain() == alea_core::transform::TransformDomain::Correlation
        {
            assert!(
                model
                    .rust_objective_validation()
                    .contains("constrain_values")
            );
        }
    }
}
