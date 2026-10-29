use alea_core::model::assembly::{AssemblyError, ModelBuilder};

#[test]
fn source_rejects_opaque_nodes_without_silently_using_their_vjp() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let f = b
        .custom(&[x], |x, g| {
            g[0] = 1.0;
            Ok(x[0])
        })
        .unwrap();
    b.factor(f).unwrap();
    assert!(matches!(
        b.compile().unwrap().rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
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
