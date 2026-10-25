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
