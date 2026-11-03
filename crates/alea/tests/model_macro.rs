//! The model macro and arithmetic do not require the distributions feature.
#![allow(clippy::needless_borrows_for_generic_args)] // borrowed input contract
use alea::model::assembly::AssemblyError;

#[test]
fn core_only_macro_uses_checked_builder_without_optional_features() {
    let model = alea::model! { |b| parameters { x: real() } {
        let square = b.multiply(&x,&x)?;
        let factor = b.multiply(-0.5,square)?;
        b.prior("normal up to constant",factor)?;
    }}
    .unwrap();
    assert_eq!(model.coordinates()[0].name, "x");
    let failed = alea::model! { |b| parameters { x: real() } {
        let _invalid = b.add(&x,f64::NAN)?;
    }};
    assert!(matches!(failed, Err(AssemblyError::Numerical)));
}
