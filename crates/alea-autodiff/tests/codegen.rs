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
