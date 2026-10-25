use alea_core::model::assembly::ModelBuilder;
use alea_distributions::modeling::normal_observation;
fn main() {
    let mut builder = ModelBuilder::new();
    let location = builder.real("location").unwrap();
    let invalid_scale = builder.real("scale").unwrap();
    let _ = normal_observation(&mut builder, 0.0, location.expr(), &invalid_scale);
}
