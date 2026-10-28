use alea::model::assembly::{ModelBuilder, Real, Simplex, VectorParameter};
use alea::transform::Transform;
fn requires_simplex(_: VectorParameter<Simplex>) {}
fn main() {
    let mut builder = ModelBuilder::new();
    let real = builder
        .vector::<Real>(
            "x",
            "coordinate",
            &["a", "b"],
            "x",
            Transform::identity(2).unwrap(),
        )
        .unwrap();
    requires_simplex(real);
}
