use alea::distributions::modeling::ModelBuilderExt;
fn main() {
    let _model = alea::model! { |b| parameters { mu: real(), sigma: real() } {
        b.normal_observations("data", &[1.,2.], mu, &sigma)?;
    }};
}
