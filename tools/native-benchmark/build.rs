#[cfg(feature = "enzyme")]
#[path = "src/conformance.rs"]
mod conformance;

fn main() {
    println!("cargo:rerun-if-changed=src/conformance.rs");
    #[cfg(feature = "enzyme")]
    {
        let (x, y) = conformance::covariance_data();
        let model =
            conformance::build("covariance", 32, 2, &x, &y).expect("valid covariance fixture");
        let source =
            alea::autodiff::codegen::enzyme_source(&model).expect("supported native normals");
        let path = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
        std::fs::write(path.join("covariance.rs"), source).expect("write generated model");
    }
}
