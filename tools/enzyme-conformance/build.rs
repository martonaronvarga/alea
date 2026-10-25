#[path = "models.rs"]
mod models;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=models.rs");
    println!("cargo:rerun-if-changed=build.rs");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    for (name, model) in models::all()? {
        let source = alea_autodiff::codegen::enzyme_source(&model)?;
        std::fs::write(output.join(format!("{name}.rs")), source)?;
    }
    Ok(())
}
