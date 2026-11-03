#[path = "models.rs"]
mod models;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=models.rs");
    println!("cargo:rerun-if-changed=build.rs");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    for (name, model) in models::all()? {
        if name.starts_with("tensor_") {
            let dimension = model.layout().unconstrained_dimension();
            let primal = format!(
                "pub fn log_density(q: &[f64; {dimension}]) -> f64 {{ {} }}",
                model.rust_primal()?
            );
            std::fs::write(output.join(format!("{name}_primal.rs")), primal)?;
        }
        let mut source = alea::autodiff::codegen::enzyme_source(&model)?;
        if name == "tensor_control" {
            // Test the raw nested JVP's zero/linearity laws, independently of
            // dense tensor assembly. This wrapper is tool-only, not public API.
            let dimension = model.layout().unconstrained_dimension();
            let (_, work) = model.rust_primal_with_workspace()?;
            source.push_str(&format!(r#"
pub fn hvp_position_jvp_for_test(q: &[f64; {dimension}], v: &[f64; {dimension}], w: &[f64; {dimension}], h: &mut [f64; {dimension}], t: &mut [f64; {dimension}]) {{
    h.fill(0.0); t.fill(0.0);
    let [a, da, b, db, c, dc, d, dd] = &mut [[0.0; {work}]; 8];
    __hvp_jvp(q, w, v, &[0.0; {dimension}], &mut [0.0; {dimension}], &mut [0.0; {dimension}], h, t, a, da, b, db, c, dc, d, dd);
}}
"#));
        }
        std::fs::write(output.join(format!("{name}.rs")), source)?;
        let objective = alea::autodiff::codegen::enzyme_objective_source(&model)?;
        std::fs::write(output.join(format!("{name}_objective.rs")), objective)?;
    }
    Ok(())
}
