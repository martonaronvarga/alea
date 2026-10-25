use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=native-template.txt");
    println!("cargo:rerun-if-env-changed=ALEA_NATIVE_PORTS");
    if env::var_os("CARGO_FEATURE_NATIVE_ENZYME").is_none() {
        return Ok(());
    }
    let directory =
        PathBuf::from(env::var_os("ALEA_NATIVE_PORTS").ok_or("missing ALEA_NATIVE_PORTS")?);
    let inventory = directory.join("models.json");
    println!("cargo:rerun-if-changed={}", inventory.display());
    let inventory: serde_json::Value = serde_json::from_slice(&fs::read(inventory)?)?;
    let output = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    let mut source = String::from(
        "type Native = Box<dyn alea_core::capability::PotentialHessianDerivative<Error=alea_autodiff::autodiff::AutodiffError> + Send + Sync>;\n",
    );
    let mut arms = String::new();
    for row in inventory["records"]
        .as_array()
        .ok_or("invalid native inventory")?
    {
        let module = row["module"].as_str().ok_or("missing module")?;
        let case = row["case"].as_str().ok_or("missing case")?;
        let dimension = row["dimension"].as_u64().ok_or("missing dimension")? as usize;
        let primal_path = directory.join(row["primal"].as_str().ok_or("missing primal")?);
        println!("cargo:rerun-if-changed={}", primal_path.display());
        let primal = Box::leak(fs::read_to_string(primal_path)?.into_boxed_str());
        let mut builder = alea_core::model::assembly::ModelBuilder::new();
        let mut coordinates = Vec::with_capacity(dimension);
        for i in 0..dimension {
            coordinates.push(builder.real(&format!("q{i}"))?.expr());
        }
        // This build-only graph exports the trusted primal. It is never an
        // analytic-tape backend; accidentally evaluating it fails explicitly.
        let factor = builder.custom_with_primal(&coordinates, primal, |_, _| {
            Err(alea_core::model::assembly::AssemblyError::Factor(
                "source-only benchmark model; evaluate generated Enzyme target".into(),
            ))
        })?;
        builder.factor(factor)?;
        let model = builder.compile()?;
        if model.layout().unconstrained_dimension() != dimension {
            return Err("native layout mismatch".into());
        }
        // Specialize the single full-coordinate factor: borrow the original
        // array directly instead of copying q through a local argument array.
        // The adapter admits finite inputs/outputs outside differentiated code.
        let primal = format!("let x = q; {primal}");
        let code = include_str!("native-template.txt")
            .replace("__DIMENSION__", &dimension.to_string())
            .replace("__PRIMAL__", &primal);
        fs::write(output.join(format!("{module}.rs")), code)?;
        source.push_str(&format!("#[allow(dead_code,unused_mut,non_snake_case)] mod {module} {{ include!(concat!(env!(\"OUT_DIR\"),\"/{module}.rs\")); }}\n"));
        arms.push_str(&format!(
            "{case:?} => Some(Box::new({module}::target())),\n"
        ));
    }
    source.push_str(&format!(
        "pub fn open(case: &str) -> Option<Native> {{ match case {{ {arms} _ => None }} }}\n"
    ));
    fs::write(output.join("native.rs"), source)?;
    Ok(())
}
