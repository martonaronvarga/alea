//! Public native GLM point fits with explicit qualification and non-success cases.
use alea_downstream_conformance::glm_workflow::{CASES, FAMILIES, write_case};
use std::error::Error;
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut out = std::io::stdout().lock();
    match args.as_slice() {
        [] => {
            for family in FAMILIES {
                for case in CASES {
                    write_case(family, case, &mut out)?;
                }
            }
        }
        [family, case] => write_case(family, case, &mut out)?,
        _ => return Err("usage: glm_workflow [logistic|poisson CASE]".into()),
    }
    Ok(())
}
