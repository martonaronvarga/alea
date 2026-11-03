//! Emits one standalone hierarchy qualification bundle; no files are overwritten.
use alea_downstream_conformance::gaussian_hierarchy::report;
use std::{error::Error, io::Write};
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let case = args.next().unwrap_or_else(|| "complete".into());
    if args.next().is_some() {
        return Err("usage: gaussian_hierarchy [complete|cancelled|initial_transform]".into());
    }
    let report = report(&case)?;
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, &report)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
