//! End-to-end native Gaussian fit -> separate intervals -> point forecasts -> JSON.
use alea_downstream_conformance::gaussian_workflow::{CASES, write_case};
use std::error::Error;
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let case = args.next().unwrap_or_else(|| "all".into());
    if args.next().is_some() {
        return Err("usage: gaussian_workflow [all|case]".into());
    }
    let cases = if case == "all" {
        CASES.to_vec()
    } else {
        vec![case.as_str()]
    };
    let mut out = std::io::stdout().lock();
    for case in cases {
        write_case(case, &mut out)?;
    }
    Ok(())
}
