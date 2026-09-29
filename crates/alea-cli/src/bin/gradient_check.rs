#![cfg_attr(feature = "std-autodiff", feature(autodiff))]
#[path = "../../../alea-autodiff/examples/support/models.rs"]
mod models;
use alea_autodiff::gradient_check::{GradientCheckOptions, check_gradient};
use alea_core::model::TransformedTarget;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: alea-gradient-check [all|gaussian|correlated|logistic|banana|funnel|wiener] [relative-step]".into());
    }
    let selected = args.first().map(String::as_str).unwrap_or("all");
    if selected != "all" && selected != "wiener" && !models::NAMES.contains(&selected) {
        return Err(format!("unknown model: {selected}").into());
    }
    let options = GradientCheckOptions {
        relative_step: args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(1e-5),
        ..Default::default()
    };
    let mut passed = true;
    for (kind, name) in models::NAMES.into_iter().enumerate() {
        if selected == "all" || selected == name {
            let report = check_gradient(&models::analytic_target(kind), &[0.3, -0.7], options)?;
            println!(
                "{name}: {} {:#?}",
                if report.passed() { "PASS" } else { "FAIL" },
                report.components
            );
            passed &= report.passed();
        }
    }
    if selected == "all" || selected == "wiener" {
        let target = TransformedTarget::new(models::wiener_primitive(), models::wiener_layout())?;
        let report = check_gradient(&target, &[0.2, -0.5, 0.3, -0.4], options)?;
        println!(
            "wiener: {} {:#?}",
            if report.passed() { "PASS" } else { "FAIL" },
            report.components
        );
        passed &= report.passed();
    }
    if !passed {
        return Err("gradient validation failed".into());
    }
    Ok(())
}
