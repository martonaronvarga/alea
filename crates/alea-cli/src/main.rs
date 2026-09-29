use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Hmc, HmcOptions};
use rand::{SeedableRng, rngs::SmallRng};
use std::ops::ControlFlow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = Gaussian::new(4);
    let mut chain = Hmc::new(
        &target,
        OwnedBuffer::new(4),
        IdentityMetric::new(4),
        HmcOptions::new(0.1, 8)?,
    )?;
    let mut rng = SmallRng::seed_from_u64(42);
    let mut accepted = 0;
    let draws = alea_runtime::run_chain(&mut chain, 1000, &mut rng, |_, transition| {
        accepted += usize::from(transition.accepted);
        ControlFlow::Continue(())
    })?;
    println!(
        "draws={draws} accepted={accepted} log_density={}",
        chain.point().log_density()
    );
    Ok(())
}
