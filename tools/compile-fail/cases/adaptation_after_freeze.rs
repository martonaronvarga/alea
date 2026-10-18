use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    HmcOptions,
    adapt::{HmcWarmup, WarmupOptions},
};
use rand::{SeedableRng, rngs::SmallRng};

mod support;
use support::Gaussian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = Gaussian::new(2);
    let warmup = HmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        HmcOptions::new(0.1, 4)?,
        WarmupOptions::new(20),
    )?;
    let mut rng = SmallRng::seed_from_u64(17);
    let _frozen = warmup.run(&mut rng)?;
    let _invalid_second_adaptation = warmup.run(&mut rng)?;
    Ok(())
}
