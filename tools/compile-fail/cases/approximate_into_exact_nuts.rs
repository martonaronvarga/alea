use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget, Nuts, config::TreeDepth,
};
use rand::{SeedableRng, rngs::SmallRng};

mod support;
use support::Gaussian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = Gaussian::new(2);
    let mut rng = SmallRng::seed_from_u64(29);
    let approximate = ApproximateMclmc::new(
        ApproximateTarget::acknowledge_bias(),
        &target,
        OwnedBuffer::new(2),
        ApproximateMclmcOptions::new(0.1, 4, None)?,
        &mut rng,
    )?;
    let depth = TreeDepth::try_from(4)?;
    let _: Nuts<'_, Gaussian, IdentityMetric> = Nuts::from_hmc(approximate, depth)?;
    Ok(())
}
