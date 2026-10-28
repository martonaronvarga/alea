use alea::math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea::sampling::{
    Nuts,
    approximate::{ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget},
    config::TreeDepth,
};
use rand::{SeedableRng, rngs::SmallRng};

mod support;
use support::StandardNormalTarget;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = StandardNormalTarget::new(2);
    let mut rng = SmallRng::seed_from_u64(29);
    let approximate = ApproximateMclmc::new(
        ApproximateTarget::acknowledge_bias(),
        &target,
        OwnedBuffer::new(2),
        ApproximateMclmcOptions::new(0.1, 4, None)?,
        &mut rng,
    )?;
    let depth = TreeDepth::try_from(4)?;
    let _: Nuts<'_, StandardNormalTarget, IdentityMetric> = Nuts::from_hmc(approximate, depth)?;
    Ok(())
}
