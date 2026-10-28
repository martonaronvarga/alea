use alea::math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea::sampling::{
    Hmc, HmcOptions,
    integrator::TwoStage,
    walnuts::{Walnuts, WalnutsOptions},
};

mod support;
use support::StandardNormalTarget;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = StandardNormalTarget::new(2);
    let hmc = Hmc::new_with_integrator(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        HmcOptions::new(0.1, 4)?,
        TwoStage::OMF,
    )?;
    let options = WalnutsOptions::new(0.1, 4)?;
    let _: Walnuts<'_, StandardNormalTarget, IdentityMetric> = Walnuts::from_hmc(hmc, options)?;
    Ok(())
}
