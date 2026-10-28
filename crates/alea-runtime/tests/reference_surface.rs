use alea_core::prelude::{EvaluationWorkspace, LogDensityGradient, Transform};
use alea_distributions::StandardNormalTarget;
use alea_math::prelude::{EuclideanMetric, IdentityMetric, OwnedBuffer};
use alea_mcmc::prelude::{
    Hmc, HmcOptions, HmcWarmup, MarkovChain, Nuts, Rwmh, RwmhOptions, TreeDepth, WarmupOptions,
};
use alea_runtime::prelude::{collect_draws, run_chain};
use rand::{SeedableRng, rngs::SmallRng};
use std::ops::ControlFlow;

#[test]
fn convenience_preludes_form_an_end_to_end_inference_path() -> Result<(), Box<dyn std::error::Error>>
{
    let target = StandardNormalTarget::new(2);
    let mut gradient = [0.0; 2];
    target.logp_grad(&[0.0; 2], &mut gradient)?;
    let _workspace = EvaluationWorkspace::new(2);
    let _identity_transform = Transform::identity(2)?;

    let metric = IdentityMetric::new(2);
    assert_eq!(metric.dimension(), 2);
    let warmup = HmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        HmcOptions::new(0.1, 4)?,
        WarmupOptions::new(0),
    )?;
    let mut rng = SmallRng::seed_from_u64(41);
    let (hmc, report) = warmup.run(&mut rng)?;
    assert_eq!(report.iterations, 0);

    let depth = TreeDepth::try_from(4)?;
    let mut nuts = Nuts::from_hmc(hmc, depth)?;
    fn assert_reference_chain<C: MarkovChain>(_: &C) {}
    assert_reference_chain(&nuts);
    let observed = run_chain(&mut nuts, 2, &mut rng, |position, transition| {
        assert_eq!(position.len(), 2);
        assert!(transition.integration_steps <= depth.max_integration_steps());
        ControlFlow::Continue(())
    })?;
    assert_eq!(observed, 2);

    let mut fixed_hmc = Hmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        HmcOptions::new(0.1, 2)?,
    )?;
    assert!(fixed_hmc.step(&mut rng)?.integration_steps <= 2);

    let mut rwmh = Rwmh::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        RwmhOptions::new(0.2)?,
    )?;
    let draws = collect_draws(&mut rwmh, 3, &mut rng)?;
    assert_eq!(draws.len(), 3);
    assert_eq!(draws.dimension(), 2);
    Ok(())
}
