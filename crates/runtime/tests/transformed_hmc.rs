use kernels::{
    buffer::OwnedBuffer,
    metric::IdentityMetric,
    model::{AnalyticModel, TransformedTarget},
    transform::{ParameterLayout, Transform},
};
use rand::{SeedableRng, rngs::SmallRng};
use runtime::mcmc::hmc_chain::{HmcChain, HmcOptions};
use std::convert::Infallible;

#[test]
#[cfg_attr(miri, ignore = "statistical integration gate runs natively")]
fn exponential_and_beta_moments_through_the_transformed_boundary() {
    let model = AnalyticModel::new(2, |x: &[f64], g: &mut [f64]| {
        g[0] = -1.0;
        g[1] = 1.0 / x[1] - 2.0 / (1.0 - x[1]);
        Ok::<_, Infallible>(-x[0] + x[1].ln() + 2.0 * (-x[1]).ln_1p())
    });
    let layout = ParameterLayout::new([
        Transform::positive(1).unwrap(),
        Transform::interval(1, 0.0, 1.0).unwrap(),
    ])
    .unwrap();
    let target = TransformedTarget::new(model, layout).unwrap();
    let mut chain = HmcChain::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        HmcOptions::new(0.3, 7).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(0x4d32);
    let mut x = [0.0; 2];
    let mut scratch = [0.0; 1];
    let mut sum = [0.0; 2];
    let mut second = [0.0; 2];
    for i in 0..22_000 {
        let _ = chain.step(&mut rng).unwrap();
        target
            .layout()
            .constrain(chain.point().position(), &mut x, &mut scratch)
            .unwrap();
        if i >= 2000 {
            for j in 0..2 {
                sum[j] += x[j];
                second[j] += x[j] * x[j];
            }
        }
    }
    assert!((sum[0] / 20_000.0 - 1.0).abs() < 0.07, "{sum:?}");
    assert!((second[0] / 20_000.0 - 2.0).abs() < 0.2, "{second:?}");
    assert!((sum[1] / 20_000.0 - 0.4).abs() < 0.02, "{sum:?}");
    assert!((second[1] / 20_000.0 - 0.2).abs() < 0.02, "{second:?}");
    let allocations = allocation_counter::measure(|| {
        for _ in 0..128 {
            let _ = chain.step(&mut rng).unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}
