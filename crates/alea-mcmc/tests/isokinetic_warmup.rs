use alea_distributions::StandardNormalTarget;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    adapt::{
        MclmcWarmupOptions, StaticWarmupOptions, warmup_mams_step_size, warmup_mclmc_step_size,
    },
    mams::{ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget, Mams, MamsOptions},
};
use rand::{SeedableRng, rngs::SmallRng};
#[test]
fn squared_local_correction_error_has_sixth_order_step_scaling() {
    let t = StandardNormalTarget::new(5);
    for refresh in [None, Some(5_f64.sqrt())] {
        let mut errors = Vec::new();
        for h in [0.02, 0.01, 0.005] {
            let mut sum = 0.;
            for seed in 0..128 {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mut s = ApproximateMclmc::new(
                    ApproximateTarget::acknowledge_bias(),
                    &t,
                    OwnedBuffer::from_fn(5, |i| 0.3 * (i + 1) as f64),
                    ApproximateMclmcOptions::new(h, 1, refresh).unwrap(),
                    &mut rng,
                )
                .unwrap();
                let o = s.step(&mut rng).unwrap();
                sum += o.correction_error.powi(2);
            }
            errors.push(sum / 128.);
        }
        for pair in errors.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!(
                (ratio / 64. - 1.).abs() < 0.06,
                "refresh={refresh:?}, ratio={ratio}"
            );
        }
    }
}
#[test]
fn corrected_and_approximate_warmup_return_frozen_steps_with_complete_work() {
    let t = StandardNormalTarget::new(4);
    let mut rng = SmallRng::seed_from_u64(18001);
    let s = Mams::new(&t, OwnedBuffer::new(4), MamsOptions::new(0.1, 8).unwrap()).unwrap();
    let (mut s, r) = warmup_mams_step_size(s, StaticWarmupOptions::new(200), &mut rng).unwrap();
    assert_eq!(r.integration_attempts, 1600);
    assert!(r.step_size.value() > 0.1);
    for _ in 0..20 {
        let _ = s.step(&mut rng).unwrap();
        assert_eq!(s.options().step_size(), r.step_size);
    }
    let s = ApproximateMclmc::new(
        ApproximateTarget::acknowledge_bias(),
        &t,
        OwnedBuffer::new(4),
        ApproximateMclmcOptions::new(0.1, 8, Some(2.)).unwrap(),
        &mut rng,
    )
    .unwrap();
    let (mut s, r) = warmup_mclmc_step_size(s, MclmcWarmupOptions::new(200), &mut rng).unwrap();
    assert_eq!(r.integration_attempts, 1600);
    assert_eq!(r.error_power, 6.);
    assert!(r.observed_error_per_dimension.is_finite());
    assert_eq!(s.options().refresh_length(), Some(2.));
    for _ in 0..20 {
        let _ = s.step(&mut rng).unwrap();
        assert_eq!(s.options().step_size(), r.step_size);
    }
}
