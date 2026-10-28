use alea_distributions::StandardNormalTarget;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Hmc, HmcOptions,
    adapt::{
        StaticWarmupOptions, warmup_drghmc_step_size, warmup_ghmc_step_size, warmup_hmc_step_size,
    },
    ghmc::{DrGhmc, DrGhmcOptions, Ghmc, GhmcOptions},
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[test]
fn fixed_geometry_warmup_freezes_each_static_kernel_and_accounts_for_ghost_work() {
    let target = StandardNormalTarget::new(2);
    let mut rng = SmallRng::seed_from_u64(18001);
    let h = Hmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        HmcOptions::new(0.1, 4).unwrap(),
    )
    .unwrap();
    let (mut h, r) = warmup_hmc_step_size(h, StaticWarmupOptions::new(64), &mut rng).unwrap();
    assert_eq!(r.iterations, 64);
    assert_eq!(r.integration_attempts, 256);
    assert!(r.accepted > 0);
    let step = h.options();
    for _ in 0..20 {
        let _ = h.step(&mut rng).unwrap();
    }
    assert_eq!(h.options(), step);
    let g = Ghmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        GhmcOptions::new(0.1, 4, 0.8).unwrap(),
        &mut rng,
    )
    .unwrap();
    let (mut g, r) = warmup_ghmc_step_size(g, StaticWarmupOptions::new(64), &mut rng).unwrap();
    assert_eq!(r.iterations, 64);
    assert_eq!(r.integration_attempts, 256);
    let step = g.options().step_size();
    for _ in 0..20 {
        let _ = g.step(&mut rng).unwrap();
    }
    assert_eq!(g.options().step_size(), step);
    let g = DrGhmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        DrGhmcOptions::new(0.1, 2., 0.8)
            .unwrap()
            .with_integration_steps(4)
            .unwrap(),
        &mut rng,
    )
    .unwrap();
    let (mut g, r) = warmup_drghmc_step_size(g, StaticWarmupOptions::new(64), &mut rng).unwrap();
    assert_eq!(r.iterations, 64);
    assert!(r.integration_attempts >= 256);
    assert!(r.integration_attempts <= 3 * 256);
    assert_eq!(
        r.integration_attempts,
        r.first_stage_integration_steps + r.delayed_integration_steps + r.ghost_integration_steps
    );
    assert!(r.delayed_integration_steps > 0);
    assert!(r.ghost_integration_steps > 0);
    let step = g.options().first_step();
    for _ in 0..20 {
        let _ = g.step(&mut rng).unwrap();
    }
    assert_eq!(g.options().first_step(), step);
}

#[test]
fn zero_static_warmup_preserves_persistent_momentum_and_random_stream() {
    let target = StandardNormalTarget::new(2);
    let mut rng = SmallRng::seed_from_u64(18001);
    let g = Ghmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        GhmcOptions::new(0.1, 4, 0.8).unwrap(),
        &mut rng,
    )
    .unwrap();
    let momentum = g.momentum().to_vec();
    let mut expected = rng.clone();
    let (g, r) = warmup_ghmc_step_size(g, StaticWarmupOptions::new(0), &mut rng).unwrap();
    assert_eq!(r.iterations, 0);
    assert_eq!(g.momentum(), momentum);
    assert_eq!(g.options().step_size().value(), 0.1);
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
}
