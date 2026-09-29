use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Rwmh, RwmhOptions,
    adapt::{DualAveraging, warmup_rwmh},
    config::{AcceptanceTarget, StepSize},
};
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn dual_averaging_matches_original_recurrence_and_rejects_without_mutation() {
    let mut controller = DualAveraging::new(
        StepSize::try_from(0.3).unwrap(),
        AcceptanceTarget::try_from(0.234).unwrap(),
    );
    let (mut h, mut average) = (0.0, 0.3_f64.ln());
    for (i, acceptance) in [0.1, 0.5, 0.8, 0.0, 1.0].into_iter().enumerate() {
        let m = (i + 1) as f64;
        let weight = 1.0 / (m + 10.0);
        h = (1.0 - weight) * h + weight * (0.234 - acceptance);
        let log_step = 10.0_f64.ln() + 0.3_f64.ln() - m.sqrt() / 0.05 * h;
        let decay = m.powf(-0.75);
        average = decay * log_step + (1.0 - decay) * average;
        assert!((controller.update(acceptance).unwrap().value() - log_step.exp()).abs() < 1e-12);
    }
    for invalid in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        let mut copy = controller.clone();
        assert!(copy.update(invalid).is_err());
        assert!((copy.finish().value() - controller.clone().finish().value()).abs() < 1e-12);
    }
    assert!((controller.finish().value() - average.exp()).abs() < 1e-12);
}

#[test]
fn warmup_is_explicit_then_scale_is_frozen_and_draws_keep_shape() {
    let target = Gaussian::new(2);
    let mut chain = Rwmh::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        RwmhOptions::new(0.3).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(93);
    let before = rng.clone();
    let empty = warmup_rwmh(
        &mut chain,
        0,
        Some(AcceptanceTarget::try_from(0.234).unwrap()),
        &mut rng,
    )
    .unwrap();
    assert_eq!(empty.step_size.value(), 0.3);
    assert_eq!(empty.acceptance_rate(), 0.0);
    assert_eq!(rng, before);
    let fixed = warmup_rwmh(&mut chain, 4, None, &mut rng).unwrap();
    assert_eq!(fixed.step_size.value(), 0.3);
    let adapted = warmup_rwmh(
        &mut chain,
        4,
        Some(AcceptanceTarget::try_from(0.234).unwrap()),
        &mut rng,
    )
    .unwrap();
    assert_eq!(adapted.iterations, 4);
    for _ in 0..8 {
        let _ = chain.step(&mut rng).unwrap();
        assert_eq!(chain.options().step_size(), adapted.step_size);
        assert_eq!(chain.point().position().len(), 2);
        assert!(chain.point().position().iter().all(|x| x.is_finite()));
    }
}

#[test]
fn extreme_initial_scales_and_probabilities_remain_finite() {
    for initial in [f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX] {
        for acceptance in [0.0, 1.0] {
            let mut adapter =
                DualAveraging::new(initial.try_into().unwrap(), 0.8.try_into().unwrap());
            for _ in 0..100 {
                let value = adapter.update(acceptance).unwrap().value();
                assert!(value.is_finite() && value > 0.0);
            }
            assert!(adapter.finish().value().is_finite());
        }
    }
}
