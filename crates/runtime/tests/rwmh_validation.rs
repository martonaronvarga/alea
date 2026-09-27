use kernels::{dist::Gaussian, metric::IdentityMetric, state::ChainState};
use rand::{SeedableRng, rngs::SmallRng};
use runtime::{Rwmh, RwmhConfig, config::SamplerConfigError};

#[test]
fn configuration_rejects_bad_inputs_even_without_adaptation() {
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            Rwmh::<_, ChainState>::try_new(
                RwmhConfig::default().with_step_size(value),
                IdentityMetric::new(2)
            )
            .is_err()
        );
    }
    for value in [0.0, 1.0, f64::NAN, f64::INFINITY] {
        let config = RwmhConfig::default()
            .with_target_accept_rate(value)
            .with_adapt_step_size(false);
        assert!(matches!(
            Rwmh::<_, ChainState>::try_new(config, IdentityMetric::new(2)),
            Err(SamplerConfigError::AcceptanceTarget)
        ));
    }
    assert!(matches!(
        Rwmh::<_, ChainState>::try_new(RwmhConfig::default(), IdentityMetric::new(0)),
        Err(SamplerConfigError::EmptyDimension)
    ));
    assert!(matches!(
        Rwmh::<_, ChainState>::try_new(RwmhConfig::default(), IdentityMetric::new(usize::MAX)),
        Err(SamplerConfigError::DimensionOverflow)
    ));
}

#[test]
fn sampling_validates_public_mutations_and_sizes_before_state_or_rng_changes() {
    let mut sampler =
        Rwmh::<_, ChainState>::try_new(RwmhConfig::default(), IdentityMetric::new(2)).unwrap();
    let mut state = ChainState::new(1);
    let mut rng = SmallRng::seed_from_u64(92);
    let before = rng.clone();
    assert!(matches!(
        sampler.try_sample(&Gaussian, &mut state, &mut rng),
        Err(SamplerConfigError::StateDimension { .. })
    ));
    assert_eq!(rng, before);
    assert!(state.log_prob.is_nan());
    assert!(sampler.try_step(&mut state, &Gaussian, &mut rng).is_err());
    assert_eq!(rng, before);
    let mut state = ChainState::new(2);
    sampler.config.n_draws = usize::MAX;
    assert!(matches!(
        sampler.try_sample(&Gaussian, &mut state, &mut rng),
        Err(SamplerConfigError::DrawCountOverflow)
    ));
    assert_eq!(rng, before);
    assert!(state.log_prob.is_nan());
    sampler.config.step_size = f64::NAN;
    assert!(sampler.try_sample(&Gaussian, &mut state, &mut rng).is_err());
    assert_eq!(rng, before);
    assert!(state.log_prob.is_nan());
}

#[test]
fn valid_fallible_sampling_produces_expected_shape() {
    let config = RwmhConfig::default().with_warmup(4).with_draws(8);
    let mut sampler = Rwmh::try_new(config, IdentityMetric::new(2)).unwrap();
    let mut state = ChainState::new(2);
    let mut rng = SmallRng::seed_from_u64(93);
    let draws = sampler.try_sample(&Gaussian, &mut state, &mut rng).unwrap();
    assert_eq!((draws.n_draws(), draws.dim()), (8, 2));
    assert!(draws.row(7).iter().all(|x| x.is_finite()));
}
