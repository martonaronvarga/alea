use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Hmc, HmcOptions, Nuts, NutsOptions,
    adapt::{
        NutsWarmup, NutsWarmupError, SearchOptions, WarmupOptions, find_reasonable_nuts_step_size,
        find_reasonable_step_size,
    },
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible};

struct Normal {
    mean: f64,
    scale: f64,
    calls: Cell<usize>,
}
impl alea_core::target::TargetSpace for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Normal {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        self.calls.set(self.calls.get() + 1);
        let z = (q[0] - self.mean) / self.scale;
        g[0] = -z / self.scale;
        Ok(-0.5 * z * z)
    }
}
fn target() -> Normal {
    Normal {
        mean: 1.,
        scale: 0.001,
        calls: Cell::new(0),
    }
}

#[test]
fn downhill_search_preserves_live_cache_and_reaches_the_ordinary_stability_scale() {
    let t = target();
    let mut nuts = Nuts::new(
        &t,
        OwnedBuffer::new(1),
        IdentityMetric::new(1),
        NutsOptions::new(0.001, 8).unwrap(),
    )
    .unwrap();
    let before = (
        nuts.point().position()[0],
        nuts.point().log_density(),
        nuts.point().gradient()[0],
    );
    let result = find_reasonable_nuts_step_size(
        &mut nuts,
        SearchOptions::default(),
        &mut SmallRng::seed_from_u64(18001),
    )
    .unwrap();
    assert!(result.step_size.value() >= 0.001);
    assert_eq!(nuts.options().step_size().value(), 0.001);
    assert_eq!(
        before,
        (
            nuts.point().position()[0],
            nuts.point().log_density(),
            nuts.point().gradient()[0]
        )
    );
    assert_eq!(t.calls.get(), result.probes + 1);
    let mut hmc = Hmc::new(
        &t,
        OwnedBuffer::new(1),
        IdentityMetric::new(1),
        HmcOptions::new(0.001, 8).unwrap(),
    )
    .unwrap();
    let old = find_reasonable_step_size(
        &mut hmc,
        SearchOptions::default(),
        &mut SmallRng::seed_from_u64(18001),
    )
    .unwrap();
    assert!(old.step_size.value() < result.step_size.value() / 2.);
}

#[test]
fn native_warmup_escapes_a_distant_sharply_scaled_start_and_freezes() {
    let t = target();
    let mut rng = SmallRng::seed_from_u64(18001);
    let mut observations = 0;
    let (mut nuts, report) = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 8).unwrap(),
        WarmupOptions::new(512),
    )
    .unwrap()
    .run_observed(&mut rng, |o| {
        observations += 1;
        assert_eq!(o.completed, observations);
        assert!(o.log_density.is_finite());
    })
    .unwrap();
    assert_eq!(observations, 512);
    assert_eq!(report.iterations, 512);
    assert!(report.metric_updates > 0);
    assert!(report.final_log_density > report.initial_log_density + 100_000.);
    let frozen = nuts.options();
    let mut sum = 0.;
    let mut square = 0.;
    for _ in 0..2048 {
        let _ = nuts.step(&mut rng).unwrap();
        let z = (nuts.point().position()[0] - t.mean) / t.scale;
        sum += z;
        square += z * z;
    }
    assert!(sum.abs() / 2048. < 0.1, "mean {}", sum / 2048.);
    assert!(
        (square / 2048. - 1.).abs() < 0.15,
        "second moment {}",
        square / 2048.
    );
    assert_eq!(nuts.options(), frozen);
}

#[test]
fn gradient_diagonal_recovers_quadratic_geometry_and_freezes_before_retention() {
    use alea_mcmc::adapt::{WarmupMetric, WarmupStage};
    let t = target();
    let mut rng = SmallRng::seed_from_u64(43001);
    let mut final_fast = 0;
    let (mut chain, report) = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 8).unwrap(),
        WarmupOptions::new(512),
    )
    .unwrap()
    .with_gradient_diagonal()
    .unwrap()
    .run_observed(&mut rng, |o| {
        if o.stage == WarmupStage::FinalFast {
            final_fast += 1;
        }
    })
    .unwrap();
    assert_eq!(report.iterations, 512);
    assert_eq!(final_fast, 512 - 512 * 85 / 100);
    let WarmupMetric::InverseDiagonal(metric) = chain.metric() else {
        panic!("diagonal profile")
    };
    assert!((metric.as_slice()[0] / t.scale.powi(2) - 1.0).abs() < 1e-10);
    let frozen = chain.metric().clone();
    let step = chain.options();
    let (mut sum, mut square) = (0.0, 0.0);
    for _ in 0..4096 {
        let _ = chain.step(&mut rng).unwrap();
        let z = (chain.point().position()[0] - t.mean) / t.scale;
        sum += z;
        square += z * z;
    }
    assert!((sum / 4096.0_f64).abs() < 0.1);
    assert!((square / 4096.0 - 1.0).abs() < 0.15);
    assert_eq!(chain.metric(), &frozen);
    assert_eq!(chain.options(), step);
}

#[test]
fn gradient_zero_warmup_and_presearch_cancellation_preserve_rng_and_cache() {
    let t = target();
    let mut rng = SmallRng::seed_from_u64(43001);
    let mut expected = rng.clone();
    let (chain, report) = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 4).unwrap(),
        WarmupOptions::new(0),
    )
    .unwrap()
    .with_gradient_diagonal()
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(t.calls.get(), 1);
    assert_eq!(report.iterations, 0);
    assert_eq!(chain.options().step_size().value(), 0.1);
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
    let result = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 4).unwrap(),
        WarmupOptions::new(512),
    )
    .unwrap()
    .with_gradient_diagonal()
    .unwrap()
    .run_controlled(&mut rng, |_| false)
    .unwrap();
    assert!(result.is_none());
    assert_eq!(t.calls.get(), 2);
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
}

#[test]
fn gradient_warmup_reuses_storage_and_observation_does_not_change_results() {
    let t = alea_distributions::StandardNormalTarget::new(4);
    for (jitter, restart, position) in [(0.0, false), (0.0, true), (0.1, false), (0.1, true)]
        .into_iter()
        .flat_map(|(j, r)| [(j, r, false), (j, r, true)])
    {
        let build = || {
            NutsWarmup::new(
                &t,
                OwnedBuffer::from_fn(4, |i| i as f64),
                NutsOptions::new(0.1, 6)
                    .unwrap()
                    .with_step_size_jitter(jitter)
                    .unwrap()
                    .with_position_turning(position),
                WarmupOptions::new(512),
            )
            .unwrap()
            .with_gradient_diagonal()
            .unwrap()
            .with_gradient_final_restart(restart)
        };
        let mut warmup = Some(build());
        let mut rng = SmallRng::seed_from_u64(43001);
        let mut observed_rng = rng.clone();
        let mut result = None;
        let allocations = allocation_counter::measure(|| {
            result = Some(warmup.take().unwrap().run(&mut rng).unwrap());
        });
        assert_eq!(allocations.count_total, 0);
        let (chain, report) = result.unwrap();
        assert_eq!(chain.options().position_turning(), position);
        let mut count = 0;
        let (observed, observed_report) = build()
            .run_observed(&mut observed_rng, |o| {
                count += 1;
                assert_eq!(count, o.completed);
            })
            .unwrap();
        assert_eq!(count, 512);
        assert_eq!(report, observed_report);
        assert_eq!(chain.point().position(), observed.point().position());
        assert_eq!(chain.point().gradient(), observed.point().gradient());
        assert_eq!(chain.options(), observed.options());
        assert_eq!(rng.random::<u64>(), observed_rng.random::<u64>());
    }
}

#[test]
fn zero_warmup_and_presearch_cancellation_consume_no_rng_or_extra_calls() {
    let t = target();
    let mut rng = SmallRng::seed_from_u64(2);
    let mut expected = rng.clone();
    let (chain, report) = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(f64::from_bits(1), 4).unwrap(),
        WarmupOptions::new(0),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(chain.options().step_size().value().to_bits(), 1);
    assert_eq!(report.iterations, 0);
    assert_eq!(t.calls.get(), 1);
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
    let result = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 4).unwrap(),
        WarmupOptions::new(512),
    )
    .unwrap()
    .run_controlled(&mut rng, |_| false)
    .unwrap();
    assert!(result.is_none());
    assert_eq!(t.calls.get(), 2);
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
}

#[test]
fn bounded_exhaustion_is_explicit() {
    let t = target();
    let search = SearchOptions::new(1e-20, 1e-19, 2, 0.8).unwrap();
    let result = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 4).unwrap(),
        WarmupOptions::new(1).with_search(Some(search)),
    )
    .unwrap()
    .run(&mut SmallRng::seed_from_u64(2));
    assert!(matches!(result, Err(NutsWarmupError::SearchExhausted)));
}

#[test]
fn frozen_conversion_preserves_cache_and_costs_no_evaluation_or_rng() {
    let t = target();
    let (chain, _) = NutsWarmup::new(
        &t,
        OwnedBuffer::new(1),
        NutsOptions::new(0.1, 8).unwrap(),
        WarmupOptions::new(128),
    )
    .unwrap()
    .run(&mut SmallRng::seed_from_u64(18001))
    .unwrap();
    let before = (
        chain.point().position().to_vec(),
        chain.point().gradient().to_vec(),
        chain.point().log_density(),
        chain.options(),
    );
    let calls = t.calls.get();
    let hmc = chain.into_hmc(8.try_into().unwrap());
    assert_eq!(t.calls.get(), calls);
    assert_eq!(hmc.point().position(), before.0);
    assert_eq!(hmc.point().gradient(), before.1);
    assert_eq!(hmc.point().log_density(), before.2);
    assert_eq!(hmc.options().step_size(), before.3.step_size());
}

#[test]
fn fisher_native_nuts_uses_paired_live_states_and_freezes_all_optimizer_modes() {
    use alea_mcmc::adapt::{DiminishingSchedule, FisherNutsWarmup, FisherOptions, StepAdaptation};
    let t = alea_distributions::StandardNormalTarget::new(2);
    for optimizer in [
        StepAdaptation::default(),
        StepAdaptation::robbins_monro(0.1, DiminishingSchedule::new(10., 0.6).unwrap()).unwrap(),
        StepAdaptation::adam(
            0.05,
            DiminishingSchedule::new(10., 0.6).unwrap(),
            0.9,
            0.999,
            1e-8,
        )
        .unwrap(),
    ] {
        let options = FisherOptions::new(256).with_step_adaptation(optimizer);
        let mut rng = SmallRng::seed_from_u64(18001);
        let (mut chain, r) = FisherNutsWarmup::new(
            &t,
            OwnedBuffer::from_fn(2, |_| 10.),
            NutsOptions::new(0.1, 8).unwrap(),
            options,
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        assert_eq!(r.iterations, 256);
        assert!(r.metric_updates > 0);
        let step = chain.options();
        let mut sum = [0.; 2];
        for _ in 0..1024 {
            let _ = chain.step(&mut rng).unwrap();
            for (a, q) in sum.iter_mut().zip(chain.point().position()) {
                *a += q;
            }
        }
        assert!(sum.iter().all(|x| x.abs() / 1024. < 0.2));
        assert_eq!(chain.options(), step);
    }
}

#[test]
fn short_native_warmup_allocates_no_transition_or_observation_storage() {
    let t = alea_distributions::StandardNormalTarget::new(2);
    let mut rng = SmallRng::seed_from_u64(18001);
    let warmup = NutsWarmup::new(
        &t,
        OwnedBuffer::new(2),
        NutsOptions::new(0.1, 4).unwrap(),
        WarmupOptions::new(16).with_search(None),
    )
    .unwrap();
    let mut result = None;
    let mut warmup = Some(warmup);
    let allocation = allocation_counter::measure(|| {
        result = Some(warmup.take().unwrap().run(&mut rng).unwrap());
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(result.unwrap().1.iterations, 16);
}

#[test]
fn native_warmup_preserves_concrete_backend_errors() {
    #[derive(Debug, thiserror::Error)]
    #[error("probe backend failure")]
    struct Fault;
    struct Failing(Cell<usize>);
    impl alea_core::target::TargetSpace for Failing {
        type Error = Fault;
        fn dimension(&self) -> usize {
            1
        }
    }
    impl LogDensityGradient for Failing {
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
            self.0.set(self.0.get() + 1);
            g[0] = -q[0];
            if self.0.get() == 3 {
                Err(Fault)
            } else {
                Ok(-0.5 * q[0] * q[0])
            }
        }
    }
    for gradient in [false, true] {
        let t = Failing(Cell::new(0));
        let warmup = NutsWarmup::new(
            &t,
            OwnedBuffer::new(1),
            NutsOptions::new(0.1, 4).unwrap(),
            WarmupOptions::new(16).with_search(None),
        )
        .unwrap();
        let warmup = if gradient {
            warmup.with_gradient_diagonal().unwrap()
        } else {
            warmup
        };
        let result = warmup.run(&mut SmallRng::seed_from_u64(18001));
        assert!(matches!(
            result,
            Err(NutsWarmupError::Nuts(alea_mcmc::nuts::NutsError::Phase {
                source: alea_mcmc::hamiltonian::PhaseError::Evaluation(
                    alea_core::target::EvaluationError::Model(Fault)
                ),
                ..
            }))
        ));
        assert_eq!(t.0.get(), 3);
    }
}

#[cfg(feature = "faer")]
#[test]
fn rank_capped_native_fisher_freezes_its_selected_rank() {
    use alea_mcmc::adapt::{FisherNutsWarmup, FisherOptions};
    let t = alea_distributions::StandardNormalTarget::new(4);
    let mut rng = SmallRng::seed_from_u64(18001);
    let (mut s, r) = FisherNutsWarmup::new(
        &t,
        OwnedBuffer::from_fn(4, |i| i as f64),
        NutsOptions::new(0.1, 8).unwrap(),
        FisherOptions::new(256).with_max_rank(2),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(r.iterations, 256);
    let frozen = s.options();
    let rank = s.metric().rank();
    for _ in 0..64 {
        let _ = s.step(&mut rng).unwrap();
    }
    assert_eq!(s.options(), frozen);
    assert_eq!(s.metric().rank(), rank);
}
