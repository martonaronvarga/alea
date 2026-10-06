use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use alea_mcmc::{
    HmcOptions,
    adapt::{
        DiminishingSchedule, DualAveraging, FisherHmcWarmup, FisherOptions, StepAdaptation,
        StepObservation, StepSizeController, WeightedFisherMoments,
    },
    hmc::Divergence,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-12 * (1.0 + b.abs()), "{a} != {b}");
}
fn settings() -> [StepAdaptation; 3] {
    let schedule = DiminishingSchedule::new(1.0, 0.6).unwrap();
    [
        StepAdaptation::default(),
        StepAdaptation::robbins_monro(1.0, schedule).unwrap(),
        StepAdaptation::adam(0.3, schedule, 0.9, 0.999, 1e-8).unwrap(),
    ]
}

#[test]
fn weighted_stream_matches_explicit_batch_weights_after_every_observation() {
    for exponent in [0.5001, 0.75, 1.0] {
        let schedule = DiminishingSchedule::new(2.0, exponent).unwrap();
        let mut moments = WeightedFisherMoments::new(2, schedule).unwrap();
        let mut weights = Vec::new();
        let mut observations = Vec::new();
        for t in 0..50 {
            let q = [(t as f64 * 0.7).sin() * 3.0, (t % 4) as f64];
            let score = [-q[0] * 0.25, 1.0 - q[1] * 2.0];
            let alpha = if t == 0 { 1.0 } else { schedule.weight(t) };
            for w in &mut weights {
                *w *= 1.0 - alpha;
            }
            weights.push(alpha);
            observations.push([q[0], q[1], score[0], score[1]]);
            moments.observe(&q, &score).unwrap();
            let means: [f64; 4] = std::array::from_fn(|j| {
                weights
                    .iter()
                    .zip(&observations)
                    .map(|(w, x)| w * x[j])
                    .sum()
            });
            let scatters: [f64; 4] = std::array::from_fn(|j| {
                weights
                    .iter()
                    .zip(&observations)
                    .map(|(w, x)| w * (x[j] - means[j]).powi(2))
                    .sum()
            });
            for j in 0..4 {
                close(moments.moments()[j], means[j]);
                close(moments.moments()[j + 4], scatters[j]);
            }
            close(
                moments.squared_weights(),
                weights.iter().map(|w| w * w).sum(),
            );
            assert_eq!(moments.count(), t + 1);
            let mut scales = [0.0; 2];
            moments.scales_into(&[2.0, 0.5], 0.01, &mut scales).unwrap();
            for (i, s) in [2.0_f64, 0.5].into_iter().enumerate() {
                close(
                    scales[i],
                    s * ((scatters[i] / s.powi(2) + 0.01) / (scatters[i + 2] * s.powi(2) + 0.01))
                        .powf(0.25),
                );
            }
        }
        let before = moments.moments().to_vec();
        let weights = moments.squared_weights();
        for q in [&[0.0][..], &[f64::NAN, 0.0], &[f64::MAX, 0.0]] {
            assert!(moments.observe(q, &[1.0; 2]).is_err());
            assert_eq!(moments.moments(), before);
            assert_eq!(moments.squared_weights(), weights);
            assert_eq!(moments.count(), 50);
        }
        moments.observe(&[0.0; 2], &[0.0; 2]).unwrap();
    }
}

#[test]
fn schedule_and_optimizer_configuration_reject_invalid_boundaries() {
    for offset in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(DiminishingSchedule::new(offset, 0.75).is_err());
    }
    for exponent in [0.5, 1.001, f64::INFINITY, f64::NAN] {
        assert!(DiminishingSchedule::new(1.0, exponent).is_err());
    }
    let schedule = DiminishingSchedule::new(1.0, 0.75).unwrap();
    assert!(schedule.weight(usize::MAX) > 0.0);
    assert!(schedule.weight(100) < schedule.weight(10));
    for rate in [0.0, -1.0, 1.01, f64::INFINITY, f64::NAN] {
        assert!(StepAdaptation::robbins_monro(rate, schedule).is_err());
    }
    for epsilon in [0.0, -1.0, f64::from_bits(1), f64::INFINITY, f64::NAN] {
        assert!(StepAdaptation::adam(0.1, schedule, 0.9, 0.99, epsilon).is_err());
    }
    for beta in [-0.1, 1.0, f64::NAN] {
        assert!(StepAdaptation::adam(0.1, schedule, beta, 0.99, 1e-8).is_err());
    }
    assert!(StepAdaptation::adam(0.1, schedule, 0.0, 0.0, 1e-8).is_ok());
}

#[test]
fn all_controllers_preserve_state_on_invalid_or_backend_observations() {
    for settings in settings() {
        let mut controller =
            StepSizeController::new(0.1.try_into().unwrap(), 0.8.try_into().unwrap(), settings);
        for a in [0.9, 0.4, 1.0] {
            controller.update(StepObservation::Acceptance(a)).unwrap();
        }
        for observation in [
            StepObservation::Acceptance(f64::NAN),
            StepObservation::Acceptance(f64::INFINITY),
            StepObservation::Acceptance(-0.1),
            StepObservation::Acceptance(1.1),
            StepObservation::BackendFailure,
        ] {
            let mut control = controller.clone();
            assert!(controller.update(observation).is_err());
            assert_eq!(
                controller.update(StepObservation::Acceptance(0.7)).unwrap(),
                control.update(StepObservation::Acceptance(0.7)).unwrap()
            );
        }
        let mut control = controller.clone();
        assert_eq!(
            controller
                .update(StepObservation::Divergence(Divergence::Energy))
                .unwrap(),
            control.update(StepObservation::Acceptance(0.0)).unwrap()
        );
    }
}

#[test]
fn dual_controller_is_exactly_the_reference_and_adam_matches_scalar_recurrence() {
    let initial = 0.2.try_into().unwrap();
    let target = 0.8.try_into().unwrap();
    let mut reference = DualAveraging::new(initial, target);
    let mut wrapper = StepSizeController::new(initial, target, StepAdaptation::default());
    let schedule = DiminishingSchedule::new(3.0, 0.75).unwrap();
    let mut adam = StepSizeController::new(
        initial,
        target,
        StepAdaptation::adam(0.1, schedule, 0.9, 0.99, 1e-8).unwrap(),
    );
    let (mut m, mut v, mut eta) = (0.0, 0.0, 0.2_f64.ln());
    for (t, a) in [0.9, 0.2, 0.8, 0.0, 1.0].into_iter().enumerate() {
        assert_eq!(
            wrapper.update(StepObservation::Acceptance(a)).unwrap(),
            reference.update(a).unwrap()
        );
        let g = 0.8 - a;
        m = 0.9 * m + 0.1 * g;
        v = 0.99 * v + 0.01 * g * g;
        eta -= 0.1 * (3.0 + t as f64).powf(-0.75) * (m / (1.0 - 0.9_f64.powi(t as i32 + 1)))
            / ((v / (1.0 - 0.99_f64.powi(t as i32 + 1))).sqrt() + 1e-8);
        close(
            adam.update(StepObservation::Acceptance(a)).unwrap().value(),
            eta.exp(),
        );
    }
    assert_eq!(wrapper.finish(), reference.finish());
}

#[test]
fn controllers_find_the_same_log_step_root() {
    // Deterministic decreasing acceptance curve: p(eta)=1/(1+exp(eta)).
    // At target .8, eta=ln(.25). This is an optimizer test, not an HMC proof.
    for settings in settings() {
        let mut c =
            StepSizeController::new(2.0.try_into().unwrap(), 0.8.try_into().unwrap(), settings);
        let mut step = 2.0;
        for _ in 0..100_000 {
            step = c
                .update(StepObservation::Acceptance(1.0 / (1.0 + step)))
                .unwrap()
                .value();
        }
        assert!((c.finish().value().ln() - 0.25_f64.ln()).abs() < 0.01);
    }
}

#[test]
fn weighted_warmup_runs_each_controller_then_freezes_geometry() {
    let target = Gaussian::new(3);
    for settings in settings() {
        let options = FisherOptions::new(1000)
            .with_step_adaptation(settings)
            .with_weighted_moments(DiminishingSchedule::new(1.0, 0.75).unwrap());
        let mut rng = SmallRng::seed_from_u64(6123);
        let (mut chain, report) = FisherHmcWarmup::new(
            &target,
            OwnedBuffer::new(3),
            HmcOptions::new(0.1, 5).unwrap(),
            options,
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        assert_eq!(report.metric_updates, 850);
        let scales = chain.metric().scales().to_vec();
        let step = chain.options().step_size();
        let mut mean = [0.0; 3];
        let mut second = [0.0; 3];
        for _ in 0..8192 {
            let _ = chain.step(&mut rng).unwrap();
            for (i, &x) in chain.point().position().iter().enumerate() {
                mean[i] += x / 8192.0;
                second[i] += x * x / 8192.0;
            }
            assert_eq!(chain.metric().scales(), scales);
            assert_eq!(chain.options().step_size(), step);
        }
        for i in 0..3 {
            assert!(
                mean[i].abs() < 0.1 && (second[i] - 1.0).abs() < 0.15,
                "{settings:?}: {mean:?} {second:?}"
            );
        }
        assert!(chain.metric().log_det().is_finite());
    }
}

#[test]
fn zero_weighted_warmup_preserves_rng_and_initial_step_for_all_controllers() {
    let target = Gaussian::new(2);
    for settings in settings() {
        let mut rng = SmallRng::seed_from_u64(712);
        let mut control = rng.clone();
        let options = FisherOptions::new(0)
            .with_step_adaptation(settings)
            .with_weighted_moments(DiminishingSchedule::new(1.0, 0.75).unwrap());
        let (chain, report) = FisherHmcWarmup::new(
            &target,
            OwnedBuffer::new(2),
            HmcOptions::new(0.123, 5).unwrap(),
            options,
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        assert_eq!(rng.next_u64(), control.next_u64());
        assert_eq!(chain.options().step_size().value(), 0.123);
        assert_eq!(chain.metric().scales(), &[1.0; 2]);
        assert_eq!(report.metric_updates, 0);
        assert_eq!(report.integration_attempts, 0);
    }
}

#[test]
fn weighted_observations_and_controllers_allocate_nothing() {
    let schedule = DiminishingSchedule::new(1.0, 0.75).unwrap();
    let mut moments = WeightedFisherMoments::new(2, schedule).unwrap();
    let mut c = StepSizeController::new(
        0.1.try_into().unwrap(),
        0.8.try_into().unwrap(),
        settings()[2],
    );
    let mut out = [0.0; 2];
    let counts = allocation_counter::measure(|| {
        for _ in 0..100 {
            moments.observe(&[1.0, 2.0], &[-1.0, -2.0]).unwrap();
            moments.scales_into(&[1.0; 2], 1e-5, &mut out).unwrap();
            c.update(StepObservation::Acceptance(0.7)).unwrap();
        }
    });
    assert_eq!(counts.count_total, 0);
}

#[cfg(feature = "faer")]
#[test]
fn weighted_unweighted_low_rank_combination_is_explicitly_rejected() {
    let options = FisherOptions::new(10)
        .with_max_rank(1)
        .with_weighted_moments(DiminishingSchedule::new(1.0, 0.75).unwrap());
    assert!(
        FisherHmcWarmup::new(
            &Gaussian::new(2),
            OwnedBuffer::new(2),
            HmcOptions::default(),
            options
        )
        .is_err()
    );
    assert!(FisherOptions::new(10).with_window_budget(0, 10).is_err());
    assert!(FisherOptions::new(10).with_window_budget(10, 1).is_err());
}
