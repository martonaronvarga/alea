use alea_core::target::{EvaluationError, LogDensityGradient};
use alea_distributions::Gaussian;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric},
};
use alea_mcmc::{
    Hmc, HmcOptions,
    adapt::{
        CovarianceError, HmcWarmup, MetricKind, OnlineCovariance, SearchOptions, WarmupError,
        WarmupOptions, WarmupStage, WindowSchedule, find_reasonable_step_size,
    },
    hmc::HmcError,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-10 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn windows_cover_short_default_and_overflow_sized_schedules() {
    for total in [0, 1, 19, 20, 21, 149, 150, 200, 1000] {
        let stages: Vec<_> = WindowSchedule::new(total).collect();
        assert_eq!(stages.len(), total);
        if total < 20 {
            assert!(stages.iter().all(|s| *s == WarmupStage::InitialFast));
        } else {
            let initial = if total < 150 { total * 15 / 100 } else { 75 };
            let final_size = if total < 150 { total / 10 } else { 50 };
            assert!(
                stages[..initial]
                    .iter()
                    .all(|s| *s == WarmupStage::InitialFast)
            );
            assert!(
                stages[total - final_size..]
                    .iter()
                    .all(|s| *s == WarmupStage::FinalFast)
            );
            assert_eq!(
                stages[total - final_size - 1],
                WarmupStage::Slow { window_end: true }
            );
        }
    }
    let ends: Vec<_> = WindowSchedule::new(1000)
        .enumerate()
        .filter_map(|(i, s)| (s == WarmupStage::Slow { window_end: true }).then_some(i + 1))
        .collect();
    assert_eq!(ends, [100, 150, 250, 450, 950]);
    let mut huge = WindowSchedule::new(usize::MAX);
    assert_eq!(huge.len(), usize::MAX);
    assert_eq!(huge.nth(100), Some(WarmupStage::Slow { window_end: false }));
    assert_eq!(huge.len(), usize::MAX - 101);
}

#[test]
fn online_covariance_matches_batch_and_metric_uses_inverse_mass() {
    let data = [[1.0, 2.0], [2.0, -1.0], [-3.0, 4.0], [0.0, 3.0]];
    for kind in [MetricKind::Diagonal, MetricKind::Dense] {
        let mut estimate = OnlineCovariance::new(2, kind).unwrap();
        for x in data {
            estimate.update(&x).unwrap();
        }
        close(estimate.mean()[0], 0.0);
        close(estimate.mean()[1], 2.0);
        let expected = [[14.0 / 3.0, -4.0], [-4.0, 14.0 / 3.0]];
        let covariance = estimate.regularized_covariance().unwrap();
        let metric = estimate.metric().unwrap();
        let mut velocity = [0.0; 2];
        metric.velocity(&[0.3, -0.7], &mut velocity).unwrap();
        for row in 0..2 {
            let diagonal = expected[row][row] * 4.0 / 9.0 + 0.005 / 9.0;
            let cross = if kind == MetricKind::Dense {
                expected[row][1 - row] * 4.0 / 9.0
            } else {
                0.0
            };
            close(
                velocity[row],
                diagonal * [0.3, -0.7][row] + cross * [0.3, -0.7][1 - row],
            );
            close(
                covariance[if kind == MetricKind::Dense {
                    row * 3
                } else {
                    row
                }],
                diagonal,
            );
        }
        let mut p = [0.0; 2];
        metric.sample_momentum(&[0.3, -0.7], &mut p).unwrap();
        close(
            metric.kinetic_energy(&p, &mut velocity).unwrap(),
            0.5 * (0.09 + 0.49),
        );
    }
}

#[test]
fn covariance_failures_are_transactional_and_constant_windows_regularize() {
    for kind in [MetricKind::Diagonal, MetricKind::Dense] {
        assert!(matches!(
            OnlineCovariance::new(0, kind),
            Err(CovarianceError::Dimension)
        ));
        assert!(matches!(
            OnlineCovariance::new(usize::MAX, kind),
            Err(CovarianceError::Dimension)
        ));
        let mut e = OnlineCovariance::new(2, kind).unwrap();
        assert!(matches!(e.metric(), Err(CovarianceError::TooFewSamples)));
        e.update(&[1.0, 2.0]).unwrap();
        e.update(&[1.0, 2.0]).unwrap();
        let before = e.regularized_covariance().unwrap();
        for bad in [
            &[1.0][..],
            &[f64::NAN, 0.0],
            &[0.0, f64::INFINITY],
            &[f64::MAX, 0.0],
        ] {
            assert!(e.update(bad).is_err());
            assert_eq!(e.count(), 2);
            assert_eq!(e.mean(), [1.0, 2.0]);
            assert_eq!(
                e.regularized_covariance().unwrap().as_slice(),
                before.as_slice()
            );
        }
        let metric = e.metric().unwrap();
        let mut v = [0.0; 2];
        metric.velocity(&[1.0, 1.0], &mut v).unwrap();
        close(v[0], 0.005 / 7.0);
        e.reset();
        assert_eq!(e.count(), 0);
        assert_eq!(e.mean(), [0.0, 0.0]);
    }
}

#[test]
fn search_is_bounded_replayable_and_never_commits_a_proposal() {
    let target = Gaussian::new(2);
    for initial in [1e-8, 100.0] {
        let mut chain = Hmc::new(
            &target,
            OwnedBuffer::new(2),
            IdentityMetric::new(2),
            HmcOptions::new(initial, 4).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(51);
        let mut replay = rng.clone();
        let result =
            find_reasonable_step_size(&mut chain, SearchOptions::default(), &mut rng).unwrap();
        let again =
            find_reasonable_step_size(&mut chain, SearchOptions::default(), &mut replay).unwrap();
        assert_eq!(result.step_size, again.step_size);
        assert_eq!(result.probes, again.probes);
        assert_eq!(rng, replay);
        assert!(result.probes <= 80);
        assert_eq!(chain.point().position(), [0.0, 0.0]);
        assert_eq!(chain.options().step_size().value(), initial);
        close(chain.point().log_density(), 0.0);
    }
    for args in [
        (0.0, 1.0, 3, 0.8),
        (1.0, 1.0, 3, 0.8),
        (1.0, 2.0, 1, 0.8),
        (1.0, 2.0, 3, 1.0),
    ] {
        assert!(SearchOptions::new(args.0, args.1, args.2, args.3).is_err());
    }
}

#[derive(Debug, thiserror::Error)]
#[error("backend failed")]
struct BackendError;
struct Failing {
    fail: Cell<bool>,
}
impl LogDensityGradient for Failing {
    type Error = BackendError;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, BackendError> {
        g[0] = -q[0];
        if self.fail.get() {
            Err(BackendError)
        } else {
            Ok(-0.5 * q[0] * q[0])
        }
    }
}
struct Flat;
impl LogDensityGradient for Flat {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = 0.0;
        Ok(0.0)
    }
}

#[test]
fn search_and_warmup_preserve_backend_errors_and_report_search_exhaustion() {
    let mut rng = SmallRng::seed_from_u64(1);
    let target = Failing {
        fail: Cell::new(false),
    };
    let mut chain = Hmc::new(
        &target,
        OwnedBuffer::new(1),
        IdentityMetric::new(1),
        HmcOptions::default(),
    )
    .unwrap();
    target.fail.set(true);
    assert!(matches!(
        find_reasonable_step_size(&mut chain, SearchOptions::default(), &mut rng),
        Err(WarmupError::Hmc(HmcError::Evaluation {
            source: EvaluationError::Model(BackendError),
            ..
        }))
    ));
    assert_eq!(chain.point().position(), [0.0]);
    target.fail.set(false);
    let controller = HmcWarmup::new(
        &target,
        OwnedBuffer::new(1),
        HmcOptions::default(),
        WarmupOptions::new(100),
    )
    .unwrap();
    target.fail.set(true);
    assert!(matches!(controller.run(&mut rng), Err(WarmupError::Hmc(_))));
    let mut flat = Hmc::new(
        &Flat,
        OwnedBuffer::new(1),
        IdentityMetric::new(1),
        HmcOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        find_reasonable_step_size(
            &mut flat,
            SearchOptions::new(0.01, 10.0, 3, 0.8).unwrap(),
            &mut rng
        ),
        Err(WarmupError::SearchExhausted)
    ));
    assert_eq!(flat.point().position(), [0.0]);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "exact replay needs deterministic floats; run separately in CI"
)]
fn warmup_freezes_is_deterministic_and_zero_iterations_consume_no_rng() {
    let target = Gaussian::new(2);
    for kind in [MetricKind::Diagonal, MetricKind::Dense] {
        for iterations in [0, 19, 20, 200] {
            let run = |rng: &mut SmallRng| {
                HmcWarmup::new(
                    &target,
                    OwnedBuffer::new(2),
                    HmcOptions::new(0.2, 3).unwrap(),
                    WarmupOptions::new(iterations).with_metric(kind),
                )
                .unwrap()
                .run(rng)
                .unwrap()
            };
            let mut rng = SmallRng::seed_from_u64(119);
            let original = rng.clone();
            let (mut chain, report) = run(&mut rng);
            let mut replay = original.clone();
            let (other, again) = run(&mut replay);
            assert_eq!(rng, replay);
            assert_eq!(chain.point().position(), other.point().position());
            assert_eq!(report.step_size, again.step_size);
            assert_eq!(
                report.metric_updates,
                WindowSchedule::new(iterations)
                    .filter(|s| *s == WarmupStage::Slow { window_end: true })
                    .count()
            );
            if iterations == 0 {
                assert_eq!(rng, original);
                assert_eq!(report.step_size.value(), 0.2);
            }
            let mut velocity = [0.0; 2];
            chain
                .metric()
                .velocity(&[0.3, -0.7], &mut velocity)
                .unwrap();
            for _ in 0..10 {
                let _ = chain.step(&mut rng).unwrap();
            }
            let mut after = [0.0; 2];
            chain.metric().velocity(&[0.3, -0.7], &mut after).unwrap();
            assert_eq!(velocity, after);
            assert_eq!(chain.options().step_size(), report.step_size);
        }
    }
}

#[test]
#[cfg(not(miri))]
fn online_updates_search_and_frozen_sampling_allocate_nothing() {
    for kind in [MetricKind::Diagonal, MetricKind::Dense] {
        let mut e = OnlineCovariance::new(2, kind).unwrap();
        let count = allocation_counter::measure(|| {
            for i in 0..100 {
                e.update(&[i as f64, (i % 7) as f64]).unwrap();
            }
            e.reset();
        });
        assert_eq!(count.count_total, 0);
        let target = Gaussian::new(2);
        let mut rng = SmallRng::seed_from_u64(119);
        let (mut chain, _) = HmcWarmup::new(
            &target,
            OwnedBuffer::new(2),
            HmcOptions::new(0.2, 3).unwrap(),
            WarmupOptions::new(200).with_metric(kind),
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        let count = allocation_counter::measure(|| {
            find_reasonable_step_size(&mut chain, SearchOptions::default(), &mut rng).unwrap();
            for _ in 0..100 {
                let _ = chain.step(&mut rng).unwrap();
            }
        });
        assert_eq!(count.count_total, 0);
    }
}

struct OriginOnly;
impl LogDensityGradient for OriginOnly {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = 0.0;
        Ok(if q[0] == 0.0 { 0.0 } else { f64::NEG_INFINITY })
    }
}

#[test]
fn divergent_rejections_still_contribute_repeated_states_to_covariance() {
    for kind in [MetricKind::Diagonal, MetricKind::Dense] {
        let (chain, report) = HmcWarmup::new(
            &OriginOnly,
            OwnedBuffer::new(1),
            HmcOptions::new(0.1, 2).unwrap(),
            WarmupOptions::new(20).with_metric(kind).with_search(None),
        )
        .unwrap()
        .run(&mut SmallRng::seed_from_u64(90))
        .unwrap();
        assert_eq!(report.accepted, 0);
        assert_eq!(report.divergences, 20);
        assert_eq!(report.metric_updates, 1);
        assert_eq!(report.search_probes, 0);
        assert_eq!(chain.point().position(), [0.0]);
        let mut velocity = [0.0];
        chain.metric().velocity(&[1.0], &mut velocity).unwrap();
        // Fifteen slow observations, despite accepting no moves.
        close(velocity[0], 0.005 / 20.0);
    }
}
