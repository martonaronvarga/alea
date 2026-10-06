use super::*;
use crate::adapt::DualAveraging;
#[cfg(feature = "faer")]
use alea_math::metric::EuclideanMetric;
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible};

struct CountedGaussian {
    calls: Cell<usize>,
}

#[test]
fn weighted_moments_do_not_reset_at_window_switch_and_stop_at_freeze() {
    let target = CountedGaussian {
        calls: Cell::new(0),
    };
    let schedule = DiminishingSchedule::new(1.0, 0.75).unwrap();
    let mut expected = WeightedFisherMoments::new(2, schedule).unwrap();
    let mut frozen_metric = Vec::new();
    let mut rng = SmallRng::seed_from_u64(618);
    FisherHmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        HmcOptions::new(0.1, 2).unwrap(),
        FisherOptions::new(40).with_weighted_moments(schedule),
    )
    .unwrap()
    .run_observed(&mut rng, |i, state, _, _| {
        if i < 34 {
            expected
                .observe(
                    state.chain.point().position(),
                    state.chain.point().gradient(),
                )
                .unwrap();
            frozen_metric = metric_bits(state.chain.metric());
        } else {
            assert_eq!(metric_bits(state.chain.metric()), frozen_metric);
        }
        let actual = state.weighted.as_ref().unwrap();
        assert_eq!(actual.count(), (i + 1).min(34));
        for (&a, &b) in actual.moments().iter().zip(expected.moments()) {
            assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()));
        }
        assert!((actual.squared_weights() - expected.squared_weights()).abs() < 1e-12);
    })
    .unwrap();
}
impl LogDensityGradient for CountedGaussian {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        g[0] = -5.5 * q[0] + 4.5 * q[1];
        g[1] = 4.5 * q[0] - 5.5 * q[1];
        Ok(0.5 * (q[0] * g[0] + q[1] * g[1]))
    }
}

fn metric_bits(metric: &LowRankDiagonalMetric) -> Vec<u64> {
    metric
        .scales()
        .iter()
        .chain(metric.basis())
        .chain(metric.eigenvalues())
        .map(|v| v.to_bits())
        .collect()
}

fn check_schedule<I: Integrator>(
    n: usize,
    rank: usize,
    integrator: I,
    initialization: MetricInitialization,
) {
    check_schedule_target(n, rank, integrator, initialization, 0.8.try_into().unwrap());
}

fn check_schedule_target<I: Integrator>(
    n: usize,
    rank: usize,
    integrator: I,
    initialization: MetricInitialization,
    acceptance: AcceptanceTarget,
) {
    let stages = integrator.stages();
    let target = CountedGaussian {
        calls: Cell::new(0),
    };
    let mut rng = SmallRng::seed_from_u64(617);
    let mut options = FisherOptions::new(n)
        .with_initialization(initialization)
        .with_target(acceptance);
    options.max_rank = rank;
    let initial = if initialization == MetricInitialization::Identity {
        [0.0; 2]
    } else {
        [0.5, -0.25]
    };
    let fallback = if initialization == MetricInitialization::Identity {
        [1.0; 2]
    } else {
        [3.875_f64.sqrt().recip(), 3.625_f64.sqrt().recip()]
    };
    let warmup = FisherHmcWarmup::new_with_integrator(
        &target,
        OwnedBuffer::from_fn(2, |i| initial[i]),
        HmcOptions::new(0.1, 2).unwrap(),
        options,
        integrator,
    )
    .unwrap();
    let freeze = n * 85 / 100;
    let switch = n * 3 / 10;
    let mut draws: Vec<[f64; 2]> = Vec::new();
    let mut scores: Vec<[f64; 2]> = Vec::new();
    let mut previous_position = initial;
    for (&actual, &expected) in
        warmup
            .chain
            .metric()
            .scales()
            .iter()
            .zip(if n == 0 { &[1.0; 2] } else { &fallback })
    {
        assert!((actual - expected).abs() < 1e-12 * expected);
    }
    let mut last_metric = metric_bits(warmup.chain.metric());
    let mut last_step = warmup.chain.options().step_size();
    let mut final_step_adapter = None;
    let mut accepted = 0;
    let mut divergences = 0;
    let mut attempts = 0;
    let mut visits = 0;
    let mut positive_rank = false;
    #[cfg(feature = "faer")]
    let mut expected_metric = LowRankDiagonalMetric::new(
        OwnedBuffer::from_fn(2, |i| warmup.chain.metric().scales()[i]),
        OwnedBuffer::new(0),
        OwnedBuffer::new(0),
    )
    .unwrap();
    let (mut chain, report) = warmup
        .run_observed(&mut rng, |i, state, transition, report| {
            visits += 1;
            let point = state.chain.point();
            let q: [f64; 2] = point.position().try_into().unwrap();
            let score: [f64; 2] = point.gradient().try_into().unwrap();
            if !transition.accepted {
                assert_eq!(q.map(f64::to_bits), previous_position.map(f64::to_bits));
            }
            previous_position = q;
            assert!((score[0] - (-5.5 * q[0] + 4.5 * q[1])).abs() < 1e-12);
            assert!((score[1] - (4.5 * q[0] - 5.5 * q[1])).abs() < 1e-12);
            assert!(
                (point.log_density() - 0.5 * (q[0] * score[0] + q[1] * score[1])).abs() < 1e-12
            );
            accepted += usize::from(transition.accepted);
            divergences += usize::from(transition.divergence.is_some());
            attempts += transition.integration_steps;
            assert_eq!(report.accepted, accepted);
            assert_eq!(report.divergences, divergences);
            assert_eq!(report.integration_attempts, report.search_probes + attempts);
            assert_eq!(target.calls.get(), 1 + report.integration_attempts * stages);
            assert_eq!(report.metric_updates, (i + 1).min(freeze));

            if i < freeze {
                if i == switch {
                    draws.clear();
                    scores.clear();
                }
                draws.push(q);
                scores.push(score);
                let period = if i < switch { 10 } else { 80 };
                let seen = draws.len();
                let start = if seen <= 2 * period {
                    0
                } else {
                    ((seen - 1) / period - 1) * period
                };
                assert_eq!(state.adapter.count(), seen - start, "n={n}, i={i}");
                assert_eq!(state.adapter.seen, seen);
                assert_eq!(state.adapter.period, period);
                assert_eq!(
                    state.adapter.retained,
                    if rank > 0 { seen.min(2 * period) } else { 0 }
                );
                // Independent centered batch moments of actual retained states,
                // including rejections; no calls to FisherMetricAdapter here.
                for axis in 0..2 {
                    let scatter = |data: &[[f64; 2]]| {
                        let data = &data[start..];
                        let mean = data.iter().map(|v| v[axis]).sum::<f64>() / data.len() as f64;
                        data.iter().map(|v| (v[axis] - mean).powi(2)).sum::<f64>()
                    };
                    let expected = fallback[axis]
                        * ((scatter(&draws) / fallback[axis].powi(2) + 1e-5)
                            / (scatter(&scores) * fallback[axis].powi(2) + 1e-5))
                            .powf(0.25);
                    assert!(
                        (state.chain.metric().scales()[axis] - expected).abs()
                            < 1e-10 * (1.0 + expected),
                        "n={n}, i={i}, axis={axis}"
                    );
                }
                #[cfg(feature = "faer")]
                {
                    expected_metric
                        .set_scales(state.chain.metric().scales())
                        .unwrap();
                    if rank > 0 && (seen.is_multiple_of(period) || i + 1 == freeze) && seen >= 2 {
                        let start = seen.saturating_sub(2 * period);
                        let q: Vec<_> = draws[start..].iter().flatten().copied().collect();
                        let s: Vec<_> = scores[start..].iter().flatten().copied().collect();
                        expected_metric = alea_math::fisher::fit_low_rank(
                            &q,
                            &s,
                            state.chain.metric().scales(),
                            1e-5,
                            2.0,
                            rank,
                        )
                        .unwrap();
                    }
                    assert_eq!(state.chain.metric().rank(), expected_metric.rank());
                    for p in [[1.0, 0.0], [0.0, 1.0]] {
                        let mut actual = [0.0; 2];
                        let mut expected = [0.0; 2];
                        state.chain.metric().velocity(&p, &mut actual).unwrap();
                        expected_metric.velocity(&p, &mut expected).unwrap();
                        for (a, b) in actual.into_iter().zip(expected) {
                            assert!(
                                (a - b).abs() < 1e-8 * (1.0 + b.abs()),
                                "n={n}, i={i}: {a} != {b}"
                            );
                        }
                    }
                }
            } else {
                assert_eq!(
                    metric_bits(state.chain.metric()),
                    last_metric,
                    "geometry changed after freeze"
                );
                if i == freeze && i > 0 {
                    final_step_adapter = Some(DualAveraging::new(last_step, acceptance));
                }
                if let Some(adapter) = &mut final_step_adapter {
                    let expected = adapter.update(transition.acceptance_probability).unwrap();
                    // Independently recomputed transcendental results need a
                    // tolerance under Miri; frozen stored values remain bit-exact.
                    assert!(
                        (state.chain.options().step_size().value() - expected.value()).abs()
                            < 1e-12 * expected.value()
                    );
                }
            }
            positive_rank |= state.chain.metric().rank() > 0;
            last_metric = metric_bits(state.chain.metric());
            last_step = state.chain.options().step_size();
        })
        .unwrap();
    assert_eq!(visits, n);
    assert_eq!(report.metric_updates, freeze);
    assert_eq!(report.step_size, chain.options().step_size());
    if let Some(adapter) = final_step_adapter {
        let expected = adapter.finish().value();
        assert!((chain.options().step_size().value() - expected).abs() < 1e-12 * expected);
    }
    if n >= 100 {
        assert!(
            accepted > 0 && accepted < n,
            "must exercise retained rejection repeats"
        );
        assert_eq!(
            positive_rank,
            rank > 0,
            "must exercise nontrivial spectral geometry"
        );
    }
    let step = chain.options().step_size();
    for _ in 0..3 {
        let _ = chain.step(&mut rng).unwrap();
        assert_eq!(metric_bits(chain.metric()), last_metric);
        assert_eq!(chain.options().step_size(), step);
    }
}

#[test]
fn configured_acceptance_survives_fixed_geometry_restart_and_zero_warmup() {
    assert_eq!(FisherOptions::new(100).target.value(), 0.8);
    for n in [0, 12, 100] {
        check_schedule_target(
            n,
            0,
            Leapfrog,
            MetricInitialization::Identity,
            0.95.try_into().unwrap(),
        );
    }
    #[cfg(feature = "faer")]
    check_schedule_target(
        100,
        2,
        Leapfrog,
        MetricInitialization::Identity,
        0.95.try_into().unwrap(),
    );
}

#[test]
fn configured_acceptance_drives_initial_and_final_dual_averaging() {
    let target = CountedGaussian {
        calls: Cell::new(0),
    };
    let acceptance = 0.95.try_into().unwrap();
    let build = || {
        FisherHmcWarmup::new(
            &target,
            OwnedBuffer::new(2),
            HmcOptions::new(0.1, 2).unwrap(),
            FisherOptions::new(20).with_target(acceptance),
        )
        .unwrap()
    };
    let mut probe = build();
    let mut rng = SmallRng::seed_from_u64(617);
    let search =
        find_reasonable_step_size(&mut probe.chain, SearchOptions::default(), &mut rng.clone())
            .unwrap();
    let mut expected = DualAveraging::new(search.step_size, acceptance);
    let mut previous = search.step_size;
    let (chain, _) = build()
        .run_observed(&mut rng, |i, state, transition, _| {
            if i == 17 {
                // floor(0.85 * 20): final fixed-geometry restart.
                expected = DualAveraging::new(previous, acceptance);
            }
            let next = expected.update(transition.acceptance_probability).unwrap();
            assert!(
                (state.chain.options().step_size().value() - next.value()).abs()
                    < 1e-12 * next.value()
            );
            previous = next;
        })
        .unwrap();
    let final_step = expected.finish().value();
    assert!((chain.options().step_size().value() - final_step).abs() < 1e-12 * final_step);
}

#[test]
fn diagonal_warmup_matches_batch_windows_and_freezes_geometry() {
    let budgets: &[usize] = if cfg!(miri) {
        &[2, 12]
    } else {
        &[0, 1, 2, 12, 39, 40, 100, 401]
    };
    for &n in budgets {
        check_schedule(n, 0, Leapfrog, MetricInitialization::Identity);
        check_schedule(
            n,
            0,
            crate::integrator::TwoStage::OMF,
            MetricInitialization::Identity,
        );
    }
}

#[cfg(feature = "faer")]
#[test]
fn low_rank_warmup_refits_windows_and_final_nonboundary_scales() {
    for n in [2, 12, 39, 40, 100, 401] {
        for rank in [1, 2] {
            check_schedule(n, rank, Leapfrog, MetricInitialization::Identity);
            check_schedule(
                n,
                rank,
                crate::integrator::TwoStage::OMF,
                MetricInitialization::Identity,
            );
        }
    }
}

#[test]
fn clipped_initialization_remains_the_regularization_reference_through_warmup() {
    let budgets: &[usize] = if cfg!(miri) { &[12] } else { &[0, 12, 100] };
    for &n in budgets {
        for rank in 0..=if cfg!(feature = "faer") { 2 } else { 0 } {
            check_schedule(n, rank, Leapfrog, MetricInitialization::ClippedScore);
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("injected warmup model failure")]
struct ModelFailure;

struct FailingGaussian {
    inner: CountedGaussian,
    fail: Cell<bool>,
}
impl LogDensityGradient for FailingGaussian {
    type Error = ModelFailure;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        if self.fail.get() {
            g[0] = 123.0; // Failed scratch must never become an observation.
            return Err(ModelFailure);
        }
        Ok(self.inner.logp_grad(q, g).unwrap())
    }
}

#[test]
fn backend_failure_at_window_switch_and_freeze_never_returns_a_chain() {
    // The next transition fails: early, at the 30% reset, at the 85% freeze,
    // and on the last iteration. A backend error must not become a rejection.
    for completed_index in [0, 14, 41, 48] {
        let target = FailingGaussian {
            inner: CountedGaussian {
                calls: Cell::new(0),
            },
            fail: Cell::new(false),
        };
        let mut rng = SmallRng::seed_from_u64(617);
        let options = FisherOptions::new(50);
        #[cfg(feature = "faer")]
        let options = options.with_max_rank(2);
        let warmup = FisherHmcWarmup::new_with_integrator(
            &target,
            OwnedBuffer::new(2),
            HmcOptions::new(0.1, 2).unwrap(),
            options,
            crate::integrator::TwoStage::OMF,
        )
        .unwrap();
        let mut visits = 0;
        let result = warmup.run_observed(&mut rng, |i, _, _, _| {
            visits += 1;
            if i == completed_index {
                target.fail.set(true);
            }
        });
        assert!(matches!(
            result,
            Err(FisherWarmupError::Warmup(WarmupError::Hmc(
                crate::hmc::HmcError::Evaluation {
                    integration_step: 1,
                    source: alea_core::target::EvaluationError::Model(ModelFailure),
                }
            )))
        ));
        assert_eq!(visits, completed_index + 1);
    }
}

#[cfg(not(miri))]
#[test]
fn public_diagonal_warmup_run_allocates_nothing_after_construction() {
    let target = CountedGaussian {
        calls: Cell::new(0),
    };
    let mut rng = SmallRng::seed_from_u64(617);
    let warmup = FisherHmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        HmcOptions::new(0.1, 2).unwrap(),
        FisherOptions::new(100),
    )
    .unwrap();
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(warmup.run(&mut rng).unwrap());
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(result.unwrap().1.metric_updates, 85);
}

struct ScaledGaussian {
    precision: f64,
    calls: Cell<usize>,
}
impl LogDensityGradient for ScaledGaussian {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        g[0] = -self.precision * q[0];
        Ok(0.5 * q[0] * g[0])
    }
}

#[test]
fn tail_start_warmup_preserves_finite_cache_and_fixed_gaussian_geometry() {
    // No assertion of convergence: huge log densities lose energy-difference
    // precision, and clipped-score initialization is explicitly experimental.
    let magnitudes: &[f64] = if cfg!(miri) {
        &[1e6]
    } else {
        &[1e-150, 1.0, 1e6, 1e100, 1e150]
    };
    let n = if cfg!(miri) { 12 } else { 100 };
    for &magnitude in magnitudes {
        for sign in [-1.0, 1.0] {
            for initialization in [
                MetricInitialization::Identity,
                MetricInitialization::ClippedScore,
            ] {
                let target = ScaledGaussian {
                    precision: 1.0,
                    calls: Cell::new(0),
                };
                let mut rng = SmallRng::seed_from_u64(617);
                let options = FisherOptions::new(n).with_initialization(initialization);
                #[cfg(feature = "faer")]
                let options = options.with_max_rank(1);
                let initial = sign * magnitude;
                let warmup = FisherHmcWarmup::new(
                    &target,
                    OwnedBuffer::from_fn(1, |_| initial),
                    HmcOptions::new(0.1, 2).unwrap(),
                    options,
                )
                .unwrap();
                assert_eq!(target.calls.get(), 1);
                let expected_scale = match initialization {
                    MetricInitialization::Identity => 1.0,
                    MetricInitialization::ClippedScore => {
                        magnitude.clamp(1e-20, 1e20).sqrt().recip()
                    }
                };
                assert!(
                    (warmup.chain.metric().scales()[0] - expected_scale).abs()
                        < 1e-12 * expected_scale
                );
                let mut completed = 0;
                let (mut chain, report) = warmup
                    .run_observed(&mut rng, |_, state, _, _| {
                        completed += 1;
                        let point = state.chain.point();
                        let q = point.position()[0];
                        assert!(q.is_finite() && point.log_density().is_finite());
                        assert_eq!(point.gradient()[0], -q);
                        assert_eq!(point.log_density(), -0.5 * q * q);
                        assert!(state.chain.metric().scales()[0].is_finite());
                        if initialization == MetricInitialization::Identity {
                            // C = F for a unit Gaussian, including rejected repeats.
                            assert!((state.chain.metric().scales()[0] - 1.0).abs() < 1e-12);
                            assert_eq!(state.chain.metric().rank(), 0);
                        }
                    })
                    .unwrap_or_else(|error| {
                        panic!("start={initial}, init={initialization:?}: {error}")
                    });
                assert_eq!(completed, n);
                assert_eq!(report.metric_updates, n * 85 / 100);
                assert!(target.calls.get() <= 1 + 80 + 2 * n);
                let frozen = metric_bits(chain.metric());
                let step = chain.options().step_size();
                let _ = chain.step(&mut rng).unwrap();
                assert_eq!(metric_bits(chain.metric()), frozen);
                assert_eq!(chain.options().step_size(), step);
            }
        }
    }
}

#[test]
fn unrepresentable_initial_density_is_a_typed_construction_failure() {
    for initialization in [
        MetricInitialization::Identity,
        MetricInitialization::ClippedScore,
    ] {
        let target = ScaledGaussian {
            precision: 1.0,
            calls: Cell::new(0),
        };
        let result = FisherHmcWarmup::new(
            &target,
            OwnedBuffer::from_fn(1, |_| 1e200),
            HmcOptions::new(0.1, 2).unwrap(),
            FisherOptions::new(100).with_initialization(initialization),
        );
        assert!(matches!(
            result,
            Err(FisherWarmupError::Warmup(WarmupError::Hmc(
                crate::hmc::HmcError::Evaluation {
                    integration_step: 0,
                    source: alea_core::target::EvaluationError::NonFiniteLogDensity,
                }
            )))
        ));
        assert_eq!(target.calls.get(), 1);
    }
}

#[test]
fn unresolvable_initial_stiffness_exhausts_bounded_search_before_observing_draws() {
    // Unit initial geometry cannot resolve this target even at the search floor.
    // The zero initial score also makes ClippedScore select identity.
    for initialization in [
        MetricInitialization::Identity,
        MetricInitialization::ClippedScore,
    ] {
        let target = ScaledGaussian {
            precision: 1e30,
            calls: Cell::new(0),
        };
        let mut rng = SmallRng::seed_from_u64(617);
        let warmup = FisherHmcWarmup::new(
            &target,
            OwnedBuffer::new(1),
            HmcOptions::new(0.1, 2).unwrap(),
            FisherOptions::new(100).with_initialization(initialization),
        )
        .unwrap();
        let result = warmup.run_observed(&mut rng, |_, _, _, _| {
            panic!("search failure must not observe a transition")
        });
        assert!(matches!(
            result,
            Err(FisherWarmupError::Warmup(WarmupError::SearchExhausted))
        ));
        assert!(target.calls.get() > 1 && target.calls.get() <= 81);
    }
}
