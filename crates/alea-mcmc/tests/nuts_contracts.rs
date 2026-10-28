use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    HmcOptions, Nuts, NutsOptions,
    adapt::{FisherHmcWarmup, FisherOptions, HmcWarmup, WarmupOptions},
    integrator::{Leapfrog, SymmetricSplitting, TwoStage},
    nuts::{NutsError, NutsTermination},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

#[derive(Debug, thiserror::Error)]
#[error("injected model failure")]
struct Injected;
struct Target {
    calls: Cell<usize>,
    fail: Cell<usize>,
    mode: Cell<u8>,
    flat: bool,
}
impl Target {
    fn new(flat: bool) -> Self {
        Self {
            calls: Cell::new(0),
            fail: Cell::new(0),
            mode: Cell::new(0),
            flat,
        }
    }
}
impl alea_core::target::TargetSpace for Target {
    type Error = Injected;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for Target {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Injected> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.fail.get() {
            g[0] = 123.0;
            match self.mode.get() {
                1 => return Err(Injected),
                2 => panic!("injected panic"),
                3 => return Ok(f64::NAN),
                _ => {}
            }
        }
        if self.flat {
            g.fill(0.0);
            Ok(0.0)
        } else {
            g[0] = -q[0];
            g[1] = -q[1];
            Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
        }
    }
}
fn failures<I: SymmetricSplitting + Copy>(integrator: I) {
    for (mode, position) in [(1, false), (2, false), (1, true), (2, true)] {
        for call in 1..=8 {
            let target = Target::new(false);
            let mut chain = Nuts::new_with_integrator(
                &target,
                OwnedBuffer::new(2),
                IdentityMetric::new(2),
                NutsOptions::new(1e-5, 4)
                    .unwrap()
                    .with_position_turning(position),
                integrator,
            )
            .unwrap();
            target.fail.set(1 + call);
            target.mode.set(mode);
            let mut rng = SmallRng::seed_from_u64(13);
            let result = catch_unwind(AssertUnwindSafe(|| chain.step(&mut rng)));
            if mode == 1 {
                assert!(
                    matches!(result.unwrap(),Err(NutsError::Phase {integration_step,..}) if integration_step==call.div_ceil(integrator.stages()))
                );
            } else {
                assert!(result.is_err());
            }
            assert_eq!(chain.point().position(), [0.0, 0.0]);
            assert_eq!(chain.point().gradient(), [-0.0, -0.0]);
            assert_eq!(chain.point().log_density().to_bits(), (-0.0_f64).to_bits());
            target.fail.set(0);
            let before = target.calls.get();
            let info = chain.step(&mut rng).unwrap();
            assert_eq!(info.integration_steps, 15);
            assert_eq!(target.calls.get() - before, 15 * integrator.stages());
        }
    }
}
#[test]
fn every_internal_stage_failure_and_panic_preserves_cache_and_recovers() {
    failures(Leapfrog);
    failures(TwoStage::OMF);
    failures(TwoStage::BCSS);
}
#[test]
fn late_numerical_failure_discards_only_the_new_subtree() {
    let target = Target::new(false);
    let mut chain = Nuts::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        NutsOptions::new(0.01, 5).unwrap(),
    )
    .unwrap();
    target.fail.set(4);
    target.mode.set(3);
    let info = chain.step(&mut SmallRng::seed_from_u64(13)).unwrap();
    assert!(matches!(info.termination, NutsTermination::Divergence(_)));
    assert_eq!(info.integration_steps, 3);
    assert_eq!(info.tree_depth, 1);
    assert_eq!(info.valid_states, 2);
    assert_eq!(info.selected_index, 1);
    assert!(info.accepted);
    assert!(chain.point().position().iter().any(|q| *q != 0.0));
}
#[test]
fn depth_budget_counts_complete_macrosteps_not_internal_stages() {
    let target = Target::new(true);
    let mut chain = Nuts::new_with_integrator(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        NutsOptions::new(0.1, 7).unwrap(),
        TwoStage::OMF,
    )
    .unwrap();
    let info = chain.step(&mut SmallRng::seed_from_u64(12)).unwrap();
    assert_eq!(info.termination, NutsTermination::MaxDepth);
    assert_eq!(info.integration_steps, 127);
    assert_eq!(info.tree_depth, 7);
    assert_eq!(info.valid_states, 128);
    assert_eq!(info.acceptance_probability, 1.0);
    assert_eq!(target.calls.get(), 1 + 2 * 127);
}
#[test]
fn frozen_covariance_and_fisher_warmups_convert_without_evaluation() {
    let target = Target::new(false);
    let mut rng = SmallRng::seed_from_u64(19);
    let options = HmcOptions::new(0.1, 5).unwrap();
    let (hmc, _) = HmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        options,
        WarmupOptions::new(200),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    let q = hmc.point().position().to_vec();
    let step = hmc.options().step_size();
    let calls = target.calls.get();
    let mut nuts = Nuts::from_hmc(hmc, 6.try_into().unwrap()).unwrap();
    assert_eq!(target.calls.get(), calls);
    assert_eq!(nuts.point().position(), q);
    assert_eq!(nuts.options().step_size(), step);
    let _ = nuts.step(&mut rng).unwrap();
    let (hmc, _) = FisherHmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        options,
        FisherOptions::new(200),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    let calls = target.calls.get();
    let step = hmc.options().step_size();
    let mut nuts = Nuts::from_hmc(hmc, 6.try_into().unwrap()).unwrap();
    assert_eq!(target.calls.get(), calls);
    assert_eq!(nuts.options().step_size(), step);
    let _ = nuts.step(&mut rng).unwrap();
}

#[test]
fn cached_velocities_are_rebuilt_and_metric_failures_recover_without_committing() {
    use alea_math::metric::{EuclideanMetric, MetricError};
    use alea_mcmc::hamiltonian::PhaseError;
    struct Metric {
        calls: Cell<usize>,
        fail: Cell<usize>,
        panic: bool,
    }
    impl EuclideanMetric for Metric {
        fn dimension(&self) -> usize {
            2
        }
        fn velocity(&self, p: &[f64], v: &mut [f64]) -> Result<(), MetricError> {
            self.calls.set(self.calls.get() + 1);
            v[0] = f64::NAN;
            if self.calls.get() == self.fail.get() {
                assert!(!self.panic, "injected metric panic");
                return Err(MetricError::IllConditionedCorrection);
            }
            v.copy_from_slice(p);
            Ok(())
        }
        fn sample_momentum(&self, z: &[f64], p: &mut [f64]) -> Result<(), MetricError> {
            p.copy_from_slice(z);
            Ok(())
        }
        fn log_det(&self) -> f64 {
            0.
        }
    }
    // A full depth-four tree needs one initial and two per-edge applications.
    let failures: Vec<_> = if cfg!(miri) {
        vec![1, 2, 3, 16, 31]
    } else {
        (1..=31).collect()
    };
    for panic in [false, true] {
        for &fail in &failures {
            let target = Target::new(true);
            let metric = Metric {
                calls: Cell::new(0),
                fail: Cell::new(fail),
                panic,
            };
            let mut chain = Nuts::new(
                &target,
                OwnedBuffer::new(2),
                metric,
                NutsOptions::new(0.01, 4).unwrap(),
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(13);
            let result = catch_unwind(AssertUnwindSafe(|| chain.step(&mut rng)));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(matches!(
                    result.unwrap(),
                    Err(NutsError::Phase {
                        source: PhaseError::Metric(MetricError::IllConditionedCorrection),
                        ..
                    })
                ));
            }
            assert_eq!(chain.point().position(), [0., 0.]);
            chain.metric().fail.set(0);
            // Repeated transitions exercise momentum refresh and pool reuse.
            for _ in 0..3 {
                chain.metric().calls.set(0);
                let info = chain.step(&mut rng).unwrap();
                assert_eq!(info.integration_steps, 15);
                assert_eq!(chain.metric().calls.get(), 31);
                assert_eq!(info.acceptance_probability, 1.);
            }
        }
    }
}
