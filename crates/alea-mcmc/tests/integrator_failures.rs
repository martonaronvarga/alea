//! Faults are injected by evaluation index, including after a successful stage.
use alea_core::target::{EvaluationError, LogDensityGradient, PointState};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Hmc, HmcOptions,
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
    hmc::HmcError,
    integrator::{Integrator, Leapfrog, TwoStage},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

#[derive(Debug, Clone, Copy)]
enum Fault {
    Model,
    Panic,
    Partial,
    Gradient,
    LogDensity,
}
const FAULTS: [Fault; 5] = [
    Fault::Model,
    Fault::Panic,
    Fault::Partial,
    Fault::Gradient,
    Fault::LogDensity,
];

#[derive(Debug, thiserror::Error)]
#[error("injected failure at evaluation {0}")]
struct BackendError(usize);

#[derive(Debug, Default)]
struct Target {
    calls: Cell<usize>,
    fault: Cell<Option<(usize, Fault)>>,
}
impl Target {
    fn arm(&self, offset: usize, fault: Fault) {
        self.fault.set(Some((self.calls.get() + offset, fault)));
    }
}
impl LogDensityGradient for Target {
    type Error = BackendError;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, BackendError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if let Some((at, fault)) = self.fault.get()
            && call == at
        {
            g[0] = 42.0; // Deliberately dirty scratch before returning/unwinding.
            return match fault {
                Fault::Model => Err(BackendError(call)),
                Fault::Panic => panic!("injected target panic"),
                Fault::Partial => Ok(0.0),
                Fault::Gradient => {
                    g[1] = f64::INFINITY;
                    Ok(0.0)
                }
                Fault::LogDensity => {
                    g[1] = 0.0;
                    Ok(f64::NEG_INFINITY)
                }
            };
        }
        for (g, q) in g.iter_mut().zip(q) {
            *g = -q;
        }
        Ok(-0.5 * q.iter().map(|q| q * q).sum::<f64>())
    }
}
fn buf(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}
fn same_point(a: &PointState<'_, Target>, b: &PointState<'_, Target>) {
    // Transactional preservation is bit-exact, including signed zeros.
    assert!(
        a.position()
            .iter()
            .map(|v| v.to_bits())
            .eq(b.position().iter().map(|v| v.to_bits()))
    );
    assert!(
        a.gradient()
            .iter()
            .map(|v| v.to_bits())
            .eq(b.gradient().iter().map(|v| v.to_bits()))
    );
    assert_eq!(a.log_density().to_bits(), b.log_density().to_bits());
    assert!(std::ptr::eq(a.target(), b.target()));
}
fn divergence(fault: Fault) -> Divergence {
    match fault {
        Fault::Partial | Fault::Gradient => Divergence::Gradient { index: 1 },
        Fault::LogDensity => Divergence::LogDensity,
        Fault::Model | Fault::Panic => panic!("not a numerical divergence"),
    }
}

fn phase_failures<I: Integrator>(integrator: I) {
    for sign in [-1.0, 1.0] {
        for stage in 1..=integrator.stages() {
            for fault in FAULTS {
                let target = Target::default();
                let initial = PhaseState::new(
                    PointState::new(&target, buf(&[0.7, -0.2])).unwrap(),
                    buf(&[-0.3, 0.6]),
                )
                .unwrap();
                let before = initial.clone();
                let metric = IdentityMetric::new(2);
                let mut workspace = PhaseWorkspace::new(initial.point());
                let step = SignedStep::new(sign * 0.1).unwrap();
                let calls = target.calls.get();
                target.arm(stage, fault);
                let phase = workspace.start_from_phase(&initial, &metric).unwrap();
                let outcome = catch_unwind(AssertUnwindSafe(|| phase.step_with(step, &integrator)));
                match (fault, outcome) {
                    (Fault::Panic, Err(_)) => {}
                    (
                        Fault::Model,
                        Ok(Err(PhaseError::Evaluation(EvaluationError::Model(BackendError(at))))),
                    ) => assert_eq!(at, calls + stage),
                    (
                        Fault::Partial | Fault::Gradient | Fault::LogDensity,
                        Ok(Err(PhaseError::Divergence(reason))),
                    ) => assert_eq!(reason, divergence(fault)),
                    _ => panic!("wrong phase failure classification"),
                }
                assert_eq!(target.calls.get(), calls + stage);
                same_point(initial.point(), before.point());
                assert_eq!(initial.momentum(), before.momentum());

                target.fault.set(None);
                let calls = target.calls.get();
                let mut fresh = PhaseWorkspace::new(initial.point());
                let recovered = workspace
                    .start_from_phase(&initial, &metric)
                    .unwrap()
                    .step_with(step, &integrator)
                    .unwrap();
                let expected = fresh
                    .start_from_phase(&initial, &metric)
                    .unwrap()
                    .step_with(step, &integrator)
                    .unwrap();
                same_point(recovered.point(), expected.point());
                assert_eq!(recovered.momentum(), expected.momentum());
                assert_eq!(target.calls.get() - calls, 2 * integrator.stages());
            }
        }
    }
}

#[test]
fn each_internal_stage_failure_consumes_phase_and_restart_discards_partial_work() {
    phase_failures(Leapfrog);
    phase_failures(TwoStage::OMF);
    phase_failures(TwoStage::BCSS);
}

fn chain_failures<I: Integrator + Copy>(integrator: I) {
    for evaluation in 1..=3 * integrator.stages() {
        for fault in FAULTS {
            let target = Target::default();
            let options = HmcOptions::new(0.1, 3).unwrap();
            let mut chain = Hmc::new_with_integrator(
                &target,
                buf(&[0.7, -0.2]),
                IdentityMetric::new(2),
                options,
                integrator,
            )
            .unwrap();
            let before = chain.point().clone();
            let mut rng = SmallRng::seed_from_u64(619);
            let calls = target.calls.get();
            target.arm(evaluation, fault);
            let outcome = catch_unwind(AssertUnwindSafe(|| chain.step(&mut rng)));
            let macrostep = (evaluation - 1) / integrator.stages() + 1;
            match (fault, outcome) {
                (Fault::Panic, Err(_)) => {}
                (
                    Fault::Model,
                    Ok(Err(HmcError::Evaluation {
                        integration_step,
                        source: EvaluationError::Model(BackendError(at)),
                    })),
                ) => {
                    assert_eq!(integration_step, macrostep);
                    assert_eq!(at, calls + evaluation);
                }
                (Fault::Partial | Fault::Gradient | Fault::LogDensity, Ok(Ok(info))) => {
                    assert!(!info.accepted);
                    assert_eq!(info.divergence, Some(divergence(fault)));
                    assert_eq!(info.integration_steps, macrostep);
                    assert_eq!(info.acceptance_probability, 0.0);
                    assert!(info.initial_energy.is_some());
                    assert!(info.proposal_energy.is_none());
                    assert!(info.energy_error.is_none());
                }
                _ => panic!("wrong HMC failure classification"),
            }
            assert_eq!(target.calls.get() - calls, evaluation);
            same_point(chain.point(), &before);
            target.fault.set(None);
            // Compare recovery at the already-consumed RNG position, not the old seed.
            let mut control_rng = rng.clone();
            let mut control = Hmc::new_with_integrator(
                &target,
                buf(before.position()),
                IdentityMetric::new(2),
                options,
                integrator,
            )
            .unwrap();
            let calls = target.calls.get();
            let info = chain.step(&mut rng).unwrap();
            let expected = control.step(&mut control_rng).unwrap();
            assert_eq!(info.accepted, expected.accepted);
            assert_eq!(info.divergence, expected.divergence);
            #[cfg(not(miri))]
            same_point(chain.point(), control.point());
            // Miri perturbs log/sqrt in momentum generation independently even
            // for cloned RNGs. Cache preservation above stays bit-exact; fresh
            // stochastic transitions have a numerical (not replay) contract here.
            #[cfg(miri)]
            for (actual, expected) in chain
                .point()
                .position()
                .iter()
                .zip(control.point().position())
                .chain(
                    chain
                        .point()
                        .gradient()
                        .iter()
                        .zip(control.point().gradient()),
                )
                .chain(std::iter::once((
                    &chain.point().log_density(),
                    &control.point().log_density(),
                )))
            {
                assert!((actual - expected).abs() < 1e-12 * (1.0 + expected.abs()));
            }
            assert_eq!(target.calls.get() - calls, 6 * integrator.stages());
        }
    }
}

#[test]
fn hmc_classifies_failures_at_every_stage_without_committing_or_extra_evaluations() {
    chain_failures(Leapfrog);
    chain_failures(TwoStage::OMF);
    chain_failures(TwoStage::BCSS);
}

#[cfg(not(miri))]
#[test]
fn failed_internal_stages_do_not_allocate() {
    let target = Target::default();
    let mut chain = Hmc::new_with_integrator(
        &target,
        buf(&[0.7, -0.2]),
        IdentityMetric::new(2),
        HmcOptions::new(0.1, 3).unwrap(),
        TwoStage::OMF,
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(619);
    let measured = allocation_counter::measure(|| {
        for evaluation in 1..=6 {
            for fault in [
                Fault::Model,
                Fault::Partial,
                Fault::Gradient,
                Fault::LogDensity,
            ] {
                target.arm(evaluation, fault);
                let result = chain.step(&mut rng);
                assert!(result.is_err() || result.unwrap().divergence.is_some());
            }
        }
    });
    assert_eq!(measured.count_total, 0);
}

#[cfg(not(miri))]
mod properties {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, rng_seed: proptest::test_runner::RngSeed::Fixed(831), ..ProptestConfig::default() })]
        #[test]
        fn two_stage_signed_roundtrip(q in prop::array::uniform2(-2.0..2.0), p in prop::array::uniform2(-2.0..2.0), h in 0.001..0.2, lambda in 0.0..0.5, steps in 1usize..16) {
            let target = Target::default();
            let point = PointState::new(&target, buf(&q)).unwrap();
            let initial = PhaseState::new(point, buf(&p)).unwrap();
            let metric = IdentityMetric::new(2);
            let integrator = TwoStage::new(lambda).unwrap();
            let mut workspace = PhaseWorkspace::new(initial.point());
            let mut phase = workspace.start_from_phase(&initial, &metric).unwrap();
            for sign in [1.0, -1.0] {
                for _ in 0..steps { phase = phase.step_with(SignedStep::new(sign * h).unwrap(), &integrator).unwrap(); }
            }
            for (actual, expected) in phase.point().position().iter().zip(q).chain(phase.momentum().iter().zip(p)) {
                prop_assert!((actual - expected).abs() < 1e-11 * (1.0 + expected.abs()));
            }
            prop_assert_eq!(target.calls.get(), 1 + 4 * steps);
        }
    }
}
