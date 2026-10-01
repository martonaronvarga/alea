use alea_core::target::{LogDensityGradient, PointState};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Hmc, HmcOptions,
    hamiltonian::{PhaseState, PhaseWorkspace, SignedStep},
    integrator::{Integrator, Leapfrog, TwoStage},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible};
struct Normal {
    calls: Cell<usize>,
}
impl LogDensityGradient for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        self.calls.set(self.calls.get() + 1);
        g[0] = -q[0];
        Ok(-0.5 * q[0] * q[0])
    }
}
fn check<I: Integrator>(integrator: I) {
    let target = Normal {
        calls: Cell::new(0),
    };
    let point = PointState::new(&target, OwnedBuffer::from_fn(1, |_| 0.7)).unwrap();
    let snapshot = PhaseState::new(point.clone(), OwnedBuffer::from_fn(1, |_| -0.3)).unwrap();
    let metric = IdentityMetric::new(1);
    let mut workspace = PhaseWorkspace::new(&point);
    let mut phase = workspace.start_from_phase(&snapshot, &metric).unwrap();
    let calls = target.calls.get();
    let mut q = 0.7;
    let mut p = -0.3;
    for _ in 0..10 {
        for i in 0..integrator.stages() {
            p -= integrator.kick(i) * 0.1 * q;
            q += integrator.drift(i) * 0.1 * p;
        }
        p -= integrator.kick(integrator.stages()) * 0.1 * q;
        phase = phase
            .step_with(SignedStep::new(0.1).unwrap(), &integrator)
            .unwrap();
    }
    assert!((phase.point().position()[0] - q).abs() < 1e-13);
    assert!((phase.momentum()[0] - p).abs() < 1e-13);
    assert_eq!(target.calls.get() - calls, 10 * integrator.stages());
    for _ in 0..10 {
        phase = phase
            .step_with(SignedStep::new(-0.1).unwrap(), &integrator)
            .unwrap();
    }
    assert!((phase.point().position()[0] - 0.7).abs() < 1e-12);
    assert!((phase.momentum()[0] + 0.3).abs() < 1e-12);
    assert_eq!(point.position(), &[0.7]);
    let expected_calls = 10 * integrator.stages();
    let mut chain = Hmc::new_with_integrator(
        &target,
        OwnedBuffer::new(1),
        metric,
        HmcOptions::new(0.15, 10).unwrap(),
        integrator,
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(726);
    let calls = target.calls.get();
    let result = chain.step(&mut rng).unwrap();
    assert_eq!(result.integration_steps, 10);
    assert!(result.divergence.is_none());
    assert_eq!(target.calls.get() - calls, expected_calls);
}
#[test]
fn complete_macrosteps_match_scalar_oracle_reverse_and_count_gradients() {
    check(Leapfrog);
    check(TwoStage::OMF);
    check(TwoStage::BCSS);
    check(TwoStage::new(0.25).unwrap());
}
#[test]
fn coefficients_reject_invalid_values() {
    for value in [-0.1, 0.6, f64::NAN, f64::INFINITY] {
        assert!(TwoStage::new(value).is_err());
    }
}
#[cfg(not(miri))]
#[test]
fn two_stage_hmc_has_gaussian_moments_and_allocation_free_transitions() {
    let target = Normal {
        calls: Cell::new(0),
    };
    for integrator in [TwoStage::OMF, TwoStage::BCSS] {
        let mut chain = Hmc::new_with_integrator(
            &target,
            OwnedBuffer::new(1),
            IdentityMetric::new(1),
            HmcOptions::new(0.2, 8).unwrap(),
            integrator,
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(811);
        let mut sum = 0.0;
        let mut square = 0.0;
        let info = allocation_counter::measure(|| {
            for _ in 0..10000 {
                let transition = chain.step(&mut rng).unwrap();
                assert!(transition.divergence.is_none());
                let q = chain.point().position()[0];
                sum += q;
                square += q * q;
            }
        });
        assert_eq!(info.count_total, 0);
        assert!((sum / 10000.0).abs() < 0.05);
        assert!((square / 10000.0 - 1.0).abs() < 0.08);
    }
}
