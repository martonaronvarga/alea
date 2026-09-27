use std::{cell::Cell, error::Error};

use kernels::{
    buffer::OwnedBuffer,
    density::FusedLogDensity,
    dist::Gaussian,
    metric::{CholeskyFactor, DenseMetric, IdentityMetric},
    target::{EvaluationError, FusedAdapter, LogDensityGradient},
};
use rand::{SeedableRng, rngs::SmallRng};
use runtime::mcmc::hmc_chain::{
    Divergence, HmcChain, HmcConfigError, HmcError, HmcOptions, StepSize,
};

#[derive(Debug, thiserror::Error)]
#[error("injected backend failure")]
struct Failure;

#[derive(Clone, Copy)]
enum Fault {
    Error,
    Panic,
    Log,
    Gradient,
    Partial,
}

struct Target {
    calls: Cell<usize>,
    fail_at: Cell<usize>,
    fault: Cell<Fault>,
    dimension: Cell<usize>,
}

impl Target {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            fail_at: Cell::new(usize::MAX),
            fault: Cell::new(Fault::Error),
            dimension: Cell::new(2),
        }
    }
}

impl LogDensityGradient for Target {
    type Error = Failure;
    fn dimension(&self) -> usize {
        self.dimension.get()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.fail_at.get() {
            g[0] = 123.0;
            match self.fault.get() {
                Fault::Error => return Err(Failure),
                Fault::Panic => panic!("injected model panic"),
                Fault::Log => {
                    g.fill(0.0);
                    return Ok(f64::NEG_INFINITY);
                }
                Fault::Gradient => {
                    g.fill(f64::INFINITY);
                    return Ok(0.0);
                }
                Fault::Partial => return Ok(0.0),
            }
        }
        Ok(Gaussian.log_prob_and_grad(q, g))
    }
}

fn initial() -> OwnedBuffer {
    OwnedBuffer::from_fn(2, |i| [0.5, -0.3][i])
}

#[test]
fn settings_validate_once_and_cannot_store_invalid_values() {
    for bad in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(StepSize::try_from(bad), Err(HmcConfigError::StepSize));
        assert!(matches!(
            HmcOptions::new(bad, 4),
            Err(HmcConfigError::StepSize)
        ));
        assert!(matches!(
            HmcOptions::default().with_max_energy_error(bad),
            Err(HmcConfigError::EnergyErrorLimit)
        ));
    }
    assert!(matches!(
        HmcOptions::new(0.1, 0),
        Err(HmcConfigError::LeapfrogCount)
    ));
    let options = HmcOptions::new(0.2, 4)
        .unwrap()
        .with_max_energy_error(10.0)
        .unwrap();
    assert_eq!(options.step_size().value(), 0.2);
    assert_eq!(options.leapfrog_steps().get(), 4);
    assert_eq!(options.max_energy_error(), 10.0);
}

#[test]
fn construction_and_contract_errors_do_not_evaluate_or_advance_rng() {
    let target = Target::new();
    assert!(matches!(
        HmcChain::new(
            &target,
            initial(),
            IdentityMetric::new(3),
            HmcOptions::default()
        ),
        Err(HmcError::MetricDimension { .. })
    ));
    assert_eq!(target.calls.get(), 0);
    let empty = FusedAdapter::new(&Gaussian, 0);
    assert!(matches!(
        HmcChain::new(
            &empty,
            OwnedBuffer::new(0),
            IdentityMetric::new(0),
            HmcOptions::default()
        ),
        Err(HmcError::EmptyTarget)
    ));
    assert!(matches!(
        HmcChain::new(
            &target,
            OwnedBuffer::new(1),
            IdentityMetric::new(2),
            HmcOptions::default()
        ),
        Err(HmcError::Evaluation {
            leapfrog_step: 0,
            ..
        })
    ));
    assert_eq!(target.calls.get(), 0);
    let mut chain = HmcChain::new(
        &target,
        initial(),
        IdentityMetric::new(2),
        HmcOptions::default(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(54);
    let before = rng.clone();
    target.dimension.set(3);
    assert!(matches!(
        chain.step(&mut rng),
        Err(HmcError::Evaluation {
            leapfrog_step: 0,
            source: EvaluationError::TargetDimensionChanged { .. }
        })
    ));
    assert_eq!(rng, before);
    assert_eq!(target.calls.get(), 1);
    assert_eq!(chain.point().position(), &[0.5, -0.3]);
}

#[test]
fn exactly_l_evaluations_per_trajectory_and_reset_refreshes_once() {
    let target = Target::new();
    let mut chain = HmcChain::new(
        &target,
        initial(),
        IdentityMetric::new(2),
        HmcOptions::new(0.15, 4).unwrap(),
    )
    .unwrap();
    assert_eq!(target.calls.get(), 1);
    let mut rng = SmallRng::seed_from_u64(85);
    for transition in 1..=20 {
        let old_q = chain.point().position().to_vec();
        let old_g = chain.point().gradient().to_vec();
        let old_lp = chain.point().log_density();
        let info = chain.step(&mut rng).unwrap();
        assert!(info.divergence.is_none());
        assert_eq!(info.leapfrog_steps, 4);
        assert_eq!(target.calls.get(), 1 + transition * 4);
        // Transcendental results need not be bit-identical across evaluations
        // (including Miri's floating-point implementation).
        let expected_probability = (-info.energy_error.unwrap()).min(0.0).exp();
        assert!((info.acceptance_probability - expected_probability).abs() < 1e-14);
        if !info.accepted {
            assert_eq!(chain.point().position(), old_q);
            assert_eq!(chain.point().gradient(), old_g);
            assert_eq!(chain.point().log_density(), old_lp);
        }
        let mut gradient = [0.0; 2];
        assert_eq!(
            chain.point().log_density(),
            Gaussian.log_prob_and_grad(chain.point().position(), &mut gradient)
        );
        assert_eq!(chain.point().gradient(), gradient);
    }
    chain.set_position(&[0.0; 2]).unwrap();
    assert_eq!(target.calls.get(), 82);
    assert_eq!(chain.point().gradient(), &[0.0; 2]);
    assert!(chain.set_position(&[f64::NAN; 2]).is_err());
    assert_eq!(target.calls.get(), 82);
    assert_eq!(chain.point().position(), &[0.0; 2]);
}

#[test]
fn late_model_error_and_panic_leave_live_cache_unchanged_and_recover() {
    for fault in [Fault::Error, Fault::Panic] {
        let target = Target::new();
        let mut chain = HmcChain::new(
            &target,
            initial(),
            IdentityMetric::new(2),
            HmcOptions::new(0.1, 4).unwrap(),
        )
        .unwrap();
        target.fail_at.set(4); // Two proposal evaluations succeed before failure.
        target.fault.set(fault);
        let mut rng = SmallRng::seed_from_u64(65);
        let position_ptr = chain.point().position().as_ptr();
        let gradient_ptr = chain.point().gradient().as_ptr();
        let value = chain.point().log_density();
        if matches!(fault, Fault::Panic) {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| chain.step(&mut rng)))
                    .is_err()
            );
        } else {
            let error = chain.step(&mut rng).unwrap_err();
            assert!(matches!(
                error,
                HmcError::Evaluation {
                    leapfrog_step: 3,
                    source: EvaluationError::Model(Failure)
                }
            ));
            assert!(
                error
                    .source()
                    .unwrap()
                    .source()
                    .unwrap()
                    .downcast_ref::<Failure>()
                    .is_some()
            );
        }
        assert_eq!(target.calls.get(), 4);
        assert_eq!(chain.point().position(), &[0.5, -0.3]);
        assert_eq!(chain.point().gradient(), &[-0.5, 0.3]);
        assert_eq!(chain.point().log_density(), value);
        assert_eq!(chain.point().position().as_ptr(), position_ptr);
        assert_eq!(chain.point().gradient().as_ptr(), gradient_ptr);
        assert!(chain.step(&mut rng).unwrap().divergence.is_none());
        assert_eq!(target.calls.get(), 8);
    }
}

#[test]
fn invalid_target_outputs_are_typed_divergences_without_commit() {
    for (fault, expected) in [
        (Fault::Log, Divergence::LogDensity),
        (Fault::Gradient, Divergence::Gradient { index: 0 }),
        (Fault::Partial, Divergence::Gradient { index: 1 }),
    ] {
        let target = Target::new();
        let mut chain = HmcChain::new(
            &target,
            initial(),
            IdentityMetric::new(2),
            HmcOptions::new(0.1, 4).unwrap(),
        )
        .unwrap();
        target.fail_at.set(3);
        target.fault.set(fault);
        let old_lp = chain.point().log_density();
        let info = chain.step(&mut SmallRng::seed_from_u64(8)).unwrap();
        assert_eq!(info.divergence, Some(expected));
        assert!(!info.accepted);
        assert_eq!(info.acceptance_probability, 0.0);
        assert_eq!(info.leapfrog_steps, 2);
        assert!(info.proposal_energy.is_none());
        assert_eq!(chain.point().position(), &[0.5, -0.3]);
        assert_eq!(chain.point().gradient(), &[-0.5, 0.3]);
        assert_eq!(chain.point().log_density(), old_lp);
    }
}

#[test]
fn finite_energy_divergence_and_ordinary_rejection_are_distinct() {
    let target = FusedAdapter::new(&Gaussian, 2);
    let mut divergent = HmcChain::new(
        &target,
        initial(),
        IdentityMetric::new(2),
        HmcOptions::new(0.5, 3)
            .unwrap()
            .with_max_energy_error(1e-30)
            .unwrap(),
    )
    .unwrap();
    let info = divergent.step(&mut SmallRng::seed_from_u64(13)).unwrap();
    assert_eq!(info.divergence, Some(Divergence::EnergyErrorLimit));
    assert_eq!(info.acceptance_probability, 0.0);
    assert!(info.energy_error.unwrap().abs() > 1e-30);
    assert_eq!(divergent.point().position(), &[0.5, -0.3]);

    let mut rejected = 0;
    for seed in 0..20 {
        let mut chain = HmcChain::new(
            &target,
            initial(),
            IdentityMetric::new(2),
            HmcOptions::new(3.0, 2)
                .unwrap()
                .with_max_energy_error(1e100)
                .unwrap(),
        )
        .unwrap();
        let value = chain.point().log_density();
        let info = chain.step(&mut SmallRng::seed_from_u64(seed)).unwrap();
        assert!(info.divergence.is_none());
        if !info.accepted {
            rejected += 1;
            assert_eq!(chain.point().position(), &[0.5, -0.3]);
            assert_eq!(chain.point().gradient(), &[-0.5, 0.3]);
            assert_eq!(chain.point().log_density(), value);
        }
    }
    assert!(rejected > 0);
}

struct Correlated;
impl LogDensityGradient for Correlated {
    type Error = std::convert::Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        g[0] = -1.25 * q[0] + 0.75 * q[1];
        g[1] = 0.75 * q[0] - 1.25 * q[1];
        Ok(0.5 * (q[0] * g[0] + q[1] * g[1]))
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "statistical test runs natively; Miri covers deterministic transitions"
)]
fn correlated_gaussian_moments_with_dense_mass() {
    let factor =
        CholeskyFactor::new_lower(2, OwnedBuffer::from_fn(4, |i| [2.0, 0.5, 0.0, 1.3][i])).unwrap();
    let mut chain = HmcChain::new(
        &Correlated,
        OwnedBuffer::new(2),
        DenseMetric::new(factor),
        HmcOptions::new(0.2, 12).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(6539);
    let mut sums = [0.0; 2];
    let mut squares = [0.0; 2];
    let mut cross = 0.0;
    for i in 0..22_000 {
        let info = chain.step(&mut rng).unwrap();
        assert!(info.divergence.is_none());
        if i >= 2000 {
            let q = chain.point().position();
            for j in 0..2 {
                sums[j] += q[j];
                squares[j] += q[j] * q[j];
            }
            cross += q[0] * q[1];
        }
    }
    for j in 0..2 {
        assert!((sums[j] / 20_000.0).abs() < 0.08);
        assert!((squares[j] / 20_000.0 - 1.25).abs() < 0.08);
    }
    assert!((cross / 20_000.0 - 0.75).abs() < 0.08);
}
