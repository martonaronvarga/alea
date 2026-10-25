use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    HmcOptions,
    adapt::{FisherHmcWarmup, FisherOptions, HmcWarmup, WarmupOptions},
    walnuts::{Walnuts, WalnutsError, WalnutsOptions, WalnutsTermination},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

#[derive(Debug, thiserror::Error)]
#[error("injected target failure")]
struct Injected;

struct Target {
    calls: Cell<usize>,
    fail_at: Cell<usize>,
    mode: Cell<u8>,
}

impl Target {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            fail_at: Cell::new(0),
            mode: Cell::new(0),
        }
    }
}

impl LogDensityGradient for Target {
    type Error = Injected;

    fn dimension(&self) -> usize {
        2
    }

    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        if self.mode.get() == 3 && position.iter().any(|value| *value != 0.0) {
            gradient.fill(0.0);
            return Ok(f64::NAN);
        }
        if self.calls.get() == self.fail_at.get() {
            gradient[0] = 123.0;
            match self.mode.get() {
                1 => return Err(Injected),
                2 => panic!("injected target panic"),
                _ => {}
            }
        }
        for (gradient, position) in gradient.iter_mut().zip(position) {
            *gradient = -*position;
        }
        Ok(-0.5 * position.iter().map(|value| value * value).sum::<f64>())
    }
}

fn chain<'a>(target: &'a Target) -> Walnuts<'a, Target, IdentityMetric> {
    Walnuts::new(
        target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        WalnutsOptions::new(0.1, 3).unwrap(),
    )
    .unwrap()
}

#[test]
fn model_errors_and_panics_preserve_the_live_cache_and_recover() {
    for mode in [1, 2] {
        let target = Target::new();
        let mut chain = chain(&target);
        target.mode.set(mode);
        target.fail_at.set(2);
        let mut rng = SmallRng::seed_from_u64(17);
        let outcome = catch_unwind(AssertUnwindSafe(|| chain.step(&mut rng)));
        if mode == 1 {
            assert!(matches!(
                outcome.unwrap(),
                Err(WalnutsError::Phase { macro_step: 1, .. })
            ));
        } else {
            assert!(outcome.is_err());
        }
        assert_eq!(chain.point().position(), &[0.0, 0.0]);
        assert_eq!(chain.point().gradient(), &[-0.0, -0.0]);
        target.fail_at.set(0);
        target.mode.set(0);
        let info = chain.step(&mut rng).unwrap();
        assert!(info.valid_states > 1);
    }
}

#[test]
fn exhausted_numerical_refinement_is_typed_and_does_not_commit() {
    let target = Target::new();
    let mut chain = chain(&target);
    target.mode.set(3);
    target.fail_at.set(2);
    let info = chain.step(&mut SmallRng::seed_from_u64(11)).unwrap();
    assert_eq!(info.termination, WalnutsTermination::RefinementLimit);
    assert_eq!(info.macro_steps, 1);
    assert_eq!(info.valid_states, 1);
    assert!(!info.accepted);
    assert_eq!(chain.point().position(), &[0.0, 0.0]);
    assert_eq!(chain.point().gradient(), &[-0.0, -0.0]);
}

#[test]
fn covariance_and_fisher_warmup_handoffs_do_not_reevaluate_or_reseed() {
    let target = Target::new();
    let hmc_options = HmcOptions::new(0.1, 5).unwrap();
    let walnuts_options = WalnutsOptions::new(0.2, 5).unwrap();
    let mut rng = SmallRng::seed_from_u64(29);

    let (hmc, _) = HmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        hmc_options,
        WarmupOptions::new(200),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    let calls = target.calls.get();
    let position = hmc.point().position().to_vec();
    let replay = rng.clone();
    let walnuts = Walnuts::from_hmc(hmc, walnuts_options).unwrap();
    assert_eq!(target.calls.get(), calls);
    assert_eq!(walnuts.point().position(), position);
    assert_eq!(rng, replay);

    let (hmc, _) = FisherHmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        hmc_options,
        FisherOptions::new(200),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    let calls = target.calls.get();
    let position = hmc.point().position().to_vec();
    let replay = rng.clone();
    let walnuts = Walnuts::from_hmc(hmc, walnuts_options).unwrap();
    assert_eq!(target.calls.get(), calls);
    assert_eq!(walnuts.point().position(), position);
    assert_eq!(rng, replay);
}
