use alea_core::target::{EvaluationError, LogDensityGradient};
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    adapt::{
        MclmcWarmupOptions, StaticWarmupOptions, warmup_mams_step_size, warmup_mclmc_step_size,
    },
    mams::{
        ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget, Mams, MamsError, MamsOptions,
    },
};
use rand::{SeedableRng, rngs::SmallRng};
use std::cell::Cell;
#[derive(Debug, thiserror::Error)]
#[error("original backend failure")]
struct Backend;
struct Target {
    mode: Cell<u8>,
    calls: Cell<usize>,
}
impl LogDensityGradient for Target {
    type Error = Backend;
    fn dimension(&self) -> usize {
        3
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Backend> {
        self.calls.set(self.calls.get() + 1);
        for (g, q) in g.iter_mut().zip(q) {
            *g = -*q;
        }
        match self.mode.get() {
            1 => Ok(f64::NEG_INFINITY),
            2 => Ok(f64::INFINITY),
            3 => Ok(f64::NAN),
            4 => {
                g[1] = f64::NAN;
                Ok(0.)
            }
            5 => Err(Backend),
            6 => {
                self.mode.set(0);
                Ok(f64::NAN)
            }
            _ => Ok(-0.5 * q.iter().map(|x| x * x).sum::<f64>()),
        }
    }
}
fn target() -> Target {
    Target {
        mode: Cell::new(0),
        calls: Cell::new(0),
    }
}
#[test]
fn mams_rejects_all_numerical_outputs_but_preserves_backend_errors_and_cache() {
    for mode in 1..=5 {
        let t = target();
        let mut rng = SmallRng::seed_from_u64(99);
        let mut chain =
            Mams::new(&t, OwnedBuffer::new(3), MamsOptions::new(0.1, 4).unwrap()).unwrap();
        t.mode.set(mode);
        let result = chain.step(&mut rng);
        if mode == 5 {
            assert!(matches!(
                result,
                Err(MamsError::Evaluation {
                    source: EvaluationError::Model(Backend),
                    ..
                })
            ));
        } else {
            let o = result.unwrap();
            assert!(!o.accepted);
            assert_eq!(o.acceptance_probability, 0.);
            assert!(o.divergence.is_some());
            assert_eq!(o.integration_steps, 1);
        }
        assert_eq!(chain.point().position(), &[0.; 3]);
        assert_eq!(chain.point().gradient(), &[0.; 3]);
        assert_eq!(chain.point().log_density(), 0.);
        assert_eq!(t.calls.get(), 2);
    }
}
#[test]
fn mclmc_recovers_only_in_warmup_and_charges_failed_macrosteps() {
    let t = target();
    let mut rng = SmallRng::seed_from_u64(99);
    let chain = ApproximateMclmc::new(
        ApproximateTarget::acknowledge_bias(),
        &t,
        OwnedBuffer::new(3),
        ApproximateMclmcOptions::new(0.1, 4, Some(2.)).unwrap(),
        &mut rng,
    )
    .unwrap();
    t.mode.set(6);
    let (mut chain, r) =
        warmup_mclmc_step_size(chain, MclmcWarmupOptions::new(100), &mut rng).unwrap();
    assert_eq!(r.recovered_numerical_failures, 1);
    assert_eq!(r.integration_attempts, 1 + 99 * 4);
    let before = chain.point().clone();
    t.mode.set(3);
    assert!(chain.step(&mut rng).is_err());
    assert_eq!(chain.point().position(), before.position());
    assert_eq!(chain.point().gradient(), before.gradient());
    assert_eq!(chain.point().log_density(), before.log_density());
    t.mode.set(5);
    assert!(matches!(
        warmup_mclmc_step_size(chain, MclmcWarmupOptions::new(10), &mut rng),
        Err(alea_mcmc::adapt::StaticWarmupError::Kernel(
            MamsError::Evaluation {
                source: EvaluationError::Model(Backend),
                ..
            }
        ))
    ));
}
#[test]
fn mams_scale_duration_adapts_count_and_freezes_it_before_retention() {
    let t = target();
    let mut rng = SmallRng::seed_from_u64(99);
    let options = MamsOptions::new(0.1, 1)
        .unwrap()
        .with_trajectory_time(0.25, 32)
        .unwrap();
    assert_eq!(options.integration_steps().get(), 3);
    let chain = Mams::new(&t, OwnedBuffer::new(3), options).unwrap();
    let (mut chain, r) =
        warmup_mams_step_size(chain, StaticWarmupOptions::new(100), &mut rng).unwrap();
    let n = (0.25 / r.step_size.value()).ceil().clamp(1., 32.) as usize;
    assert_eq!(chain.options().integration_steps().get(), n);
    for _ in 0..20 {
        let _ = chain.step(&mut rng).unwrap();
        assert_eq!(chain.options().integration_steps().get(), n);
        assert_eq!(chain.options().step_size(), r.step_size);
    }
    for (time, cap) in [(0., 32), (f64::NAN, 32), (1., 0), (1., 1025)] {
        assert!(
            MamsOptions::new(0.1, 1)
                .unwrap()
                .with_trajectory_time(time, cap)
                .is_err()
        );
    }
}
