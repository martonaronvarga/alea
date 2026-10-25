use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    MarkovChain,
    adapt::{CheesCriterion, CheesEnsembleAdapter},
    chees::{CheesHmc, CheesHmcOptions},
};
use alea_runtime::{
    chees::warmup,
    ensemble::{Chain, Execution, chain_rng},
    output::ChainId,
};

fn adapter() -> CheesEnsembleAdapter {
    CheesEnsembleAdapter::new(
        2,
        0.15.try_into().unwrap(),
        0.9,
        0.15,
        2.0,
        CheesCriterion::Global,
    )
    .unwrap()
}

fn chains(target: &Gaussian) -> Vec<Chain<CheesHmc<'_, Gaussian, IdentityMetric>, ()>> {
    let frozen = adapter().finish().unwrap();
    (0..4)
        .map(|id| {
            let chain_id = ChainId(id);
            Chain {
                id: chain_id,
                sampler: CheesHmc::new(
                    target,
                    OwnedBuffer::from_fn(2, |coordinate| {
                        (id as f64 + 1.0) * if coordinate == 0 { 0.2 } else { -0.1 }
                    }),
                    IdentityMetric::new(2),
                    CheesHmcOptions::new(frozen),
                )
                .unwrap(),
                rng: chain_rng([7; 32], chain_id),
                sink: (),
            }
        })
        .collect()
}

#[test]
fn shared_warmup_is_bitwise_scheduler_deterministic_and_freezes() {
    let target = Gaussian::new(2);
    let mut sequential = chains(&target);
    let mut parallel = chains(&target);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let (sequential_frozen, sequential_report) =
        warmup(&mut sequential, adapter(), 40, Execution::Sequential).unwrap();
    let (parallel_frozen, parallel_report) =
        warmup(&mut parallel, adapter(), 40, Execution::Parallel(&pool)).unwrap();

    assert_eq!(sequential_report.iterations, 40);
    assert_eq!(
        sequential_report.integration_steps,
        parallel_report.integration_steps
    );
    assert_eq!(
        sequential_frozen.step_size().value().to_bits(),
        parallel_frozen.step_size().value().to_bits()
    );
    assert_eq!(
        sequential_frozen.max_trajectory_time().to_bits(),
        parallel_frozen.max_trajectory_time().to_bits()
    );
    for (left, right) in sequential.iter().zip(&parallel) {
        assert_eq!(left.sampler.position(), right.sampler.position());
    }
}

struct ScaledGaussian;
impl alea_core::target::LogDensityGradient for ScaledGaussian {
    type Error = std::convert::Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let z = [(q[0] - 3.) / 20., (q[1] + 2.) / 0.05];
        g[0] = -z[0] / 20.;
        g[1] = -z[1] / 0.05;
        Ok(-0.5 * (z[0] * z[0] + z[1] * z[1]))
    }
}
// Each chain owns one lock and one RefCell workspace; no global numerical state.
struct LockedTransport<'a>(
    std::sync::Mutex<alea_core::transport::FrozenTransportTarget<'a, ScaledGaussian>>,
);
impl alea_core::target::LogDensityGradient for LockedTransport<'_> {
    type Error = alea_core::transport::FrozenTransportError<std::convert::Infallible>;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.0.lock().unwrap().logp_grad(q, g)
    }
}
fn whitened_targets(t: &ScaledGaussian) -> Vec<LockedTransport<'_>> {
    (0..4)
        .map(|_| {
            let map = alea_math::affine::LowRankAffineMap::new(
                OwnedBuffer::from_fn(2, |i| if i == 0 { 3. } else { -2. }),
                OwnedBuffer::from_fn(2, |i| if i == 0 { 20. } else { 0.05 }),
                OwnedBuffer::new(0),
                OwnedBuffer::new(0),
            )
            .unwrap();
            LockedTransport(std::sync::Mutex::new(
                alea_core::transport::FrozenTransportTarget::new(t, map).unwrap(),
            ))
        })
        .collect()
}
fn whitened_chains<'a>(
    targets: &'a [LockedTransport<'a>],
) -> Vec<Chain<CheesHmc<'a, LockedTransport<'a>, IdentityMetric>, ()>> {
    let frozen = adapter().finish().unwrap();
    targets
        .iter()
        .enumerate()
        .map(|(id, t)| {
            let id = ChainId(id as u64);
            Chain {
                id,
                sampler: CheesHmc::new(
                    t,
                    OwnedBuffer::from_fn(2, |k| 0.2 * (id.0 + 1) as f64 * (k + 1) as f64),
                    IdentityMetric::new(2),
                    CheesHmcOptions::new(frozen),
                )
                .unwrap(),
                rng: chain_rng([7; 32], id),
                sink: (),
            }
        })
        .collect()
}
#[test]
fn shared_frozen_whitening_replays_across_schedulers_and_preserves_latent_scores() {
    use alea_core::target::evaluate;
    let target = ScaledGaussian;
    let serial_targets = whitened_targets(&target);
    let mut gradient = [0.; 2];
    evaluate(&serial_targets[0], &[0.4, -0.7], &mut gradient).unwrap();
    assert!((gradient[0] + 0.4).abs() < 1e-13 && (gradient[1] - 0.7).abs() < 1e-13);
    let mut serial = whitened_chains(&serial_targets);
    let (expected, r) = warmup(
        &mut serial,
        adapter().with_max_integration_steps(128).unwrap(),
        80,
        Execution::Sequential,
    )
    .unwrap();
    for workers in [1, 2, 3] {
        let parallel_targets = whitened_targets(&target);
        let mut parallel = whitened_chains(&parallel_targets);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let (actual, report) = warmup(
            &mut parallel,
            adapter().with_max_integration_steps(128).unwrap(),
            80,
            Execution::Parallel(&pool),
        )
        .unwrap();
        assert_eq!(r.integration_steps, report.integration_steps);
        assert_eq!(
            expected.step_size().value().to_bits(),
            actual.step_size().value().to_bits()
        );
        assert_eq!(
            expected.max_trajectory_time().to_bits(),
            actual.max_trajectory_time().to_bits()
        );
        for (a, b) in serial.iter().zip(&parallel) {
            assert_eq!(a.sampler.position(), b.sampler.position());
        }
        for _ in 0..20 {
            for (a, b) in serial.iter_mut().zip(&mut parallel) {
                let _ = a.sampler.step(&mut a.rng).unwrap();
                let _ = b.sampler.step(&mut b.rng).unwrap();
                assert_eq!(a.sampler.position(), b.sampler.position());
            }
        }
        // Restore serial replay to its warmup boundary for the next scheduler.
        serial = whitened_chains(&serial_targets);
        let _ = warmup(
            &mut serial,
            adapter().with_max_integration_steps(128).unwrap(),
            80,
            Execution::Sequential,
        )
        .unwrap();
    }
}
