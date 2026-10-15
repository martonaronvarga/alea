use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    CheesHmc, CheesHmcOptions, MarkovChain,
    adapt::{CheesCriterion, CheesEnsembleAdapter},
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
