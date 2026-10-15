use alea_mcmc::MarkovChain;
use alea_runtime::{
    ensemble::{self, Chain, Execution, RunError, RunOptions, Termination, chain_rng},
    monitor::{Monitor, StopPolicy},
    output::{ChainId, Draw, Phase, TransitionStats},
};
use std::{
    num::NonZeroUsize,
    ops::ControlFlow,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

fn options(draws: usize) -> RunOptions {
    RunOptions {
        warmup: 7,
        draws,
        chunk: NonZeroUsize::new(16).unwrap(),
        window: NonZeroUsize::new(64).unwrap(),
        coordinates: vec![0, 1],
    }
}

#[derive(Debug, thiserror::Error)]
#[error("injected error")]
struct Failure;
struct TestChain {
    q: [f64; 2],
    steps: usize,
    fail: usize,
    panic: usize,
}
impl TestChain {
    fn new() -> Self {
        Self {
            q: [0.0; 2],
            steps: 0,
            fail: usize::MAX,
            panic: usize::MAX,
        }
    }
}
impl MarkovChain for TestChain {
    type Error = Failure;
    type Transition = TransitionStats;
    fn position(&self) -> &[f64] {
        &self.q
    }
    fn step<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Failure> {
        assert_ne!(self.steps, self.panic, "injected panic");
        if self.steps == self.fail {
            return Err(Failure);
        }
        self.steps += 1;
        for x in &mut self.q {
            *x = (rng.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
        }
        Ok(TransitionStats {
            accepted: true,
            acceptance_probability: 1.0,
            energy: Some(self.q[0]),
            divergent: Some(false),
            depth_saturated: Some(false),
            ..TransitionStats::default()
        })
    }
}

type Records = Arc<Mutex<Vec<(ChainId, Phase, usize, Vec<f64>)>>>;
fn chains(
    records: &Records,
) -> Vec<Chain<TestChain, impl for<'a> FnMut(Draw<'a, TransitionStats>) -> ControlFlow<()> + Send>>
{
    (0..4)
        .map(|i| {
            let records = Arc::clone(records);
            Chain {
                id: ChainId(i),
                sampler: TestChain::new(),
                rng: chain_rng([19; 32], ChainId(i)),
                sink: move |draw: Draw<'_, TransitionStats>| {
                    assert_eq!(draw.schema_version, 1);
                    records.lock().unwrap().push((
                        draw.chain,
                        draw.phase,
                        draw.index,
                        draw.position.to_vec(),
                    ));
                    ControlFlow::Continue(())
                },
            }
        })
        .collect()
}
fn sorted(records: &Records) -> Vec<(ChainId, Phase, usize, Vec<f64>)> {
    let mut out = records.lock().unwrap().clone();
    out.sort_by_key(|(id, phase, i, _)| (*id, usize::from(*phase == Phase::Sampling), *i));
    out
}

#[test]
fn sequential_parallel_streams_replay_with_stable_chain_identity() {
    let baseline = Arc::new(Mutex::new(Vec::new()));
    let mut a = chains(&baseline);
    let options = options(127);
    let report = ensemble::run(
        &mut a,
        &options,
        Execution::Sequential,
        &CancellationToken::new(),
        None,
    )
    .unwrap();
    assert_eq!(report.termination, Termination::Completed);
    for threads in [1, 2, 4] {
        let output = Arc::new(Mutex::new(Vec::new()));
        let mut b = chains(&output);
        // Input order is arbitrary; RNG identity is not array position.
        b.reverse();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = ensemble::run(
            &mut b,
            &options,
            Execution::Parallel(&pool),
            &CancellationToken::new(),
            None,
        )
        .unwrap();
        assert_eq!(
            result.chains.iter().map(|x| x.chain).collect::<Vec<_>>(),
            vec![ChainId(3), ChainId(2), ChainId(1), ChainId(0)]
        );
        assert_eq!(sorted(&baseline), sorted(&output));
        for chain in &b {
            assert_eq!(chain.rng, a.iter().find(|c| c.id == chain.id).unwrap().rng);
        }
    }
    assert_ne!(a[0].sampler.q, a[1].sampler.q);
}

#[test]
fn invalid_configuration_and_pre_cancelled_run_do_not_advance_rng() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut chains = chains(&output);
    let before = chain_rng([19; 32], ChainId(0));
    let mut bad = options(20);
    bad.coordinates = vec![0, 0];
    assert!(matches!(
        ensemble::run(
            &mut chains,
            &bad,
            Execution::Sequential,
            &CancellationToken::new(),
            None
        ),
        Err(RunError::Configuration)
    ));
    assert_eq!(chains[0].rng, before);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let result = ensemble::run(
        &mut chains,
        &options(20),
        Execution::Sequential,
        &cancel,
        None,
    )
    .unwrap();
    assert_eq!(result.termination, Termination::Cancelled);
    assert!(output.lock().unwrap().is_empty());
    assert_eq!(chains[0].rng, before);
    chains[1].id = chains[0].id;
    assert!(matches!(
        ensemble::run(
            &mut chains,
            &options(20),
            Execution::Sequential,
            &CancellationToken::new(),
            None
        ),
        Err(RunError::Configuration)
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn subscription_is_latest_only_final_snapshot_is_drained_and_warmup_excluded() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut chains = chains(&output);
    let cancel = CancellationToken::new();
    let monitor = Monitor::start(None, cancel.clone()).unwrap();
    let publisher = monitor.publisher();
    let mut subscription = monitor.subscribe();
    let slow = subscription.clone();
    // A deliberately unread subscriber cannot block the sampler or shutdown.
    ensemble::run(
        &mut chains,
        &options(257),
        Execution::Sequential,
        &cancel,
        Some(&publisher),
    )
    .unwrap();
    monitor.finish().unwrap();
    let report = subscription.changed().await.unwrap();
    assert!(report.finished);
    assert!(!report.failed);
    for c in &report.chains {
        assert_eq!(c.range, 193..257);
        assert_eq!(c.counts.draws, 257);
        assert!(matches!(c.ebfmi, Some(Ok(_))));
    }
    assert!(report.diagnostics.iter().all(Result::is_ok));
    assert!(
        report
            .progress
            .iter()
            .all(|c| c.warmup == 7 && c.draws == 257)
    );
    assert!(Arc::ptr_eq(&report, &slow.latest().unwrap()));
    assert!(subscription.changed().await.is_none());
    let records = sorted(&output);
    for (j, diagnostic) in report.diagnostics.iter().enumerate() {
        let values: Vec<Vec<_>> = (0..4)
            .map(|c| {
                records
                    .iter()
                    .filter(|(id, phase, i, _)| id.0 == c && *phase == Phase::Sampling && *i >= 193)
                    .map(|(_, _, _, q)| q[j])
                    .collect()
            })
            .collect();
        let expected = alea_runtime::diagnostics::modern::summarize(
            &values.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        );
        assert_eq!(*diagnostic, expected);
    }
}

#[test]
fn typed_failure_and_ragged_final_report_never_pass_stop_gates() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut chains = chains(&output);
    chains[1].sampler.fail = 25;
    let cancel = CancellationToken::new();
    let monitor = Monitor::start(None, cancel.clone()).unwrap();
    let subscription = monitor.subscribe();
    let result = ensemble::run(
        &mut chains,
        &options(100),
        Execution::Sequential,
        &cancel,
        Some(&monitor.publisher()),
    );
    assert!(matches!(
        result,
        Err(RunError::Chain {
            chain: ChainId(1),
            ..
        })
    ));
    monitor.finish().unwrap();
    let report = subscription.latest().unwrap();
    assert!(report.finished && report.failed);
    assert!(report.diagnostics.iter().all(Result::is_err));
    let policy = StopPolicy {
        minimum_draws: 4,
        maximum_rhat: 10.0,
        minimum_ess: 1.0,
        maximum_mcse: None,
        minimum_ebfmi: None,
    };
    assert!(!policy.satisfied(&report));
    assert_eq!(chains[1].sampler.steps, 25);
}

#[test]
fn panics_join_parallel_work_and_leave_completed_transitions_intact() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut chains = chains(&output);
    chains[1].sampler.panic = 3;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ensemble::run(
                &mut chains,
                &options(100),
                Execution::Parallel(&pool),
                &CancellationToken::new(),
                None,
            )
        }))
        .is_err()
    );
    assert_eq!(chains[1].sampler.steps, 3);
    // All borrowed Rayon jobs have joined before control returns.
    let completed = output.lock().unwrap().len();
    assert_eq!(
        completed,
        chains.iter().map(|c| c.sampler.steps).sum::<usize>()
    );
}

#[test]
#[cfg(not(miri))]
fn streaming_peak_live_allocation_is_independent_of_draw_budget() {
    let measure = |draws| {
        let mut chains: Vec<_> = (0..4)
            .map(|i| Chain {
                id: ChainId(i),
                sampler: TestChain::new(),
                rng: chain_rng([7; 32], ChainId(i)),
                sink: |_: Draw<'_, TransitionStats>| ControlFlow::Continue(()),
            })
            .collect();
        let options = options(draws);
        let cancel = CancellationToken::new();
        allocation_counter::measure(|| {
            let result =
                ensemble::run(&mut chains, &options, Execution::Sequential, &cancel, None).unwrap();
            assert_eq!(result.termination, Termination::Completed);
        })
    };
    let small = measure(128);
    let large = measure(16_384);
    assert_eq!(small.bytes_max, large.bytes_max);
    assert_eq!(small.bytes_current, 0);
    assert_eq!(large.bytes_current, 0);
}

#[test]
fn minimum_draw_policy_and_explicit_cancellation_are_enforced() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut chains = chains(&output);
    let cancel = CancellationToken::new();
    let policy = StopPolicy {
        minimum_draws: 64,
        maximum_rhat: 2.0,
        minimum_ess: 1.0,
        maximum_mcse: None,
        minimum_ebfmi: Some(0.01),
    };
    let monitor = Monitor::start(Some(policy), cancel.clone()).unwrap();
    let sub = monitor.subscribe();
    ensemble::run(
        &mut chains,
        &options(32),
        Execution::Sequential,
        &cancel,
        Some(&monitor.publisher()),
    )
    .unwrap();
    monitor.finish().unwrap();
    assert!(!cancel.is_cancelled());
    assert!(!sub.latest().unwrap().stop_requested);
    let monitor = Monitor::start(Some(policy), cancel.clone()).unwrap();
    let sub = monitor.subscribe();
    ensemble::run(
        &mut chains,
        &options(64),
        Execution::Sequential,
        &cancel,
        Some(&monitor.publisher()),
    )
    .unwrap();
    monitor.finish().unwrap();
    assert!(cancel.is_cancelled());
    assert!(sub.latest().unwrap().stop_requested);
    let good = sub.latest().unwrap();
    for rejected in 0..6 {
        let mut report = (*good).clone();
        match rejected {
            0 => report.chains[0].counts.divergences = 1,
            1 => report.chains[0].counts.depth_saturations = 1,
            2 => report.chains[0].ebfmi = None,
            3 => report.diagnostics[0] = Err(alea_runtime::diagnostics::modern::Error::Undefined),
            4 => report.chains[0].range.start += 1,
            _ => report.failed = true,
        }
        assert!(!policy.satisfied(&report));
    }
}

#[test]
fn actual_nuts_draws_statistics_and_rng_match_across_execution_modes() {
    use alea_distributions::Gaussian;
    use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
    use alea_mcmc::{Nuts, NutsOptions, NutsTransition};
    let target = Gaussian::new(4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let mut baseline = Vec::new();
    for execution in [Execution::Sequential, Execution::Parallel(&pool)] {
        let records = Arc::new(Mutex::new(Vec::new()));
        let mut chains: Vec<_> = (0..4)
            .map(|i| {
                let records = Arc::clone(&records);
                Chain {
                    id: ChainId(i),
                    rng: chain_rng([71; 32], ChainId(i)),
                    sampler: Nuts::new(
                        &target,
                        OwnedBuffer::new(4),
                        IdentityMetric::new(4),
                        NutsOptions::new(0.2, 6).unwrap(),
                    )
                    .unwrap(),
                    sink: move |draw: Draw<'_, NutsTransition>| {
                        records.lock().unwrap().push((
                            draw.chain,
                            draw.phase,
                            draw.index,
                            draw.position.to_vec(),
                            draw.stats,
                        ));
                        ControlFlow::Continue(())
                    },
                }
            })
            .collect();
        ensemble::run(
            &mut chains,
            &options(64),
            execution,
            &CancellationToken::new(),
            None,
        )
        .unwrap();
        let mut output = records.lock().unwrap().clone();
        output
            .sort_by_key(|(id, phase, i, _, _)| (*id, usize::from(*phase == Phase::Sampling), *i));
        if baseline.is_empty() {
            baseline = output;
        } else {
            assert_eq!(baseline, output);
        }
    }
}

#[test]
fn observer_stop_and_mid_chunk_cancellation_emit_completed_step_only() {
    for observer_stop in [true, false] {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let mut chain = [Chain {
            id: ChainId(0),
            sampler: TestChain::new(),
            rng: chain_rng([3; 32], ChainId(0)),
            sink: move |draw: Draw<'_, TransitionStats>| {
                if draw.phase == Phase::Sampling && draw.index == 2 {
                    if observer_stop {
                        return ControlFlow::Break(());
                    }
                    token.cancel();
                }
                ControlFlow::Continue(())
            },
        }];
        let result = ensemble::run(
            &mut chain,
            &options(100),
            Execution::Sequential,
            &cancel,
            None,
        )
        .unwrap();
        assert_eq!(
            result.termination,
            if observer_stop {
                Termination::Observer
            } else {
                Termination::Cancelled
            }
        );
        assert_eq!(result.chains[0].draws, 3);
        assert_eq!(chain[0].sampler.steps, 10);
    }
}
