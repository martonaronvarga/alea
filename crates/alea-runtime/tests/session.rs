use alea_core::{density::LogDensity, target::LogDensityGradient};
use alea_mcmc::{
    HmcOptions, NutsOptions, RwmhOptions, adapt::MetricKind, config::AcceptanceTarget,
};
use alea_runtime::{
    ensemble::Execution,
    output::ChainId,
    session::{
        Checkpoint, CheckpointError, Diagnostics, Failure, FrozenGeometry, Kernel, Resources,
        SessionBuilder, Status, Warmup,
        storage::{ChannelSink, Discard, DrawReader, DurableSink, Observer, Sink, SinkError},
    },
};
use std::{
    cell::Cell,
    convert::Infallible,
    fs::File,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;

struct Normal {
    dimension: usize,
}
impl LogDensityGradient for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        for (g, q) in g.iter_mut().zip(q) {
            *g = -*q;
        }
        Ok(-0.5 * q.iter().map(|x| x * x).sum::<f64>())
    }
}
impl LogDensity for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp(&self, q: &[f64]) -> Result<f64, Infallible> {
        Ok(-0.5 * q.iter().map(|x| x * x).sum::<f64>())
    }
}
fn builder(kernel: Kernel, chains: usize, draws: usize) -> SessionBuilder {
    SessionBuilder::new(
        "normal-v1;test-build",
        3,
        (0..chains).map(|i| vec![i as f64 * 0.1; 3]).collect(),
    )
    .seed([151; 32])
    .kernel(kernel)
    .draws(draws)
    .diagnostics(Diagnostics {
        window: 64,
        maximum_rhat: 10.0,
        minimum_ess: 1.0,
    })
}
fn covariance(metric: MetricKind) -> Warmup {
    Warmup::Covariance {
        iterations: 80,
        metric,
        target: AcceptanceTarget::try_from(0.8).unwrap(),
        search: None,
    }
}
fn hmc() -> Kernel {
    Kernel::Hmc(HmcOptions::new(0.25, 4).unwrap())
}
fn nuts() -> Kernel {
    Kernel::Nuts(NutsOptions::new(0.25, 5).unwrap())
}
fn rwmh() -> Kernel {
    Kernel::Rwmh(RwmhOptions::new(0.5).unwrap())
}

#[test]
fn nuts_session_native_warmup_reaches_a_narrow_posterior_from_distant_starts() {
    struct Narrow;
    impl LogDensityGradient for Narrow {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            1
        }
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
            let z = (q[0] - 1.) / 0.001;
            g[0] = -z / 0.001;
            Ok(-0.5 * z * z)
        }
    }
    let session = SessionBuilder::new("narrow-native-warmup-v1", 1, vec![vec![0.]; 4])
        .kernel(Kernel::Nuts(NutsOptions::new(0.1, 8).unwrap()))
        .warmup(Warmup::Covariance {
            iterations: 512,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into().unwrap(),
            search: Some(Default::default()),
        })
        .draws(1024)
        .build()
        .unwrap();
    let out = session
        .run(|_| Ok(Narrow), Execution::Sequential, &Discard)
        .unwrap();
    for chain in out.chains {
        assert!(chain.failure.is_none(), "{:?}", chain.failure);
        assert_eq!(chain.warmup_completed, 512);
        assert_eq!(chain.completed, 1024);
        let mean = chain.draws.iter().map(|d| d.position[0]).sum::<f64>() / 1024.;
        assert!((mean - 1.).abs() < 0.00015, "mean {mean}");
    }
}

#[test]
fn serial_rayon_and_streamed_collected_outputs_match_for_all_reference_kernels() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for kernel in [hmc(), nuts(), rwmh()] {
        let warmup = match kernel {
            Kernel::Rwmh(_) => Warmup::RwmhScale {
                iterations: 80,
                target: 0.234.try_into().unwrap(),
            },
            _ => covariance(MetricKind::Diagonal),
        };
        let session = builder(kernel, 4, 256).warmup(warmup).build().unwrap();
        let serial = match kernel {
            Kernel::Rwmh(_) => session.run_density(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &Discard,
            ),
            _ => session.run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &Discard,
            ),
        }
        .unwrap();
        let records = Mutex::new(Vec::new());
        let observer = Observer(|d: &alea_runtime::session::OwnedDraw| {
            records.lock().unwrap().push(d.clone());
            Ok(())
        });
        let streamed = builder(kernel, 4, 256)
            .warmup(warmup)
            .collect(false)
            .build()
            .unwrap();
        let parallel = match kernel {
            Kernel::Rwmh(_) => streamed.run_density(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Parallel(&pool),
                &observer,
            ),
            _ => streamed.run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Parallel(&pool),
                &observer,
            ),
        }
        .unwrap();
        assert_eq!(serial.status, Status::Success);
        assert_eq!(parallel.status, serial.status);
        let records = records.into_inner().unwrap();
        for (a, b) in serial.chains.iter().zip(&parallel.chains) {
            let received: Vec<_> = records
                .iter()
                .filter(|d| d.chain == a.chain)
                .cloned()
                .collect();
            assert_eq!(a.draws, received);
            assert!(b.draws.is_empty());
            assert_eq!(a.checkpoint, b.checkpoint);
            assert_eq!(a.warmup_completed, 80);
            assert_eq!(b.completed, 256);
        }
    }
}

#[test]
fn cancelled_durable_checkpoints_replay_odd_dimensional_normal_cache_and_dense_mass() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let dir = Directory::new();
    for (case, (kernel, kind)) in [
        (hmc(), MetricKind::Dense),
        (nuts(), MetricKind::Dense),
        (hmc(), MetricKind::Diagonal),
        (nuts(), MetricKind::Diagonal),
        (rwmh(), MetricKind::Diagonal),
    ]
    .into_iter()
    .enumerate()
    {
        let warmup = if matches!(kernel, Kernel::Rwmh(_)) {
            Warmup::RwmhScale {
                iterations: 20,
                target: 0.234.try_into().unwrap(),
            }
        } else {
            covariance(kind)
        };
        let full_session = builder(kernel, 2, 100).warmup(warmup).build().unwrap();
        let full = if matches!(kernel, Kernel::Rwmh(_)) {
            full_session.run_density(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &Discard,
            )
        } else {
            full_session.run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &Discard,
            )
        }
        .unwrap();
        let cancel = CancellationToken::new();
        let observer = Observer(|d: &alea_runtime::session::OwnedDraw| {
            if d.chain == ChainId(0) && d.index == 26 {
                cancel.cancel();
            }
            Ok(())
        });
        let session = builder(kernel, 2, 100)
            .warmup(warmup)
            .cancellation(cancel.clone())
            .build()
            .unwrap();
        let partial = if matches!(kernel, Kernel::Rwmh(_)) {
            session.run_density(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &observer,
            )
        } else {
            session.run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &observer,
            )
        }
        .unwrap();
        assert_eq!(partial.status, Status::Cancelled);
        // Replay chain zero with its original logical ID; sequential cancellation
        // prevented chain one from starting.
        let cp = partial.chains[0].checkpoint.as_ref().unwrap();
        if !matches!(kernel, Kernel::Rwmh(_)) {
            if kind == MetricKind::Dense {
                assert!(matches!(cp.geometry(), FrozenGeometry::InverseDense { .. }));
            } else {
                assert!(matches!(cp.geometry(), FrozenGeometry::InverseDiagonal(_)));
            }
        }
        let path = dir.0.join(format!("{case}.checkpoint"));
        cp.write_new(&path, 65_536).unwrap();
        assert!(cp.write_new(&path, 65_536).is_err());
        assert!(Checkpoint::read(&path, 8).is_err());
        let decoded = Checkpoint::read(&path, 65_536).unwrap();
        assert_eq!(&decoded, cp);
        let resumed = builder(kernel, 1, 100)
            .restart(vec![decoded])
            .build()
            .unwrap();
        let rest = if matches!(kernel, Kernel::Rwmh(_)) {
            resumed.run_density(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Parallel(&pool),
                &Discard,
            )
        } else {
            resumed.run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Parallel(&pool),
                &Discard,
            )
        }
        .unwrap();
        let cut = partial.chains[0].completed;
        assert_eq!(partial.chains[0].draws, full.chains[0].draws[..cut]);
        assert_eq!(rest.chains[0].draws, full.chains[0].draws[cut..]);
        assert_eq!(rest.chains[0].checkpoint, full.chains[0].checkpoint);
        assert_eq!(rest.chains[0].warmup_completed, 0);
        assert_eq!(rest.diagnostics.chains[0].range, 36..100);
        let mut corrupt = std::fs::read(&path).unwrap();
        corrupt[12] ^= 1;
        std::fs::write(dir.0.join("corrupt"), &corrupt).unwrap();
        assert!(Checkpoint::read(dir.0.join("corrupt"), 65_536).is_err());
        for length in [0, 8, 16, corrupt.len() - 1] {
            std::fs::write(dir.0.join("truncated"), &corrupt[..length]).unwrap();
            assert!(Checkpoint::read(dir.0.join("truncated"), 65_536).is_err());
        }
    }
}

#[test]
fn nuts_options_survive_durable_restart_and_reject_changed_policy() {
    let directory = Directory::new();
    for (case, warmup) in [
        Warmup::None,
        covariance(MetricKind::Diagonal),
        Warmup::GradientDiagonal {
            iterations: 80,
            target: 0.8.try_into().unwrap(),
            search: None,
            final_restart: false,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let options = NutsOptions::new(0.2, 5)
            .unwrap()
            .with_step_size_jitter(0.1)
            .unwrap()
            .with_position_turning(true);
        let kernel = Kernel::Nuts(options);
        let run = |draws| {
            builder(kernel, 1, draws)
                .warmup(warmup)
                .build()
                .unwrap()
                .run(
                    |_| Ok(Normal { dimension: 3 }),
                    Execution::Sequential,
                    &Discard,
                )
                .unwrap()
        };
        let full = run(80);
        let prefix = run(31);
        let checkpoint = prefix.chains[0].checkpoint.as_ref().unwrap();
        let Kernel::Nuts(frozen) = checkpoint.kernel() else {
            panic!("NUTS checkpoint expected")
        };
        assert_eq!(frozen.step_size_jitter(), 0.1);
        assert!(frozen.position_turning());
        let path = directory.0.join(format!("nuts-options-{case}"));
        checkpoint.write_new(&path, 65_536).unwrap();
        let restored = Checkpoint::read(&path, 65_536).unwrap();
        assert_eq!(&restored, checkpoint, "durable settings must be lossless");
        let resumed = builder(kernel, 1, 80)
            .restart(vec![restored.clone()])
            .build()
            .unwrap()
            .run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Sequential,
                &Discard,
            )
            .unwrap();
        assert_eq!(resumed.chains[0].draws, full.chains[0].draws[31..]);
        assert_eq!(resumed.chains[0].checkpoint, full.chains[0].checkpoint);
        for changed in [
            options.with_position_turning(false),
            options.with_step_size_jitter(0.0).unwrap(),
        ] {
            assert!(
                builder(Kernel::Nuts(changed), 1, 80)
                    .restart(vec![restored.clone()])
                    .build()
                    .is_err()
            );
        }
        let mut legacy = std::fs::read(&path).unwrap();
        legacy[..8].copy_from_slice(b"ALEACP01");
        let old = directory.0.join(format!("old-nuts-options-{case}"));
        std::fs::write(&old, legacy).unwrap();
        assert!(matches!(
            Checkpoint::read(old, 65_536),
            Err(CheckpointError::Format)
        ));
    }
}

#[test]
fn gradient_diagonal_warmup_is_nuts_only_and_budgeted() {
    let warmup = Warmup::GradientDiagonal {
        iterations: 80,
        target: 0.8.try_into().unwrap(),
        search: None,
        final_restart: false,
    };
    for kernel in [hmc(), rwmh()] {
        assert!(builder(kernel, 1, 16).warmup(warmup).build().is_err());
    }
    let limits = Resources {
        warmup: 79,
        ..Resources::default()
    };
    assert!(
        builder(nuts(), 1, 16)
            .warmup(warmup)
            .resources(limits)
            .build()
            .is_err()
    );
}

#[test]
fn native_model_factory_needs_neither_send_nor_sync_target() {
    use alea_core::model::assembly::ModelBuilder;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let session = SessionBuilder::new("compiled-normal", 1, vec![vec![0.0]; 2])
        .draws(32)
        .build()
        .unwrap();
    let result = session
        .run(
            |_| {
                let mut b = ModelBuilder::new();
                let q = b.real("q").unwrap();
                let square = b.multiply(q.expr(), q.expr()).unwrap();
                let minus = b.constant(-0.5).unwrap();
                let logp = b.multiply(square, minus).unwrap();
                b.factor(logp).unwrap();
                Ok::<_, alea_core::model::ModelError<alea_core::model::assembly::AssemblyError>>(
                    b.compile().unwrap().into_target().unwrap(),
                )
            },
            Execution::Parallel(&pool),
            &Discard,
        )
        .unwrap();
    assert!(
        result
            .chains
            .iter()
            .all(|c| c.completed == 32 && c.failure.is_none())
    );
}

#[derive(Debug, thiserror::Error)]
#[error("original backend fault")]
struct Fault;
struct Faulty {
    calls: Cell<usize>,
    fail: usize,
    panic: bool,
    cancel: Option<CancellationToken>,
}
impl LogDensityGradient for Faulty {
    type Error = Fault;
    fn dimension(&self) -> usize {
        3
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
        let count = self.calls.get();
        self.calls.set(count + 1);
        if count == self.fail {
            if self.panic {
                panic!("worker fault");
            }
            return Err(Fault);
        }
        if count == 4
            && let Some(token) = &self.cancel
        {
            token.cancel();
        }
        for (g, q) in g.iter_mut().zip(q) {
            *g = -*q;
        }
        Ok(-0.5 * q.iter().map(|x| x * x).sum::<f64>())
    }
}
#[test]
fn failures_preserve_completed_draws_original_errors_and_join_all_workers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for panic in [false, true] {
        let session = builder(hmc(), 3, 30).build().unwrap();
        let out = session
            .run(
                |id| {
                    Ok(Faulty {
                        calls: Cell::new(0),
                        fail: if id == ChainId(0) { 10 } else { usize::MAX },
                        panic,
                        cancel: None,
                    })
                },
                Execution::Parallel(&pool),
                &Discard,
            )
            .unwrap();
        assert_eq!(out.status, Status::BackendFailure);
        assert_eq!(out.chains[0].completed, 2);
        assert_eq!(out.chains[0].draws.len(), 2);
        assert_eq!(
            out.chains[0].checkpoint.as_ref().unwrap().position(),
            out.chains[0].draws[1].position
        );
        if panic {
            assert!(matches!(out.chains[0].failure, Some(Failure::WorkerPanic)));
        } else {
            assert!(matches!(
                out.chains[0].failure,
                Some(Failure::Hmc(alea_mcmc::hmc::HmcError::Evaluation {
                    source: alea_core::target::EvaluationError::Model(Fault),
                    ..
                }))
            ));
        }
        assert_eq!(out.chains[1].completed, 30);
        assert_eq!(out.chains[2].completed, 30);
    }
}

#[test]
fn cancellation_during_warmup_has_no_partial_geometry_or_checkpoint() {
    let token = CancellationToken::new();
    let session = builder(hmc(), 1, 30)
        .warmup(covariance(MetricKind::Dense))
        .cancellation(token.clone())
        .build()
        .unwrap();
    let out = session
        .run(
            |_| {
                Ok(Faulty {
                    calls: Cell::new(0),
                    fail: usize::MAX,
                    panic: false,
                    cancel: Some(token.clone()),
                })
            },
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    assert_eq!(out.status, Status::Cancelled);
    assert_eq!(out.chains[0].warmup_completed, 1);
    assert!(out.chains[0].checkpoint.is_none());
    assert!(out.chains[0].draws.is_empty());
}

#[test]
fn pre_cancel_and_zero_deadline_do_not_create_targets() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    for session in [
        builder(hmc(), 2, 10).cancellation(cancel).build().unwrap(),
        builder(hmc(), 2, 10)
            .wall_time(Duration::ZERO)
            .build()
            .unwrap(),
    ] {
        let out = session
            .run(
                |_| -> Result<Normal, Infallible> {
                    panic!("must not construct");
                },
                Execution::Sequential,
                &Discard,
            )
            .unwrap();
        assert_eq!(out.status, Status::Cancelled);
        assert!(
            out.chains
                .iter()
                .all(|c| c.completed == 0 && c.checkpoint.is_none())
        );
    }
    // Cancellation after the final completed draw cannot uncomplete the run.
    let token = CancellationToken::new();
    let session = builder(hmc(), 1, 4)
        .cancellation(token.clone())
        .build()
        .unwrap();
    let out = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Observer(|d: &alea_runtime::session::OwnedDraw| {
                if d.index == 3 {
                    token.cancel();
                }
                Ok(())
            }),
        )
        .unwrap();
    assert_eq!(out.status, Status::DiagnosticWarning);
    assert_eq!(out.chains[0].completed, 4);
}

#[test]
fn bounded_channel_backpressure_subscriber_loss_and_observer_panics_return_pending_draw() {
    let session = builder(hmc(), 1, 10).collect(false).build().unwrap();
    let (sink, receiver) = ChannelSink::bounded(1).unwrap();
    let out = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &sink,
        )
        .unwrap();
    assert_eq!(out.status, Status::BackendFailure);
    assert_eq!(out.chains[0].completed, 2);
    assert_eq!(receiver.try_iter().count(), 1);
    assert_eq!(out.chains[0].undelivered.as_ref().unwrap().index, 1);
    assert!(matches!(
        out.chains[0].failure,
        Some(Failure::Sink(SinkError::Backpressure))
    ));
    drop(receiver);
    let out = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &sink,
        )
        .unwrap();
    assert_eq!(out.chains[0].completed, 1);
    assert!(matches!(
        out.chains[0].failure,
        Some(Failure::Sink(SinkError::Disconnected))
    ));
    let out = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Observer(
                |_: &alea_runtime::session::OwnedDraw| -> Result<(), SinkError> {
                    panic!("observer fault");
                },
            ),
        )
        .unwrap();
    assert!(matches!(
        out.chains[0].failure,
        Some(Failure::ObserverPanic)
    ));
    assert_eq!(out.chains[0].undelivered.as_ref().unwrap().index, 0);
    assert_eq!(out.chains[0].checkpoint.as_ref().unwrap().completed(), 1);
}

#[test]
fn numerical_exhaustion_retains_rejected_states_without_invented_hamiltonian_stats() {
    let session = builder(Kernel::Rwmh(RwmhOptions::new(f64::MAX).unwrap()), 1, 20)
        .numerical_limit(3)
        .build()
        .unwrap();
    let out = session
        .run_density(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    assert_eq!(out.status, Status::NumericalExhaustion);
    assert_eq!(out.chains[0].completed, 3);
    assert_eq!(out.chains[0].numerical_rejections, 3);
    assert!(
        out.chains[0]
            .draws
            .iter()
            .all(|d| d.position == vec![0.0; 3]
                && d.stats.divergent.is_none()
                && d.numerical_rejection)
    );
    let resumed = builder(rwmh(), 1, 20)
        .numerical_limit(3)
        .restart(vec![out.chains[0].checkpoint.clone().unwrap()])
        .build()
        .unwrap();
    let rest = resumed
        .run_density(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    assert_eq!(rest.chains[0].completed, 3);
    assert_eq!(rest.status, Status::NumericalExhaustion);
}

#[test]
fn durable_prefix_roundtrip_storage_budget_and_incompatible_provenance() {
    let dir = Directory::new();
    let session = builder(nuts(), 2, 30).build().unwrap();
    let path = dir.0.join("draws");
    let sink = DurableSink::create_new(&path, session.provenance(), 1_000_000).unwrap();
    assert!(DurableSink::create_new(&path, session.provenance(), 1_000_000).is_err());
    let out = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &sink,
        )
        .unwrap();
    let mut reader = DrawReader::new(File::open(&path).unwrap(), 65_536).unwrap();
    assert_eq!(reader.provenance(), &out.provenance);
    let mut records = Vec::new();
    while let Some(draw) = reader.next_draw().unwrap() {
        records.push(draw);
    }
    assert_eq!(
        records,
        out.chains
            .into_iter()
            .flat_map(|c| c.draws)
            .collect::<Vec<_>>()
    );
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.pop();
    let mut reader = DrawReader::new(bytes.as_slice(), 65_536).unwrap();
    let mut count = 0;
    loop {
        match reader.next_draw() {
            Ok(Some(_)) => count += 1,
            Err(_) => break,
            Ok(None) => panic!("truncation was hidden"),
        }
    }
    assert_eq!(count, 59);
    assert!(DrawReader::new(bytes.as_slice(), 8).is_err());
    let budget_sink =
        DurableSink::create_new(dir.0.join("budget"), session.provenance(), 300).unwrap();
    let budget = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &budget_sink,
        )
        .unwrap();
    assert_eq!(budget.status, Status::BackendFailure);
    assert!(budget.chains.iter().any(|c| c.undelivered.is_some()));
    let wrong = SessionBuilder::new("wrong-target", 3, vec![vec![0.0; 3]])
        .build()
        .unwrap();
    let fresh_sink =
        DurableSink::create_new(dir.0.join("identity"), session.provenance(), 1_000_000).unwrap();
    let out = wrong
        .run(
            |_| -> Result<Normal, Infallible> {
                panic!("sink identity rejects first");
            },
            Execution::Sequential,
            &fresh_sink,
        )
        .unwrap();
    assert_eq!(out.status, Status::BackendFailure);
    assert_eq!(out.chains[0].completed, 0);
    assert!(matches!(out.sink_failure, Some(SinkError::Format)));
}

#[test]
fn restart_refuses_changed_target_cache_ids_kernel_shape_or_seed() {
    let session = builder(hmc(), 1, 10).build().unwrap();
    let out = session
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    let cp = out.chains[0].checkpoint.clone().unwrap();
    assert!(
        builder(hmc(), 1, 20)
            .restart(vec![cp.clone()])
            .warmup(covariance(MetricKind::Diagonal))
            .build()
            .is_err()
    );
    assert!(
        builder(nuts(), 1, 20)
            .restart(vec![cp.clone()])
            .build()
            .is_err()
    );
    assert!(
        builder(hmc(), 1, 20)
            .seed([2; 32])
            .restart(vec![cp.clone()])
            .build()
            .is_err()
    );
    struct Changed;
    impl LogDensityGradient for Changed {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            3
        }
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
            Normal { dimension: 3 }.logp_grad(q, g).map(|v| v + 1.0)
        }
    }
    let resumed = builder(hmc(), 1, 20).restart(vec![cp]).build().unwrap();
    let rest = resumed
        .run(|_| Ok(Changed), Execution::Sequential, &Discard)
        .unwrap();
    assert_eq!(rest.status, Status::BackendFailure);
    assert!(matches!(
        rest.chains[0].failure,
        Some(Failure::RestartMismatch)
    ));
    assert!(rest.chains[0].draws.is_empty());
}

#[test]
fn all_resource_products_and_pool_limits_are_admitted_before_work() {
    assert!(builder(hmc(), 1, usize::MAX).build().is_err());
    assert!(
        builder(hmc(), 1, 10)
            .resources(Resources {
                working_values: 1,
                ..Resources::default()
            })
            .build()
            .is_err()
    );
    assert!(
        builder(nuts(), 1, 10)
            .resources(Resources {
                tree_depth: 4,
                ..Resources::default()
            })
            .build()
            .is_err()
    );
    assert!(
        builder(hmc(), 1, 10)
            .diagnostics(Diagnostics {
                window: usize::MAX,
                ..Diagnostics::default()
            })
            .build()
            .is_err()
    );
    assert!(
        builder(rwmh(), 1, 10)
            .warmup(covariance(MetricKind::Dense))
            .build()
            .is_err()
    );
    let session = builder(hmc(), 1, 10)
        .resources(Resources {
            threads: 1,
            ..Resources::default()
        })
        .build()
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    assert!(
        session
            .run(
                |_| -> Result<Normal, Infallible> {
                    panic!("pool rejected first");
                },
                Execution::Parallel(&pool),
                &Discard
            )
            .is_err()
    );
}

#[test]
fn streaming_peak_memory_does_not_grow_with_draw_budget_and_finish_runs_after_join() {
    // Some platforms initialize a retained standard-library thread-local block
    // on the first session. Exclude that one-time runtime setup, then retain the
    // strict zero-live-bytes and equal-peak checks for both complete steady runs.
    builder(rwmh(), 1, 64)
        .collect(false)
        .build()
        .unwrap()
        .run_density(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    fn peak(draws: usize) -> u64 {
        let session = builder(rwmh(), 1, draws).collect(false).build().unwrap();
        let info = allocation_counter::measure(|| {
            let out = session
                .run_density(
                    |_| Ok(Normal { dimension: 3 }),
                    Execution::Sequential,
                    &Discard,
                )
                .unwrap();
            assert!(out.chains[0].draws.is_empty());
            assert_eq!(out.chains[0].completed, draws);
        });
        assert_eq!(info.bytes_current, 0);
        info.bytes_max
    }
    // Both runs fill exactly the same 64-draw window. No output-history allocation.
    assert_eq!(peak(1000), peak(100_000));
    struct Joined {
        count: AtomicUsize,
        finished: AtomicUsize,
    }
    impl Sink for Joined {
        fn draw(&self, _: &alea_runtime::session::OwnedDraw) -> Result<(), SinkError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn finish(&self) -> Result<(), SinkError> {
            assert_eq!(self.count.load(Ordering::SeqCst), 400);
            self.finished.fetch_add(1, Ordering::SeqCst);
            Err(SinkError::Disconnected)
        }
    }
    let sink = Joined {
        count: AtomicUsize::new(0),
        finished: AtomicUsize::new(0),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let out = builder(hmc(), 4, 100)
        .build()
        .unwrap()
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Parallel(&pool),
            &sink,
        )
        .unwrap();
    assert_eq!(sink.finished.load(Ordering::SeqCst), 1);
    assert_eq!(out.status, Status::BackendFailure);
    assert!(out.chains.iter().all(|c| c.completed == 100));
}

struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "alea-session-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn concurrent_cancellation_restarts_each_logical_stream_at_its_own_completed_index() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let baseline = builder(hmc(), 2, 120)
        .build()
        .unwrap()
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    for _ in 0..8 {
        let cancel = CancellationToken::new();
        let barrier = std::sync::Barrier::new(2);
        let session = builder(hmc(), 2, 120)
            .cancellation(cancel.clone())
            .build()
            .unwrap();
        let partial = session
            .run(
                |_| {
                    barrier.wait();
                    Ok(Normal { dimension: 3 })
                },
                Execution::Parallel(&pool),
                &Observer(|d: &alea_runtime::session::OwnedDraw| {
                    if d.chain == ChainId(0) && d.index == 20 {
                        cancel.cancel();
                    }
                    Ok(())
                }),
            )
            .unwrap();
        assert_eq!(partial.status, Status::Cancelled);
        let checkpoints = partial
            .chains
            .iter()
            .map(|c| c.checkpoint.clone().unwrap())
            .collect();
        let resumed = builder(hmc(), 2, 120)
            .restart(checkpoints)
            .build()
            .unwrap()
            .run(
                |_| Ok(Normal { dimension: 3 }),
                Execution::Parallel(&pool),
                &Discard,
            )
            .unwrap();
        for ((a, b), full) in partial
            .chains
            .iter()
            .zip(&resumed.chains)
            .zip(&baseline.chains)
        {
            assert_eq!(a.draws, full.draws[..a.completed]);
            assert_eq!(b.draws, full.draws[a.completed..]);
            assert_eq!(b.checkpoint, full.checkpoint);
        }
    }
}

#[test]
fn sink_finalization_panic_is_an_outcome_after_every_worker_has_joined() {
    struct Panics;
    impl Sink for Panics {
        fn draw(&self, _: &alea_runtime::session::OwnedDraw) -> Result<(), SinkError> {
            Ok(())
        }
        fn finish(&self) -> Result<(), SinkError> {
            panic!("finish fault");
        }
    }
    let out = builder(hmc(), 2, 10)
        .build()
        .unwrap()
        .run(
            |_| Ok(Normal { dimension: 3 }),
            Execution::Sequential,
            &Panics,
        )
        .unwrap();
    assert_eq!(out.status, Status::BackendFailure);
    assert!(matches!(out.sink_failure, Some(SinkError::ObserverPanic)));
    assert!(
        out.chains
            .iter()
            .all(|c| c.completed == 10 && c.draws.len() == 10)
    );
}

#[test]
fn nonfinite_initial_evaluations_are_numerical_while_factory_errors_remain_backend_failures() {
    struct Nonfinite;
    impl LogDensity for Nonfinite {
        type Error = Fault;
        fn dimension(&self) -> usize {
            3
        }
        fn logp(&self, _: &[f64]) -> Result<f64, Fault> {
            Ok(f64::NAN)
        }
    }
    impl LogDensityGradient for Nonfinite {
        type Error = Fault;
        fn dimension(&self) -> usize {
            3
        }
        fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
            g.fill(0.0);
            Ok(f64::NAN)
        }
    }
    let gradient = builder(hmc(), 1, 10)
        .build()
        .unwrap()
        .run(|_| Ok(Nonfinite), Execution::Sequential, &Discard)
        .unwrap();
    let density = builder(rwmh(), 1, 10)
        .build()
        .unwrap()
        .run_density(|_| Ok(Nonfinite), Execution::Sequential, &Discard)
        .unwrap();
    for out in [gradient, density] {
        assert_eq!(out.status, Status::NumericalExhaustion);
        assert_eq!(out.chains[0].completed, 0);
        assert!(out.chains[0].checkpoint.is_none());
    }
    let factory = builder(hmc(), 1, 10)
        .build()
        .unwrap()
        .run(
            |_| Err::<Nonfinite, _>(Fault),
            Execution::Sequential,
            &Discard,
        )
        .unwrap();
    assert_eq!(factory.status, Status::BackendFailure);
    assert!(matches!(
        factory.chains[0].failure,
        Some(Failure::Target(Fault))
    ));
}
