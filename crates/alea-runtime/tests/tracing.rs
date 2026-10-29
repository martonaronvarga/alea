#![cfg(feature = "tracing")]
use alea_core::{
    density::LogDensity,
    target::{LogDensityGradient, TargetSpace},
};
use alea_mcmc::{HmcOptions, RwmhOptions};
use alea_runtime::{
    ensemble::Execution,
    monitor::Monitor,
    optimize::{
        LbfgsOptions, ObjectiveDescription, ObjectiveKind, OptimizationTermination, maximize,
        maximize_with_cancel,
    },
    session::{
        SamplerConfig, SessionBuilder, Status,
        storage::{Discard, Sink, SinkError},
    },
};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id},
};
use tracing_subscriber::{Layer, Registry, layer::Context, prelude::*, registry::LookupSpan};

#[derive(Clone, Default, Debug)]
struct Fields(BTreeMap<String, String>);
impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), value.into());
    }
}
#[derive(Clone, Debug)]
struct Record {
    target: &'static str,
    fields: Fields,
    scope: Vec<(&'static str, Fields)>,
}
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<Record>>>);
impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        ctx.span(id).unwrap().extensions_mut().insert(fields);
    }
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let scope = ctx
            .event_span(event)
            .map(|s| {
                s.scope()
                    .from_root()
                    .map(|s| (s.name(), s.extensions().get::<Fields>().unwrap().clone()))
                    .collect()
            })
            .unwrap_or_default();
        self.0.lock().unwrap().push(Record {
            target: event.metadata().target(),
            fields,
            scope,
        });
    }
}
fn subscriber(parent: bool) -> (tracing::Dispatch, Capture) {
    let capture = Capture::default();
    let layer = capture
        .clone()
        .with_filter(tracing_subscriber::filter::filter_fn(move |m| {
            parent || m.target().starts_with("alea::")
        }));
    (
        tracing::Dispatch::new(Registry::default().with(layer)),
        capture,
    )
}
struct Gaussian;
impl TargetSpace for Gaussian {
    type Error = std::io::Error;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensity for Gaussian {
    fn logp(&self, q: &[f64]) -> Result<f64, Self::Error> {
        Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
    }
}
impl LogDensityGradient for Gaussian {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        g[0] = -q[0];
        g[1] = -q[1];
        self.logp(q)
    }
}
fn session(density: bool) -> alea_runtime::session::Session {
    SessionBuilder::new("SECRET_MODEL", 2, vec![vec![0.; 2]; 2])
        .sampler(if density {
            SamplerConfig::Rwmh(RwmhOptions::new(0.3).unwrap())
        } else {
            SamplerConfig::Hmc(HmcOptions::new(0.2, 3).unwrap())
        })
        .draws(32)
        .seed([17; 32])
        .collect(true)
        .build()
        .unwrap()
}
fn run(
    s: &alea_runtime::session::Session,
    pool: &rayon::ThreadPool,
    density: bool,
) -> alea_runtime::session::RunOutcome<std::io::Error> {
    if density {
        s.run_density(|_| Ok(Gaussian), Execution::Parallel(pool), &Discard)
            .unwrap()
    } else {
        s.run(|_| Ok(Gaussian), Execution::Parallel(pool), &Discard)
            .unwrap()
    }
}
fn checkpoint_bytes(cp: &alea_runtime::session::Checkpoint) -> Vec<u8> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "alea-trace-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    cp.write_new(&path, 1 << 20).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    bytes
}

#[test]
fn scoped_subscribers_follow_reused_pools_and_preserve_draws_rng_and_checkpoint_bytes() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for density in [false, true] {
        let s = session(density);
        let baseline = run(&s, &pool, density);
        for parent in [false, true] {
            let (dispatch, capture) = subscriber(parent);
            let traced = tracing::dispatcher::with_default(&dispatch, || {
                let _parent = tracing::info_span!("caller").entered();
                run(&s, &pool, density)
            });
            assert_eq!(baseline.status, traced.status);
            for (a, b) in baseline.chains.iter().zip(&traced.chains) {
                assert_eq!(a.draws, b.draws);
                assert_eq!(a.completed, b.completed);
                assert_eq!(
                    checkpoint_bytes(a.checkpoint.as_ref().unwrap()),
                    checkpoint_bytes(b.checkpoint.as_ref().unwrap())
                );
            }
            let records = capture.0.lock().unwrap();
            let chains: Vec<_> = records
                .iter()
                .filter(|r| {
                    r.fields
                        .0
                        .get("event")
                        .is_some_and(|v| v == "chain_finished")
                })
                .collect();
            assert_eq!(chains.len(), 2);
            for record in chains {
                assert_eq!(record.target, "alea::session");
                assert_eq!(record.fields.0["schema_version"], "1");
                let names: Vec<_> = record.scope.iter().map(|s| s.0).collect();
                assert_eq!(
                    names,
                    if parent {
                        vec!["caller", "session", "chain"]
                    } else {
                        vec!["session", "chain"]
                    }
                );
                assert!(record.scope.last().unwrap().1.0.contains_key("chain_id"));
            }
            assert!(!format!("{records:?}").contains("SECRET_MODEL"));
            let count = records.len();
            drop(records);
            pool.install(|| tracing::info!(target:"alea::session",event="must_not_leak"));
            assert_eq!(capture.0.lock().unwrap().len(), count);
        }
    }
}

fn objective() -> ObjectiveDescription {
    ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "SECRET_MEASURE".into(),
        model: "SECRET_MODEL".into(),
        data: "SECRET_DATA".into(),
        source: "SECRET_SOURCE".into(),
    }
}
#[test]
fn optimizer_completion_cancellation_and_failure_events_do_not_disclose_payloads() {
    let (dispatch, capture) = subscriber(true);
    tracing::dispatcher::with_default(&dispatch, || {
        let options = LbfgsOptions::new(50, 200).unwrap();
        let result = maximize(&Gaussian, &[1., 2.], objective(), options.clone()).unwrap();
        assert_eq!(result.termination, OptimizationTermination::ScoreTolerance);
        let result =
            maximize_with_cancel(&Gaussian, &[1., 2.], objective(), options.clone(), || true)
                .unwrap();
        assert_eq!(result.termination, OptimizationTermination::Cancelled);
        assert!(maximize(&Gaussian, &[1.], objective(), options).is_err());
    });
    let records = capture.0.lock().unwrap();
    let finished: Vec<_> = records
        .iter()
        .filter(|r| r.fields.0.get("event").is_some_and(|s| s == "finished"))
        .collect();
    assert_eq!(finished.len(), 2);
    assert_eq!(finished[1].fields.0["termination"], "Cancelled");
    assert!(records.iter().any(|r| {
        r.fields
            .0
            .get("failure")
            .is_some_and(|v| v == "configuration")
    }));
    assert!(!format!("{records:?}").contains("SECRET"));
}

struct FailingTarget;
impl TargetSpace for FailingTarget {
    type Error = std::io::Error;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for FailingTarget {
    fn logp_grad(&self, _: &[f64], _: &mut [f64]) -> Result<f64, Self::Error> {
        Err(std::io::Error::other("SECRET_BACKEND_PAYLOAD"))
    }
}
#[test]
fn original_backend_error_is_returned_without_becoming_a_trace_payload() {
    let (dispatch, capture) = subscriber(true);
    let error = tracing::dispatcher::with_default(&dispatch, || {
        maximize(
            &FailingTarget,
            &[0., 0.],
            objective(),
            LbfgsOptions::new(10, 100).unwrap(),
        )
        .unwrap_err()
    });
    assert!(error.to_string().contains("SECRET_BACKEND_PAYLOAD"));
    let records = capture.0.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|r| r.fields.0.get("failure").is_some_and(|v| v == "evaluation"))
    );
    assert!(!format!("{records:?}").contains("SECRET_BACKEND_PAYLOAD"));
}

#[test]
fn monitor_worker_keeps_the_scoped_dispatch_and_parent_until_join() {
    let (dispatch, capture) = subscriber(true);
    tracing::dispatcher::with_default(&dispatch, || {
        let _parent = tracing::info_span!("monitor_caller").entered();
        Monitor::start(None, tokio_util::sync::CancellationToken::new())
            .unwrap()
            .finish()
            .unwrap();
    });
    let records = capture.0.lock().unwrap();
    let started = records
        .iter()
        .find(|r| {
            r.fields
                .0
                .get("event")
                .is_some_and(|v| v == "worker_started")
        })
        .unwrap();
    assert_eq!(
        started.scope.iter().map(|s| s.0).collect::<Vec<_>>(),
        vec!["monitor_caller", "monitor"]
    );
}

struct RejectSink;
impl Sink for RejectSink {
    fn draw(&self, _: &alea_runtime::session::OwnedDraw) -> Result<(), SinkError> {
        Err(SinkError::Backpressure)
    }
}
#[test]
fn sink_failure_remains_a_typed_outcome_and_emits_one_terminal_summary() {
    let (dispatch, capture) = subscriber(true);
    let s = session(false);
    let result = tracing::dispatcher::with_default(&dispatch, || {
        s.run(|_| Ok(Gaussian), Execution::Sequential, &RejectSink)
            .unwrap()
    });
    assert_eq!(result.status, Status::BackendFailure);
    assert!(result.chains.iter().all(|c| c.undelivered.is_some()));
    let records = capture.0.lock().unwrap();
    assert_eq!(
        records
            .iter()
            .filter(|r| r.target == "alea::session"
                && r.fields.0.get("event").is_some_and(|v| v == "finished"))
            .count(),
        1
    );
    assert!(records.iter().any(|r| {
        r.fields
            .0
            .get("status")
            .is_some_and(|v| v == "BackendFailure")
    }));
}

#[test]
fn traced_checkpoint_restart_keeps_the_uninterrupted_numerical_stream() {
    use alea_runtime::session::storage::Observer;
    use tokio_util::sync::CancellationToken;
    let build = |cancel: CancellationToken| {
        SessionBuilder::new("restart-trace", 2, vec![vec![0.; 2]])
            .sampler(SamplerConfig::Hmc(HmcOptions::new(0.2, 3).unwrap()))
            .draws(32)
            .seed([17; 32])
            .collect(true)
            .cancellation(cancel)
    };
    let full = build(CancellationToken::new())
        .build()
        .unwrap()
        .run(|_| Ok(Gaussian), Execution::Sequential, &Discard)
        .unwrap();
    let (dispatch, capture) = subscriber(true);
    let resumed = tracing::dispatcher::with_default(&dispatch, || {
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        let partial = build(cancel)
            .build()
            .unwrap()
            .run(
                |_| Ok(Gaussian),
                Execution::Sequential,
                &Observer(move |draw: &alea_runtime::session::OwnedDraw| {
                    if draw.index == 7 {
                        stop.cancel();
                    }
                    Ok(())
                }),
            )
            .unwrap();
        assert_eq!(partial.status, Status::Cancelled);
        let cp = partial.chains[0].checkpoint.clone().unwrap();
        let suffix = build(CancellationToken::new())
            .restart(vec![cp])
            .build()
            .unwrap()
            .run(|_| Ok(Gaussian), Execution::Sequential, &Discard)
            .unwrap();
        let mut draws = partial.chains[0].draws.clone();
        draws.extend_from_slice(&suffix.chains[0].draws);
        assert_eq!(draws, full.chains[0].draws);
        checkpoint_bytes(suffix.chains[0].checkpoint.as_ref().unwrap())
    });
    assert_eq!(
        resumed,
        checkpoint_bytes(full.chains[0].checkpoint.as_ref().unwrap())
    );
    let records = capture.0.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|r| r.target == "alea::storage"
                && r.scope.iter().any(|s| s.0 == "checkpoint_write"))
    );
}

#[test]
fn cold_model_and_exact_search_boundaries_are_subscribable_without_payloads() {
    let (dispatch, capture) = subscriber(true);
    tracing::dispatcher::with_default(&dispatch, || {
        use alea_math::exact::{ConeSearch, ConeSearchOptions, strict_cone_direction};
        assert!(matches!(
            strict_cone_direction(1, &[1.], &[], ConeSearchOptions::default()).unwrap(),
            ConeSearch::Direction(_)
        ));
        let mut b = alea_core::model::assembly::ModelBuilder::new();
        let p = b.real("SECRET_PARAMETER").unwrap();
        b.factor(p.expr()).unwrap();
        let _ = b.compile().unwrap();
    });
    let records = capture.0.lock().unwrap();
    assert!(records.iter().any(|r| {
        r.fields
            .0
            .get("outcome")
            .is_some_and(|v| v == "improving_ray")
    }));
    assert!(records.iter().any(|r| r.target == "alea::model"));
    assert!(!format!("{records:?}").contains("SECRET_PARAMETER"));
}

struct ToyChain {
    position: [f64; 2],
}
impl alea_mcmc::MarkovChain for ToyChain {
    type Error = std::io::Error;
    type Transition = alea_runtime::output::TransitionStats;
    fn position(&self) -> &[f64] {
        &self.position
    }
    fn step<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<Self::Transition, Self::Error> {
        self.position[0] = (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        self.position[1] += 1.;
        Ok(Self::Transition::default())
    }
}
#[test]
fn direct_ensemble_and_monitor_reports_share_the_scoped_subscriber() {
    use alea_runtime::{
        ensemble::{self, Chain, RunOptions, chain_rng},
        output::{ChainId, Draw, TransitionStats},
    };
    use std::{num::NonZeroUsize, ops::ControlFlow};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let (dispatch, capture) = subscriber(true);
    tracing::dispatcher::with_default(&dispatch, || {
        let _parent = tracing::info_span!("ensemble_caller").entered();
        let cancel = tokio_util::sync::CancellationToken::new();
        let monitor = Monitor::start(None, cancel.clone()).unwrap();
        let publisher = monitor.publisher();
        let mut chains: Vec<_> = (0..2)
            .map(|i| Chain {
                id: ChainId(i),
                sampler: ToyChain { position: [0.; 2] },
                rng: chain_rng([5; 32], ChainId(i)),
                sink: |_: Draw<'_, TransitionStats>| ControlFlow::Continue(()),
            })
            .collect();
        ensemble::run(
            &mut chains,
            &RunOptions {
                warmup: 0,
                draws: 32,
                chunk: NonZeroUsize::new(8).unwrap(),
                window: NonZeroUsize::new(16).unwrap(),
                coordinates: vec![0],
            },
            Execution::Parallel(&pool),
            &cancel,
            Some(&publisher),
        )
        .unwrap();
        monitor.finish().unwrap();
    });
    let records = capture.0.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|r| r.target == "alea::ensemble"
                && r.scope.iter().any(|s| s.0 == "ensemble_caller"))
    );
    assert!(records.iter().any(|r| {
        r.fields
            .0
            .get("event")
            .is_some_and(|v| v == "report_completed")
            && r.scope.iter().any(|s| s.0 == "ensemble_caller")
    }));
}

#[test]
fn fully_filtered_callbacks_and_caught_worker_panics_preserve_context_isolation() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let s = session(false);
    let off = tracing::Dispatch::new(
        Registry::default()
            .with(Capture::default().with_filter(tracing::level_filters::LevelFilter::OFF)),
    );
    let baseline = run(&s, &pool, false);
    let silent = tracing::dispatcher::with_default(&off, || run(&s, &pool, false));
    for (a, b) in baseline.chains.iter().zip(&silent.chains) {
        for (a, b) in a.draws.iter().zip(&b.draws) {
            assert_eq!(
                a.position.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.position.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    let (dispatch, capture) = subscriber(true);
    tracing::dispatcher::with_default(&dispatch, || {
        let _parent = tracing::info_span!("panic_caller").entered();
        let outcome = s
            .run::<Gaussian, _, std::io::Error, _>(
                |_| panic!("test factory panic"),
                Execution::Parallel(&pool),
                &Discard,
            )
            .unwrap();
        assert_eq!(outcome.status, Status::BackendFailure);
        let good = run(&s, &pool, false);
        assert!(good.chains.iter().all(|c| c.failure.is_none()));
    });
    let count = capture.0.lock().unwrap().len();
    pool.install(|| tracing::error!(target:"alea::session",event="must_not_leak"));
    assert_eq!(capture.0.lock().unwrap().len(), count);
}
