//! Workflow boundaries, not subtraction-based estimates of internal costs.
//! Session run includes target factory, initialization, transition and monitoring.
use alea_distributions::StandardNormalTarget;
use alea_mcmc::{NutsOptions, adapt::MetricKind};
use alea_runtime::{
    diagnostics::modern,
    ensemble::Execution,
    optimize::{LbfgsOptions, ObjectiveDescription, ObjectiveKind, maximize},
    session::{
        SamplerConfig, SessionBuilder, Warmup,
        storage::{ChannelSink, Discard, Sink},
    },
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn builder(dimension: usize, warmup: usize, draws: usize, collect: bool) -> SessionBuilder {
    SessionBuilder::new(
        "criterion-standard-normal-v1",
        dimension,
        vec![vec![0.1; dimension]; 2],
    )
    .seed([91; 32])
    .sampler(SamplerConfig::Nuts(NutsOptions::new(0.2, 6).unwrap()))
    .warmup(if warmup == 0 {
        Warmup::None
    } else {
        Warmup::Covariance {
            iterations: warmup,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into().unwrap(),
            search: None,
        }
    })
    .draws(draws)
    .collect(collect)
}

fn workflow(c: &mut Criterion) {
    for dimension in [4, 64] {
        let mut group = c.benchmark_group(format!("workflow/{dimension}"));
        group.bench_function("session_configuration", |b| {
            b.iter(|| black_box(builder(dimension, 80, 128, false).build().unwrap()))
        });
        // Session requires retained draws. These names explicitly include the
        // one-draw tail; isolated adaptation is in the existing warmup target.
        // Never subtract these timings to manufacture a warmup-only estimate.
        for (name, warmup, draws, collect) in [
            ("initialize_sample1", 0, 1, false),
            ("warmup80_sample1", 80, 1, false),
            ("sample128_discard", 0, 128, false),
            ("sample128_collect", 0, 128, true),
            ("warmup80_sample128", 80, 128, true),
        ] {
            let session = builder(dimension, warmup, draws, collect).build().unwrap();
            let run = || {
                session
                    .run(
                        |_| Ok(StandardNormalTarget::new(dimension)),
                        Execution::Sequential,
                        &Discard,
                    )
                    .unwrap()
            };
            let admitted = run(); // Same seed/session on every invocation.
            assert!(admitted.chains.iter().all(|c| c.failure.is_none()
                && c.completed == draws
                && c.warmup_completed == warmup));
            group.bench_function(name, |b| b.iter(|| black_box(run())));
        }
        let retained = builder(dimension, 0, 1, true)
            .build()
            .unwrap()
            .run(
                |_| Ok(StandardNormalTarget::new(dimension)),
                Execution::Sequential,
                &Discard,
            )
            .unwrap();
        let draw = &retained.chains[0].draws[0];
        group.bench_function("owned_draw_clone", |b| {
            b.iter(|| black_box(black_box(draw).clone()))
        });
        let (sink, receiver) = ChannelSink::bounded(1).unwrap();
        group.bench_function("channel_clone_enqueue_dequeue", |b| {
            b.iter(|| {
                sink.draw(black_box(draw)).unwrap();
                black_box(receiver.recv().unwrap())
            })
        });
        let target = StandardNormalTarget::new(dimension);
        let options = LbfgsOptions::new(100, 1000).unwrap();
        let initial = vec![0.5; dimension];
        let objective = ObjectiveDescription {
            kind: ObjectiveKind::Other,
            measure: "real".into(),
            model: "native-standard-normal".into(),
            data: "none".into(),
            source: "criterion".into(),
        };
        group.bench_function("lbfgs_complete_fit", |b| {
            b.iter_batched(
                || (objective.clone(), options.clone()),
                |(objective, options)| {
                    black_box(maximize(&target, black_box(&initial), objective, options).unwrap())
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.finish();
    }
    // Deterministic non-degenerate fixtures; diagnostics data preparation excluded.
    for n in [128, 1024, 8192] {
        let chains: Vec<Vec<_>> = (0..4)
            .map(|k| (0..n).map(|i| ((i * 17 + k * 13) as f64).sin()).collect())
            .collect();
        let refs: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        assert!(modern::summarize(&refs).unwrap().bulk_ess.is_finite());
        let mut group = c.benchmark_group("diagnostics");
        group.throughput(Throughput::Elements((4 * n) as u64));
        group.bench_function(format!("rank_rhat_bulk_tail_ess/{n}"), |b| {
            b.iter(|| black_box(modern::summarize(black_box(&refs)).unwrap()))
        });
        group.finish();
    }
}

#[cfg(feature = "analysis")]
fn serialization(c: &mut Criterion) {
    use alea_core::model::{assembly::ModelBuilder, prediction::NamedOutput};
    use alea_runtime::{
        analysis::{AnalysisSpec, AnalysisWriter, ChainStatus, Provenance},
        output::{ChainId, Draw, Phase, SCHEMA_VERSION, TransitionStats},
    };
    use criterion::BatchSize;
    use std::collections::BTreeMap;
    let spec = || {
        let mut b = ModelBuilder::new();
        let x = b.positive("scale").unwrap().expr();
        b.factor(x).unwrap();
        let model = b.compile().unwrap();
        AnalysisSpec::new(
            NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap(),
            Provenance {
                model: "criterion".into(),
                data: "none".into(),
                source: "native".into(),
                method: "fixture".into(),
                settings: BTreeMap::new(),
            },
            vec![ChainId(0)],
        )
    };
    let row = |i| Draw {
        schema_version: SCHEMA_VERSION,
        chain: ChainId(0),
        phase: Phase::Sampling,
        index: i,
        position: &[0.2],
        transition: &(),
        stats: TransitionStats::default(),
    };
    let create = || AnalysisWriter::new(Vec::with_capacity(65536), spec()).unwrap();
    let mut group = c.benchmark_group("analysis_memory");
    group.bench_function("header_setup", |b| {
        b.iter_batched(
            || (Vec::with_capacity(65536), spec()),
            |(bytes, spec)| black_box(AnalysisWriter::new(bytes, spec).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("one_row_encode_hash_buffered", |b| {
        b.iter_batched(
            create,
            |mut w| {
                w.write_draw(&row(0), &[]).unwrap();
                black_box(w)
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("flush_manifest_after128", |b| {
        b.iter_batched(
            || {
                let mut w = create();
                for i in 0..128 {
                    w.write_draw(&row(i), &[]).unwrap();
                }
                w
            },
            |w| {
                let mut manifest = Vec::new();
                let csv = w
                    .finish(&mut manifest, &[(ChainId(0), ChainStatus::Completed)])
                    .unwrap();
                black_box((csv, manifest))
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}
#[cfg(not(feature = "analysis"))]
fn serialization(_: &mut Criterion) {}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=workflow, serialization}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
