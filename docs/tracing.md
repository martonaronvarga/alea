# Tracing execution

Enable Alea's `tracing` feature to subscribe to structured execution events and
spans. Use it to follow a session, distinguish its chains, see how optimization
stopped, or investigate output and worker failures.

```toml
[dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", branch = "main", features = ["tracing", "distributions"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

Your application chooses the subscriber, filters, and output destination. Alea
does not install a global subscriber or read environment variables. The tracing
feature alone does not enable probability distributions or require a C compiler.

## Subscribe to one analysis

A scoped subscriber is useful when Alea is part of a larger application. This
complete program traces a two-chain normal-target run:

```rust
use alea::{
    distributions::StandardNormalTarget,
    ensemble::Execution,
    sampling::NutsOptions,
    session::{SamplerConfig, SessionBuilder, storage::Discard},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter("alea::session=info")
        .with_writer(std::io::stderr)
        .finish();

    tracing::subscriber::with_default(subscriber, || -> Result<(), Box<dyn std::error::Error>> {
        let session = SessionBuilder::new("normal-example", 2, vec![vec![0.0; 2]; 2])
            .sampler(SamplerConfig::Nuts(NutsOptions::new(0.2, 6)?))
            .draws(100)
            .collect(false)
            .build()?;
        let outcome = session.run(
            |_| Ok::<_, alea::target::DimensionError>(StandardNormalTarget::new(2)),
            Execution::Sequential,
            &Discard,
        )?;
        println!("session outcome: {:?}", outcome.status);
        Ok(())
    })
}
```

The same scoped dispatcher and parent span follow work onto a caller-supplied
Rayon pool. The diagnostics monitor captures the context active when it starts
and keeps that context on its dedicated worker. Reusing a pool for another
analysis restores the new caller's context; Alea does not change pool settings.

An application can instead install one global subscriber using
`tracing_subscriber::fmt().with_env_filter(...).try_init()`. Only the application
should do this. Multiple subscriber layers can format events, count them, or send
them to an external collector without changing Alea's inference APIs.

## Choose targets and levels

| Target | Spans | Events |
| --- | --- | --- |
| `alea::session` | `session`, `chain`; DEBUG `admission`, `target_factory`, `initialize` | `started`, `chain_started`, `chain_finished`, `sampling_started`, `finished`, `failed`, `admitted`, `admission_failed` |
| `alea::optimization` | `maximize` | `started`, `finished`, `failed`; DEBUG `iteration_completed` |
| `alea::ensemble` | `ensemble`; DEBUG `chunk` | `started`, `finished`, `failed` |
| `alea::run_chain` | `run_chain` | `started`, `finished`, `failed` |
| `alea::adaptation` | INFO `hmc_warmup`, `nuts_warmup`, `chees_warmup`; DEBUG `fisher_hmc_warmup`, `fisher_nuts_warmup`, `joint_hmc_warmup`, `joint_nuts_warmup`, `static_step_warmup`, `pathfinder` | `started`, `finished`, `failed`, reference warmup `cancelled`; DEBUG `metric_updated` |
| `alea::monitor` | `monitor` | `worker_started`; DEBUG `report_completed` |
| `alea::storage` | DEBUG `checkpoint_read`, `checkpoint_write`, `draw_file_create`, `draw_file_finish`, `draw_reader_create`, `analysis_create`, `analysis_seal` | `started`, `finished`, `failed` at operation boundaries |
| `alea::model` | DEBUG `compile`, `enzyme_lowering` | `started`, `finished`, `failed` |
| `alea::numerics` | DEBUG `recession_search`, `least_squares`, `bingham_normalization`; TRACE `wiener7_scaled_integral` | Cold-operation outcomes; TRACE `wiener7_integral_completed`, `wiener7_work_limit`, `wiener7_recalibration`, `wiener7_nonfinite_evaluation`, `cubature_backend_failure` |

INFO records operation boundaries and completed counts. DEBUG adds phase/chunk
context and diagnostic report summaries. Session numerical exhaustion and final
diagnostic warnings use WARN; terminal backend/configuration failures use ERROR.
A cancelled run is an ordinary outcome. Individual rejections and divergences do
not each produce a warning. Span close notifications also describe unwinding;
a panic need not produce a terminal event.

Every event has `schema_version = 1` and a static string `event` field. Other
fields contain counts, logical chain IDs, restart flags, and termination/status
codes. Chain IDs are local to the enclosing session, not globally unique. Parent
spans let applications attach their own analysis identity. No observations,
parameter vectors, gradients, RNG state, paths, caller provenance strings, or
arbitrary backend error messages are emitted.

Numerical events preserve the method's error interpretation. For example, a
Wiener quadrature discrepancy is labelled `estimated_quadrature`; a Bingham
normalization report is labelled `estimated`. Neither is promoted to a certified
error bound by tracing. Recession searches distinguish an improving ray, absence
of such a ray, and an unresolved resource limit. Least-squares events identify
the numerical rank policy. Inference-method guarantees remain those documented
by each method's API.

To read `RUST_LOG`, configure an application subscriber with
`tracing_subscriber::EnvFilter::try_from_default_env()`. For example,
`RUST_LOG=alea::session=info,alea::optimization=info,alea::monitor=debug`
selects those targets when the subscriber uses that filter.

## Performance and subscriber responsibilities

Instrumentation is compiled out when the feature is disabled. Enabled tracing
reports operation and chunk boundaries; it does not trace every draw, target
evaluation, leapfrog stage, graph node, or numerical series term. Filters avoid
formatting disabled fields. Alea does not perform extra target evaluations or
calculate additional diagnostics for tracing.

Ordinary Wiener4/5 evaluations have no tracing callsites. Wiener7 tracing is
restricted to scaled-integration summaries and recovery/nonfinite paths. Generated
Enzyme primal and derivative code contains no tracing instrumentation.

Subscriber callbacks run synchronously on the thread emitting an event. A slow,
allocating, or panicking subscriber can affect execution. For asynchronous output,
choose bounded buffering in the application and decide how to handle overflow.
Alea supplies no tracing queue, automatic exporter, or delivery guarantee. Detailed
formatting and output costs belong to the selected subscriber.

For fixed-work runs, a noninterfering subscriber leaves numerical results and RNG
state unchanged. Added execution time can affect wall-clock deadlines, concurrent
cancellation, and monitor-driven stopping. Tracing does not change the meanings of
optimizer termination, diagnostic warnings, or numerical accuracy reports.

Use [monitor subscriptions](diagnostics.md) for computed diagnostic reports and
[draw sinks](sessions.md) for samples. Traces provide execution context; typed
results and errors remain the source of numerical outcomes.

## Measuring tracing in your workload

The source checkout includes
`tools/downstream-conformance/examples/tracing_benchmark.rs` and
`tools/downstream-conformance/python/benchmark_tracing.py`. Build the example in
release mode with `faer`, copy the executable, then build with `faer,tracing` and
copy that executable separately. Pass those paths as `--off` and `--on` to the
Python runner, with a new `--output` directory. Both builds must use the same
compiler, source and other features.

The runner compares feature-off execution with no subscriber, a subscriber that
filters out Alea, and an INFO counting subscriber. It retains failed executions,
checks output fingerprints before comparing timings, and records source and
binary hashes. Each process performs three repetitions: the first is reported
separately, and the other two supply steady timings. Fifteen randomized paired
rounds are the default. Timing includes workload setup and output collection;
process startup is excluded. The counting subscriber buffers no event records.
These measurements do not measure peak memory or formatting/export costs.

The Bernoulli and Poisson optimization cases retain their actual termination
codes. Matching unsuccessful optimizations establish execution identity, not
successful fitting performance. Do not exclude them from an experiment or relax
convergence requirements to obtain a favorable timing.

### Measurements on October 29, 2026

On one x86-64 Linux host, all 1,020 planned processes completed across 17
workloads and four modes. All 3,060 repetitions matched their workload's output
fingerprint, including optimizer candidates, termination codes and counters.
Independent tests also compare draw/RNG and serialized checkpoint identity.

The following ratios are median paired steady times divided by the feature-off
time. Values below one should not be interpreted as an algorithmic speedup:
no numerical work was removed. These are measurements from one host, not bounds.

| Workload | No subscriber | Filtered | INFO counter |
| --- | ---: | ---: | ---: |
| Bernoulli optimization¹ | 0.999× | 0.989× | 0.990× |
| Exact cone search | 1.004× | 1.010× | 1.018× |
| hmc | 0.966× | 0.979× | 0.975× |
| nuts | 1.020× | 1.039× | 1.013× |
| 50D parallel NUTS | 1.009× | 1.029× | 0.967× |
| Poisson optimization¹ | 0.991× | 0.991× | 0.998× |
| rwmh | 1.024× | 0.981× | 1.006× |
| 100k HMC transitions | 0.920× | 0.965× | 0.950× |
| Small SVD solves | 0.984× | 0.963× | 0.995× |
| wiener4 density | 1.004× | 1.022× | 1.019× |
| wiener4 fused | 0.982× | 0.994× | 0.990× |
| wiener5 density | 1.014× | 1.013× | 0.993× |
| wiener5 fused | 1.017× | 0.988× | 0.989× |
| wiener7 density | 1.006× | 0.997× | 1.007× |
| wiener7 fused | 0.933× | 0.967× | 0.927× |
| wiener7 rare density | 1.024× | 0.996× | 1.010× |
| wiener7 rare fused | 1.003× | 0.995× | 1.025× |

¹ Both optimization workloads terminated with `LineSearchFailed` in every mode.
Their timings describe identical unsuccessful executions.

The predeclared objectives were at most 1% median overhead with a 95% upper
ratio of 1.02 for absent/filtered subscribers, and 2% median with an upper ratio
of 1.05 for INFO. **These objectives were not met across the workload suite.**
For example, filtered NUTS measured 1.040×, with a paired-bootstrap 95% interval
of [1.011, 1.060]. An earlier isolated 45-round cohort also measured a filtered
NUTS regression: 1.028× [1.022, 1.033]. Several other intervals were wide enough
that the small-overhead objective remains inconclusive. Ordinary Wiener4/5
paths emit no events, but that alone does not establish timing equivalence
between separately compiled binaries.

Keep tracing disabled when its observability is unnecessary. Measure enabled
tracing with your subscriber and workload before relying on an overhead budget.
Formatting, file/network output and peak-memory costs were not measured here.

A separate 15-round counting-subscriber comparison measured DEBUG cone searches
at 125.2 ms versus 118.3 ms without a subscriber (40,000 events per workload),
and DEBUG small SVD solves at 6.16 ms versus 5.34 ms (4,000 events). The differences
are approximately 0.17 and 0.20 microseconds per event, respectively, including
span overhead and measurement variation. The rare Wiener7 fused workload emitted
200 TRACE events; its 30.7 ms TRACE and 31.3 ms unsubscribed medians did not resolve
a positive overhead. These figures cover counting callbacks, not formatted output.
