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
| `alea::adaptation` | `chees_warmup` | `started`, `finished`, `failed` |
| `alea::monitor` | `monitor` | `worker_started`; DEBUG `report_completed` |

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
