# alea-runtime

Sessions, parallel chains, diagnostics, and durable storage for Alea.

A session combines a model factory with a sampler, warmup settings, initial
positions, and a seed. It returns per-chain outcomes and diagnostics, and can
collect draws in memory or stream them to a sink.

```rust
use alea::session::SessionBuilder;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let session = SessionBuilder::new("my-model-v1", 2, vec![vec![0.0; 2]; 4])
    .seed([17; 32])
    .draws(1_000)
    .build()?;
# let _ = session;
# Ok(())
# }
```

This specifies four chains and uses the default NUTS settings without warmup.
Select warmup explicitly before fitting a model. Follow
[your first model](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/getting-started.md) for a complete fit, then
[diagnostics](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/diagnostics.md) and [saving a run](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/sessions.md).
Parallel execution uses a Rayon pool owned by your application.

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
