# alea-math

Aligned storage and numerical geometry for Alea.

The crate provides initialized buffers, Euclidean metrics, Cholesky factors,
interpolation, and numerical helpers. Most applications reach these through the
samplers; import them directly when selecting a metric or supplying storage.

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use alea::math::{buffer::OwnedBuffer, metric::IdentityMetric};

    let position = OwnedBuffer::from_fn(3, |i| i as f64);
    let metric = IdentityMetric::new(3);
    assert_eq!(position.len(), 3);
    let _ = metric;

    Ok(())
}
```

The default backend uses stable Rust 1.95 or later. Optional `faer`, `openblas`,
and `simd` features have separate compiler and library requirements. See
[installation](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/installation.md) and [metric conventions](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/metric-contract.md).

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
