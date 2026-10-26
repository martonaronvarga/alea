# alea-cli

Command-line tools for gradient checks and numerical experiments.

Run the tools from a source checkout:

```sh
cargo run --release --manifest-path crates/Cargo.toml -p alea-cli --bin alea
cargo run --release --manifest-path crates/Cargo.toml -p alea-cli \
  --bin alea-gradient-check -- all
```

These commands exercise supplied targets and derivative checks. For fitting your
own data, start with the [Rust user guide](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/getting-started.md).

## License

GPL-3.0-or-later; see [LICENSE](LICENSE). This package includes or depends on
GPL-covered cubature; upstream terms remain intact.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
