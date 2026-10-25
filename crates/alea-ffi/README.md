# alea-ffi

Checked Rust callbacks for synchronous native cubature.

This crate owns the C build and callback boundary used by Wiener integration.
It checks callback dimensions, reports backend failures, and keeps foreign
allocation ownership within the calling operation. Applications normally use it
through `alea-distributions`.

A C compiler is required. Source checkouts must include the cubature submodule:

```sh
git submodule update --init --recursive
cargo test --manifest-path crates/Cargo.toml -p alea-ffi --locked
```

See [installation](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/installation.md) for platform requirements.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE). This package includes or depends on
GPL-covered cubature; upstream terms remain intact.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
