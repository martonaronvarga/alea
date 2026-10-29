# Installation

Alea requires Rust 1.95 or later and Cargo. The default build compiles
with stable Rust and do not require a C compiler, BLAS, Enzyme, or Nix.

## Add Alea to a project

Create an application:

```sh
cargo new my-fit
cd my-fit
```

Add these entries to its `Cargo.toml`:

```toml
[dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", rev = "717053a49172a0179b8c759914ea8ac1fc9e8979" }
```

The tag pins a specific source version. Keep the generated `Cargo.lock` with your
application so that dependency updates do not silently change a saved analysis.
Alea is installed from Git or a source checkout, rather than crates.io.

Copy the program in [Your first model](getting-started.md) into `src/main.rs`, then
run it:

```sh
cargo run --release
```

Release builds matter for numerical work. Cargo's default debug build is useful
for debugging, but its execution time does not represent sampling performance.

## Add probability distributions

Enable distribution laws and typed likelihood factors on the same dependency:

```toml
alea = { git = "https://github.com/martonaronvarga/alea", rev = "717053a49172a0179b8c759914ea8ac1fc9e8979", features = ["distributions"] }
rand = "0.10.1"
```

The `distributions` feature builds native cubature, so install a C compiler. On
Windows, use the MSVC Rust target and Visual Studio C++ build tools. Add `rand`
when your code supplies its own random-number generator.

Import the public API through `alea`:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use alea::distributions::scalar::Normal;
    use alea::prelude::*;
    let law = Normal::new(0.0, 1.0)?;
    let _ = law;
    Ok::<(), Box<dyn std::error::Error>>(())
}
```

The internal workspace crates do not need to appear in your application's
manifest. Alea uses GPL-3.0-or-later; see [Licensing](../LICENSING.md) for the terms
of the included components.

## Build from a checkout

```sh
git clone --recurse-submodules https://github.com/martonaronvarga/alea.git
cd alea
cargo test --manifest-path crates/Cargo.toml --workspace --locked
```

If you already cloned without submodules:

```sh
git submodule update --init --recursive
```

When using an unpacked source archive, add path dependencies instead:

```toml
alea = { path = "../alea/crates/alea", features = ["distributions"] }
```

## Optional backends

Start with the default build. Add a backend when your workload needs it.

| Feature | Requirements | Typical use |
|---|---|---|
| `analysis` | Stable Rust | Named CSV/JSON sampling artifacts |
| `faer` | Stable Rust | Optional matrix operations and fitted geometry |
| `openblas` | LP64 OpenBLAS headers and libraries | BLAS-backed matrix operations |
| `simd` | Pinned nightly Rust | Portable SIMD kernels |
| `branch-hints` | Pinned nightly Rust | Additional branch-probability hints |
| `std-autodiff` | Matching nightly Rust and Enzyme | Compiled automatic derivatives |

Wiener kernels already mark rare recovery paths as cold on stable Rust. The
`branch-hints` feature additionally selects nightly probability intrinsics.

For example, enable Faer:

```toml
alea = { git = "https://github.com/martonaronvarga/alea", rev = "717053a49172a0179b8c759914ea8ac1fc9e8979", features = ["faer"] }
```

The checkout includes pinned Nix environments for optional compilers and native
libraries:

```sh
nix develop .#stable
nix develop .#blas
nix develop .#autodiff
```

The default `nix develop` environment supplies the pinned nightly compiler.
Do not enable every feature on stable Rust: some features deliberately require
nightly. See [compatibility](releases.md) for the tested platforms and compiler policy.
