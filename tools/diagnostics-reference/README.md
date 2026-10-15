# M6 independent diagnostic fixtures

`generate.py` executes ArviZ 0.23.4 / NumPy 2.4.2 / SciPy 1.17.1 from the
existing content-pinned `tools/blackjax-reference/default.nix` environment.
It does not invoke or translate Alea's implementation. Seed20261013 and all
cases are fixed before comparison; no statistical threshold fitting.

```sh
nix-build tools/blackjax-reference/default.nix --no-out-link
env XDG_CACHE_HOME=/tmp/alea-reference-cache MPLCONFIGDIR=/tmp/alea-reference-matplotlib PYTHON tools/diagnostics-reference/generate.py --check
nix develop .#stable --command cargo test --locked --manifest-path crates/Cargo.toml -p alea-runtime --test modern
```

Replace `PYTHON` with the environment's `bin/python3`. Without `--check`, the
generator regenerates the three checked-in `alea-runtime/tests/fixtures/modern-*`
files. Provenance records versions and SHA256 of raw draws and reference results.
Rust tests require relative/absolute tolerance2e-9, not diagnostic threshold agreement.

Cases: independent normal, correlated AR(1), antithetic AR(1), shifted chains,
different scales, odd length, minimal length, ties, heavy tails, drift, constant,
differently stuck, ragged, and nonfinite chains. Compare rank/folded/max R-hat,
bulk/tail/mean ESS, mean MCSE, mean and E-BFMI. Ragged inputs are deliberately
rejected, not fed to a silently rectangularized oracle. Constant/stuck R-hat
is undefined/nonfinite in ArviZ and a typed error in Alea. ArviZ's constant
indicator ESS convention is retained for discrete tail quantiles.

Definitions: [Vehtari et al.](https://arxiv.org/abs/1903.08008),
[ArviZ source](https://github.com/arviz-devs/arviz/blob/v0.23.4/arviz/stats/diagnostics.py).
