# Alea compile-fail contracts

This standalone package compiles as an ordinary downstream consumer. Its seven
examples are intentionally invalid and pin existing public type boundaries:

- a consumed Hamiltonian phase token cannot be reused;
- consuming warmup freezes adaptation and prevents a second adaptive run;
- WALNUTS accepts only the leapfrog HMC proposal family for which its local
  refinement and reverse-selection correction are implemented;
- an explicitly approximate MCLMC chain cannot be converted into exact NUTS;
- an evolving NUTS termination reason cannot be matched exhaustively downstream;
- an emitted HMC transition cannot be assembled from unchecked public fields; and
- a real-valued parameter handle cannot supply a required positive scale handle.

`tools/ci/compile_fail.py` checks the compiler's structured JSON diagnostics,
including the expected Rust error code and the primary source span. The examples
must not be built as ordinary successful Cargo examples.

Run the CI contract from the repository root:

```console
python3 tools/ci/compile_fail.py
```
