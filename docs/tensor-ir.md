# Tensor regions, immutable data and explicit lowering

Alea now has a shape-specialized deterministic tensor SSA IR alongside its
scalar probability assembly tape. It represents whole array operations and
structured control flow; it is not a generative modeling language. Native
distributions, parameter charts, Jacobians and samplers are unchanged.

This is research infrastructure, not a claim of theoretical novelty, optimal
tensor kernels, or speed parity with Stan/JAX. Those claims require independent
measurements and broader qualification than a richer representation supplies.

## Target-owned workspace admission

`TensorProgram::workspace_bytes()` and `BoundProgram::workspace_bytes()` expose
the checked native per-execution estimate, including retained scan frames and
both conditional branches. `binding.workspace_with_limit(bytes)` rejects an
insufficient budget before allocating any execution frames; it returns typed
required/allowed bytes. `TensorFactor::check_workspace_limit(bytes)` permits the
same admission before registering a factor into a model. Rebinding keeps the
fixed plan and its estimate. Budget each simultaneous chain separately.

These limits complement session sampler/storage admission, which excludes
target-owned scratch. They exclude immutable data, allocator bookkeeping,
generated AD storage and unrelated process memory; they are not an OOM guarantee.
Ordinary `scan` retains O(steps × body workspace) reverse storage. Opt-in
`checkpointed_scan` uses segmented replay for substantial tapes; see the
[checkpoint follow-up](#checkpointed-scans-and-matrix-vector-kernels). Neither
mode performs general liveness compaction. A budget never shortens a scan or
changes its math.

## What the reference implementations teach

[Stan's matrix multiply reverse implementation](https://github.com/stan-dev/math/blob/develop/stan/math/rev/fun/multiply.hpp)
registers matrix-level reverse callbacks, retaining whole values rather than
requiring a scalar AD object per matrix entry. Its
[callback implementation](https://github.com/stan-dev/math/blob/develop/stan/math/rev/core/callback_vari.hpp)
and [arena-backed stack](https://github.com/stan-dev/math/blob/develop/stan/math/rev/core/autodiffstackstorage.hpp)
also make lifetime ownership part of correctness. Alea uses immutable shared
plans and execution-local owned scratch instead of adopting a global arena stack.

[JAX's jaxpr](https://docs.jax.dev/en/latest/601/jaxpr.html) makes numerical
operations and subregions explicit. Its
[scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) keeps carry
shapes fixed and retains a loop instead of host-language unrolling.
[cond](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) distinguishes
conditional execution from evaluating both sides then selecting a value. Alea
adopts these structural ideas, not JAX's tracing, batching, XLA or GPU backend.
These are implementation references, not evidence that the systems are identical.

## Fixed-shape block programming

```rust
use alea::model::tensor::{BinaryOp, Shape, TensorBuilder};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut b = TensorBuilder::new();
let x = b.input("x", Shape::new(vec![3])?)?;
let w = b.data("weights", Shape::new(vec![3])?)?;
let product = b.binary(BinaryOp::Multiply, x, w)?;
let total = b.sum(product)?;
let plan = b.compile(total)?;
assert_eq!(plan.node_count(), 4); // Not one graph per element.
let old = plan.bind(&[("weights", &[1.0, 2.0, 3.0])])?;
let new = old.rebind("weights", &[3.0, 2.0, 1.0])?;
let mut scratch = old.workspace(); // Same-plan snapshots may reuse scratch.
let mut value = [0.0];
let mut score = [0.0; 3];
old.value_and_vjp(&[1.0, 2.0, 3.0], &[1.0], &mut value, &mut score, &mut scratch)?;
assert_eq!(value, [14.0]);
assert_eq!(score, [1.0, 2.0, 3.0]);
new.value_and_vjp(&[1.0, 2.0, 3.0], &[1.0], &mut value, &mut score, &mut scratch)?;
assert_eq!(value, [10.0]);
assert_eq!(score, [3.0, 2.0, 1.0]);
# Ok(())
# }
```

Shapes are fixed, row-major f64 arrays; an empty axis list means scalar and
zero-sized dimensions are allowed. Inputs flatten in declaration order. Data are
finite immutable snapshots, not parameters. Named binding replacement validates
before returning a new snapshot and shares the original plan/unchanged arrays.
It never mutates another target. Nested regions own their bindings: pass parent
data as explicit operands when rebinding must affect them.

Operations include elementary unary maps, four arithmetic binary maps with
trailing-axis broadcasting, full sum, constant-index gather, zero-copy reshape,
rank-two matrix multiplication, scalar comparison/conditional and fixed-trip scan.
Reshape aliases both value and adjoint storage. Gather repeats scatter-add in
reverse. Unreachable operations remain inspectable but do not execute or export.
Only the scalar result bridge `TensorFactor` connects to `ModelBuilder`; arbitrary
tensor outputs use `BoundProgram::evaluate` / `value_and_vjp` directly.

Broadcast maps are specialized at construction. If output element j reads input
element m(j), its pullback accumulates `input_bar[m(j)] += output_bar[j] * local_score`.
Matrix products use the usual `A_bar += C_bar B^T` and `B_bar += A^T C_bar`
pullbacks, without materializing a Jacobian. Current matrix execution is a
compensated reference kernel, **not** tuned BLAS, batched GEMM or a GPU kernel.

## Structured conditional and loop semantics

```rust
use alea::model::tensor::{BinaryOp, Shape, TensorBuilder};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut body = TensorBuilder::new();
let carry = body.input("carry", Shape::scalar())?;
let row = body.input("row", Shape::scalar())?;
let gain = body.input("gain", Shape::scalar())?;
let scaled = body.binary(BinaryOp::Multiply, carry, gain)?;
let next = body.binary(BinaryOp::Add, scaled, row)?;
let body = body.compile(next)?.bind(&[])?;

let mut b = TensorBuilder::new();
let initial = b.input("initial", Shape::scalar())?;
let gain = b.input("gain", Shape::scalar())?;
let xs = b.data("xs", Shape::new(vec![3])?)?;
let final_carry = b.scan(initial, xs, &[gain], body)?;
let program = b.compile(final_carry)?.bind(&[("xs", &[1.0, 2.0, 3.0])])?;
let mut out = [0.0];
program.evaluate(&[0.0, 2.0], &mut out, &mut program.workspace())?;
assert_eq!(out, [11.0]); // ((0*2 + 1)*2 + 2)*2 + 3
# Ok(())
# }
```

`cond` takes a scalar comparison, explicit operands and two bound regions with
matching input/output shapes. Only the chosen branch executes, in either pass.
The derivative is pathwise; predicates are not differentiated and no smoothness
claim applies at a switching surface. An invalid unselected log branch must not
fail the valid selected branch. This is not eager elementwise `select` or vmap.

`scan` body inputs are carry, one leading-axis row and captures. The final carry
is returned; zero trips are the identity. For `c[t+1]=f(c[t],xs[t],a)`, reverse
propagates the carry adjoint backwards and accumulates row/capture adjoints per
step. Ordinary scans retain O(trips * body workspace) storage; nested frame/storage
overflow is rejected before allocation. `checkpointed_scan` explicitly opts into
native segment replay (below). There is no dynamic while, automatic checkpoint
selection for ordinary scans or reverse-through-discrete-state claim.
Finite materialized values/adjoints are required; errors preserve caller buffers.
Construction specializes derivative activity through region operands. Immutable
data and predicates do not acquire spurious derivatives (for example, an unrelated
`sqrt(0)` data constant does not create an infinite parameter gradient).
Panics propagate, and scratch can be reused afterwards. This is not full-range
floating-point certification. Unused-data bindings are still shape-validated.

## Rebind an existing native model without rebuilding topology

`CompiledModel::rebind::<P>(expression, update)` consumes the model, downcasts its
descriptor and validates identical expression inputs and full semantic signature.
The optional batch rule and its row count must remain unchanged too.
It prepares the new callback, scratch, source and batch rules before replacement.
Failure returns `RebindError { model, source }`, preserving the unchanged model
and original typed error. A panic propagates and consumes the model. Recover a
model from its target before rebinding; never reuse a PointState from an old
density. External generated functions must be regenerated/recompiled too.

`NormalRegression`, `BernoulliLogitRegression` and `PoissonLogRegression` expose
`with_data` for fixed-row/fixed-column replacements, retaining typed observations,
design orientation, coefficients and offsets. `TensorFactor::with_data` rebinds
a named tensor binding. Changing parameter layout, dimensions or dependency
handles requires rebuilding, not pretending it is data-only rebinding.

## Automatic native batch reporting and source lowering

The primitive author supplies optional `batch_rule` / `source_rule` capabilities.
The standard scalar Normal and all three native regression families implement
these rules. Registration keeps one fused native analytic likelihood/VJP;
**reporting does not scalarize the sampler's graph**. Pointwise rules reuse Alea's
existing normalized scalar distributions; no likelihood formula is reimplemented.

`model.batch_likelihood()` consumes/selects the likelihood and applies the
registered rules automatically. `units()` reports `(factor name, local row)` in
declaration/row order. `evaluate(q, rows)` constrains coordinates without a density
Jacobian, admits the required forward values/scores, then stages/validates every
row before publication. Priors and potentials are excluded. The extra forward
plus row pass is for reporting, not a new faster sampler evaluation. Compensated
row/fused reductions can differ in floating-point rounding.

Local rows are **not** global observation IDs, marginal likelihoods, leave-group-
out units or proof of independence/propriety. Existing explicitly labelled
`PointwiseLikelihood` remains available. Opaque or algebraically composed joint
factors without an explicit decomposition return `MissingPointwise`; no partition
is guessed from shape, a name or distributive algebra.

Source generation is lazy: capability inspection and model construction never
format captured data. Export emits counted loops and structured regions with
data frozen from that binding. Normal uses its existing trusted scalar primal;
link regressions call the existing distribution implementations. Opaque active
nodes still return `MissingPrimal`. No finite-difference/analytic-VJP fallback is
invented. Exported source uses checked admission and preserves empty-region errors
through a separate validity flag, not a meaningless zero-length NaN sentinel.

## Qualification, limitations and reproduction

New tests cover block VJPs vs central differences, matrix inputs, every elementary
operation, rank-three broadcasting, shared reshapes/repeated gathers, lazy log
branches, zero/long scans and captures, immutable rebinding, dead operations,
empty-matrix/recursive-storage overflow, and zero allocations in repeated calls.
Batch tests cover all three laws, original typed causes, incomplete callback
writes, panic/recovery, output staging and lazy source generation. Model-level
tests stay independent of sampler/performance runs.

The final binary-kernel tests also compare scalar **bits**, including aliased
reshapes, inactive data, scalar broadcasts, zero seeds, extreme finite values and
nonzero preexisting adjoints. Contiguous fast paths must retain the original
interleaved two additions for repeated operands and ordered scalar reductions.

Actual compiled tensor source and compiler-generated derivatives are checked
against native execution and independent references. The original matrix compiler
failure and a subsequently exposed scan third-derivative defect are fixed as
described below. Matrix, scalar-carry scan, both lazy branch directions, zero-trip
scan and empty-result error fixtures run in the default Enzyme gate. These are
executed examples, not a guarantee that the experimental compiler handles every
shape/program, nor statistical qualification of a tensor inference workload.

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml -p alea-core --test tensor_ir --test primitive_lowering
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml -p alea-distributions --test batch_lowering
nix develop .#autodiff --command cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked --test tensor_source --test generated --test tensor_matrix --test tensor_control
```

No dynamic shapes, multi-dtype tracing, general graph rewrite optimizer, GPU/JIT,
BLAS dispatch or generative site language is claimed. The small tensor/native-law
workflow qualified below is not arbitrary-model qualification. The next optimization must follow a measured bottleneck,
not remove numerical checks to make a compiler or timing look better.

## Compiler AD failure and fix (2026-11-02)

There were two distinct defects. Fixing compilation alone would have shipped an
incorrect third derivative.

The original forward-over-reverse matrix fixture failed Enzyme type analysis with
`Illegal updateAnalysis ... Float@double ... Integer`. A standalone 2×2 matrix
calculation, with no Alea dependency, distributions, compensation or validation,
reproduced it. In the emitted LLVM, local floating buffers and their shadows were
byte-initialized or copied through integer storage; nested differentiation then
conflicted with floating loads/stores. Updating to a newer content-pinned compiler
alone reproduced the original error. Neither probability mathematics nor NUTS
was necessary for this failure.

Forensic minimal reproducer (the intentionally failing version, not the fixed
lowering): save this as `repro.rs`, then run the command below on the pinned
toolchain. It has no Alea dependency.

```rust,ignore
#![feature(autodiff)]
use std::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_reverse(score, Duplicated, Active)]
fn primal(x: &[f64; 2]) -> f64 {
    let a = [[1.0, 0.5], [-0.5, 1.5]];
    let mut product = [0.0; 2];
    for i in 0..2 {
        for j in 0..2 { product[i] += a[i][j] * x[j]; }
    }
    let mut sum = 0.0;
    for p in product { sum += p * p; }
    -0.5 * sum
}
#[autodiff_forward(score_jvp, Dual, Dual, Const)]
fn gradient(x: &[f64; 2], g: &mut [f64; 2]) -> f64 { score(x, g, 1.0) }
fn main() {
    let mut g = [0.0; 2];
    let mut h = [0.0; 2];
    score_jvp(&[0.3, -0.4], &[1.0, 0.0], &mut g, &mut h);
    assert!((h[0] + 1.25).abs() < 1e-12);
    assert!((h[1] - 0.25).abs() < 1e-12);
}
```

```sh
nix develop .#autodiff --command rustc --edition 2024 -O -C lto=fat -C embed-bitcode=yes -Zautodiff=Enable repro.rs
```

The source boundary now preserves typed storage explicitly:

- `CompiledModel::rust_primal_with_workspace` and its objective counterpart return
  a body plus a checked `f64` element count. `Primitive::source_workspace` declares
  source-only scratch independently of native evaluation scratch. Combined sizes,
  pruning and consuming rebinding are tested; source formatting remains lazy.
- Factor arguments and tensor mutable intermediates live in an external float
  arena, passed as a duplicated argument to compiler AD. Each differentiated
  wrapper also receives its working buffers explicitly. Inputs, constants, data
  and reshapes are borrowed views. Copies are typed loops, with direct operations
  for one-element blocks. Larger maps and scan iterations are not source-unrolled.
- Generated region helpers remain visible to AD through inlining. The standalone
  `BoundProgram::rust_function` retains its old primal wrapper signature, but owns
  local scratch: use the compiled-model exporter for nested compiler AD.

After compilation succeeded, the scan's density, score and Hessian passed, but
its third derivative failed. For the three-step recurrence
`r = a*b^3 - 0.1*b^2 + 0.25*b + 0.5` and potential `-log(abs(r))`, at
`[a,b]=[-0.7,-0.57]`, the expected `U_aaa` is about `0.1351718172151034`.
The generated result was `0.3010938670629443`: an extra Hessian term of
`0.1659220498478410`. Other mixed entries broke permutation symmetry.
A raw JVP with **zero position direction** still returned a nonzero derivative.
This is a differentiation defect, not finite-difference noise or a tolerable
floating-point difference.

The effective correction is in `alea-autodiff/src/codegen.rs`: the HVP direction
is passed to the outer forward AD as `Dual` with an **explicit zero tangent**,
instead of `Const`. This is mathematically the same constant-direction partial
derivative, but avoids the pinned backend's erroneous nested treatment of that
reference. Independent recurrence derivatives, all mixed third entries, zero
direction, directional linearity, both branches and zero trips now test this
contract. Changing the reverse seed, marking the discarded score accumulator
constant, or giving every scan iteration fresh scratch did not fix the defect;
those experiments were reverted.

No compensated reduction, finite/domain check, probability kernel, analytic
probability score, transform, Jacobian or sampler was changed. No loose type mode,
fast math, parameter clipping, finite-difference fallback or altered tolerance was
used. The [Rust autodiff flags](https://doc.rust-lang.org/unstable-book/compiler-flags/autodiff.html)
explicitly warn that loose type mode can produce incorrect gradients;
[type trees](https://rustc-dev-guide.rust-lang.org/autodiff/type-trees.html) explain
why memory layout matters to compiler AD.

The wrapper's stack scratch is now explicitly O(d+S), where S is the exported
source workspace count; third-order calls supply eight S-sized float buffers.
This does not bound Enzyme's internal reverse tape or promise arbitrary-size
programs fit a thread stack. Matrix regression tests count Rust allocations on
repeated successful calls; that counter is not a general proof about backend C
allocations. Compiler throughput and end-to-end tensor inference performance were
not benchmarked by this fix. The executor timings below predate this cold-source
change and retain their original source fingerprints.

The implementation lesson is to test each claimed capability separately:
native VJP correctness → executable primal → compiler-generated score/HVP →
nonzero mixed third derivatives and directional laws. A quadratic's zero third
tensor alone missed the scan defect. Keep an independent nonlinear reference,
test zero seeds and both branch directions, preserve invalid-state recovery,
and reject compiler success as a substitute for numerical evidence.

## Measured block-executor deficit and fix

The first implementation had fewer nodes but was slower. On the same deterministic
dyadic objective `sum((slope * weights + 0.125)^2)`, the clean three-process baseline
put the tensor factor at 33.953/339.558 microseconds for 1,000/10,000 elements,
versus 24.150/263.869 for scalar assembly. The native regression likelihoods were
already fused and are **not** this comparison. No probability law or derivative
was reimplemented in the benchmark.

User-space `perf` instruction-address sampling at 499 Hz for five seconds found
60.72% in tensor reverse and 38.56% in forward at 10,000 elements. Annotated
assembly exposed repeated operation/map/activity checks and indexed adjoint
accesses inside element loops. Stack sampling was denied by host permissions;
flat `--all-user` sampling worked, with zero lost samples in these captures.

The fix is local to `model/tensor/kernels.rs`: dispatch arithmetic and common maps
once per block; use contiguous zipped slices for identity/scalar broadcasts;
specialize inactive-side pullbacks; keep the general gather/scatter fallback.
Disjoint destinations may use separate passes because their accumulation orders
do not interact. Aliased operands keep two ordered gradient additions, not
`seed * (da + db)`. No fast-math, clipping, removed finite checks, distribution
rewrite or change to the compensated sum was used. Sampled generated forward
code contains packed `mulpd`/`addpd` instructions; this is not a claim that every
reverse kernel or reduction is SIMD-vectorized.

Each cell below is the median of three process medians, with the observed
process-median range in parentheses, in microseconds. These ranges are **not**
confidence intervals or a full noise model.

| Elements / path | Initial block executor | Hoisted block executor |
|---|---:|---:|
| 1,000 / scalar control | 24.150 (24.051–24.531) | 24.341 (24.120–24.892) |
| 1,000 / tensor factor | 33.953 (33.709–34.241) | 11.517 (11.428–11.653) |
| 1,000 / tensor direct | 33.682 (33.632–33.848) | 11.619 (11.496–11.630) |
| 10,000 / scalar control | 263.869 (263.729–266.805) | 263.749 (262.603–272.056) |
| 10,000 / tensor factor | 339.558 (339.385–343.647) | 117.557 (117.169–118.937) |
| 10,000 / tensor direct | 338.959 (338.409–419.324) | 117.469 (116.816–118.577) |

Tensor-factor throughput changed from 29,453 to 86,825 evaluations/second at
1,000 elements and 2,945 to 8,507 at 10,000. The measured factor improvement is
2.95×/2.89× over the **first implementation in this increment**, not a historical
Alea or Stan speedup. The final factor is 2.11×/2.24× faster than the corresponding
scalar control on this fixture. Afterward, the profile was 43.73% forward,
52.86% reverse including the extracted kernel, and 3.30% memory copy. Relative
profile shares alone do not establish absolute improvement; the timings do.

Protocol: three sequential processes, forward/reverse/forward scalar-factor
ordering; 30 Criterion samples per case, 0.5-second warmup and 2-second measurement.
Direct tensors follow the two assembly cases. No concurrent builds or other
benchmark jobs; shared desktop background load remains. No affinity or frequency
control, Intel i5-8250U/x86-64, powersave governor, one evaluation thread,
Rust 1.95.0/LLVM 22.1.2 from the pinned stable Nix shell, release opt-level 3/fat
LTO, no custom RUSTFLAGS. Compilation/construction/source export and samplers are
outside the timed closure; every path checks exact fixture value/score equality
before timing. This is a hot repeated-evaluation test, not an end-to-end inference
benchmark. An overlapping exploratory run and an earlier-source run were excluded
before this comparison; a slow final direct-process result remains in the range.

```sh
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml -p alea-core --bench tensor_tape -- --save-baseline tensor-hoisted-forward-1
nix develop .#stable --command env ALEA_TENSOR_REVERSE=1 cargo bench --manifest-path crates/Cargo.toml -p alea-core --bench tensor_tape -- --save-baseline tensor-hoisted-reverse-1
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml -p alea-core --bench tensor_tape -- --save-baseline tensor-hoisted-forward-2
```

## Execution-plan follow-up: immutable bindings and active adjoints

The 2026-11-02 follow-up separates three questions: whether supported IR semantics
are correct, whether native execution is cheaper, and whether generated source
is fast. A representation with fewer nodes does not establish any of these alone.

A fresh release Callgrind profile of 1,000 evaluations of
`sum((slope * weights)^2)`, with 1,000 weights, records 126,338,272 instructions.
About 19.1% are attributed to memory clearing. The executor also recopied and
rechecked immutable arrays and checked adjoints for inactive data on every call.
This is instruction attribution, not CPU-time percentages.

The change is local to `tensor/execute.rs`:

- Workspace construction initializes constants and immutable data once. Cached
  data retain owning `Arc`s. Same-plan rebinding compares Arc identity and
  refreshes changed data before use; a bare cached address would be incorrect
  after an old snapshot is dropped and the allocator reuses its address.
- Derivative activity lowers to merged, owned adjoint ranges per workspace.
  Clear/check those ranges, not inactive data slots. Reshape aliases do not
  duplicate ranges; repeated-operand accumulation order is unchanged. Every
  input gradient is still initialized and every materialized arithmetic result
  is still admitted, including operations with parameter-independent results.
- Input/binding/constant finiteness guarantees are reused at their admission
  boundary. Arithmetic results and live adjoints use the existing runtime-SIMD
  nonfinite checker. This does not remove domain checks or reorder reductions.
- Workspace-byte admission includes range storage and cached Arc metadata,
  recursively through both branches and scan frames. Tests compare actual
  retained allocator bytes with the estimate. Dense buffers and retained scan
  frames are not liveness-compacted; the improvement is memory **traffic**, not
  an assertion of smaller total storage. A workspace can retain its last binding
  arrays until refresh/drop; immutable payload bytes remain outside its budget.

No probability law, score, chart, Jacobian, matrix summation, scalar reduction or
source differentiation rule changes. Core continues to forbid unsafe code. New
tests repeatedly drop/rebind snapshots, switch seeds, fail inside a log block,
recover with the same workspace, and verify zero evaluation allocations. Existing
alias, singular-inactive-data, empty-output, branch and scan contracts still pass.

### Current native timings

Evidence: `../alea-inference-evidence/tensor-lowering-2026-11-02.32pDo4`.
Baseline production is the previous sufficient-statistics qualified source
archive `0712bb70d0a4f48cf9d729e2aea9b4dbbec0fec09bb31ff4f2afbfc89a0b9769`.
Both saved executables use the **identical benchmark-only harness overlay** for
the new matrix/branch/scan/scalar controls. Seven alternating pairs per shape,
140 jobs/70 pairs, all successful and exact-checksum compatible, no missing jobs.
Each call consumes the value and all gradient coordinates. Checksums are not a
replacement for the separate coordinate-level correctness tests.

| Native path / size | Before / after median µs per call | Paired before/after median [range] |
| --- | ---: | ---: |
| Elementwise tensor / 1,000 | 10.123 / 6.299 | 1.628 [1.554,1.785] |
| Elementwise tensor / 10,000 | 104.668 / 63.035 | 1.660 [1.625,2.150] |
| Matrix-vector tensor, 8 coefficients / 1,000 rows | 89.012 / 64.579 | 1.403 [1.365,1.495] |
| Matrix-vector tensor, 8 coefficients / 10,000 rows | 915.401 / 604.830 | 1.493 [1.219,1.544] |
| Lazy branch / 1,000 elements | 12.507 / 6.991 | 1.772 [1.661,1.810] |
| Lazy branch / 10,000 elements | 129.547 / 79.580 | 1.665 [1.515,1.884] |
| Scalar-carry scan / 8 steps | 1.360 / 1.346 | 1.008 [0.988,1.033] |
| Scalar-carry scan / 128 steps | 19.491 / 18.858 | 1.026 [0.994,1.048] |
| Scalar-tape control / 1,000 terms | 19.997 / 22.534 | 0.881 [0.745,0.948] |
| Scalar-tape control / 10,000 terms | 223.710 / 233.822 | 0.973 [0.884,1.010] |

Elementwise, matrix and branch paths win every measured pair; scans have no
consistent material gain. The scalar-tape control **regresses**, especially at
1,000 terms, despite no scalar-assembly implementation change. Its causal
attribution remains unresolved; do not remove this control or claim a universal
improvement. Final elementwise Callgrind count is 76,470,496 instructions, about
39.5% fewer than baseline. Instrumented durations are excluded from wall results.
With the identical expanded harness, baseline records 126,340,190 instructions;
the scalar control records 218,870,622→218,868,686. Thus the scalar timing loss
does not correspond to extra counted arithmetic/work; its hardware/code-layout
cause is still not established by these instruction profiles.
The remaining reverse arithmetic, ordered compensated reduction and clearing
are still substantial work; this is not general expression fusion or tuned GEMM.

Pinned stable Rust 1.95/LLVM 22.1.2, fat LTO/one codegen unit, CPU 2, one thread,
shared i5-8250U with variable clocks/unisolated SMT sibling. No own builds, tests
or profilers overlap wall measurements. Construction is separate. Ranges retain
all observations and are not confidence intervals. These fixed-work timings are
not a fresh Stan/JAX comparison, sampler speedup or ESS/sec ranking.

Preparation is not free: construction medians for the elementwise factor are
52.5→58.3 µs (1,000) and 125.3→172.4 µs (10,000); matrix-vector at 10,000 rows
is 959.0→1,274.4 µs, branch at 10,000 elements 247.3→335.9 µs, and scans at
8/128 steps 27.3→36.9 µs / 86.1→106.2 µs. These include model construction,
binding and workspace setup, not compilation. Repeated elementwise/matrix/branch
calls amortize this quickly on the measured shapes; tiny scans may simply pay
more setup for negligible runtime benefit. Do not generalize hot-loop gains to
one-shot evaluation, stack/storage reduction or an optimal execution planner.

### Generated lowering and a statistical consumer

The actual pinned Enzyme suite executes 20 passing tests: native batch source,
primal tensor parity, both branch directions, nonzero/zero-trip scans, matrix
scores/HVPs, nonlinear mixed third derivatives, zero directions, error recovery
and sampler consumers. The compiler still prints its `freeing without malloc`
diagnostics for the nested control-flow fixture. Passing is not a guarantee for
arbitrary compiler programs or an assertion that those diagnostics are fixed.

The profiling example now also measures tensor fixtures: seven alternating
rounds of one million complete value/gradient calls per backend, using the same
nightly compiler for native and generated targets (Rust 1.98 nightly `4429659e4`,
LLVM 22.1.7), CPU 2, no overlapping own workload. Native/generated median ns are
512.488/268.771 for the small matrix, 1,209.477/270.568 for the three-step control
fixture, and 392.266/67.048 for zero-trip control. Paired native/generated ratios
are 1.940 [1.553,2.787], 4.400 [3.843,4.576], and 5.830 [4.352,7.779]. All 42
rows across the six profiling fixtures pass the example's density/g0 checksum
assertion; output gradients are blackboxed and the independent conformance suite
checks all coordinates. This is descriptive backend comparison, **not** measured
before/after improvement of source generation or the large tensor workloads.

`alea/tests/tensor_workflow.rs` now composes a 64-element tensor map/reduction,
lazy conditional and three-step scan into the mean of an existing native Normal
likelihood, with a native Normal prior. Independent equations verify full density
and score; no probability distribution or sampler is reimplemented. Four seeded
chains run 500 native NUTS warmup and 1,000 retained iterations each. All 4,000
retained transitions terminate with U-turns (no retained divergence/depth limit).
For the known posterior N(1/2, variance 1/2), q and q² summaries have R-hat
1.00099/1.00074, bulk ESS 1,614/2,045, tail ESS 1,911/1,941; both moments agree
within six estimated MCSE. This is one correctness workflow, not coverage,
universal convergence, zero warmup failures or an ESS/sec comparison. The stable
CI facade-feature gate includes it explicitly.

The supported fixed-shape/control-flow slice is therefore executable and
numerically tested end to end. Remaining boundaries are deliberate and visible:
no dynamic shapes, generic optimizer/fusion pass, cost-optimal scheduling or tuned
BLAS/GPU lowering. The subsequent increment below adds opt-in native checkpointed
scans. Native batch distributions still lower
through author-provided rules; arbitrary opaque closures cannot be automatically
batched. Native/source performance parity and arbitrary-program compiler AD
remain separate, unfinished qualifications.

Final validation also passes the full default core release suite and doctests,
the facade's distribution-feature modeling/macro/tensor-workflow targets,
all-target core Clippy, the workflow Clippy target, related native batch suites,
40 benchmark protocol tests, source-policy tests and scoped formatting. This is
local validation, not a claim that every remote CI/feature/ISA gate was run.

Local Criterion evidence uses `tensor-final-{forward-1,reverse-1,forward-2}`
(before) and `tensor-hoisted-{forward-1,reverse-1,forward-2}` (after) under
`crates/target/criterion/tensor_tape_{1000,10000}`. These are ignored local results,
not a sealed/shipped evidence capsule. Timings above came from directly executing
the Cargo-built benchmark binary from `crates/`; Cargo invocations above rebuild
when necessary. Source fingerprints hash sorted repository-relative paths plus
NUL, bytes plus NUL, for all `.rs` under core/math `src`, this benchmark, both
package manifests and the workspace manifest/lock:

- Before: `955dc2f40235ed119fe52b25379616fa8c79b733f3ec676794fda8b40e8513f9`.
- After: `581b3547d1eb0cf2dc13897dcd24fb0ecd725b0ae0b45ef54721065704127952`.

There is no committed revision for these dirty-checkout states. Matching source
archives, raw Criterion results and perf captures are retained in the private
local evidence capsule `tensor-ir-2026-11-01.PcGa0s`, not bundled in releases. The
before kernel state was reconstructed from the isolated change and verified
against the **original full benchmark code-bundle fingerprint** above; it is not
an approximation inferred from the timings. Supporting tests/report text can be
newer than the measured kernel state. A fresh before/after replay needs those
archives; current source alone only reproduces the after implementation. These
unsigned local snapshots are not published release artifacts. Do not promote
these measurements into a release-level claim or infer ESS/second from them.

## Checkpointed scans and matrix-vector kernels

The follow-up adds an explicit memory/computation choice without changing the
ordinary `scan` default:

```rust,ignore
let end = builder.checkpointed_scan(initial, rows, &captures, body)?;
```

This uses the same fixed-shape recurrence, error admission and pathwise VJP.
It is not a new modeling language. The later source-replay increment below adds
an explicit first-order source lowering; higher-order compiler tapes remain separate.
[JAX's checkpointing guide](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
explains why scan bodies need explicit residual-storage policies.
[Chen et al., 2016](https://arxiv.org/abs/1604.06174) provides the foundational
segmented recomputation tradeoff. This implementation is not a new checkpointing
algorithm or an optimal Revolve scheduler.

For T trips, B estimated bytes per body workspace and C carry bytes, a segment
length K costs approximately `(K+1)*B + ceil(T/K)*C`. The extra frame preserves
first-iteration derivative activity. The byte-aware planner starts near
`sqrt(T*C/B)`, using integer arithmetic and checked allocation estimates, rather
than treating a tiny carry and a complete body tape as equally expensive. It
enables replay only for explicit checkpointed scans with at least 64 trips,
at least 1 MiB estimated retained body storage and a candidate smaller than half
the retained estimate. These are conservative memory-policy boundaries, **not**
a fitted CPU cost model or a globally optimal discrete schedule. Small tapes
continue to retain frames.

Forward saves only segment-boundary carries. Reverse restores each boundary,
replays that segment and applies the original body VJPs in descending trip order.
The native reverse-body storage is sublinear in trip count for a fixed body;
the enclosing dense buffers and immutable observations remain additional storage.
One extra outer-body forward pass per trip can be necessary for a VJP; nested
checkpointed bodies can compound recomputation. Value-only evaluation does not
replay. A first implementation automatically checkpointed substantial tapes;
seven alternating pairs showed its 10,000-step small-body case about 31% slower
(median before/after ratio 0.765). **That automatic default was rejected.**
Ordinary `scan` retains its original storage choice; users opt into the tradeoff.

The first carry can be nondifferentiable even though later carries depend on a
parameter. Activity has the form `(body_depends_on_carry && carry_active) ||
other_operands_active` and stabilizes after the first trip. Keep a separate
first-trip frame and steady-activity replay frames, including specialized nested
regions. Otherwise `sqrt(0)` can acquire a spurious derivative, or a genuine
singular derivative can be hidden. Owning binding caches, selected-branch domain
checks, gradient admission, reverse accumulation order and failure staging remain
in force. No parameter clipping, reduced trip count or numerical-error suppression
is used.

The new tests compare all outputs/gradient coordinates with retained execution,
an independent nonlinear recurrence/reverse, and central finite differences.
They cover odd segment tails, row/capture/initial adjoints, both lazy branches,
zero/short trips, zero seeds, singular gradients, bad-data rebinding followed by
snapshot drop/recovery, budget rejection and zero hot allocations. For the tested
16-wide nonlinear body, estimated total native workspace drops from 1,508,684 to
100,796 bytes at 257 trips and 6,012,236 to 325,004 bytes at 1,025 trips (15.0× and
18.5×). These are per-execution workspace estimates, checked against retained
allocation bytes; they are not process RSS or a universal memory ratio. Exported
Rust source is identical between the two storage choices. Source AD still owns
its own tape policy and memory bounds.

Matrix-vector profiling found substantial work in compensated products, generic
indexing and reverse-loop control. Private block kernels now use contiguous row
slices/zipped coefficients for the single-column case and whole-block pullbacks,
following the organization of
[Stan Math's matrix reverse callbacks](https://github.com/stan-dev/math/blob/develop/stan/math/rev/fun/multiply.hpp).
Compensated summation, TwoProductFMA product residuals, each coordinate's original
addition order, zero-seed skipping and inactive-side admission are unchanged.
Shared operand storage keeps the original interleaved fallback; zero dimensions
and general matrix products keep the reference path. The block loop bodies stay
out of the shared interpreter to limit unrelated code-size effects. No unsafe
core code, new dependency, BLAS substitution or blanket SIMD claim is involved.

Checkpoint-only interval/carry metadata is boxed only when enabled, following the
Rust Performance Book's advice on
[boxing fields to reduce type size](https://nnethercote.github.io/perf-book/heap-allocations.html).
The initial inline layout enlarged every region slot, including ordinary scans
and branches; the compact layout reduces that added metadata from four words to
one word per slot at the cost of one cold allocation per checkpointed region.
Both layouts' measurements remain in the evidence capsule; this is not free
memory or a claim that heap allocation is generally faster.

Source-identical-harness Callgrind controls (1,000 full calls, 1,000 rows, eight
coefficients, including construction) count 739,183,448 before versus 527,177,684
after instructions in the isolated-matrix candidate. The final retained-path
candidate counts 527,183,295, also a 28.7% reduction. The baseline differs from the earlier
sealed source only by the current benchmark harness and an additive compatibility
shim forwarding `checkpointed_scan` to the unmodified retained `scan`. The
shim exists only in the external baseline reconstruction, never shipping code.
Both binary/source states and all failed/rejected trial records are retained in
the local `tensor-checkpoint-2026-11-02.i9wLiu` evidence capsule. Instrumented time
is not used as wall-speed evidence. The Rust Performance Book's
[profile-first guidance](https://nnethercote.github.io/perf-book/profiling.html)
informed this separation of measured work, wall time and storage.

The final implementation also selects retained versus checkpointed execution
once per scan. Ordinary trips no longer perform checkpoint-index lookups, and
the scan loop bodies stay outside the shared pointwise interpreter. Earlier
inline/compact candidates showed roughly 2–5% elementwise/branch regressions;
those candidates and raw records remain available. The final source restores
these hot controls to approximate parity rather than hiding them behind matrix
speedups. Callgrind's elementwise control is 76,470,452 versus 76,442,136
instructions; this is not evidence of a new elementwise algorithm or a causal
hardware-level explanation for every timing fluctuation.

Final `paired-retained-path` campaign: seven alternating pairs for each of 14
shapes, 196 successful jobs/98 exact-checksum-compatible pairs, no failed or
unstarted work. Raw outputs, unique identities, declared work, binary/source
hashes, compatibility decisions and summaries were independently rederived.
Both public-native evaluation and gradient outputs are consumed. A checksum is
not a proof of every coordinate; independent model-level tests supplement it.
One i5-8250U host, CPU 2 affinity, one thread, generic stable Rust 1.95.0,
LLVM 22.1.2, fat LTO/one codegen unit, shared desktop/variable clocks/unisolated
SMT sibling. No own builds, tests, archives or profiles overlap the timing cohort.
Intervals below are the full observed paired range, not confidence intervals.
Ratios are medians of paired before/after times, not ratios of the two medians.

| Fixed-work objective | Size | Before µs/call | After µs/call | Paired gain median [range] |
| --- | ---: | ---: | ---: | ---: |
| Elementwise tensor | 1,000 | 8.887 | 8.887 | 1.002 [0.998, 1.055] |
| Elementwise tensor | 10,000 | 89.406 | 89.210 | 1.002 [0.959, 1.025] |
| Scalar-tape control | 1,000 | 30.102 | 26.976 | 1.115 [1.103, 1.998] |
| Scalar-tape control | 10,000 | 314.057 | 288.581 | 1.092 [1.054, 1.096] |
| Matrix-vector, eight coefficients | 1,000 | 101.408 | 74.628 | 1.343 [1.295, 1.522] |
| Matrix-vector, eight coefficients | 10,000 | 974.905 | 745.341 | 1.319 [1.231, 1.395] |
| Lazy branch | 1,000 | 9.571 | 9.595 | 1.000 [0.992, 1.011] |
| Lazy branch | 10,000 | 107.646 | 107.923 | 0.996 [0.954, 1.006] |
| Retained scan | 8 trips | 1.992 | 1.925 | 1.027 [0.932, 1.154] |
| Retained scan | 128 trips | 29.660 | 27.258 | 1.101 [0.929, 1.277] |
| Retained scan | 1,025 trips | 221.581 | 210.124 | 1.042 [0.952, 1.088] |
| Retained scan | 10,000 trips | 2,598.199 | 2,478.821 | 1.051 [0.938, 1.124] |
| Requested checkpoint, small-tape fallback | 1,025 trips | 215.740 | 200.028 | 1.074 [1.037, 1.164] |
| Checkpointed scan versus old retained scan | 10,000 trips | 2,526.243 | 3,087.060 | 0.818 [0.619, 0.851] |

The scalar implementation was not changed; its descriptive improvement is not
claimed as an algorithmic speedup or an explanation of the preceding increment's
scalar-control regret. Matrix-vector wins every final pair. Ordinary scans have
modest median gains, not an all-pair guarantee. Branch medians include a 0.4%
slowdown at 10,000 elements, with both directions represented in the range.
Construction is separate: the 10,000-step checkpointed case falls from 8.092 ms
to 0.268 ms, whereas ordinary 10,000-step construction grows from 8.496 to
8.981 ms (about 5.7%). Smaller ordinary scans have smaller/mixed cold changes.
These costs and retained-frame metadata are not erased by the zero-allocation
hot-evaluation result. The 10,000-step checkpointed evaluation is about 22%
slower than the old retained path in paired latency terms; choose it for memory,
not a promised CPU win. Its small scalar-body timing fixture is distinct from the
16-wide nonlinear workspace-budget fixture above.

Final local gates pass the full default core release suite/doctests, all-target
core Clippy, 16 tensor/scan tests, the four-chain native-law tensor workflow,
20 executed Enzyme conformance tests, 41 benchmark protocol tests, source-policy
tests, API extraction/compatibility adjudication and scoped formatting. The
existing nested-control Enzyme compiler still emits `freeing without malloc`
diagnostics while its executed tests pass; this does not establish arbitrary
program/compiler correctness. No remote CI, all-feature/ISA matrix, full tensor
statistical calibration or new inference/ESS/sec comparison was performed.

At this increment, general pointwise fusion, tuned multi-column GEMM/BLAS, GPU
lowering, automatic global CPU scheduling and source-AD checkpoint placement
remained unimplemented. The follow-up below addresses pointwise scheduling and
first-order source replay, not the other compiler/backend gaps.
[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) is a useful reference
for dependency/index-map-aware fusion; it is not an Alea backend. This increment
does not close those gaps or establish Stan/JAX inference or ESS/sec parity.

## Implementation lessons

1. A semantic signature is not a faster execution plan. Preserve blocks and
   regions in the representation, then measure dispatch, arithmetic and storage
   separately. The unchanged native regression kernels were already fused; a
   scalar-versus-block deterministic benchmark cannot establish a regression or
   Stan sampling speedup. The [native-model case study](native-model-performance.md)
   remains the separate record for those inference workloads.
2. Data rebinding changes a mathematical function. A mutable setter on a borrowed
   target can make a cached position/value/gradient inconsistent. Immutable
   snapshots and consuming model replacement make that boundary explicit.
3. Correct reverse propagation needs both reachability and derivative activity.
   An active branch can require a data value without differentiating it; a scan
   carry can become differentiable only after its first iteration. Treating every
   region input as a parameter creates spurious singular derivatives.
4. Empty arrays are not error sentinels. Preserve failure status independently,
   including across chosen branches and loop bodies. Omit statically empty pure
   loops, not the upstream domain checks that make their dependencies valid.
5. Source export is a distinct capability from compiler AD. Successful Rust
   emission, matching gradients on one backend and metadata declarations cannot
   establish nested-AD support. Keep actual compilation/execution tests and a
   truthful unsupported-backend boundary when the compiler fails.
6. Automatic reporting must apply an explicit mathematical decomposition, never
   invent one from names or tensor axes. Conditional rows, global study IDs and
   marginal leave-out units solve different problems.

## Dependency-safe fusion and first-order source replay

The November 3 follow-up separates two mechanisms which must not be conflated.

### Fusion is an execution schedule, not a new density

`model/tensor/fusion.rs::schedule` identifies contiguous, reachable pointwise
DAGs with equal flat output extents and an optional terminal sum. Unary and
binary nodes can share operands; external operands retain their static broadcast
maps. An internal producer must be consumed at the same element index. Gather,
reshape, matrix operations, constants and structured-region boundaries are
barriers. Child regions compile their own schedules. This is not arbitrary
callback analysis, distribution recognition, matrix-chain optimization or GPU/JIT.

The native executor uses cache-sized blocks of 256 elements, restricted to at
least four maps and 4,096 elements after the unconditional policy regressed the
small controls. These are a measured conservative envelope on one host, not a
universal cost model. Generated Rust uses one straight-line element loop per
eligible group, independently of that native threshold. Both check every intermediate,
including an overflowing exponential followed by its reciprocal: a finite final
answer never legitimizes an invalid intermediate. Terminal sums retain original
element order and compensation. There is no floating-point reassociation.

Dense primal storage and the original reverse-node/operand accumulation order
are retained. In particular, tiling a broadcast scalar's reverse accumulation
would change its addition order. This increment does not fuse that reverse
reduction, perform lifetime compaction, or claim zero intermediates. A smaller
number of loops is not by itself a performance result.

### Source checkpoints require an explicit reverse program

The pinned Rust/Enzyme frontend has no supported rematerialization directive or
general custom-VJP registry. See the
[Rust frontend limitations](https://enzyme.mit.edu/rust/limitations.html) and
[compiler AD flags](https://rustc-dev-guide.rust-lang.org/autodiff/flags.html).
Neither outlining a function nor adding `inline(never)` proves that a reverse
tape has bounded storage. Unsafe type-analysis relaxation is not a remedy.

`tensor/pullback.rs` emits an explicit VJP for the tensor primitives. Its scan
reverse stores boundary carries and replays segments of approximately sqrt(T)
trips, retaining their input carries. First-trip and steady-state derivative
activity are separate. Selected branches, repeated capture/row adjoints,
zero trips, alias storage and independent failure status remain explicit.
Nested regions recurse through the same lowering. Source generation freezes
the binding; rebinding still requires regeneration.

`assembly/source/checkpoint.rs` composes these tensor VJPs with the scalar graph.
Other factors use their existing scalar source plus compiler reverse; no
probability laws or analytic distribution scores are reimplemented. Parameter
charts use the existing native `Transform::pullback` or
`pullback_composition`, so Jacobians remain present exactly once for densities
and absent for objectives. Every local factor score is admitted before graph
propagation, even if its eventual scalar multiplier is zero.

The generated `SOURCE_CHECKPOINTED` constant reports whether this explicit
first-order path was selected. Its float-arena extent is checked, but is not a
whole-process memory bound. Original input/output storage and dense active tensor
adjoints remain; opaque source factors can retain compiler-owned tapes. Replay
also costs computation: the current model gradient can execute five body-primal
passes per trip (model admission, VJP admission, boundary preparation, segment
replay and local VJP admission). Nested replay can multiply that cost. This is a
memory/work tradeoff, not a claimed inference speedup or optimal scheduler.

### The higher-order compiler boundary is still open

Actual forward-over-explicit-reverse compilation failed on the pinned compiler:
first with an unresolved `enzyme_primal_return` global for nested reverse
helpers, then with an `Illegal updateAnalysis` pointer/integer type conflict for
the rematerialized program. Reusing native chart pullbacks avoids duplicated
mathematics, but does not make that compiler failure disappear.

Consequently **HVPs and third derivatives retain the original qualified compiler
reverse**. They are not source-checkpointed. Their values remain covered by the
independent quadratic/recurrence fixtures, including zero directions and both
branches. Do not describe general higher-order source checkpointing as finished.
The existing nested-control `freeing without malloc` compiler diagnostics also
remain visible; passing selected tests is not arbitrary-program qualification.

The source fixtures additionally exercise 257-trip replay, a native Normal prior
composed with a squared tensor factor and positive chart, objective Jacobian
exclusion, matrix scores, empty selected errors and recovery. Pure 257-trip
first-order execution allocates no heap memory. The retained compiler fixture
also allocates no heap memory on this compiler, so that observation alone does
not establish a measured peak-memory improvement.

### Source adjoint storage follow-up

The explicit first-order source pullback now assigns compact ranges only to
derivative-active nodes. Immutable bindings, constants and inactive intermediate
results have no adjoint range. Reshape aliases share their
producer's range. Conditional argument and scan row/capture/initial propagation
skip inactive destinations without changing the order of active additions.
Primal storage and every selected arithmetic/domain check are unchanged; a
constant result still evaluates its primal and validates its seed.

An exact layout regression covers zero, one, 257 and 10,000 immutable elements.
For `gain * sum(data)` with a reshape alias, the isolated tensor pullback's
scratch requirement is five doubles regardless of data length, rather than
`N + 6` doubles with the former dense adjoint arena. At N=10,000 this is 40
versus 80,048 bytes of declared scratch, excluding inputs, embedded data, caller
buffers and compiler temporaries. This is a checked storage calculation, **not
a measured peak-RSS or latency speedup**. Dense active intermediates still need
adjoints, and the enclosing source model takes the maximum of its other scratch
requirements, so this ratio must not be extrapolated to whole models.

Executed compiler fixtures check alias gradients against finite differences,
`sqrt(0)` on immutable data without spurious singular derivatives, entirely
parameter-independent checkpoint results, overflow rejection and recovery.
An active-row recurrence checks every coordinate analytically and by central
finite differences, including repeated capture aliases following an inactive
capture in the body argument list.
The existing long-scan, first/steady activity, matrix, mixed-prior/chart and
higher-order tests remain the independent composition controls. The higher-order
compiler boundary and five-pass replay cost above are not resolved by compaction.

### Fixed-work performance qualification

The final boxed-schedule candidate was compared with the original `07c99a2`
checkpoint tree (also exactly reproduced by the scoped reconstruction ending
at `bce2d91`). Both builds use the identical current `kernel_audit.rs` harness;
the baseline overlay changes only that harness, not a distribution, derivative
or scan implementation. The new shared-DAG case contains eighteen binary maps.
All fourteen previous checkpoint controls remain.

Seven alternating-order pairs per shape produced 238 successful jobs and
119 exact-checksum-compatible pairs, with zero failed or unstarted jobs. Complete
coordinate/error tests are separate from checksum-based timing admission.
CPU affinity is logical CPU 2 on a shared x86-64 i5-8250U, one thread, Rust
1.95.0/LLVM 22.1.2, generic release/fat LTO/one codegen unit. There were no own
builds, tests, profiles or archive jobs during wall cohorts; clocks/SMT/other
desktop activity were not isolated. Ranges below are observed paired ranges,
not confidence intervals. Gain is before/after per pair, so its median is not
necessarily the ratio of the two marginal medians.

| Case | Elements/trips | Before µs/eval | After µs/eval | Paired gain median [range] |
|---|---:|---:|---:|---:|
| tensor | 1,000 | 6.289 | 6.251 | 1.004 [0.989, 1.023] |
| tensor | 10,000 | 65.165 | 65.149 | 1.001 [0.989, 1.008] |
| tensor_scalar | 1,000 | 22.118 | 20.339 | 1.094 [1.061, 1.133] |
| tensor_scalar | 10,000 | 259.515 | 242.215 | 1.072 [1.049, 1.109] |
| tensor_matmul | 1,000 | 47.114 | 47.336 | 0.994 [0.969, 1.001] |
| tensor_matmul | 10,000 | 463.241 | 471.913 | 0.983 [0.914, 1.010] |
| tensor_cond | 1,000 | 6.932 | 6.958 | 0.994 [0.973, 1.002] |
| tensor_cond | 10,000 | 77.155 | 77.608 | 0.999 [0.987, 1.004] |
| tensor_scan | 8 | 1.375 | 1.416 | 0.974 [0.941, 0.988] |
| tensor_scan | 128 | 18.664 | 18.861 | 0.992 [0.950, 1.022] |
| tensor_scan | 1,025 | 147.537 | 148.810 | 0.992 [0.968, 1.005] |
| tensor_scan | 10,000 | 2173.226 | 2312.253 | 0.960 [0.915, 0.965] |
| tensor_scan_checkpoint | 1,025 | 149.029 | 150.922 | 0.997 [0.965, 1.016] |
| tensor_scan_checkpoint | 10,000 | 2226.093 | 2223.557 | 1.009 [0.996, 1.033] |
| tensor_dag | 1,000 | 37.685 | 38.166 | 0.986 [0.969, 0.996] |
| tensor_dag | 10,000 | 427.869 | 402.571 | 1.078 [1.039, 1.101] |
| tensor_dag | 100,000 | 8455.446 | 6742.074 | 1.259 [1.230, 1.274] |

Large-DAG throughput rose from about 118 to 148 fused value/gradient evaluations
per second at 100,000 elements. At 10,000 it rose from about 2,337 to 2,484.
This is an approximately 26% and 8% paired-median gain, not a Stan/JAX or ESS/sec
result. Construction is separate: the 100,000-element DAG increased from
1.599 to 1.735 ms, and the 10,000-trip retained scan from 6.621 to 6.680 ms.
Dense workspace did not shrink.

Do not omit the losses: the retained 10,000-trip scan's paired median is 0.960
(about 4.2% greater latency), the eight-trip scan 0.974, the small DAG 0.986,
and the large matvec 0.983. The unchanged scalar controls improve in this cohort,
but no scalar algorithm changed; code layout/host variation is not an attributed
optimization. This is not a regression-free or universally optimal planner.

Callgrind on ten 100,000-element evaluations, including construction and the
harness warm evaluation, records instructions increasing from 348,017,558 to
361,487,285, while simulated D1 misses fall from 23,326,689 to 16,529,745.
Last-level misses are essentially unchanged. This supports a cache-locality
explanation rather than fewer arithmetic operations; simulated misses are not
hardware-counter measurements or proof of a particular real cache's behavior.
Instrumented timings are excluded from the wall cohort.

The first unconditional candidate is retained as a rejected experiment: its
large DAG won about 27%, but small scan/control latency regressed roughly 5–12%.
Restricting native groups recovered short pointwise controls; selecting the
interpreter once outside scan trips and boxing optional schedule metadata
reduced, but did not eliminate, unrelated regressions. The intermediate
selective, dispatch and plain-interpreter cohorts are retained too. Do not
choose their stronger individual gains over the final whole-cohort numbers.

Evidence capsule: `fusion-source-2026-11-03.t8QwOJ` under the maintainer's external
`alea-inference-evidence` directory. It retains saved executables, manifests,
all raw jobs/pair decisions/summaries, compiler failures, profiles and source
archives. Final wall evidence is `paired-boxed/`, executable SHA-256
`b01efb1b459c47f03b42e55ed1e4a9d1b71331ab4a85f3d5642aaaf963e25aad`;
before is `200d272b000a2b03fbce9a90c144a408baefd565197f9f127f90bfeb26facf5d`.
The measured candidate archive is
`1252ed4fc185c6cdc131a81b37300df874a2c52b2a0c1aa89589cdfaef01f079`;
baseline-plus-identical-harness archive is
`f9cbeb81e41736aea738997e49a704b724df5306ac9eadfc501ed00be7208fbc`.
These are local evidence identifiers, not a claim of independently published
or freshly rerun Stan inference results.

The engineering lesson is to qualify the abstraction boundary, not just the
winning kernel. Shared optional metadata, iterator bounds, dispatch frequency,
body activity and source-compiler tape ownership can outweigh removed tensor
passes. Preserve arithmetic and failure admission, measure controls at every
candidate, and report remaining losses rather than call a partial speedup
general engine parity.
