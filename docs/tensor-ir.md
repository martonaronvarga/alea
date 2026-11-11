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
required/allowed bytes. `BoundProgram::check_workspace_limit(bytes)` permits the
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
use alea::model::{
    assembly::ModelBuilder,
    tensor::{BinaryOp, Shape},
};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut b = ModelBuilder::new().region();
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
`ModelBuilder::call_region` connects whole model values to a region's declared
inputs. It checks shape and ownership before adding the call. Its result can
feed multiple consumers, which accumulate their cotangents before one reverse
call. Matrix products, reshapes and scalar reductions are available directly
on the model builder. A vector chart can supply a whole value through
`ModelBuilder::vector_value`.

```rust
use alea::model::{
    ConstrainedModel,
    assembly::ModelBuilder,
    tensor::Shape,
};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut model = ModelBuilder::new();
let beta = model.real_vector("beta", &["a", "b"])?;
let beta = model.vector_value(&beta, Shape::new(vec![2, 1])?)?;
let design = model.constant_block(Shape::new(vec![2, 2])?, vec![1., 2., 3., 4.])?;
let eta = model.matmul(design, beta)?;
let total = model.sum_value(eta)?;
model.factor(total)?;
let model = model.compile()?;
let mut gradient = [0.; 2];
assert_eq!(model.logp_grad(&[2., 3.], &mut gradient)?, 26.);
assert_eq!(gradient, [4., 6.]);
# Ok(())
# }
```

Use `BlockRules` to register a custom forward rule and an optional native VJP.
The descriptor carries immutable configuration, and each execution frame owns
its typed workspace. Region callbacks can read real, Boolean, count and index
arguments in their original types. VJPs write one contribution per real input
element in declared argument order, including repeated inputs. Native-only
callbacks cannot be checkpoint-replayed or exported as Rust source.

Whole-value model calls currently execute natively. Existing scalar-result
source models use `ModelBuilder::apply_region`. Native `NormalBatch` and
logit composition with whole tensor predictors remains a separate cutover.

### Numerical admission is part of the execution contract

Only the selected conditional branch executes. Conversely, every reachable
forward intermediate must be finite: `exp(-exp(x))` is rejected when its inner
exponential overflows, even though unchecked arithmetic could finish at zero.
A zero reverse seed does not bypass this forward admission. Failed evaluation
and pullbacks preserve caller outputs and the workspace remains reusable.

Derivative admission is deliberately not uniform across interfaces. Native
built-in tensor pullbacks skip zero seeds and inactive data operands; for example,
`sqrt(0)` can have a finite value and a zero-seeded VJP, but its active unit-seeded
VJP fails. Opaque supplied local scores are admitted independently of downstream
weights, including zero. Retained and checkpoint-replayed scans preserve those
distinctions. An unselected scalar assembly plan also admits unattached scalar
nodes; explicit statistical selection prunes them. This differs from tensor
root reachability and is not silently changed by these tests.

The existing `tensor_ir`, `tensor_checkpoint`, `factor_roles`, `model_source` and
private registration tests cover these boundaries, error/unwind recovery, repeated
gather accumulation and tolerant VJP seed linearity with independent derivatives.
Native first order remains usable when source is unavailable. It does not confer
a generic JVP, HVP or third-derivative capability; the native target's unsupported
curvature traits have compile-fail tests. No native JVP/VJP dot-product identity
is claimed where there is no JVP implementation. These checks do not qualify
generated Enzyme derivatives or establish a performance improvement.

Broadcast maps are specialized at construction. If output element j reads input
element m(j), its pullback accumulates `input_bar[m(j)] += output_bar[j] * local_score`.
Regular broadcasts retain O(rank) index descriptors, not one integer per output
element. Native forward/reverse execution and fusion share the same descriptor;
the primal and explicit-pullback source emitters share its index renderer.
Explicit irregular `gather` still retains the caller's indices.
Matrix products use the usual `A_bar += C_bar B^T` and `B_bar += A^T C_bar`
pullbacks, without materializing a Jacobian. The default matrix execution is an
ordered compensated reference kernel. Explicit accelerated kernels are available
as described below; this is not batched GEMM or a GPU backend.

### Compact broadcast indexing: contracts and evidence

For each non-unit input axis, the private descriptor retains its extent `n`,
row-major input stride `s`, and corresponding output stride `d`. An output flat
index maps to `sum(((j / d) % n) * s)`. Shape/broadcast admission precedes map
construction; stride multiplication is checked. Empty output domains need no
strides or addresses, including when a suffix product after a zero axis would
otherwise overflow. A distinct empty map prevents contiguous fast paths from
addressing nonexistent inputs or adjoint ranges. Leading unit dimensions with
no repeated elements canonicalize to the identity path.

Native kernels split work at the two operands' innermost run boundaries, unrank
only at those boundaries, and advance constant or contiguous input indices in
the inner loop. This avoids per-element rank-dependent division without a
heap-allocated cursor. Fusion tiles can start/end inside runs. Reverse traversal
keeps the original element and operand accumulation order: a repeated read map
is not an injective write map and does not justify unordered scatter stores.
Generated code uses compact, shape-specialized integer expressions instead of
index-array literals. Frozen data literals and primal/adjoint storage remain
separate costs and can still grow with problem size.

This follows the distinction between indexing and storage in
[Futhark's memory work](https://futhark-lang.org/publications/sc22-mem.pdf) and
the explicit reindexer obligations in
[ATL's verified lowering](https://adam.chlipala.net/papers/AtlPLDI24/AtlPLDI24.pdf).
Neither is a formal proof of Alea or of floating-point AD. No tensor-layout,
copy-elision, speculative branch, reassociation or checkpoint-policy change is
included here.

Tests use an independent dense coordinate oracle, arbitrary-rank property cases,
partial runs, exact-bit explicit-gather comparisons with cancellation-sensitive
seeds, finite differences, empty axes, immutable rebinding and staged failure
recovery. Downstream compiler conformance includes a rank-three shared-input
quartic with independent value/score/Hessian/third derivatives on ordinary and
explicit first-order checkpoint source routes. Higher derivatives still use the
qualified compiler reverse, not differentiation of the checkpointed pullback.

The post-cutover admission inventory is below. Native integration names refer to
`crates/alea-core/tests/tensor_ir.rs`; compiler cases live in the existing
`tools/enzyme-conformance` fixtures and tests.

| Case | Native evidence | Compiled-source evidence |
| --- | --- | --- |
| Rank-two rows/columns, rank-three coordinates, partial runs | `executable_broadcast_grid_matches_dense_coordinates_across_contexts_and_immutable_rebinds`; exact-bit explicit-gather controls | `tensor_broadcast` and `tensor_broadcast_tail`, ordinary and checkpoint variants; independent dense value and polynomial score/Hessian/third/HVP answers |
| Empty axes | Executable grid includes zero rows and zero width; `empty_broadcasts_do_not_address_inputs_or_adjoint_ranges` | `tensor_broadcast_empty` and checkpoint variant, including zero HVP directions |
| Invalid shapes, overflow, owner and index checks | `admission_checks_owners_shapes_empty_tensors_and_dead_operations`; empty scan metadata overflow test | Source generation consumes the same admitted plan; runtime nonfinite inputs remain typed errors |
| Repeated IDs, reshape aliases, inactive operands | Executable grid; rank-three gather reverse test; specialized binary scalar-bit controls | Shared-input quartic fixtures on both source routes; compiler tensor admission tests for inactive scores |
| Zero seeds, lazy branches, selected intermediate errors | Repeated-gather seed linearity; selected overflow test; executable lazy-broadcast test covers empty and zero-weight results | `tensor_control`, `tensor_source`, and compiler tensor admission tests retain selected-branch failures and zero-seed behavior |
| Failure/recovery, immutable rebind, independent contexts | Executable grid checks rejected shapes/nonfinite data, new binding identities, unchanged staged scores and old-binding recovery; checkpoint replay recovery tests | Generated tail/empty tests reject nonfinite inputs then recover; existing source rebinding/admission controls remain |

Whole-value `call_region` executable edges in the new grid are native-only and
explicitly reject source emission. Compiler fixtures exercise the existing
scalar-result adapter, including ordinary and explicit checkpoint source routes;
they do not establish generated derivatives for whole-value model edges. No
writable buffer reuse or copy-elision behavior changes in this grid.

`tensor_tape` retains `tensor_broadcast/{128,4096,65536}` width-eight cases and
adds rows 1/3/64/1024/65536 crossed with widths 1/3/16. Each has an explicit
repeated-gather control under `tensor_broadcast_gather/{rows}x{width}`. Existing
scalar/block/direct and matrix controls remain. The phases separate `plan_bind`,
`primal_source`, `frozen_source`, `workspace`, `first_value_vjp`, `forward`, and
steady `value_vjp`. The first-call Criterion case uses a fresh workspace per
iteration: setup allocation is excluded, but workspace destruction is included.
The existing fixed-work receipt additionally distinguishes one first and four
steady target calls; optional phase timing still requires a compute allocation.
Run these
through `cargo bench -p alea-core --bench tensor_tape -- tensor_broadcast` from
the crate workspace, on an ELTE HPC compute allocation. Compare saved before/after
sources on the same host, retaining nonbroadcast/scan controls. Storage/source
complexity is established by code and allocation tests; the remote native latency
receipt below does not establish inference speedups.

Cold `ALEA_BROADCAST_STORAGE_V1` records are retained by the existing
`three_engine.py cargo-report` reader as `broadcast_storage`. They distinguish
complete plan heap ownership (including graph/scheduler metadata), runtime-input
source bytes, frozen-row source bytes, frozen f64 payload bytes, checked workspace
budget, measured workspace ownership and first/steady allocation counts/bytes.
Source lengths include scaffolding; their difference is not an isolated literal
text measurement. The private `map_storage_and_source_depend_on_rank_not_output_cardinality`
test separately emits `ALEA_BROADCAST_MAP_V1` allocation and integer-expression
length records without allocating tensor elements. Those map-only counts must
not be substituted for total plan or workspace costs. None bounds compiler tape,
provider scratch or process RSS.

Untimed allocation-counter observations on x86_64 (bytes, not timings):

| Fixture | Plan heap ownership | Runtime-input source | Workspace heap ownership | First / steady allocations |
| --- | ---: | ---: | ---: | ---: |
| Compact 1x1 | 1,494 | 1,144 | 581 | 0 / 0 |
| Compact 65536x16 | 1,574 | 1,113 | 35,128,181 | 0 / 0 |
| Explicit gather 65536x16 | 16,778,776 | 10,702,693 | 68,158,495 | 0 / 0 |

These are scoped fixture ownership counts, not peak allocation or matched
before/after results. At width 16, frozen row payload is 128 bytes; the compact
frozen-source wrapper is 1,194 bytes, including scaffolding and literals.

### Remote indexing qualification (November 11, 2026)

ELTE job 2778970 completed on exclusive cn18-13 (Xeon Gold 6240, KVM guest),
with one pinned evaluation CPU and all provider/thread budgets set to one.
Both trees used stable Rust 1.95.0, `-Cembed-bitcode=yes -Ctarget-cpu=native`,
no default features, the Rust provider and the same expanded harness. The old
harness only substitutes its required `CompiledModel` type name for `NativeModel`.
The reference is the preserved I0.4 candidate source; the current tree includes
subsequent ownership/registration work. **Both already contain compact maps**:
these timings compare revisions and cannot attribute a gain to map compaction.

Both source manifests were verified unchanged after execution. Native tests
passed (reference 26, current 28), map property/storage tests passed on both,
and all 261 untimed benchmark cases passed before timing. Seven alternating
reference/current pairs yielded 3,654 Criterion observations (20 samples each),
1,176 first/steady phase records, 2,940 counted target calls and 504 storage
records. No run failed. The existing report reader replayed all records without
relaxing source/query/data identities. The isolated map grid retains at most
40 heap bytes per nontrivial operand and at most 10 bytes of index expression;
full plans, frozen data and workspaces remain the larger separate costs above.
All recorded steady target allocation counts were zero; ownership/source counts
were identical across the two trees and seven pairs.

Steady fused VJP times below are medians of the seven per-run Criterion medians.
Ratios are medians of matched reference/current ratios; their min/max show
observed pair dispersion, **not confidence intervals**. Each original Criterion
interval/sample set is retained. Warmup was 0.1 s and requested measurement time
0.3 s per case; Criterion extended expensive cases when needed. Outlier and
insufficient-time warnings are preserved. One host and one source pair do not
support a universal performance claim.

| Compact shape | Reference VJP µs | Current VJP µs | Paired reference/current median [min, max] |
| --- | ---: | ---: | --- |
| 128×8 | 16.392 | 16.603 | 0.989 [0.985, 1.019] |
| 4096×8 | 549.764 | 539.890 | 1.022 [1.015, 1.028] |
| 65536×8 | 9234.872 | 9117.819 | 1.005 [0.968, 1.043] |
| 1×1 | 0.152 | 0.148 | 1.025 [1.016, 1.156] |
| 1×3 | 0.237 | 0.232 | 1.025 [1.021, 1.034] |
| 1×16 | 0.365 | 0.349 | 1.015 [0.937, 1.067] |
| 3×1 | 0.236 | 0.233 | 1.013 [1.006, 1.032] |
| 3×3 | 0.454 | 0.438 | 1.037 [1.017, 1.093] |
| 3×16 | 0.836 | 0.805 | 1.043 [1.032, 1.071] |
| 64×1 | 0.860 | 0.779 | 1.108 [1.072, 1.115] |
| 64×3 | 5.114 | 4.916 | 1.043 [1.039, 1.048] |
| 64×16 | 13.539 | 12.814 | 1.057 [1.041, 1.061] |
| 1024×1 | 11.343 | 10.255 | 1.098 [1.060, 1.173] |
| 1024×3 | 81.485 | 78.433 | 1.036 [1.012, 1.046] |
| 1024×16 | 220.597 | 205.852 | 1.059 [1.055, 1.114] |
| 65536×1 | 1071.552 | 997.842 | 1.068 [1.049, 1.084] |
| 65536×3 | 5261.891 | 5099.288 | 1.033 [1.027, 1.040] |
| 65536×16 | 17751.031 | 16964.764 | 1.039 [1.037, 1.054] |

Cold/fresh-workspace controls (µs, seven-run medians):

| Shape | Phase | Reference | Current |
| --- | --- | ---: | ---: |
| 1×1 | plan/bind | 0.964 | 0.911 |
| 1×1 | runtime-input source emission | 1.836 | 1.929 |
| 1×1 | frozen-row source emission | 1.931 | 2.011 |
| 1×1 | workspace construction | 0.414 | 0.385 |
| 1×1 | fresh-workspace VJP (setup excluded, destruction included) | 0.278 | 0.276 |
| 65536×16 | plan/bind | 1.100 | 1.044 |
| 65536×16 | runtime-input source emission | 2.205 | 2.258 |
| 65536×16 | frozen-row source emission | 3.471 | 3.506 |
| 65536×16 | workspace construction | 5,679.949 | 5,416.313 |
| 65536×16 | fresh-workspace VJP (setup excluded, destruction included) | 17,690.071 | 17,190.857 |

The separate fixed-work first call includes poisoning, admission and output
copies; it does not isolate pure kernel latency. For compact 65536×16 its current
median was 17.600 ms [17.436, 18.091] versus 16.946 ms [16.884, 17.228] per steady
call across seven observations. For 1×1 these were 0.567 µs [0.498, 0.751] and
0.424 µs [0.359, 0.539], illustrating timer/harness sensitivity at tiny sizes.

Retained counterexamples and controls: 1×16 forward increased from 0.112 to
0.122 µs (paired ratio 0.920 [0.900, 0.938]); compensated 8×8 GEMM increased
3.274 to 3.515 µs (0.932 [0.926, 0.936]). Some source-emission cases also regressed.
The 10k assembled tensor control was effectively unchanged, 67.113 to 67.363 µs
(0.997 [0.991, 1.014]); the 1k scalar tape improved 31.116 to 26.966 µs
(1.154 [1.150, 1.155]). Explicit-gather 65536×16 VJP was 32.100 to 31.379 ms
(1.015 [1.002, 1.029]), versus current compact 16.965 ms. This gather comparison
also changes intermediate storage and graph structure; it is a control, not an
isolated index-arithmetic speedup or a before/after map-cutover measurement.

Single build observations were 107.54 s reference and 87.11 s current, with
630,984/634,432 KiB maximum RSS. Builds ran in that order and share dependency
caches; these are not qualified cold-build improvements. Fourteen complete
benchmark-process RSS readings ranged 265,288–268,632 KiB and include fixture
construction/source generation and all shapes, not per-target memory. Compiler
AD/provider scratch remains unbounded here; no generated derivatives were changed
or measured. No scan timing, inference/ESS or other-host qualification follows.

Raw evidence is preserved on ELTE at `/users/usumusu/alea-i24-mttkQlKa`:
`receipt.tar.gz` SHA256
`4e7c63b1161fffc2b10c90353a885b24ce9a13d345506459cf0f6c3e404248f4`
and `build-receipt.tar.gz` SHA256
`8151f9c1ec1e2d3a6b6cf5196091349ce3e3bb01ad5c6c88444c4dfaa84065e9`.
They contain raw samples/estimates, logs, host/flag/thread metadata, preflight
results and both source manifests. Existing `three_engine.py cargo-report --output
<extracted-results>` replays the receipt; use matched pair IDs for comparisons,
not its sequential feature-cell ratios as a paired uncertainty estimate.


### Explicit Faer and BLAS matrix kernels

Enable `alea/faer` or `alea/openblas` (direct core users: `alea-core/faer` or
`alea-core/openblas`), then select a kernel on the matrix operation:

```rust,ignore
use alea::model::tensor::MatrixKernel;
let product = builder.matmul_with_kernel(a, b, MatrixKernel::Faer)?;
// MatrixKernel::Blas selects LP64 CBLAS instead.
```

The choice is retained with the fixed-shape matrix node. Forward execution uses
GEMM and reverse execution uses the two transposed GEMMs above, accumulating into
existing active adjoints without transpose copies. Faer uses `Par::Seq`; BLAS
uses the provider's thread policy. Configure `OPENBLAS_NUM_THREADS=1` when running
parallel chains unless a deliberate global CPU budget says otherwise. Neither
adapter changes a process-global thread setting. Exact row-major lengths and
LP64 dimension conversions are checked before foreign calls. Unsupported BLAS
dimensions decline to the reference implementation before writes.

This policy **explicitly allows different floating-point reduction orders** in
values and derivatives. It does not promise compensated accuracy, bitwise replay
across providers/CPUs, or identical NUTS trajectories. Enabling a Cargo feature
alone never changes `matmul`'s default. Shared-storage reverse operands and any
zero seed retain the ordered scalar pullback, preserving interleaved aliases and
skipped derivatives. Empty inner dimensions produce zero without calling BLAS;
ordinary nonfinite admission, typed failures and caller-output staging remain.
The safe adapter lives in `alea-math::gemm`; core remains free of unsafe code.

Workspace budgets count Alea-owned storage, **not** provider packing buffers,
thread-local allocations or thread stacks. Value and explicit-first source export
retain the selected Faer/BLAS provider, including resolved `Auto`; they do not
substitute compensated primal arithmetic. First-order lowering uses the existing
checked GEMM adapter for the forward product and both accumulating transposed
operand pullbacks. Aliases and zero seeds retain the ordered scalar pullback,
matching native admission and accumulation behavior. Every selected provider
output is finite-checked before a later expression can mask overflow.

Provider factors require explicit first-order composition. This selection is
factor-local: an ordinary companion tensor keeps compiler reverse plus checked
score admission. `SourceCapabilities::unqualified_curvature` identifies active
provider factors; HVP/third `SourceRequest`s return `UnqualifiedDerivative` even
when `OriginalCompilerReverse` is requested. Use an explicit value/first request,
not the highest-order convenience exporter. This does not qualify compiler AD
through provider calls or differentiation of the explicit pullback.

```rust,ignore
use alea::model::assembly::{CurvatureRoute, DerivativeOrder, SourceMode, SourceRequest};
let source = alea::autodiff::codegen::source(&model, SourceRequest {
    mode: SourceMode::Density,
    order: DerivativeOrder::First,
    curvature: CurvatureRoute::Unavailable,
})?;
```

These rules apply to the existing scalar-result region adapter, including nested
branches and nonempty scans. Whole-value model edges remain native-only.
Unreachable nodes and zero-trip bodies require no lowering. Unsupported nonempty
LP64 dimensions reject with `MissingPrimal` before emission/workspace allocation.
Default compensated source remains supported. `BoundProgram::rust_function`
returns `Result<String, AssemblyError>`; no reference-export policy is implied.

The original bounded provider probes in the existing Enzyme `tensor_matrix` test
use `try_gemm` directly, independently of tensor source lowering. On
`nightly-2026-06-23` (rustc `4429659e4`, Enzyme 22), release/fat LTO and embedded
bitcode, a 2×3 by 3×1 product has these outcomes:

| Provider | Primal execution | Compiler reverse |
| --- | --- | --- |
| Faer 0.24.0 | Independent answers pass for four input scales | Cargo exits 101 after rustc SIGSEGV in Enzyme `TypeTree::allFloat` / `TypeAnalyzer::visitAtomicRMWInst`; no gradient executable/result |
| LP64 OpenBLAS 0.3.31 | Independent answers pass; executable loads `libopenblas.so.0` and calls `cblas_dgemm` | Cargo exits 101 at link: undefined `cblas_dlascl`, referenced by the generated `blas_primal` derivative; no numerical gradient result |

The inputs are `A=[[1,2,-1],[0.5,-2,3]]`, `b=[2,-1,0.5]` and the scalar
result is `(Ab)[0]+2*(Ab)[1]=8.5`. Both operands are differentiated; independent
score checks are present for zero, negative and positive inputs and three seeds,
but neither compiler-gradient probe reached execution. The checked adapter and
its LP64 C static assertion are unchanged. Faer enters Rust matrix code (this
shape dispatches to row-major matvec); the symbol table confirms that path.
CBLAS uses 32-bit dimension/stride declarations and a foreign call. The loaded
OpenBLAS exports `cblas_dgemm` and Fortran `dlascl_`, but not `cblas_dlascl`.
No provider-specific AD rule was found in the inspected Alea/Faer/CBLAS wrappers;
no custom rule, helper shim or alternative arithmetic was added for these probes.

These are bounded configuration-specific failures, not proofs that all Faer or
BLAS differentiation is impossible. Larger/transposed products, other compilers,
custom-rule composition, higher derivatives and inference are unqualified.
Primal success and generated derivative code alone are not derivative evidence.
Direct compiler reverse and curvature remain unsupported; the separate explicit
first-order lowering above avoids these compiler paths.
The test target's opt-in `alea_probe_reverse` cfg isolates deliberate failing
reverse probes; `alea_probe_blas` selects the explicit BLAS feature cell. See the existing
[Enzyme conformance instructions](../tools/enzyme-conformance/README.md) for
commands; no provider feature-forwarding redesign is implied.

The explicit first-order composition is independently exercised on the same
pinned compiler for Faer and LP64-BLAS, in density and objective modes. Fixtures
cover 2×3 by 3×1 and 2×3 by 3×2 products, both active operands, existing adjoints,
overlapping gathers, shared reshape aliases, a captured inactive matrix, zero
seeds, empty rows/columns/inner dimensions, both branch paths, ordinary scans and
checkpointed scans. Independent scalar polynomials and coordinate derivatives
check every output score. Nonfinite input, finite-primal/VJP overflow, selected
matrix overflow hidden by a reciprocal or zero weight, and successful recovery
are checked separately. A mixed `sqrt(s*s)` factor retains its original compiler
helper and singular-score rejection instead of inheriting the provider route.

Primal and operand VJP calls use the actual selected library. First-only targets
do not implement curvature capabilities, and unsupported higher source entries
are not emitted. No new provider shim, custom Enzyme rule, global thread change
or feature forwarding is introduced. The source-owned arena includes the seed
copy needed by provider accumulation; provider packing/compiler storage remain
outside that budget. This is numerical/capability evidence, not timing, general
program qualification or inference/ESS evidence.

Reference contracts: [Faer matrix multiplication](https://docs.rs/faer/0.24.0/faer/linalg/matmul/fn.matmul.html)
and [BLAS DGEMM](https://netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html).
`alea-core`'s `tensor_gemm` Criterion group compares compensated/Faer/BLAS
value-and-VJP execution on identical inputs, validates values and both scores
before timing, and reuses workspaces. It excludes construction and does not
measure ESS/sec, source AD, Stan or JAX.

```sh
nix develop .#blas --command env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench --manifest-path crates/Cargo.toml -p alea-core --features faer,openblas --bench tensor_tape -- tensor_gemm
```

November 3 microbenchmark qualification: same-revision policy comparison, three
separate process runs, 20 Criterion samples each, 0.3 s warmup / 0.7 s requested
measurement per case. Saved release binary, fat LTO, rustc 1.97 nightly
`7af3402cd`, LLVM 22.1.2, Faer 0.24.0 and LP64 OpenBLAS 0.3.31; i5-8250U x86-64,
CPU 2 affinity, one BLAS thread. No builds/profiles/archive compression overlapped
timing. Shared desktop, unisolated SMT sibling and variable frequency remain
limitations. Each case multiplies two square parameter matrices and computes
both input VJPs with a dense nonzero seed; validation precedes timing.

Times below are **microseconds**, median of three per-process median estimates;
brackets show their observed range, not confidence intervals:

| Matrix order | Compensated reference | Faer | BLAS |
| --- | ---: | ---: | ---: |
| 8 | 6.480 [6.375, 6.493] | 1.029 [1.029, 1.045] | 1.381 [1.371, 1.396] |
| 64 | 3,184 [3,089, 3,231] | 100.3 [98.5, 103.3] | 109.0 [106.3, 170.0] |
| 128 | 35,774 [34,462, 36,646] | 715.6 [697.1, 721.2] | 659.6 [655.6, 667.1] |

This is not a before/after commit comparison or equal-accuracy speedup claim:
external GEMM changes reduction arithmetic as well as using optimized kernels.
It does not establish default-path parity, general tensor speedups, source-AD
performance or ESS/sec. The BLAS 64-order variation is retained, not discarded.
No automatic selection thresholds are inferred from these three shapes.
Reproduce the short cohort from the built benchmark executable with
`--bench tensor_gemm --warm-up-time 0.3 --measurement-time 0.7 --sample-size 20 --noplot --save-baseline gemm-1`
(repeat with distinct baseline names under the same affinity/thread settings).
Raw Criterion samples, estimates, all three process logs and the executed binary
are retained in the external `tensor-gemm-2026-11-03.saxTKU` evidence capsule.
Full release core/math tests and doctests with both backends passed (192 tests),
as did both-crate all-target Clippy with warnings denied. Additional default-only
checks retain the ordered matvec bit tests. Fractional rectangular fixtures,
central finite differences, immutable rebinding, existing adjoints, aliases,
zero seeds, empty dimensions and overflow/output-staging recovery are covered.

## Structured conditional and loop semantics

```rust
use alea::model::{
    assembly::ModelBuilder,
    tensor::{BinaryOp, Shape},
};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut body = ModelBuilder::new().region();
let carry = body.input("carry", Shape::scalar())?;
let row = body.input("row", Shape::scalar())?;
let gain = body.input("gain", Shape::scalar())?;
let scaled = body.binary(BinaryOp::Multiply, carry, gain)?;
let next = body.binary(BinaryOp::Add, scaled, row)?;
let body = body.compile(next)?.bind(&[])?;

let mut b = ModelBuilder::new().region();
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

`DataBinding::rebind::<P>(expression, update)` borrows an immutable snapshot,
validates the same semantic signature and reporting layout, and returns a new
snapshot with a fresh identity. Compile it to obtain a new `Executable`, then
construct contexts or owned targets for it. Descriptor captures are immutable;
mutable native/reporting workspaces are allocated independently by each context.
Failure preserves the old snapshot and returns the original typed error directly.
A panic also leaves the borrowed snapshot usable. Do not reuse a context or
PointState from the old executable. Frozen generated functions require regeneration.

`NormalRegression`, `BernoulliLogitRegression` and `PoissonLogRegression` expose
`with_data` for fixed-row/fixed-column replacements, retaining typed observations,
design orientation, coefficients and offsets. `DataBinding::rebind_region` rebinds
a named tensor binding. Changing parameter layout, dimensions or dependency
handles requires rebuilding, not pretending it is data-only rebinding.

## Automatic native batch reporting and source lowering

The operation author installs optional `NativeRules::batch` and
`NativeRules::source` capabilities in the typed rule bundle.
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
automatic BLAS scheduling or generative site language is claimed. The small tensor/native-law
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

- `Executable::source_primal(SourceMode)` returns a named `SourceProgram`
  containing source and a checked `FloatWorkspace` element count. `SourceRule::workspace` declares
  source-only scratch independently of native evaluation scratch. Combined sizes,
  pruning and consuming rebinding are tested; source formatting remains lazy.
- Factor arguments and tensor mutable intermediates live in an external float
  arena, passed as a duplicated argument to compiler AD. Each differentiated
  wrapper also receives its working buffers explicitly. Inputs, constants, data
  and reshapes are borrowed views. Copies are typed loops, with direct operations
  for one-element blocks. Larger maps and scan iterations are not source-unrolled.
- Generated region helpers remain visible to AD through inlining. The standalone
  `BoundProgram::rust_function` returns `Result<String, AssemblyError>` and rejects
  missing source capabilities before formatting. Its generated primal wrapper
  owns local scratch: use the compiled-model exporter for nested compiler AD.

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

First-order derivative selection is factor-local inside a composed model.
A trusted score rule supplies its own factor's scores. A checkpointed tensor
uses segmented replay for its own pullback. Ordinary tensors and eligible
opaque source factors use compiler reverse. Ordinary tensors in a composed
model still run the checked tensor pullback for local-score admission. The
executor discards those scores before computing the compiler gradient.
The Enzyme `route_*` fixtures check the emitted helpers and calls with trusted,
checkpointed, opaque and excluded companions.

The admission investigation exposed a counterexample: with A equal to
`sqrt(x*x)`, native evaluation rejects the singular local score at zero, whereas
the unguarded pinned compiler-only generated target returned value and score zero.
Inspecting the actual pinned compiler's emitted LLVM IR identifies the mechanism:
the reverse rule for `sqrt` compares its **input** with zero and selects no
adjoint increment at zero instead of propagating `seed / (2 * sqrt(input))`.
The product's reverse then receives zero. This is an explicit branch in generated
derivative code; it is not evidence that Alea lost the aliased input or that the
compiler simplified the entire expression to an absolute-value operation. The
emitted derivative arithmetic also carries LLVM `fast` flags; this inspection
does not establish that those flags cause the zero-input selection.

Compiler-only tensor derivative entries now run the existing checked tensor
pullback in a separate admission helper, outside compiler differentiation. The
helper reuses the scalar composition and tensor VJP; its score is discarded.
The actual returned gradient and curvature remain compiler-derived. Value-only
evaluation retains primal admission and can return zero at this point. Hybrid
first-order entries already admit tensor scores and do not add another helper.
`SOURCE_TENSOR_SCORE_VALIDATION` reports this extra work separately from
`SOURCE_EXPLICIT_FIRST_ORDER` and `SOURCE_CHECKPOINTED`. The existing benchmark
receipt records derivative route L with admission H and an explicit extra-work
reason, rather than reporting pure compiler execution costs.

The validation adds a complete checked composition evaluation and its scratch
before each compiler-only tensor gradient, including the first-order check used
by curvature. Its timing and total process memory have **not been measured**.
It preserves selected branches, static activity and dynamic zero-seed behavior;
it does not reject every occurrence of `sqrt(0)` in a forward pass. The source
rules for arbitrary opaque callbacks remain the author's responsibility, and
this check does not establish equivalence to a hidden handwritten native VJP.

The existing `route_*` tests cover all nine compositions through both density and
objective entries, independent nonzero derivatives through third order, invalid
supplied scores, singular rejection and recovery. The ordinary tensor admission
fixture additionally tests lazy selection between a singular active root and a
masked root with zero seed plus an immutable `sqrt(0)`. Admission failures are
collected until all derivative and recovery checks run. This qualification does
not extend to different compilers or higher differentiation of a manual VJP.

The generated `SOURCE_EXPLICIT_FIRST_ORDER` constant reports whether this explicit
first-order path was selected; `SOURCE_CHECKPOINTED` reports tensor checkpointing
within that path. Its float-arena extent is checked, but is not a
whole-process memory bound. Original input/output storage and dense active tensor
adjoints remain; opaque source factors can retain compiler-owned tapes. Replay
also costs computation: the initial implementation executed five body-primal
passes per trip (model admission, VJP admission, boundary preparation, segment
replay and local VJP admission). Segment-primal reuse below removes the last of
these passes; nested replay can still multiply the cost. This is a
memory/work tradeoff, not a claimed inference speedup or optimal scheduler.

### The higher-order compiler boundary is still open

Actual forward-over-explicit-reverse compilation failed on the pinned compiler:
first with an unresolved `enzyme_primal_return` global for nested reverse
helpers, then with an `Illegal updateAnalysis` pointer/integer type conflict for
the rematerialized program. Reusing native chart pullbacks avoids duplicated
mathematics, but did not make that compiler failure disappear. The follow-up
below isolates one type-analysis trigger without qualifying the higher-order path.

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

### Higher-order follow-up: compilation is not a correctness gate by itself

The saved LLVM trace localizes the pointer/integer conflict to the final
`g.iter().chain(a.iter()).any(...)` finite check in a differentiated tensor
pullback. Enzyme's differentiated `Chain` state has pointer fields, but its
analysis also encounters bytewise initialization of that state. Replacing the
check with `g.iter().any(...) || a.iter().any(...)` removes this observed trigger
in the actual generated checkpoint model. It checks exactly the same values in
the same order, with the same short-circuit rejection. A tiny iterator-only
probe already compiles successfully, so this is not evidence that every Rust
chained iterator fails AD.

This allowed standalone forward-over-explicit-reverse and third-derivative
executables to compile, including a mixed native Normal prior/positive chart.
It did **not** qualify those routes. Forcing the explicit checkpoint pullback
into the full conformance binary exposed two independent failures:

- A zero-trip `log(abs(q[0]))` fixture at `q=[-10,1]` returned Hessian entries
  `H00=0.11` and `H01=0.1`, instead of the independent values `0.01` and `0`.
  The three-trip recurrence also returned `H00=0.1183905745088507` instead of
  `0.011438702851096686`. First-order scores were correct. A standalone
  zero-trip executable returned correct higher derivatives, so the observed
  failure depends on compilation context; the underlying cause is not resolved.
- A larger mixed-factor conformance binary still failed preprocessing with
  `enzyme_primal_return`, despite its standalone counterpart compiling. Explicit
  helper inlining did not resolve the mixed-factor failure or the wrong
  negative-branch derivatives.

Both experimental routings were rejected and preserved as diagnostic artifacts.
The shipped generator still uses the original compiler reverse for HVPs and
third derivatives. A generator regression asserts that routing boundary. Expanded
executed tests compare checkpoint-model Hessians to central differences of the
independent native analytic scores, and third derivatives to Hessian differences.
They cover long/vector scans, shared captures, constants, matrices, mixed priors,
positive charts, and objective Jacobian exclusion. The singular first/steady
activity fixture additionally has an independent nested-square-root Hessian and
third derivative, plus zero/nonzero-direction numerical-error checks.

These tests qualify the **shipped mixed implementation**, not higher-order
checkpointing. No `LooseTypes`, numerical clipping, validation removal, or
finite-difference implementation was introduced. The
[Rust compiler's AD flag documentation](https://rustc-dev-guide.rust-lang.org/autodiff/flags.html)
explicitly warns that relaxed type analysis can produce incorrect derivatives;
these experiments retained strict type analysis and still required independent
mathematical checks after compilation. Rust-side custom-derivative/checkpoint
controls also remain limited in the
[Enzyme interface](https://enzyme.mit.edu/rust/limitations.html).

The reproduction capsule `source-higher-2026-11-03.MlD5Pv` preserves the pinned
compiler commands, generated sources, failed LLVM logs, successful standalone
outputs, rejected routing/inlining overlays on `d0b88fe`, and passing fallback
tests. This increment makes no latency, peak-memory, sampler, or ESS/sec claim.

The expanded tests also found a separate bug in the shipped fallback: for the
three-step recurrence `c <- sqrt(c) + g`, starting from constant zero, the value
at `g=0` is finite but its score is singular. `logp_grad` rejected it while a
zero-direction HVP returned `Ok(())` and `-0.0`. Checking only primal support
was insufficient. Merely inspecting the gradient side output of the curvature
AD call did not fix this counterexample either.

Generated HVP and dense higher-derivative calls now admit the point using the
same value-and-gradient entry point as sampling, before running curvature AD.
That replaces their former primal-only admission pass, and rejects a singular
score even when a zero direction would hide it. The result remains a typed
`AutodiffError::NonFinite`; positive-gain derivatives and recovery are checked
separately. This can cost additional reverse work compared with primal-only
admission. No zero-overhead claim is made, and the first-order sampling path is
unchanged. Future reuse must preserve this actual first-order admission contract,
not infer it from a finite curvature result or a discarded AD side output.

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
compiler boundary and replay cost above are not resolved by compaction.

### Reuse validated segment primals instead of evaluating them twice

The next source-level profile exposed a redundant pass: segment reconstruction
evaluated the body primal, discarded its intermediate arrays, then immediately
evaluated the same body again inside its local VJP. The new source lowering
separates the normal, validating pullback wrapper from a private prepared reverse
helper. Segment reconstruction retains the body's validated materialized primals;
reverse consumes those exact primals with the matching restored carry, row and
captures. It never uses a prepared helper after a failed primal. Seed and local
score validation still run, and first/steady activity and reverse accumulation
order are unchanged. General callbacks do not acquire this capability.

For a scan with T trips, C carry doubles and B source-primal doubles per body,
the segment storage term is now approximately
`ceil(T/K)*C + K*(C+B)`. Source lowering selects the integer heuristic
`K = clamp(floor(sqrt(T*C/(C+B))), 1, T)`, with widened integer arithmetic and
checked arena extents. This accounts for body primals, unlike simply choosing
`sqrt(T)`. It is not an optimal checkpoint scheduler or a promise that the new
scratch arena is smaller than the previous carry-only replay arena: retaining
more primals trades storage for less recomputation. Active intermediate lifetimes
are still not compacted.

At this scan level the model gradient now needs four rather than five complete
body-primal passes per trip. Model admission, VJP admission and boundary
preparation remain separate. Nested scans and branch pullbacks can add further
recomputation; HVP/third derivatives still use the original compiler reverse.
This increment does not resolve higher-order source checkpointing.

This follows the general saved-residual/recomputation distinction described in
the [JAX checkpointing guide](https://docs.jax.dev/en/latest/gradient-checkpointing.html):
checkpoint placement and the residuals retained for reverse must be considered
together. It is not an implementation of JAX's optimizer or a new AD algorithm.

Correctness controls include a 65-trip nonlinear two-element carry, non-divisible
segments, independent recurrence derivatives, all-coordinate finite differences,
first/steady singular-score admission, repeated captures, active rows, zero-trip
and empty-result errors, and the existing matrix/mixed-prior/chart checks. The
new vector fixture also exposed ambiguous generated assignment formatting; the
emitter now writes `*a = *b` rather than `*a=*b`, without suppressing the lint.

Seven alternating pairs per case (84 successful fresh processes, 42 exact
density/all-coordinate-checksum-compatible pairs) compared saved release binaries.
The timer covers only the fixed-work evaluation/checksum loop, not compilation,
process startup, construction, warmup or serialization. These are deterministic
target evaluations, not sampling or ESS/sec results:

| Source case | Before µs/evaluation | After µs/evaluation | Median paired gain [min, max] |
|---|---:|---:|---:|
| Checkpoint, 3 trips | 0.395 | 0.348 | 1.136 [1.124, 1.176] |
| Checkpoint, 257 trips | 17.425 | 16.090 | 1.095 [1.034, 1.102] |
| Retained compiler reverse, 257 trips | 7.351 | 7.598 | 0.970 [0.961, 0.975] |
| Checkpoint + native prior/positive chart | 0.432 | 0.385 | 1.124 [1.087, 1.176] |
| Matrix + zero-trip checkpoint | 0.176 | 0.173 | 1.015 [0.992, 1.357] |
| Active rows/shared captures, 3 trips | 0.568 | 0.511 | 1.112 [1.105, 1.249] |

The uninstrumented cohort ran on CPU 2 of the same shared i5-8250U host, one
evaluation thread, variable clocks and unisolated SMT. No own builds, profilers or
other benchmark jobs overlapped it. The ranges are observed pair ranges, not
confidence intervals. All failures/pending jobs and stdout/stderr are retained
by the runner; there were no failed or unstarted jobs in this cohort.

Separate Callgrind runs of the 257-trip checkpoint case, 10,000 timed calls plus
64 untimed calls/startup, report 1,884,479,892 -> 1,641,293,351 instructions
(about 12.9% fewer). Instrumented times are excluded from the table. The retained
control instead has essentially identical work: 500,698,531 -> 500,698,412
instructions. Its 3,903-byte generated reverse function has the same 728
relocation-normalized instructions, checked switch targets and sampled constant
prefixes; its address/alignment changes. That rules out increased instruction
work in this hot function as the explanation for the observed timing loss, but
does not establish a hardware cache/front-end cause. The approximately 3% retained
control regression remains disclosed; this is not a regression-free result.

The retained compiler reverse is still faster than explicit checkpoint replay on
this small-body example. Checkpointing is a bounded-storage/work tradeoff, not an
automatic speed optimization. No peak-RSS reduction, universal shape gain,
higher-order checkpoint bound, native interpreter improvement or Stan/JAX
performance comparison follows from this cohort.

Reproduction capsule: `source-replay-2026-11-03.gIaEsR`, with source overlays on
`d5f31ba`, the executable `examples/source_checkpoint.rs`, pinned compiler
`4429659e4`/LLVM 22.1.7, fat LTO, raw paired outcomes, instruction profiles and the
relocation comparison. Before binary SHA-256:
`315c5fb6b87ca115c1017aa27cf0fa3757b150d30a9df1cf673fac513aae239f`;
after: `5ddd3d4c356069b1400193dc41a3ab14ab1b760b8334d53f099247bc058b5781`.
The before/after measured source-overlay archive hashes are respectively
`3997c5702f81c4b5881fb75e1a5b5cf2f85f10cb9c27d848798706a312bea85b` and
`67dee1f2d9e9af0dbd880825872fc9ce88a7eb2c4d96933003e88eec39d19287`.

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

## Backward storage and source residual selection

The native workspace can use a distinct adjoint layout: immutable and inactive
primal arrays have no adjoint allocation, and reshape aliases share their
producer's adjoints. If no fused/reused plan is needed and the saved storage does
not exceed the layout metadata, it retains the original dense adjoints and a
separately specialized dense reverse path. In particular, tiny scalar scan bodies
do not allocate an address plan merely to save one or two doubles. Optional
metadata stays outside hot retained-frame headers. Dense primal storage and
complete forward admission remain; workspace byte budgets conservatively cover
either selected representation.
For unfused regions with at least 1 MiB of active adjoints, the planner computes
inclusive backward live intervals over alias classes and uses a best-fit arena.
An output seed and its input destinations must coexist during its VJP. Reused
storage is cleared only after admitting the retiring seed; binary and matrix
pullbacks use a preallocated seed buffer when needed. The candidate is declined
unless arena plus seed scratch is smaller than compact non-reused storage.
This is bounded adjoint reuse, not global primal-buffer reuse or an optimal
allocation algorithm. Planning uses ordered free blocks and a lifetime heap,
with O((V+E) log V) scheduling; fragmentation can decline reuse.

Existing dependency-safe pointwise groups can also run tiled reverse operations.
Only active identity-indexed edges qualify; active broadcast/gather reductions
retain the original reverse traversal. Eligibility is cached with workspace
activity, not rediscovered per element. Reverse-node and operand order within
each coordinate, zero-seed skipping, all intermediate finite checks and staged
caller outputs remain. All primal arrays are still materialized. No floating
reassociation or callback interpretation is introduced by this pass.

The explicit first-order source pullback separately selects its saved residuals
from the values its VJP rules actually read. Inputs and immutable bindings remain
borrowed, aliases share their producer, and control predicates/arguments remain
available for complete selected-child admission. Every forward operation still
runs and is validated. Root scratch is compacted in place after admission;
checkpointed scans pack only selected residuals into each segment frame, using
one shared full-primal replay frame. First-trip and steady-state activity can
require different residual layouts. This does not reduce the four whole-body
passes per gradient trip, establish optimal checkpointing, or qualify higher
derivatives through the explicit reverse program. The original tested compiler
reverse remains the HVP/third-order route.

For matrix products, `RegionBuilder::with_matrix_kernel` sets the policy for
subsequent products in that lexical builder. Existing nodes and child regions
retain their own policies. Optional `MatrixKernel::Auto` resolves once to an
available provider, preferring sequential Faer when both providers are enabled.
It is **availability dispatch**, not a fitted shape/cost model. Opting in permits
provider reassociation; default construction remains compensated. Qualified
value/explicit-first source export preserves the resolved provider and rejects
unqualified curvature. BLAS threads/provider scratch remain host-owned. Feature
availability alone never changes default arithmetic.

### Remote qualification of the adjoint planner

This increment compares production baseline `5b29e2f` with the saved final
candidate, using the identical updated `kernel_audit.rs` harness in both builds.
The six new vector/lifetime cases are a harness-only baseline overlay; all
seventeen prior tensor controls remain. Seven alternating pairs per shape yielded
322 successful jobs and 161 exact-checksum-compatible pairs, with no failed or
unstarted jobs. Complete-coordinate, finite-difference, alias and error-recovery
tests remain separate from checksum-based timing admission.

Measurements ran on **Dusk**, logical CPU 0, Intel i7-7500U, Rust 1.95.0/LLVM
22.1.2, generic release/fat LTO/one codegen unit and single-thread environment.
Both AC supplies reported online; the battery reported not charging. The host
used its existing powersave governor and shared background services, with initial
load averages about 1.83/1.49/1.61. Clocks, SMT and services were not isolated or
changed. No own builds, tests, profiles or archive jobs overlapped on Dusk.
Local correctness builds were separate. Laptop exploratory timings are retained
but **excluded** from these claims because variable AC/battery policy invalidates
pooling them with server measurements. All runners now share descriptive host,
affinity, governor, load and power metadata without changing host policy.

Values below are marginal median microseconds per fused evaluation. Gain is
paired before/after median [observed range], not the ratio of marginal medians,
a confidence interval, or a Stan/JAX/ESS comparison.

| Case | Elements/trips | Before µs | After µs | Paired gain [range] |
|---|---:|---:|---:|---:|
| tensor | 1,000 | 7.061 | 6.263 | 0.976 [0.813, 1.222] |
| tensor | 10,000 | 62.858 | 66.084 | 0.951 [0.905, 1.034] |
| tensor_scalar | 1,000 | 19.315 | 22.580 | 0.895 [0.693, 1.049] |
| tensor_scalar | 10,000 | 249.439 | 247.872 | 0.961 [0.849, 1.396] |
| tensor_matmul | 1,000 | 65.641 | 60.982 | 1.073 [0.859, 1.615] |
| tensor_matmul | 10,000 | 634.877 | 568.145 | 1.194 [0.819, 1.340] |
| tensor_cond | 1,000 | 10.710 | 10.886 | 0.964 [0.656, 1.003] |
| tensor_cond | 10,000 | 81.377 | 114.787 | 0.962 [0.575, 0.993] |
| tensor_scan | 8 | 1.567 | 1.955 | 0.854 [0.600, 1.096] |
| tensor_scan | 128 | 22.196 | 20.003 | 0.995 [0.816, 1.283] |
| tensor_scan | 1,025 | 192.750 | 159.852 | 1.043 [0.834, 1.489] |
| tensor_scan | 10,000 | 1846.482 | 1894.727 | 1.052 [0.927, 1.125] |
| tensor_scan_checkpoint | 1,025 | 226.743 | 221.654 | 0.968 [0.611, 1.275] |
| tensor_scan_checkpoint | 10,000 | 2657.989 | 2712.457 | 1.066 [0.787, 1.191] |
| tensor_dag | 1,000 | 36.665 | 37.002 | 1.003 [0.944, 1.144] |
| tensor_dag | 10,000 | 410.132 | 396.213 | 1.066 [0.936, 1.094] |
| tensor_dag | 100,000 | 7692.197 | 7656.597 | 0.958 [0.942, 1.086] |
| tensor_vector | 1,000 | 868.451 | 840.454 | 1.012 [0.956, 1.134] |
| tensor_vector | 10,000 | 8900.442 | 9026.827 | 0.952 [0.888, 1.067] |
| tensor_vector | 100,000 | 91332.242 | 90565.510 | 1.030 [0.907, 1.086] |
| tensor_reuse | 1,000 | 555.862 | 525.406 | 1.017 [0.864, 1.328] |
| tensor_reuse | 10,000 | 6008.758 | 5210.081 | 1.144 [1.069, 2.089] |
| tensor_reuse | 100,000 | 61409.082 | 53621.392 | 1.127 [1.061, 1.588] |

The 100,000-element lifetime control rose from about 16.3 to 18.6 evaluations/s;
its paired median gain is 1.127 [1.061, 1.588]. The 10,000-element lifetime control
gained 1.144 [1.069, 2.089]. These are supported fixed-work observations on this
host, not a general speed advantage. Vector/Tanh and shared-DAG cases mostly
remain near parity. Negative findings remain: the scalar controls have paired
medians 0.895 and 0.961, the eight-trip scan 0.854, and the large shared DAG 0.958.
The 10,000-element branch control is 0.962 with all seven pairs slower.
This is **not regression-free** and does not qualify an optimal scheduler or
attribute every changed scalar control to a new algorithm.

Preceding server candidates remain in the evidence. Unconditional metadata gave
a 0.845 paired median on the 10,000-trip retained scan; boxing alone gave 0.910.
Storage-cost admission restored that control to 1.024 in its own cohort, but
left a 0.932 paired median on the 1,025-trip retained scan. These cohorts are not
pooled. The final specialization candidate reports 1.043 and 1.052 respectively,
with wide ranges; do not select stronger individual gains from earlier trials.

A separate Dusk Callgrind control (1,025 trips, 100 evaluations, including
construction and the harness warm evaluation) records 161,637,115 instructions
before and 171,641,985 with the admitted storage candidate: about 6.2% extra work.
The shared binary reverse no longer specialized as in the baseline. Forcing
only its dispatch wrapper inline did **not** fix this (171,710,817 instructions).
Specializing its shared derivative loop too reduced the final count to
164,014,369: about 4.5% below that wrapper-only trial, but still 1.5% above the
baseline. This counterexample is why a favorable noisy wall cohort is not enough
to attribute a fix. Instrumented timings are excluded from wall results.

Common tiled binary maps now delegate to the existing specialized kernels;
generic map/operation dispatch per element was rejected. Unary VJPs share
bounded disjoint-slice loops rather than repeatedly indexing independent primal
and adjoint layouts. Optional address metadata is selected by storage cost and
kept outside hot retained-frame headers. Remaining control losses are not hidden.

Memory evidence is deliberately narrower than timing evidence. The forced-reuse
unit verifies more than eightfold reduction of adjoint arena plus seed scratch
versus compact non-reused adjoints for its 64-map fixture. Primals remain dense;
this is not an eightfold whole-model/RSS reduction. The additive source fixture
retains only its output instead of more than 32 vector-widths of full primals.
Actual generated-source tests qualify correctness, not a new source-AD latency
or higher-order memory claim.

Final qualification: 197 core/math release tests with both Faer and BLAS,
27 actual pinned strict-Enzyme conformance tests, the existing four-chain
native-law tensor workflow, 46 Python benchmark-protocol tests, backend and
benchmark Clippy, formatting, API compatibility/maturity and source-link guards.
No API maturity promotion follows.

Local evidence directory: `integrated-planner-2026-11-03.TneYsi`.
Final wall journals/manifest are `paired-inline-loop-dusk/`; preceding server
cohorts are `paired-dusk/`, `paired-boxed-dusk/`, `paired-admission-dusk/` and
`paired-inline-dusk/`. Baseline executable SHA-256
is `1d7c402155d42f84e71924ddf063cb1fc02d1a21b9e8e78c8c38ac335133d8a0`;
final is `984c5b3a057fdcfb4c1b6dac352ab624eb2b33dba2e66e4c67d9de08db45f5b0`.
The measured source delta is `candidate-inline-loop.patch`, SHA-256
`96574fa205ab506ec7b4ff9171ca0240b359790a73c05954dbe4a673be9220d5`,
against the sealed baseline bundle, with identical-harness overlay and exact
Cubature submodule sources retained. Later report edits do not change these
measured receipts. Capsules remain local; they are not independently published
or an offline toolchain distribution.

The final local `measurement-capsule-complete/` seals 124 files (98,048,646
uncompressed bytes) in three segments, including baseline source/bundle, saved
binaries, all five server cohorts, source deltas, profiles and validation logs.
Restoring it with the generic tool and replaying its final journal reproduces
the saved summary exactly. Its index SHA-256 is
`e836aa7a566ed7e7e002990e71657c2961f43008beb68e6ec8989fc91135f17f`.
The separate local `pdb-capsule/` retains the historical PosteriorDB payloads.
Nothing here asserts public artifact availability or independent reproduction.

Remaining limits include generic primal lifetime planning, profitable automatic
matrix cost models, broad model/compiler conformance, higher-order checkpoint AD,
reference-functional agreement and SBC. Availability dispatch, passing tests and
a smaller arena must not be described as solving these separate obligations.
