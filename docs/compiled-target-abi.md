# Compiled target ABI v1 — specification, not an implemented backend

2026-09-30. CPU float64 first. The sampler only sees checked Rust target
capabilities. FFI ownership/unsafe auditing lives in `alea-ffi`, with an adapter
above the boundary; no foreign ABI is introduced into core/math/MCMC hot paths.

Proposed descriptor header: `uint32_t abi_version`, `uint32_t struct_size`,
`uint64_t capabilities`, `size_t dimension`, opaque model/context pointer and
function pointers. Negotiate version/size/capabilities before touching optional
fields. Reject unknown mandatory capability bits. The descriptor is immutable.

Mandatory function:

```c
int32_t logp_grad(void *chain_context, const double *q, size_t dim,
                 double *logp_out, double *gradient_out);
```

Optional functions:

```c
int32_t potential_hvp(void *chain_context, const double *q, const double *v,
                      size_t dim, double *out);
int32_t batch_logp_grad(void *batch_context, const double *q, size_t chains,
                       size_t dim, uint32_t layout, double *logp_out,
                       double *grad_out, int32_t *lane_status);
```

The HVP is for **-logp**, unlike the score written by logp_grad. Layout values
0/1 denote chain-major/coordinate-major; reject all others. Distinct contexts
must be supplied per simultaneously executing chain unless the provider explicitly
advertises safe shared access. Backend factories/destructors have matching ownership;
never free foreign allocations with Rust's allocator. Context lifetime exceeds all
calls and cached points; changing model semantics invalidates caches.

Status 0 means complete finite output, positive values model-domain failures,
negative values backend/ABI failures. Reserve exact numeric ranges before freezing
a public header. Batch return status describes the call; lane_status describes
individual evaluations. Failed-lane outputs are unspecified and cannot be cached.
Use a separate caller-buffer error-message function, not foreign-owned strings.

Inputs are immutable and do not alias outputs. Every pointer/length, byte-size
product and alignment is checked on the Rust side before references are formed.
Zero dimensions/batches have an explicit no-dereference convention. Providers may
not retain caller buffers, unwind/throw across the ABI, or invoke Rust callbacks
from unknown threads. C++/Rust providers catch exceptions/panics at their boundary.
GPU completion/streams require a later capability/version, not implicit asynchronous
use of borrowed slices.

Candidate providers: analytic Rust, Enzyme/LLVM, BridgeStan, JAX, Numba, C/C++,
experimental Futhark. Each needs lifetime, concurrent-context, partial-write,
nonfinite/error, layout and leak tests, plus a tiny independently compiled fixture.
Do not ship a safe wrapper before these obligations are implemented and audited.
