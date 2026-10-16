# M13 implemented-map exactness audit

Reviewed: 2026-10-16

This audit records what each public `MarkovChain` implementation actually does at
finite precision. It separates a target guarantee from validation maturity: an S1
kernel may implement a mathematically complete exact-target correction while still
lacking broad statistical, performance, API, or operational evidence. Conversely,
an energy-based accept/reject step is not enough to claim exactness when the
implemented proposal map has an open reverse-support or Jacobian obligation.

The machine-readable source is
[`tools/ci/exactness-ledger.json`](../tools/ci/exactness-ledger.json). CI discovers
the public implementations from the accepted `alea-mcmc` API snapshot, rejects
missing or stale entries, and requires a claim for:

- refresh law and extended-state density;
- forward/reverse support;
- implemented-map reversibility or involution;
- volume preservation, Jacobian, or proposal-density terms;
- candidate selection or Metropolis correction; and
- failure/rejection publication behavior.

Every entry names executable evidence. Evidence establishes the stated regression
scope; it is not automatically a proof over all floating-point inputs.

## Current transition guarantees

| Transition | Guarantee | Maturity | Implemented correction or limitation |
|---|---|---:|---|
| `ForceGradientHmc` | exact target | S1 | Canonical endpoint MH; both declared maps have signed-reversal/unit-volume evidence. |
| `GaussianSplitHmc` | exact target | S1 | Symmetric residual kick/exact Gaussian flow with canonical endpoint MH. |
| `SplitForceHmc` | exact target | S1 | Symmetric nested split with full endpoint density and canonical MH. |
| `CheesHmc` | exact target | S2 | Frozen state-independent path-length randomization followed by endpoint MH. |
| `DrGhmc` | exact target | S1 | Stage-two ratio includes the executed reverse first-stage rejection probability. |
| `Ghmc` | exact target | S1 | Gaussian autoregressive refresh, reversible proposal and persistent rejection convention. |
| `Hmc` | exact target | S2 | Full momentum refresh, symmetric unit-volume integration and endpoint MH. |
| `ApproximateMclmc` | approximate target | S1 | No MH step; the discarded noncanonical correction and bias acknowledgement are public. |
| `Mams` | exact target | S1 | Analytic isokinetic Jacobian/kinetic correction enters MH selection. |
| `Nuts` | exact target | S2 | Symmetric tree edges and multiplicity-preserving multinomial selection. |
| `Rmhmc` | exactness pending | S1 | Energy-only MH depends on unclosed numerical reverse-support, involution and volume assumptions. |
| `Rwmh` | exact target | S2 | Symmetric Gaussian proposal density cancels in the target-density MH ratio. |
| `Walnuts` | exact target | S1 policy | Candidate mass includes explicit reverse/forward refinement probability. |

“Exact target” is scoped to the declared fixed kernel and its assumptions. It does
not promote a kernel to S2, recommend it, or make warmup adaptive during retained
sampling. Finite warmups remain exact only because adaptation is consumed before a
fixed retained chain is returned.

## The RMHMC boundary

`Rmhmc` is deliberately the only `exactness_pending` transition. Its bounded Picard
iterations and forward residual make work deterministic, and sampled reversal plus
low-dimensional numerical-Jacobian tests remain useful regressions. They do not
show that a reverse finite-tolerance solve chooses the same branch for every
proposal. The present energy-only Metropolis correction therefore cannot close the
implemented-map exactness argument.

Promotion requires a per-proposal momentum-flipped reverse/involution check or
another derived correction, transactional rejection when reverse support fails,
and adversarial validation of the corresponding volume/Jacobian property. Until
then RMHMC remains an explicitly research-only geometry experiment.

## Remaining audit work

The inventory covers every public default-feature `MarkovChain` implementation and
the correction used by each transition. M13 still needs feature-specific API
coverage, reusable law/conformance suites for third-party implementations, and an
explicit audit of standalone proposal/integrator components that are public but do
not themselves implement `MarkovChain`.
