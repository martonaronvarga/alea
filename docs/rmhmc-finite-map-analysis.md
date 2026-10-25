# Finite RMHMC solver support and volume

The implemented Newton/replay kernel remains `ExactnessPendingRmhmc`. The
analysis below gives a negative result for the general finite-tolerance algorithm:
its accepted numerical map need not preserve volume, even when full tolerant
reverse replay succeeds. Tight settings are useful numerical controls; they are
not an exact-target proof.

The distinction between ideal root selection and practical Newton stopping is
explicit in [Lelièvre, Santet and Stoltz, sections 2.2 and 4.3](https://arxiv.org/html/2303.15918).
Their ideal volume and reversibility assumptions must be established before
applying the corresponding invariant-measure argument. The examples and finite-map
derivation here are independent calculations for Alea's actual equations.

## Exact-root accepted support

Let `F` denote one complete generalized-leapfrog map and `R(q,p)=(q,-p)`.
For an exact, regular implicit root branch, generalized leapfrog is canonical.
On that branch `|det DF|=1`. This is a statement about solving the implicit
equations exactly, not about arbitrary finite Newton iterates.

Set `P=R∘F`. Valid deterministic Metropolis proposals require a measurable
accepted support `D` on which `P(P(z))=z`, `P(D)=D`, and the map preserves phase
volume. Rejecting outside such a support gives the identity there. The usual
energy acceptance then balances the pair of densities `exp(-H(z))` and
`exp(-H(P(z)))`. A numerical branch that chooses another implicit root on replay
has no such pair and must be excluded.

Checking endpoint reversal alone does not establish consistent intermediate
branches. Alea therefore checks every saved phase and intermediate half-momentum.
The exact ideal argument requires exact reverse identities; current finite
checks use a declared scaled tolerance.

## Multiple roots are real algorithm choices

Use one coordinate, potential `U(q)=q²/2`, and metric `G(q)=exp(q)`. Its phase
Hamiltonian is

`H(q,p) = q²/2 + q/2 + p² exp(-q)/2`.

The force is `f(q,p)=-q-1/2+p² exp(-q)/2`. Half-momentum `u` satisfies

`A u² - u + B = 0`, where `A=(h/4)exp(-q)` and `B=p-(h/2)(q+1/2)`.

A positive discriminant gives two roots. The near root is stably evaluated as
`2B/(1+sqrt(1-4AB))`; the other is `(1+sqrt(1-4AB))/(2A)`.
The momentum Jacobian is `1-(h/2)exp(-q)u`, which can vanish. At initial
`q=0,p=2,h=1`, Newton's first Jacobian is exactly zero and the proposal is rejected.

The position equation is `v=q+(h/2)u(exp(-q)+exp(-v))`. When `u<0`, define
`a=-hu/2>0`: the residual has its minimum at `v=log(a)` and can also have two
roots. Selecting a near momentum root does not remove position-root ambiguity.
The independent 80-digit fixtures enumerate both momentum branches and the
available position branches; they do not assume that root convergence identifies
a unique reversible map.

## A finite accepted map with non-unit determinant

Suppose the finite residual policy accepts the initial Newton guesses `u=p` and
`v=q`. On the open neighborhood where the same stopping predicates hold, the
implemented map is

`Fε(q,p) = (q, p + (h/2)f(q,p))`.

Its Jacobian determinant is exactly

`det DFε(q,p) = 1 + (h/2)p exp(-q)`.

At `q=.2,p=-.35,h=.4`, this is approximately `.94269`, rather than one. This
is a derivative of the implemented finite map, not the ideal implicit map.
With forward and full-replay tolerances `.2`, the same example and its finite-
difference neighbors pass all boundary and half-momentum replay checks. The
executable test verifies both the analytic determinant and replay acceptance.
Thus tolerant replay does not establish either exact involution or volume.

The finite support is defined by solver and reverse-tolerance predicates. Those
predicates alone do not prove that it is invariant under the numerical proposal,
that the proposal is an exact involution, or that its determinant is one. A
Jacobian factor alone would not repair a non-involution without an implemented
inverse/paired proposal construction. Alea does not apply such an unsupported
correction.

This counterexample uses deliberately loose tolerances. It does not estimate
bias at the benchmark's `1e-12` residual and `1e-8` reverse settings. The solver
ladder records finite behavior at several tolerances; numerical closeness cannot
replace the missing structural guarantee.

## Executed evidence

- `tools/riemannian-reference/solver_ladder.py`: 72 independent Decimal cases
  across Picard/Newton, four tolerances and three budgets, plus eight valid root
  branches. `--check` verifies frozen fixture bytes without Alea output.
- `singular_newton_system_rejects_without_publishing`: singular linear-system
  rejection and unchanged live cache.
- `solver_tolerance_ladder_matches_independent_decimal_maps_and_failures`:
  selected maps, iteration counts, failure stages and no partial live commit.
- `multiple_implicit_roots_are_valid_but_do_not_define_one_selected_map`:
  alternate roots satisfy the actual momentum, position and final-kick equations.
- `finite_map_can_pass_full_tolerant_replay_and_still_change_volume`:
  accepted replay plus independent numerical and analytic Jacobians.

The derivation and tests complete the requested investigation with a negative
finding. They do not close the kernel's exactness obligation. Its type, ledger
and benchmark labels continue to expose that boundary.
