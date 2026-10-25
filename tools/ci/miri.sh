#!/usr/bin/env bash
set -euo pipefail
export MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check'
args=(--manifest-path crates/Cargo.toml --locked)
scope=${1:-all}
case "$scope" in
  all|storage|targets|hmc|nuts|research|runtime) ;;
  *) echo "usage: $0 [all|storage|targets|hmc|nuts|research|runtime]" >&2; exit 2 ;;
esac
selected() { [[ "$scope" == all || "$scope" == "$1" ]]; }

# Exact replay commands opt into deterministic floats; the default perturbed
# mode remains the numerical-robustness gate for all other commands.
if selected storage; then
cargo miri test "${args[@]}" -p alea-ffi --lib
cargo miri test "${args[@]}" -p alea-math --lib --test metrics --test inverse_metrics --test interpolation --test low_rank
fi
if selected targets; then
cargo miri test "${args[@]}" -p alea-core --test target --test transforms --test density --test capability --test transport
cargo miri test "${args[@]}" -p alea-autodiff --test models
cargo miri test "${args[@]}" -p alea-distributions --lib latent::tests
cargo miri test "${args[@]}" -p alea-distributions --test latent
fi
# Keep interpreter work bounded to memory, ownership, rollback, and deterministic
# algorithmic contracts. Native CI owns long statistical/calibration tests.
if selected hmc; then
cargo miri test "${args[@]}" -p alea-mcmc --test hmc_chain --test hmc_validation --test hmc_stress --test phase_state --test rwmh --test adaptation --test warmup --test warmup_reference --test fisher --test integrators --test integrator_failures
MIRIFLAGS="$MIRIFLAGS -Zmiri-deterministic-floats" cargo miri test "${args[@]}" -p alea-mcmc --test warmup warmup_freezes_is_deterministic_and_zero_iterations_consume_no_rng -- --ignored
fi
if selected nuts; then
cargo miri test "${args[@]}" -p alea-mcmc --test nuts_contracts
cargo miri test "${args[@]}" -p alea-mcmc --lib nuts::tests::iterative_carries_match_independent_recursive_scalar_oracle -- --exact
fi
if selected research; then
cargo miri test "${args[@]}" -p alea-mcmc --test walnuts_contracts
cargo miri test "${args[@]}" -p alea-mcmc --test isokinetic_failures
cargo miri test "${args[@]}" -p alea-mcmc --lib singular_newton_system_rejects_without_publishing
cargo miri test "${args[@]}" -p alea-mcmc --lib walnuts::tests::selected_reverse_level_is_replayed_at_floating_point_thresholds -- --exact
cargo miri test "${args[@]}" -p alea-mcmc --lib hamiltonian::tests::partial_step_error_and_unwind_require_restart_and_leave_live_cache_untouched -- --exact
cargo miri test "${args[@]}" -p alea-mcmc --lib advanced::tests::force_gradient_variants_reverse_and_preserve_unit_volume -- --exact
cargo miri test "${args[@]}" -p alea-mcmc --lib mams::tests::analytic_kick_preserves_unit_speed_and_reverse_correction -- --exact
MIRIFLAGS="$MIRIFLAGS -Zmiri-deterministic-floats" cargo miri test "${args[@]}" -p alea-mcmc --lib advanced::tests::exact_gaussian_flow_matches_closed_orbit_and_reverses -- --exact --ignored
fi
if selected runtime; then
cargo miri test "${args[@]}" -p alea-runtime --test streaming --test collection
cargo miri test "${args[@]}" -p alea-runtime --lib ring_is_bounded_contiguous_and_does_not_bridge_missing_energy
fi
