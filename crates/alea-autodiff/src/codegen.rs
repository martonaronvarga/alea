//! Ahead-of-time lowering of typed native models to generated Enzyme targets.
use alea_core::model::assembly::{AssemblyError, CompiledModel};

/// Emits a complete Rust module body with `log_density` and `target` functions.
///
/// Call from `build.rs`, write to `OUT_DIR`, then `include!` it in a module of a
/// crate with `#![feature(autodiff)]` on Alea's pinned `.#autodiff` toolchain.
/// The consuming crate needs the `alea` dependency with its `std-autodiff` feature.
/// `target()` supplies fused gradients, forward-over-reverse HVPs and
/// forward-over-forward-over-reverse third derivatives. All derivatives include
/// transforms and Jacobians; the dense outputs are derivatives of **potential**.
/// The wrapper uses O(d+S) fixed-size stack scratch, where S is the source
/// workspace element count, and O(d²) directional third-derivative evaluations.
/// This does not bound compiler-generated reverse tapes or guarantee arbitrary
/// programs fit a thread stack. Native analytic evaluation is unaffected.
/// Explicit `checkpointed_scan` regions select a source-generated first-order
/// pullback, exposed by the generated `SOURCE_CHECKPOINTED` receipt. HVPs and
/// third derivatives retain the compiler-owned reverse path: they are not
/// checkpointed. Other factors' compiler tapes are not bounded by the receipt.
/// Trusted primitive first-order rules reuse native value/score kernels where
/// provided; `SOURCE_EXPLICIT_FIRST_ORDER` distinguishes this from compiler AD.
/// Frozen regression kernels initialize a cache once per calling thread, then
/// reuse their native workspace. That heap storage and initialization cost are
/// outside the fixed source scratch receipt. Their original scalar primals
/// remain the independent compiler path for higher derivatives.
/// HVP/tensor calls admit the same value and gradient entry point as sampling:
/// a finite directional result cannot admit an invalid primal or singular score,
/// including for a zero direction. This replaces primal-only admission and can
/// cost additional reverse work; AD side outputs are not an admission oracle.
///
/// # Errors
/// Rejects zero unconstrained dimension and opaque nodes without a Rust primal.
pub fn enzyme_source(model: &CompiledModel) -> Result<String, AssemblyError> {
    mapped_source(model, true)
}

/// Emits `value` and `objective` functions for f(T(q)), without transform
/// Jacobians in the value or any derivative. The model's selected factors are
/// retained; use explicit statistical selection for likelihood or MAP views.
/// The returned higher derivatives are of negative objective, as required by
/// `PotentialHessianDerivative`. Uses the same explicit stack buffers and
/// first-order admission as `enzyme_source`; compiler-owned tapes retain the
/// limitations documented there.
/// # Errors
/// Rejects zero free dimension or an opaque factor without a scalar primal.
pub fn enzyme_objective_source(model: &CompiledModel) -> Result<String, AssemblyError> {
    mapped_source(model, false)
}

fn mapped_source(model: &CompiledModel, density: bool) -> Result<String, AssemblyError> {
    #[cfg(feature = "tracing")]
    let _span =
        tracing::debug_span!(target: "alea::model", "enzyme_lowering", density_jacobian = density)
            .entered();
    #[cfg(feature = "tracing")]
    tracing::debug!(target: "alea::model", schema_version = 1_u64, event = "started");
    let outcome = mapped_source_inner(model, density);
    #[cfg(feature = "tracing")]
    match &outcome {
        Ok(_) => tracing::debug!(target: "alea::model", schema_version = 1_u64, event = "finished"),
        Err(_) => tracing::error!(target: "alea::model", schema_version = 1_u64, event = "failed"),
    }
    outcome
}
fn mapped_source_inner(model: &CompiledModel, density: bool) -> Result<String, AssemblyError> {
    let (body, mut workspace) = if density {
        model.rust_primal_with_workspace()?
    } else {
        model.rust_objective_primal_with_workspace()?
    };
    let dimension = model.layout().unconstrained_dimension();
    if dimension == 0 {
        return Err(AssemblyError::Empty);
    }
    let validation = if density {
        model.rust_validation()
    } else {
        model.rust_objective_validation()
    };
    let checkpoint = model.rust_first_order_gradient(density)?;
    let mut template = TEMPLATE.to_owned();
    template = template.replace(
        "__CHECKPOINTED__",
        if checkpoint
            .as_ref()
            .is_some_and(|(_, _, checkpointed)| *checkpointed)
        {
            "true"
        } else {
            "false"
        },
    );
    template = template.replace(
        "__EXPLICIT_FIRST_ORDER__",
        if checkpoint.is_some() {
            "true"
        } else {
            "false"
        },
    );
    if let Some((source, count, _)) = checkpoint {
        workspace = workspace.max(count);
        template = template.replace(
            "#[::std::autodiff::autodiff_reverse(__gradient, Duplicated, Duplicated, Active)]",
            &(source + "\n#[::std::autodiff::autodiff_reverse(__higher_gradient, Duplicated, Duplicated, Active)]"),
        );
        // The explicit VJP can now pass forward type analysis, but independent
        // negative-branch derivatives still fail and mixed-factor compilation
        // remains context-dependent. Keep the qualified compiler reverse.
        template = template.replace(
            "__gradient(q, g, work, reverse, 1.0)",
            "__higher_gradient(q, g, work, reverse, 1.0)",
        );
    }
    Ok(template
        .replace("__DIMENSION__", &dimension.to_string())
        .replace("__WORKSPACE__", &workspace.to_string())
        .replace("__PRIMAL__", &body)
        .replace("__VALIDATION__", &validation)
        .replace(
            "__PUBLIC_VALUE__",
            if density { "log_density" } else { "value" },
        )
        .replace(
            "__PUBLIC_TARGET__",
            if density { "target" } else { "objective" },
        ))
}

const TEMPLATE: &str = r#"
pub const DIMENSION: usize = __DIMENSION__;
/// Whether first-order gradients use explicit segmented tensor replay.
/// Higher derivatives still use the compiler-owned reverse tape.
pub const SOURCE_CHECKPOINTED: bool = __CHECKPOINTED__;
/// Whether sampling uses an explicit first-order pullback rather than compiler reverse.
pub const SOURCE_EXPLICIT_FIRST_ORDER: bool = __EXPLICIT_FIRST_ORDER__;

// Each float buffer must stay explicit across nested AD. The generated Enzyme
// ABI duplicates arguments; a user-facing argument-count convention does not
// apply to these private compiler entry points (including macro-generated ones).
#[allow(clippy::too_many_arguments)]
// Shape-specialized generated loops intentionally retain zero offsets, empty
// ranges, explicit float copies and identical first/steady replay arms. These
// are generator style artifacts, not hand-written numerical validation waivers.
#[allow(clippy::identity_op, clippy::manual_memcpy, clippy::modulo_one,
        clippy::reversed_empty_ranges, clippy::if_same_then_else,
        clippy::possible_missing_else, clippy::approx_constant)]
mod __enzyme {
use super::DIMENSION;

#[::std::autodiff::autodiff_reverse(__gradient, Duplicated, Duplicated, Active)]
// Numeric literals serialize model data/bounds exactly; they are not approximations
// to named mathematical constants, even when their decimal representations match.
#[allow(clippy::approx_constant)]
pub(super) fn __log_density_work(q: &[f64; DIMENSION], scratch: &mut [f64; __WORKSPACE__]) -> f64 {
    let _ = &scratch;
    __PRIMAL__
}

pub(super) fn __log_density(q: &[f64; DIMENSION]) -> f64 {
    __log_density_work(q, &mut [0.0; __WORKSPACE__])
}

#[::std::autodiff::autodiff_forward(__gradient_jvp, Dual, Dual, Dual, Dual, Const)]
pub(super) fn __gradient_body(q: &[f64; DIMENSION], g: &mut [f64; DIMENSION], work: &mut [f64; __WORKSPACE__], reverse: &mut [f64; __WORKSPACE__]) -> f64 {
    __gradient(q, g, work, reverse, 1.0)
}

// The direction needs an explicit zero tangent at the call site. Marking this
// reference Const makes the pinned compiler's nested AD of scans inject a
// spurious Hessian term, even for a zero position direction.
#[::std::autodiff::autodiff_forward(__hvp_jvp, Dual, Dual, Dual, Dual, Dual, Dual, Dual, Dual, Const)]
pub(super) fn __hvp(q: &[f64; DIMENSION], v: &[f64; DIMENSION], g: &mut [f64; DIMENSION], h: &mut [f64; DIMENSION], work: &mut [f64; __WORKSPACE__], reverse: &mut [f64; __WORKSPACE__], tangent: &mut [f64; __WORKSPACE__], reverse_tangent: &mut [f64; __WORKSPACE__]) -> f64 {
    __gradient_jvp(q, v, g, h, work, tangent, reverse, reverse_tangent)
}
}
use __enzyme::{__gradient, __hvp, __hvp_jvp, __log_density};

fn __valid(q: &[f64; DIMENSION]) -> bool {
    let _ = q;
    __VALIDATION__
}

fn __first_order_valid(q: &[f64; DIMENSION]) -> bool {
    let mut score = [0.0; DIMENSION];
    let value = __gradient(q, &mut score, &mut [0.0; __WORKSPACE__], &mut [0.0; __WORKSPACE__], 1.0);
    value.is_finite() && score.iter().all(|v| v.is_finite())
}

pub fn __PUBLIC_VALUE__(q: &[f64; DIMENSION]) -> f64 {
    if q.iter().any(|x| !x.is_finite()) || !__valid(q) { f64::NAN } else { __log_density(q) }
}

pub fn __PUBLIC_TARGET__() -> impl ::alea::capability::PotentialHessianDerivative<Error = ::alea::autodiff::autodiff::AutodiffError> {
    ::alea::autodiff::autodiff::EnzymeHigherOrderTarget::new(
        DIMENSION,
        |q: &[f64], g: &mut [f64], seed: f64| {
            let q = q.try_into().expect("validated position");
            if !__valid(q) { g.fill(f64::NAN); return f64::NAN; }
            __gradient(q, g.try_into().expect("validated gradient"), &mut [0.0; __WORKSPACE__], &mut [0.0; __WORKSPACE__], seed)
        },
        |q: &[f64], v: &[f64], out: &mut [f64]| {
            let q = q.try_into().expect("validated position");
            // A zero directional result can hide an undefined gradient, and
            // AD-produced side outputs need not preserve first-order admission.
            if !__valid(q) || !__first_order_valid(q) { out.fill(f64::NAN); return; }
            let [work, reverse, tangent, reverse_tangent] = &mut [[0.0; __WORKSPACE__]; 4];
            let value = __hvp(q, v.try_into().expect("validated vector"), &mut [0.0; DIMENSION], out.try_into().expect("validated product"), work, reverse, tangent, reverse_tangent);
            if value.is_finite() { for x in out { *x = -*x; } } else { out.fill(f64::NAN); }
        },
        |q: &[f64], hessian: &mut [f64], third: &mut [f64]| {
            let q = q.try_into().expect("validated position");
            if !__valid(q) || !__first_order_valid(q) { hessian.fill(f64::NAN); third.fill(f64::NAN); return; }
            let mut v = [0.0; DIMENSION];
            let mut w = [0.0; DIMENSION];
            let mut h = [0.0; DIMENSION];
            let mut t = [0.0; DIMENSION];
            for j in 0..DIMENSION {
                v[j] = 1.0;
                for k in 0..DIMENSION {
                    w[k] = 1.0;
                    h.fill(0.0);
                    t.fill(0.0);
                    let [work, work_jvp, reverse, reverse_jvp, tangent, tangent_jvp, reverse_tangent, reverse_tangent_jvp] = &mut [[0.0; __WORKSPACE__]; 8];
                    let value = __hvp_jvp(q, &w, &v, &[0.0; DIMENSION], &mut [0.0; DIMENSION], &mut [0.0; DIMENSION], &mut h, &mut t, work, work_jvp, reverse, reverse_jvp, tangent, tangent_jvp, reverse_tangent, reverse_tangent_jvp);
                    if !value.is_finite() { hessian.fill(f64::NAN); third.fill(f64::NAN); return; }
                    for i in 0..DIMENSION {
                        if k == 0 { hessian[i * DIMENSION + j] = -h[i]; }
                        third[k * DIMENSION * DIMENSION + i * DIMENSION + j] = -t[i];
                    }
                    w[k] = 0.0;
                }
                v[j] = 0.0;
            }
        },
    )
}
"#;
