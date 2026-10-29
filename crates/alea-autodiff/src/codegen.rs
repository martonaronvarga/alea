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
/// Successful evaluations allocate nothing. Dense tensor assembly uses O(d)
/// fixed-size stack scratch and O(d²) directional third-derivative evaluations.
/// HVP/tensor calls also evaluate the scalar primal once for domain admission:
/// derivatives of a non-finite-return branch cannot establish valid support.
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
/// `PotentialHessianDerivative`. Uses the same bounded, allocation-free generated
/// derivative path as `enzyme_source`.
/// # Errors
/// Rejects zero free dimension or an opaque factor without a scalar primal.
pub fn enzyme_objective_source(model: &CompiledModel) -> Result<String, AssemblyError> {
    mapped_source(model, false)
}

fn mapped_source(model: &CompiledModel, density: bool) -> Result<String, AssemblyError> {
    let body = if density {
        model.rust_primal()?
    } else {
        model.rust_objective_primal()?
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
    Ok(TEMPLATE
        .replace("__DIMENSION__", &dimension.to_string())
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

#[::std::autodiff::autodiff_reverse(__gradient, Duplicated, Active)]
// Numeric literals serialize model data/bounds exactly; they are not approximations
// to named mathematical constants, even when their decimal representations match.
#[allow(clippy::approx_constant)]
fn __log_density(q: &[f64; DIMENSION]) -> f64 {
    __PRIMAL__
}

fn __valid(q: &[f64; DIMENSION]) -> bool {
    let _ = q;
    __VALIDATION__
}

pub fn __PUBLIC_VALUE__(q: &[f64; DIMENSION]) -> f64 {
    if q.iter().any(|x| !x.is_finite()) || !__valid(q) { f64::NAN } else { __log_density(q) }
}

#[::std::autodiff::autodiff_forward(__gradient_jvp, Dual, Dual, Const)]
fn __gradient_body(q: &[f64; DIMENSION], g: &mut [f64; DIMENSION]) -> f64 {
    __gradient(q, g, 1.0)
}

#[::std::autodiff::autodiff_forward(__hvp_jvp, Dual, Const, Dual, Const)]
fn __hvp(q: &[f64; DIMENSION], v: &[f64; DIMENSION], h: &mut [f64; DIMENSION]) -> f64 {
    let mut g = [0.0; DIMENSION];
    __gradient_jvp(q, v, &mut g, h)
}

pub fn __PUBLIC_TARGET__() -> impl ::alea::capability::PotentialHessianDerivative<Error = ::alea::autodiff::autodiff::AutodiffError> {
    ::alea::autodiff::autodiff::EnzymeHigherOrderTarget::new(
        DIMENSION,
        |q: &[f64], g: &mut [f64], seed: f64| {
            let q = q.try_into().expect("validated position");
            if !__valid(q) { g.fill(f64::NAN); return f64::NAN; }
            __gradient(q, g.try_into().expect("validated gradient"), seed)
        },
        |q: &[f64], v: &[f64], out: &mut [f64]| {
            let q = q.try_into().expect("validated position");
            // A higher derivative can be finite on a NaN-return branch. Check
            // primal admission explicitly before evaluating the HVP.
            if !__valid(q) || !__log_density(q).is_finite() { out.fill(f64::NAN); return; }
            let value = __hvp(q, v.try_into().expect("validated vector"), out.try_into().expect("validated product"));
            if value.is_finite() { for x in out { *x = -*x; } } else { out.fill(f64::NAN); }
        },
        |q: &[f64], hessian: &mut [f64], third: &mut [f64]| {
            let q = q.try_into().expect("validated position");
            if !__valid(q) || !__log_density(q).is_finite() { hessian.fill(f64::NAN); third.fill(f64::NAN); return; }
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
                    let value = __hvp_jvp(q, &w, &v, &mut h, &mut t);
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
