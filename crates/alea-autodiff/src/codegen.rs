//! Ahead-of-time lowering of typed native models to generated Enzyme targets.
use alea_core::model::assembly::{
    AssemblyError, CurvatureRoute, DerivativeOrder, Executable, FirstOrderSource, FloatWorkspace,
    SourceMode, SourceProgram, SourceRequest,
};

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
/// Frozen regression kernels construct typed state once per generated target,
/// then reuse that target's private native workspace. Heap storage and setup are
/// outside the fixed source scratch receipt. Their original scalar primals
/// remain the independent compiler path for higher derivatives.
/// HVP/tensor calls admit the same value and gradient entry point as sampling:
/// a finite directional result cannot admit an invalid primal or singular score,
/// including for a zero direction. This replaces primal-only admission and can
/// cost additional reverse work; AD side outputs are not an admission oracle.
/// Compiler-only tensor gradients additionally run the checked tensor pullback
/// for admission outside AD, reported by `SOURCE_TENSOR_SCORE_VALIDATION`.
/// That extra evaluation has its own workspace; it does not replace compiler
/// derivatives or change value-only admission. Its cost has not been benchmarked.
///
/// # Errors
/// Rejects zero unconstrained dimension and opaque nodes without a Rust primal.
pub fn enzyme_source(model: &Executable) -> Result<String, AssemblyError> {
    source(model, legacy_request(SourceMode::Density))
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
pub fn enzyme_objective_source(model: &Executable) -> Result<String, AssemblyError> {
    source(model, legacy_request(SourceMode::Objective))
}

fn legacy_request(mode: SourceMode) -> SourceRequest {
    SourceRequest {
        mode,
        order: DerivativeOrder::Third,
        curvature: CurvatureRoute::OriginalCompilerReverse,
    }
}

/// Emits only the requested value/derivative entry points, for either density or
/// objective. First-order composition retains factor-local trusted, checkpointed
/// and compiler rules. Curvature requires explicit original compiler reverse;
/// no trusted first-order rule is differentiated to manufacture higher support.
/// Eligibility inspection is not a numerical qualification of arbitrary source.
/// # Errors
/// Rejects missing active primals, zero-dimensional derivative targets and
/// curvature requests without an explicit qualified route.
pub fn source(model: &Executable, request: SourceRequest) -> Result<String, AssemblyError> {
    #[cfg(feature = "tracing")]
    let density = request.mode == SourceMode::Density;
    #[cfg(feature = "tracing")]
    let _span =
        tracing::debug_span!(target: "alea::model", "enzyme_lowering", density_jacobian = density)
            .entered();
    #[cfg(feature = "tracing")]
    tracing::debug!(target: "alea::model", schema_version = 1_u64, event = "started");
    let outcome = model
        .source_capabilities()
        .resolve(request)
        .and_then(|request| GenerationPlan::new(model, request))
        .map(|plan| plan.render());
    #[cfg(feature = "tracing")]
    match &outcome {
        Ok(_) => tracing::debug!(target: "alea::model", schema_version = 1_u64, event = "finished"),
        Err(_) => tracing::error!(target: "alea::model", schema_version = 1_u64, event = "failed"),
    }
    outcome
}
// These are source-generation choices, not a second model graph. Factor calls
// and local derivative routes are checked by the existing core lowerers.
struct GenerationPlan {
    request: SourceRequest,
    dimension: usize,
    primal: SourceProgram,
    first: Option<FirstOrderSource>,
    admission: ScoreAdmission,
    validation: String,
    workspace: FloatWorkspace,
    data: SourceData,
}

enum ScoreAdmission {
    None,
    Tensor(SourceProgram),
}

enum SourceData {
    Frozen,
}

impl GenerationPlan {
    fn new(model: &Executable, request: SourceRequest) -> Result<Self, AssemblyError> {
        let dimension = model.layout().unconstrained_dimension();
        if dimension == 0 && request.order != DerivativeOrder::Value {
            return Err(AssemblyError::Empty);
        }
        let primal = model.source_primal(request.mode)?;
        let first = if request.order == DerivativeOrder::Value {
            None
        } else {
            model.source_first_order(request.mode)?
        };
        let admission = if request.order != DerivativeOrder::Value && first.is_none() {
            model
                .source_score_admission(request.mode)?
                .map_or(ScoreAdmission::None, ScoreAdmission::Tensor)
        } else {
            ScoreAdmission::None
        };
        let validation = match request.mode {
            SourceMode::Density => model.rust_validation(),
            SourceMode::Objective => model.rust_objective_validation(),
        };
        let workspace = FloatWorkspace {
            elements: primal
                .workspace
                .elements
                .max(first.as_ref().map_or(0, |f| f.program.workspace.elements)),
        };
        Ok(Self {
            request,
            dimension,
            primal,
            first,
            admission,
            validation,
            workspace,
            data: SourceData::Frozen,
        })
    }
    fn higher(&self) -> bool {
        matches!(
            self.request.order,
            DerivativeOrder::Hvp | DerivativeOrder::Third
        )
    }
    fn third(&self) -> bool {
        self.request.order == DerivativeOrder::Third
    }
    fn first_workspace(&self) -> usize {
        self.first
            .as_ref()
            .map_or(self.workspace.elements, |f| f.program.workspace.elements)
    }
    fn stateful(&self) -> bool {
        self.first.as_ref().is_some_and(|f| f.stateful)
    }
    fn first_context_argument(&self, context: &str) -> String {
        if self.stateful() {
            format!(",&mut {context}.borrow_mut()")
        } else if self.first.is_some() {
            ",&mut __enzyme::__first_context()".into()
        } else {
            String::new()
        }
    }
    fn first_valid_call(&self, context: &str) -> String {
        format!(
            "__first_order_valid(q{})",
            if self.stateful() {
                format!(",&mut {context}.borrow_mut()")
            } else {
                String::new()
            }
        )
    }
    fn target_state(&self) -> String {
        if !self.stateful() {
            return String::new();
        }
        let mut state = "let __for_gradient=::std::rc::Rc::new(::std::cell::RefCell::new(__enzyme::__first_context()));\n".to_owned();
        if self.higher() {
            state.push_str("let __for_hvp=::std::rc::Rc::clone(&__for_gradient);\n");
        }
        if self.third() {
            state.push_str("let __for_third=::std::rc::Rc::clone(&__for_gradient);\n");
        }
        state
    }
    fn reverse_declaration(&self) -> &'static str {
        match (self.request.order, self.first.is_some()) {
            (DerivativeOrder::Value, _) | (DerivativeOrder::First, true) => "",
            (_, true) => {
                "#[::std::autodiff::autodiff_reverse(__higher_gradient, Duplicated, Duplicated, Active)]"
            }
            (_, false) => {
                "#[::std::autodiff::autodiff_reverse(__gradient, Duplicated, Duplicated, Active)]"
            }
        }
    }
    fn render_admission(&self) -> String {
        match &self.admission {
            ScoreAdmission::None if self.request.order == DerivativeOrder::Value => String::new(),
            ScoreAdmission::None => {
                "pub(super) fn __score_valid(_: &[f64;DIMENSION])->bool { true }".into()
            }
            ScoreAdmission::Tensor(program) => {
                let source = &program.source;
                let count = program.workspace.elements;
                format!(
                    "mod __tensor_admission {{ use super::DIMENSION; {source}\n\
                    pub(super) fn valid(q:&[f64;DIMENSION])->bool {{\
                    let mut score=[0.0;DIMENSION];\
                    let value=__gradient(q,&mut score,&mut [0.0;{count}],&mut [0.0;{count}],1.0,&mut __first_context());\
                    value.is_finite() && score.iter().all(|v|v.is_finite()) }} }}\n\
                    pub(super) fn __score_valid(q:&[f64;DIMENSION])->bool {{__tensor_admission::valid(q)}}"
                )
            }
        }
    }
    fn imports(&self) -> &'static str {
        match self.request.order {
            DerivativeOrder::Value => "use __enzyme::__log_density;",
            DerivativeOrder::First => "use __enzyme::{__gradient, __log_density, __score_valid};",
            DerivativeOrder::Hvp => {
                "use __enzyme::{__gradient, __hvp, __log_density, __score_valid};"
            }
            DerivativeOrder::Third => {
                "use __enzyme::{__gradient, __hvp, __hvp_jvp, __log_density, __score_valid};"
            }
        }
    }
    fn target_trait(&self) -> &'static str {
        match self.request.order {
            DerivativeOrder::First => "::alea::target::LogDensityGradient",
            DerivativeOrder::Hvp => "::alea::capability::PotentialHessianVector",
            DerivativeOrder::Third => "::alea::capability::PotentialHessianDerivative",
            DerivativeOrder::Value => unreachable!("value-only plan has no target"),
        }
    }
    fn render_target(&self) -> String {
        if self.request.order == DerivativeOrder::Value {
            String::new()
        } else {
            render_target(self)
        }
    }
    fn render_higher(&self) -> String {
        if self.higher() {
            render_higher(self)
        } else {
            String::new()
        }
    }
    fn render(&self) -> String {
        render_module(self)
    }
}

fn render_module(plan: &GenerationPlan) -> String {
    format!(
        r#"// Source data: {data_mode}.

pub const DIMENSION: usize = {dimension};
/// Whether first-order gradients use explicit segmented tensor replay.
/// Higher derivatives still use the compiler-owned reverse tape.
pub const SOURCE_CHECKPOINTED: bool = {checkpointed};
/// Whether sampling uses an explicit first-order pullback rather than compiler reverse.
pub const SOURCE_EXPLICIT_FIRST_ORDER: bool = {explicit_first_order};
/// Whether a compiler-only tensor gradient also runs checked pullback admission.
/// This is extra first-order work, not a switch of the returned derivative route.
pub const SOURCE_TENSOR_SCORE_VALIDATION: bool = {tensor_score_validation};

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
mod __enzyme {{
use super::DIMENSION;

{score_validation}

{first_order_source}
{reverse_entry}
// Numeric literals serialize model data/bounds exactly; they are not approximations
// to named mathematical constants, even when their decimal representations match.
#[allow(clippy::approx_constant)]
pub(super) fn __log_density_work(q: &[f64; DIMENSION], scratch: &mut [f64; {workspace}]) -> f64 {{
    let _ = &scratch;
    {primal}
}}

pub(super) fn __log_density(q: &[f64; DIMENSION]) -> f64 {{
    __log_density_work(q, &mut [0.0; {workspace}])
}}

{higher_entries}
}}
{imports}

fn __valid(q: &[f64; DIMENSION]) -> bool {{
    let _ = q;
    {validation}
}}

{first_order_valid}
pub fn {public_value}(q: &[f64; DIMENSION]) -> f64 {{
    if q.iter().any(|x| !x.is_finite()) || !__valid(q) {{ f64::NAN }} else {{ __log_density(q) }}
}}

{target}
"#,
        data_mode = match plan.data {
            SourceData::Frozen => "frozen at generation",
        },
        dimension = plan.dimension,
        checkpointed = plan.first.as_ref().is_some_and(|f| f.checkpointed),
        explicit_first_order = plan.first.is_some(),
        tensor_score_validation = matches!(plan.admission, ScoreAdmission::Tensor(_)),
        score_validation = plan.render_admission(),
        first_order_source = plan
            .first
            .as_ref()
            .map_or("", |f| f.program.source.as_str()),
        reverse_entry = plan.reverse_declaration(),
        workspace = plan.workspace.elements,
        primal = &plan.primal.source,
        higher_entries = plan.render_higher(),
        imports = plan.imports(),
        validation = &plan.validation,
        first_order_valid = if plan.higher() {
            render_first_order_valid(plan)
        } else {
            String::new()
        },
        public_value = if plan.request.mode == SourceMode::Density {
            "log_density"
        } else {
            "value"
        },
        target = plan.render_target(),
    )
}

fn render_higher(plan: &GenerationPlan) -> String {
    format!(
        r#"#[::std::autodiff::autodiff_forward(__gradient_jvp, Dual, Dual, Dual, Dual, Const)]
pub(super) fn __gradient_body(q: &[f64; DIMENSION], g: &mut [f64; DIMENSION], work: &mut [f64; {workspace}], reverse: &mut [f64; {workspace}]) -> f64 {{
    {curvature_gradient}(q, g, work, reverse, 1.0)
}}

// The direction needs an explicit zero tangent at the call site. Marking this
// reference Const makes the pinned compiler's nested AD of scans inject a
// spurious Hessian term, even for a zero position direction.
{third_attribute}
pub(super) fn __hvp(q: &[f64; DIMENSION], v: &[f64; DIMENSION], g: &mut [f64; DIMENSION], h: &mut [f64; DIMENSION], work: &mut [f64; {workspace}], reverse: &mut [f64; {workspace}], tangent: &mut [f64; {workspace}], reverse_tangent: &mut [f64; {workspace}]) -> f64 {{
    __gradient_jvp(q, v, g, h, work, tangent, reverse, reverse_tangent)
}}"#,
        workspace = plan.workspace.elements,
        curvature_gradient = if plan.first.is_some() {
            "__higher_gradient"
        } else {
            "__gradient"
        },
        third_attribute = if plan.third() { THIRD_ATTRIBUTE } else { "" },
    )
}

const THIRD_ATTRIBUTE: &str = r#"#[::std::autodiff::autodiff_forward(__hvp_jvp, Dual, Dual, Dual, Dual, Dual, Dual, Dual, Dual, Const)]"#;

fn render_first_order_valid(plan: &GenerationPlan) -> String {
    format!(
        r#"fn __first_order_valid(q: &[f64; DIMENSION]{context_parameter}) -> bool {{
    if !__score_valid(q) {{ return false; }}
    let mut score = [0.0; DIMENSION];
    let value = __gradient(q, &mut score, &mut [0.0; {first_workspace}], &mut [0.0; {first_workspace}], 1.0{context_argument});
    value.is_finite() && score.iter().all(|v| v.is_finite())
}}

"#,
        first_workspace = plan.first_workspace(),
        context_parameter = if plan.stateful() {
            ",context:&mut __enzyme::__SourceContext"
        } else {
            ""
        },
        context_argument = if plan.stateful() {
            ",context"
        } else if plan.first.is_some() {
            ",&mut __enzyme::__first_context()"
        } else {
            ""
        },
    )
}

fn render_target(plan: &GenerationPlan) -> String {
    format!(
        r#"pub fn {public_target}() -> impl {target_trait}<Error = ::alea::autodiff::autodiff::AutodiffError> {{
    {state_setup}
    ::alea::autodiff::autodiff::EnzymeHigherOrderTarget::new(
        DIMENSION,
        move |q: &[f64], g: &mut [f64], seed: f64| {{
            let q = q.try_into().expect("validated position");
            if !__valid(q) || !__score_valid(q) {{ g.fill(f64::NAN); return f64::NAN; }}
            __gradient(q, g.try_into().expect("validated gradient"), &mut [0.0; {first_workspace}], &mut [0.0; {first_workspace}], seed{context_argument})
        }},
        {hvp_callback},
        {third_callback},

    )
}}
"#,
        public_target = if plan.request.mode == SourceMode::Density {
            "target"
        } else {
            "objective"
        },
        target_trait = plan.target_trait(),
        state_setup = plan.target_state(),
        context_argument = plan.first_context_argument("__for_gradient"),
        first_workspace = plan.first_workspace(),
        hvp_callback = if plan.higher() {
            render_hvp_callback(plan)
        } else {
            "()".into()
        },
        third_callback = if plan.third() {
            render_third_callback(plan)
        } else {
            "()".into()
        },
    )
}

fn render_hvp_callback(plan: &GenerationPlan) -> String {
    format!(
        r#"move |q: &[f64], v: &[f64], out: &mut [f64]| {{
            let q = q.try_into().expect("validated position");
            // A zero directional result can hide an undefined gradient, and
            // AD-produced side outputs need not preserve first-order admission.
            if !__valid(q) || !{first_valid} {{ out.fill(f64::NAN); return; }}
            let [work, reverse, tangent, reverse_tangent] = &mut [[0.0; {workspace}]; 4];
            let value = __hvp(q, v.try_into().expect("validated vector"), &mut [0.0; DIMENSION], out.try_into().expect("validated product"), work, reverse, tangent, reverse_tangent);
            if value.is_finite() {{ for x in out {{ *x = -*x; }} }} else {{ out.fill(f64::NAN); }}
        }}"#,
        workspace = plan.workspace.elements,
        first_valid = plan.first_valid_call("__for_hvp"),
    )
}

fn render_third_callback(plan: &GenerationPlan) -> String {
    format!(
        r#"move |q: &[f64], hessian: &mut [f64], third: &mut [f64]| {{
            let q = q.try_into().expect("validated position");
            if !__valid(q) || !{first_valid} {{ hessian.fill(f64::NAN); third.fill(f64::NAN); return; }}
            let mut v = [0.0; DIMENSION];
            let mut w = [0.0; DIMENSION];
            let mut h = [0.0; DIMENSION];
            let mut t = [0.0; DIMENSION];
            for j in 0..DIMENSION {{
                v[j] = 1.0;
                for k in 0..DIMENSION {{
                    w[k] = 1.0;
                    h.fill(0.0);
                    t.fill(0.0);
                    let [work, work_jvp, reverse, reverse_jvp, tangent, tangent_jvp, reverse_tangent, reverse_tangent_jvp] = &mut [[0.0; {workspace}]; 8];
                    let value = __hvp_jvp(q, &w, &v, &[0.0; DIMENSION], &mut [0.0; DIMENSION], &mut [0.0; DIMENSION], &mut h, &mut t, work, work_jvp, reverse, reverse_jvp, tangent, tangent_jvp, reverse_tangent, reverse_tangent_jvp);
                    if !value.is_finite() {{ hessian.fill(f64::NAN); third.fill(f64::NAN); return; }}
                    for i in 0..DIMENSION {{
                        if k == 0 {{ hessian[i * DIMENSION + j] = -h[i]; }}
                        third[k * DIMENSION * DIMENSION + i * DIMENSION + j] = -t[i];
                    }}
                    w[k] = 0.0;
                }}
                v[j] = 0.0;
            }}
        }}"#,
        workspace = plan.workspace.elements,
        first_valid = plan.first_valid_call("__for_third"),
    )
}
