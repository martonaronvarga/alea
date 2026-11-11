//! Cold source lowering. No graph, callbacks or reverse tape survives in the primal.
use super::{AssemblyError, Executable, Op};
use std::fmt::Write;
mod checkpoint;

/// Highest derivative entry point requested from source generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeOrder {
    Value,
    First,
    Hvp,
    Third,
}

/// Scalar being differentiated, including the choice of chart Jacobian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceMode {
    Density,
    Objective,
}

/// Explicit curvature route. First-order rules never imply curvature support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureRoute {
    /// No curvature route has been selected or qualified by the caller.
    Unavailable,
    /// Retain the original compiler reverse path, independently of trusted or
    /// checkpointed first-order rules. Requires actual downstream qualification;
    /// selecting this route is not a claim about arbitrary compiler programs.
    OriginalCompilerReverse,
}

/// Source entry points to emit. Lower-order requests do not emit higher entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceRequest {
    pub mode: SourceMode,
    pub order: DerivativeOrder,
    pub curvature: CurvatureRoute,
}

/// Fixed, explicitly typed mutable f64 source scratch. This excludes compiler tapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatWorkspace {
    pub elements: usize,
}

/// Lowered Rust leaf and its checked fixed f64 scratch requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceProgram {
    pub source: String,
    pub workspace: FloatWorkspace,
}

/// Checked factor-local first-order composition and its replay receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstOrderSource {
    pub program: SourceProgram,
    pub checkpointed: bool,
    /// Whether generated native helpers require per-target typed state.
    pub stateful: bool,
}

impl super::SourceCapabilities {
    /// Checks source availability and explicit derivative-route selection without
    /// evaluating callbacks, rendering source, or invoking an external compiler.
    /// This checks emission eligibility, not numerical/toolchain qualification.
    /// # Errors
    /// Reports the first active missing primal, or an unqualified curvature order.
    pub fn resolve(&self, request: SourceRequest) -> Result<SourceRequest, AssemblyError> {
        if let Some(value) = self.missing_primals.first() {
            return Err(AssemblyError::MissingPrimal { node: value.node });
        }
        if matches!(request.order, DerivativeOrder::Hvp | DerivativeOrder::Third)
            && request.curvature == CurvatureRoute::Unavailable
        {
            return Err(AssemblyError::UnqualifiedDerivative {
                order: request.order,
            });
        }
        Ok(request)
    }
}

impl Executable {
    pub(super) fn has_primal(&self, node: usize) -> bool {
        self.generated_primals
            .binary_search_by_key(&node, |(i, _)| *i)
            .is_ok()
            || self
                .primals
                .binary_search_by_key(&node, |(i, _)| *i)
                .is_ok()
    }
    pub(super) fn primal_source(&self, node: usize) -> Option<std::borrow::Cow<'_, str>> {
        self.generated_primals
            .binary_search_by_key(&node, |(i, _)| *i)
            .ok()
            .map(|i| std::borrow::Cow::Owned((self.generated_primals[i].1)()))
            .or_else(|| {
                self.primals
                    .binary_search_by_key(&node, |(i, _)| *i)
                    .ok()
                    .map(|i| std::borrow::Cow::Borrowed(self.primals[i].1))
            })
    }
    /// Emits the body of an unconstrained Rust scalar log-density function.
    ///
    /// The caller supplies `q: &[f64; layout().unconstrained_dimension()]` and a
    /// dependency named `alea`. The body includes every parameter transform
    /// and exactly one log Jacobian per block. Data/plates are specialized at
    /// source-generation time. Invalid numerical evaluations return NaN. Pair
    /// with [`Self::rust_validation`] before evaluation to retain the rounded-SPD
    /// domain check without differentiating a numerical admission algorithm.
    ///
    /// Use `alea::autodiff::codegen::enzyme_source` to emit a complete generated
    /// target for inclusion from a build script on the pinned Enzyme compiler.
    ///
    /// # Errors
    /// Returns [`AssemblyError::MissingPrimal`] for any active opaque node without an
    /// explicit Rust primal. There is no analytic or finite-difference fallback.
    pub fn rust_primal(&self) -> Result<String, AssemblyError> {
        self.rust_local_primal(true)
    }

    /// Emits the scalar composed objective f(T(q)), with no log-Jacobian code.
    /// The active factor sum is retained without changing its statistical selection.
    /// # Errors
    /// Rejects opaque nodes without an explicit primal, as `rust_primal` does.
    pub fn rust_objective_primal(&self) -> Result<String, AssemblyError> {
        self.rust_local_primal(false)
    }

    fn rust_local_primal(&self, density: bool) -> Result<String, AssemblyError> {
        let program = self.rust_mapped_primal(density)?;
        Ok(format!(
            "let mut scratch = [0.0_f64; {}];\n{}",
            program.workspace.elements, program.source
        ))
    }

    /// Lowers a density or objective primal with checked fixed f64 scratch.
    /// The caller supplies `q` and `scratch` with the returned extent. Data is
    /// frozen into source. Scratch is overwritten and must have a zero shadow
    /// when compiler-differentiated. No callbacks or reverse tape survive here.
    /// # Errors
    /// Rejects missing primals or overflowing combined scratch storage.
    pub fn source_primal(&self, mode: SourceMode) -> Result<SourceProgram, AssemblyError> {
        self.rust_mapped_primal(mode == SourceMode::Density)
    }

    fn rust_mapped_primal(&self, density: bool) -> Result<SourceProgram, AssemblyError> {
        let mut source = String::new();
        let mut workspace = 0_usize;
        if density {
            source.push_str("let mut jacobian = 0.0_f64;\n");
        }
        for block in self.layout.blocks() {
            source.push_str(&block.transform().rust_primal(
                block.unconstrained_range().start,
                block.constrained_range().start,
                density,
            ));
        }
        for (i, op) in self.nodes.iter().enumerate() {
            if self.active_nodes.as_ref().is_some_and(|active| !active[i]) {
                continue;
            }
            let expression = match op {
                Op::Parameter(slot) => format!("x{slot}"),
                Op::Constant(value) => format!("{value:?}_f64"),
                Op::Add(a, b) => format!("n{a} + n{b}"),
                Op::Multiply(a, b) => format!("n{a} * n{b}"),
                Op::Exp(a) => format!("n{a}.exp()"),
                Op::Log(a) => format!("n{a}.ln()"),
                Op::Region(_) => return Err(AssemblyError::MissingPrimal { node: i }),
                Op::Factor { inputs, .. } => {
                    let primal = self
                        .primal_source(i)
                        .ok_or(AssemblyError::MissingPrimal { node: i })?;
                    let size = self
                        .primitives
                        .binary_search_by_key(&i, |(node, _)| *node)
                        .ok()
                        .map_or(0, |j| self.primitives[j].1.source_workspace);
                    let start = workspace;
                    let arguments = inputs.len();
                    let size = size.checked_add(arguments).ok_or(AssemblyError::Shape)?;
                    workspace = workspace
                        .checked_add(size)
                        .filter(|n| *n <= isize::MAX as usize / size_of::<f64>())
                        .ok_or(AssemblyError::Shape)?;
                    writeln!(source,
                        "fn factor{i}(x: &[f64; {}], scratch: &mut [f64]) -> f64 {{ let _ = &scratch; {primal} }}",
                        inputs.len()).unwrap();
                    for (argument, node) in inputs.iter().enumerate() {
                        let offset = start + argument;
                        writeln!(source, "scratch[{offset}]=n{node};").unwrap();
                    }
                    format!(
                        "{{ let (args, work)=scratch[{start}..{workspace}].split_at_mut({arguments}); factor{i}((&*args).try_into().expect(\"fixed arguments\"), work) }}"
                    )
                }
            };
            writeln!(
                source,
                "let n{i} = {expression};\nif !n{i}.is_finite() {{ return f64::NAN; }}"
            )
            .unwrap();
        }
        source.push_str("let mut total = 0.0_f64;\n");
        for i in self.retained_factors() {
            writeln!(source, "total += n{i};").unwrap();
        }
        source.push_str(if density {
            "total + jacobian\n"
        } else {
            "total\n"
        });
        Ok(SourceProgram {
            source,
            workspace: FloatWorkspace {
                elements: workspace,
            },
        })
    }

    /// Emits a boolean function body for cold/opaque numerical admission checks.
    ///
    /// The caller supplies the same `q` as [`Self::rust_primal`]. Call this before
    /// any generated derivative; it preserves the original rounded covariance
    /// positive-definiteness checks outside AD. Other transforms validate their
    /// support in the primal itself. This body performs no model/score evaluation.
    pub fn rust_validation(&self) -> String {
        self.rust_mapped_validation(true)
    }

    /// Emits rounded-domain admission for `rust_objective_primal` without
    /// requesting a density-change determinant.
    pub fn rust_objective_validation(&self) -> String {
        self.rust_mapped_validation(false)
    }

    fn rust_mapped_validation(&self, density: bool) -> String {
        let mut source = String::new();
        for block in self.layout.blocks() {
            if let Some(validation) = block.transform().rust_validation(density) {
                let input = block.unconstrained_range();
                writeln!(
                    source,
                    "{{ let q = &q[{}..{}]; {validation} }}",
                    input.start, input.end
                )
                .unwrap();
            }
        }
        source.push_str("true");
        source
    }
}
