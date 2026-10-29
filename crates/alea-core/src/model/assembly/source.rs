//! Cold source lowering. No graph, callbacks or reverse tape survives in the primal.
use super::{AssemblyError, CompiledModel, Op};
use std::fmt::Write;

impl CompiledModel {
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
    /// Returns [`AssemblyError::MissingPrimal`] for any opaque node without an
    /// explicit Rust primal. There is no analytic or finite-difference fallback.
    pub fn rust_primal(&self) -> Result<String, AssemblyError> {
        self.rust_mapped_primal(true)
    }

    /// Emits the scalar composed objective f(T(q)), with no log-Jacobian code.
    /// All factors are retained; this does not classify them as likelihoods.
    /// # Errors
    /// Rejects opaque nodes without an explicit primal, as `rust_primal` does.
    pub fn rust_objective_primal(&self) -> Result<String, AssemblyError> {
        self.rust_mapped_primal(false)
    }

    fn rust_mapped_primal(&self, density: bool) -> Result<String, AssemblyError> {
        let mut source = String::new();
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
            let expression = match op {
                Op::Parameter(slot) => format!("x{slot}"),
                Op::Constant(value) => format!("{value:?}_f64"),
                Op::Add(a, b) => format!("n{a} + n{b}"),
                Op::Multiply(a, b) => format!("n{a} * n{b}"),
                Op::Exp(a) => format!("n{a}.exp()"),
                Op::Log(a) => format!("n{a}.ln()"),
                Op::Factor { inputs, .. } => {
                    let index = self
                        .primals
                        .binary_search_by_key(&i, |&(node, _)| node)
                        .map_err(|_| AssemblyError::MissingPrimal { node: i })?;
                    let primal = self.primals[index].1;
                    writeln!(
                        source,
                        "fn factor{i}(x: &[f64; {}]) -> f64 {{ {primal} }}",
                        inputs.len()
                    )
                    .unwrap();
                    let inputs = inputs
                        .iter()
                        .map(|j| format!("n{j}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("factor{i}(&[{inputs}])")
                }
            };
            writeln!(
                source,
                "let n{i} = {expression};\nif !n{i}.is_finite() {{ return f64::NAN; }}"
            )
            .unwrap();
        }
        source.push_str("let mut total = 0.0_f64;\n");
        for i in &self.factors {
            writeln!(source, "total += n{i};").unwrap();
        }
        source.push_str(if density {
            "total + jacobian\n"
        } else {
            "total\n"
        });
        Ok(source)
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
