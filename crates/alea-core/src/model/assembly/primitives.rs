//! Semantic primitives lowered to the existing scalar analytic execution tape.
//!
//! Descriptors and signatures are cold immutable data. Their presence is not a
//! proof of normalization, a generative ordering, or support for another backend.
use super::{AssemblyError, CompiledModel, Expr, ModelBuilder, Op};
use crate::distribution::{EventShape, ReferenceMeasure, Support};
use std::{any::Any, sync::Arc};

/// Logical argument shape, distinct from probability event and batch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArgumentShape {
    /// One scalar expression.
    Scalar,
    /// Ordered scalar expressions, including an empty coefficient vector.
    Vector(usize),
    /// Full row-major scalar expressions.
    Matrix { rows: usize, columns: usize },
}
impl ArgumentShape {
    fn elements(self) -> Option<usize> {
        match self {
            Self::Scalar => Some(1),
            Self::Vector(n) => Some(n),
            Self::Matrix { rows, columns } => rows.checked_mul(columns),
        }
    }
}
/// One named logical argument. Arguments partition the primitive's flattened
/// expression inputs in declaration order; constants/data may live in its descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveArgument {
    /// Semantic argument name, not a model namespace or Rust identifier.
    pub name: &'static str,
    /// Storage shape of this argument.
    pub shape: ArgumentShape,
}
/// Declared probability semantics of a scalar log-probability reduction.
/// Event measure/support describe each event, not a parameter chart. Batch axes
/// do not by themselves declare conditional independence or leave-out units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbabilitySemantics {
    /// Shape of one event, before batch reduction.
    pub event: EventShape,
    /// Ordered batch dimensions; empty denotes one event.
    pub batch: Vec<usize>,
    /// Support of an event, not of every expression argument.
    pub support: Support,
    /// Reference measure of one event.
    pub measure: ReferenceMeasure,
}
/// Mathematical signature, retained independently of executable callbacks.
/// A signature is a declaration by the primitive author, not an independent
/// verification of arbitrary user implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveSignature {
    /// Human-readable operation name. Typed identity is available by downcasting.
    pub name: &'static str,
    /// Logical arguments in flattened input order.
    pub arguments: Vec<PrimitiveArgument>,
    /// Present for a declared log-probability primitive; absent for deterministic
    /// computations or arbitrary potentials with no probability interpretation.
    pub probability: Option<ProbabilitySemantics>,
}
impl PrimitiveSignature {
    fn validate(&self, inputs: usize) -> Result<(), AssemblyError> {
        let mut count = 0_usize;
        if self.name.is_empty() {
            return Err(AssemblyError::Name);
        }
        for (i, argument) in self.arguments.iter().enumerate() {
            if argument.name.is_empty()
                || self.arguments[..i].iter().any(|a| a.name == argument.name)
            {
                return Err(AssemblyError::Name);
            }
            count = count
                .checked_add(argument.shape.elements().ok_or(AssemblyError::Shape)?)
                .ok_or(AssemblyError::Shape)?;
        }
        if count != inputs {
            return Err(AssemblyError::Shape);
        }
        if let Some(probability) = &self.probability {
            probability.batch.iter().try_fold(1_usize, |n, &axis| {
                n.checked_mul(axis).ok_or(AssemblyError::Shape)
            })?;
        }
        Ok(())
    }
}
/// Immutable mathematical operation with fused value/local-score evaluation.
///
/// Implementations retain their typed arguments and data, and lower through the
/// same checked tape as custom callbacks. Inputs/semantics must stay fixed.
/// `evaluate` must overwrite every derivative and depend only on inputs and
/// immutable data, never previous scratch contents. No sampling, pointwise or
/// higher-derivative capability follows automatically from this trait.
pub trait Primitive: Send + Sync + 'static {
    /// Per-execution mutable scratch; never stored in the semantic descriptor.
    type Workspace: Send + Sync + 'static;
    /// Flattened constrained expression arguments.
    fn inputs(&self) -> &[Expr];
    /// Logical mathematical meaning of the arguments and scalar result.
    fn signature(&self) -> PrimitiveSignature;
    /// Allocates/initializes scratch at construction, not during evaluation.
    fn workspace(&self) -> Self::Workspace;
    /// Scratch for conditional row reporting. Defaults to ordinary evaluation
    /// scratch; primitives whose row rule needs less storage may specialize it.
    fn batch_workspace(&self) -> Self::Workspace {
        self.workspace()
    }
    /// Fused scalar value and local derivatives in flattened input order.
    /// # Errors
    /// Preserves the primitive's numerical and structural failures as typed errors.
    fn evaluate(
        &self,
        inputs: &[f64],
        gradient: &mut [f64],
        workspace: &mut Self::Workspace,
    ) -> Result<f64, AssemblyError>;
    /// Optional trusted Rust primal, with the same contract as
    /// [`ModelBuilder::custom_with_primal`]. This is not a generated HVP promise.
    fn rust_primal(&self) -> Option<&'static str> {
        None
    }
    /// Optional lazy data-specialized Rust source rule. Overrides the static
    /// source; called only during export, never retained evaluation/construction.
    fn source_rule(&self) -> Option<fn(&Self) -> String>
    where
        Self: Sized,
    {
        None
    }
    /// Number of mutable `f64` scratch elements required by the Rust source.
    /// When nonzero, the source receives `scratch: &mut [f64]` of this length.
    /// It must initialize every element before reading it, independently of prior
    /// calls. Export keeps this buffer explicit so compiler AD knows its type.
    /// This is separate from the native evaluation workspace.
    fn source_workspace(&self) -> usize {
        0
    }
    /// Explicit conditional pointwise decomposition, never inferred from shape.
    fn batch_rule(&self) -> Option<BatchRule<Self>>
    where
        Self: Sized,
    {
        None
    }
}
/// Author-supplied conditional pointwise decomposition of a summed primitive.
pub type BatchEvaluator<P> =
    fn(&P, &[f64], &mut [f64], &mut <P as Primitive>::Workspace) -> Result<(), AssemblyError>;
/// A fixed-size rule for a primitive's conditional row values.
pub struct BatchRule<P: Primitive> {
    /// Number of entries in fixed row order.
    pub units: usize,
    /// Writes every entry. Reporting stages/validates outputs before publication.
    pub evaluate: BatchEvaluator<P>,
}
impl<P: Primitive> BatchRule<P> {
    pub(super) fn validate(&self) -> Result<(), AssemblyError> {
        if self.units > isize::MAX as usize / size_of::<super::BatchUnit>() {
            return Err(AssemblyError::Shape);
        }
        Ok(())
    }
}
/// Cold metadata with recoverable typed descriptor and immutable captured data.
pub struct PrimitiveMetadata {
    pub(super) signature: PrimitiveSignature,
    pub(super) descriptor: Arc<dyn Any + Send + Sync>,
    pub(super) source_workspace: usize,
}
impl PrimitiveMetadata {
    /// Validated argument layout and declared probability semantics.
    pub fn signature(&self) -> &PrimitiveSignature {
        &self.signature
    }
    /// Recovers a concrete descriptor without interpreting source strings or
    /// guessing an operation from its user-supplied factor name.
    pub fn descriptor<P: Primitive>(&self) -> Option<&P> {
        self.descriptor.downcast_ref()
    }
}
impl ModelBuilder {
    /// Validates and lowers an immutable semantic primitive using existing
    /// callbacks and chain-local scratch. Retains its typed descriptor outside
    /// the hot operation tape; no extra runtime graph traversal is introduced.
    /// No prior, likelihood role, transform or Jacobian is inferred.
    /// # Errors
    /// Rejects foreign handles, malformed argument layouts/names, or overflowing
    /// storage before adding a node. Workspace construction panics propagate.
    pub fn primitive<P: Primitive>(&mut self, primitive: P) -> Result<Expr, AssemblyError> {
        let signature = primitive.signature();
        signature.validate(primitive.inputs().len())?;
        for &input in primitive.inputs() {
            self.index(input)?;
        }
        let primal = primitive.rust_primal();
        let source = primitive.source_rule();
        let source_workspace = primitive.source_workspace();
        if source_workspace > isize::MAX as usize / size_of::<f64>() {
            return Err(AssemblyError::Shape);
        }
        let batch = primitive.batch_rule();
        if let Some(rule) = &batch {
            rule.validate()?;
        }
        let batch = batch.map(|rule| (rule, primitive.batch_workspace()));
        let mut workspace = primitive.workspace();
        let descriptor = Arc::new(primitive);
        let executable = Arc::clone(&descriptor);
        let expression = self.custom_node(
            descriptor.inputs(),
            move |x, g| executable.evaluate(x, g, &mut workspace),
            primal,
        )?;
        if let Some(source) = source {
            let descriptor = Arc::clone(&descriptor);
            self.generated_primals
                .push((expression.node, Box::new(move || source(&descriptor))));
        }
        if let Some((rule, mut workspace)) = batch {
            let descriptor = Arc::clone(&descriptor);
            self.batch_rules.push((
                expression.node,
                rule.units,
                Box::new(move |x, out| (rule.evaluate)(&descriptor, x, out, &mut workspace)),
            ));
        }
        self.primitives.push((
            expression.node,
            PrimitiveMetadata {
                signature,
                descriptor,
                source_workspace,
            },
        ));
        Ok(expression)
    }
}
/// Scalar execution operation identity. Logical array arguments and probability
/// semantics, when available, are retained separately in [`GraphNode::primitive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationKind {
    /// Constrained coordinate slot.
    Parameter(usize),
    /// Literal scalar value.
    Constant,
    /// Binary sum.
    Add,
    /// Binary product.
    Multiply,
    /// Scalar exponential.
    Exp,
    /// Scalar logarithm.
    Log,
    /// Registered mathematical primitive.
    Primitive,
    /// Custom callback without a semantic descriptor.
    Opaque,
}
/// Read-only cold graph snapshot entry. Inspecting the graph allocates input
/// handle vectors; retained model evaluation never constructs these entries.
pub struct GraphNode<'a> {
    /// Owner-checked handle usable when matching descriptor inputs to nodes.
    pub expression: Expr,
    /// Execution operation.
    pub operation: OperationKind,
    /// Arguments in original order, including repeated shared expressions.
    pub inputs: Vec<Expr>,
    /// Literal value for a constant node, absent otherwise.
    pub constant: Option<f64>,
    /// Semantic identity/data for a registered primitive; absent for opaque nodes.
    pub primitive: Option<&'a PrimitiveMetadata>,
    /// Factor declarations rooted at this expression, preserving declaration
    /// order and multiplicity, including declarations excluded by selection.
    pub factors: Vec<&'a super::FactorMetadata>,
    /// Whether this node is included in the current execution view. The generic
    /// all-factor view executes even unattached expressions, matching the tape.
    pub active: bool,
}
/// Source-export capability of the current execution view. This does not assert
/// that an external compiler is installed or that any derivative backend succeeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceCapabilities {
    /// Active callback nodes missing a trusted Rust primal, in graph order.
    pub missing_primals: Vec<Expr>,
}
impl SourceCapabilities {
    /// All active nodes have source-lowering support.
    pub fn supports_rust_primal(&self) -> bool {
        self.missing_primals.is_empty()
    }
}
impl CompiledModel {
    /// Inspects all declared nodes without evaluating callbacks or copying captured
    /// datasets. Statistical selections preserve descriptors for inactive nodes.
    pub fn graph(&self) -> impl Iterator<Item = GraphNode<'_>> {
        self.nodes.iter().enumerate().map(|(node, op)| {
            let primitive = self
                .primitives
                .binary_search_by_key(&node, |(i, _)| *i)
                .ok()
                .map(|i| &self.primitives[i].1);
            let (operation, inputs, constant) = match op {
                Op::Parameter(slot) => (OperationKind::Parameter(*slot), vec![], None),
                Op::Constant(value) => (OperationKind::Constant, vec![], Some(*value)),
                Op::Add(a, b) => (OperationKind::Add, vec![*a, *b], None),
                Op::Multiply(a, b) => (OperationKind::Multiply, vec![*a, *b], None),
                Op::Exp(a) => (OperationKind::Exp, vec![*a], None),
                Op::Log(a) => (OperationKind::Log, vec![*a], None),
                Op::Factor { inputs, .. } => (
                    if primitive.is_some() {
                        OperationKind::Primitive
                    } else {
                        OperationKind::Opaque
                    },
                    inputs.to_vec(),
                    None,
                ),
            };
            GraphNode {
                expression: Expr {
                    owner: self.owner,
                    node,
                },
                operation,
                inputs: inputs
                    .into_iter()
                    .map(|node| Expr {
                        owner: self.owner,
                        node,
                    })
                    .collect(),
                constant,
                primitive,
                factors: self
                    .factors
                    .iter()
                    .zip(&self.factor_metadata)
                    .filter_map(|(root, metadata)| (*root == node).then_some(metadata))
                    .collect(),
                active: self.active_nodes.as_ref().is_none_or(|active| active[node]),
            }
        })
    }
    /// Reports missing source primals for the selected graph without generating
    /// source or touching evaluation scratch. Actual export retains typed errors.
    pub fn source_capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            missing_primals: self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(node, op)| {
                    (matches!(op, Op::Factor { .. })
                        && self.active_nodes.as_ref().is_none_or(|active| active[node])
                        && !self.has_primal(node))
                    .then_some(Expr {
                        owner: self.owner,
                        node,
                    })
                })
                .collect(),
        }
    }
}
