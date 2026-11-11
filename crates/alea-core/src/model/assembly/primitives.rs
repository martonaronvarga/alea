//! Semantic primitives lowered to the existing scalar analytic execution tape.
//!
//! Descriptors and signatures are cold immutable data. Their presence is not a
//! proof of normalization, a generative ordering, or support for another backend.
use super::{AssemblyError, ModelBuilder, Op, Value};
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
/// Graph-independent operation descriptor with a typed native rule bundle.
/// Descriptors contain immutable configuration/data, never model value handles.
/// Graph calls supply their ordered edges separately. Rules declare capabilities;
/// native first-order evaluation implies no source, JVP or higher derivatives.
pub trait OperationDescriptor: Send + Sync + Sized + 'static {
    /// Execution-local mutable scratch, never part of shared descriptor data.
    type Workspace: Send + 'static;
    /// Resolves immutable signature and optional rules without evaluating data or
    /// allocating execution workspaces. Source generation remains lazy.
    fn rules(&self) -> NativeRules<Self>;
}
/// Fused native scalar value/local-score rule. Every score must be overwritten;
/// admission and staged publication remain the executor's responsibility.
pub type NativeEvaluator<D> = fn(
    &D,
    &[f64],
    &mut [f64],
    &mut <D as OperationDescriptor>::Workspace,
) -> Result<f64, AssemblyError>;
/// Typed rules erased once after graph/layout validation. The fused native rule
/// uses existing typed workspaces and must preserve all local score checks,
/// including zero downstream cotangents. Optional fields are the capability
/// authority; no support booleans or inferred probability/replay guarantees.
pub struct NativeRules<D: OperationDescriptor> {
    /// Argument layout and declared probability semantics.
    pub signature: PrimitiveSignature,
    /// Allocates native scratch after structural validation.
    pub workspace: fn(&D) -> D::Workspace,
    /// Allocates reporting scratch only when a reporting rule is present.
    pub batch_workspace: fn(&D) -> D::Workspace,
    /// Fused value and local scores in graph-call input order.
    pub evaluate: NativeEvaluator<D>,
    /// Optional pure scalar preparation; never an admission waiver.
    pub preparation: Option<ScalarPreparation<D>>,
    /// Explicit conditional row reporting.
    pub batch: Option<BatchRule<D>>,
    /// Trusted source lowering, independently optional from native evaluation.
    pub source: Option<SourceRule<D>>,
}
impl<D: OperationDescriptor> NativeRules<D> {
    /// Creates native-only rules. Reporting uses the ordinary workspace factory
    /// unless explicitly specialized. No source or replay capability is inferred.
    pub fn new(
        signature: PrimitiveSignature,
        workspace: fn(&D) -> D::Workspace,
        evaluate: NativeEvaluator<D>,
    ) -> Self {
        Self {
            signature,
            workspace,
            batch_workspace: workspace,
            evaluate,
            preparation: None,
            batch: None,
            source: None,
        }
    }
}
/// Trusted Rust source body, rendered only when source export is requested.
pub enum SourceBody<D> {
    /// Static application-supplied body, with the custom-primal trust contract.
    Static(&'static str),
    /// Immutable-data-specialized body; not called by registration/evaluation.
    Generated(fn(&D) -> String),
}
impl<D> SourceBody<D> {
    /// Renders the trusted body. This does not validate arbitrary Rust or claim
    /// differentiation support on any particular compiler.
    pub fn render(&self, descriptor: &D) -> String {
        match self {
            Self::Static(body) => (*body).into(),
            Self::Generated(render) => render(descriptor),
        }
    }
}
/// Optional trusted source capability. The ordinary primal remains the source
/// of compiler higher derivatives; a supplied first-order rule is never
/// differentiated for curvature. Every call initializes its own float scratch.
/// Explicit typed state owned by one generated target, never a thread/global cache.
/// Rust type and initializer are trusted scoped leaves, like source rule bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceState {
    pub rust_type: String,
    pub initializer: String,
}
/// First-order body and optional per-target state. Bodies receive `x`, `g`,
/// fixed f64 `scratch`, and `state` when declared. No higher rule is inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceGradient {
    pub body: String,
    pub state: Option<SourceState>,
}
impl From<String> for SourceGradient {
    fn from(body: String) -> Self {
        Self { body, state: None }
    }
}
impl From<&str> for SourceGradient {
    fn from(body: &str) -> Self {
        body.to_owned().into()
    }
}
pub struct SourceRule<D> {
    /// Ordinary scalar primal with native value/domain semantics.
    pub body: SourceBody<D>,
    /// Optional FIRST-ORDER fused value/local scores, preserving admission.
    pub gradient: Option<fn(&D) -> SourceGradient>,
    /// Explicit source scratch extent in f64 elements, checked at registration.
    pub workspace: usize,
}
/// Author-supplied conditional pointwise decomposition of a summed primitive.
pub type BatchEvaluator<P> = fn(
    &P,
    &[f64],
    &mut [f64],
    &mut <P as OperationDescriptor>::Workspace,
) -> Result<(), AssemblyError>;
/// Optional local evaluation using a shared, pure scalar calculation.
pub type PreparedEvaluator<P> = fn(
    &P,
    &[f64],
    &mut [f64],
    &mut <P as OperationDescriptor>::Workspace,
    f64,
) -> Result<f64, AssemblyError>;
/// A trusted preparation rule. `prepare` must depend only on its argument and
/// have no side effects. Nonfinite results remain scratch: `evaluate` must admit
/// or reject them exactly as ordinary evaluation would, at the original node.
pub struct ScalarPreparation<P: OperationDescriptor> {
    /// Flattened scalar input index.
    pub argument: usize,
    /// Pure preparation, e.g. the logarithm of a shared Normal scale.
    pub prepare: fn(f64) -> f64,
    /// Original value/local-score calculation using the prepared scalar.
    pub evaluate: PreparedEvaluator<P>,
}
/// A fixed-size rule for a primitive's conditional row values.
pub struct BatchRule<P: OperationDescriptor> {
    /// Number of entries in fixed row order.
    pub units: usize,
    /// Writes every entry. Reporting stages/validates outputs before publication.
    pub evaluate: BatchEvaluator<P>,
}
impl<P: OperationDescriptor> BatchRule<P> {
    pub(super) fn validate(&self) -> Result<(), AssemblyError> {
        if self.units > isize::MAX as usize / size_of::<super::BatchUnit>() {
            return Err(AssemblyError::Shape);
        }
        Ok(())
    }
}
/// Cold metadata with recoverable typed descriptor and immutable captured data.
#[derive(Clone)]
pub struct PrimitiveMetadata {
    pub(super) signature: PrimitiveSignature,
    pub(super) descriptor: Arc<dyn Any + Send + Sync>,
    pub(super) source_workspace: usize,
    pub(super) preparation: Option<super::prepare::Rule>,
    pub(super) source_gradient: Option<Arc<super::SourceGradientCallback>>,
}
impl PrimitiveMetadata {
    /// Validated argument layout and declared probability semantics.
    pub fn signature(&self) -> &PrimitiveSignature {
        &self.signature
    }
    /// Immutable program and data of a registered scalar-result region.
    pub fn region_binding(&self) -> Option<&crate::model::tensor::BoundProgram> {
        self.descriptor::<crate::model::tensor::TensorFactor>()
            .map(crate::model::tensor::TensorFactor::binding)
    }

    /// Recovers a concrete descriptor without interpreting source strings or
    /// guessing an operation from its user-supplied factor name.
    pub fn descriptor<P: Any>(&self) -> Option<&P> {
        self.descriptor.downcast_ref()
    }
}
/// Prepared once after structural validation; native and source rules share this
/// path at construction and replacement. No graph mutation precedes preparation.
pub(super) struct Registration {
    pub metadata: PrimitiveMetadata,
    pub callback: Arc<super::CallbackFactory>,
    pub batch: Option<(usize, Arc<super::BatchFactory>)>,
    pub primal: Option<&'static str>,
    pub source: Option<Arc<super::SourceCallback>>,
}
pub(super) enum ReportingConstraint {
    New,
    SameUnits(Option<usize>),
}
impl Registration {
    pub fn prepare<P: OperationDescriptor>(
        primitive: P,
        rules: NativeRules<P>,
        input_count: usize,
        reporting: ReportingConstraint,
    ) -> Result<Self, AssemblyError> {
        let NativeRules {
            signature,
            workspace,
            batch_workspace,
            evaluate,
            preparation,
            batch,
            source,
        } = rules;
        signature.validate(input_count)?;
        let source_workspace = source.as_ref().map_or(0, |rule| rule.workspace);
        let (primal, source, source_gradient) = match source {
            None => (None, None, None),
            Some(SourceRule {
                body: SourceBody::Static(body),
                gradient,
                ..
            }) => (Some(body), None, gradient),
            Some(SourceRule {
                body: SourceBody::Generated(body),
                gradient,
                ..
            }) => (None, Some(body), gradient),
        };
        if preparation
            .as_ref()
            .is_some_and(|p| p.argument >= input_count)
        {
            return Err(AssemblyError::Shape);
        }
        let preparation_metadata = preparation.as_ref().map(|p| super::prepare::Rule {
            argument: p.argument,
            prepare: p.prepare,
        });
        if source_workspace > isize::MAX as usize / size_of::<f64>() {
            return Err(AssemblyError::Shape);
        }
        if let Some(rule) = &batch {
            rule.validate()?;
        }
        if let ReportingConstraint::SameUnits(units) = reporting
            && batch.as_ref().map(|rule| rule.units) != units
        {
            return Err(AssemblyError::Shape);
        }
        let descriptor = Arc::new(primitive);
        let executable = Arc::clone(&descriptor);
        let prepared = preparation.map(|rule| rule.evaluate);
        let source_gradient = source_gradient.map(|rule| {
            let descriptor = Arc::clone(&descriptor);
            Arc::new(move || rule(&descriptor)) as Arc<super::SourceGradientCallback>
        });
        let callback: Arc<super::CallbackFactory> = Arc::new(move || {
            let mut scratch = workspace(&executable);
            let executable = Arc::clone(&executable);
            Box::new(
                move |x: &[f64], g: &mut [f64], value| match (prepared, value) {
                    (Some(rule), Some(value)) => rule(&executable, x, g, &mut scratch, value),
                    _ => evaluate(&executable, x, g, &mut scratch),
                },
            ) as Box<super::Callback>
        });
        let batch = batch.map(|rule| {
            let descriptor = Arc::clone(&descriptor);
            let factory: Arc<super::BatchFactory> = Arc::new(move || {
                let mut scratch = batch_workspace(&descriptor);
                let descriptor = Arc::clone(&descriptor);
                let evaluate = rule.evaluate;
                Box::new(move |x: &[f64], out: &mut [f64]| {
                    evaluate(&descriptor, x, out, &mut scratch)
                }) as Box<super::BatchCallback>
            });
            (rule.units, factory)
        });
        let source = source.map(|rule| {
            let descriptor = Arc::clone(&descriptor);
            Arc::new(move || rule(&descriptor)) as Arc<super::SourceCallback>
        });
        Ok(Self {
            metadata: PrimitiveMetadata {
                signature,
                descriptor,
                source_workspace,
                preparation: preparation_metadata,
                source_gradient,
            },
            callback,
            batch,
            primal,
            source,
        })
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
    pub fn primitive<P: OperationDescriptor>(
        &mut self,
        inputs: &[Value],
        primitive: P,
    ) -> Result<Value, AssemblyError> {
        // Check edges before any rule can allocate a workspace or execute.
        let rules = primitive.rules();
        rules.signature.validate(inputs.len())?;
        for &input in inputs {
            self.index(input)?;
        }
        let Registration {
            metadata,
            callback,
            batch,
            primal,
            source,
        } = Registration::prepare(primitive, rules, inputs.len(), ReportingConstraint::New)?;
        let expression = self.custom_erased_node(inputs, callback, primal)?;
        if let Some(source) = source {
            self.generated_primals.push((expression.node, source));
        }
        if let Some((units, callback)) = batch {
            self.batch_rules.push((expression.node, units, callback));
        }
        self.primitives.push((expression.node, metadata));
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
    /// Whole-value native region call.
    Region,
}
/// Read-only cold graph snapshot entry. Inspecting the graph allocates input
/// handle vectors; retained model evaluation never constructs these entries.
pub struct GraphNode<'a> {
    /// Owner-checked handle usable when matching descriptor inputs to nodes.
    pub expression: Value,
    /// Fixed shape of the output, including rank-zero scalars.
    pub shape: crate::model::tensor::Shape,
    /// Execution operation.
    pub operation: OperationKind,
    /// Arguments in original order, including repeated shared expressions.
    pub inputs: Vec<Value>,
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
    pub missing_primals: Vec<Value>,
    /// Active matrix-provider factors without a qualified compiler curvature route.
    pub unqualified_curvature: Vec<Value>,
}
impl SourceCapabilities {
    /// All active nodes have source-lowering support.
    /// Use [`Self::resolve`] to also check a requested derivative route.
    pub fn supports_rust_primal(&self) -> bool {
        self.missing_primals.is_empty()
    }
}
impl super::Executable {
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
                Op::Region(call) => (OperationKind::Region, call.inputs.to_vec(), None),
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
                expression: Value {
                    owner: self.owner,
                    node,
                },
                shape: match op {
                    Op::Region(call) => call.binding.output_shape().clone(),
                    _ => crate::model::tensor::Shape::scalar(),
                },
                operation,
                inputs: inputs
                    .into_iter()
                    .map(|node| Value {
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
            unqualified_curvature: self
                .primitives
                .iter()
                .filter_map(|(node, primitive)| {
                    (self.active_nodes.as_ref().is_none_or(|a| a[*node])
                        && primitive
                            .region_binding()
                            .is_some_and(|t| t.requires_explicit_source()))
                    .then_some(Value {
                        owner: self.owner,
                        node: *node,
                    })
                })
                .collect(),
            missing_primals: self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(node, op)| {
                    (matches!(op, Op::Factor { .. } | Op::Region(_))
                        && self.active_nodes.as_ref().is_none_or(|active| active[node])
                        && !self.has_primal(node))
                    .then_some(Value {
                        owner: self.owner,
                        node,
                    })
                })
                .collect(),
        }
    }
}
