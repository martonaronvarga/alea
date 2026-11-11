//! Typed, immutable scalar expression plans lowered through one ParameterLayout.
//! Construction owns names and graph storage; each compiled chain owns its scratch.
//! Shared expressions accumulate reverse derivatives. Jacobians are added only by
//! TransformedTarget, once per declared parameter, regardless of its use count.
use super::{ConstrainedModel, TransformedTarget};
use crate::transform::{ParameterLayout, Transform, TransformError};
use alea_math::numeric::CompensatedSum;
use std::{
    cell::RefCell,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

mod batch;
mod blocks;
mod factors;
mod inputs;
mod pointwise;
mod prepare;
mod primitives;
mod rebinding;
mod reverse;
mod scope;
mod source;
pub use batch::{BatchLikelihood, BatchUnit};
pub use factors::{FactorMetadata, FactorRole};
pub use inputs::ScalarInput;
pub use pointwise::{PointwiseError, PointwiseLikelihood, PredictiveUnit};
pub use primitives::{
    ArgumentShape, BatchEvaluator, BatchRule, GraphNode, NativeEvaluator, NativeRules,
    OperationDescriptor, OperationKind, PreparedEvaluator, PrimitiveArgument, PrimitiveMetadata,
    PrimitiveSignature, ProbabilitySemantics, ScalarPreparation, SourceBody, SourceCapabilities,
    SourceGradient, SourceRule, SourceState,
};

pub use scope::ModelScope;
pub use source::{
    CurvatureRoute, DerivativeOrder, FirstOrderSource, FloatWorkspace, SourceMode, SourceProgram,
    SourceRequest,
};

static NEXT_VALUE_OWNER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_value_owner() -> u64 {
    // The fetch_update -> try_update rename is newer than the MSRV. Spell
    // out its same checked retry loop to support both compiler generations.
    let mut owner = NEXT_VALUE_OWNER.load(Ordering::Relaxed);
    loop {
        let next = owner
            .checked_add(1)
            .expect("value identity space exhausted");
        match NEXT_VALUE_OWNER.compare_exchange_weak(
            owner,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => owner = actual,
        }
    }
    owner
}
/// Unconstrained real scalar parameter marker.
#[derive(Debug, Clone, Copy)]
pub enum Real {}
/// Positive scalar parameter marker, lowered with the exponential transform.
#[derive(Debug, Clone, Copy)]
pub enum Positive {}
/// Unit interval parameter marker, lowered with the logit transform.
#[derive(Debug, Clone, Copy)]
pub enum UnitInterval {}
mod domain_sealed {
    pub trait Sealed {}
}
/// Domain marker accepted by a structured parameter chart.
/// Implementations are sealed so a domain cannot silently accept another chart.
pub trait VectorDomain: domain_sealed::Sealed {
    fn accepts(transform: Transform) -> bool;
}
macro_rules! domains {
    ($($marker:ident => $domain:ident),* $(,)?) => {$(
        /// Structured parameter domain, validated against its chart at construction.
        #[derive(Debug, Clone, Copy)]
        pub enum $marker {}
        impl domain_sealed::Sealed for $marker {}
        impl VectorDomain for $marker {
            fn accepts(transform: Transform) -> bool { transform.domain() == crate::transform::TransformDomain::$domain }
        }
    )*};
}
impl domain_sealed::Sealed for Real {}
impl VectorDomain for Real {
    fn accepts(transform: Transform) -> bool {
        transform.domain() == crate::transform::TransformDomain::Real
    }
}
impl domain_sealed::Sealed for Positive {}
impl VectorDomain for Positive {
    fn accepts(transform: Transform) -> bool {
        Transform::positive(transform.constrained_dimension())
            .is_ok_and(|expected| expected == transform)
    }
}
impl domain_sealed::Sealed for UnitInterval {}
impl VectorDomain for UnitInterval {
    fn accepts(transform: Transform) -> bool {
        Transform::interval(transform.constrained_dimension(), 0., 1.)
            .is_ok_and(|expected| expected == transform)
    }
}
domains!(Simplex => Simplex, UnitSphere => UnitSphere, Ordered => Ordered,
    PositiveOrdered => PositiveOrdered, CholeskyCovariance => CholeskyCovariance,
    CholeskyCorrelation => CholeskyCorrelation, PositiveDefinite => PositiveDefinite,
    Correlation => Correlation);
/// Explicitly runtime-selected domain. Its validated chart retains the domain
/// metadata; use a concrete marker when the model's domain is known statically.
#[derive(Debug, Clone, Copy)]
pub enum DynamicDomain {}
impl domain_sealed::Sealed for DynamicDomain {}
impl VectorDomain for DynamicDomain {
    fn accepts(_: Transform) -> bool {
        true
    }
}
/// A constrained vector or matrix parameter with one transform block.
#[derive(Debug, Clone)]
#[must_use]
pub struct VectorParameter<S: VectorDomain> {
    marker: PhantomData<S>,
    transform: Transform,
    expressions: Vec<Value>,
}
impl<S: VectorDomain> VectorParameter<S> {
    /// The validated chart, including runtime dimensions and geometric domain.
    pub fn transform(&self) -> Transform {
        self.transform
    }
    pub fn expressions(&self) -> &[Value] {
        &self.expressions
    }
}
/// A read-only value belonging to one model or lexical region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct Value {
    pub(super) owner: u64,
    pub(super) node: usize,
}
/// A support-typed handle. Handles cannot create mutable aliases of chain state.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Parameter<S> {
    expression: Value,
    slot: usize,
    marker: PhantomData<S>,
}
impl<S> Parameter<S> {
    pub fn expr(&self) -> Value {
        self.expression
    }
    pub fn slot(&self) -> usize {
        self.slot
    }
}
/// Cold coordinate metadata, shared by model targets and output/prediction clients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coordinate {
    pub name: String,
    pub dimension: Option<String>,
    pub coordinate: Option<String>,
    pub block: Option<String>,
}
/// Construction and fused plan evaluation failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AssemblyError {
    #[error("parameter or expression belongs to another model")]
    ForeignHandle,
    /// A named statistical view cannot infer the meaning of an unclassified factor.
    #[error("factor {index} has no declared statistical role")]
    UnclassifiedFactor { index: usize },
    /// An arbitrary potential is not automatically a prior density.
    #[error("factor {index} is an arbitrary potential, not a declared posterior term")]
    PosteriorPotential { index: usize },
    #[error("duplicate or empty parameter or factor name")]
    Name,
    #[error("model dimension or plate shape mismatch")]
    Shape,
    #[error("model plan is empty or exhausted its identity space")]
    Empty,
    #[error("model workspace is already in use")]
    WorkspaceBusy,
    #[error("evaluation context belongs to another executable")]
    ContextMismatch,
    #[error("prepared evaluation is stale, consumed or belongs to another context")]
    StaleEvaluation,
    #[error("non-finite model expression or derivative")]
    Numerical,
    #[error("opaque model factor failed: {0}")]
    Factor(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A custom node has no Rust primal available for ahead-of-time compilation.
    #[error("model node {node} has no Rust primal for source generation")]
    MissingPrimal { node: usize },
    /// The requested derivative order has no explicitly selected source route.
    #[error("source derivative order {order:?} requires a qualified curvature route")]
    UnqualifiedDerivative { order: DerivativeOrder },
    /// A likelihood root has no explicitly supplied pointwise rule.
    #[error("model node {node} has no batch pointwise lowering")]
    MissingPointwise { node: usize },
    #[error(transparent)]
    Transform(#[from] TransformError),
}
type Callback = dyn FnMut(&[f64], &mut [f64], Option<f64>) -> Result<f64, AssemblyError> + Send;
enum Op {
    Parameter(usize),
    Constant(f64),
    Add(usize, usize),
    Multiply(usize, usize),
    Exp(usize),
    Log(usize),
    Region(Box<blocks::Call>),
    Factor {
        inputs: Box<[usize]>,
        callback: usize,
        start: usize,
    },
}
/// Owned cold builder. Plates are expanded at construction; retained evaluation
/// traverses a fixed plan and allocates nothing unless a user factor does.
#[must_use = "a model builder must be compiled before it can be evaluated"]
pub struct ModelBuilder {
    owner: u64,
    nodes: Vec<Op>,
    transforms: Vec<Transform>,
    coordinates: Vec<Coordinate>,
    factors: Vec<usize>,
    factor_metadata: Vec<FactorMetadata>,
    local_count: usize,
    max_inputs: usize,
    callbacks: Vec<Arc<CallbackFactory>>,
    // Cold metadata stays out of the retained operation-tape stride.
    primals: Vec<(usize, &'static str)>,
    generated_primals: Vec<(usize, Arc<SourceCallback>)>,
    batch_rules: Vec<(usize, usize, Arc<BatchFactory>)>,
    // Semantic descriptors are cold; never increase the hot Op stride.
    primitives: Vec<(usize, PrimitiveMetadata)>,
    namespace: String,
    block_count: usize,
}
impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            owner: next_value_owner(),
            nodes: Vec::new(),
            transforms: Vec::new(),
            coordinates: Vec::new(),
            factors: Vec::new(),
            factor_metadata: Vec::new(),
            local_count: 0,
            max_inputs: 0,
            callbacks: Vec::new(),
            primals: Vec::new(),
            generated_primals: Vec::new(),
            batch_rules: Vec::new(),
            primitives: Vec::new(),
            namespace: String::new(),
            block_count: 0,
        }
    }
    /// Starts a lexical region with its own values and explicit inputs.
    pub fn region(&self) -> super::tensor::RegionBuilder {
        super::tensor::RegionBuilder::new()
    }

    /// Applies an immutable scalar-result region to ordered scalar values.
    /// Captures follow the region's flattened input order. Repeated values
    /// contribute once per input slot to the same model adjoint.
    /// # Errors
    /// Rejects foreign captures, a mismatched input count or a non-scalar result.
    pub fn apply_region(
        &mut self,
        captures: &[Value],
        binding: super::tensor::BoundProgram,
    ) -> Result<Value, AssemblyError> {
        self.primitive(captures, super::tensor::TensorFactor::new(binding)?)
    }

    fn push(&mut self, op: Op) -> Value {
        let node = self.nodes.len();
        self.nodes.push(op);
        Value {
            owner: self.owner,
            node,
        }
    }
    fn index(&self, x: Value) -> Result<usize, AssemblyError> {
        let index = self.value_index(x)?;
        if matches!(&self.nodes[index], Op::Region(call) if !call.binding.output_shape().axes().is_empty())
        {
            return Err(AssemblyError::Shape);
        }
        Ok(index)
    }
    fn value_index(&self, x: Value) -> Result<usize, AssemblyError> {
        if x.owner != self.owner || x.node >= self.nodes.len() {
            Err(AssemblyError::ForeignHandle)
        } else {
            Ok(x.node)
        }
    }
    fn parameter<S>(
        &mut self,
        name: &str,
        transform: Transform,
    ) -> Result<Parameter<S>, AssemblyError> {
        if name.is_empty() {
            return Err(AssemblyError::Name);
        }
        let name = self.qualified_name(name);
        if self.coordinates.iter().any(|x| x.name == name) {
            return Err(AssemblyError::Name);
        }
        let slot = self.coordinates.len();
        self.transforms.push(transform);
        self.coordinates.push(Coordinate {
            name,
            dimension: None,
            coordinate: None,
            block: None,
        });
        Ok(Parameter {
            expression: self.push(Op::Parameter(slot)),
            slot,
            marker: PhantomData,
        })
    }
    /// # Errors
    /// Rejects empty/duplicate names and exhausted layout dimensions.
    pub fn real(&mut self, name: &str) -> Result<Parameter<Real>, AssemblyError> {
        self.parameter(name, Transform::identity(1)?)
    }
    /// # Errors
    /// Rejects empty/duplicate names and invalid layout dimensions.
    pub fn positive(&mut self, name: &str) -> Result<Parameter<Positive>, AssemblyError> {
        self.parameter(name, Transform::positive(1)?)
    }
    /// # Errors
    /// Rejects empty/duplicate names and invalid layout dimensions.
    pub fn unit_interval(&mut self, name: &str) -> Result<Parameter<UnitInterval>, AssemblyError> {
        self.parameter(name, Transform::interval(1, 0.0, 1.0)?)
    }
    /// Declares one vector/matrix transform and metadata for all constrained entries.
    /// Coordinate order is the transform's full output order, including derived
    /// matrix diagonals. A simplex therefore receives one Jacobian, not one per entry.
    /// # Errors
    /// Rejects dimension mismatch, empty/duplicate names or coordinate labels.
    pub fn vector<S: VectorDomain>(
        &mut self,
        name: &str,
        dimension: &str,
        labels: &[&str],
        block: &str,
        transform: Transform,
    ) -> Result<VectorParameter<S>, AssemblyError> {
        if !S::accepts(transform) {
            return Err(AssemblyError::Shape);
        }
        if name.is_empty() || labels.len() != transform.constrained_dimension() || labels.is_empty()
        {
            return Err(AssemblyError::Shape);
        }
        let name = self.qualified_name(name);
        let names: Vec<_> = labels
            .iter()
            .map(|label| format!("{name}[{label}]"))
            .collect();
        if names.iter().enumerate().any(|(i, n)| {
            labels[i].is_empty()
                || names[..i].contains(n)
                || self.coordinates.iter().any(|c| &c.name == n)
        }) {
            return Err(AssemblyError::Name);
        }
        let mut expressions = Vec::with_capacity(labels.len());
        for (label, name) in labels.iter().zip(names) {
            let slot = self.coordinates.len();
            self.coordinates.push(Coordinate {
                name,
                dimension: Some(dimension.into()),
                coordinate: Some((*label).into()),
                block: Some(block.into()),
            });
            expressions.push(self.push(Op::Parameter(slot)));
        }
        self.transforms.push(transform);
        Ok(VectorParameter {
            expressions,
            transform,
            marker: PhantomData,
        })
    }
    /// Names a coordinate and geometry block without changing numerical storage.
    /// # Errors
    /// Rejects a handle owned by another builder.
    pub fn annotate<S>(
        &mut self,
        p: &Parameter<S>,
        dimension: &str,
        coordinate: &str,
        block: &str,
    ) -> Result<(), AssemblyError> {
        self.index(p.expr())?;
        let c = &mut self.coordinates[p.slot];
        c.dimension = Some(dimension.into());
        c.coordinate = Some(coordinate.into());
        c.block = Some(block.into());
        Ok(())
    }
    /// # Errors
    /// Rejects non-finite constants.
    pub fn constant(&mut self, x: f64) -> Result<Value, AssemblyError> {
        if !x.is_finite() {
            return Err(AssemblyError::Numerical);
        }
        Ok(self.push(Op::Constant(x)))
    }
    /// Adds literals, expressions or scalar parameter handles.
    /// # Errors
    /// Rejects foreign handles/nonfinite literals before mutation.
    pub fn add(
        &mut self,
        a: impl Into<ScalarInput>,
        b: impl Into<ScalarInput>,
    ) -> Result<Value, AssemblyError> {
        let [a, b] = self.inputs([a.into(), b.into()])?;
        Ok(self.push(Op::Add(a.node, b.node)))
    }
    /// Multiplies literals, expressions or scalar parameter handles.
    /// # Errors
    /// Rejects foreign handles/nonfinite literals before mutation.
    pub fn multiply(
        &mut self,
        a: impl Into<ScalarInput>,
        b: impl Into<ScalarInput>,
    ) -> Result<Value, AssemblyError> {
        let [a, b] = self.inputs([a.into(), b.into()])?;
        Ok(self.push(Op::Multiply(a.node, b.node)))
    }
    /// # Errors
    /// Rejects foreign handles/nonfinite literals. Runtime overflow is a typed evaluation failure.
    pub fn exp(&mut self, a: impl Into<ScalarInput>) -> Result<Value, AssemblyError> {
        let a = self.expression(a)?;
        Ok(self.push(Op::Exp(a.node)))
    }
    /// # Errors
    /// Rejects foreign handles/nonfinite literals. Runtime non-positive input is an evaluation error.
    pub fn log(&mut self, a: impl Into<ScalarInput>) -> Result<Value, AssemblyError> {
        let a = self.expression(a)?;
        Ok(self.push(Op::Log(a.node)))
    }
    /// Fuses `offset + sum(coefficients[i] * inputs[i])` into one native node.
    /// Product residuals and addition residuals are accumulated before the final
    /// rounding, avoiding loss from large cancelling terms in linear predictors.
    /// The analytic derivatives are the supplied coefficients. This native node
    /// has no generated Rust primal; source lowering returns `MissingPrimal`.
    /// # Errors
    /// Rejects mismatched shapes, nonfinite coefficients/offsets, foreign handles,
    /// and nonrepresentable evaluation. No output derivatives are published on error.
    pub fn linear_combination(
        &mut self,
        inputs: &[Value],
        coefficients: &[f64],
        offset: f64,
    ) -> Result<Value, AssemblyError> {
        if inputs.len() != coefficients.len() {
            return Err(AssemblyError::Shape);
        }
        if !offset.is_finite() || coefficients.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        let coefficients = coefficients.to_vec();
        self.custom(inputs, move |values, gradient| {
            let mut sum = CompensatedSum::default();
            sum.add(offset);
            for (&value, &coefficient) in values.iter().zip(&coefficients) {
                sum.add_product(value, coefficient);
            }
            let result = sum.total();
            if !result.is_finite() {
                return Err(AssemblyError::Numerical);
            }
            gradient.copy_from_slice(&coefficients);
            Ok(result)
        })
    }
    /// Adds a deterministic/custom scalar node with an explicit local VJP. The
    /// callback must fill every derivative and preserve the mathematical model.
    /// # Errors
    /// Rejects foreign handles or overflowing derivative storage dimensions.
    pub fn custom<F>(&mut self, inputs: &[Value], callback: F) -> Result<Value, AssemblyError>
    where
        F: Fn(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        self.custom_node(inputs, callback, None)
    }
    /// Adds a scalar node with an explicit VJP and owned reusable scratch.
    ///
    /// The factory creates fresh scratch for each evaluation context. Scratch is
    /// borrowed exclusively for each call and is never shared between contexts. The callback must initialize
    /// all scratch it reads and overwrite every local derivative, including after
    /// failures or unwinding; scratch is not rolled back. Results must depend only
    /// on the inputs and immutable model data, never previous scratch contents.
    /// Like [`Self::custom`], this node has no generated Rust primal.
    ///
    /// # Errors
    /// Rejects foreign handles or overflowing derivative storage dimensions.
    pub fn custom_with_scratch<W, F>(
        &mut self,
        inputs: &[Value],
        workspace: impl Fn() -> W + Send + Sync + 'static,
        callback: F,
    ) -> Result<Value, AssemblyError>
    where
        W: Send + 'static,
        F: Fn(&[f64], &mut [f64], &mut W) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        let callback = Arc::new(callback);
        let factory: Arc<CallbackFactory> = Arc::new(move || {
            let mut scratch = workspace();
            let callback = Arc::clone(&callback);
            Box::new(move |x, g, _| callback(x, g, &mut scratch))
        });
        self.custom_erased_node(inputs, factory, None)
    }
    /// Adds an opaque VJP together with its differentiable Rust scalar primal.
    ///
    /// `primal` is a Rust function body whose `x: &[f64; inputs.len()]` argument
    /// has the same order as `inputs`. It must compute the same value and domain
    /// as `callback`, return NaN for invalid inputs, and use only names available
    /// in the generated module (prefer fully qualified paths). Captured data must
    /// be explicit input expressions. Alea differentiates this primal, never the
    /// supplied VJP. The source is compiled as ordinary trusted application code.
    ///
    /// # Errors
    /// Rejects foreign handles or overflowing derivative storage dimensions.
    pub fn custom_with_primal<F>(
        &mut self,
        inputs: &[Value],
        primal: &'static str,
        callback: F,
    ) -> Result<Value, AssemblyError>
    where
        F: Fn(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        self.custom_node(inputs, callback, Some(primal))
    }
    fn custom_node<F>(
        &mut self,
        inputs: &[Value],
        callback: F,
        primal: Option<&'static str>,
    ) -> Result<Value, AssemblyError>
    where
        F: Fn(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        let callback = Arc::new(callback);
        let factory: Arc<CallbackFactory> = Arc::new(move || {
            let callback = Arc::clone(&callback);
            Box::new(move |x, g, _| callback(x, g))
        });
        self.custom_erased_node(inputs, factory, primal)
    }
    fn custom_erased_node(
        &mut self,
        inputs: &[Value],
        callback: Arc<CallbackFactory>,
        primal: Option<&'static str>,
    ) -> Result<Value, AssemblyError> {
        let inputs = inputs
            .iter()
            .map(|x| self.index(*x))
            .collect::<Result<Vec<_>, _>>()?;
        let start = self.local_count;
        self.local_count = start
            .checked_add(inputs.len())
            .ok_or(AssemblyError::Shape)?;
        self.max_inputs = self.max_inputs.max(inputs.len());
        let callback_index = self.callbacks.len();
        self.callbacks.push(callback);
        let expression = self.push(Op::Factor {
            inputs: inputs.into_boxed_slice(),
            callback: callback_index,
            start,
        });
        if let Some(primal) = primal {
            self.primals.push((expression.node, primal));
        }
        Ok(expression)
    }
    /// Declares a log factor (normalized prior/likelihood or explicit potential).
    /// # Errors
    /// Rejects a foreign expression.
    pub fn factor(&mut self, log_factor: Value) -> Result<(), AssemblyError> {
        let i = self.index(log_factor)?;
        self.factors.push(i);
        self.factor_metadata.push(FactorMetadata::unclassified());
        Ok(())
    }
    /// Expands an independent plate at construction, with no retained iterator state.
    /// # Errors
    /// Propagates the factor-construction error at the first failing observation.
    pub fn plate<T, F>(&mut self, data: &[T], mut factor: F) -> Result<(), AssemblyError>
    where
        F: FnMut(&mut Self, usize, &T) -> Result<Value, AssemblyError>,
    {
        for (i, row) in data.iter().enumerate() {
            let x = factor(self, i, row)?;
            self.factor(x)?;
        }
        Ok(())
    }
    /// Reduces a tuple/plate of deterministic expressions to their sum.
    /// # Errors
    /// Rejects foreign handles.
    pub fn sum(&mut self, values: &[Value]) -> Result<Value, AssemblyError> {
        for &value in values {
            self.index(value)?;
        }
        let mut total = self.constant(0.0)?;
        for x in values {
            total = self.add(total, *x)?;
        }
        Ok(total)
    }
    /// Consumes construction and lowers one immutable constrained model.
    /// # Errors
    /// Rejects an empty parameter/factor plan or an invalid transform layout.
    pub fn compile(self) -> Result<NativeModel, AssemblyError> {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!(target: "alea::model", "compile", expressions = self.nodes.len(), factors = self.factors.len()).entered();
        #[cfg(feature = "tracing")]
        tracing::debug!(target: "alea::model", schema_version = 1_u64, event = "started");
        let outcome = self.compile_inner();
        #[cfg(feature = "tracing")]
        match &outcome {
            Ok(_) => {
                tracing::debug!(target: "alea::model", schema_version = 1_u64, event = "finished")
            }
            Err(_) => {
                tracing::error!(target: "alea::model", schema_version = 1_u64, event = "failed")
            }
        }
        outcome
    }
    fn compile_inner(self) -> Result<NativeModel, AssemblyError> {
        let definition = self.finish()?;
        Ok(definition.bind().compile()?.into_native_model())
    }

    /// Freezes the authoring graph and binding schema. No execution scratch is
    /// retained in the definition; ordinary `compile` also constructs an owned
    /// evaluator for the simple path.
    /// # Errors
    /// Rejects an empty graph or invalid parameter layout.
    pub fn finish(self) -> Result<Arc<ModelDefinition>, AssemblyError> {
        if self.transforms.is_empty() || self.factors.is_empty() {
            return Err(AssemblyError::Empty);
        }
        let layout = ParameterLayout::new(self.transforms)?;
        let regions = self
            .nodes
            .iter()
            .filter_map(|node| match node {
                Op::Region(call) => Some(call.binding.clone()),
                _ => None,
            })
            .collect();
        let defaults = BindingContents {
            callbacks: self.callbacks,
            batch_rules: self.batch_rules,
            primals: self.primals,
            generated_primals: self.generated_primals,
            primitives: self.primitives,
            regions,
        };
        Ok(Arc::new(ModelDefinition {
            owner: self.owner,
            nodes: self.nodes,
            factors: self.factors,
            factor_metadata: self.factor_metadata,
            coordinates: self.coordinates.into(),
            layout,
            local_count: self.local_count,
            max_inputs: self.max_inputs,
            defaults,
        }))
    }
}

/// Immutable graph, parameter charts, schema and default captured data.
/// Bindings share this one topology; they never own a second authoring graph.
pub struct ModelDefinition {
    owner: u64,
    nodes: Vec<Op>,
    factors: Vec<usize>,
    factor_metadata: Vec<FactorMetadata>,
    coordinates: Arc<[Coordinate]>,
    layout: ParameterLayout,
    local_count: usize,
    max_inputs: usize,
    defaults: BindingContents,
}

#[derive(Clone)]
struct BindingContents {
    callbacks: Vec<Arc<CallbackFactory>>,
    batch_rules: Vec<(usize, usize, Arc<BatchFactory>)>,
    primals: Vec<(usize, &'static str)>,
    generated_primals: Vec<(usize, Arc<SourceCallback>)>,
    primitives: Vec<(usize, PrimitiveMetadata)>,
    regions: Vec<crate::model::tensor::BoundProgram>,
}

/// An immutable, validated captured-data snapshot, with non-recycled identity.
/// Native evaluation and frozen source export resolve the same registered rules.
#[derive(Clone)]
pub struct DataBinding {
    definition: Arc<ModelDefinition>,
    identity: u64,
    callbacks: Vec<Arc<CallbackFactory>>,
    batch_rules: Vec<(usize, usize, Arc<BatchFactory>)>,
    primals: Vec<(usize, &'static str)>,
    generated_primals: Vec<(usize, Arc<SourceCallback>)>,
    primitives: Vec<(usize, PrimitiveMetadata)>,
    regions: Vec<crate::model::tensor::BoundProgram>,
}
impl std::ops::Deref for DataBinding {
    type Target = ModelDefinition;
    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}
impl ModelDefinition {
    /// Creates a fresh snapshot identity even when captured values are identical.
    pub fn bind(self: &Arc<Self>) -> DataBinding {
        let BindingContents {
            callbacks,
            batch_rules,
            primals,
            generated_primals,
            primitives,
            regions,
        } = self.defaults.clone();
        DataBinding {
            definition: Arc::clone(self),
            identity: next_value_owner(),
            callbacks,
            batch_rules,
            primals,
            generated_primals,
            primitives,
            regions,
        }
    }
    /// Compiles an immutable snapshot. Binding-dependent plans are rebuilt;
    /// parameter-dependent preparations belong exclusively to each context.
    /// # Errors
    /// Rejects a binding belonging to another definition.
    pub fn compile(&self, binding: &DataBinding) -> Result<Executable, AssemblyError> {
        if self.owner != binding.definition.owner {
            return Err(AssemblyError::ForeignHandle);
        }
        let (reverse, dynamic) = reverse::Plan::new(&self.nodes);
        Ok(Executable {
            binding: binding.clone(),
            reverse,
            dynamic,
            identity: next_value_owner(),
            active_nodes: None,
            active_factors: None,
            selection: factors::Selection::All,
        })
    }
}
impl DataBinding {
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn definition(&self) -> &Arc<ModelDefinition> {
        &self.definition
    }
    /// Compiles this immutable snapshot without consuming it.
    /// # Errors
    /// Preserves definition/schema validation errors.
    pub fn compile(&self) -> Result<Executable, AssemblyError> {
        self.definition.compile(self)
    }
}

/// Immutable bound executable. Share this object, never mutable execution state.
#[derive(Clone)]
pub struct Executable {
    binding: DataBinding,
    identity: u64,
    reverse: reverse::Plan,
    dynamic: Vec<bool>,
    active_nodes: Option<Vec<bool>>,
    active_factors: Option<Vec<usize>>,
    selection: factors::Selection,
}
impl std::ops::Deref for Executable {
    type Target = DataBinding;
    fn deref(&self) -> &Self::Target {
        &self.binding
    }
}

/// Exclusive evaluation scratch. It has no Sync bound and cannot be used with
/// another executable, even when the graph, binding or shapes happen to match.
pub struct EvaluationContext {
    executable: u64,
    identity: u64,
    epoch: u64,
    ready: Option<u64>,
    work: PlanWorkspace,
}

/// One-use publication token for an admitted native value/gradient evaluation.
/// It does not expose residual arrays or grant higher derivative capabilities.
pub struct PreparedEvaluation {
    executable: u64,
    context: u64,
    epoch: u64,
    value: f64,
}

/// Owned constrained evaluator. Interior mutability is local to this adapter;
/// the shared executable and all registered descriptor captures are immutable.
#[must_use]
pub struct NativeModel {
    executable: Arc<Executable>,
    workspace: RefCell<EvaluationContext>,
}
impl std::ops::Deref for NativeModel {
    type Target = Executable;
    fn deref(&self) -> &Self::Target {
        &self.executable
    }
}
impl Executable {
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn binding(&self) -> &DataBinding {
        &self.binding
    }
    /// Allocates one independent context and fresh operation/reporting workspaces.
    pub fn context(&self) -> EvaluationContext {
        let mut blocks = Vec::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if let Op::Region(call) = node {
                blocks.push(
                    self.active_nodes
                        .as_ref()
                        .is_none_or(|a| a[index])
                        .then(|| call.workspace(&self.dynamic, &self.binding.regions[call.slot])),
                );
            }
        }
        let mut active_callbacks = vec![true; self.callbacks.len()];
        if let Some(active) = &self.active_nodes {
            for (index, node) in self.nodes.iter().enumerate() {
                if let Op::Factor { callback, .. } = node {
                    active_callbacks[*callback] = active[index];
                }
            }
        }
        EvaluationContext {
            executable: self.identity,
            identity: next_value_owner(),
            epoch: 0,
            ready: None,
            work: PlanWorkspace {
                blocks,
                callbacks: self
                    .callbacks
                    .iter()
                    .enumerate()
                    .map(|(i, make)| {
                        if active_callbacks[i] {
                            make()
                        } else {
                            Box::new(|_: &[f64], _: &mut [f64], _: Option<f64>| {
                                Err(AssemblyError::Empty)
                            }) as Box<Callback>
                        }
                    })
                    .collect(),
                batch_rules: self
                    .batch_rules
                    .iter()
                    .map(|(node, units, make)| {
                        (
                            *node,
                            *units,
                            if self.active_nodes.as_ref().is_none_or(|a| a[*node]) {
                                make()
                            } else {
                                Box::new(|_: &[f64], _: &mut [f64]| Err(AssemblyError::Empty))
                                    as Box<BatchCallback>
                            },
                        )
                    })
                    .collect(),
                values: vec![0.; self.nodes.len()],
                adjoints: vec![0.; self.nodes.len()],
                locals: vec![0.; self.local_count],
                inputs: vec![0.; self.max_inputs],
                gradient: vec![0.; self.layout.constrained_dimension()],
                preparation: prepare::Plan::new(
                    &self.nodes,
                    &self.primitives,
                    self.callbacks.len(),
                ),
            },
        }
    }
    pub fn into_native_model(self) -> NativeModel {
        Arc::new(self).native_model()
    }
    /// Constructs an independent owned evaluator while sharing this immutable
    /// executable. Each worker owns its mutable context, with no Sync requirement.
    pub fn native_model(self: &Arc<Self>) -> NativeModel {
        NativeModel {
            executable: Arc::clone(self),
            workspace: RefCell::new(self.context()),
        }
    }
    /// Constructs a per-chain owned sampler target from a shared executable.
    /// # Errors
    /// Rejects a model/layout dimension mismatch.
    pub fn target(
        self: &Arc<Self>,
    ) -> Result<TransformedTarget<NativeModel>, super::ModelError<AssemblyError>> {
        self.native_model().into_target()
    }
    /// Constructs the ordinary owned sampling adapter, with one chart boundary.
    /// # Errors
    /// Rejects a model/layout dimension mismatch.
    pub fn into_target(
        self,
    ) -> Result<TransformedTarget<NativeModel>, super::ModelError<AssemblyError>> {
        self.into_native_model().into_target()
    }
    fn check_context(&self, context: &EvaluationContext) -> Result<(), AssemblyError> {
        if context.executable != self.identity {
            return Err(AssemblyError::ContextMismatch);
        }
        Ok(())
    }
    /// Computes and admits the complete constrained value and first-order score
    /// in exclusive scratch. Any failure or unwind invalidates prior publication.
    /// # Errors
    /// Preserves shape/native errors and rejects a foreign context or exhausted epoch.
    pub fn prepare(
        &self,
        context: &mut EvaluationContext,
        q: &[f64],
    ) -> Result<PreparedEvaluation, AssemblyError> {
        self.check_context(context)?;
        context.ready = None;
        context.epoch = context.epoch.checked_add(1).ok_or(AssemblyError::Empty)?;
        let value = if self.active_nodes.is_some() {
            self.evaluate_selected::<true>(q, &mut context.work)?
        } else {
            self.evaluate_selected::<false>(q, &mut context.work)?
        };
        context.ready = Some(context.epoch);
        Ok(PreparedEvaluation {
            executable: self.identity,
            context: context.identity,
            epoch: context.epoch,
            value,
        })
    }
    /// Publishes an admitted score once. Failed publication leaves caller output
    /// untouched. A matching publication attempt consumes its token; a foreign
    /// token does not invalidate another context's valid preparation.
    /// # Errors
    /// Rejects a foreign context, stale token or wrong output shape.
    pub fn publish(
        &self,
        context: &mut EvaluationContext,
        token: &PreparedEvaluation,
        g: &mut [f64],
    ) -> Result<f64, AssemblyError> {
        self.check_context(context)?;
        let ready = context.ready;
        if token.executable != self.identity
            || token.context != context.identity
            || ready != Some(token.epoch)
        {
            return Err(AssemblyError::StaleEvaluation);
        }
        context.ready = None;
        if g.len() != self.layout.constrained_dimension() {
            return Err(AssemblyError::Shape);
        }
        g.copy_from_slice(&context.work.gradient);
        Ok(token.value)
    }
    /// Fused constrained evaluation with transactional output publication.
    /// # Errors
    /// Preserves native numerical/structural errors and context identity failures.
    pub fn evaluate(
        &self,
        context: &mut EvaluationContext,
        q: &[f64],
        g: &mut [f64],
    ) -> Result<f64, AssemblyError> {
        self.check_context(context)?;
        context.ready = None;
        if g.len() != self.layout.constrained_dimension() {
            return Err(AssemblyError::Shape);
        }
        let token = self.prepare(context, q)?;
        self.publish(context, &token, g)
    }
}
impl NativeModel {
    pub fn executable(&self) -> &Executable {
        &self.executable
    }
    pub fn into_executable(self) -> Executable {
        Arc::try_unwrap(self.executable).unwrap_or_else(|shared| (*shared).clone())
    }
}

struct PlanWorkspace {
    blocks: Vec<Option<blocks::Work>>,
    callbacks: Vec<Box<Callback>>,
    batch_rules: Vec<(usize, usize, Box<BatchCallback>)>,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    locals: Vec<f64>,
    inputs: Vec<f64>,
    gradient: Vec<f64>,
    preparation: Option<Box<prepare::Plan>>,
}
type BatchCallback = dyn FnMut(&[f64], &mut [f64]) -> Result<(), AssemblyError> + Send;
type CallbackFactory = dyn Fn() -> Box<Callback> + Send + Sync;
type BatchFactory = dyn Fn() -> Box<BatchCallback> + Send + Sync;
type SourceCallback = dyn Fn() -> String + Send + Sync;
type SourceGradientCallback = dyn Fn() -> SourceGradient + Send + Sync;
impl Executable {
    /// Number of scalar expressions, including opaque batch nodes.
    /// Captured dataset bytes are not counted as expressions.
    pub fn expression_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn coordinates(&self) -> Arc<[Coordinate]> {
        Arc::clone(&self.coordinates)
    }
    pub fn layout(&self) -> &ParameterLayout {
        &self.layout
    }
}
impl NativeModel {
    /// Composes the active factor sum with its parameter charts without density
    /// Jacobians. This does not change the statistical selection: use
    /// `select_likelihood`, `select_prior` or `select_posterior` explicitly when
    /// appropriate.
    /// # Errors
    /// Rejects a model/layout dimension mismatch.
    pub fn into_objective(
        self,
    ) -> Result<super::ComposedObjective<Self>, super::ModelError<AssemblyError>> {
        let layout = self.layout.clone();
        super::ComposedObjective::new(self, layout)
    }
    /// Applies the single Jacobian boundary for all declared coordinates.
    ///
    /// The native target provides fused first-order evaluation, not automatic
    /// curvature capabilities. Even a smooth graph with analytic local scores
    /// does not acquire a Hessian-vector implementation:
    ///
    /// ```compile_fail,E0277
    /// use alea_core::{capability::PotentialHessianVector, model::assembly::ModelBuilder};
    /// let mut b = ModelBuilder::new();
    /// let x = b.real("x").unwrap().expr();
    /// let square = b.multiply(x, x).unwrap();
    /// b.factor(square).unwrap();
    /// let native = b.compile().unwrap().into_target().unwrap();
    /// fn needs_hvp(_: &impl PotentialHessianVector) {}
    /// needs_hvp(&native);
    /// ```
    ///
    /// Nor does it provide a potential Hessian and third derivatives. Request
    /// those only from a separately qualified backend implementing the trait:
    ///
    /// ```compile_fail,E0277
    /// use alea_core::{capability::PotentialHessianDerivative, model::assembly::ModelBuilder};
    /// let mut b = ModelBuilder::new();
    /// let x = b.real("x").unwrap().expr();
    /// let square = b.multiply(x, x).unwrap();
    /// b.factor(square).unwrap();
    /// let native = b.compile().unwrap().into_target().unwrap();
    /// fn needs_third(_: &impl PotentialHessianDerivative) {}
    /// needs_third(&native);
    /// ```
    /// # Errors
    /// Returns a model/layout dimension error.
    pub fn into_target(self) -> Result<TransformedTarget<Self>, super::ModelError<AssemblyError>> {
        let layout = self.layout.clone();
        TransformedTarget::new(self, layout)
    }
}
impl ConstrainedModel for NativeModel {
    type Error = AssemblyError;
    fn is_numerical_error(&self, error: &Self::Error) -> bool {
        match error {
            AssemblyError::Numerical => true,
            AssemblyError::Factor(source) => matches!(
                source.downcast_ref::<crate::distribution::DistributionError>(),
                Some(crate::distribution::DistributionError::Numerical)
            ),
            _ => false,
        }
    }
    fn dimension(&self) -> usize {
        self.layout.constrained_dimension()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, AssemblyError> {
        let mut context = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| AssemblyError::WorkspaceBusy)?;
        self.executable.evaluate(&mut context, q, g)
    }
}
impl Executable {
    fn evaluate_selected<const FILTERED: bool>(
        &self,
        q: &[f64],
        work: &mut PlanWorkspace,
    ) -> Result<f64, AssemblyError> {
        if q.len() != self.layout.constrained_dimension() {
            return Err(AssemblyError::Shape);
        }
        if q.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        self.forward::<FILTERED>(q, work)?;
        let PlanWorkspace {
            values,
            adjoints,
            locals,
            gradient,
            blocks,
            ..
        } = work;
        adjoints.fill(0.);
        gradient.fill(0.);
        let mut total = CompensatedSum::default();
        for i in self.retained_factors() {
            total.add(values[*i]);
            adjoints[*i] += 1.;
        }
        self.reverse.apply::<FILTERED>(
            values,
            adjoints,
            locals,
            gradient,
            self.active_nodes.as_deref(),
            |node, adjoints| {
                let Op::Region(call) = &self.nodes[node] else {
                    unreachable!()
                };
                call.reverse(&self.nodes, adjoints, blocks, node)
            },
        )?;
        let total = total.total();
        if !total.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        Ok(total)
    }

    // Shared forward tape. Pointwise reporting skips reverse propagation but
    // retains the existing callback value/VJP admission contract.
    fn forward<const FILTERED: bool>(
        &self,
        q: &[f64],
        work: &mut PlanWorkspace,
    ) -> Result<(), AssemblyError> {
        let PlanWorkspace {
            values,
            locals,
            inputs,
            callbacks,
            preparation,
            blocks,
            ..
        } = work;
        if let Some(preparation) = preparation {
            preparation.begin_evaluation();
        }
        for (i, op) in self.nodes.iter().enumerate() {
            if FILTERED && !self.active_nodes.as_ref().expect("selected graph")[i] {
                continue;
            }
            values[i] = match op {
                Op::Region(call) => {
                    let Some(value) = call.forward(&self.nodes, values, blocks)? else {
                        continue;
                    };
                    value
                }
                Op::Parameter(s) => q[*s],
                Op::Constant(x) => *x,
                Op::Add(a, b) => values[*a] + values[*b],
                Op::Multiply(a, b) => values[*a] * values[*b],
                Op::Exp(a) => values[*a].exp(),
                Op::Log(a) => values[*a].ln(),
                Op::Factor {
                    inputs: indices,
                    callback,
                    start,
                    ..
                } => {
                    let derivatives = &mut locals[*start..*start + indices.len()];
                    let mut evaluate = |arguments: &[f64]| {
                        derivatives.fill(f64::NAN);
                        let prepared = preparation
                            .as_mut()
                            .and_then(|p| p.value(*callback, arguments));
                        let value = callbacks[*callback](arguments, derivatives, prepared)?;
                        if derivatives.iter().any(|x| !x.is_finite()) {
                            return Err(AssemblyError::Numerical);
                        }
                        Ok(value)
                    };
                    // Fixed scalar argument packets avoid scratch gathering
                    // loops. Keep semantic nodes, callback order, poisoning and
                    // all local-score admission unchanged, including aliases.
                    match indices.as_ref() {
                        [a] => evaluate(&[values[*a]])?,
                        [a, b] => evaluate(&[values[*a], values[*b]])?,
                        [a, b, c] => evaluate(&[values[*a], values[*b], values[*c]])?,
                        [a, b, c, d] => {
                            evaluate(&[values[*a], values[*b], values[*c], values[*d]])?
                        }
                        _ => {
                            for (slot, index) in inputs.iter_mut().zip(indices.iter()) {
                                *slot = values[*index];
                            }
                            evaluate(&inputs[..indices.len()])?
                        }
                    }
                }
            };
            if !values[i].is_finite() {
                return Err(AssemblyError::Numerical);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    #[test]
    fn exhausted_epoch_invalidates_preparation_and_owned_adapter_reentrancy_is_typed() {
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        b.factor(x).unwrap();
        let model = b.compile().unwrap();
        let locked = model.workspace.borrow_mut();
        let mut g = [19.];
        assert!(matches!(
            model.logp_grad(&[2.], &mut g),
            Err(AssemblyError::WorkspaceBusy)
        ));
        assert_eq!(g, [19.]);
        drop(locked);
        assert_eq!(model.logp_grad(&[2.], &mut g).unwrap(), 2.);
        assert_eq!(g, [1.]);
        let mut context = model.context();
        let old = model.prepare(&mut context, &[2.]).unwrap();
        context.epoch = u64::MAX;
        assert!(matches!(
            model.prepare(&mut context, &[2.]),
            Err(AssemblyError::Empty)
        ));
        assert!(context.ready.is_none());
        assert!(model.publish(&mut context, &old, &mut g).is_err());
        assert!(!model.is_numerical_error(&AssemblyError::ContextMismatch));
        assert!(!model.is_numerical_error(&AssemblyError::StaleEvaluation));
    }
    #[test]
    fn selected_executable_never_allocates_excluded_operation_workspace() {
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let bad = b
            .custom_with_scratch(
                &[x],
                || -> () { panic!("excluded factory") },
                |x, g, _| {
                    g[0] = 1.;
                    Ok(x[0])
                },
            )
            .unwrap();
        b.prior("excluded", bad).unwrap();
        b.likelihood("retained", x).unwrap();
        let definition = b.finish().unwrap();
        let executable = definition
            .bind()
            .compile()
            .unwrap()
            .select_likelihood()
            .unwrap();
        let mut context = executable.context();
        let mut g = [19.];
        assert_eq!(
            executable.evaluate(&mut context, &[2.], &mut g).unwrap(),
            2.
        );
        assert_eq!(g, [1.]);
    }
}
