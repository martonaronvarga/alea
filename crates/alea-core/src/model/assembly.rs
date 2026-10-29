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

mod factors;
mod pointwise;
mod source;
pub use factors::{FactorMetadata, FactorRole};
pub use pointwise::{PointwiseError, PointwiseLikelihood, PredictiveUnit};

static NEXT_MODEL: AtomicU64 = AtomicU64::new(1);
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
    expressions: Vec<Expr>,
}
impl<S: VectorDomain> VectorParameter<S> {
    /// The validated chart, including runtime dimensions and geometric domain.
    pub fn transform(&self) -> Transform {
        self.transform
    }
    pub fn expressions(&self) -> &[Expr] {
        &self.expressions
    }
}
/// A read-only scalar expression belonging to one model builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct Expr {
    owner: u64,
    node: usize,
}
/// A support-typed handle. Handles cannot create mutable aliases of chain state.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Parameter<S> {
    expression: Expr,
    slot: usize,
    marker: PhantomData<S>,
}
impl<S> Parameter<S> {
    pub fn expr(&self) -> Expr {
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
    #[error("non-finite model expression or derivative")]
    Numerical,
    #[error("opaque model factor failed: {0}")]
    Factor(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A custom node has no Rust primal available for ahead-of-time compilation.
    #[error("model node {node} has no Rust primal for source generation")]
    MissingPrimal { node: usize },
    #[error(transparent)]
    Transform(#[from] TransformError),
}
type Callback = dyn FnMut(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync;
enum Op {
    Parameter(usize),
    Constant(f64),
    Add(usize, usize),
    Multiply(usize, usize),
    Exp(usize),
    Log(usize),
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
    callbacks: Vec<Box<Callback>>,
    // Cold metadata stays out of the retained operation-tape stride.
    primals: Vec<(usize, &'static str)>,
}
impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl ModelBuilder {
    pub fn new() -> Self {
        // The fetch_update -> try_update rename is newer than the MSRV. Spell
        // out its same checked retry loop to support both compiler generations.
        let mut owner = NEXT_MODEL.load(Ordering::Relaxed);
        loop {
            let next = owner
                .checked_add(1)
                .expect("model identity space exhausted");
            match NEXT_MODEL.compare_exchange_weak(
                owner,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => owner = actual,
            }
        }
        Self {
            owner,
            nodes: Vec::new(),
            transforms: Vec::new(),
            coordinates: Vec::new(),
            factors: Vec::new(),
            factor_metadata: Vec::new(),
            local_count: 0,
            max_inputs: 0,
            callbacks: Vec::new(),
            primals: Vec::new(),
        }
    }
    fn push(&mut self, op: Op) -> Expr {
        let node = self.nodes.len();
        self.nodes.push(op);
        Expr {
            owner: self.owner,
            node,
        }
    }
    fn index(&self, x: Expr) -> Result<usize, AssemblyError> {
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
        if name.is_empty() || self.coordinates.iter().any(|x| x.name == name) {
            return Err(AssemblyError::Name);
        }
        let slot = self.coordinates.len();
        self.transforms.push(transform);
        self.coordinates.push(Coordinate {
            name: name.into(),
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
    pub fn constant(&mut self, x: f64) -> Result<Expr, AssemblyError> {
        if !x.is_finite() {
            return Err(AssemblyError::Numerical);
        }
        Ok(self.push(Op::Constant(x)))
    }
    /// # Errors
    /// Rejects foreign handles.
    pub fn add(&mut self, a: Expr, b: Expr) -> Result<Expr, AssemblyError> {
        let a = self.index(a)?;
        let b = self.index(b)?;
        Ok(self.push(Op::Add(a, b)))
    }
    /// # Errors
    /// Rejects foreign handles.
    pub fn multiply(&mut self, a: Expr, b: Expr) -> Result<Expr, AssemblyError> {
        let a = self.index(a)?;
        let b = self.index(b)?;
        Ok(self.push(Op::Multiply(a, b)))
    }
    /// # Errors
    /// Rejects foreign handles. Runtime overflow is a typed evaluation failure.
    pub fn exp(&mut self, a: Expr) -> Result<Expr, AssemblyError> {
        let a = self.index(a)?;
        Ok(self.push(Op::Exp(a)))
    }
    /// # Errors
    /// Rejects foreign handles. Runtime non-positive input is an evaluation error.
    pub fn log(&mut self, a: Expr) -> Result<Expr, AssemblyError> {
        let a = self.index(a)?;
        Ok(self.push(Op::Log(a)))
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
        inputs: &[Expr],
        coefficients: &[f64],
        offset: f64,
    ) -> Result<Expr, AssemblyError> {
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
    pub fn custom<F>(&mut self, inputs: &[Expr], callback: F) -> Result<Expr, AssemblyError>
    where
        F: Fn(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        self.custom_node(inputs, callback, None)
    }
    /// Adds a scalar node with an explicit VJP and owned reusable scratch.
    ///
    /// Scratch moves into the compiled evaluator and is borrowed exclusively for
    /// each call. It is not shared between models. The callback must initialize
    /// all scratch it reads and overwrite every local derivative, including after
    /// failures or unwinding; scratch is not rolled back. Results must depend only
    /// on the inputs and immutable model data, never previous scratch contents.
    /// Like [`Self::custom`], this node has no generated Rust primal.
    ///
    /// # Errors
    /// Rejects foreign handles or overflowing derivative storage dimensions.
    pub fn custom_with_scratch<W, F>(
        &mut self,
        inputs: &[Expr],
        mut scratch: W,
        callback: F,
    ) -> Result<Expr, AssemblyError>
    where
        W: Send + Sync + 'static,
        F: Fn(&[f64], &mut [f64], &mut W) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        self.custom_node(inputs, move |x, g| callback(x, g, &mut scratch), None)
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
        inputs: &[Expr],
        primal: &'static str,
        callback: F,
    ) -> Result<Expr, AssemblyError>
    where
        F: Fn(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
        self.custom_node(inputs, callback, Some(primal))
    }
    fn custom_node<F>(
        &mut self,
        inputs: &[Expr],
        callback: F,
        primal: Option<&'static str>,
    ) -> Result<Expr, AssemblyError>
    where
        F: FnMut(&[f64], &mut [f64]) -> Result<f64, AssemblyError> + Send + Sync + 'static,
    {
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
        self.callbacks.push(Box::new(callback));
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
    pub fn factor(&mut self, log_factor: Expr) -> Result<(), AssemblyError> {
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
        F: FnMut(&mut Self, usize, &T) -> Result<Expr, AssemblyError>,
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
    pub fn sum(&mut self, values: &[Expr]) -> Result<Expr, AssemblyError> {
        let mut total = self.constant(0.0)?;
        for x in values {
            total = self.add(total, *x)?;
        }
        Ok(total)
    }
    /// Consumes construction and lowers one immutable constrained model.
    /// # Errors
    /// Rejects an empty parameter/factor plan or an invalid transform layout.
    pub fn compile(self) -> Result<CompiledModel, AssemblyError> {
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
    fn compile_inner(self) -> Result<CompiledModel, AssemblyError> {
        if self.transforms.is_empty() || self.factors.is_empty() {
            return Err(AssemblyError::Empty);
        }
        let layout = ParameterLayout::new(self.transforms)?;
        let workspace = RefCell::new(PlanWorkspace {
            callbacks: self.callbacks,
            values: vec![0.0; self.nodes.len()],
            adjoints: vec![0.0; self.nodes.len()],
            locals: vec![0.0; self.local_count],
            inputs: vec![0.0; self.max_inputs],
            gradient: vec![0.0; layout.constrained_dimension()],
        });
        Ok(CompiledModel {
            nodes: self.nodes,
            factors: self.factors,
            factor_metadata: self.factor_metadata,
            active_nodes: None,
            active_factors: None,
            selection: factors::Selection::All,
            coordinates: self.coordinates.into(),
            layout,
            workspace,
            primals: self.primals,
        })
    }
}
struct PlanWorkspace {
    callbacks: Vec<Box<Callback>>,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    locals: Vec<f64>,
    inputs: Vec<f64>,
    gradient: Vec<f64>,
}
/// Fixed analytic reverse plan with chain-local scratch and named layout metadata.
#[must_use = "a compiled model must be evaluated or lowered to a target"]
pub struct CompiledModel {
    nodes: Vec<Op>,
    active_nodes: Option<Vec<bool>>,
    active_factors: Option<Vec<usize>>,
    selection: factors::Selection,
    factors: Vec<usize>,
    factor_metadata: Vec<FactorMetadata>,
    coordinates: Arc<[Coordinate]>,
    layout: ParameterLayout,
    workspace: RefCell<PlanWorkspace>,
    primals: Vec<(usize, &'static str)>,
}
impl CompiledModel {
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
    /// Composes the active factor sum with its parameter charts without density
    /// Jacobians. This does not change the statistical selection: use
    /// `select_likelihood` or `select_posterior` explicitly when appropriate.
    /// # Errors
    /// Rejects a model/layout dimension mismatch.
    pub fn into_objective(
        self,
    ) -> Result<super::ComposedObjective<Self>, super::ModelError<AssemblyError>> {
        let layout = self.layout.clone();
        super::ComposedObjective::new(self, layout)
    }
    /// Applies the single Jacobian boundary for all declared coordinates.
    /// # Errors
    /// Returns a model/layout dimension error.
    pub fn into_target(self) -> Result<TransformedTarget<Self>, super::ModelError<AssemblyError>> {
        let layout = self.layout.clone();
        TransformedTarget::new(self, layout)
    }
}
impl ConstrainedModel for CompiledModel {
    type Error = AssemblyError;
    fn dimension(&self) -> usize {
        self.layout.constrained_dimension()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, AssemblyError> {
        if self.active_nodes.is_some() {
            self.evaluate_selected::<true>(q, g)
        } else {
            self.evaluate_selected::<false>(q, g)
        }
    }
}
impl CompiledModel {
    fn evaluate_selected<const FILTERED: bool>(
        &self,
        q: &[f64],
        g: &mut [f64],
    ) -> Result<f64, AssemblyError> {
        if q.len() != self.dimension() || g.len() != q.len() {
            return Err(AssemblyError::Shape);
        }
        if q.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        let mut work = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| AssemblyError::WorkspaceBusy)?;
        self.forward::<FILTERED>(q, &mut work)?;
        let PlanWorkspace {
            values,
            adjoints,
            locals,
            gradient,
            ..
        } = &mut *work;
        adjoints.fill(0.0);
        gradient.fill(0.0);
        let mut total = CompensatedSum::default();
        for i in self.retained_factors() {
            total.add(values[*i]);
            adjoints[*i] += 1.0;
        }
        for (i, op) in self.nodes.iter().enumerate().rev() {
            if FILTERED && !self.active_nodes.as_ref().expect("selected graph")[i] {
                continue;
            }
            let a = adjoints[i];
            match op {
                Op::Parameter(s) => gradient[*s] += a,
                Op::Constant(_) => {}
                Op::Add(x, y) => {
                    adjoints[*x] += a;
                    adjoints[*y] += a;
                }
                Op::Multiply(x, y) => {
                    adjoints[*x] += a * values[*y];
                    adjoints[*y] += a * values[*x];
                }
                Op::Exp(x) => adjoints[*x] += a * values[i],
                Op::Log(x) => adjoints[*x] += a / values[*x],
                Op::Factor { inputs, start, .. } => {
                    for (j, x) in inputs.iter().enumerate() {
                        adjoints[*x] += a * locals[start + j];
                    }
                }
            }
        }
        let total = total.total();
        if !total.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        g.copy_from_slice(gradient);
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
            ..
        } = work;
        for (i, op) in self.nodes.iter().enumerate() {
            if FILTERED && !self.active_nodes.as_ref().expect("selected graph")[i] {
                continue;
            }
            values[i] = match op {
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
                    for (slot, index) in inputs.iter_mut().zip(indices.iter()) {
                        *slot = values[*index];
                    }
                    let derivatives = &mut locals[*start..*start + indices.len()];
                    derivatives.fill(f64::NAN);
                    let value = callbacks[*callback](&inputs[..indices.len()], derivatives)?;
                    if derivatives.iter().any(|x| !x.is_finite()) {
                        return Err(AssemblyError::Numerical);
                    }
                    value
                }
            };
            if !values[i].is_finite() {
                return Err(AssemblyError::Numerical);
            }
        }
        Ok(())
    }
}
