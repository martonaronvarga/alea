//! Checked typed registration for native operations on the region SSA.
use super::*;
use std::any::Any;

#[cfg(test)]
mod binding_tests;
#[cfg(test)]
mod observations;

// Immutable payloads belong to binding snapshots, not execution scratch.
type Prepared = dyn Any + Send + Sync;
pub(super) struct BindingState {
    prepared: Box<[Option<Arc<Prepared>>]>,
    pub(super) fixed: Box<[Option<binding::Fixed>]>,
}
pub(super) fn bind_snapshot(plan: Arc<Plan>, data: Vec<Arc<[f64]>>) -> Result<BoundProgram, E> {
    let mut prepared = Vec::new();
    let end = plan.last_native_block().map_or(0, |last| last + 1);
    for (i, node) in plan.nodes[..end].iter().enumerate() {
        let rule = match &node.op {
            Op::Registered(op) if plan.active[i] => match &**op {
                Operation::Native(call) => call.binding.as_ref(),
                Operation::Data(_) => None,
            },
            _ => None,
        };
        if let Some(rule) = rule {
            // Allocate only for bindings with an opted-in recipe. Preserve node
            // indexing without charging every ordinary binding for a table.
            prepared.resize_with(i + 1, || None);
            prepared[i] = Some((rule.prepare)(BindingInputs {
                call: op_inputs(node),
                arguments: &rule.arguments,
                nodes: &plan.nodes,
                data: &data,
            })?);
        }
    }
    let fixed = binding::materialize(&plan, &data);
    Ok(BoundProgram {
        plan,
        data,
        snapshot: (!prepared.is_empty() || !fixed.is_empty()).then(|| {
            Arc::new(BindingState {
                prepared: prepared.into_boxed_slice(),
                fixed: fixed.into_boxed_slice(),
            })
        }),
    })
}
fn op_inputs(node: &Node) -> &[Value] {
    let Op::Registered(op) = &node.op else {
        unreachable!("prepared registered call")
    };
    op.inputs()
}
impl BoundProgram {
    pub(super) fn fixed(&self, node: usize) -> Option<&binding::Fixed> {
        self.snapshot.as_ref()?.fixed.get(node)?.as_ref()
    }
    pub(super) fn has_fixed(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| !s.fixed.is_empty())
    }
    pub(super) fn preparation(&self, node: usize) -> Option<&Arc<Prepared>> {
        self.snapshot.as_ref()?.prepared.get(node)?.as_ref()
    }
}

/// Storage type of a block argument. Only real values have cotangents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    Real,
    Bool,
    Count,
    Index,
}
/// Fixed shape and element type required by a block rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockType {
    /// Storage type of each element.
    pub dtype: ElementType,
    /// Row-major extents of the argument.
    pub shape: Shape,
}
impl BlockType {
    pub(super) fn of(node: &Node) -> Self {
        let dtype = match &node.op {
            Op::Registered(op) => match &**op {
                Operation::Data(Data::Bool(_)) => ElementType::Bool,
                Operation::Data(Data::Count(_)) => ElementType::Count,
                Operation::Data(Data::Index(_)) => ElementType::Index,
                Operation::Native(_) => ElementType::Real,
            },
            _ => ElementType::Real,
        };
        Self {
            dtype,
            shape: node.shape.clone(),
        }
    }
}
pub(super) enum Data {
    Bool(Box<[bool]>),
    Count(Box<[u64]>),
    Index(Box<[usize]>),
}
impl Data {
    fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Count(v) => v.len(),
            Self::Index(v) => v.len(),
        }
    }
}

// Descriptors contain types/configuration, never graph handles. Registration
// resolves their edges and prepares the native rule only AFTER structural checks.
/// Native block rules with an immutable descriptor and a typed execution workspace.
pub struct BlockRules<D, W> {
    /// Ordered argument types, including discrete data without cotangent slots.
    pub inputs: Vec<BlockType>,
    /// Shape of the real result.
    pub output: Shape,
    /// Creates private workspace for each execution frame.
    pub workspace: fn(&D) -> W,
    /// Additional heap bytes owned by the workspace, excluding allocator metadata.
    pub heap_bytes: usize,
    /// Writes every element of the result.
    pub forward: fn(&D, BlockInputs<'_>, &mut [f64], &mut W) -> Result<(), E>,
    /// Optional eager binding recipe, independent of the VJP. Only declared
    /// direct immutable leaves may be read; this never runs `forward` early.
    pub preparation: Option<BlockPreparation<D>>,
    /// Optional reverse rule. Writes contributions for real arguments in order.
    pub vjp: Option<BlockVjp<D, W>>,
}

/// Reverse rule receiving the primal output and its cotangent seed. Real argument
/// contributions are flattened in argument order, including repeated inputs.
pub type BlockVjp<D, W> =
    fn(&D, BlockInputs<'_>, &[f64], &[f64], &mut [f64], &mut W) -> Result<(), E>;

/// Explicit directional rule over already admitted primals and read-only
/// forward workspace. The rule writes the entire output tangent in row-major
/// order. It does not imply a transpose, replay, source or higher derivatives.
pub type BlockJvp<D, W> =
    fn(&D, BlockInputs<'_>, BlockTangents<'_>, &[f64], &mut [f64], &W) -> Result<(), E>;

/// Directional derivative of a registered VJP at the admitted forward point.
/// The seeds are the output cotangent and its direction, respectively.
/// Overwrite all real argument contributions in flattened occurrence order.
/// In the full-input curvature evaluator, a missing input direction denotes a
/// structurally inactive occurrence and its contribution can be written as zero.
/// The workspace is the read-only forward residual, before the local VJP runs.
pub type BlockVjpJvp<D, W> = fn(
    &D,
    BlockInputs<'_>,
    BlockTangents<'_>,
    &[f64],
    [&[f64]; 2],
    &mut [f64],
    &W,
) -> Result<(), E>;

/// Directions for the real argument occurrences of one registered call.
pub struct BlockTangents<'a> {
    arguments: &'a [Value],
    nodes: &'a [Node],
    ranges: &'a [Option<Range<usize>>],
    values: &'a [f64],
}
impl BlockTangents<'_> {
    /// Returns a real direction, or `None` for a symbolic zero.
    /// # Errors
    /// Rejects nonexistent or discrete arguments.
    pub fn real(&self, i: usize) -> Result<Option<&[f64]>, E> {
        let argument = self.arguments.get(i).ok_or(E::Shape)?;
        if self.nodes[argument.node].discrete() {
            return Err(E::Shape);
        }
        Ok(self.ranges[i].as_ref().map(|r| &self.values[r.clone()]))
    }
}

type ErasedJvp = dyn Fn(BlockInputs<'_>, BlockTangents<'_>, &[f64], &mut [f64], &Workspace) -> Result<(), E>
    + Send
    + Sync;

type ErasedVjpJvp = dyn Fn(
        BlockInputs<'_>,
        BlockTangents<'_>,
        &[f64],
        [&[f64]; 2],
        &mut [f64],
        &Workspace,
    ) -> Result<(), E>
    + Send
    + Sync;

type Workspace = dyn Any + Send;
type ErasedForward =
    dyn Fn(BlockInputs<'_>, &mut [f64], &mut Workspace) -> Result<(), E> + Send + Sync;
type ErasedVjp = dyn Fn(BlockInputs<'_>, &[f64], &[f64], &mut [f64], &mut Workspace) -> Result<(), E>
    + Send
    + Sync;

#[cfg(test)]
struct Signature {
    inputs: Vec<BlockType>,
    output: BlockType,
    scratch: usize,
}
#[cfg(test)]
type Forward = dyn Fn(BlockInputs<'_>, &mut [f64], &mut [f64]) -> Result<(), E> + Send + Sync;
#[cfg(test)]
type Vjp =
    dyn Fn(BlockInputs<'_>, &[f64], &[f64], &mut [f64], &mut [f64]) -> Result<(), E> + Send + Sync;
#[cfg(test)]
struct Kernel {
    forward: Box<Forward>,
    // An optional native VJP implies neither JVP, source nor higher derivatives.
    vjp: Option<Box<Vjp>>,
}
type Prepare<D> = dyn Fn(&D, BindingInputs<'_>) -> Result<Arc<Prepared>, E> + Send + Sync;
/// Explicit opt-in to eager preparation at `bind`/`rebind`, not at evaluation.
/// The callback must validate its own typed payload and return any domain error.
/// Failed preparation (or unwind) leaves old snapshots/contexts intact. Ordinary
/// guarded forward callbacks are never inferred pure or hoisted by this recipe.
/// Clones of a binding share the returned immutable payload; per-point residuals
/// and mutable scratch must remain in the execution workspace.
pub struct BlockPreparation<D> {
    arguments: Box<[usize]>,
    prepare: Box<Prepare<D>>,
}
impl<D: 'static> BlockPreparation<D> {
    /// Declares logical block argument positions available to the recipe.
    /// Registration rejects parameters, derived values and bad positions before
    /// graph mutation. The function runs once per active call per new binding;
    /// the model convenience registration may bind immediately during construction.
    /// P is erased once; `BlockInputs::prepared::<P>` checks its type at access.
    pub fn new<P: Any + Send + Sync>(
        arguments: Vec<usize>,
        prepare: fn(&D, BindingInputs<'_>) -> Result<P, E>,
    ) -> Self {
        Self {
            arguments: arguments.into_boxed_slice(),
            prepare: Box::new(move |d, inputs| Ok(Arc::new(prepare(d, inputs)?) as Arc<Prepared>)),
        }
    }
}
type ErasedPrepare = dyn Fn(BindingInputs<'_>) -> Result<Arc<Prepared>, E> + Send + Sync;
struct PreparationCall {
    arguments: Box<[usize]>,
    prepare: Box<ErasedPrepare>,
}
enum BindingValue<'a> {
    Real(&'a [f64]),
    Bool(&'a [bool]),
    Count(&'a [u64]),
    Index(&'a [usize]),
}
/// Read-only immutable leaves in the recipe's declared argument order. Parameters
/// and undeclared call arguments are not available, even if currently constant.
pub struct BindingInputs<'a> {
    call: &'a [Value],
    arguments: &'a [usize],
    nodes: &'a [Node],
    data: &'a [Arc<[f64]>],
}
impl BindingInputs<'_> {
    fn get(&self, i: usize) -> Result<BindingValue<'_>, E> {
        let argument = *self.arguments.get(i).ok_or(E::Shape)?;
        let value = self.call.get(argument).ok_or(E::Shape)?;
        match &self.nodes.get(value.node).ok_or(E::Shape)?.op {
            Op::Data(i) => Ok(BindingValue::Real(&self.data[*i])),
            Op::Constant(v) => Ok(BindingValue::Real(v)),
            Op::Registered(op) => match &**op {
                Operation::Data(Data::Bool(v)) => Ok(BindingValue::Bool(v)),
                Operation::Data(Data::Count(v)) => Ok(BindingValue::Count(v)),
                Operation::Data(Data::Index(v)) => Ok(BindingValue::Index(v)),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
    /// Reads a declared real data/constant leaf.
    /// # Errors
    /// Rejects a wrong position or element type.
    pub fn real(&self, i: usize) -> Result<&[f64], E> {
        match self.get(i)? {
            BindingValue::Real(v) => Ok(v),
            _ => Err(E::Shape),
        }
    }
    /// Reads a declared Boolean leaf.
    /// # Errors
    /// Rejects a wrong position or element type.
    pub fn booleans(&self, i: usize) -> Result<&[bool], E> {
        match self.get(i)? {
            BindingValue::Bool(v) => Ok(v),
            _ => Err(E::Shape),
        }
    }
    /// Reads a declared count leaf without floating conversion.
    /// # Errors
    /// Rejects a wrong position or element type.
    pub fn counts(&self, i: usize) -> Result<&[u64], E> {
        match self.get(i)? {
            BindingValue::Count(v) => Ok(v),
            _ => Err(E::Shape),
        }
    }
    /// Reads a declared index leaf without floating conversion.
    /// # Errors
    /// Rejects a wrong position or element type.
    pub fn indices(&self, i: usize) -> Result<&[usize], E> {
        match self.get(i)? {
            BindingValue::Index(v) => Ok(v),
            _ => Err(E::Shape),
        }
    }
}
pub(super) struct Call {
    inputs: Box<[Value]>,
    contributions: Box<[Range<usize>]>,
    scores: usize,
    workspace_bytes: usize,
    output: usize,
    workspace: Box<dyn Fn() -> Box<Workspace> + Send + Sync>,
    forward: Box<ErasedForward>,
    vjp: Option<Box<ErasedVjp>>,
    jvp: Option<Box<ErasedJvp>>,
    vjp_jvp: Option<Box<ErasedVjpJvp>>,
    binding: Option<PreparationCall>,
    source: Option<Box<dyn Fn() -> String + Send + Sync>>,
}
pub(super) enum Operation {
    Data(Data),
    Native(Call),
}
impl Operation {
    pub(super) fn inputs(&self) -> &[Value] {
        match self {
            Self::Data(_) => &[],
            Self::Native(call) => &call.inputs,
        }
    }
}

/// Read-only block arguments in call order. Discrete inputs keep their original
/// storage types and do not receive cotangents.
pub struct BlockInputs<'a> {
    arguments: &'a [Value],
    nodes: &'a [Node],
    values: BlockValues<'a>,
    prepared: Option<&'a Prepared>,
    evaluation_prepared: Option<&'a Prepared>,
}
enum BlockValues<'a> {
    Arena(lowering::PrimalView<'a>),
    Residuals(lowering::PrimalInputs<'a>),
}
impl BlockInputs<'_> {
    /// Borrows this block's typed evaluation payload during forward.
    /// # Errors
    /// Rejects an absent payload or a different Rust type.
    pub fn evaluation_prepared<P: Any>(&self) -> Result<&P, E> {
        self.evaluation_prepared
            .and_then(|p| p.downcast_ref::<P>())
            .ok_or(E::Shape)
    }
    /// Borrows the immutable payload returned by this call's binding recipe.
    /// # Errors
    /// Rejects an absent recipe or a different Rust payload type.
    pub fn prepared<P: Any + Send + Sync>(&self) -> Result<&P, E> {
        self.prepared
            .and_then(|p| p.downcast_ref::<P>())
            .ok_or(E::Shape)
    }
    fn node(&self, i: usize) -> Result<&Node, E> {
        let v = self.arguments.get(i).ok_or(E::Shape)?;
        self.nodes.get(v.node).ok_or(E::Shape)
    }
    /// Returns the real block at this argument position.
    /// # Errors
    /// Rejects a wrong argument position or element type.
    pub fn real(&self, i: usize) -> Result<&[f64], E> {
        let n = self.node(i)?;
        if n.discrete() {
            return Err(E::Shape);
        }
        match self.values {
            BlockValues::Arena(values) => values.get(n.primal_range()),
            BlockValues::Residuals(inputs) => inputs.get(i),
        }
        .ok_or(E::Shape)
    }
    /// Returns the Boolean block at this argument position.
    /// # Errors
    /// Rejects a wrong argument position or element type.
    pub fn booleans(&self, i: usize) -> Result<&[bool], E> {
        match &self.node(i)?.op {
            Op::Registered(op) => match &**op {
                Operation::Data(Data::Bool(v)) => Ok(v),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
    /// Returns the count block at this argument position.
    /// # Errors
    /// Rejects a wrong argument position or element type.
    pub fn counts(&self, i: usize) -> Result<&[u64], E> {
        match &self.node(i)?.op {
            Op::Registered(op) => match &**op {
                Operation::Data(Data::Count(v)) => Ok(v),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
    /// Returns the index block at this argument position.
    /// # Errors
    /// Rejects a wrong argument position or element type.
    pub fn indices(&self, i: usize) -> Result<&[usize], E> {
        match &self.node(i)?.op {
            Op::Registered(op) => match &**op {
                Operation::Data(Data::Index(v)) => Ok(v),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
}
fn extent(a: usize, b: usize) -> Result<usize, E> {
    a.checked_add(b)
        .filter(|&v| v <= isize::MAX as usize / size_of::<f64>())
        .ok_or(E::Shape)
}
fn checked_workspace_bytes<W>(elements: usize, heap_bytes: usize) -> Result<usize, E> {
    (elements * size_of::<f64>())
        .checked_add(size_of::<W>())
        .and_then(|bytes| bytes.checked_add(heap_bytes))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(E::Shape)
}
impl RegionBuilder {
    /// Adds immutable Boolean data with the supplied shape.
    /// # Errors
    /// Rejects a shape that disagrees with the value count.
    pub fn boolean_data(&mut self, shape: Shape, values: Vec<bool>) -> Result<Value, E> {
        self.typed_data(shape, Data::Bool(values.into_boxed_slice()))
    }

    /// Adds immutable counts without a conversion through floating point.
    /// # Errors
    /// Rejects a shape that disagrees with the value count.
    pub fn count_data(&mut self, shape: Shape, values: Vec<u64>) -> Result<Value, E> {
        self.typed_data(shape, Data::Count(values.into_boxed_slice()))
    }

    /// Registers an operation with ordered, whole-value inputs and a real output.
    /// # Errors
    /// Rejects foreign handles, type or shape mismatches and workspace overflow.
    pub fn register_block<D, W>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
    {
        self.register_block_rules(inputs, descriptor, rules, None, None, None)
    }

    /// Registers an explicitly supplied directional rule alongside native rules.
    /// The callback reads admitted forward residuals and is independent of VJP.
    /// # Errors
    /// Returns the same structural errors as [Self::register_block].
    pub fn register_block_with_jvp<D, W>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
        jvp: BlockJvp<D, W>,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
    {
        self.register_block_rules(inputs, descriptor, rules, Some(jvp), None, None)
    }

    /// Registers a JVP and an explicit directional derivative of the supplied VJP.
    /// All three rules must describe the same operation. HVP evaluation first
    /// runs ordinary admission and may evaluate the operation again at the
    /// same point, so these callbacks must describe a fixed mathematical map.
    /// This registration does not enable checkpoint replay or source lowering.
    /// # Errors
    /// Rejects a missing VJP or the structural errors of [Self::register_block].
    pub fn register_block_with_vjp_jvp<D, W>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
        jvp: BlockJvp<D, W>,
        vjp_jvp: BlockVjpJvp<D, W>,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
    {
        if rules.vjp.is_none() {
            return Err(E::Factor(Box::new(MissingVjp)));
        }
        self.register_block_rules(inputs, descriptor, rules, Some(jvp), Some(vjp_jvp), None)
    }

    /// Registers native rules and trusted Rust for the same real-input block.
    ///
    /// The source body receives `args: &[&[f64]]`, `out: &mut [f64]`,
    /// `seed: Option<&[f64]>` and `scores: &mut [f64]`. With no seed it
    /// writes the complete primal and receives empty scores. With a seed it
    /// writes ordered, flattened argument cotangents and receives empty
    /// primal storage. Both calls must admit the full primal and local scores
    /// and return `false` on failure. The body must be safe to replay and
    /// serialize any captured immutable data as typed Rust literals.
    /// Source supports value and explicit first order for this registration.
    ///
    /// # Errors
    /// Rejects binding preparation, discrete inputs, a missing native VJP,
    /// and the structural errors of [Self::register_block].
    pub fn register_block_with_source<D, W>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
        jvp: BlockJvp<D, W>,
        vjp_jvp: BlockVjpJvp<D, W>,
        source: fn(&D) -> String,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
    {
        if rules.preparation.is_some()
            || rules.inputs.iter().any(|ty| ty.dtype != ElementType::Real)
        {
            return Err(E::Shape);
        }
        if rules.vjp.is_none() {
            return Err(E::Factor(Box::new(MissingVjp)));
        }
        self.register_block_rules(
            inputs,
            descriptor,
            rules,
            Some(jvp),
            Some(vjp_jvp),
            Some(source),
        )
    }

    fn register_block_rules<D, W>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
        jvp: Option<BlockJvp<D, W>>,
        vjp_jvp: Option<BlockVjpJvp<D, W>>,
        source: Option<fn(&D) -> String>,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
    {
        let (scores, contributions) =
            self.check_block_edges(inputs, &rules.inputs, &rules.output)?;
        if let Some(recipe) = &rules.preparation {
            for &i in &recipe.arguments {
                let input = *inputs.get(i).ok_or(E::Shape)?;
                let node = self.node(input)?;
                if !matches!(&node.op, Op::Data(_) | Op::Constant(_)) && !node.discrete() {
                    return Err(E::Shape);
                }
            }
        }
        let elements = extent(scores, rules.output.elements)?;
        let workspace_bytes = checked_workspace_bytes::<W>(elements, rules.heap_bytes)?;
        let descriptor = Arc::new(descriptor);
        let forward_descriptor = Arc::clone(&descriptor);
        let workspace_descriptor = Arc::clone(&descriptor);
        let source = source.map(|render| {
            let descriptor = Arc::clone(&descriptor);
            Box::new(move || render(&descriptor)) as Box<dyn Fn() -> String + Send + Sync>
        });
        let binding = rules.preparation.map(|recipe| {
            let descriptor = Arc::clone(&descriptor);
            PreparationCall {
                arguments: recipe.arguments,
                prepare: Box::new(move |inputs| (recipe.prepare)(&descriptor, inputs)),
            }
        });
        let jvp = jvp.map(|rule| -> Box<ErasedJvp> {
            let descriptor = Arc::clone(&descriptor);
            Box::new(move |inputs, tangents, primal, out, workspace| {
                rule(
                    &descriptor,
                    inputs,
                    tangents,
                    primal,
                    out,
                    workspace.downcast_ref::<W>().expect("typed workspace"),
                )
            })
        });
        let vjp_jvp = vjp_jvp.map(|rule| -> Box<ErasedVjpJvp> {
            let descriptor = Arc::clone(&descriptor);
            Box::new(move |inputs, tangents, primal, seeds, out, workspace| {
                rule(
                    &descriptor,
                    inputs,
                    tangents,
                    primal,
                    seeds,
                    out,
                    workspace.downcast_ref::<W>().expect("typed workspace"),
                )
            })
        });
        let output = rules.output.elements;
        self.push(
            rules.output,
            Op::Registered(Box::new(Operation::Native(Call {
                inputs: inputs.into(),
                contributions: contributions.into_boxed_slice(),
                scores,
                workspace_bytes,
                output,
                workspace: Box::new(move || Box::new((rules.workspace)(&workspace_descriptor))),
                forward: Box::new(move |inputs, output, workspace| {
                    (rules.forward)(
                        &forward_descriptor,
                        inputs,
                        output,
                        workspace.downcast_mut::<W>().expect("typed workspace"),
                    )
                }),
                vjp: rules.vjp.map(|rule| -> Box<ErasedVjp> {
                    Box::new(move |inputs, primal, seed, scores, workspace| {
                        rule(
                            &descriptor,
                            inputs,
                            primal,
                            seed,
                            scores,
                            workspace.downcast_mut::<W>().expect("typed workspace"),
                        )
                    })
                }),
                jvp,
                vjp_jvp,
                binding,
                source,
            }))),
        )
    }

    fn check_block_edges(
        &self,
        inputs: &[Value],
        types: &[BlockType],
        output: &Shape,
    ) -> Result<(usize, Vec<Range<usize>>), E> {
        if inputs.len() != types.len() {
            return Err(E::Shape);
        }
        // Check ALL handles before types or executing user preparation.
        for &input in inputs {
            self.node(input)?;
        }
        let mut scores = 0;
        let mut contributions = Vec::with_capacity(inputs.len());
        for (&input, expected) in inputs.iter().zip(types) {
            let actual = BlockType::of(self.node(input)?);
            if &actual != expected {
                return Err(E::Shape);
            }
            let start = scores;
            if actual.dtype == ElementType::Real {
                scores = extent(scores, actual.shape.elements)?;
            }
            contributions.push(start..scores);
        }
        extent(self.slots, output.elements)?;
        Ok((scores, contributions))
    }

    fn typed_data(&mut self, shape: Shape, data: Data) -> Result<Value, E> {
        if data.len() != shape.elements {
            return Err(E::Shape);
        }
        self.push(shape, Op::Registered(Box::new(Operation::Data(data))))
    }
    /// Adds immutable indices, checking each against the exclusive upper bound.
    /// # Errors
    /// Rejects an out-of-bounds index or a shape mismatch.
    pub fn indices(&mut self, shape: Shape, values: Vec<usize>, bound: usize) -> Result<Value, E> {
        if values.iter().any(|&i| i >= bound) {
            return Err(E::Shape);
        }
        self.typed_data(shape, Data::Index(values.into_boxed_slice()))
    }
    #[cfg(test)]
    fn register(
        &mut self,
        signature: Signature,
        inputs: &[Value],
        prepare: impl FnOnce() -> Result<Kernel, E>,
    ) -> Result<Value, E> {
        if signature.output.dtype != ElementType::Real {
            return Err(E::Shape);
        }
        let (scores, _) =
            self.check_block_edges(inputs, &signature.inputs, &signature.output.shape)?;
        let heap_bytes = signature
            .scratch
            .checked_mul(size_of::<f64>())
            .ok_or(E::Shape)?;
        let elements = extent(scores, signature.output.shape.elements)?;
        checked_workspace_bytes::<Vec<f64>>(elements, heap_bytes)?;
        let kernel = prepare()?;
        let Kernel { forward, vjp } = kernel;
        struct Fixture {
            forward: Box<Forward>,
            vjp: Option<Box<Vjp>>,
            scratch: usize,
        }
        let has_vjp = vjp.is_some();
        let value = self.register_block(
            inputs,
            Fixture {
                forward,
                vjp,
                scratch: signature.scratch,
            },
            BlockRules {
                preparation: None,
                inputs: signature.inputs,
                output: signature.output.shape,
                workspace: |fixture| vec![0.; fixture.scratch],
                heap_bytes,
                forward: |fixture, inputs, output, workspace| {
                    (fixture.forward)(inputs, output, workspace)
                },
                vjp: has_vjp.then_some(
                    |fixture: &Fixture, inputs, primal, seed, scores, workspace: &mut Vec<f64>| {
                        fixture.vjp.as_ref().expect("fixture VJP")(
                            inputs, primal, seed, scores, workspace,
                        )
                    },
                ),
            },
        )?;
        Ok(value)
    }
}

fn same_preparation(old: Option<&Arc<Prepared>>, current: Option<&Arc<Prepared>>) -> bool {
    match (old, current) {
        (None, None) => true,
        (Some(old), Some(current)) => Arc::ptr_eq(old, current),
        _ => false,
    }
}

pub(super) struct Work {
    scratch: Box<Workspace>,
    contributions: Vec<f64>,
    seed: Vec<f64>,
    binding: Option<Arc<Prepared>>,
}
impl Call {
    pub(super) fn has_source(&self) -> bool {
        self.source.is_some()
    }
    pub(super) fn source_definition(&self, name: &str) -> String {
        let body = self.source.as_ref().expect("checked block source")();
        format!(
            "fn {name}(args:&[&[f64]],out:&mut[f64],seed:Option<&[f64]>,scores:&mut[f64])->bool {{ {body} }}\n"
        )
    }
    pub(super) fn source_score_slots(&self) -> usize {
        self.scores
    }
    pub(super) fn supports_jvp(&self) -> bool {
        self.jvp.is_some()
    }
    pub(super) fn supports_curvature(&self) -> bool {
        self.vjp_jvp.is_some()
    }
    pub(super) fn arguments(&self) -> &[Value] {
        &self.inputs
    }
    pub(super) fn directional(
        &self,
        (nodes, values): (&[Node], &[f64]),
        (ranges, tangents): (&[Option<Range<usize>>], &[f64]),
        primal: &[f64],
        output: &mut [f64],
        work: &Work,
    ) -> Result<(), E> {
        output.fill(f64::NAN);
        self.jvp.as_ref().expect("compiled JVP")(
            BlockInputs {
                arguments: &self.inputs,
                nodes,
                values: BlockValues::Arena(lowering::PrimalView::whole(values)),
                prepared: work.binding.as_deref(),
                evaluation_prepared: None,
            },
            BlockTangents {
                arguments: &self.inputs,
                nodes,
                ranges,
                values: tangents,
            },
            primal,
            output,
            work.scratch.as_ref(),
        )
    }
    pub(super) fn reverse_direction(
        &self,
        (nodes, values): (&[Node], &[f64]),
        (ranges, tangents): (&[Option<Range<usize>>], &[f64]),
        primal: &[f64],
        seeds: [&[f64]; 2],
        work: &mut Work,
    ) -> Result<(), E> {
        if seeds.iter().any(|seed| seed.iter().any(|x| !x.is_finite())) {
            return Err(E::Numerical);
        }
        work.contributions.fill(f64::NAN);
        self.vjp_jvp.as_ref().expect("compiled VJP direction")(
            BlockInputs {
                arguments: &self.inputs,
                nodes,
                values: BlockValues::Arena(lowering::PrimalView::whole(values)),
                prepared: work.binding.as_deref(),
                evaluation_prepared: None,
            },
            BlockTangents {
                arguments: &self.inputs,
                nodes,
                ranges,
                values: tangents,
            },
            primal,
            seeds,
            &mut work.contributions,
            work.scratch.as_ref(),
        )?;
        if work.contributions.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(())
    }
    pub(super) fn workspace(&self) -> Work {
        Work {
            scratch: (self.workspace)(),
            contributions: vec![0.; self.scores],
            seed: vec![0.; self.output],
            binding: None,
        }
    }
    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    pub(super) fn forward(
        &self,
        nodes: &[Node],
        node: &Node,
        values: &mut [f64],
        w: &mut Work,
        prepared: Option<&Arc<Prepared>>,
        evaluation_prepared: Option<&Prepared>,
    ) -> Result<(), E> {
        if !same_preparation(w.binding.as_ref(), prepared) {
            w.binding = prepared.cloned();
        }
        let (inputs, output) = lowering::PrimalView::split(values, node.primal_range());
        output.fill(f64::NAN);
        (self.forward)(
            BlockInputs {
                arguments: &self.inputs,
                nodes,
                values: BlockValues::Arena(inputs),
                prepared: prepared.map(AsRef::as_ref),
                evaluation_prepared,
            },
            output,
            w.scratch.as_mut(),
        )
    }
    pub(super) fn reverse(
        &self,
        nodes: &[Node],
        residuals: lowering::NativePrimals<'_>,
        seed: &[f64],
        w: &mut Work,
        prepared: Option<&Arc<Prepared>>,
    ) -> Result<(), E> {
        if !same_preparation(w.binding.as_ref(), prepared) {
            return Err(E::Shape);
        }
        let vjp = self
            .vjp
            .as_ref()
            .ok_or_else(|| E::Factor(Box::new(MissingVjp)))?;
        w.seed.copy_from_slice(seed);
        w.contributions.fill(f64::NAN);
        vjp(
            BlockInputs {
                arguments: &self.inputs,
                nodes,
                values: BlockValues::Residuals(residuals.inputs),
                prepared: prepared.map(AsRef::as_ref),
                evaluation_prepared: None,
            },
            residuals.output.ok_or(E::Shape)?,
            &w.seed,
            &mut w.contributions,
            w.scratch.as_mut(),
        )?;
        if w.contributions.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(())
    }
    // Local contributions are complete/admitted BEFORE graph accumulation. Each
    // occurrence has a separate range, including repeated argument handles.
    pub(super) fn accumulate(&self, w: &Work, mut add: impl FnMut(Value, &[f64])) {
        for (&input, range) in self.inputs.iter().zip(&self.contributions) {
            if !range.is_empty() {
                add(input, &w.contributions[range.clone()]);
            }
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("native block operation has no VJP rule")]
struct MissingVjp;

pub(super) fn missing_source(plan: &Plan) -> Option<usize> {
    plan.facts.missing_source
}
// Native callbacks have no purity/replay contract, regardless of VJP support.
pub(super) fn blocks_replay(plan: &Plan) -> bool {
    plan.facts.blocks_replay()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn shape(n: usize) -> Shape {
        Shape::new(vec![n]).unwrap()
    }
    fn ty(dtype: ElementType, shape: Shape) -> BlockType {
        BlockType { dtype, shape }
    }
    fn unary_signature(n: usize) -> Signature {
        Signature {
            inputs: vec![ty(ElementType::Real, shape(n))],
            output: ty(ElementType::Real, shape(n)),
            scratch: 1,
        }
    }
    fn identity() -> Kernel {
        Kernel {
            forward: Box::new(|x, out, _| {
                out.copy_from_slice(x.real(0)?);
                Ok(())
            }),
            vjp: Some(Box::new(|_, _, seed, g, _| {
                g.copy_from_slice(seed);
                Ok(())
            })),
        }
    }
    fn selected(i: usize) -> bool {
        i % 3 != 1 && i.is_multiple_of(2) && i.is_multiple_of(5)
    }
    fn model(n: usize) -> (TensorProgram, Arc<AtomicUsize>) {
        let mut b = RegionBuilder::new();
        let x = b.input("x", shape(n)).unwrap();
        let neg = b.unary(UnaryOp::Neg, x).unwrap();
        let squared = b.binary(BinaryOp::Multiply, neg, neg).unwrap();
        let mask = b
            .typed_data(shape(n), Data::Bool((0..n).map(|i| i % 3 != 1).collect()))
            .unwrap();
        let counts = b
            .typed_data(
                shape(n),
                Data::Count(
                    (0..n)
                        .map(|i| {
                            if i.is_multiple_of(2) {
                                u64::MAX
                            } else {
                                u64::MAX - 1
                            }
                        })
                        .collect(),
                ),
            )
            .unwrap();
        let indices = b
            .indices(
                shape(n),
                (0..n)
                    .map(|i| {
                        if i.is_multiple_of(5) {
                            usize::MAX - 1
                        } else {
                            usize::MAX - 2
                        }
                    })
                    .collect(),
                usize::MAX,
            )
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let signature = Signature {
            inputs: [
                ElementType::Real,
                ElementType::Real,
                ElementType::Bool,
                ElementType::Count,
                ElementType::Index,
            ]
            .into_iter()
            .map(|dtype| ty(dtype, shape(n)))
            .collect(),
            output: ty(ElementType::Real, shape(n)),
            scratch: 1,
        };
        let result = b
            .register(
                signature,
                &[squared, squared, mask, counts, indices],
                || {
                    Ok(Kernel {
                        forward: Box::new(move |x, out, scratch| {
                            count.fetch_add(1, Ordering::Relaxed);
                            scratch[0] = 1.; // Reinitialize, regardless of previous execution.
                            let (a, b, m, c, ix) = (
                                x.real(0)?,
                                x.real(1)?,
                                x.booleans(2)?,
                                x.counts(3)?,
                                x.indices(4)?,
                            );
                            for (i, out) in out.iter_mut().enumerate() {
                                *out = if m[i] && c[i] == u64::MAX && ix[i] == usize::MAX - 1 {
                                    a[i] * b[i]
                                } else {
                                    a[i] + b[i]
                                };
                            }
                            Ok(())
                        }),
                        vjp: Some(Box::new(move |x, primal, seed, g, scratch| {
                            let (a, b, m, c, ix) = (
                                x.real(0)?,
                                x.real(1)?,
                                x.booleans(2)?,
                                x.counts(3)?,
                                x.indices(4)?,
                            );
                            if primal.len() != n || g.len() != 2 * n {
                                return Err(E::Shape);
                            }
                            let (ga, gb) = g.split_at_mut(n);
                            for i in 0..n {
                                let multiply = m[i] && c[i] == u64::MAX && ix[i] == usize::MAX - 1;
                                ga[i] = seed[i] * if multiply { b[i] } else { scratch[0] };
                                gb[i] = seed[i] * if multiply { a[i] } else { scratch[0] };
                            }
                            Ok(())
                        })),
                    })
                },
            )
            .unwrap();
        let total = b.sum(result).unwrap();
        (b.compile(total).unwrap(), calls)
    }

    #[test]
    fn typed_block_and_opaque_vjp_compose_on_existing_ssa_without_scalarizing() {
        let mut node_count = None;
        for n in [0, 1, 3, 17, 257] {
            let (program, calls) = model(n);
            assert_eq!(
                *node_count.get_or_insert(program.node_count()),
                program.node_count()
            );
            assert_eq!(
                program
                    .nodes()
                    .filter(|node| node.operation == TensorOperation::Registered)
                    .count(),
                1
            );
            let native = program
                .plan
                .nodes
                .iter()
                .find_map(|node| match &node.op {
                    Op::Registered(op) => match &**op {
                        Operation::Native(call) => Some(call),
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap();
            assert_eq!(native.inputs[0], native.inputs[1]);
            assert_eq!(native.scores, 2 * n); // Discrete arguments have no cotangents.
            assert_eq!(
                program
                    .plan
                    .nodes
                    .iter()
                    .filter(|node| node.discrete())
                    .count(),
                3
            );
            assert!(
                program
                    .plan
                    .nodes
                    .iter()
                    .filter(|node| node.discrete())
                    .all(|node| node.range().is_empty())
            );
            let p = program.bind(&[]).unwrap();
            let mut w = p.workspace();
            let x: Vec<_> = (0..n).map(|i| 0.1 + (i % 7) as f64 * 0.07).collect();
            let expected: f64 = x
                .iter()
                .enumerate()
                .map(|(i, x)| if selected(i) { x.powi(4) } else { 2. * x * x })
                .sum();
            let mut out = [91.];
            let mut g = vec![92.; n];
            for seed in [0., 1., -0.7] {
                let before = calls.load(Ordering::Relaxed);
                p.value_and_vjp(&x, &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!(calls.load(Ordering::Relaxed), before + 1); // Not two forward scans.
                assert!((out[0] - expected).abs() < 2e-10 * (1. + expected.abs()));
                for (i, (&x, &g)) in x.iter().zip(&g).enumerate() {
                    let reference = seed * if selected(i) { 4. * x.powi(3) } else { 4. * x };
                    assert!(
                        (g - reference).abs() < 2e-12 * (1. + reference.abs()),
                        "{i}: {g} vs {reference}"
                    );
                }
            }
            let wrong = vec![0.; n + 1];
            let before = calls.load(Ordering::Relaxed);
            assert!(matches!(
                p.value_and_vjp(&wrong, &[1.], &mut out, &mut g, &mut w),
                Err(E::Shape)
            ));
            if n != 0 {
                let mut invalid = x.clone();
                invalid[0] = f64::NAN;
                assert!(matches!(
                    p.value_and_vjp(&invalid, &[1.], &mut out, &mut g, &mut w),
                    Err(E::Numerical)
                ));
            }
            assert!(matches!(
                p.value_and_vjp(&x, &[f64::NAN], &mut out, &mut g, &mut w),
                Err(E::Numerical)
            ));
            assert_eq!(calls.load(Ordering::Relaxed), before);
            let allocation = allocation_counter::measure(|| {
                for _ in 0..8 {
                    p.value_and_vjp(&x, &[1.], &mut out, &mut g, &mut w)
                        .unwrap();
                }
            });
            assert_eq!(allocation.count_total, 0);
            let bytes = p.workspace_bytes();
            assert!(p.workspace_with_limit(bytes - 1).is_err());
            let measured = allocation_counter::measure(|| {
                let _work = p.workspace();
            });
            assert!(measured.bytes_max as usize <= bytes);
            if n == 3 {
                use crate::model::{ConstrainedModel, assembly::ModelBuilder};
                let mut builder = ModelBuilder::new();
                let params: Vec<_> = (0..n)
                    .map(|i| builder.real(&format!("x{i}")).unwrap().expr())
                    .collect();
                let factor = TensorFactor::new(p.clone()).unwrap();
                assert!(factor.source_rule().is_none());
                let root = builder.primitive(&params, factor).unwrap();
                builder.potential("deterministic", root).unwrap();
                let model = builder.compile().unwrap();
                let mut expected_gradient = vec![0.; n];
                p.value_and_vjp(&x, &[1.], &mut out, &mut expected_gradient, &mut w)
                    .unwrap();
                let value = model.logp_grad(&x, &mut g).unwrap();
                assert_eq!(value.to_bits(), out[0].to_bits());
                assert_eq!(g, expected_gradient);
            }
            // Independent central differences at several scales, not shared VJP math.
            for i in [0, n / 2, n.saturating_sub(1)] {
                if n == 0 {
                    break;
                }
                for h in [1e-4, 1e-5, 1e-6] {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[i] += h;
                    minus[i] -= h;
                    let mut a = [0.];
                    let mut b = [0.];
                    p.evaluate(&plus, &mut a, &mut w).unwrap();
                    p.evaluate(&minus, &mut b, &mut w).unwrap();
                    assert!(
                        ((a[0] - b[0]) / (2. * h) - g[i]).abs() < 3e-7,
                        "n{n}, coordinate{i}, h{h}"
                    );
                }
            }
        }
    }

    #[test]
    fn registration_rejects_foreign_dtype_shape_arity_and_overflow_before_preparation_or_mutation()
    {
        let mut b = RegionBuilder::new();
        let x = b.input("x", shape(3)).unwrap();
        let other = b.input("other", shape(2)).unwrap();
        let same_elements = b.input("matrix", Shape::new(vec![1, 3]).unwrap()).unwrap();
        let scalar = b.input("scalar", Shape::scalar()).unwrap();
        let mask = b
            .typed_data(shape(3), Data::Bool(vec![true; 3].into()))
            .unwrap();
        let foreign = RegionBuilder::new().input("foreign", shape(3)).unwrap();
        let calls = AtomicUsize::new(0);
        for (args, foreign_error) in [
            (vec![foreign], true),
            (
                vec![Value {
                    owner: b.owner,
                    node: usize::MAX,
                }],
                true,
            ),
            (vec![mask], false),
            (vec![other], false),
            (vec![same_elements], false),
            (vec![], false),
            (vec![x, x], false),
        ] {
            let before = (b.nodes.len(), b.slots);
            let error = b
                .register(unary_signature(3), &args, || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(identity())
                })
                .unwrap_err();
            if foreign_error {
                assert!(matches!(error, E::ForeignHandle));
            } else {
                assert!(matches!(error, E::Shape));
            }
            assert_eq!((b.nodes.len(), b.slots), before);
        }
        assert!(matches!(
            b.register(unary_signature(1), &[scalar], || panic!(
                "rank-zero is not rank-one"
            )),
            Err(E::Shape)
        ));
        let mut bad = unary_signature(3);
        bad.scratch = usize::MAX;
        assert!(matches!(
            b.register(bad, &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        let mut bad = unary_signature(3);
        bad.output.dtype = ElementType::Bool;
        assert!(matches!(
            b.register(bad, &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        // The same handle in two REAL positions is legal; a repeated handle
        // cannot pretend to satisfy an INDEX argument with identical shape.
        let bad = Signature {
            inputs: vec![
                ty(ElementType::Real, shape(3)),
                ty(ElementType::Index, shape(3)),
            ],
            output: ty(ElementType::Real, shape(3)),
            scratch: 0,
        };
        assert!(matches!(
            b.register(bad, &[x, x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let old = b.slots;
        b.slots = isize::MAX as usize / size_of::<f64>();
        assert!(matches!(
            b.register(unary_signature(3), &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        b.slots = old;
        let before = (b.nodes.len(), b.slots);
        assert!(matches!(
            b.register(unary_signature(3), &[x], || Err(E::Name)),
            Err(E::Name)
        ));
        assert_eq!((b.nodes.len(), b.slots), before);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let y = b
            .register(unary_signature(3), &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut out = [0.; 3];
        let mut inputs = vec![0.; p.dimension()];
        inputs[..3].copy_from_slice(&[1., 2., 3.]);
        p.evaluate(&inputs, &mut out, &mut p.workspace()).unwrap();
        assert_eq!(out, [1., 2., 3.]);
        let mut large = RegionBuilder::new();
        let n = isize::MAX as usize / size_of::<f64>() / 2 + 1;
        let huge = large.input("huge", shape(n)).unwrap(); // Metadata only, no dense allocation.
        let sig = Signature {
            inputs: vec![ty(ElementType::Real, shape(n)); 2],
            output: ty(ElementType::Real, Shape::scalar()),
            scratch: 0,
        };
        assert!(matches!(
            large.register(sig, &[huge, huge], || panic!(
                "overflow must precede preparation"
            )),
            Err(E::Shape)
        ));
        assert!(Shape::new(vec![usize::MAX, 2]).is_err());
    }

    #[test]
    fn discrete_data_cannot_leak_into_real_builtins_or_get_rounded() {
        let mut b = RegionBuilder::new();
        let scalar = Shape::scalar();
        let count = b
            .typed_data(scalar.clone(), Data::Count(vec![u64::MAX].into()))
            .unwrap();
        let mask = b
            .typed_data(scalar.clone(), Data::Bool(vec![true].into()))
            .unwrap();
        let index = b
            .indices(scalar.clone(), vec![usize::MAX - 1], usize::MAX)
            .unwrap();
        let x = b.input("x", scalar.clone()).unwrap();
        let before = (b.nodes.len(), b.slots);
        assert!(matches!(b.unary(UnaryOp::Neg, count), Err(E::Shape)));
        assert!(matches!(b.binary(BinaryOp::Add, mask, x), Err(E::Shape)));
        assert!(matches!(b.sum(index), Err(E::Shape)));
        assert!(matches!(b.reshape(count, shape(1)), Err(E::Shape)));
        assert!(matches!(
            b.gather(count, scalar.clone(), vec![0]),
            Err(E::Shape)
        ));
        assert!(matches!(
            b.typed_data(shape(2), Data::Count(vec![1].into())),
            Err(E::Shape)
        ));
        assert!(matches!(
            b.indices(scalar.clone(), vec![3], 3),
            Err(E::Shape)
        ));
        assert_eq!((b.nodes.len(), b.slots), before);
        assert_eq!(BlockType::of(b.node(count).unwrap()).shape, scalar);
        let sig = Signature {
            inputs: vec![
                ty(ElementType::Count, scalar.clone()),
                ty(ElementType::Bool, scalar.clone()),
                ty(ElementType::Index, scalar.clone()),
            ],
            output: ty(ElementType::Real, scalar),
            scratch: 0,
        };
        let y = b
            .register(sig, &[count, mask, index], || {
                Ok(Kernel {
                    forward: Box::new(|x, out, _| {
                        assert_eq!(x.counts(0)?, [u64::MAX]);
                        assert_eq!(x.indices(2)?, [usize::MAX - 1]);
                        assert!(matches!(x.real(0), Err(E::Shape)));
                        assert!(matches!(x.counts(2), Err(E::Shape)));
                        assert!(matches!(x.real(9), Err(E::Shape)));
                        out[0] = if x.booleans(1)?[0] { 1. } else { 0. };
                        Ok(())
                    }),
                    vjp: None,
                })
            })
            .unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut out = [0.];
        // Unused real input remains in the fixed input schema, not a discrete tangent.
        let mut g = [17.];
        p.value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut p.workspace())
            .unwrap();
        assert_eq!(out, [1.]);
        assert_eq!(g, [0.]);
        let mut b = RegionBuilder::new();
        let v = b
            .typed_data(Shape::scalar(), Data::Bool(vec![true].into()))
            .unwrap();
        assert!(matches!(b.compile(v), Err(E::Shape)));
    }

    #[test]
    fn opaque_failures_preserve_caller_outputs_and_recover_even_with_zero_seeds() {
        let mode = Arc::new(AtomicUsize::new(0));
        let f = Arc::clone(&mode);
        let r = Arc::clone(&mode);
        let mut b = RegionBuilder::new();
        let x = b.input("x", shape(1)).unwrap();
        let y = b
            .register(unary_signature(1), &[x], || {
                Ok(Kernel {
                    forward: Box::new(move |x, out, scratch| {
                        scratch[0] = f64::NAN;
                        match f.load(Ordering::Relaxed) {
                            1 => return Err(E::Shape),
                            2 => return Ok(()),
                            6 => panic!("opaque callback"),
                            _ => {}
                        }
                        scratch[0] = x.real(0)?[0];
                        out[0] = scratch[0] * scratch[0];
                        Ok(())
                    }),
                    vjp: Some(Box::new(move |_, _, seed, g, scratch| {
                        match r.load(Ordering::Relaxed) {
                            3 => return Err(E::Name),
                            4 => return Ok(()),
                            5 => {
                                g[0] = f64::INFINITY;
                                return Ok(());
                            }
                            _ => {}
                        }
                        g[0] = 2. * scratch[0] * seed[0];
                        Ok(())
                    })),
                })
            })
            .unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        for failure in 1..=6 {
            for seed in [0., 1.] {
                mode.store(failure, Ordering::Relaxed);
                let mut out = [19.];
                let mut g = [23.];
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    p.value_and_vjp(&[0.5], &[seed], &mut out, &mut g, &mut w)
                }));
                match failure {
                    1 => assert!(matches!(result, Ok(Err(E::Shape)))),
                    3 => assert!(matches!(result, Ok(Err(E::Name)))),
                    6 => assert!(result.is_err()),
                    _ => assert!(matches!(result, Ok(Err(E::Numerical)))),
                }
                assert_eq!(out, [19.]);
                assert_eq!(g, [23.]);
                mode.store(0, Ordering::Relaxed);
                p.value_and_vjp(&[0.75], &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!(out, [0.5625]);
                assert_eq!(g, [1.5 * seed]);
            }
        }
    }

    #[test]
    fn missing_vjp_is_structural_not_a_fake_zero_gradient_or_source_capability() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", shape(1)).unwrap();
        let y = b
            .register(unary_signature(1), &[x], || {
                let mut k = identity();
                k.vjp = None;
                Ok(k)
            })
            .unwrap();
        let total = b.sum(y).unwrap();
        let p = b.compile(total).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        let mut out = [17.];
        let mut g = [19.];
        p.evaluate(&[2.], &mut out, &mut w).unwrap();
        assert_eq!(out, [2.]);
        let error = p
            .value_and_vjp(&[2.], &[0.], &mut out, &mut g, &mut w)
            .unwrap_err();
        assert!(matches!(error,E::Factor(e) if e.downcast_ref::<MissingVjp>().is_some()));
        assert_eq!(out, [2.]);
        assert_eq!(g, [19.]);
        assert!(matches!(
            p.pullback_source("unsupported"),
            Err(E::MissingPrimal { .. })
        ));
        assert!(matches!(p.rust_function(), Err(E::MissingPrimal { .. })));
        let mut model = super::super::super::assembly::ModelBuilder::new();
        let x = model.real("x").unwrap().expr();
        let factor = TensorFactor::new(p).unwrap();
        assert!(factor.source_rule().is_none());
        let root = model.primitive(&[x], factor).unwrap();
        model.potential("deterministic", root).unwrap();
        assert!(matches!(
            model
                .compile()
                .unwrap()
                .source_primal(crate::model::assembly::SourceMode::Density),
            Err(E::MissingPrimal { .. })
        ));
    }

    #[test]
    fn lazy_native_regions_skip_unchosen_opaque_failures_and_recover_from_selected_errors_and_unwind()
     {
        let mode = Arc::new(AtomicUsize::new(1));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut branch = RegionBuilder::new();
        let x = branch.input("x", Shape::scalar()).unwrap();
        let flag = Arc::clone(&mode);
        let count = Arc::clone(&calls);
        let out = branch
            .register(
                Signature {
                    inputs: vec![ty(ElementType::Real, Shape::scalar())],
                    output: ty(ElementType::Real, Shape::scalar()),
                    scratch: 0,
                },
                &[x],
                || {
                    let mut k = identity();
                    k.forward = Box::new(move |x, out, _| {
                        count.fetch_add(1, Ordering::Relaxed);
                        match flag.load(Ordering::Relaxed) {
                            1 => return Err(E::Name),
                            2 => panic!("selected opaque region"),
                            _ => {}
                        }
                        out.copy_from_slice(x.real(0)?);
                        Ok(())
                    });
                    Ok(k)
                },
            )
            .unwrap();
        let yes = branch.compile(out).unwrap().bind(&[]).unwrap();
        let mut branch = RegionBuilder::new();
        let x = branch.input("x", Shape::scalar()).unwrap();
        let no = branch.compile(x).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
        let out = b.cond(x, zero, Comparison::Greater, &[x], yes, no).unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        for failure in [1, 2] {
            for seed in [0., 1.] {
                mode.store(failure, Ordering::Relaxed);
                let (mut out, mut g) = ([91.], [92.]);
                let before = calls.load(Ordering::Relaxed);
                p.value_and_vjp(&[-2.], &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!((out, g), ([-2.], [seed]));
                assert_eq!(calls.load(Ordering::Relaxed), before);
                let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    p.value_and_vjp(&[2.], &[seed], &mut out, &mut g, &mut w)
                }));
                if failure == 1 {
                    assert!(matches!(failed, Ok(Err(E::Name))));
                } else {
                    assert!(failed.is_err());
                }
                assert_eq!((out, g), ([-2.], [seed]));
                mode.store(0, Ordering::Relaxed);
                p.value_and_vjp(&[3.], &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!((out, g), ([3.], [seed]));
            }
        }
        assert!(matches!(
            p.pullback_source("native_only"),
            Err(E::MissingPrimal { .. })
        ));
    }

    #[test]
    fn unqualified_opaque_callbacks_cannot_be_silently_checkpoint_replayed() {
        let mut body = RegionBuilder::new();
        let carry = body.input("carry", Shape::scalar()).unwrap();
        let _ = body.input("row", Shape::scalar()).unwrap();
        let signature = Signature {
            inputs: vec![ty(ElementType::Real, Shape::scalar())],
            output: ty(ElementType::Real, Shape::scalar()),
            scratch: 0,
        };
        let output = body
            .register(signature, &[carry], || Ok(identity()))
            .unwrap();
        let body = body.compile(output).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let initial = b.input("initial", Shape::scalar()).unwrap();
        let xs = b.constant(shape(128), vec![0.; 128]).unwrap();
        let before = (b.nodes.len(), b.slots);
        assert!(matches!(
            b.checkpointed_scan(initial, xs, &[], body.clone()),
            Err(E::Shape)
        ));
        assert!(matches!(
            b.scan_with_replay_budget(initial, xs, &[], body.clone(), usize::MAX, 128),
            Err(E::ScanReplayUnavailable)
        ));
        assert_eq!((b.nodes.len(), b.slots), before);
        // Rejection of replay is not rejection of ordinary native composition.
        let output = b.scan(initial, xs, &[], body).unwrap();
        let p = b.compile(output).unwrap().bind(&[]).unwrap();
        let mut out = [0.];
        let mut g = [0.];
        p.value_and_vjp(&[0.75], &[2.], &mut out, &mut g, &mut p.workspace())
            .unwrap();
        assert_eq!(out, [0.75]);
        assert_eq!(g, [2.]);
        let factor = TensorFactor::new(p.clone()).unwrap();
        assert!(factor.source_rule().is_none());
        assert!(matches!(
            p.pullback_source("nested_opaque"),
            Err(E::MissingPrimal { .. })
        ));
    }

    #[test]
    fn repeated_argument_contributions_keep_declared_order_before_graph_accumulation() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let signature = Signature {
            inputs: vec![ty(ElementType::Real, Shape::scalar()); 3],
            output: ty(ElementType::Real, Shape::scalar()),
            scratch: 0,
        };
        let y = b
            .register(signature, &[x, x, x], || {
                Ok(Kernel {
                    forward: Box::new(|args, out, _| {
                        out[0] =
                            (1e16 * args.real(0)?[0] + args.real(1)?[0]) - 1e16 * args.real(2)?[0];
                        Ok(())
                    }),
                    vjp: Some(Box::new(|_, _, seed, g, _| {
                        for (g, score) in g.iter_mut().zip([1e16, 1., -1e16]) {
                            *g = score * seed[0];
                        }
                        Ok(())
                    })),
                })
            })
            .unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut out = [0.];
        let mut g = [0.];
        p.value_and_vjp(&[1.], &[1.], &mut out, &mut g, &mut p.workspace())
            .unwrap();
        let mut ordered = 0.0_f64;
        for contribution in [1e16, 1., -1e16] {
            ordered += contribution;
        }
        // This is an ORDER oracle, not a claim of exact real-arithmetic scores.
        assert_eq!(g[0].to_bits(), ordered.to_bits());
    }
}
