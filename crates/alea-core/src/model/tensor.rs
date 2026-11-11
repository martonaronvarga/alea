//! Shape-specialized tensor SSA regions, immutable data bindings and block VJPs.
//!
//! This is a deterministic numerical IR, not a generative language. TensorFactor
//! connects scalar results to the existing model tape; no inference algorithm,
//! probability distribution or parameter chart is reimplemented here.
use super::assembly::{
    ArgumentShape, AssemblyError as E, PrimitiveArgument, PrimitiveSignature, Value,
    next_value_owner,
};
use crate::model::assembly::{NativeRules, OperationDescriptor, SourceBody, SourceRule};
use std::{ops::Range, sync::Arc};
mod execute;
mod fusion;
mod kernels;
mod pullback;
mod registration;
mod source;
pub use alea_math::gemm::MatrixKernel;
pub use execute::TensorWorkspace;
pub use registration::{BlockInputs, BlockRules, BlockType, BlockVjp, ElementType};

/// Validated dense row-major tensor shape. An empty axis list is a scalar;
/// zero-length axes are allowed. Shapes are fixed throughout one compiled plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    axes: Arc<[usize]>,
    elements: usize,
}
impl Shape {
    /// Checks element-count and byte-size overflow before any tensor allocation.
    /// # Errors
    /// Rejects shapes whose dense f64 storage is not addressable.
    pub fn new(axes: impl Into<Vec<usize>>) -> Result<Self, E> {
        let axes = axes.into();
        let elements = axes
            .iter()
            .try_fold(1_usize, |n, &axis| n.checked_mul(axis).ok_or(E::Shape))?;
        if elements > isize::MAX as usize / size_of::<f64>() {
            return Err(E::Shape);
        }
        Ok(Self {
            axes: axes.into(),
            elements,
        })
    }
    /// Shape of a scalar, with one stored value.
    pub fn scalar() -> Self {
        Self {
            axes: Arc::from([]),
            elements: 1,
        }
    }
    /// Ordered axis extents, not probability batch/event declarations.
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
    /// Number of row-major stored elements.
    pub fn elements(&self) -> usize {
        self.elements
    }
}
/// Elementwise elementary operation. Domain errors remain explicit; no clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Log1p,
    Expm1,
    Tanh,
    Sin,
    Cos,
    Sqrt,
}
/// Elementwise binary operation with NumPy-style trailing-axis broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}
/// Scalar branch comparison. Reverse AD follows the executed branch; it does not
/// differentiate a predicate or certify smoothness at a switching surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}
/// Stable operation identity for cold graph inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorOperation {
    Input,
    Data,
    Constant,
    Unary(UnaryOp),
    Binary(BinaryOp),
    Sum,
    Reshape,
    Gather,
    Matmul,
    Cond,
    Scan,
    Registered,
}
/// Cold SSA node description; input handles preserve sharing and order.
pub struct TensorNode<'a> {
    /// Value produced by this node.
    pub value: Value,
    /// Fixed output storage shape.
    pub shape: &'a Shape,
    /// Block operation, not one scalar tape instruction per element.
    pub operation: TensorOperation,
    /// Ordered dependencies in this region. Child-region inputs are explicit operands.
    pub inputs: Vec<Value>,
    /// Reachable from the compiled result, including dependencies of either branch.
    pub active: bool,
}
#[derive(Clone)]
enum Map {
    Empty,
    Identity,
    Scalar,
    Broadcast(Arc<[BroadcastAxis]>),
}
// Read maps may repeat indices. They are NOT independent write maps: the VJP
// must still accumulate repeated destinations in the original element order.
#[derive(Clone, Copy)]
struct BroadcastAxis {
    extent: usize,
    output_stride: usize,
    input_stride: usize,
}
impl Map {
    // Within this many output elements the input index is constant or advances
    // by one. Native kernels unrank only at run boundaries, not per element.
    fn run(&self) -> (usize, usize) {
        match self {
            Self::Empty => (1, 0),
            Self::Identity => (usize::MAX, 1),
            Self::Scalar => (usize::MAX, 0),
            Self::Broadcast(axes) => {
                let inner = axes
                    .first()
                    .expect("nonempty broadcast has a nonunit input axis");
                if inner.output_stride == 1 {
                    (inner.extent, 1)
                } else {
                    (inner.output_stride, 0)
                }
            }
        }
    }
    #[inline]
    fn at(&self, i: usize) -> usize {
        match self {
            Self::Empty => unreachable!("empty broadcast domain has no indices"),
            Self::Identity => i,
            Self::Scalar => 0,
            Self::Broadcast(axes) => axes
                .iter()
                .map(|axis| (i / axis.output_stride % axis.extent) * axis.input_stride)
                .sum(),
        }
    }
    // Both primal and explicit pullback source lower the SAME checked map.
    // This emits O(rank) code, never an O(output elements) integer literal.
    fn index_source(&self) -> String {
        match self {
            Self::Empty => "0".into(),
            Self::Identity => "j".into(),
            Self::Scalar => "0".into(),
            Self::Broadcast(axes) => axes
                .iter()
                .map(|axis| {
                    let coordinate = if axis.output_stride == 1 {
                        format!("j%{}", axis.extent)
                    } else {
                        format!("j/{}%{}", axis.output_stride, axis.extent)
                    };
                    if axis.input_stride == 1 {
                        coordinate
                    } else {
                        format!("({coordinate})*{}", axis.input_stride)
                    }
                })
                .collect::<Vec<_>>()
                .join("+"),
        }
    }
}
struct Binary {
    kind: BinaryOp,
    a: Value,
    b: Value,
    left: Map,
    right: Map,
}
struct Cond {
    lhs: Value,
    rhs: Value,
    comparison: Comparison,
    yes: BoundProgram,
    no: BoundProgram,
    args: Box<[Value]>,
}
struct Scan {
    initial: Value,
    xs: Value,
    captures: Box<[Value]>,
    body: BoundProgram,
    steps: usize,
    row: usize,
    checkpointed: bool,
}
enum Op {
    Registered(Box<registration::Operation>),
    Input(usize),
    Data(usize),
    Constant(Arc<[f64]>),
    Unary(UnaryOp, Value),
    Binary(Box<Binary>),
    Sum(Value),
    Alias(Value),
    Gather(Value, Arc<[usize]>),
    Matmul {
        a: Value,
        b: Value,
        m: usize,
        k: usize,
        n: usize,
        kernel: MatrixKernel,
    },
    Cond(Box<Cond>),
    Scan(Box<Scan>),
}
struct Node {
    shape: Shape,
    start: usize,
    op: Op,
}
impl Node {
    fn range(&self) -> Range<usize> {
        if self.discrete() {
            return self.start..self.start;
        }
        self.start..self.start + self.shape.elements
    }
    fn discrete(&self) -> bool {
        matches!(&self.op, Op::Registered(op) if matches!(&**op, registration::Operation::Data(_)))
    }
    fn native_block(&self) -> Option<&registration::Call> {
        match &self.op {
            Op::Registered(op) => match &**op {
                registration::Operation::Native(call) => Some(call),
                registration::Operation::Data(_) => None,
            },
            _ => None,
        }
    }
    fn inputs(&self) -> Vec<Value> {
        match &self.op {
            Op::Registered(op) => op.inputs().to_vec(),
            Op::Input(_) | Op::Data(_) | Op::Constant(_) => vec![],
            Op::Unary(_, a) | Op::Sum(a) | Op::Alias(a) | Op::Gather(a, _) => vec![*a],
            Op::Binary(b) => vec![b.a, b.b],
            Op::Matmul { a, b, .. } => vec![*a, *b],
            Op::Cond(c) => {
                let mut v = vec![c.lhs, c.rhs];
                v.extend_from_slice(&c.args);
                v
            }
            Op::Scan(s) => {
                let mut v = vec![s.initial, s.xs];
                v.extend_from_slice(&s.captures);
                v
            }
        }
    }
}
#[derive(Clone)]
struct Input {
    name: String,
    shape: Shape,
    start: usize,
}
struct Plan {
    owner: u64,
    nodes: Vec<Node>,
    inputs: Vec<Input>,
    data: Vec<Input>,
    dimension: usize,
    slots: usize,
    output: Value,
    active: Vec<bool>,
    workspace_bytes: usize,
    source_slots: usize,
    execution: Option<Box<fusion::Schedule>>,
}
impl Plan {
    fn last_native_block(&self) -> Option<usize> {
        self.nodes
            .iter()
            .zip(&self.active)
            .rposition(|(node, &active)| active && node.native_block().is_some())
    }
}
/// Construction of shape-checked, deterministic tensor/control-flow regions.
/// No global AD stack is used. Builder identities and escaped values are never recycled.
#[must_use]
pub struct RegionBuilder {
    owner: u64,
    nodes: Vec<Node>,
    inputs: Vec<Input>,
    data: Vec<Input>,
    dimension: usize,
    slots: usize,
    matrix_kernel: MatrixKernel,
}
impl RegionBuilder {
    pub(super) fn new() -> Self {
        Self {
            owner: next_value_owner(),
            nodes: vec![],
            inputs: vec![],
            data: vec![],
            dimension: 0,
            slots: 0,
            matrix_kernel: MatrixKernel::Compensated,
        }
    }
    /// Sets the execution policy for subsequent `matmul` operations in this
    /// lexical builder. Existing nodes and child regions retain their policies.
    /// `Auto` opts into accelerated reduction arithmetic and selects an available
    /// provider; it does not promise a cost-optimal choice or bound provider
    /// scratch/thread use. Default construction remains compensated.
    pub fn with_matrix_kernel(mut self, kernel: MatrixKernel) -> Self {
        self.matrix_kernel = kernel.resolved();
        self
    }
    fn node(&self, value: Value) -> Result<&Node, E> {
        if value.owner != self.owner {
            return Err(E::ForeignHandle);
        }
        self.nodes.get(value.node).ok_or(E::ForeignHandle)
    }
    /// Shape of a value declared or computed in this region.
    /// # Errors
    /// Rejects values from a different scope.
    pub fn value_shape(&self, value: Value) -> Result<&Shape, E> {
        Ok(&self.node(value)?.shape)
    }

    fn real_node(&self, value: Value) -> Result<&Node, E> {
        let node = self.node(value)?;
        if node.discrete() {
            return Err(E::Shape);
        }
        Ok(node)
    }
    fn push(&mut self, shape: Shape, op: Op) -> Result<Value, E> {
        let start = self.slots;
        let elements = if matches!(&op, Op::Registered(p) if matches!(&**p, registration::Operation::Data(_)))
        {
            0
        } else {
            shape.elements
        };
        let slots = start.checked_add(elements).ok_or(E::Shape)?;
        if slots > isize::MAX as usize / size_of::<f64>() {
            return Err(E::Shape);
        }
        let result = Value {
            owner: self.owner,
            node: self.nodes.len(),
        };
        self.nodes.push(Node { shape, start, op });
        self.slots = slots;
        Ok(result)
    }
    fn name(&self, name: &str) -> Result<(), E> {
        if name.is_empty() || self.inputs.iter().chain(&self.data).any(|i| i.name == name) {
            Err(E::Name)
        } else {
            Ok(())
        }
    }
    /// Declares differentiable numerical input, flattened in declaration order.
    /// # Errors
    /// Rejects duplicate/empty names or storage overflow.
    pub fn input(&mut self, name: &str, shape: Shape) -> Result<Value, E> {
        self.name(name)?;
        let end = self.dimension.checked_add(shape.elements).ok_or(E::Shape)?;
        let v = self.push(shape.clone(), Op::Input(self.inputs.len()))?;
        self.inputs.push(Input {
            name: name.into(),
            shape,
            start: self.dimension,
        });
        self.dimension = end;
        Ok(v)
    }
    /// Declares a non-differentiable, immutable data binding of fixed shape.
    /// # Errors
    /// Rejects duplicate/empty names or storage overflow.
    pub fn data(&mut self, name: &str, shape: Shape) -> Result<Value, E> {
        self.name(name)?;
        let v = self.push(shape.clone(), Op::Data(self.data.len()))?;
        self.data.push(Input {
            name: name.into(),
            shape,
            start: 0,
        });
        Ok(v)
    }
    /// Inserts owned finite row-major constants.
    /// # Errors
    /// Rejects element-count mismatch/nonfinite values before mutation.
    pub fn constant(&mut self, shape: Shape, values: Vec<f64>) -> Result<Value, E> {
        if values.len() != shape.elements {
            return Err(E::Shape);
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(E::Numerical);
        }
        self.push(shape, Op::Constant(values.into()))
    }
    /// Applies an elementary operation to an entire tensor block.
    /// # Errors
    /// Rejects foreign values or overflowing storage.
    pub fn unary(&mut self, kind: UnaryOp, value: Value) -> Result<Value, E> {
        self.push(self.real_node(value)?.shape.clone(), Op::Unary(kind, value))
    }
    /// Applies a broadcasted binary operation, with maps specialized once.
    /// # Errors
    /// Rejects incompatible shapes, foreign values or storage overflow.
    pub fn binary(&mut self, kind: BinaryOp, a: Value, b: Value) -> Result<Value, E> {
        let lhs = &self.real_node(a)?.shape;
        let rhs = &self.real_node(b)?.shape;
        let shape = broadcast(lhs, rhs)?;
        let left = map(lhs, &shape)?;
        let right = map(rhs, &shape)?;
        self.push(
            shape,
            Op::Binary(Box::new(Binary {
                kind,
                a,
                b,
                left,
                right,
            })),
        )
    }
    /// Ordered full-tensor sum, with an analytic broadcast pullback.
    /// # Errors
    /// Rejects foreign values or storage overflow.
    pub fn sum(&mut self, value: Value) -> Result<Value, E> {
        self.real_node(value)?;
        self.push(Shape::scalar(), Op::Sum(value))
    }
    /// Zero-copy reshape in the native tape; value/adjoint storage is aliased.
    /// # Errors
    /// Rejects changed element count or foreign values.
    pub fn reshape(&mut self, value: Value, shape: Shape) -> Result<Value, E> {
        let old = self.real_node(value)?;
        if old.shape.elements != shape.elements {
            return Err(E::Shape);
        }
        let start = old.start;
        let result = Value {
            owner: self.owner,
            node: self.nodes.len(),
        };
        self.nodes.push(Node {
            shape,
            start,
            op: Op::Alias(value),
        });
        Ok(result)
    }
    /// Constant-index row-major gather. Repeated indices scatter-add in reverse.
    /// # Errors
    /// Rejects invalid indices, shape mismatch or foreign values.
    pub fn gather(&mut self, value: Value, shape: Shape, indices: Vec<usize>) -> Result<Value, E> {
        let old = self.real_node(value)?;
        if indices.len() != shape.elements || indices.iter().any(|&i| i >= old.shape.elements) {
            return Err(E::Shape);
        }
        self.push(shape, Op::Gather(value, indices.into()))
    }
    /// Dense row-major matrix multiplication as one block operation. Both inputs
    /// must have rank two; arbitrary-rank elementwise broadcasting is separate.
    /// # Errors
    /// Rejects incompatible matrix dimensions or foreign values.
    pub fn matmul(&mut self, a: Value, b: Value) -> Result<Value, E> {
        self.matmul_with_kernel(a, b, self.matrix_kernel)
    }
    /// Dense product with an explicit execution policy. Faer/BLAS permit
    /// reassociated reductions in both primal and VJP; they do not promise
    /// compensated accuracy or identical trajectories. The default `matmul`
    /// remains compensated. Source export rejects accelerated providers until
    /// their source lowering is qualified; it never substitutes compensated
    /// arithmetic. Shared-storage reverse
    /// operands and zero seeds retain the scalar pullback.
    /// Workspace budgets cover Alea-owned buffers, not external provider packing
    /// or thread-local scratch. BLAS thread count must be configured by the host.
    /// # Errors
    /// Rejects incompatible matrix dimensions or foreign values.
    pub fn matmul_with_kernel(
        &mut self,
        a: Value,
        b: Value,
        kernel: MatrixKernel,
    ) -> Result<Value, E> {
        let kernel = kernel.resolved();
        let lhs = &self.real_node(a)?.shape;
        let rhs = &self.real_node(b)?.shape;
        if lhs.axes.len() != 2 || rhs.axes.len() != 2 || lhs.axes[1] != rhs.axes[0] {
            return Err(E::Shape);
        }
        let (m, k, n) = (lhs.axes[0], lhs.axes[1], rhs.axes[1]);
        self.push(
            Shape::new(vec![m, n])?,
            Op::Matmul {
                a,
                b,
                m,
                k,
                n,
                kernel,
            },
        )
    }
    fn arguments(&self, region: &BoundProgram, args: &[Value]) -> Result<(), E> {
        if args.len() != region.plan.inputs.len() {
            return Err(E::Shape);
        }
        for (&arg, input) in args.iter().zip(&region.plan.inputs) {
            if self.real_node(arg)?.shape != input.shape {
                return Err(E::Shape);
            }
        }
        Ok(())
    }
    /// Lazy scalar conditional with explicit region operands and equal branch
    /// output shapes. Only the selected branch runs, including in reverse AD.
    /// # Errors
    /// Rejects non-scalar predicates, argument/output mismatches or foreign values.
    pub fn cond(
        &mut self,
        lhs: Value,
        rhs: Value,
        comparison: Comparison,
        args: &[Value],
        yes: BoundProgram,
        no: BoundProgram,
    ) -> Result<Value, E> {
        if self.real_node(lhs)?.shape != Shape::scalar()
            || self.real_node(rhs)?.shape != Shape::scalar()
            || yes.output_shape() != no.output_shape()
        {
            return Err(E::Shape);
        }
        self.arguments(&yes, args)?;
        self.arguments(&no, args)?;
        self.push(
            yes.output_shape().clone(),
            Op::Cond(Box::new(Cond {
                lhs,
                rhs,
                comparison,
                yes,
                no,
                args: args.into(),
            })),
        )
    }
    /// Fixed-trip recurrence returning the final carry, without graph unrolling.
    /// Body arguments are carry, one leading-axis slice of xs, then captures.
    /// Retains every body frame for reverse propagation. Use
    /// [`Self::checkpointed_scan`] to trade additional computation for less native
    /// reverse storage. Zero steps return the initial carry.
    /// # Errors
    /// Rejects body/carry/input-shape mismatch, foreign values or size overflow.
    pub fn scan(
        &mut self,
        initial: Value,
        xs: Value,
        captures: &[Value],
        body: BoundProgram,
    ) -> Result<Value, E> {
        self.scan_with_storage(initial, xs, captures, body, false)
    }
    /// Fixed-trip scan with segmented native reverse-mode checkpointing.
    ///
    /// Like [`Self::scan`], but substantial tapes retain boundary carries and
    /// replay each segment during reverse propagation. For T trips, B body bytes
    /// and C carry elements, reverse storage is O(sqrt(T) * (B + C)); immutable
    /// inputs and the enclosing value buffers are additional storage. A VJP may
    /// perform one extra body forward pass per trip. Small tapes keep all frames.
    /// Arithmetic, error admission and pathwise derivatives are unchanged. Source
    /// export preserves the same loop; a source AD compiler owns its tape policy.
    /// # Errors
    /// Same shape, ownership and workspace-overflow errors as [`Self::scan`].
    pub fn checkpointed_scan(
        &mut self,
        initial: Value,
        xs: Value,
        captures: &[Value],
        body: BoundProgram,
    ) -> Result<Value, E> {
        self.scan_with_storage(initial, xs, captures, body, true)
    }
    fn scan_with_storage(
        &mut self,
        initial: Value,
        xs: Value,
        captures: &[Value],
        body: BoundProgram,
        checkpointed: bool,
    ) -> Result<Value, E> {
        let shape = self.real_node(initial)?.shape.clone();
        let data = self.real_node(xs)?.shape.clone();
        if data.axes.is_empty()
            || body.plan.inputs.len() != captures.len() + 2
            || body.output_shape() != &shape
            || body.plan.inputs[0].shape != shape
            || body.plan.inputs[1].shape != Shape::new(data.axes[1..].to_vec())?
        {
            return Err(E::Shape);
        }
        for (&arg, input) in captures.iter().zip(&body.plan.inputs[2..]) {
            if self.real_node(arg)?.shape != input.shape {
                return Err(E::Shape);
            }
        }
        let steps = data.axes[0];
        if checkpointed && steps != 0 && registration::blocks_replay(&body.plan) {
            return Err(E::Shape);
        }
        let row = body.plan.inputs[1].shape.elements;
        steps
            .checked_mul(body.plan.workspace_bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(E::Shape)?;
        self.push(
            shape,
            Op::Scan(Box::new(Scan {
                initial,
                xs,
                captures: captures.into(),
                body,
                steps,
                row,
                checkpointed,
            })),
        )
    }
    /// Compiles a fixed-shape region, pruning unreachable operations. Regions
    /// retain structured conditionals/loops; tensor elements are not scalar nodes.
    /// # Errors
    /// Rejects a foreign output value or overflowing nested workspace storage.
    pub fn compile(self, output: Value) -> Result<TensorProgram, E> {
        self.real_node(output)?;
        let mut active = vec![false; self.nodes.len()];
        active[output.node] = true;
        for (i, node) in self.nodes.iter().enumerate().rev() {
            if active[i] {
                for input in node.inputs() {
                    active[input.node] = true;
                }
            }
        }
        let mut plan = Plan {
            owner: self.owner,
            nodes: self.nodes,
            inputs: self.inputs,
            data: self.data,
            dimension: self.dimension,
            slots: self.slots,
            output,
            active,
            workspace_bytes: 0,
            source_slots: 0,
            execution: None,
        };
        let schedule = fusion::schedule(&plan, true);
        if !schedule.is_empty() {
            plan.execution = Some(Box::new(fusion::Schedule(schedule)));
        }
        plan.workspace_bytes = execute::workspace_bytes(&plan)?;
        plan.source_slots = source::workspace_elements(&plan)?;
        Ok(TensorProgram {
            plan: Arc::new(plan),
        })
    }
}
/// Shared immutable typed SSA/control-flow plan, independent of bound data/scratch.
#[derive(Clone)]
pub struct TensorProgram {
    plan: Arc<Plan>,
}
/// Immutable binding snapshot. Rebinding never changes another snapshot or target.
#[derive(Clone)]
pub struct BoundProgram {
    plan: Arc<Plan>,
    data: Vec<Arc<[f64]>>,
    #[cfg(test)]
    snapshot: Arc<registration::BindingState>,
}
/// Target-owned tensor scratch exceeds the caller's explicit per-execution budget.
/// Counts include native value/adjoint buffers and retained control-flow frames,
/// not immutable data, generated AD storage or allocator bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("tensor workspace requires {required} bytes, exceeding limit {limit}")]
pub struct WorkspaceBudgetError {
    /// Checked native workspace estimate.
    pub required: usize,
    /// Caller-supplied per-execution byte limit.
    pub limit: usize,
}
impl TensorProgram {
    /// Native per-execution workspace estimate, including retained scan frames.
    /// Excludes immutable bindings, allocator overhead and generated AD storage.
    pub fn workspace_bytes(&self) -> usize {
        self.plan.workspace_bytes
    }
    /// Binds every named data slot exactly once; values are copied once into shared
    /// immutable storage. Different bindings reuse the same compiled plan.
    /// # Errors
    /// Rejects missing/duplicate/unknown names, wrong lengths or nonfinite data.
    pub fn bind(&self, bindings: &[(&str, &[f64])]) -> Result<BoundProgram, E> {
        if bindings.len() != self.plan.data.len() {
            return Err(E::Shape);
        }
        let mut data = vec![];
        for input in &self.plan.data {
            let matches: Vec<_> = bindings
                .iter()
                .filter(|(name, _)| *name == input.name)
                .collect();
            if matches.len() != 1 {
                return Err(E::Name);
            }
            let values = matches[0].1;
            if values.len() != input.shape.elements {
                return Err(E::Shape);
            }
            if values.iter().any(|x| !x.is_finite()) {
                return Err(E::Numerical);
            }
            data.push(Arc::from(values));
        }
        #[cfg(test)]
        return registration::bind_snapshot(Arc::clone(&self.plan), data);
        #[cfg(not(test))]
        Ok(BoundProgram {
            plan: Arc::clone(&self.plan),
            data,
        })
    }
    /// Number of block SSA nodes, independent of tensor element/trip counts.
    pub fn node_count(&self) -> usize {
        self.plan.nodes.len()
    }
    /// Cold graph inspection, preserving structured region operation identity.
    pub fn nodes(&self) -> impl Iterator<Item = TensorNode<'_>> {
        self.plan.nodes.iter().enumerate().map(|(i, n)| TensorNode {
            value: Value {
                owner: self.plan.owner,
                node: i,
            },
            shape: &n.shape,
            inputs: n.inputs(),
            active: self.plan.active[i],
            operation: match &n.op {
                Op::Registered(op) => match &**op {
                    registration::Operation::Data(_) => TensorOperation::Constant,
                    registration::Operation::Native(_) => TensorOperation::Registered,
                },
                Op::Input(_) => TensorOperation::Input,
                Op::Data(_) => TensorOperation::Data,
                Op::Constant(_) => TensorOperation::Constant,
                Op::Unary(op, _) => TensorOperation::Unary(*op),
                Op::Binary(b) => TensorOperation::Binary(b.kind),
                Op::Sum(_) => TensorOperation::Sum,
                Op::Alias(_) => TensorOperation::Reshape,
                Op::Gather(..) => TensorOperation::Gather,
                Op::Matmul { .. } => TensorOperation::Matmul,
                Op::Cond(_) => TensorOperation::Cond,
                Op::Scan(_) => TensorOperation::Scan,
            },
        })
    }
}
impl BoundProgram {
    /// Fixed flattened differentiable input dimension.
    pub fn dimension(&self) -> usize {
        self.plan.dimension
    }
    pub(super) fn input_shapes(&self) -> impl ExactSizeIterator<Item = &Shape> {
        self.plan.inputs.iter().map(|input| &input.shape)
    }
    /// Shape of the result, independent of the current binding values.
    pub fn output_shape(&self) -> &Shape {
        &self.plan.nodes[self.plan.output.node].shape
    }
    /// Checks this immutable target's per-execution native workspace budget
    /// before graph registration allocates its scratch. Rebinding keeps shape
    /// and plan, hence this estimate; concurrent chains require separate budgets.
    /// # Errors
    /// Reports required/allowed bytes without allocating execution frames.
    pub fn check_workspace_limit(&self, limit: usize) -> Result<(), WorkspaceBudgetError> {
        let required = self.workspace_bytes();
        if required > limit {
            return Err(WorkspaceBudgetError { required, limit });
        }
        Ok(())
    }
    /// Returns a new same-plan snapshot, preserving all unchanged data arrays.
    /// Nested regions have their own bindings; pass parent data as operands when
    /// rebinding should affect a branch or loop body.
    /// # Errors
    /// Rejects unknown names, shape changes or nonfinite replacement values.
    pub fn rebind(&self, name: &str, values: &[f64]) -> Result<Self, E> {
        let index = self
            .plan
            .data
            .iter()
            .position(|i| i.name == name)
            .ok_or(E::Name)?;
        if values.len() != self.plan.data[index].shape.elements {
            return Err(E::Shape);
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(E::Numerical);
        }
        let mut data = self.data.clone();
        data[index] = Arc::from(values);
        #[cfg(test)]
        return registration::bind_snapshot(Arc::clone(&self.plan), data);
        #[cfg(not(test))]
        Ok(Self {
            plan: Arc::clone(&self.plan),
            data,
        })
    }
}
/// Scalar-result tensor region lowered as one native model primitive. Arbitrary
/// tensor results remain available through BoundProgram's value/VJP interface.
pub(super) struct TensorFactor {
    binding: BoundProgram,
}
impl TensorFactor {
    /// Describes a scalar deterministic region; it adds no probability role or chart.
    /// # Errors
    /// Rejects non-scalar output. Registration separately checks the flattened
    /// graph input count against the immutable binding dimension.
    pub(super) fn new(binding: BoundProgram) -> Result<Self, E> {
        if binding.output_shape() != &Shape::scalar() {
            return Err(E::Shape);
        }
        Ok(Self { binding })
    }
    /// Immutable program/data snapshot for inspection and same-plan rebinding.
    pub(super) fn binding(&self) -> &BoundProgram {
        &self.binding
    }
    /// Same-shape named data replacement, suitable for NativeModel::rebind.
    /// # Errors
    /// Preserves BoundProgram's binding validation errors.
    pub(super) fn with_data(&self, name: &str, values: &[f64]) -> Result<Self, E> {
        Ok(Self {
            binding: self.binding.rebind(name, values)?,
        })
    }
}
impl TensorFactor {
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "tensor_region",
            arguments: vec![PrimitiveArgument {
                name: "inputs",
                shape: ArgumentShape::Vector(self.binding.dimension()),
            }],
            probability: None,
        }
    }
    /// Native kernel workspace.
    pub fn workspace(&self) -> TensorWorkspace {
        self.binding.workspace()
    }
    /// Native kernel evaluate.
    pub fn evaluate(&self, x: &[f64], g: &mut [f64], w: &mut TensorWorkspace) -> Result<f64, E> {
        let mut out = [0.];
        self.binding.value_and_vjp(x, &[1.], &mut out, g, w)?;
        Ok(out[0])
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        if registration::missing_source(&self.binding.plan).is_some() {
            return None;
        }
        Some(|this| this.binding.scalar_source())
    }
    fn source_workspace(&self) -> usize {
        self.binding.plan.source_slots
    }
}
impl OperationDescriptor for TensorFactor {
    type Workspace = TensorWorkspace;
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        let body = self.source_rule().map(SourceBody::Generated);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: None,
            workspace: self.source_workspace(),
        });
        rules
    }
}
fn broadcast(a: &Shape, b: &Shape) -> Result<Shape, E> {
    let rank = a.axes.len().max(b.axes.len());
    let mut axes = vec![1; rank];
    for (i, axis) in axes.iter_mut().enumerate() {
        let x = if i + a.axes.len() >= rank {
            a.axes[i + a.axes.len() - rank]
        } else {
            1
        };
        let y = if i + b.axes.len() >= rank {
            b.axes[i + b.axes.len() - rank]
        } else {
            1
        };
        *axis = if x == y || y == 1 {
            x
        } else if x == 1 {
            y
        } else {
            return Err(E::Shape);
        };
    }
    Shape::new(axes)
}
// Called only after broadcast() has admitted the shapes. Nonempty shape products
// bound every stride and computed offset. Empty domains require no strides (and
// may have otherwise unrepresentable suffix products following a zero axis).
fn map(input: &Shape, output: &Shape) -> Result<Map, E> {
    if output.elements == 0 {
        return Ok(Map::Empty);
    }
    if input.elements == output.elements {
        return Ok(Map::Identity);
    }
    if input.elements == 1 {
        return Ok(Map::Scalar);
    }
    let leading = output.axes.len() - input.axes.len();
    let mut axes = Vec::with_capacity(input.axes.len());
    let (mut input_stride, mut output_stride) = (1_usize, 1_usize);
    for (&size, &extent) in input.axes.iter().zip(&output.axes[leading..]).rev() {
        if size != 1 {
            axes.push(BroadcastAxis {
                extent,
                output_stride,
                input_stride,
            });
        }
        input_stride = input_stride.checked_mul(size).ok_or(E::Shape)?;
        output_stride = output_stride.checked_mul(extent).ok_or(E::Shape)?;
    }
    Ok(Map::Broadcast(axes.into()))
}

#[cfg(test)]
mod indexing_tests {
    use super::*;
    use proptest::prelude::*;

    // Independent coordinate expansion, retained only as a test oracle. Do not
    // materialize this table in the planner or either source renderer.
    fn expanded(input: &Shape, output: &Shape) -> Vec<usize> {
        (0..output.elements)
            .map(|mut flat| {
                let mut coordinates = vec![0; output.axes.len()];
                for (coordinate, &extent) in coordinates.iter_mut().zip(output.axes.iter()).rev() {
                    *coordinate = flat % extent;
                    flat /= extent;
                }
                input
                    .axes
                    .iter()
                    .zip(&coordinates[output.axes.len() - input.axes.len()..])
                    .fold(0, |index, (&extent, &coordinate)| {
                        index * extent + if extent == 1 { 0 } else { coordinate }
                    })
            })
            .collect()
    }

    proptest! {
        #[test]
        fn compact_broadcast_matches_coordinate_oracle(
            axes in prop::collection::vec(0_usize..5, 0..7),
            mask in any::<u8>(),
            leading in 0_usize..7,
        ) {
            let output = Shape::new(axes.clone()).unwrap();
            let input_axes: Vec<_> = axes.iter().enumerate()
                .map(|(i, &extent)| if mask & (1 << i) == 0 { 1 } else { extent })
                .collect();
            // Only leading unit axes may be omitted under trailing broadcasting.
            let drop = input_axes.iter().take(leading).take_while(|&&n| n == 1).count();
            let input = Shape::new(input_axes[drop..].to_vec()).unwrap();
            prop_assert_eq!(broadcast(&input, &output).unwrap(), output.clone());
            let compact = map(&input, &output).unwrap();
            let expected = expanded(&input, &output);
            for (i, &expected) in expected.iter().enumerate() {
                prop_assert_eq!(compact.at(i), expected);
                prop_assert!(compact.at(i) < input.elements);
            }
            let b = Binary {
                kind: BinaryOp::Add,
                a: Value { owner: 0, node: 0 },
                b: Value { owner: 0, node: 1 },
                left: compact.clone(),
                right: Map::Identity,
            };
            // Fusion blocks can begin/end inside a broadcast run.
            for start in [0, output.elements.min(1), output.elements / 2, output.elements] {
                let len = (output.elements - start).min(257);
                let mut visited = Vec::new();
                kernels::for_each_index(&b, start, len, |j, left, right| visited.push((j, left, right)));
                prop_assert_eq!(visited.len(), len);
                for (j, left, right) in visited {
                    prop_assert_eq!(left, expected[start + j]);
                    prop_assert_eq!(right, start + j);
                }
            }
            if let Map::Broadcast(terms) = compact {
                prop_assert!(!terms.is_empty());
                prop_assert!(terms.len() <= input.axes.len());
            }
        }
    }

    #[test]
    fn map_storage_and_source_depend_on_rank_not_output_cardinality() {
        // Match the remote fixture grid without allocating tensor elements.
        // Isolate read-map ownership and index text from complete plan storage,
        // frozen literals, and primal/adjoint workspace costs.
        for n in [1, 3, 64, 128, 1024, 4096, 65_536] {
            for width in [1, 3, 8, 16] {
                let output = Shape::new(vec![n, width]).unwrap();
                for (operand, input) in [
                    ("column", Shape::new(vec![n, 1]).unwrap()),
                    ("row", Shape::new(vec![width]).unwrap()),
                ] {
                    let mut retained = None;
                    let allocation = allocation_counter::measure(|| {
                        retained = Some(map(&input, &output).unwrap());
                    });
                    let compact = retained.unwrap();
                    let source_bytes = compact.index_source().len();
                    assert!(allocation.bytes_total < 256);
                    assert!(source_bytes < 64);
                    if let Map::Broadcast(terms) = &compact {
                        assert_eq!(terms.len(), 1);
                    }
                    eprintln!(
                        "ALEA_BROADCAST_MAP_V1\t{n}x{width}\t{operand}\towned={}\tallocated={}\tindex_source={source_bytes}",
                        allocation.bytes_current, allocation.bytes_total,
                    );
                }
            }
        }
        for n in [1_000, 1_000_000] {
            let output = Shape::new(vec![n, 8]).unwrap();
            for input in [
                Shape::new(vec![n, 1]).unwrap(),
                Shape::new(vec![8]).unwrap(),
            ] {
                let mut retained = None;
                let allocation = allocation_counter::measure(|| {
                    retained = Some(map(&input, &output).unwrap());
                });
                assert!(allocation.bytes_total < 256);
                let compact = retained.unwrap();
                assert!(compact.index_source().len() < 64);
                let Map::Broadcast(terms) = compact else {
                    panic!("nontrivial broadcast")
                };
                assert_eq!(terms.len(), 1);
            }
        }
        let input = Shape::new(vec![3]).unwrap();
        assert!(matches!(
            map(&input, &Shape::new(vec![1, 1, 3]).unwrap()).unwrap(),
            Map::Identity
        ));
        // Valid empty shapes need no suffix products, divisions or index arrays.
        let empty = Shape::new(vec![0, usize::MAX, usize::MAX]).unwrap();
        let input = Shape::new(vec![0, 1, 1]).unwrap();
        assert_eq!(broadcast(&input, &empty).unwrap(), empty);
        assert!(matches!(map(&input, &empty).unwrap(), Map::Empty));
    }
}
