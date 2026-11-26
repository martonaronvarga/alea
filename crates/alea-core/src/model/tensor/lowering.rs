//! Checked physical calls for the canonical region. These are cold compilation
//! metadata, not another authoring graph or an interpreter instruction set.
use super::analysis::BackwardRead;
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Leaf,
    Unary,
    Binary,
    Sum,
    Alias,
    Gather,
    Matrix,
    Control,
    Native,
}
// Graph argument contributions keep their declared additive order. Internal
// matrix reductions still obey the separately recorded provider request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Contribution {
    None,
    OrderedAdd,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Write {
    Immutable,
    Overwrite,
    Alias(Value),
}
#[derive(Clone, PartialEq, Eq)]
enum Access {
    Whole,
    Broadcast(Map),
    Gather(Arc<[usize]>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Region {
    ty: BlockType,
    range: Range<usize>,
}
#[derive(Clone)]
pub(super) struct Read {
    pub(super) value: Value,
    region: Region,
    access: Access,
    contribution: Contribution,
}
impl Read {
    pub(super) fn elements(&self) -> usize {
        self.region.ty.shape.elements
    }

    fn broadcast(&self) -> &Map {
        let Access::Broadcast(map) = &self.access else {
            unreachable!("checked broadcast")
        };
        map
    }
}

// Borrow the semantic payload for rendering and take edges, access maps and
// matrix calls from verified lowering. No additional owned operation list.
#[derive(Clone, Copy)]
pub(super) enum Operation<'a> {
    Input(usize),
    Data(usize),
    Constant(&'a [f64]),
    Alias(Value),
    Unary(UnaryOp, Value),
    Binary(BinaryOp, (Value, &'a Map), (Value, &'a Map)),
    Sum(&'a Read),
    Gather(Value, &'a [usize]),
    Matrix(Value, Value, &'a MatrixCalls),
    Cond(&'a Cond, [Value; 2], &'a [Read]),
    Scan(&'a Scan, [Value; 2], &'a [Read]),
    Registered(&'a registration::Call, &'a [Read]),
    NativeOnly,
}

impl Operation<'_> {
    pub(super) fn child_scratch(self) -> (usize, usize) {
        match self {
            Self::Cond(c, ..) => (
                c.yes.dimension(),
                c.yes.plan.source_slots.max(c.no.plan.source_slots),
            ),
            Self::Scan(s, ..) if s.steps != 0 => (s.body.dimension(), s.body.plan.source_slots),
            _ => (0, 0),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Residuals {
    Explicit(Range<usize>),
    Opaque,
}
// This is the executor's slice access contract, not a sparsity claim about
// mathematical derivatives. In particular the existing unary and binary
// kernels receive entire operand views.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeResiduals {
    None,
    Inputs,
    Unary,
    ScanReplay,
    Callback,
}

fn native_residuals(node: &Node) -> NativeResiduals {
    match &node.op {
        Op::Unary(..) => NativeResiduals::Unary,
        Op::Binary(_) => NativeResiduals::Inputs,
        Op::Matmul { .. } if node.shape.elements != 0 => NativeResiduals::Inputs,
        Op::Scan(scan) if scan.replay.is_some() => NativeResiduals::ScanReplay,
        _ if node.native_block().is_some() => NativeResiduals::Callback,
        _ => NativeResiduals::None,
    }
}

#[derive(Clone, Copy)]
pub(super) struct PrimalView<'a> {
    before: &'a [f64],
    after: &'a [f64],
    resume: usize,
}
impl<'a> PrimalView<'a> {
    #[inline]
    pub(super) fn whole(values: &'a [f64]) -> Self {
        Self {
            before: values,
            after: &[],
            resume: values.len(),
        }
    }
    #[inline]
    pub(super) fn split(values: &'a mut [f64], range: Range<usize>) -> (Self, &'a mut [f64]) {
        let (before, tail) = values.split_at_mut(range.start);
        let (output, after) = tail.split_at_mut(range.len());
        (
            Self {
                before,
                after,
                resume: range.end,
            },
            output,
        )
    }
    #[inline]
    pub(super) fn get(self, range: Range<usize>) -> Option<&'a [f64]> {
        if range.is_empty() {
            Some(&[])
        } else if range.end <= self.before.len() {
            self.before.get(range)
        } else {
            self.after
                .get(range.start.checked_sub(self.resume)?..range.end - self.resume)
        }
    }
    #[inline]
    pub(super) fn read(self, range: Range<usize>) -> &'a [f64] {
        self.get(range).expect("checked disjoint primal range")
    }
}

#[derive(Clone, Copy)]
pub(super) struct PrimalInputs<'a> {
    reads: &'a [Read],
    values: PrimalView<'a>,
}
impl<'a> PrimalInputs<'a> {
    #[inline]
    pub(super) fn get(self, index: usize) -> Option<&'a [f64]> {
        self.values.get(self.reads.get(index)?.region.range.clone())
    }
}

pub(super) struct NativePrimals<'a> {
    pub(super) inputs: PrimalInputs<'a>,
    pub(super) output: Option<&'a [f64]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourcePolicy {
    pub(super) supported: bool,
    pub(super) explicit: bool,
}

/// Logical product dimensions over dense row-major storage. A transpose changes
/// the view for the provider, without packing or copying an operand in Alea.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatrixCall {
    pub(super) dimensions: (usize, usize, usize),
    pub(super) transpose: (bool, bool),
    pub(super) add: bool,
}

impl MatrixCall {
    #[inline]
    pub(super) fn run(
        self,
        kernel: ResolvedMatrix,
        left: &[f64],
        right: &[f64],
        out: &mut [f64],
    ) -> bool {
        alea_math::gemm::try_gemm(
            kernel,
            left,
            right,
            out,
            self.dimensions,
            self.transpose,
            self.add,
        )
    }
}

/// Lower a typed product and its two derivative contractions. The cotangent
/// enters on the left of the first contraction and the right of the second.
/// Slice placement and active destinations are supplied by the executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MatrixCalls {
    request: MatrixRequest,
    pub(super) kernel: ResolvedMatrix,
    pub(super) primal: MatrixCall,
    pub(super) left_adjoint: MatrixCall,
    pub(super) right_adjoint: MatrixCall,
}

impl MatrixCalls {
    fn new((m, k, n): (usize, usize, usize), kernel: ResolvedMatrix) -> Self {
        Self {
            request: kernel.requested(),
            kernel,
            primal: MatrixCall {
                dimensions: (m, k, n),
                transpose: (false, false),
                add: false,
            },
            left_adjoint: MatrixCall {
                dimensions: (m, n, k),
                transpose: (false, true),
                add: true,
            },
            right_adjoint: MatrixCall {
                dimensions: (k, m, n),
                transpose: (true, false),
                add: true,
            },
        }
    }
}

// Keep source admission and provider requirements in one routine. Child plans
// already carry checked summaries; zero-trip bodies remain unexecuted.
pub(super) fn source_policy(node: &Node) -> SourcePolicy {
    let (supported, explicit) = match &node.op {
        Op::Registered(_) => {
            let supported = node
                .native_block()
                .is_some_and(registration::Call::has_source);
            (supported, supported)
        }
        Op::Matmul { kernel, .. } => {
            #[allow(unused_mut)]
            let mut supported = true;
            #[cfg(feature = "openblas")]
            if let Op::Matmul { m, k, n, .. } = &node.op {
                supported = kernel.provider() != MatrixProvider::Blas
                    || *m == 0
                    || *k == 0
                    || *n == 0
                    || [*m, *k, *n].into_iter().all(|d| i32::try_from(d).is_ok());
            }
            (supported, kernel.is_accelerated())
        }
        Op::Cond(c) => (
            c.yes.plan.facts.missing_source.is_none() && c.no.plan.facts.missing_source.is_none(),
            c.yes.requires_explicit_source() || c.no.requires_explicit_source(),
        ),
        Op::Scan(s) if s.steps != 0 => (
            s.body.plan.facts.missing_source.is_none(),
            s.body.requires_explicit_source(),
        ),
        _ => (true, false),
    };
    SourcePolicy {
        supported,
        explicit,
    }
}

#[derive(Clone)]
pub(super) struct Call {
    origin: Value,
    kind: Kind,
    reads: Box<[Read]>,
    destination: Region,
    write: Write,
    matrix: Option<Box<MatrixCalls>>,
    source: SourcePolicy,
    residuals: Residuals,
    native_residuals: NativeResiduals,
}
impl Call {
    pub(super) fn origin(&self) -> Value {
        self.origin
    }

    pub(super) fn elements(&self) -> usize {
        self.destination.ty.shape.elements
    }

    pub(super) fn alias(&self) -> Option<Value> {
        match self.write {
            Write::Alias(value) => Some(value),
            _ => None,
        }
    }

    pub(super) fn operation<'a>(&'a self, nodes: &'a [Node]) -> Operation<'a> {
        match &nodes[self.origin.node].op {
            Op::Input(i) => Operation::Input(*i),
            Op::Data(i) => Operation::Data(*i),
            Op::Constant(data) => Operation::Constant(data),
            Op::Registered(_) => match nodes[self.origin.node].native_block() {
                Some(call) if call.has_source() => Operation::Registered(call, &self.reads),
                _ => Operation::NativeOnly,
            },
            Op::Alias(_) => Operation::Alias(self.alias().expect("checked alias")),
            Op::Unary(op, _) => Operation::Unary(*op, self.reads[0].value),
            Op::Binary(b) => {
                let [left, right] = &*self.reads else {
                    unreachable!("checked binary arity")
                };
                Operation::Binary(
                    b.kind,
                    (left.value, left.broadcast()),
                    (right.value, right.broadcast()),
                )
            }
            Op::Sum(_) => Operation::Sum(&self.reads[0]),
            Op::Gather(..) => {
                let read = &self.reads[0];
                let Access::Gather(indices) = &read.access else {
                    unreachable!("checked gather")
                };
                Operation::Gather(read.value, indices)
            }
            Op::Matmul { .. } => Operation::Matrix(
                self.reads[0].value,
                self.reads[1].value,
                self.matrix().expect("checked matrix calls"),
            ),
            Op::Cond(cond) => Operation::Cond(
                cond,
                [self.reads[0].value, self.reads[1].value],
                &self.reads[2..],
            ),
            Op::Scan(scan) => Operation::Scan(
                scan,
                [self.reads[0].value, self.reads[1].value],
                &self.reads[2..],
            ),
        }
    }

    pub(super) fn matrix(&self) -> Option<&MatrixCalls> {
        self.matrix.as_deref()
    }
    fn retained_inputs(&self) -> &[Read] {
        match self.native_residuals {
            NativeResiduals::None => &[],
            NativeResiduals::ScanReplay => &self.reads[1..],
            _ => &self.reads,
        }
    }

    fn retains_output(&self) -> bool {
        matches!(
            self.native_residuals,
            NativeResiduals::Unary | NativeResiduals::Callback
        )
    }

    pub(super) fn retained_primals(&self) -> impl Iterator<Item = Value> {
        self.retained_inputs()
            .iter()
            .map(|read| read.value)
            .chain(self.retains_output().then_some(self.origin))
    }

    #[inline]
    pub(super) fn native_primals<'a>(&'a self, values: &'a [f64]) -> NativePrimals<'a> {
        let output = self
            .retains_output()
            .then(|| &values[self.destination.range.clone()]);
        NativePrimals {
            inputs: PrimalInputs {
                reads: self.retained_inputs(),
                values: PrimalView::whole(values),
            },
            output,
        }
    }

    #[inline]
    pub(super) fn scan_inputs<'a>(&'a self, values: PrimalView<'a>) -> PrimalInputs<'a> {
        debug_assert_eq!(self.kind, Kind::Control);
        PrimalInputs {
            reads: &self.reads[1..],
            values,
        }
    }

    pub(super) fn source_storage(&self) -> Option<usize> {
        // Source-ineligible typed literals keep their conservative extent in
        // the budget. An empty materialized result still has a placement.
        (self.write == Write::Overwrite || self.destination.ty.dtype != ElementType::Real)
            .then_some(self.destination.ty.shape.elements)
    }
    // Only deterministic builtins with the existing ordered/reference policy.
    // Provider calls and opaque/control operations retain their original scope.
    pub(super) fn binding_builtin(&self) -> bool {
        matches!(
            self.kind,
            Kind::Unary | Kind::Binary | Kind::Sum | Kind::Gather
        ) || (self.kind == Kind::Matrix && !self.source.explicit)
    }
    // Native directional eligibility is independent of source/AD qualification.
    pub(super) fn linear_builtin(&self) -> bool {
        match self.kind {
            Kind::Leaf => self.destination.ty.dtype == ElementType::Real,
            Kind::Unary | Kind::Alias | Kind::Binary | Kind::Gather | Kind::Sum | Kind::Matrix => {
                true
            }
            Kind::Native | Kind::Control => false,
        }
    }
    pub(super) fn operands(&self) -> impl Iterator<Item = usize> + '_ {
        self.reads.iter().map(|read| read.value.node)
    }

    // None denotes an incompatible map. Some(true) identifies an actual
    // producer edge into the candidate group.
    pub(super) fn pointwise(
        &self,
        elements: usize,
        internal: impl Fn(usize) -> bool,
    ) -> Option<bool> {
        if self.destination.ty.shape.elements != elements
            || !matches!(self.kind, Kind::Unary | Kind::Binary)
        {
            return None;
        }
        let mut connected = false;
        for read in &self.reads {
            if internal(read.value.node) {
                if !matches!(
                    read.access,
                    Access::Whole | Access::Broadcast(Map::Identity)
                ) {
                    return None;
                }
                connected = true;
            }
        }
        Some(connected)
    }

    pub(super) fn reduction_of(&self, accepts: impl FnOnce(usize) -> bool) -> bool {
        self.kind == Kind::Sum && accepts(self.reads[0].value.node)
    }

    pub(super) fn independent_adjoint_lanes(&self, active: &[bool]) -> bool {
        self.reads.iter().all(|read| {
            !active[read.value.node]
                || matches!(
                    read.access,
                    Access::Whole | Access::Broadcast(Map::Identity)
                )
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdjointWrite {
    Add,
    Replace,
}

pub(super) struct AdjointWrites {
    // Writer and destination indices refer to original nodes. A destination
    // is always the owner of the complete adjoint alias class.
    first_writer: Box<[(usize, usize)]>,
    replacement: Box<[Option<usize>]>,
}

impl AdjointWrites {
    pub(super) fn compile(p: &Plan) -> Option<Box<Self>> {
        let owners = super::storage::alias_owners(p);
        let mut seen = vec![false; p.nodes.len()];
        seen[owners[p.output.node]] = true; // Seed publication is the first write.
        let mut first_writer = Vec::new();
        let mut replacement = vec![None; p.nodes.len()];
        for (i, call) in p.facts.lowered.calls.iter().enumerate().rev() {
            let Some(call) = call else { continue };
            for read in &call.reads {
                if read.contribution != Contribution::OrderedAdd {
                    continue;
                }
                let target = owners[read.value.node];
                if !std::mem::replace(&mut seen[target], true)
                    && call.reads.len() == 1
                    && matches!(call.kind, Kind::Unary | Kind::Sum)
                    && read.access == Access::Whole
                    && target != owners[i]
                    && !p.nodes[target].range().is_empty()
                    && !matches!(p.nodes[target].op, Op::Data(_) | Op::Constant(_))
                {
                    first_writer.push((target, i));
                    replacement[i] = Some(target);
                }
            }
        }
        first_writer.sort_unstable();
        (!first_writer.is_empty()).then(|| {
            Box::new(Self {
                first_writer: first_writer.into_boxed_slice(),
                replacement: replacement.into_boxed_slice(),
            })
        })
    }

    pub(super) fn mode(&self, node: usize, active: &[bool]) -> AdjointWrite {
        if self.replacement[node].is_some_and(|target| active[target]) {
            AdjointWrite::Replace
        } else {
            AdjointWrite::Add
        }
    }

    pub(super) fn replaced<'a>(&'a self, active: &'a [bool]) -> impl Iterator<Item = usize> + 'a {
        self.first_writer
            .iter()
            .filter_map(|&(target, writer)| (active[target] && active[writer]).then_some(target))
    }
}

pub(super) struct Contract {
    calls: Box<[Option<Call>]>,
    reads: Box<[BackwardRead]>,
}
fn kind(node: &Node) -> Kind {
    match &node.op {
        Op::Input(_) | Op::Data(_) | Op::Constant(_) => Kind::Leaf,
        Op::Registered(op) => match &**op {
            registration::Operation::Data(_) => Kind::Leaf,
            _ => Kind::Native,
        },
        Op::Unary(..) => Kind::Unary,
        Op::Binary(_) => Kind::Binary,
        Op::Sum(_) => Kind::Sum,
        Op::Alias(_) => Kind::Alias,
        Op::Gather(..) => Kind::Gather,
        Op::Matmul { .. } => Kind::Matrix,
        Op::Cond(_) | Op::Scan(_) => Kind::Control,
    }
}
fn region(node: &Node, slots: usize, compact: bool) -> Result<Region, E> {
    if Shape::new(node.shape.axes.to_vec())? != node.shape {
        return Err(E::Shape);
    }
    let start = if compact {
        node.primal_start
    } else {
        node.start
    };
    let end = start
        .checked_add(if node.discrete() {
            0
        } else {
            node.shape.elements
        })
        .ok_or(E::Shape)?;
    if end > slots {
        return Err(E::Shape);
    }
    Ok(Region {
        ty: BlockType::of(node),
        range: start..end,
    })
}
fn contribution(node: &Node, index: usize, input: &Node) -> Contribution {
    if input.discrete()
        || matches!(node.op, Op::Alias(_))
        || (matches!(node.op, Op::Cond(_)) && index < 2)
        || (matches!(&node.op, Op::Scan(s) if s.steps == 0) && index != 0)
    {
        Contribution::None
    } else {
        Contribution::OrderedAdd
    }
}
fn access(node: &Node, index: usize) -> Access {
    match &node.op {
        Op::Binary(b) => Access::Broadcast(if index == 0 {
            b.left.clone()
        } else {
            b.right.clone()
        }),
        Op::Gather(_, indices) => Access::Gather(Arc::clone(indices)),
        _ => Access::Whole,
    }
}
fn write(node: &Node) -> Write {
    match node.op {
        Op::Alias(v) => Write::Alias(v),
        _ if kind(node) == Kind::Leaf => Write::Immutable,
        _ => Write::Overwrite,
    }
}
fn arguments(binding: &BoundProgram, shapes: &[&Shape]) -> bool {
    let inputs = binding.input_shapes();
    shapes.len() == inputs.len() && shapes.iter().zip(inputs).all(|(a, b)| *a == b)
}
impl Contract {
    // Called after the physical overlap sweep has checked every alias class.
    pub(super) fn place_primals(&mut self, nodes: &[Node]) {
        for call in self.calls.iter_mut().flatten() {
            call.destination.range = nodes[call.origin.node].primal_range();
            for read in &mut call.reads {
                read.region.range = nodes[read.value.node].primal_range();
            }
        }
    }

    pub(super) fn compile(
        nodes: &[Node],
        active: &[bool],
        output: Value,
        slots: usize,
        reads: Vec<BackwardRead>,
    ) -> Result<Self, E> {
        let mut cursor = 0;
        let mut calls = Vec::with_capacity(nodes.len());
        for (i, node) in nodes.iter().enumerate() {
            if !active[i] {
                calls.push(None);
                continue;
            }
            let origin = Value {
                owner: output.owner,
                node: i,
            };
            let start = cursor;
            while cursor < reads.len() && reads[cursor].reader == origin {
                cursor += 1;
            }
            let opaque = node.native_block().is_some_and(|call| !call.has_source());
            let mut inputs = Vec::new();
            for (index, value) in node.inputs().into_iter().enumerate() {
                let input = nodes.get(value.node).ok_or(E::Shape)?;
                inputs.push(Read {
                    value,
                    region: region(input, slots, false)?,
                    access: access(node, index),
                    contribution: contribution(node, index, input),
                });
            }
            calls.push(Some(Call {
                origin,
                kind: kind(node),
                reads: inputs.into_boxed_slice(),
                destination: region(node, slots, false)?,
                write: write(node),
                matrix: match node.op {
                    Op::Matmul {
                        m, k, n, kernel, ..
                    } => Some(Box::new(MatrixCalls::new((m, k, n), kernel))),
                    _ => None,
                },
                source: source_policy(node),
                native_residuals: native_residuals(node),
                residuals: if opaque {
                    Residuals::Opaque
                } else {
                    Residuals::Explicit(start..cursor)
                },
            }));
        }
        let contract = Self {
            calls: calls.into_boxed_slice(),
            reads: reads.into_boxed_slice(),
        };
        contract.verify(nodes, active, output, slots, false)?;
        Ok(contract)
    }
    pub(super) fn call(&self, node: usize) -> Option<&Call> {
        self.calls.get(node).and_then(Option::as_ref)
    }
    pub(super) fn calls(&self) -> impl DoubleEndedIterator<Item = &Call> {
        self.calls.iter().flatten()
    }
    pub(super) fn matrix(&self, node: usize) -> &MatrixCalls {
        self.call(node)
            .and_then(Call::matrix)
            .expect("checked matrix calls")
    }
    pub(super) fn backward_reads(&self) -> &[BackwardRead] {
        &self.reads
    }
    pub(super) fn missing_source(&self) -> Option<usize> {
        self.calls
            .iter()
            .flatten()
            .find(|call| !call.source.supported)
            .map(|call| call.origin.node)
    }
    pub(super) fn requires_explicit_source(&self) -> bool {
        self.calls.iter().flatten().any(|call| call.source.explicit)
    }
    fn verify(
        &self,
        nodes: &[Node],
        active: &[bool],
        output: Value,
        slots: usize,
        compact: bool,
    ) -> Result<(), E> {
        if self.calls.len() != nodes.len()
            || active.len() != nodes.len()
            || !active.get(output.node).copied().unwrap_or(false)
        {
            return Err(E::Shape);
        }
        let mut allocations = Vec::new();
        let mut residual_cursor = 0;
        for (i, node) in nodes.iter().enumerate() {
            let Some(call) = &self.calls[i] else {
                if active[i] {
                    return Err(E::Shape);
                }
                continue;
            };
            if !active[i]
                || call.origin
                    != (Value {
                        owner: output.owner,
                        node: i,
                    })
                || call.kind != kind(node)
                || call.destination != region(node, slots, compact)?
                || call.write != write(node)
                || call.source != source_policy(node)
                || call.native_residuals != native_residuals(node)
            {
                return Err(E::Shape);
            }
            let edges = node.inputs();
            if edges.len() != call.reads.len() {
                return Err(E::Shape);
            }
            for (index, (read, &value)) in call.reads.iter().zip(&edges).enumerate() {
                if value.owner != output.owner
                    || value.node >= i
                    || !active[value.node]
                    || read.value != value
                    || read.region != region(&nodes[value.node], slots, compact)?
                    || read.access != access(node, index)
                    || read.contribution != contribution(node, index, &nodes[value.node])
                {
                    return Err(E::Shape);
                }
                if call.write == Write::Overwrite
                    && !read.region.range.is_empty()
                    && !call.destination.range.is_empty()
                    && read.region.range.start < call.destination.range.end
                    && call.destination.range.start < read.region.range.end
                {
                    return Err(E::Shape);
                }
            }
            if call.kind != Kind::Native
                && call.kind != Kind::Leaf
                && (call.destination.ty.dtype != ElementType::Real
                    || call
                        .reads
                        .iter()
                        .any(|r| r.region.ty.dtype != ElementType::Real))
            {
                return Err(E::Shape);
            }
            match &node.op {
                Op::Alias(v)
                    if call.destination.range != region(&nodes[v.node], slots, compact)?.range
                        || node.shape.elements != nodes[v.node].shape.elements =>
                {
                    return Err(E::Shape);
                }
                Op::Unary(_, v) if node.shape != nodes[v.node].shape => {
                    return Err(E::Shape);
                }
                Op::Binary(b)
                    if broadcast(&nodes[b.a.node].shape, &nodes[b.b.node].shape)? != node.shape
                        || b.left != map(&nodes[b.a.node].shape, &node.shape)?
                        || b.right != map(&nodes[b.b.node].shape, &node.shape)? =>
                {
                    return Err(E::Shape);
                }
                Op::Cond(c) => {
                    let shapes: Vec<_> = c.args.iter().map(|v| &nodes[v.node].shape).collect();
                    if nodes[c.lhs.node].shape != Shape::scalar()
                        || nodes[c.rhs.node].shape != Shape::scalar()
                        || c.yes.output_shape() != &node.shape
                        || c.no.output_shape() != &node.shape
                        || !arguments(&c.yes, &shapes)
                        || !arguments(&c.no, &shapes)
                    {
                        return Err(E::Shape);
                    }
                }
                Op::Scan(s) => {
                    let xs = &nodes[s.xs.node].shape;
                    if xs.axes.first() != Some(&s.steps) {
                        return Err(E::Shape);
                    }
                    let row = Shape::new(xs.axes[1..].to_vec())?;
                    let mut shapes = vec![&nodes[s.initial.node].shape, &row];
                    shapes.extend(s.captures.iter().map(|v| &nodes[v.node].shape));
                    if row.elements != s.row
                        || nodes[s.initial.node].shape != node.shape
                        || s.body.output_shape() != &node.shape
                        || !arguments(&s.body, &shapes)
                    {
                        return Err(E::Shape);
                    }
                }
                Op::Sum(_) if node.shape != Shape::scalar() => {
                    return Err(E::Shape);
                }
                Op::Gather(v, indices)
                    if indices.len() != node.shape.elements
                        || indices.iter().any(|&j| j >= nodes[v.node].shape.elements) =>
                {
                    return Err(E::Shape);
                }
                _ => {}
            }
            match (&node.op, call.matrix()) {
                (
                    Op::Matmul {
                        a,
                        b,
                        m,
                        k,
                        n,
                        kernel,
                    },
                    Some(matrix),
                ) => {
                    let resolved = matrix.request.resolve()?;
                    if resolved != *kernel
                        || *matrix != MatrixCalls::new((*m, *k, *n), resolved)
                        || nodes[a.node].shape.axes.as_ref() != [*m, *k]
                        || nodes[b.node].shape.axes.as_ref() != [*k, *n]
                        || node.shape.axes.as_ref() != [*m, *n]
                    {
                        return Err(E::Shape);
                    }
                }
                (Op::Matmul { .. }, None) | (_, Some(_)) => return Err(E::Shape),
                _ => {}
            }
            let mut expected = Vec::new();
            super::analysis::backward_reads(node, call.origin, &mut expected);
            match &call.residuals {
                Residuals::Opaque if call.kind == Kind::Native && expected.is_empty() => {}
                Residuals::Explicit(range)
                    if call.kind != Kind::Native || call.source.supported =>
                {
                    if range.start != residual_cursor
                        || self.reads.get(range.clone()) != Some(expected.as_slice())
                    {
                        return Err(E::Shape);
                    }
                    residual_cursor = range.end;
                }
                _ => return Err(E::Shape),
            }
            // Compact placement has a separate inclusive overlap sweep in storage.
            if !compact
                && !matches!(call.write, Write::Alias(_))
                && !call.destination.range.is_empty()
            {
                allocations.push(call.destination.range.clone());
            }
        }
        if residual_cursor != self.reads.len() {
            return Err(E::Shape);
        }
        allocations.sort_unstable_by_key(|r| r.start);
        if allocations.windows(2).any(|r| r[0].end > r[1].start) {
            return Err(E::Shape);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> BoundProgram {
        let mut b = RegionBuilder::new();
        let shape = Shape::new(vec![2, 2]).unwrap();
        let x = b.input("x", shape.clone()).unwrap();
        let squared = b.binary(BinaryOp::Multiply, x, x).unwrap();
        let alias = b.reshape(squared, shape.clone()).unwrap();
        let gathered = b.gather(alias, shape, vec![3, 1, 1, 0]).unwrap();
        let matrix = b.matmul(gathered, x).unwrap();
        let out = b.sum(matrix).unwrap();
        b.compile(out).unwrap().bind(&[]).unwrap()
    }
    fn draft(p: &Plan) -> Contract {
        Contract {
            calls: p.facts.lowered.calls.clone(),
            reads: p.facts.lowered.reads.clone(),
        }
    }
    fn check(c: &Contract, p: &Plan) -> Result<(), E> {
        c.verify(
            &p.nodes,
            &p.active,
            p.output,
            p.primal_slots,
            p.primal_slots != p.slots,
        )
    }
    #[test]
    fn checked_calls_preserve_shared_alias_gather_and_matrix_answers() {
        let p = fixture();
        let mut out = [0.];
        let mut gradient = [0.; 4];
        p.value_and_vjp(
            &[1., 2., 3., 4.],
            &[1.],
            &mut out,
            &mut gradient,
            &mut p.workspace(),
        )
        .unwrap();
        assert_eq!(out, [95.]);
        assert_eq!(gradient, [34., 60., 5., 29.]);
        let contract = &p.plan.facts.lowered;
        check(contract, &p.plan).unwrap();
        let alias = contract
            .calls
            .iter()
            .flatten()
            .find(|c| c.kind == Kind::Alias)
            .unwrap();
        assert_eq!(alias.source_storage(), None);
        let binary = contract
            .calls
            .iter()
            .flatten()
            .find(|c| c.kind == Kind::Binary)
            .unwrap();
        assert_eq!(binary.reads[0].value, binary.reads[1].value);
        assert!(
            binary
                .reads
                .iter()
                .all(|r| r.contribution == Contribution::OrderedAdd)
        );
        assert!(p.rust_function().is_ok());
    }
    #[test]
    fn verifier_rejects_forged_typed_regions_writes_maps_and_accumulation() {
        let p = fixture();
        let index = p
            .plan
            .facts
            .lowered
            .calls
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| c.kind == Kind::Binary))
            .unwrap();
        let corruptions: &[fn(&mut Call)] = &[
            |c| c.origin.owner = c.origin.owner.wrapping_add(1),
            |c| c.destination.range.end = usize::MAX,
            |c| c.destination.range = c.reads[0].region.range.clone(),
            |c| c.destination.ty.dtype = ElementType::Bool,
            |c| c.destination.ty.shape = Shape::scalar(),
            |c| c.write = Write::Alias(c.reads[0].value),
            |c| c.reads[0].value.node = usize::MAX,
            |c| c.reads[0].region.range = 0..0,
            |c| c.reads[0].region.ty.dtype = ElementType::Count,
            |c| c.reads[0].access = Access::Broadcast(Map::Scalar),
            |c| c.reads[1].contribution = Contribution::None,
            |c| c.residuals = Residuals::Explicit(0..0),
            |c| c.residuals = Residuals::Opaque,
            |c| c.native_residuals = NativeResiduals::None,
            |c| c.source.supported = false,
        ];
        for corrupt in corruptions {
            let mut candidate = draft(&p.plan);
            corrupt(candidate.calls[index].as_mut().unwrap());
            assert!(matches!(check(&candidate, &p.plan), Err(E::Shape)));
        }
        let mut candidate = draft(&p.plan);
        candidate.reads[0].reader.node = usize::MAX;
        assert!(matches!(check(&candidate, &p.plan), Err(E::Shape)));
        let mut candidate = draft(&p.plan);
        candidate.calls[index] = None;
        assert!(matches!(check(&candidate, &p.plan), Err(E::Shape)));
    }

    #[test]
    fn native_reverse_borrows_aliases_and_reserves_parent_reads_only_for_scan_replay() {
        let mut body = RegionBuilder::new();
        let shape = Shape::new(vec![64]).unwrap();
        let carry = body.input("carry", shape.clone()).unwrap();
        let row = body.input("row", shape.clone()).unwrap();
        let capture = body.input("capture", shape.clone()).unwrap();
        let added = body.binary(BinaryOp::Add, carry, row).unwrap();
        let next = body.binary(BinaryOp::Add, added, capture).unwrap();
        let body = body.compile(next).unwrap().bind(&[]).unwrap();

        for (steps, requested_checkpoint) in [(0, true), (3, true), (1025, false), (1025, true)] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", shape.clone()).unwrap();
            let alias = b.reshape(x, shape.clone()).unwrap();
            let square = b.binary(BinaryOp::Multiply, x, alias).unwrap();
            let initial = b.unary(UnaryOp::Tanh, square).unwrap();
            let rows = b
                .input("rows", Shape::new(vec![steps, 64]).unwrap())
                .unwrap();
            let captures = [x];
            let scan = if requested_checkpoint {
                b.checkpointed_scan(initial, rows, &captures, body.clone())
                    .unwrap()
            } else {
                b.scan(initial, rows, &captures, body.clone()).unwrap()
            };
            let reduced = b.sum(scan).unwrap();
            let p = b.compile(reduced).unwrap();
            let plan = &p.plan;
            let values: Vec<_> = (0..plan.slots).map(|i| i as f64).collect();
            let primals = |v: Value| {
                plan.facts
                    .lowered
                    .call(v.node)
                    .unwrap()
                    .native_primals(&values)
            };
            let pair = primals(square);
            let left = pair.inputs.get(0).unwrap();
            let right = pair.inputs.get(1).unwrap();
            assert!(std::ptr::eq(left, right));
            assert!(pair.output.is_none());
            let unary = primals(initial);
            assert_eq!(
                unary.inputs.get(0),
                Some(&values[plan.nodes[square.node].range()])
            );
            assert_eq!(
                unary.output,
                Some(&values[plan.nodes[initial.node].range()])
            );
            assert!(primals(reduced).inputs.get(0).is_none());

            let scan = primals(scan);
            assert!(scan.output.is_none());
            if requested_checkpoint && steps == 1025 {
                assert_eq!(
                    scan.inputs.get(0),
                    Some(&values[plan.nodes[rows.node].range()])
                );
                assert!(std::ptr::eq(scan.inputs.get(1).unwrap(), left));
                assert!(scan.inputs.get(2).is_none());
            } else {
                assert!(scan.inputs.get(0).is_none());
            }
        }
    }
    #[test]
    fn verifier_rejects_invalid_physical_aliases_indices_and_canonical_maps() {
        for corruption in 0..5 {
            let mut p = fixture();
            let plan = Arc::get_mut(&mut p.plan).unwrap();
            match corruption {
                0 => {
                    let i = plan
                        .nodes
                        .iter()
                        .position(|n| matches!(n.op, Op::Binary(_)))
                        .unwrap();
                    plan.nodes[i].start = 0;
                }
                1 => {
                    let i = plan
                        .nodes
                        .iter()
                        .position(|n| matches!(n.op, Op::Alias(_)))
                        .unwrap();
                    plan.nodes[i].start += 1;
                }
                2 => {
                    let node = plan
                        .nodes
                        .iter_mut()
                        .find(|n| matches!(n.op, Op::Gather(..)))
                        .unwrap();
                    let Op::Gather(_, indices) = &mut node.op else {
                        unreachable!()
                    };
                    *indices = Arc::from([4, 1, 1, 0]);
                }
                3 => {
                    let node = plan
                        .nodes
                        .iter_mut()
                        .find(|n| matches!(n.op, Op::Binary(_)))
                        .unwrap();
                    let Op::Binary(b) = &mut node.op else {
                        unreachable!()
                    };
                    b.left = Map::Scalar;
                }
                _ => {
                    let node = plan
                        .nodes
                        .iter_mut()
                        .find(|n| matches!(n.op, Op::Matmul { .. }))
                        .unwrap();
                    let Op::Matmul { k, .. } = &mut node.op else {
                        unreachable!()
                    };
                    *k += 1;
                }
            }
            // A self-consistent candidate built from the malformed regions must
            // fail too, not merely a candidate compared with an older descriptor.
            assert!(matches!(
                Contract::compile(
                    &plan.nodes,
                    &plan.active,
                    plan.output,
                    plan.slots,
                    plan.facts.lowered.reads.to_vec()
                ),
                Err(E::Shape)
            ));
        }
    }
    #[test]
    fn matrix_policy_must_resolve_and_match_the_original_numerics() {
        let p = fixture();
        let index = p
            .plan
            .facts
            .lowered
            .calls
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| c.kind == Kind::Matrix))
            .unwrap();
        let mut candidate = draft(&p.plan);
        candidate.calls[index]
            .as_mut()
            .unwrap()
            .matrix
            .as_mut()
            .unwrap()
            .request = MatrixRequest::standard(MatrixProvider::Reference);
        assert!(matches!(
            check(&candidate, &p.plan),
            Err(E::MatrixPolicy(_))
        ));
        let mut candidate = draft(&p.plan);
        candidate.calls[index]
            .as_mut()
            .unwrap()
            .matrix
            .as_mut()
            .unwrap()
            .right_adjoint
            .add = false;
        assert!(matches!(check(&candidate, &p.plan), Err(E::Shape)));
        let mut candidate = draft(&p.plan);
        candidate.calls[index].as_mut().unwrap().matrix = None;
        assert!(matches!(check(&candidate, &p.plan), Err(E::Shape)));
        #[cfg(not(feature = "faer"))]
        {
            let mut candidate = draft(&p.plan);
            candidate.calls[index]
                .as_mut()
                .unwrap()
                .matrix
                .as_mut()
                .unwrap()
                .request = MatrixRequest::standard(MatrixProvider::Faer);
            assert!(matches!(
                check(&candidate, &p.plan),
                Err(E::MatrixPolicy(_))
            ));
        }
    }
    #[test]
    fn verifier_rejects_child_region_signature_mismatch_before_planning() {
        let mut child = RegionBuilder::new();
        let x = child.input("x", Shape::scalar()).unwrap();
        let child = child.compile(x).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let out = b
            .cond(x, x, Comparison::Less, &[x], child.clone(), child.clone())
            .unwrap();
        let mut p = b.compile(out).unwrap().bind(&[]).unwrap();
        let plan = Arc::get_mut(&mut p.plan).unwrap();
        let Op::Cond(c) = &mut plan.nodes[out.node].op else {
            unreachable!()
        };
        c.args = Box::new([]);
        // Control outputs remain guarded; a signature mismatch cannot create
        // an executable even when all the candidate's own read regions match.
        let mut reads = Vec::new();
        for (i, node) in plan.nodes.iter().enumerate() {
            super::super::analysis::backward_reads(
                node,
                Value {
                    owner: plan.owner,
                    node: i,
                },
                &mut reads,
            );
        }
        assert!(matches!(
            Contract::compile(&plan.nodes, &plan.active, plan.output, plan.slots, reads),
            Err(E::Shape)
        ));
    }
}
