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
struct Read {
    value: Value,
    region: Region,
    access: Access,
    contribution: Contribution,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Residuals {
    Explicit(Range<usize>),
    Opaque,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourcePolicy {
    pub(super) supported: bool,
    pub(super) explicit: bool,
}

// Keep source admission and provider requirements in one routine. Child plans
// already carry checked summaries; zero-trip bodies remain unexecuted.
pub(super) fn source_policy(node: &Node) -> SourcePolicy {
    let (supported, explicit) = match &node.op {
        Op::Registered(_) => (false, false),
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
    // None retains existing builtin arithmetic, including compensated sums.
    // A matrix request
    // retains arithmetic/CPU/thread preferences as well as the resolved provider.
    matrix: Option<MatrixRequest>,
    source: SourcePolicy,
    residuals: Residuals,
}
impl Call {
    pub(super) fn source_storage(&self) -> usize {
        // Retain the old conservative source extent for source-ineligible typed
        // literals too; this never supplies a lowering for their storage type.
        if self.write == Write::Overwrite || self.destination.ty.dtype != ElementType::Real {
            self.destination.ty.shape.elements
        } else {
            0
        }
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
    pub(super) fn pointwise(&self, first: usize, elements: usize) -> bool {
        if self.destination.ty.shape.elements != elements {
            return false;
        }
        match self.kind {
            Kind::Unary => true,
            Kind::Binary => self.reads.iter().all(|r| {
                r.value.node < first || matches!(r.access, Access::Broadcast(Map::Identity))
            }),
            _ => false,
        }
    }
    pub(super) fn reduction_of(&self, maps: Range<usize>) -> bool {
        self.kind == Kind::Sum && maps.contains(&self.reads[0].value.node)
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
fn region(node: &Node, slots: usize) -> Result<Region, E> {
    if Shape::new(node.shape.axes.to_vec())? != node.shape {
        return Err(E::Shape);
    }
    let end = node
        .start
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
        range: node.start..end,
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
            let opaque = matches!(&node.op, Op::Registered(op) if matches!(&**op, registration::Operation::Native(_)));
            let mut inputs = Vec::new();
            for (index, value) in node.inputs().into_iter().enumerate() {
                let input = nodes.get(value.node).ok_or(E::Shape)?;
                inputs.push(Read {
                    value,
                    region: region(input, slots)?,
                    access: access(node, index),
                    contribution: contribution(node, index, input),
                });
            }
            calls.push(Some(Call {
                origin,
                kind: kind(node),
                reads: inputs.into_boxed_slice(),
                destination: region(node, slots)?,
                write: write(node),
                matrix: match node.op {
                    Op::Matmul { kernel, .. } => Some(kernel.requested()),
                    _ => None,
                },
                source: source_policy(node),
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
        contract.verify(nodes, active, output, slots)?;
        Ok(contract)
    }
    pub(super) fn call(&self, node: usize) -> Option<&Call> {
        self.calls.get(node).and_then(Option::as_ref)
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
                || call.destination != region(node, slots)?
                || call.write != write(node)
                || call.source != source_policy(node)
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
                    || read.region != region(&nodes[value.node], slots)?
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
                    if call.destination.range != region(&nodes[v.node], slots)?.range
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
            match (&node.op, call.matrix) {
                (
                    Op::Matmul {
                        a,
                        b,
                        m,
                        k,
                        n,
                        kernel,
                    },
                    Some(request),
                ) => {
                    let resolved = request.resolve()?;
                    if resolved != *kernel
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
                Residuals::Explicit(range) if call.kind != Kind::Native => {
                    if range.start != residual_cursor
                        || self.reads.get(range.clone()) != Some(expected.as_slice())
                    {
                        return Err(E::Shape);
                    }
                    residual_cursor = range.end;
                }
                _ => return Err(E::Shape),
            }
            if !matches!(call.write, Write::Alias(_)) && !call.destination.range.is_empty() {
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
        c.verify(&p.nodes, &p.active, p.output, p.slots)
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
        assert_eq!(alias.source_storage(), 0);
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
        candidate.calls[index].as_mut().unwrap().matrix =
            Some(MatrixRequest::standard(MatrixProvider::Reference));
        assert!(matches!(
            check(&candidate, &p.plan),
            Err(E::MatrixPolicy(_))
        ));
        let mut candidate = draft(&p.plan);
        candidate.calls[index].as_mut().unwrap().matrix = None;
        assert!(matches!(check(&candidate, &p.plan), Err(E::Shape)));
        #[cfg(not(feature = "faer"))]
        {
            let mut candidate = draft(&p.plan);
            candidate.calls[index].as_mut().unwrap().matrix =
                Some(MatrixRequest::standard(MatrixProvider::Faer));
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
