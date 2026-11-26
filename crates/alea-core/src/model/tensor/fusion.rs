//! Dependency-safe pointwise DAG/map-reduction scheduling. No reassociation.
use super::*;
use alea_math::numeric::{CompensatedSum, first_nonfinite};
use std::num::NonZeroUsize;

pub(super) enum Step {
    Nodes(Range<usize>),
    Fused(Group),
    #[cfg(any(alea_l42_joint, test))]
    Joint(Box<Joint>),
}
// Most plans (especially scalar scan bodies) have no profitable native group.
// Keep optional scheduler metadata out of their hot immutable plan layout.
pub(super) struct Schedule {
    pub(super) materialized: Vec<Step>,
    #[cfg(any(alea_l42_joint, test))]
    joint: Option<Box<[Option<Step>; JointTile::ALL.len()]>>,
}

impl Schedule {
    pub(super) fn new(_p: &Plan, materialized: Vec<Step>) -> Self {
        Self {
            #[cfg(any(alea_l42_joint, test))]
            joint: JointGroup::check(_p, &materialized).map(|group| {
                Box::new(JointTile::ALL.map(|tile| {
                    group
                        .place(_p, &materialized, tile)
                        .map(|step| Step::Joint(Box::new(step)))
                }))
            }),
            materialized,
        }
    }

    #[cfg(any(alea_l42_joint, test))]
    pub(super) fn joint_step(&self, tile: JointTile) -> Option<&Step> {
        self.joint.as_ref()?[tile as usize].as_ref()
    }

    #[cfg(any(alea_l42_joint, test))]
    pub(super) fn joint(&self, tile: JointTile) -> Option<&Joint> {
        match self.joint_step(tile)? {
            Step::Joint(step) => Some(step),
            _ => unreachable!("joint candidate contains one joint step"),
        }
    }
}

const REFERENCE_TILE_WIDTH: NonZeroUsize = NonZeroUsize::new(256).unwrap();

/// The finite set of widths used to qualify a terminal joint group.
#[cfg(any(alea_l42_joint, test))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum JointTile {
    Reference,
    Width128,
    Width512,
    Width1024,
}

#[cfg(any(alea_l42_joint, test))]
impl JointTile {
    pub(super) const ALL: [Self; 4] = [
        Self::Reference,
        Self::Width128,
        Self::Width512,
        Self::Width1024,
    ];

    fn width(self) -> NonZeroUsize {
        match self {
            Self::Reference => REFERENCE_TILE_WIDTH,
            Self::Width128 => NonZeroUsize::new(128).unwrap(),
            Self::Width512 => NonZeroUsize::new(512).unwrap(),
            Self::Width1024 => NonZeroUsize::new(1024).unwrap(),
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Reference => "joint",
            Self::Width128 => "joint_128",
            Self::Width512 => "joint_512",
            Self::Width1024 => "joint_1024",
        }
    }
}

pub(super) struct Group {
    maps: Box<[usize]>,
    pub(super) loads: Box<[usize]>,
    pub(super) domain: LoopDomain,
    pub(super) reduction: Option<usize>,
}
// A split of one ordered, half-open element domain. Tiling changes execution
// granularity without permuting lanes or the order of an attached reduction.
pub(super) struct LoopDomain {
    pub(super) end: usize,
    tile: NonZeroUsize,
}
impl LoopDomain {
    #[cfg(alea_l42_joint)]
    pub(super) fn tile_width(&self) -> usize {
        self.tile.get()
    }

    fn tiles(&self) -> impl Iterator<Item = Range<usize>> {
        (0..self.end)
            .step_by(self.tile.get())
            .map(|start| start..start.saturating_add(self.tile.get()).min(self.end))
    }
}

impl Group {
    pub(super) fn start(&self) -> usize {
        self.maps[0]
    }

    pub(super) fn end(&self) -> usize {
        self.reduction.unwrap_or(self.maps[self.maps.len() - 1]) + 1
    }

    pub(super) fn maps(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
        self.maps.iter().copied()
    }
}

struct Uses {
    owners: Vec<usize>,
    last: Vec<usize>,
}

impl Uses {
    fn new(p: &Plan) -> Self {
        let owners = storage::alias_owners(p);
        let mut last = vec![0; p.nodes.len()];
        for (i, node) in p.nodes.iter().enumerate() {
            // A reshape only names its owner's storage. Charge actual readers
            // through the view, plus publication, to that same owner.
            if !matches!(node.op, Op::Alias(_))
                && let Some(call) = p.facts.lowered.call(i)
            {
                for input in call.operands() {
                    last[owners[input]] = i;
                }
            }
        }
        last[owners[p.output.node]] = p.nodes.len();
        Self { owners, last }
    }
}

fn pointwise(
    p: &Plan,
    i: usize,
    elements: usize,
    internal: impl Fn(usize) -> bool,
) -> Option<bool> {
    if p.facts.binding_fixed(i) {
        return None;
    }
    p.facts.lowered.call(i)?.pointwise(elements, internal)
}

pub(super) fn schedule(p: &Plan, native: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut uses = None;
    let (mut i, mut plain) = (0, 0);
    while i < p.nodes.len() {
        let first = i;
        let elements = p.nodes[i].shape.elements;
        // Preserve the measured native size gate and the existing source policy.
        if elements < if native { 4096 } else { 2 }
            || pointwise(p, i, elements, |node| node >= first).is_none()
        {
            i += 1;
            continue;
        }
        let facts = native.then(|| &*uses.get_or_insert_with(|| Uses::new(p)));
        let mut maps = vec![i];
        let mut loads = Vec::new();
        i += 1;
        while i < p.nodes.len() {
            if let Some(facts) = facts {
                if !p.active[i] {
                    i += 1;
                    continue;
                }
                match p.nodes[i].op {
                    Op::Alias(_) | Op::Constant(_) => {
                        i += 1;
                        continue;
                    }
                    Op::Input(_) | Op::Data(_) => {
                        loads.push(i);
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
                if pointwise(p, i, elements, |node| {
                    maps.binary_search(&facts.owners[node]).is_ok()
                }) != Some(true)
                {
                    break;
                }
            } else if pointwise(p, i, elements, |node| node >= first).is_none() {
                break;
            }
            maps.push(i);
            i += 1;
        }
        let reduction = p
            .facts
            .lowered
            .call(i)
            .is_some_and(|call| {
                call.reduction_of(|node| match facts {
                    Some(facts) => {
                        let owner = facts.owners[node];
                        facts.last[owner] == i && maps.binary_search(&owner).is_ok()
                    }
                    None => (first..i).contains(&node),
                })
            })
            .then_some(i);
        if (native && maps.len() < 4) || (maps.len() < 2 && reduction.is_none()) {
            continue;
        }

        if plain != first {
            steps.push(Step::Nodes(plain..first));
        }
        let end = reduction.unwrap_or(*maps.last().expect("nonempty group")) + 1;
        loads.retain(|&load| load < end);
        let group = Group {
            maps: maps.into_boxed_slice(),
            loads: loads.into_boxed_slice(),
            domain: LoopDomain {
                end: elements,
                tile: REFERENCE_TILE_WIDTH,
            },
            reduction,
        };
        i = i.max(end);
        plain = end;
        steps.push(Step::Fused(group));
    }
    if !steps.is_empty() && plain != p.nodes.len() {
        steps.push(Step::Nodes(plain..p.nodes.len()));
    }
    steps
}

// Tile scheduling retains vector loops/dispatch per block, never an interpreted
// instruction dispatch per observation. Materialized primals remain available to
// the ORIGINAL reverse order. Shared/scalar adjoints must not be tiled/reordered.
pub(super) fn forward(p: &Plan, group: &Group, values: &mut [f64]) -> Result<(), E> {
    let mut sum = CompensatedSum::default();
    for range in group.domain.tiles() {
        let Range { start, end } = range;
        for i in group.maps() {
            let node = &p.nodes[i];
            let (before, after) = values.split_at_mut(node.start);
            let out = &mut after[start..end];
            match &node.op {
                Op::Unary(op, a) => {
                    let base = p.nodes[a.node].start;
                    for (out, &x) in out.iter_mut().zip(&before[base + start..base + end]) {
                        *out = execute::unary(*op, x);
                    }
                }
                Op::Binary(b) => binary(
                    b,
                    &before[p.nodes[b.a.node].range()],
                    &before[p.nodes[b.b.node].range()],
                    out,
                    start,
                ),
                _ => unreachable!("compiled pointwise group"),
            }
            // Check EVERY intermediate, including values whose consumer would
            // turn infinity back into a finite value. Caller buffers stay staged.
            if first_nonfinite(out).is_some() {
                return Err(E::Numerical);
            }
        }
        if let Some(i) = group.reduction {
            let Op::Sum(a) = p.nodes[i].op else {
                unreachable!("compiled reduction")
            };
            let base = p.nodes[a.node].start;
            for &value in &values[base + start..base + end] {
                sum.add(value);
            }
        }
    }
    if let Some(i) = group.reduction {
        let total = sum.total();
        if !total.is_finite() {
            return Err(E::Numerical);
        }
        values[p.nodes[i].start] = total;
    }
    Ok(())
}

fn binary(b: &Binary, a: &[f64], r: &[f64], out: &mut [f64], start: usize) {
    let end = start + out.len();
    match (&b.left, &b.right) {
        (Map::Identity, Map::Identity) => kernels::forward(b, &a[start..end], &r[start..end], out),
        (Map::Identity, Map::Scalar) => kernels::forward(b, &a[start..end], r, out),
        (Map::Scalar, Map::Identity) => kernels::forward(b, a, &r[start..end], out),
        _ => match b.kind {
            BinaryOp::Add => kernels::mapped(b, a, r, out, start, |a, b| a + b),
            BinaryOp::Subtract => kernels::mapped(b, a, r, out, start, |a, b| a - b),
            BinaryOp::Multiply => kernels::mapped(b, a, r, out, start, |a, b| a * b),
            BinaryOp::Divide => kernels::mapped(b, a, r, out, start, |a, b| a / b),
        },
    }
}

// Reverse tiling is legal only when active adjoint writes are independent
// across elements. Inactive scalar/broadcast operands may still supply primals.
// Never tile shared scalar or gather reductions: their addition order matters.
pub(super) fn reverse_compatible(p: &Plan, group: &Group, active: &[bool]) -> bool {
    group.maps().all(|i| {
        p.facts
            .lowered
            .call(i)
            .expect("checked pointwise map")
            .independent_adjoint_lanes(active)
    })
}

pub(super) fn reverse(
    p: &Plan,
    group: &Group,
    values: &[f64],
    adjoints: &mut [f64],
    active: &[bool],
    starts: &[usize],
) {
    use execute::linear::{Reverse, Rule};
    for range in group.domain.tiles() {
        for i in group.maps().chain(group.reduction).rev() {
            if active[i] {
                Rule::at(p, i).transpose::<true, true>(
                    i,
                    Reverse {
                        plan: p,
                        primals: values,
                        adjoints,
                        active,
                        starts,
                    },
                    range.clone(),
                    None,
                    lowering::AdjointWrite::Add,
                );
            }
        }
    }
}

#[cfg(any(alea_l42_joint, test))]
enum TileRead<Local> {
    Input(Range<usize>),
    Data(usize),
    Constant(usize),
    Local(Local),
}

// Checked terminal group with logical reads. Local reads still name an alias
// owner here, so a change in loop width cannot reuse a stale physical offset.
#[cfg(any(alea_l42_joint, test))]
struct JointGroup {
    group: usize,
    reads: Vec<Option<TileRead<Value>>>,
    owners: Vec<usize>,
    active: Vec<bool>,
    inputs: Vec<(usize, usize)>,
    writes: Option<Box<lowering::AdjointWrites>>,
    adjoint_writes: Vec<lowering::AdjointWrite>,
}

#[cfg(any(alea_l42_joint, test))]
impl JointGroup {
    fn check(p: &Plan, steps: &[Step]) -> Option<Self> {
        let mut groups = steps.iter().enumerate().filter_map(|(i, step)| match step {
            Step::Fused(group) => Some((i, group)),
            _ => None,
        });
        let (group_index, group) = groups.next()?;
        if groups.next().is_some() || group.reduction != Some(p.output.node) {
            return None;
        }
        let active = p.facts.activity(&vec![true; p.inputs.len()]);
        if !reverse_compatible(p, group, &active) {
            return None;
        }
        let owners = storage::alias_owners(p);
        let mut reads: Vec<_> = (0..p.nodes.len()).map(|_| None).collect();
        let mut inputs = Vec::new();
        for (i, node) in p.nodes.iter().enumerate().filter(|(i, _)| p.active[*i]) {
            reads[i] = match &node.op {
                Op::Input(index) => {
                    let start = p.inputs[*index].start;
                    if active[i] {
                        if node.shape.elements != group.domain.end {
                            return None;
                        }
                        inputs.push((start, i));
                    }
                    Some(TileRead::Input(start..start + node.shape.elements))
                }
                Op::Data(index) => Some(TileRead::Data(*index)),
                Op::Constant(_) => Some(TileRead::Constant(i)),
                Op::Alias(_) => None,
                _ if i == p.output.node => None,
                _ if group.maps.binary_search(&i).is_ok() => Some(TileRead::Local(Value {
                    owner: p.owner,
                    node: i,
                })),
                _ => return None,
            };
        }
        for i in group.maps() {
            if let Op::Binary(b) = &p.nodes[i].op
                && !matches!(
                    (&b.left, &b.right),
                    (Map::Identity, Map::Identity)
                        | (Map::Identity, Map::Scalar)
                        | (Map::Scalar, Map::Identity)
                )
            {
                return None;
            }
        }
        let writes = lowering::AdjointWrites::compile(p);
        let adjoint_writes = (0..p.nodes.len())
            .map(|node| {
                writes
                    .as_ref()
                    .map_or(lowering::AdjointWrite::Add, |writes| {
                        writes.mode(node, &active)
                    })
            })
            .collect();
        Some(Self {
            group: group_index,
            reads,
            owners,
            active,
            inputs,
            writes,
            adjoint_writes,
        })
    }

    fn place(&self, p: &Plan, steps: &[Step], tile: JointTile) -> Option<Joint> {
        let Step::Fused(group) = &steps[self.group] else {
            unreachable!("checked joint group");
        };
        let domain = LoopDomain {
            end: group.domain.end,
            tile: tile.width(),
        };
        let width = domain.tile.get();
        // Reserve a materialized scalar escape at offset zero.
        let (primals, primal_slots) =
            storage::tile_slots(p.nodes.len(), group.maps(), width, 1).ok()?;
        let adjoints = storage::tiled_adjoints(
            p,
            &self.active,
            &self.owners,
            self.inputs
                .iter()
                .map(|&(_, node)| node)
                .chain(group.maps().filter(|&i| self.active[i])),
            width,
            self.writes.as_deref(),
        )
        .ok()?;
        let reads = self
            .owners
            .iter()
            .map(|&owner| {
                Some(match self.reads[owner].as_ref()? {
                    TileRead::Input(range) => TileRead::Input(range.clone()),
                    TileRead::Data(index) => TileRead::Data(*index),
                    TileRead::Constant(node) => TileRead::Constant(*node),
                    TileRead::Local(value) => TileRead::Local(primals[value.node]),
                })
            })
            .collect();
        Some(Joint {
            group: self.group,
            domain,
            reads,
            active: self.active.clone(),
            inputs: self.inputs.clone(),
            adjoint_writes: self.adjoint_writes.clone(),
            primal_slots,
            adjoints,
        })
    }
}

/// One placed joint step. Leaves are borrowed from the admitted call/binding.
#[cfg(any(alea_l42_joint, test))]
pub(super) struct Joint {
    group: usize,
    pub(super) domain: LoopDomain,
    reads: Vec<Option<TileRead<usize>>>,
    active: Vec<bool>,
    inputs: Vec<(usize, usize)>,
    adjoint_writes: Vec<lowering::AdjointWrite>,
    pub(super) primal_slots: usize,
    pub(super) adjoints: storage::AdjointPlacement,
}

#[cfg(any(alea_l42_joint, test))]
impl Joint {
    pub(super) fn group<'a>(&self, schedule: &'a Schedule) -> &'a Group {
        let Step::Fused(group) = &schedule.materialized[self.group] else {
            unreachable!("compiled joint group")
        };
        group
    }

    // Keep the tile loops out of callers shared with the materialized path.
    #[inline(never)]
    pub(super) fn run(
        &self,
        binding: &BoundProgram,
        x: &[f64],
        seed: Option<f64>,
        primals: &mut [f64],
        adjoints: &mut [f64],
        gradient: &mut [f64],
    ) -> Result<(), E> {
        let p = &binding.plan;
        let group = self.group(p.execution.as_ref().expect("joint schedule"));
        let Op::Sum(result) = p.nodes[p.output.node].op else {
            unreachable!("compiled joint sum")
        };
        let mut sum = CompensatedSum::default();
        for range in self.domain.tiles() {
            let n = range.len();
            let tile = Tile {
                binding,
                x,
                range: &range,
            };
            for i in group.maps() {
                let Some(TileRead::Local(start)) = self.reads[i] else {
                    unreachable!("compiled tile destination")
                };
                let (before, tail) = primals.split_at_mut(start);
                let dest = &mut tail[..n];
                match &p.nodes[i].op {
                    Op::Unary(op, a) => kernels::unary_forward(
                        *op,
                        tile.read(&self.reads, a.node, &Map::Identity, before),
                        dest,
                    ),
                    Op::Binary(b) => kernels::forward(
                        b,
                        tile.read(&self.reads, b.a.node, &b.left, before),
                        tile.read(&self.reads, b.b.node, &b.right, before),
                        dest,
                    ),
                    _ => unreachable!("compiled pointwise map"),
                }
                if first_nonfinite(dest).is_some() {
                    return Err(E::Numerical);
                }
            }
            for &value in tile.read(&self.reads, result.node, &Map::Identity, primals) {
                sum.add(value);
            }
            let Some(seed) = seed.filter(|_| self.active[result.node]) else {
                continue;
            };
            let result = self.adjoints.starts[result.node];
            if n == self.domain.tile.get() {
                for clear in &self.adjoints.clear {
                    adjoints[clear.clone()].fill(0.);
                }
            } else {
                // Replacement writes cover the active lanes. Initialize the
                // padded lanes too so the whole-arena finite check is valid.
                adjoints.fill(0.);
            }
            adjoints[result..result + n].fill(0. + seed);
            for i in group.maps().rev() {
                if !self.active[i] {
                    continue;
                }
                let start = self.adjoints.starts[i];
                match &p.nodes[i].op {
                    Op::Unary(op, a) => kernels::unary_reverse_range(
                        *op,
                        tile.read(&self.reads, a.node, &Map::Identity, primals),
                        tile.read(&self.reads, i, &Map::Identity, primals),
                        adjoints,
                        (self.adjoints.starts[a.node], start),
                        self.adjoint_writes[i],
                    ),
                    Op::Binary(b) => {
                        let (before, tail) = adjoints.split_at_mut(start);
                        kernels::reverse(
                            b,
                            kernels::Pullback {
                                a: tile.read(&self.reads, b.a.node, &b.left, primals),
                                r: tile.read(&self.reads, b.b.node, &b.right, primals),
                                seeds: &tail[..n],
                                adjoints: before,
                                starts: (
                                    self.adjoints.starts[b.a.node],
                                    self.adjoints.starts[b.b.node],
                                ),
                                active: (self.active[b.a.node], self.active[b.b.node]),
                            },
                        );
                    }
                    _ => unreachable!("compiled pointwise map"),
                }
            }
            if first_nonfinite(adjoints).is_some() {
                return Err(E::Numerical);
            }
            for &(input, node) in &self.inputs {
                let score = self.adjoints.starts[node];
                gradient[input + range.start..input + range.end]
                    .copy_from_slice(&adjoints[score..score + n]);
            }
        }
        let total = sum.total();
        if !total.is_finite() {
            return Err(E::Numerical);
        }
        primals[0] = total;
        Ok(())
    }
}

#[cfg(any(alea_l42_joint, test))]
struct Tile<'a> {
    binding: &'a BoundProgram,
    x: &'a [f64],
    range: &'a Range<usize>,
}

#[cfg(any(alea_l42_joint, test))]
impl Tile<'_> {
    fn read<'a>(
        &'a self,
        reads: &[Option<TileRead<usize>>],
        node: usize,
        map: &Map,
        locals: &'a [f64],
    ) -> &'a [f64] {
        let external = match reads[node].as_ref().expect("compiled tile read") {
            TileRead::Input(input) => &self.x[input.clone()],
            TileRead::Data(index) => &self.binding.data[*index],
            TileRead::Constant(i) => match &self.binding.plan.nodes[*i].op {
                Op::Constant(values) => values,
                _ => unreachable!("compiled constant"),
            },
            TileRead::Local(start) => return &locals[*start..*start + self.range.len()],
        };
        if matches!(map, Map::Identity) {
            &external[self.range.clone()]
        } else {
            external
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_tiles_match_independent_scores_and_reuse_across_bindings() {
        for n in [4096, 4097, 8193] {
            let mut b = RegionBuilder::new();
            let shape = Shape::new(vec![n]).unwrap();
            let x = b.input("x", shape.clone()).unwrap();
            let data = b.data("d", shape.clone()).unwrap();
            let added = b.binary(BinaryOp::Add, x, data).unwrap();
            let view = b.reshape(added, Shape::new(vec![n, 1]).unwrap()).unwrap();
            let square = b.binary(BinaryOp::Multiply, view, view).unwrap();
            let half = b.constant(Shape::scalar(), vec![0.5]).unwrap();
            let energy = b.binary(BinaryOp::Multiply, half, square).unwrap();
            let negated = b.unary(UnaryOp::Neg, energy).unwrap();
            let output = b.sum(negated).unwrap();
            let bound = b
                .compile(output)
                .unwrap()
                .bind(&[("d", &vec![0.25; n])])
                .unwrap();
            let rebound = bound.rebind("d", &vec![-0.5; n]).unwrap();
            let schedule = bound.plan.execution.as_ref().unwrap();
            let q: Vec<_> = (0..n).map(|i| (i % 13) as f64 / 16. - 0.25).collect();
            for (tile, width) in JointTile::ALL.into_iter().zip([256, 128, 512, 1024]) {
                let joint = schedule.joint(tile).unwrap();
                assert_eq!(joint.domain.tile.get(), width);
                assert_eq!(
                    (joint.primal_slots, joint.adjoints.slots),
                    (1 + 4 * width, 5 * width)
                );
                assert_eq!(joint.adjoints.clear.len(), 1);
                assert_eq!(joint.adjoints.clear[0], 0..3 * width);
                let mut primals = vec![0.; joint.primal_slots];
                let mut adjoints = vec![f64::NAN; joint.adjoints.slots];
                let mut staged = vec![f64::NAN; n];
                joint
                    .run(
                        &bound,
                        &q,
                        Some(0.5),
                        &mut primals,
                        &mut adjoints,
                        &mut staged,
                    )
                    .unwrap();
                assert!(staged.iter().zip(&q).all(|(g, x)| *g == -0.5 * (x + 0.25)));
            }
            assert_eq!(
                bound.qualification_workspaces().count(),
                3 + JointTile::ALL.len()
            );
            let mut reference = bound
                .qualification_workspaces()
                .find(|(name, _)| *name == "materialized")
                .unwrap()
                .1;
            let mut reference_scores = vec![0.; n];
            for (name, mut work) in bound.qualification_workspaces() {
                #[cfg(alea_l42_joint)]
                if let Some(tile) = JointTile::ALL.into_iter().find(|tile| tile.label() == name) {
                    let width = tile.width().get();
                    assert_eq!(work.qualification_arena_bytes(), (n + 1 + 9 * width) * 8);
                    assert_eq!(
                        bound.qualification_work(&work).unwrap().map_passes,
                        8 * n.div_ceil(width)
                    );
                }
                let mut scores = vec![17.; n];
                let mut value = [19.];
                for (p, offset, seed) in [
                    (&bound, 0.25, 0.5),
                    (&rebound, -0.5, -2.),
                    (&bound, 0.25, 0.),
                ] {
                    let measured = allocation_counter::measure(|| {
                        p.value_and_vjp(&q, &[seed], &mut value, &mut scores, &mut work)
                            .unwrap();
                    });
                    assert_eq!(measured.count_total, 0);
                    let expected: f64 = q.iter().map(|x| -0.5 * (x + offset).powi(2)).sum();
                    assert_eq!(value[0].to_bits(), expected.to_bits(), "{name}");
                    p.value_and_vjp(
                        &q,
                        &[seed],
                        &mut [0.],
                        &mut reference_scores,
                        &mut reference,
                    )
                    .unwrap();
                    assert!(
                        scores
                            .iter()
                            .zip(&q)
                            .zip(&reference_scores)
                            .all(|((g, x), reference)| *g == -seed * (x + offset)
                                && g.to_bits() == reference.to_bits()),
                        "{name}"
                    );
                }
                let mut invalid = q.clone();
                invalid[n - 1] = 1e200;
                value[0] = 19.;
                scores.fill(17.);
                assert!(matches!(
                    bound.value_and_vjp(&invalid, &[0.], &mut value, &mut scores, &mut work),
                    Err(E::Numerical)
                ));
                assert_eq!(value, [19.]);
                assert!(scores.iter().all(|&g| g == 17.));
                invalid[n - 1] = 0.5;
                bound
                    .value_and_vjp(&invalid, &[2.], &mut value, &mut scores, &mut work)
                    .unwrap();
                assert_eq!(scores[n - 1], -1.5);
            }
        }
    }

    #[test]
    fn joint_tiles_stage_reverse_failure_and_decline_shared_writes() {
        let n = 4097;
        for shared in [false, true] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![n]).unwrap()).unwrap();
            let offset = if shared {
                b.input("offset", Shape::scalar()).unwrap()
            } else {
                b.constant(Shape::scalar(), vec![0.]).unwrap()
            };
            let shifted = b.binary(BinaryOp::Add, x, offset).unwrap();
            let root = b.unary(UnaryOp::Sqrt, shifted).unwrap();
            let negated = b.unary(UnaryOp::Neg, root).unwrap();
            let output = b.unary(UnaryOp::Neg, negated).unwrap();
            let output = b.sum(output).unwrap();
            let bound = b.compile(output).unwrap().bind(&[]).unwrap();
            let mut values = vec![1.; bound.dimension()];
            let mut scores = vec![31.; values.len()];
            let mut out = [37.];
            if shared {
                assert_eq!(bound.qualification_workspaces().count(), 1);
                values[n] = 0.;
                bound
                    .value_and_vjp(
                        &values,
                        &[1.],
                        &mut out,
                        &mut scores,
                        &mut bound.workspace(),
                    )
                    .unwrap();
                assert_eq!(scores[n], n as f64 / 2.);
                continue;
            }
            for (_, mut work) in bound
                .qualification_workspaces()
                .filter(|(name, _)| name.starts_with("joint"))
            {
                scores.fill(31.);
                out = [37.];
                values[n - 1] = 0.;
                assert!(matches!(
                    bound.value_and_vjp(&values, &[1.], &mut out, &mut scores, &mut work),
                    Err(E::Numerical)
                ));
                assert_eq!(out, [37.]);
                assert!(scores.iter().all(|&g| g == 31.));
                bound
                    .value_and_vjp(&values, &[0.], &mut out, &mut scores, &mut work)
                    .unwrap();
                assert_eq!(out, [(n - 1) as f64]);
                assert!(scores.iter().all(|&g| g.to_bits() == 0));
                let measured = allocation_counter::measure(|| {
                    bound.evaluate(&values, &mut out, &mut work).unwrap();
                });
                assert_eq!(measured.count_total, 0);
                assert_eq!(out, [(n - 1) as f64]);
                assert!(matches!(
                    bound.reverse_point(&[0.], &mut scores, &mut work),
                    Err(E::Shape)
                ));
                bound
                    .evaluate_with_preparation(&values, &mut out, &mut work, None)
                    .unwrap();
                bound.reverse_point(&[0.], &mut scores, &mut work).unwrap();
                assert!(scores.iter().all(|&g| g.to_bits() == 0));
                assert!(matches!(
                    bound.reverse_point(&[0.], &mut scores, &mut work),
                    Err(E::Shape)
                ));
            }
        }
    }

    #[test]
    fn connected_group_crosses_live_views_and_leaf_loads_with_rebind_and_reverse() {
        let n = 4097;
        let mut b = RegionBuilder::new();
        let vector = Shape::new(vec![n]).unwrap();
        let column = Shape::new(vec![n, 1]).unwrap();
        let x = b.input("x", vector.clone()).unwrap();
        let first = b.unary(UnaryOp::Neg, x).unwrap();
        let view = b.reshape(first, column.clone()).unwrap();
        let late_input = b.input("y", column.clone()).unwrap();
        let second = b.binary(BinaryOp::Add, view, late_input).unwrap();
        let data = b.data("d", column).unwrap();
        let third = b.binary(BinaryOp::Subtract, second, data).unwrap();
        let fourth = b.unary(UnaryOp::Neg, third).unwrap();
        let view = b.reshape(fourth, vector).unwrap();
        let output = b.sum(view).unwrap();
        let mut program = b.compile(output).unwrap();
        let [Step::Nodes(_), Step::Fused(group)] = program
            .plan
            .execution
            .as_ref()
            .unwrap()
            .materialized
            .as_slice()
        else {
            panic!("one connected group");
        };
        assert_eq!(
            group.maps().collect::<Vec<_>>(),
            [first.node, second.node, third.node, fourth.node]
        );
        assert_eq!(group.loads.as_ref(), [late_input.node, data.node]);
        assert_eq!(group.reduction, Some(output.node));
        let mut x = vec![2.; n * 2];
        x[n..].fill(3.);
        let seed = [0.5];
        let mut actual = Vec::new();
        let source = {
            let bound = program.bind(&[("d", &vec![1.; n])]).unwrap();
            let rebound = bound.rebind("d", &vec![2.; n]).unwrap();
            let mut w = bound.workspace();
            for binding in [&bound, &rebound, &bound] {
                let mut value = [19.];
                let mut gradient = vec![19.; n * 2];
                binding
                    .value_and_vjp(&x, &seed, &mut value, &mut gradient, &mut w)
                    .unwrap();
                assert!(gradient[..n].iter().all(|&v| v == 0.5));
                assert!(gradient[n..].iter().all(|&v| v == -0.5));
                actual.push((value, gradient));
            }
            bound.rust_function().unwrap()
        };
        assert_eq!(
            [actual[0].0[0], actual[1].0[0], actual[2].0[0]],
            [0., n as f64, 0.]
        );
        Arc::get_mut(&mut program.plan).unwrap().execution = None;
        let baseline = program.bind(&[("d", &vec![1.; n])]).unwrap();
        assert_eq!(baseline.rust_function().unwrap(), source);
        let rebound = baseline.rebind("d", &vec![2.; n]).unwrap();
        let mut w = baseline.workspace();
        for (binding, (value, gradient)) in [&baseline, &rebound, &baseline].into_iter().zip(actual)
        {
            let mut expected = [0.];
            let mut expected_gradient = vec![0.; n * 2];
            binding
                .value_and_vjp(&x, &seed, &mut expected, &mut expected_gradient, &mut w)
                .unwrap();
            assert_eq!(value.map(f64::to_bits), expected.map(f64::to_bits));
            assert!(
                gradient
                    .iter()
                    .zip(expected_gradient)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn unrelated_checked_maps_and_shared_reduction_stay_outside_the_group() {
        let n = 4097;
        let mut b = RegionBuilder::new();
        let shape = Shape::new(vec![n]).unwrap();
        let x = b.input("x", shape.clone()).unwrap();
        let other = b.input("other", shape.clone()).unwrap();
        let mut last = x;
        for _ in 0..4 {
            last = b.unary(UnaryOp::Neg, last).unwrap();
        }
        let view = b.reshape(last, shape).unwrap();
        let early_sum = b.sum(view).unwrap();
        let unrelated = b.unary(UnaryOp::Log, other).unwrap();
        let mut next = view;
        for _ in 0..4 {
            next = b.unary(UnaryOp::Neg, next).unwrap();
        }
        let combined = b.binary(BinaryOp::Add, unrelated, next).unwrap();
        let later_sum = b.sum(combined).unwrap();
        let root = b.binary(BinaryOp::Add, early_sum, later_sum).unwrap();
        let p = b.compile(root).unwrap().bind(&[]).unwrap();
        let [
            Step::Nodes(_),
            Step::Fused(first),
            Step::Nodes(cut),
            Step::Fused(second),
            Step::Nodes(_),
        ] = p.plan.execution.as_ref().unwrap().materialized.as_slice()
        else {
            panic!("two groups and their admission cut");
        };
        assert_eq!(first.reduction, None);
        assert!(cut.contains(&early_sum.node) && cut.contains(&unrelated.node));
        assert_eq!(second.reduction, Some(later_sum.node));
        let mut w = p.workspace();
        let (mut value, mut g) = ([17.], vec![19.; n * 2]);
        let mut q = vec![1.; n * 2];
        q[n] = -1.;
        assert!(matches!(
            p.value_and_vjp(&q, &[0.], &mut value, &mut g, &mut w),
            Err(E::Numerical)
        ));
        assert_eq!(value, [17.]);
        assert!(g.iter().all(|&g| g == 19.));
        q[n] = 1.;
        p.value_and_vjp(&q, &[1.], &mut value, &mut g, &mut w)
            .unwrap();
        assert_eq!(value, [2. * n as f64]);
        assert!(g[..n].iter().all(|&g| g == 2.));
        assert!(g[n..].iter().all(|&g| g == 1.));
    }

    #[test]
    fn numerical_admission_precedes_an_intervening_opaque_effect() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let n = 4097;
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![n]).unwrap()).unwrap();
        let mut last = b.unary(UnaryOp::Exp, x).unwrap();
        for _ in 0..3 {
            last = b.unary(UnaryOp::Neg, last).unwrap();
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let effect = b
            .register_block(
                &[],
                Arc::clone(&calls),
                BlockRules {
                    inputs: vec![],
                    output: Shape::scalar(),
                    workspace: |_| (),
                    heap_bytes: 0,
                    forward: |calls, _, output, _| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        output[0] = 2.;
                        Ok(())
                    },
                    preparation: None,
                    vjp: None,
                },
            )
            .unwrap();
        last = b.binary(BinaryOp::Add, last, effect).unwrap();
        for _ in 0..3 {
            last = b.unary(UnaryOp::Neg, last).unwrap();
        }
        let output = b.sum(last).unwrap();
        let p = b.compile(output).unwrap().bind(&[]).unwrap();
        let [
            Step::Nodes(_),
            Step::Fused(_),
            Step::Nodes(cut),
            Step::Fused(_),
        ] = p.plan.execution.as_ref().unwrap().materialized.as_slice()
        else {
            panic!("opaque boundary");
        };
        assert_eq!(*cut, effect.node..effect.node + 1);
        let mut w = p.workspace();
        let mut y = [19.];
        let mut q = vec![0.; n];
        q[n - 1] = 1000.;
        assert!(matches!(p.evaluate(&q, &mut y, &mut w), Err(E::Numerical)));
        assert_eq!(y, [19.]);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        q[n - 1] = 0.;
        p.evaluate(&q, &mut y, &mut w).unwrap();
        assert_eq!(y, [-(n as f64)]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn ordered_tiles_cross_dead_nodes_but_stop_at_live_reductions() {
        let n = 4097;
        for barrier in [false, true] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![n]).unwrap()).unwrap();
            let mut nodes = vec![b.unary(UnaryOp::Neg, x).unwrap()];
            let _dead = b.unary(UnaryOp::Log, x).unwrap();
            nodes.push(b.unary(UnaryOp::Neg, nodes[0]).unwrap());
            let s = if barrier {
                b.sum(x).unwrap()
            } else {
                // A constant must be emitted before the maps to participate
                // in fusion, so the no-barrier branch simply uses the maps.
                x
            };
            nodes.push(b.binary(BinaryOp::Add, nodes[1], s).unwrap());
            nodes.push(b.unary(UnaryOp::Neg, nodes[2]).unwrap());
            let sum = b.sum(nodes[3]).unwrap();
            let p = b.compile(sum).unwrap().bind(&[]).unwrap();
            if barrier {
                assert!(p.plan.execution.is_none());
            } else {
                let schedule = p.plan.execution.as_ref().unwrap();
                let group = schedule
                    .materialized
                    .iter()
                    .find_map(|step| match step {
                        Step::Fused(group) => Some(group),
                        _ => None,
                    })
                    .unwrap();
                assert!(group.maps().eq(nodes.iter().map(|v| v.node)));
                assert_eq!(group.domain.tiles().last(), Some(n - 1..n));
                assert_eq!(group.reduction, Some(sum.node));
            }
            let q = vec![-1.; n];
            let (mut result, mut score) = ([0.], vec![0.; n]);
            p.value_and_vjp(&q, &[1.], &mut result, &mut score, &mut p.workspace())
                .unwrap();
            let expected = if barrier { -(1. + n as f64) } else { -2. };
            assert_eq!(result, [-expected * n as f64]);
            assert!(score.iter().all(|&x| x == expected));
        }
    }

    #[test]
    fn reverse_tiles_preserve_shared_alias_contributions_and_inactive_broadcasts() {
        for scalar_parameter in [false, true] {
            let n = 8193;
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![n]).unwrap()).unwrap();
            let scale = if scalar_parameter {
                b.input("scale", Shape::scalar()).unwrap()
            } else {
                b.constant(Shape::scalar(), vec![1.25]).unwrap()
            };
            let a = b.unary(UnaryOp::Tanh, x).unwrap();
            let square = b.binary(BinaryOp::Multiply, a, a).unwrap();
            let shifted = b.binary(BinaryOp::Add, square, a).unwrap();
            let product = b.binary(BinaryOp::Multiply, shifted, scale).unwrap();
            let result = b.sum(product).unwrap();
            let mut program = b.compile(result).unwrap();
            let fused = program.bind(&[]).unwrap();
            let mut w = fused.workspace();
            let schedule = &program.plan.execution.as_ref().unwrap().materialized;
            let group = schedule
                .iter()
                .find_map(|s| {
                    if let Step::Fused(g) = s {
                        Some(g)
                    } else {
                        None
                    }
                })
                .unwrap();
            assert_eq!(
                reverse_compatible(
                    &program.plan,
                    group,
                    &fused.activity(&vec![true; program.plan.inputs.len()])
                ),
                !scalar_parameter
            );
            let mut x: Vec<_> = (0..fused.dimension())
                .map(|i| ((i % 31) as f64 / 17.).sin())
                .collect();
            if scalar_parameter {
                *x.last_mut().unwrap() = 1.25;
            }
            let mut out = [0.];
            let mut g = vec![0.; x.len()];
            fused
                .value_and_vjp(&x, &[0.7], &mut out, &mut g, &mut w)
                .unwrap();
            assert_eq!(
                allocation_counter::measure(|| {
                    fused
                        .value_and_vjp(&x, &[0.7], &mut out, &mut g, &mut w)
                        .unwrap();
                })
                .count_total,
                0
            );
            drop(fused);
            Arc::get_mut(&mut program.plan).unwrap().execution = None;
            let reference = program.bind(&[]).unwrap();
            let mut expected = [0.];
            let mut scores = vec![0.; x.len()];
            let mut reference_work = reference.workspace();
            reference
                .value_and_vjp(&x, &[0.7], &mut expected, &mut scores, &mut reference_work)
                .unwrap();
            assert_eq!(out.map(f64::to_bits), expected.map(f64::to_bits));
            assert!(
                g.iter()
                    .zip(scores)
                    .all(|(g, s)| g.to_bits() == s.to_bits())
            );
        }
    }
    #[test]
    fn fused_dags_match_unfused_values_every_score_and_error_admission() {
        for rows in [1, 2, 7, 257, 1025, 4097] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![rows, 3]).unwrap()).unwrap();
            let bias = b.input("bias", Shape::new(vec![1, 3]).unwrap()).unwrap();
            let scale = b.input("scale", Shape::scalar()).unwrap();
            let a = b.binary(BinaryOp::Add, x, bias).unwrap();
            let t = b.unary(UnaryOp::Tanh, a).unwrap();
            let product = b.binary(BinaryOp::Multiply, a, t).unwrap();
            let divided = b.binary(BinaryOp::Divide, product, scale).unwrap();
            let wave = b.unary(UnaryOp::Sin, divided).unwrap();
            let out = b.sum(wave).unwrap();
            let mut plan = b.compile(out).unwrap();
            assert_eq!(plan.plan.execution.is_some(), rows * 3 >= 4096);
            let fused = plan.bind(&[]).unwrap();
            let fused_source = fused.rust_function().unwrap();
            let mut x: Vec<_> = (0..fused.dimension())
                .map(|i| (i % 17) as f64 / 16.)
                .collect();
            *x.last_mut().unwrap() = 1.25;
            let mut w = fused.workspace();
            let (mut out, mut g) = ([0.], vec![0.; x.len()]);
            fused
                .value_and_vjp(&x, &[0.7], &mut out, &mut g, &mut w)
                .unwrap();
            assert_eq!(
                allocation_counter::measure(|| {
                    fused
                        .value_and_vjp(&x, &[0.7], &mut out, &mut g, &mut w)
                        .unwrap();
                })
                .count_total,
                0
            );
            drop(fused);
            // Same semantic graph, no fused scheduling. No arithmetic oracle
            // shares the new tile executor or source-fusion implementation.
            Arc::get_mut(&mut plan.plan).unwrap().execution = None;
            let reference = plan.bind(&[]).unwrap();
            let (mut expected, mut scores) = ([0.], vec![0.; x.len()]);
            reference
                .value_and_vjp(&x, &[0.7], &mut expected, &mut scores, &mut w)
                .unwrap();
            assert_eq!(out.map(f64::to_bits), expected.map(f64::to_bits));
            assert_eq!(
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            // AOT keeps straight-line fusion even when native blocking is
            // declined by its independently measured working-set policy.
            assert_eq!(fused_source.matches("for j in").count(), 1);
        }
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![8193]).unwrap()).unwrap();
        let one = b.constant(Shape::scalar(), vec![1.]).unwrap();
        let huge = b.unary(UnaryOp::Exp, x).unwrap();
        let finite_consumer = b.binary(BinaryOp::Divide, one, huge).unwrap();
        let square = b
            .binary(BinaryOp::Multiply, finite_consumer, finite_consumer)
            .unwrap();
        let finite_consumer = b.unary(UnaryOp::Sin, square).unwrap();
        let out = b.sum(finite_consumer).unwrap();
        let bound = b.compile(out).unwrap().bind(&[]).unwrap();
        let mut out = [41.];
        let mut g = vec![41.; 8193];
        let mut w = bound.workspace();
        assert!(matches!(
            bound.value_and_vjp(&vec![1000.; 8193], &[0.], &mut out, &mut g, &mut w),
            Err(E::Numerical)
        ));
        assert_eq!(out, [41.]);
        assert!(g.iter().all(|&g| g == 41.));
        bound
            .value_and_vjp(&vec![0.; 8193], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
    }
}
