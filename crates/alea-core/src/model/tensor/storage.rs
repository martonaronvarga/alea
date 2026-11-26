//! Physical slot assignment derived from reachability and inclusive lifetimes.
use super::*;

fn add(a: usize, b: usize) -> Result<usize, E> {
    a.checked_add(b)
        .filter(|&v| v <= isize::MAX as usize / size_of::<f64>())
        .ok_or(E::Shape)
}

// Preserve allocation order for the native slice kernels. Dead values acquire
// no physical slots, but retain their logical identities and formal inputs.
pub(super) fn pack_primals(nodes: &mut [Node], active: &[bool]) -> Result<usize, E> {
    let mut slots = 0;
    for i in 0..nodes.len() {
        nodes[i].start = if !active[i] {
            0
        } else if let Op::Alias(value) = nodes[i].op {
            nodes[value.node].start
        } else {
            let start = slots;
            if !nodes[i].discrete() {
                slots = add(slots, nodes[i].shape.elements)?;
            }
            start
        };
        nodes[i].primal_start = nodes[i].start;
    }
    Ok(slots)
}

pub(super) fn alias_owners(p: &Plan) -> Vec<usize> {
    let mut owners: Vec<_> = (0..p.nodes.len()).collect();
    for call in p.facts.lowered.calls() {
        if let Some(value) = call.alias() {
            owners[call.origin().node] = owners[value.node];
        }
    }
    owners
}

// One real allocation per alias class. The final forward position stands for
// every later read: only the forward phase writes the primal arena.
pub(super) fn reuse_primals(p: &mut Plan) -> Result<(), E> {
    let count = p.nodes.len();
    let owner = alias_owners(p);
    let mut birth: Vec<_> = (1..=count).collect();
    let mut death = birth.clone();
    death[owner[p.output.node]] = count;
    for (i, node) in p.nodes.iter().enumerate() {
        let Some(call) = p.facts.lowered.call(i) else {
            continue;
        };
        if matches!(node.op, Op::Data(_) | Op::Constant(_)) || p.facts.binding_fixed(i) {
            // These may be populated before forward, then kept across calls.
            birth[owner[i]] = 0;
            death[owner[i]] = count;
        }
        for value in node.inputs() {
            let end = &mut death[owner[value.node]];
            *end = (*end).max(i + 1);
        }
        for value in call.retained_primals() {
            death[owner[value.node]] = count;
        }
    }
    let mut intervals = Vec::new();
    for (i, node) in p.nodes.iter().enumerate() {
        if p.active[i] && owner[i] == i && !node.range().is_empty() {
            intervals.push((birth[i], death[i], i, node.shape.elements));
        }
    }
    intervals.sort_unstable();
    let (mut offsets, slots) = assign_intervals(&intervals, count)?;
    if slots >= p.slots {
        return Ok(());
    }
    verify_intervals(&intervals, &offsets, slots)?;
    for (i, &root) in owner.iter().enumerate() {
        if root != i {
            offsets[i] = offsets[root];
        }
    }
    for (node, start) in p.nodes.iter_mut().zip(offsets) {
        node.primal_start = start;
    }
    p.primal_slots = slots;
    p.facts.lowered.place_primals(&p.nodes);
    Ok(())
}

// One traversal assigns the source output/child ranges and checks the arena
// extent. Callers that only need the budget do not allocate an offset table.
pub(super) fn source_forward_layout(
    p: &Plan,
    mut place: impl FnMut(usize, Range<usize>),
) -> Result<usize, E> {
    let mut slots = p.nodes[p.output.node].shape.elements;
    for call in p.facts.lowered.calls() {
        if let Some(len) = call.source_storage() {
            let end = add(slots, len)?;
            place(call.origin().node, slots..end);
            slots = end;
        }
        let (arguments, child) = call.operation(&p.nodes).child_scratch();
        slots = add(slots, add(arguments, child)?)?;
    }
    Ok(slots)
}

// Which computed primals are actually read by the explicit reverse rules?
// Forward admission still evaluates every selected operation. Borrowed inputs,
// immutable bindings and constants remain available without retained copies.
// Control arguments/predicates are retained because child reverse replay admits
// the complete selected primal, not just its differentiable operands.
pub(super) struct SourceResidualLayout {
    pub(super) offsets: Vec<usize>,
    pub(super) owners: Vec<usize>,
    pub(super) slots: usize,
    pub(super) copies: Vec<(usize, usize, usize)>,
}
pub(super) fn source_residuals(p: &Plan, active: &[bool]) -> Result<SourceResidualLayout, E> {
    let mut full = vec![0; p.nodes.len()];
    let end = source_forward_layout(p, |i, range| full[i] = range.start)?;
    debug_assert_eq!(end, p.source_slots);
    let mut needed = vec![false; p.nodes.len()];
    let owner = alias_owners(p);
    let demand = p.facts.demands(active);
    for read in demand.backward_reads() {
        debug_assert_eq!(read.reader.owner, p.owner);
        debug_assert_eq!(read.value.owner, p.owner);
        needed[owner[read.value.node]] = true;
    }
    let output = owner[demand.result().node];
    let mut offsets = vec![0; p.nodes.len()];
    let mut slots = p.nodes[p.output.node].shape.elements;
    let mut copies = Vec::new();
    for call in p.facts.lowered.calls() {
        let i = call.origin().node;
        if !needed[i] || i == output {
            continue;
        }
        if let Some(len) = call.source_storage() {
            offsets[i] = slots;
            copies.push((full[i], slots, len));
            slots = add(slots, len)?;
        }
    }
    for (i, &root) in owner.iter().enumerate() {
        if root != i {
            offsets[i] = offsets[root];
        }
    }
    Ok(SourceResidualLayout {
        offsets,
        owners: owner,
        slots,
        copies,
    })
}
// Primal slots include immutable bindings and constants; their adjoints do not
// exist. Aliases share their producer's range, exactly as in the semantic plan.
pub(super) fn adjoint_layout(p: &Plan, active: &[bool]) -> Result<(Vec<usize>, usize), E> {
    let mut offsets = vec![0; p.nodes.len()];
    let mut slots = 0;
    for call in p.facts.lowered.calls() {
        let i = call.origin().node;
        #[cfg(test)]
        if p.nodes[i].discrete() {
            continue;
        }
        if !p.facts.demands(active).cotangent(i)
            || matches!(
                call.operation(&p.nodes),
                lowering::Operation::Data(_) | lowering::Operation::Constant(_)
            )
        {
            continue;
        }
        if let Some(v) = call.alias() {
            offsets[i] = offsets[v.node];
        } else {
            offsets[i] = slots;
            slots = add(slots, call.elements())?;
        }
    }
    Ok((offsets, slots))
}

pub(super) struct AdjointPlacement {
    pub(super) starts: Vec<usize>,
    pub(super) slots: usize,
    pub(super) seed_slots: Option<usize>,
    pub(super) clear: Vec<Range<usize>>,
    pub(super) planned: bool,
}

#[cfg(any(alea_l42_joint, test))]
pub(super) fn tile_slots(
    count: usize,
    nodes: impl Iterator<Item = usize>,
    width: usize,
    prefix: usize,
) -> Result<(Vec<usize>, usize), E> {
    // The current tile kernels check the whole arena at tile retirement. Treat
    // all slots as live until then, including a seed still read by its VJP.
    let intervals: Vec<_> = nodes
        .enumerate()
        .map(|(birth, node)| (birth, count, node, width))
        .collect();
    let (mut starts, slots) = assign_intervals(&intervals, count)?;
    verify_intervals(&intervals, &starts, slots)?;
    let slots = add(prefix, slots)?;
    for &(_, _, node, _) in &intervals {
        starts[node] += prefix;
    }
    Ok((starts, slots))
}

#[cfg(any(alea_l42_joint, test))]
pub(super) fn tiled_adjoints(
    p: &Plan,
    active: &[bool],
    owners: &[usize],
    nodes: impl Iterator<Item = usize>,
    width: usize,
    writes: Option<&lowering::AdjointWrites>,
) -> Result<AdjointPlacement, E> {
    let (mut starts, slots) = tile_slots(p.nodes.len(), nodes, width, 0)?;
    let clear = initial_adjoint_ranges(p, active, &starts, writes, |i| {
        if i == p.output.node { 0 } else { width }
    });
    for (i, &owner) in owners.iter().enumerate() {
        starts[i] = starts[owner];
    }
    Ok(AdjointPlacement {
        starts,
        slots,
        seed_slots: None,
        clear,
        planned: true,
    })
}

pub(super) fn materialized_adjoints(
    p: &Plan,
    active: &[bool],
    tiled_reverse: bool,
    metadata_bytes: usize,
) -> Result<AdjointPlacement, E> {
    let (mut starts, mut slots) = adjoint_layout(p, active)?;
    let mut seed_slots = None;
    if !tiled_reverse && let Some((offsets, span, seed)) = reused_adjoint_layout(p, active, slots)?
    {
        starts = offsets;
        slots = span;
        seed_slots = Some(seed);
    }
    let planned = tiled_reverse
        || seed_slots.is_some()
        || (p.slots - slots) * size_of::<f64>() > metadata_bytes;
    if !planned {
        slots = p.slots;
        for (start, node) in starts.iter_mut().zip(&p.nodes) {
            *start = node.start;
        }
    }
    let clear = if seed_slots.is_some() {
        std::iter::once(0..slots).collect()
    } else {
        initial_adjoint_ranges(p, active, &starts, p.adjoint_writes.as_deref(), |i| {
            p.nodes[i].shape.elements
        })
    };
    Ok(AdjointPlacement {
        starts,
        slots,
        seed_slots,
        clear,
        planned,
    })
}

fn initial_adjoint_ranges(
    p: &Plan,
    active: &[bool],
    starts: &[usize],
    writes: Option<&lowering::AdjointWrites>,
    width: impl Fn(usize) -> usize,
) -> Vec<Range<usize>> {
    let mut clear: Vec<Range<usize>> = Vec::with_capacity(p.nodes.len());
    let mut replaced = writes.iter().flat_map(|w| w.replaced(active)).peekable();
    for (i, node) in p.nodes.iter().enumerate() {
        let len = width(i);
        if p.facts.admission(i)
            && active[i]
            && !matches!(node.op, Op::Alias(_))
            && len != 0
            && replaced.next_if_eq(&i).is_none()
        {
            let start = starts[i];
            if let Some(last) = clear.last_mut().filter(|last| last.end == start) {
                last.end = start + len;
            } else {
                clear.push(start..start + len);
            }
        }
    }
    clear
}

// Backward live intervals are inclusive: an output seed and its input
// destinations must coexist during their VJP. Alias classes die at the original
// producer, not at a reshape. Best-fit free blocks give O((V+E) log V) planning;
// fragmentation may decline reuse, never exceed the dense reference budget.
pub(super) fn reused_adjoint_layout(
    p: &Plan,
    active: &[bool],
    dense: usize,
) -> Result<Option<(Vec<usize>, usize, usize)>, E> {
    // Keep tiny/control-model construction out of the interval sweep.
    if dense < 1024 * 1024 / size_of::<f64>() {
        return Ok(None);
    }
    let count = p.nodes.len();
    let owner = alias_owners(p);
    let mut first: Vec<_> = (0..count).collect();
    for (i, n) in p.nodes.iter().enumerate() {
        if !p.active[i] || !active[i] {
            continue;
        }
        for input in n.inputs() {
            if active[input.node] {
                first[owner[input.node]] = first[owner[input.node]].max(i);
            }
        }
    }
    if active[p.output.node] {
        first[owner[p.output.node]] = count;
    }
    let mut intervals = Vec::new();
    let mut scratch = 0;
    for (i, n) in p.nodes.iter().enumerate() {
        #[cfg(test)]
        if n.discrete() {
            continue;
        }
        if p.active[i]
            && active[i]
            && owner[i] == i
            && n.shape.elements != 0
            && !matches!(n.op, Op::Data(_) | Op::Constant(_))
        {
            intervals.push((count - first[i], count - i, i, n.shape.elements));
            scratch = scratch.max(n.shape.elements);
        }
    }
    intervals.sort_unstable();
    let (mut offsets, slots) = assign_intervals(&intervals, count)?;
    if add(slots, scratch)? >= dense {
        return Ok(None);
    }
    verify_intervals(&intervals, &offsets, slots)?;
    for (i, &root) in owner.iter().enumerate() {
        if root != i {
            offsets[i] = offsets[root];
        }
    }
    Ok(Some((offsets, slots, scratch)))
}

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
};

// (inclusive birth, inclusive death, owner, length), in birth order.
type Interval = (usize, usize, usize, usize);

fn assign_intervals(intervals: &[Interval], count: usize) -> Result<(Vec<usize>, usize), E> {
    let mut free: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut live = BinaryHeap::new();
    let mut slots = 0;
    let mut offsets = vec![0; count];
    for &(birth, death, i, len) in intervals {
        while live.peek().is_some_and(|Reverse((end, _, _))| *end < birth) {
            let Reverse((_, base, size)) = live.pop().expect("peeked live interval");
            free.entry(size).or_default().push(base);
        }
        let block = free.range(len..).next().map(|(&size, _)| size);
        let base = if let Some(size) = block {
            let entries = free.get_mut(&size).expect("selected free block");
            let base = entries.pop().expect("nonempty free block");
            if entries.is_empty() {
                free.remove(&size);
            }
            if size > len {
                free.entry(size - len).or_default().push(add(base, len)?);
            }
            base
        } else {
            let base = slots;
            slots = add(slots, len)?;
            base
        };
        offsets[i] = base;
        live.push(Reverse((death, base, len)));
    }
    Ok((offsets, slots))
}

fn verify_intervals(intervals: &[Interval], offsets: &[usize], slots: usize) -> Result<(), E> {
    let mut retire = BinaryHeap::new();
    let mut live = BTreeMap::new();
    for &(birth, death, i, len) in intervals {
        while retire.peek().is_some_and(|Reverse((end, _))| *end < birth) {
            let Reverse((_, start)) = retire.pop().expect("checked retirement");
            live.remove(&start);
        }
        let start = offsets[i];
        let end = add(start, len)?;
        if end > slots
            || live
                .range(..end)
                .next_back()
                .is_some_and(|(_, &last)| last > start)
        {
            return Err(E::Shape);
        }
        live.insert(start, end);
        retire.push(Reverse((death, start)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_overlap_requires_strict_retirement_even_with_distinct_owners() {
        let first = (1, 3, 0, 8);
        let simultaneous = (3, 4, 1, 4);
        let after = (4, 4, 1, 4);
        assert!(verify_intervals(&[first, simultaneous], &[0, 4], 8).is_err());
        assert!(verify_intervals(&[first, simultaneous], &[0, 8], 12).is_ok());
        assert!(verify_intervals(&[first, after], &[0, 4], 8).is_ok());
        assert!(verify_intervals(&[first, after], &[0, usize::MAX], 8).is_err());

        let mut storage = [0., 1., 2., 3., 4., 5.];
        let (reads, write) = lowering::PrimalView::split(&mut storage, 2..4);
        assert_eq!(reads.read(0..2), [0., 1.]);
        assert_eq!(reads.read(4..6), [4., 5.]);
        assert!(reads.get(1..3).is_none());
        assert!(reads.get(2..5).is_none());
        write.fill(7.);
        assert_eq!(storage, [0., 1., 7., 7., 4., 5.]);
    }

    #[test]
    fn source_offsets_reserve_child_frames_and_keep_only_reverse_reachable_owners() {
        let branch = |extra| {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::scalar()).unwrap();
            let mut y = b.unary(UnaryOp::Sin, x).unwrap();
            if extra {
                y = b.unary(UnaryOp::Exp, y).unwrap();
            }
            b.compile(y).unwrap().bind(&[]).unwrap()
        };
        let mut body = RegionBuilder::new();
        let carry = body.input("carry", Shape::scalar()).unwrap();
        let row = body.input("row", Shape::scalar()).unwrap();
        let next = body.binary(BinaryOp::Add, carry, row).unwrap();
        let body = body.compile(next).unwrap().bind(&[]).unwrap();

        for steps in [0, 3] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::scalar()).unwrap();
            let rows = b.input("rows", Shape::new(vec![steps]).unwrap()).unwrap();
            let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
            let empty = b.gather(x, Shape::new(vec![0]).unwrap(), vec![]).unwrap();
            let empty_total = b.sum(empty).unwrap();
            let initial = b
                .cond(x, zero, Comparison::Less, &[x], branch(false), branch(true))
                .unwrap();
            let alias = b.reshape(initial, Shape::scalar()).unwrap();
            let scanned = b.scan(alias, rows, &[], body.clone()).unwrap();
            let output = b.binary(BinaryOp::Add, scanned, empty_total).unwrap();
            let p = b.compile(output).unwrap();
            let mut placements = Vec::new();
            let slots = source_forward_layout(&p.plan, |i, r| placements.push((i, r))).unwrap();
            let total_start = if steps == 0 { 8 } else { 12 };
            assert_eq!(slots, total_start + 1);
            assert_eq!(slots, p.plan.source_slots);
            assert_eq!(
                placements,
                [
                    (empty.node, 1..1),
                    (empty_total.node, 1..2),
                    (initial.node, 2..3),
                    (scanned.node, 7..8),
                    (output.node, total_start..total_start + 1),
                ]
            );
            let active = p.plan.facts.activity(&[true, true]);
            let residuals = source_residuals(&p.plan, &active).unwrap();
            if steps == 0 {
                assert_eq!(residuals.slots, 1);
                assert!(residuals.copies.is_empty());
            } else {
                assert_eq!(residuals.slots, 2);
                assert_eq!(residuals.copies, [(2, 1, 1)]);
                assert_eq!(residuals.offsets[initial.node], 1);
                assert_eq!(residuals.offsets[alias.node], 1);
            }
        }
    }
}
