//! Dependency-safe pointwise DAG/map-reduction scheduling. No reassociation.
use super::*;
use alea_math::numeric::{CompensatedSum, first_nonfinite};

pub(super) enum Step {
    Nodes(Range<usize>),
    Fused(Group),
}
// Most plans (especially scalar scan bodies) have no profitable native group.
// Keep optional scheduler metadata out of their hot immutable plan layout.
pub(super) struct Schedule(pub(super) Vec<Step>);
pub(super) struct Group {
    pub(super) maps: Range<usize>,
    pub(super) elements: usize,
    pub(super) reduction: Option<usize>,
}
impl Group {
    pub(super) fn end(&self) -> usize {
        self.reduction.map_or(self.maps.end, |i| i + 1)
    }
}

fn identity(map: &Map) -> bool {
    match map {
        Map::Identity => true,
        Map::Empty | Map::Scalar => false,
        // Equal-element broadcast shapes (including leading unit axes) are
        // canonicalized to Identity at construction. Every other nonempty
        // broadcast repeats destinations and cannot use independent VJP tiles.
        Map::Broadcast(_) => false,
    }
}
fn pointwise(p: &Plan, i: usize, first: usize, elements: usize) -> bool {
    let node = &p.nodes[i];
    if !p.active[i] || node.shape.elements != elements {
        return false;
    }
    match &node.op {
        Op::Unary(_, _) => true,
        Op::Binary(b) => {
            // An in-group value must be read at the same element. An external
            // operand can use ANY static broadcast map, including repeats.
            (b.a.node < first || identity(&b.left)) && (b.b.node < first || identity(&b.right))
        }
        _ => false,
    }
}

pub(super) fn schedule(p: &Plan, native: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    let (mut i, mut plain) = (0, 0);
    while i < p.nodes.len() {
        let first = i;
        let elements = p.nodes[i].shape.elements;
        if elements < 2 || !pointwise(p, i, first, elements) {
            i += 1;
            continue;
        }
        while i < p.nodes.len() && pointwise(p, i, first, elements) {
            i += 1;
        }
        let reduction = (i < p.nodes.len() && p.active[i])
            .then(|| match p.nodes[i].op {
                Op::Sum(a) if (first..i).contains(&a.node) => Some(i),
                _ => None,
            })
            .flatten();
        if i - first < 2 && reduction.is_none() {
            continue;
        }
        // Paired controls reject short-chain/small-array blocking: it adds
        // dispatch without a useful working-set reduction. AOT straight-line
        // fusion has a different cost model and retains all eligible groups.
        if native && (elements < 4096 || i - first < 4) {
            continue;
        }
        if plain != first {
            steps.push(Step::Nodes(plain..first));
        }
        let group = Group {
            maps: first..i,
            elements,
            reduction,
        };
        i = group.end();
        plain = i;
        steps.push(Step::Fused(group));
    }
    if steps.is_empty() {
        return steps;
    }
    if plain != p.nodes.len() {
        steps.push(Step::Nodes(plain..p.nodes.len()));
    }
    steps
}

// Tile scheduling retains vector loops/dispatch per block, never an interpreted
// instruction dispatch per observation. Materialized primals remain available to
// the ORIGINAL reverse order. Shared/scalar adjoints must not be tiled/reordered.
pub(super) fn forward(p: &Plan, group: &Group, values: &mut [f64]) -> Result<(), E> {
    const TILE: usize = 256;
    let mut sum = CompensatedSum::default();
    for start in (0..group.elements).step_by(TILE) {
        let end = (start + TILE).min(group.elements);
        for i in group.maps.clone() {
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
    group.maps.clone().all(|i| match &p.nodes[i].op {
        Op::Unary(_, _) => true,
        Op::Binary(b) => {
            (!active[b.a.node] || identity(&b.left)) && (!active[b.b.node] || identity(&b.right))
        }
        _ => false,
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
    const TILE: usize = 256;
    for start in (0..group.elements).step_by(TILE) {
        let end = (start + TILE).min(group.elements);
        if let Some(i) = group.reduction.filter(|&i| active[i]) {
            let Op::Sum(a) = p.nodes[i].op else {
                unreachable!("compiled reduction")
            };
            let seed = adjoints[starts[i]];
            for g in &mut adjoints[starts[a.node] + start..starts[a.node] + end] {
                *g += seed;
            }
        }
        for i in group.maps.clone().rev() {
            if !active[i] {
                continue;
            }
            let node = &p.nodes[i];
            match &node.op {
                Op::Unary(op, a) => {
                    kernels::unary_reverse_range(
                        *op,
                        &values[p.nodes[a.node].start + start..p.nodes[a.node].start + end],
                        &values[node.start + start..node.start + end],
                        adjoints,
                        (starts[a.node] + start, starts[i] + start),
                    );
                }
                Op::Binary(b) => {
                    let (before, after) = adjoints.split_at_mut(starts[i]);
                    kernels::reverse_range(
                        b,
                        kernels::Pullback {
                            a: &values[p.nodes[b.a.node].range()],
                            r: &values[p.nodes[b.b.node].range()],
                            seeds: &after[start..end],
                            adjoints: before,
                            starts: (starts[b.a.node], starts[b.b.node]),
                            active: (active[b.a.node], active[b.b.node]),
                        },
                        start,
                    );
                }
                _ => unreachable!("compiled pointwise group"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            let schedule = &program.plan.execution.as_ref().unwrap().0;
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
            reference
                .value_and_vjp(&x, &[0.7], &mut expected, &mut scores, &mut w)
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
            let fused_source = fused.rust_function();
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
