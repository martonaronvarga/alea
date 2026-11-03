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
        Map::Scalar => false,
        Map::Gather(indices) => indices.iter().enumerate().all(|(i, &j)| i == j),
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
            BinaryOp::Add => mapped(b, a, r, out, start, |a, b| a + b),
            BinaryOp::Subtract => mapped(b, a, r, out, start, |a, b| a - b),
            BinaryOp::Multiply => mapped(b, a, r, out, start, |a, b| a * b),
            BinaryOp::Divide => mapped(b, a, r, out, start, |a, b| a / b),
        },
    }
}
fn mapped(
    b: &Binary,
    a: &[f64],
    r: &[f64],
    out: &mut [f64],
    start: usize,
    f: impl Fn(f64, f64) -> f64,
) {
    for (j, out) in out.iter_mut().enumerate() {
        *out = f(a[b.left.at(start + j)], r[b.right.at(start + j)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_dags_match_unfused_values_every_score_and_error_admission() {
        for rows in [1, 2, 7, 257, 1025, 4097] {
            let mut b = TensorBuilder::new();
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
        let mut b = TensorBuilder::new();
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
