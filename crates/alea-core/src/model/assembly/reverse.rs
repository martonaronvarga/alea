//! Compile derivative edges once. Keep the semantic/forward tape untouched:
//! even constant/inactive derivatives do not authorize skipping primal admission.
use super::{AssemblyError, Op};

#[derive(Clone)]
enum Edge {
    Parameter(usize, usize),
    Add(usize, usize),
    Multiply(usize, usize, usize),
    Log(usize, usize),
    Region(usize),
    Local(usize, Box<[(usize, usize)]>),
}

#[derive(Clone)]
pub(super) struct Plan(Vec<Edge>);
impl Plan {
    pub(super) fn new(nodes: &[Op]) -> (Self, Vec<bool>) {
        let mut dynamic = vec![false; nodes.len()];
        for (i, op) in nodes.iter().enumerate() {
            dynamic[i] = match op {
                Op::Parameter(_) => true,
                Op::Constant(_) => false,
                Op::Add(a, b) | Op::Multiply(a, b) => dynamic[*a] || dynamic[*b],
                Op::Exp(a) | Op::Log(a) => dynamic[*a],
                Op::Factor { inputs, .. } => inputs.iter().any(|i| dynamic[*i]),
                Op::Region(call) => call.inputs.iter().any(|i| dynamic[*i]),
            };
        }
        let mut edges = Vec::new();
        for (i, op) in nodes.iter().enumerate().rev() {
            // Preserve the old reverse node and operand order, including
            // repeated destinations; only mathematically constant edges vanish.
            match op {
                Op::Parameter(s) => edges.push(Edge::Parameter(i, *s)),
                Op::Constant(_) => {}
                Op::Region(_) if dynamic[i] => edges.push(Edge::Region(i)),
                Op::Region(_) => {}
                Op::Add(a, b) => {
                    for &out in [a, b] {
                        if dynamic[out] {
                            edges.push(Edge::Add(i, out));
                        }
                    }
                }
                Op::Multiply(a, b) => {
                    for (&out, &other) in [(a, b), (b, a)] {
                        if dynamic[out] {
                            edges.push(Edge::Multiply(i, out, other));
                        }
                    }
                }
                Op::Exp(a) => {
                    if dynamic[*a] {
                        edges.push(Edge::Multiply(i, *a, i));
                    }
                }
                Op::Log(a) => {
                    if dynamic[*a] {
                        edges.push(Edge::Log(i, *a));
                    }
                }
                Op::Factor { inputs, start, .. } => {
                    let destinations: Box<[_]> = inputs
                        .iter()
                        .enumerate()
                        .filter(|(_, out)| dynamic[**out])
                        .map(|(j, &out)| (out, start + j))
                        .collect();
                    if !destinations.is_empty() {
                        edges.push(Edge::Local(i, destinations));
                    }
                }
            }
        }
        (Self(edges), dynamic)
    }
    pub(super) fn apply<const FILTERED: bool>(
        &self,
        values: &[f64],
        adjoints: &mut [f64],
        locals: &[f64],
        gradient: &mut [f64],
        active: Option<&[bool]>,
        mut region: impl FnMut(usize, &mut [f64]) -> Result<(), AssemblyError>,
    ) -> Result<(), AssemblyError> {
        for edge in &self.0 {
            let (Edge::Parameter(node, _)
            | Edge::Add(node, _)
            | Edge::Multiply(node, _, _)
            | Edge::Log(node, _)
            | Edge::Region(node)
            | Edge::Local(node, _)) = edge;
            if FILTERED && !active.expect("selected graph")[*node] {
                continue;
            }
            let a = adjoints[*node];
            match edge {
                Edge::Region(node) => region(*node, adjoints)?,
                Edge::Parameter(_, slot) => gradient[*slot] += a,
                Edge::Add(_, out) => adjoints[*out] += a,
                Edge::Multiply(_, out, other) => adjoints[*out] += a * values[*other],
                Edge::Log(_, out) => adjoints[*out] += a / values[*out],
                Edge::Local(_, destinations) => {
                    let mut add = |&(out, local): &(usize, usize)| {
                        adjoints[out] += a * locals[local];
                    };
                    // Each factor's active operands stay in their original
                    // order, including aliases. Admit the source node once.
                    match destinations.as_ref() {
                        [x] => add(x),
                        [x, y] => {
                            add(x);
                            add(y);
                        }
                        [x, y, z] => {
                            add(x);
                            add(y);
                            add(z);
                        }
                        [x, y, z, w] => {
                            add(x);
                            add(y);
                            add(z);
                            add(w);
                        }
                        many => many.iter().for_each(add),
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_derivative_edges_do_not_remove_primal_or_local_score_admission() {
        use crate::model::{
            ConstrainedModel,
            assembly::{AssemblyError, ModelBuilder},
        };
        for malformed in [false, true] {
            let mut b = ModelBuilder::new();
            let _q = b.real("q").unwrap();
            let value = b.constant(1.).unwrap();
            let root = b
                .custom(&[value], move |_, g| {
                    g[0] = if malformed { f64::NAN } else { 1. };
                    if malformed {
                        Ok(0.)
                    } else {
                        Err(AssemblyError::Shape)
                    }
                })
                .unwrap();
            b.factor(root).unwrap();
            let model = b.compile().unwrap();
            let mut g = [17.];
            let result = model.logp_grad(&[0.], &mut g);
            assert!(if malformed {
                matches!(result, Err(AssemblyError::Numerical))
            } else {
                matches!(result, Err(AssemblyError::Shape))
            });
            assert_eq!(g, [17.]);
        }
    }
    #[test]
    fn compiled_edges_preserve_interpreter_gradient_bits_with_sharing_and_constants() {
        for round in 0..32 {
            let mut nodes = vec![Op::Parameter(0), Op::Parameter(1), Op::Constant(0.5)];
            let mut local_count = 0;
            let mut random = round + 17_u64;
            for i in 3..400 {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = (random as usize) % i;
                let b = ((random >> 32) as usize) % i;
                nodes.push(match i % 6 {
                    0 => Op::Add(a, b),
                    1 => Op::Multiply(a, b),
                    2 => Op::Exp(a),
                    3 => Op::Log(a),
                    4 => Op::Constant(1.),
                    _ => {
                        let start = local_count;
                        local_count += 3;
                        Op::Factor {
                            inputs: vec![a, b, a].into(),
                            callback: 0,
                            start,
                        }
                    }
                });
            }
            let values: Vec<_> = (0..nodes.len())
                .map(|i| [0.25, 0.5, 1., 2.][i % 4])
                .collect();
            let locals: Vec<_> = (0..local_count)
                .map(|i| [0., 0.25, -0.5, 1.][i % 4])
                .collect();
            let mut expected = vec![0.; nodes.len()];
            for i in (0..nodes.len()).step_by(11) {
                expected[i] = 1.;
            }
            let mut actual = expected.clone();
            let mut a = [0.; 2];
            let mut b = a;
            for (i, op) in nodes.iter().enumerate().rev() {
                let seed = expected[i];
                match op {
                    Op::Parameter(s) => a[*s] += seed,
                    Op::Constant(_) => {}
                    Op::Region(_) => unreachable!(),
                    Op::Add(x, y) => {
                        expected[*x] += seed;
                        expected[*y] += seed;
                    }
                    Op::Multiply(x, y) => {
                        expected[*x] += seed * values[*y];
                        expected[*y] += seed * values[*x];
                    }
                    Op::Exp(x) => expected[*x] += seed * values[i],
                    Op::Log(x) => expected[*x] += seed / values[*x],
                    Op::Factor { inputs, start, .. } => {
                        for (j, x) in inputs.iter().enumerate() {
                            expected[*x] += seed * locals[start + j];
                        }
                    }
                }
            }
            Plan::new(&nodes)
                .0
                .apply::<false>(
                    &values,
                    &mut actual,
                    &locals,
                    &mut b,
                    None,
                    |_, _| unreachable!(),
                )
                .unwrap();
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits), "round={round}");
        }
    }
}
