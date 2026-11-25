//! Immutable, conservative facts for the existing region lowerings. Formal
//! inputs are parameter-capable; callers substitute their own activity masks.
//! Purity never implies totality or permission to skip forward admission.
use super::*;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Sources(u8);
impl Sources {
    const INPUT: Self = Self(1);
    const DATA: Self = Self(2);
    const CONSTANT: Self = Self(4);
    // A callback can read captured state absent from its explicit arguments.
    const OPAQUE: Self = Self(8);
    fn include(&mut self, other: Self) {
        self.0 |= other.0;
    }
    fn closed(self) -> Self {
        Self(self.0 & !Self::INPUT.0)
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Effects {
    // Requires checked forward admission, regardless of derivative activity.
    checked: bool,
    // Unknown effects: neither referential purity nor replay may be inferred.
    opaque: bool,
}
impl Effects {
    fn include(&mut self, other: Self) {
        self.checked |= other.checked;
        self.opaque |= other.opaque;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FrameInstance {
    Branch(usize),
    Iteration(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResidualGuard {
    pub(super) site: Value,
    pub(super) instance: FrameInstance,
    pub(super) child: u64,
}
#[derive(Clone, Copy)]
enum Control {
    None,
    Branch([u64; 2]),
    Scan { child: u64, steps: usize },
}

/// A read remains attached to its original SSA reader, even through aliases.
/// The condition describes derivative demand, never permission to skip admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BackwardRead {
    pub(super) reader: Value,
    pub(super) value: Value,
    when: Option<Value>,
}

pub(super) struct Demands<'a> {
    facts: &'a CompileFacts,
    activity: &'a [bool],
}
impl Demands<'_> {
    pub(super) fn admission(&self, node: usize) -> bool {
        self.facts.admission(node)
    }
    // Current first-order planners conservatively use the same structural mask
    // for tangent dependence and cotangent execution. Numeric zero seeds do not
    // narrow it. This does not add a JVP capability.
    pub(super) fn tangent(&self, node: usize) -> bool {
        self.activity[node]
    }
    pub(super) fn cotangent(&self, node: usize) -> bool {
        self.admission(node) && self.tangent(node)
    }
    pub(super) fn result(&self) -> Value {
        self.facts.result
    }
    pub(super) fn backward_reads(&self) -> impl Iterator<Item = BackwardRead> + '_ {
        self.facts
            .lowered
            .backward_reads()
            .iter()
            .copied()
            .filter(|r| {
                self.cotangent(r.reader.node) && r.when.is_none_or(|v| self.tangent(v.node))
            })
    }
}

// Only explicit source reverse reads are specialized here. Native opaque rules
// retain their complete admitted input/output views; no callback read is guessed.
pub(super) fn backward_reads(node: &Node, reader: Value, reads: &mut Vec<BackwardRead>) {
    let mut read = |value, when| {
        reads.push(BackwardRead {
            reader,
            value,
            when,
        })
    };
    match &node.op {
        Op::Unary(op, v) => match op {
            UnaryOp::Neg => {}
            UnaryOp::Exp | UnaryOp::Tanh | UnaryOp::Sqrt => read(reader, None),
            _ => read(*v, None),
        },
        Op::Binary(b) => match b.kind {
            BinaryOp::Add | BinaryOp::Subtract => {}
            BinaryOp::Multiply => {
                read(b.b, Some(b.a));
                read(b.a, Some(b.b));
            }
            BinaryOp::Divide => {
                read(b.b, None);
                read(b.a, Some(b.b));
            }
        },
        Op::Matmul { a, b, .. } => {
            read(*b, Some(*a));
            read(*a, Some(*b));
        }
        Op::Cond(c) => {
            read(c.lhs, None);
            read(c.rhs, None);
            for &v in &c.args {
                read(v, None);
            }
        }
        Op::Scan(s) if s.steps != 0 => {
            read(s.initial, None);
            read(s.xs, None);
            for &v in &s.captures {
                read(v, None);
            }
        }
        // Opaque callbacks are source-ineligible, not assigned fake read sets.
        _ => {}
    }
}

struct ValueFacts {
    input: Option<usize>,
    derivatives: Range<usize>,
    sources: Sources,
    effects: Effects,
    admission: bool,
    binding_fixed: bool,
    control: Control,
}

pub(super) struct CompileFacts {
    values: Box<[ValueFacts]>,
    pub(super) lowered: lowering::Contract,
    pub(super) binding_data: Box<[usize]>,
    result: Value,
    derivatives: Box<[usize]>,
    output_inputs: Box<[usize]>,
    output_derivatives: Box<[usize]>,
    output_sources: Sources,
    last_binding_fixed: Option<usize>,
    effects: Effects,
    pub(super) missing_source: Option<usize>,
    pub(super) explicit_source: bool,
    pub(super) source_checkpoints: bool,
    pub(super) last_native_block: Option<usize>,
}

// Extract formal dependence without a dense per-value/formal-input matrix.
fn output_inputs(nodes: &[Node], edges: &[Vec<usize>], output: usize) -> Box<[usize]> {
    let mut needed = vec![false; nodes.len()];
    needed[output] = true;
    let mut inputs = Vec::new();
    for i in (0..nodes.len()).rev() {
        if !needed[i] {
            continue;
        }
        if let Op::Input(j) = nodes[i].op {
            inputs.push(j);
        }
        for &j in &edges[i] {
            needed[j] = true;
        }
    }
    inputs.sort_unstable();
    inputs.dedup();
    inputs.into_boxed_slice()
}

impl CompileFacts {
    pub(super) fn compile(
        nodes: &[Node],
        active: &[bool],
        output: Value,
        slots: usize,
    ) -> Result<Self, E> {
        let mut values: Vec<ValueFacts> = Vec::with_capacity(nodes.len());
        let mut value_edges = vec![Vec::new(); nodes.len()];
        let mut derivative_edges = vec![Vec::new(); nodes.len()];
        let mut derivatives = Vec::new();
        let mut reads = Vec::new();
        let mut binding_data = Vec::new();
        let mut effects = Effects::default();
        let mut source_checkpoints = false;
        let mut last_native_block = None;
        for (i, node) in nodes.iter().enumerate() {
            let mut sources = Sources::default();
            let mut local_effects = Effects::default();

            let mut input = None;
            if active[i] {
                backward_reads(
                    node,
                    Value {
                        owner: output.owner,
                        node: i,
                    },
                    &mut reads,
                );
                let operands = node.inputs();
                value_edges[i] = operands.iter().map(|v| v.node).collect();
                derivative_edges[i].clone_from(&value_edges[i]);
                // Admission follows all original operands, including predicates
                // and unused region arguments. This does not change reachability.
                for v in &operands {
                    local_effects.include(values[v.node].effects);
                }
                match &node.op {
                    Op::Input(j) => {
                        input = Some(*j);
                        sources = Sources::INPUT;
                    }
                    Op::Data(j) => {
                        sources = Sources::DATA;
                        binding_data.push(*j);
                    }
                    Op::Constant(_) => sources = Sources::CONSTANT,
                    Op::Registered(op) => match &**op {
                        registration::Operation::Data(_) => sources = Sources::CONSTANT,
                        registration::Operation::Native(_) => {
                            sources = Sources::OPAQUE;
                            local_effects.opaque = true;
                            local_effects.checked = true;
                            last_native_block = Some(i);
                        }
                    },
                    Op::Cond(c) => {
                        local_effects.checked = true;
                        value_edges[i] = vec![c.lhs.node, c.rhs.node];
                        derivative_edges[i].clear();
                        for branch in [&c.yes, &c.no] {
                            let child = &branch.plan.facts;
                            value_edges[i]
                                .extend(child.output_inputs.iter().map(|&j| c.args[j].node));
                            derivative_edges[i]
                                .extend(child.output_derivatives.iter().map(|&j| c.args[j].node));
                            sources.include(child.output_sources.closed());
                            local_effects.include(child.effects);
                            source_checkpoints |= child.source_checkpoints;
                        }
                    }
                    Op::Scan(s) => {
                        local_effects.checked = true;
                        let child = &s.body.plan.facts;
                        // Preserve the existing conservative higher-source policy
                        // even when the checkpointed body executes zero times.
                        source_checkpoints |= s.checkpointed || child.source_checkpoints;
                        value_edges[i].clear();
                        derivative_edges[i].clear();
                        if s.steps == 0 {
                            value_edges[i].push(s.initial.node);
                            derivative_edges[i].push(s.initial.node);
                        } else {
                            let args: Vec<_> = [s.initial, s.xs]
                                .into_iter()
                                .chain(s.captures.iter().copied())
                                .collect();
                            value_edges[i]
                                .extend(child.output_inputs.iter().map(|&j| args[j].node));
                            derivative_edges[i]
                                .extend(child.output_derivatives.iter().map(|&j| args[j].node));
                            sources.include(child.output_sources.closed());
                            local_effects.include(child.effects);
                        }
                    }
                    _ => local_effects.checked = true,
                }
                for &j in &value_edges[i] {
                    sources.include(values[j].sources);
                }
                effects.include(local_effects);
            }
            let start = derivatives.len();
            derivatives.extend_from_slice(&derivative_edges[i]);
            values.push(ValueFacts {
                input,
                derivatives: start..derivatives.len(),
                sources,
                effects: local_effects,
                admission: active[i],
                binding_fixed: false,
                control: match &node.op {
                    Op::Cond(c) => Control::Branch([c.yes.plan.owner, c.no.plan.owner]),
                    Op::Scan(s) => Control::Scan {
                        child: s.body.plan.owner,
                        steps: s.steps,
                    },
                    _ => Control::None,
                },
            });
        }
        let value_inputs = output_inputs(nodes, &value_edges, output.node);
        let output_derivatives = output_inputs(nodes, &derivative_edges, output.node);
        let output_sources = values[output.node].sources;
        let lowered = lowering::Contract::compile(nodes, active, output, slots, reads)?;
        let mut available = vec![false; nodes.len()];
        let mut last_binding_fixed = None;
        for (i, node) in nodes.iter().enumerate() {
            if !active[i] {
                continue;
            }
            available[i] = match &node.op {
                Op::Data(_) | Op::Constant(_) => true,
                Op::Alias(v) => available[v.node],
                _ => {
                    let fixed = lowered.call(i).is_some_and(|c| c.binding_builtin())
                        && node.inputs().iter().all(|v| available[v.node]);
                    values[i].binding_fixed = fixed;
                    if fixed {
                        last_binding_fixed = Some(i);
                    }
                    fixed
                }
            };
        }

        Ok(Self {
            values: values.into_boxed_slice(),
            binding_data: binding_data.into_boxed_slice(),
            result: output,
            derivatives: derivatives.into_boxed_slice(),
            output_inputs: value_inputs,
            output_derivatives,
            output_sources,
            last_binding_fixed,
            effects,
            missing_source: lowered.missing_source(),
            explicit_source: lowered.requires_explicit_source(),
            source_checkpoints,
            last_native_block,
            lowered,
        })
    }

    pub(super) fn binding_fixed(&self, node: usize) -> bool {
        self.values[node].binding_fixed
    }
    pub(super) fn last_binding_fixed(&self) -> Option<usize> {
        self.last_binding_fixed
    }
    pub(super) fn guard(&self, site: Value, instance: FrameInstance) -> Result<ResidualGuard, E> {
        if site.owner != self.result.owner {
            return Err(E::Shape);
        }
        let value = self.values.get(site.node).ok_or(E::Shape)?;
        if !value.admission {
            return Err(E::Shape);
        }
        let child = match (value.control, instance) {
            (Control::Branch(children), FrameInstance::Branch(choice)) => {
                *children.get(choice).ok_or(E::Shape)?
            }
            (Control::Scan { child, steps }, FrameInstance::Iteration(step)) if step < steps => {
                child
            }
            _ => return Err(E::Shape),
        };
        Ok(ResidualGuard {
            site,
            instance,
            child,
        })
    }

    pub(super) fn admission(&self, node: usize) -> bool {
        self.values[node].admission
    }
    pub(super) fn demands<'a>(&'a self, activity: &'a [bool]) -> Demands<'a> {
        Demands {
            facts: self,
            activity,
        }
    }

    pub(super) fn activity(&self, inputs: &[bool]) -> Vec<bool> {
        let mut activity = vec![false; self.values.len()];
        for (i, value) in self.values.iter().enumerate() {
            activity[i] = value.input.map_or_else(
                || {
                    self.derivatives[value.derivatives.clone()]
                        .iter()
                        .any(|&j| activity[j])
                },
                |j| inputs[j],
            );
        }
        activity
    }

    // Follow derivatives of the actual output. Admission-only dependencies
    // such as a condition's comparison operands stay in the primal schedule.
    pub(super) fn result_derivative_path(&self) -> Vec<bool> {
        let mut path = vec![false; self.values.len()];
        path[self.result.node] = true;
        for (i, value) in self.values.iter().enumerate().rev() {
            if path[i] {
                for &input in &self.derivatives[value.derivatives.clone()] {
                    path[input] = true;
                }
            }
        }
        path
    }

    pub(super) fn blocks_replay(&self) -> bool {
        self.effects.opaque
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent control: the original structured activity interpretation.
    fn interpreted(p: &Plan, inputs: &[bool]) -> Vec<bool> {
        let mut result = vec![false; p.nodes.len()];
        for (i, node) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                continue;
            }
            result[i] = match &node.op {
                Op::Input(j) => inputs[*j],
                Op::Cond(c) => {
                    let args: Vec<_> = c.args.iter().map(|v| result[v.node]).collect();
                    interpreted(&c.yes.plan, &args)[c.yes.plan.output.node]
                        || interpreted(&c.no.plan, &args)[c.no.plan.output.node]
                }
                Op::Scan(s) => {
                    let mut args: Vec<_> = [s.initial, s.xs]
                        .into_iter()
                        .chain(s.captures.iter().copied())
                        .map(|v| result[v.node])
                        .collect();
                    for _ in 0..s.steps {
                        args[0] = interpreted(&s.body.plan, &args)[s.body.plan.output.node];
                    }
                    args[0]
                }
                _ => node.inputs().iter().any(|v| result[v.node]),
            };
        }
        result
    }

    fn scalar(b: &mut RegionBuilder, name: &str) -> Value {
        b.input(name, Shape::scalar()).unwrap()
    }

    #[test]
    fn demand_separates_admission_result_derivatives_and_original_backward_reads() {
        let mut b = RegionBuilder::new();
        let x = scalar(&mut b, "x");
        let data = b.data("data", Shape::scalar()).unwrap();
        let exp = b.unary(UnaryOp::Exp, x).unwrap();
        let alias = b.reshape(exp, Shape::scalar()).unwrap();
        let out = b.binary(BinaryOp::Multiply, alias, data).unwrap();
        let p = b.compile(out).unwrap();
        for enabled in [false, true] {
            let activity = p.plan.facts.activity(&[enabled]);
            let demand = p.plan.facts.demands(&activity);
            assert!(demand.admission(exp.node));
            assert_eq!(demand.result(), out);
            assert_eq!(demand.tangent(alias.node), enabled);
            assert_eq!(demand.cotangent(out.node), enabled);
            let reads: Vec<_> = demand
                .backward_reads()
                .map(|r| (r.reader, r.value))
                .collect();
            assert_eq!(
                reads,
                if enabled {
                    vec![(exp, exp), (out, data)]
                } else {
                    vec![]
                }
            );
        }
    }

    #[test]
    fn nested_capture_transfer_matches_every_mask_and_native_scores() {
        for steps in [0, 1, 3] {
            for use_carry in [false, true] {
                let mut branch = RegionBuilder::new();
                let x = scalar(&mut branch, "x");
                let _unused = scalar(&mut branch, "unused");
                let alias = branch.reshape(x, Shape::scalar()).unwrap();
                let out = branch.binary(BinaryOp::Add, x, alias).unwrap();
                let branch = branch.compile(out).unwrap().bind(&[]).unwrap();

                let mut body = RegionBuilder::new();
                let carry = scalar(&mut body, "carry");
                let row = scalar(&mut body, "row");
                let capture = scalar(&mut body, "capture");
                let zero = body.constant(Shape::scalar(), vec![0.]).unwrap();
                let selected = body
                    .cond(
                        row,
                        zero,
                        Comparison::Less,
                        &[capture, carry],
                        branch.clone(),
                        branch,
                    )
                    .unwrap();
                let out = if use_carry {
                    body.binary(BinaryOp::Add, selected, carry).unwrap()
                } else {
                    selected
                };
                let body = body.compile(out).unwrap().bind(&[]).unwrap();

                let mut b = RegionBuilder::new();
                let initial = scalar(&mut b, "initial");
                let xs = b.input("xs", Shape::new(vec![steps]).unwrap()).unwrap();
                let capture = scalar(&mut b, "capture");
                let out = b.scan(initial, xs, &[capture], body).unwrap();
                let p = b.compile(out).unwrap().bind(&[]).unwrap();
                for bits in 0..8 {
                    let mask = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
                    let expected = interpreted(&p.plan, &mask);
                    assert_eq!(p.activity(&mask), expected);
                    let mut work = p.workspace_with_activity(&mask);
                    assert_eq!(
                        expected[out.node],
                        if steps == 0 {
                            mask[0]
                        } else {
                            mask[2] || (use_carry && mask[0])
                        }
                    );
                    let mut x = vec![1.; steps + 2];
                    x[0] = 0.5;
                    x[steps + 1] = 1.5;
                    let mut result = [0.];
                    let mut scores = vec![0.; x.len()];
                    p.value_and_vjp(&x, &[1.], &mut result, &mut scores, &mut work)
                        .unwrap();
                    assert_eq!(
                        result,
                        [if steps == 0 {
                            0.5
                        } else if use_carry {
                            0.5 + 3. * steps as f64
                        } else {
                            3.
                        }]
                    );
                    assert_eq!(scores[0], f64::from(mask[0] && (steps == 0 || use_carry)));
                    assert!(scores[1..=steps].iter().all(|&v| v == 0.));
                    assert_eq!(
                        scores[steps + 1],
                        if mask[2] && steps > 0 {
                            if use_carry { 2. * steps as f64 } else { 2. }
                        } else {
                            0.
                        }
                    );
                }
                // The predicate affects the value, but has no pathwise derivative.
                assert_eq!(
                    p.plan.facts.output_inputs.as_ref(),
                    if steps == 0 {
                        &[0][..]
                    } else if use_carry {
                        &[0, 1, 2][..]
                    } else {
                        &[1, 2][..]
                    }
                );
            }
        }
    }

    #[test]
    fn provenance_separates_bound_data_constants_and_shared_inputs() {
        let mut b = RegionBuilder::new();
        let x = scalar(&mut b, "x");
        let data = b.data("data", Shape::scalar()).unwrap();
        let constant = b.constant(Shape::scalar(), vec![2.]).unwrap();
        let alias = b.reshape(x, Shape::scalar()).unwrap();
        let shared = b.binary(BinaryOp::Multiply, x, alias).unwrap();
        let sum = b.binary(BinaryOp::Add, data, constant).unwrap();
        let out = b.binary(BinaryOp::Add, shared, sum).unwrap();
        let p = b.compile(out).unwrap();
        let facts = &p.plan.facts;
        assert_eq!(facts.values[data.node].sources, Sources::DATA);
        assert_eq!(facts.values[constant.node].sources, Sources::CONSTANT);
        assert_eq!(facts.values[alias.node].sources, Sources::INPUT);
        assert_eq!(facts.output_sources, Sources(7));
        assert_eq!(facts.output_derivatives.as_ref(), &[0]);
        assert!(!facts.blocks_replay());
        assert!(facts.values[out.node].effects.checked);
        for value in [1., 3.] {
            let bound = p.bind(&[("data", &[value])]).unwrap();
            let mut out = [0.];
            let mut score = [0.];
            bound
                .value_and_vjp(&[4.], &[1.], &mut out, &mut score, &mut bound.workspace())
                .unwrap();
            assert_eq!(out, [18. + value]);
            assert_eq!(score, [8.]);
        }
        // Closed child bindings contribute provenance without acquiring formal
        // derivative inputs when substituted through a conditional.
        let yes = p.bind(&[("data", &[1.])]).unwrap();
        let no = p.bind(&[("data", &[3.])]).unwrap();
        let mut b = RegionBuilder::new();
        let x = scalar(&mut b, "x");
        let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
        let out = b.cond(x, zero, Comparison::Less, &[x], yes, no).unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        assert_eq!(p.plan.facts.output_sources, Sources(7));
        assert!(!p.activity(&[false])[out.node]);
        let mut output = [0.];
        let mut score = [0.];
        p.value_and_vjp(&[4.], &[1.], &mut output, &mut score, &mut p.workspace())
            .unwrap();
        assert_eq!(output, [21.]);
        assert_eq!(score, [8.]);
    }

    #[test]
    fn typed_and_opaque_facts_keep_source_and_replay_barriers() {
        use registration::{BlockRules, BlockType, ElementType};
        let mut b = RegionBuilder::new();
        let counts = b.count_data(Shape::scalar(), vec![u64::MAX]).unwrap();
        let boolean = b.boolean_data(Shape::scalar(), vec![true]).unwrap();
        let x = scalar(&mut b, "x");
        let out = b
            .register_block(
                &[x, counts, boolean],
                (),
                BlockRules {
                    preparation: None,
                    inputs: vec![
                        BlockType {
                            dtype: ElementType::Real,
                            shape: Shape::scalar(),
                        },
                        BlockType {
                            dtype: ElementType::Count,
                            shape: Shape::scalar(),
                        },
                        BlockType {
                            dtype: ElementType::Bool,
                            shape: Shape::scalar(),
                        },
                    ],
                    output: Shape::scalar(),
                    workspace: |_| (),
                    heap_bytes: 0,
                    forward: |_, _, _, _| Err(E::Numerical),
                    vjp: None,
                },
            )
            .unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let facts = &p.plan.facts;
        assert_eq!(facts.values[counts.node].sources, Sources::CONSTANT);
        assert_eq!(facts.values[boolean.node].sources, Sources::CONSTANT);
        assert!(!facts.values[counts.node].effects.opaque);
        assert!(facts.values[out.node].effects.opaque);
        assert_eq!(facts.output_sources, Sources(1 | 4 | 8));
        assert!(facts.blocks_replay());
        assert_eq!(facts.missing_source, Some(counts.node));
        assert!(matches!(p.rust_function(), Err(E::MissingPrimal { node }) if node == counts.node));
        // Inactivity cannot turn an unknown callback into an admitted constant.
        let mut result = [17.];
        assert!(matches!(
            p.evaluate(&[1.], &mut result, &mut p.workspace_with_activity(&[false])),
            Err(E::Numerical)
        ));
        assert_eq!(result, [17.]);
    }
    #[test]
    fn zero_trip_body_is_not_an_execution_barrier_but_keeps_checkpoint_source_policy() {
        use registration::{BlockRules, BlockType, ElementType};
        let mut body = RegionBuilder::new();
        let carry = scalar(&mut body, "carry");
        let _row = scalar(&mut body, "row");
        let out = body
            .register_block(
                &[carry],
                (),
                BlockRules {
                    preparation: None,
                    inputs: vec![BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar(),
                    }],
                    output: Shape::scalar(),
                    workspace: |_| (),
                    heap_bytes: 0,
                    forward: |_, _, _, _| Err(E::Numerical),
                    vjp: None,
                },
            )
            .unwrap();
        let body = body.compile(out).unwrap().bind(&[]).unwrap();
        for steps in [0, 2] {
            let mut b = RegionBuilder::new();
            let initial = scalar(&mut b, "initial");
            let xs = b
                .constant(Shape::new(vec![steps]).unwrap(), vec![0.; steps])
                .unwrap();
            let out = b.scan(initial, xs, &[], body.clone()).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            assert_eq!(p.plan.facts.blocks_replay(), steps != 0);
            assert_eq!(
                p.plan.facts.missing_source,
                (steps != 0).then_some(out.node)
            );
            assert_eq!(p.rust_function().is_ok(), steps == 0);
            if steps == 0 {
                let mut b = RegionBuilder::new();
                let initial = scalar(&mut b, "initial");
                let xs = b.constant(Shape::new(vec![0]).unwrap(), vec![]).unwrap();
                let out = b.checkpointed_scan(initial, xs, &[], body.clone()).unwrap();
                let p = b.compile(out).unwrap().bind(&[]).unwrap();
                assert!(p.has_source_checkpoints());
                assert!(!p.plan.facts.blocks_replay());
                let mut output = [0.];
                let mut gradient = [0.];
                p.value_and_vjp(&[2.], &[1.], &mut output, &mut gradient, &mut p.workspace())
                    .unwrap();
                assert_eq!(output, [2.]);
                assert_eq!(gradient, [1.]);
            }
        }
    }
}
