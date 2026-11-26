//! Private directional program over admitted primals of the canonical region.
use super::*;
mod control;
pub(super) mod curvature;
use control::Control;

#[derive(Clone, Copy)]
struct Slot(usize);

struct InputDirection {
    requested: bool,
    slot: Option<Slot>,
}

/// A typed step of the canonical linear program before assigning storage.
#[derive(Clone, Copy)]
pub(in crate::model::tensor) enum Rule<'a> {
    Input(usize),
    Fixed,
    Alias(Value),
    Unary(UnaryOp, Value),
    Binary(&'a Binary),
    Gather(Value, &'a [usize]),
    Sum(Value),
    Matrix {
        a: Value,
        b: Value,
        calls: &'a lowering::MatrixCalls,
    },
    // These are orchestration boundaries, not inferred linear rules.
    Native(&'a registration::Call),
    Cond(&'a Cond),
    Scan(&'a Scan),
}

impl<'a> Rule<'a> {
    #[inline]
    pub(in crate::model::tensor) fn at(plan: &'a Plan, node: usize) -> Self {
        match &plan.nodes[node].op {
            Op::Input(i) => Self::Input(*i),
            Op::Data(_) | Op::Constant(_) => Self::Fixed,
            Op::Alias(v) => Self::Alias(*v),
            Op::Unary(kind, v) => Self::Unary(*kind, *v),
            Op::Binary(b) => Self::Binary(b),
            Op::Gather(v, indices) => Self::Gather(*v, indices),
            Op::Sum(v) => Self::Sum(*v),
            Op::Matmul { a, b, .. } => Self::Matrix {
                a: *a,
                b: *b,
                calls: plan.facts.lowered.matrix(node),
            },
            Op::Registered(op) => match &**op {
                registration::Operation::Data(_) => Self::Fixed,
                registration::Operation::Native(call) => Self::Native(call),
            },
            Op::Cond(c) => Self::Cond(c),
            Op::Scan(s) => Self::Scan(s),
        }
    }

    fn operands(self) -> [Option<Value>; 2] {
        match self {
            Self::Unary(_, v) | Self::Gather(v, _) | Self::Sum(v) => [Some(v), None],
            Self::Binary(b) => [Some(b.a), Some(b.b)],
            Self::Matrix { a, b, .. } => [Some(a), Some(b)],
            _ => [None, None],
        }
    }
}

/// The existing primal and adjoint arenas borrowed for a transpose step.
pub(in crate::model::tensor) struct Reverse<'a> {
    pub(in crate::model::tensor) plan: &'a Plan,
    pub(in crate::model::tensor) primals: &'a [f64],
    pub(in crate::model::tensor) adjoints: &'a mut [f64],
    pub(in crate::model::tensor) active: &'a [bool],
    pub(in crate::model::tensor) starts: &'a [usize],
}

impl<'a> Reverse<'a> {
    #[inline]
    fn pullback<const PLANNED: bool>(
        self,
        node: usize,
        a: Value,
        b: Value,
        range: Range<usize>,
        copied_seed: Option<&'a [f64]>,
    ) -> kernels::Pullback<'a> {
        let Self {
            plan,
            primals,
            adjoints,
            active,
            starts,
        } = self;
        let reads = plan
            .facts
            .lowered
            .call(node)
            .expect("checked transpose")
            .native_primals(primals)
            .inputs;
        let base = |i| adjoint_start::<PLANNED>(plan, starts, i);
        let (adjoints, seeds): (&mut [f64], &[f64]) = if let Some(seed) = copied_seed {
            (adjoints, &seed[range])
        } else {
            let (before, after) = adjoints.split_at_mut(base(node));
            (before, &after[range])
        };
        kernels::Pullback {
            a: reads.get(0).expect("checked left residual"),
            r: reads.get(1).expect("checked right residual"),
            seeds,
            adjoints,
            starts: (base(a.node), base(b.node)),
            active: (active[a.node], active[b.node]),
        }
    }
}

impl Rule<'_> {
    /// Interpret a transposed builtin at the same schedule position. Tiling
    /// is permitted only by the existing independent-destination proof.
    #[inline]
    pub(in crate::model::tensor) fn transpose<const PLANNED: bool, const TILED: bool>(
        self,
        node: usize,
        work: Reverse<'_>,
        range: Range<usize>,
        copied_seed: Option<&[f64]>,
        write: lowering::AdjointWrite,
    ) {
        let plan = work.plan;
        let base = |i| adjoint_start::<PLANNED>(plan, work.starts, i);
        match self {
            Rule::Fixed | Rule::Alias(_) => {}
            Rule::Unary(kind, a) => {
                let residuals = plan
                    .facts
                    .lowered
                    .call(node)
                    .expect("checked unary transpose")
                    .native_primals(work.primals);
                let input = residuals.inputs.get(0).expect("checked unary input");
                let output = residuals.output.expect("checked unary output");
                kernels::unary_reverse_range(
                    kind,
                    &input[range.clone()],
                    &output[range.clone()],
                    work.adjoints,
                    (base(a.node) + range.start, base(node) + range.start),
                    write,
                );
            }
            Rule::Binary(b) => {
                let start = range.start;
                let pullback = work.pullback::<PLANNED>(node, b.a, b.b, range, copied_seed);
                if TILED {
                    kernels::reverse_range(b, pullback, start);
                } else {
                    kernels::reverse(b, pullback);
                }
            }
            Rule::Sum(a) => {
                let seed = work.adjoints[base(node)];
                let range = if TILED {
                    range
                } else {
                    0..plan.nodes[a.node].shape.elements
                };
                let start = base(a.node);
                let output = &mut work.adjoints[start + range.start..start + range.end];
                match write {
                    lowering::AdjointWrite::Replace => output.fill(0. + seed),
                    lowering::AdjointWrite::Add => {
                        for g in output {
                            *g += seed;
                        }
                    }
                }
            }
            Rule::Gather(a, indices) => {
                let start = base(a.node);
                let seed = base(node);
                for (j, &index) in indices.iter().enumerate() {
                    work.adjoints[start + index] += work.adjoints[seed + j];
                }
            }
            Rule::Matrix { a, b, calls } => {
                if !range.is_empty() {
                    kernels::matmul_reverse(
                        work.pullback::<PLANNED>(node, a, b, range, copied_seed),
                        calls,
                    );
                }
            }
            _ => unreachable!("only builtin rules have a transpose"),
        }
    }
}

enum DirectionRanges {
    Builtin([Option<Range<usize>>; 2]),
    Native(Box<[Option<Range<usize>>]>),
}
impl DirectionRanges {
    fn as_slice(&self) -> &[Option<Range<usize>>] {
        match self {
            Self::Builtin(ranges) => ranges,
            Self::Native(ranges) => ranges,
        }
    }
}

struct Step<'a> {
    node: usize,
    primal: Range<usize>,
    tangent: Slot,
    inputs: DirectionRanges,
    rule: Rule<'a>,
    control: Option<Box<Control<'a>>>,
}

pub(super) struct Program<'a> {
    plan: &'a Plan,
    inputs: Box<[InputDirection]>,
    steps: Box<[Step<'a>]>,
    output: Option<Slot>,
    slots: usize,
    child_slots: usize,
    node_slots: Box<[Option<Slot>]>,
}

#[derive(Debug, thiserror::Error)]
#[error("tensor node {node} has no scoped native JVP rule")]
struct MissingJvp {
    node: usize,
}

impl<'a> Program<'a> {
    pub(super) fn compile(plan: &'a Plan, directions: &[bool]) -> Result<Self, E> {
        Self::compile_mode(plan, directions, false)
    }

    fn compile_mode(plan: &'a Plan, directions: &[bool], curvature: bool) -> Result<Self, E> {
        if directions.len() != plan.inputs.len() {
            return Err(E::Shape);
        }
        let activity = plan.facts.activity(directions);
        let demand = plan.facts.demands(&activity);
        let curvature_path = curvature.then(|| plan.facts.result_derivative_path());
        let mut inputs: Vec<_> = directions
            .iter()
            .map(|&requested| InputDirection {
                requested,
                slot: None,
            })
            .collect();
        let mut tangents = vec![None; plan.nodes.len()];
        let mut steps = Vec::new();
        let mut slots = 0_usize;
        let mut child_slots = 0;
        for (i, node) in plan.nodes.iter().enumerate() {
            if !demand.admission(i) {
                continue;
            }
            let rule = Rule::at(plan, i);
            if matches!(rule, Rule::Scan(s) if s.bounded_replay) {
                return Err(E::ScanReplayUnavailable);
            }
            let supported = match rule {
                Rule::Native(call) if curvature && !call.supports_curvature() => {
                    return Err(E::Factor(Box::new(curvature::MissingSecondRule {
                        node: i,
                    })));
                }
                Rule::Native(call) => call.supports_jvp(),
                Rule::Fixed | Rule::Cond(_) | Rule::Scan(_) => true,
                _ => plan
                    .facts
                    .lowered
                    .call(i)
                    .is_some_and(|call| call.linear_builtin()),
            };
            if !supported {
                return Err(E::Factor(Box::new(MissingJvp { node: i })));
            }
            let input = match rule {
                Rule::Fixed => continue,
                Rule::Alias(v) => {
                    tangents[i] = tangents[v.node];
                    continue;
                }
                Rule::Input(input) => Some(input),
                _ => None,
            };
            let range = |v: Value| {
                let Slot(start) = tangents[v.node]?;
                Some(start..start + plan.nodes[v.node].shape.elements)
            };
            let ranges = match rule {
                Rule::Native(call) => {
                    DirectionRanges::Native(call.arguments().iter().map(|&v| range(v)).collect())
                }
                Rule::Cond(c) => {
                    DirectionRanges::Native(c.args.iter().map(|&v| range(v)).collect())
                }
                Rule::Scan(s) if s.steps != 0 => DirectionRanges::Native(
                    [s.initial, s.xs]
                        .into_iter()
                        .chain(s.captures.iter().copied())
                        .map(range)
                        .collect(),
                ),
                Rule::Scan(s) => DirectionRanges::Builtin([range(s.initial), None]),
                _ => DirectionRanges::Builtin(rule.operands().map(|v| v.and_then(range))),
            };
            let control = Control::compile(rule, ranges.as_slice(), curvature)?;
            if node.shape.elements == 0
                || !demand.tangent(i)
                || curvature_path.as_ref().is_some_and(|path| !path[i])
                || (input.is_none() && ranges.as_slice().iter().all(Option::is_none))
            {
                continue;
            }
            let slot = Slot(slots);
            slots = slots.checked_add(node.shape.elements).ok_or(E::Shape)?;
            tangents[i] = Some(slot);
            if let Some(input) = input {
                inputs[input].slot = Some(slot);
            } else {
                if let Some(control) = &control {
                    child_slots = child_slots.max(control.scratch_elements());
                }
                steps.push(Step {
                    node: i,
                    primal: node.primal_range(),
                    tangent: slot,
                    inputs: ranges,
                    rule,
                    control: control.map(Box::new),
                });
            }
        }
        slots
            .checked_add(child_slots)
            .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
            .ok_or(E::Shape)?;
        Ok(Self {
            plan,
            inputs: inputs.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            output: tangents[demand.result().node],
            slots,
            child_slots,
            node_slots: if curvature {
                tangents.into_boxed_slice()
            } else {
                Box::default()
            },
        })
    }

    pub(super) fn tangent_elements(&self) -> usize {
        self.slots + self.child_slots
    }

    /// Runs the existing complete primal admission before interpreting any
    /// coefficients. The optional input slices must match the compiled mask.
    /// Both caller buffers are published together after all numerical checks.
    pub(super) fn value_and_jvp(
        &self,
        binding: &BoundProgram,
        x: &[f64],
        directions: &[Option<&[f64]>],
        outputs: (&mut [f64], &mut [f64]),
        work: &mut TensorWorkspace,
        tangents: &mut [f64],
    ) -> Result<(), E> {
        let (out, jvp) = outputs;
        work.prepared = false;
        let result = self.plan.nodes[self.plan.output.node].primal_range();
        if !std::ptr::eq(self.plan, binding.plan.as_ref())
            || out.len() != result.len()
            || jvp.len() != result.len()
            || directions.len() != self.inputs.len()
            || tangents.len() != self.tangent_elements()
        {
            return Err(E::Shape);
        }
        binding.validate(x, work)?;
        for ((input, direction), formal) in
            self.inputs.iter().zip(directions).zip(&self.plan.inputs)
        {
            if input.requested != direction.is_some() {
                return Err(E::Shape);
            }
            if let Some(direction) = direction {
                if direction.len() != formal.shape.elements {
                    return Err(E::Shape);
                }
                if direction.iter().any(|v| !v.is_finite()) {
                    return Err(E::Numerical);
                }
                if let Some(Slot(start)) = input.slot {
                    tangents[start..start + direction.len()].copy_from_slice(direction);
                }
            }
        }
        binding.execute_forward(x, work, None)?;
        self.run_prepared(binding, work, tangents)?;
        out.copy_from_slice(&work.values[result]);
        self.copy_result(jvp, tangents);
        Ok(())
    }

    fn run_prepared(
        &self,
        binding: &BoundProgram,
        work: &mut TensorWorkspace,
        tangents: &mut [f64],
    ) -> Result<(), E> {
        let admitted = std::mem::replace(&mut work.prepared, false);
        binding.check_residual(work, admitted)?;
        if !std::ptr::eq(self.plan, binding.plan.as_ref()) {
            return Err(E::Shape);
        }
        let (local, scratch) = tangents.split_at_mut(self.slots);
        for step in &self.steps {
            step.apply(self.plan, work, local, scratch, None)?;
        }
        Ok(())
    }

    fn load_direction(&self, i: usize, direction: Option<&[f64]>, tangents: &mut [f64]) {
        if let Some(Slot(start)) = self.inputs[i].slot {
            let len = self.plan.inputs[i].shape.elements;
            let output = &mut tangents[start..start + len];
            if let Some(direction) = direction {
                output.copy_from_slice(direction);
            } else {
                output.fill(0.);
            }
        }
    }

    fn copy_result(&self, output: &mut [f64], tangents: &[f64]) {
        if let Some(Slot(start)) = self.output {
            output.copy_from_slice(&tangents[start..start + output.len()]);
        } else {
            output.fill(0.);
        }
    }
}

impl Step<'_> {
    fn apply(
        &self,
        plan: &Plan,
        work: &mut TensorWorkspace,
        tangents: &mut [f64],
        scratch: &mut [f64],
        curvature: Option<&mut curvature::Region>,
    ) -> Result<(), E> {
        let primals = &work.values;
        let (inputs, after) = tangents.split_at_mut(self.tangent.0);
        let output = &mut after[..self.primal.len()];
        let tangent = |i: usize| {
            self.inputs.as_slice()[i]
                .as_ref()
                .map(|r| &inputs[r.clone()])
        };
        let primal = |v: Value| &primals[plan.nodes[v.node].primal_range()];
        match self.rule {
            Rule::Unary(kind, input) => {
                let x = primal(input);
                let y = &primals[self.primal.clone()];
                let direction = tangent(0).expect("active unary");
                for (i, out) in output.iter_mut().enumerate() {
                    *out = scaled(direction[i], || derivative(kind, x[i], y[i]));
                }
            }
            Rule::Binary(op) => {
                let (a, b) = (primal(op.a), primal(op.b));
                let (da, db) = (tangent(0), tangent(1));
                kernels::for_each_index(op, 0, output.len(), |j, l, r| {
                    let left = da.map_or(0., |t| {
                        scaled(t[l], || match op.kind {
                            BinaryOp::Add | BinaryOp::Subtract => 1.,
                            BinaryOp::Multiply => b[r],
                            BinaryOp::Divide => kernels::divide_left(b[r]),
                        })
                    });
                    let right = db.map_or(0., |t| {
                        scaled(t[r], || match op.kind {
                            BinaryOp::Add => 1.,
                            BinaryOp::Subtract => -1.,
                            BinaryOp::Multiply => a[l],
                            BinaryOp::Divide => kernels::divide_right(a[l], b[r]),
                        })
                    });
                    output[j] = left + right;
                });
            }
            Rule::Gather(_, indices) => {
                kernels::gather_forward(tangent(0).expect("active gather"), indices, output);
            }
            Rule::Sum(_) => output[0] = kernels::sum_forward(tangent(0).expect("active sum")),
            Rule::Matrix { a, b, calls } => {
                kernels::matmul_jvp(
                    (primal(a), primal(b)),
                    (tangent(0), tangent(1)),
                    output,
                    calls,
                )?;
            }
            Rule::Native(call) => {
                call.directional(
                    (&plan.nodes, primals),
                    (self.inputs.as_slice(), inputs),
                    &primals[self.primal.clone()],
                    output,
                    work.blocks[self.node]
                        .as_ref()
                        .expect("registered workspace"),
                )?;
            }
            Rule::Cond(c) => {
                let region = work.regions[self.node].as_mut().expect("conditional frame");
                self.control.as_ref().expect("conditional program").cond(
                    plan,
                    c,
                    region,
                    tangent,
                    (output, scratch),
                    curvature,
                )?;
            }
            Rule::Scan(s) => {
                if let Some(initial) = tangent(0) {
                    output.copy_from_slice(initial);
                } else {
                    output.fill(0.);
                }
                if let Some(control) = &self.control {
                    let region = work.regions[self.node].as_mut().expect("scan frame");
                    control.scan(
                        plan,
                        s,
                        (primals, region),
                        tangent,
                        (output, scratch),
                        curvature,
                    )?;
                }
            }
            _ => unreachable!("only supported operations get a tangent step"),
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(())
    }
}

fn scaled(tangent: f64, coefficient: impl FnOnce() -> f64) -> f64 {
    // As in the existing reverse kernels, a zero seed does not evaluate a
    // singular local coefficient. Primal admission has already run.
    if tangent == 0. {
        0.
    } else {
        tangent * coefficient()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() <= 2e-14 * (1. + b.abs()), "{a} != {b}");
    }

    #[test]
    fn unary_rules_match_independent_directional_answers() {
        use UnaryOp::*;
        let x = 0.6_f64;
        let z = x.exp();
        let cases = [
            (Neg, -1.),
            (Exp, z),
            (Log, 1. / x),
            (Log1p, 1. / (1. + x)),
            (Expm1, z),
            (Tanh, 1. / x.cosh().powi(2)),
            (Sin, x.cos()),
            (Cos, -x.sin()),
            (Sqrt, 1. / (2. * x.sqrt())),
        ];
        for (kind, slope) in cases {
            let mut b = RegionBuilder::new();
            let arg = b.input("x", Shape::scalar()).unwrap();
            let y = b.unary(kind, arg).unwrap();
            let p = b.compile(y).unwrap().bind(&[]).unwrap();
            let program = Program::compile(&p.plan, &[true]).unwrap();
            let mut work = p.workspace();
            let (mut out, mut jvp, mut reference) = ([f64::NAN], [f64::NAN], [0.]);
            let mut scratch = vec![f64::NAN; program.tangent_elements()];
            p.evaluate(&[x], &mut reference, &mut work).unwrap();
            for direction in [-2., 0., 3.5] {
                program
                    .value_and_jvp(
                        &p,
                        &[x],
                        &[Some(&[direction])],
                        (&mut out, &mut jvp),
                        &mut work,
                        &mut scratch,
                    )
                    .unwrap();
                assert_eq!(out.map(f64::to_bits), reference.map(f64::to_bits));
                near(jvp[0], direction * slope);
            }
        }
    }

    #[test]
    fn shared_edges_and_reshape_reuse_tangents_across_bindings() {
        let mut b = RegionBuilder::new();
        let shape = Shape::new(vec![2]).unwrap();
        let x = b.input("x", shape.clone()).unwrap();
        let y = b.input("y", shape.clone()).unwrap();
        let data = b.data("shift", shape.clone()).unwrap();
        let a = b.binary(BinaryOp::Add, x, data).unwrap();
        let alias = b.reshape(a, Shape::new(vec![1, 2]).unwrap()).unwrap();
        let a = b.reshape(alias, shape).unwrap();
        let square = b.binary(BinaryOp::Multiply, a, a).unwrap();
        let numerator = b.binary(BinaryOp::Subtract, square, y).unwrap();
        let result = b.binary(BinaryOp::Divide, numerator, y).unwrap();
        let p = b
            .compile(result)
            .unwrap()
            .bind(&[("shift", &[1., 2.])])
            .unwrap();
        let rebound = p.rebind("shift", &[3., 4.]).unwrap();

        for mask in [[true, true], [true, false], [false, true], [false, false]] {
            let program = Program::compile(&p.plan, &mask).unwrap();
            let expected_nodes = match mask {
                [true, true] => 6,
                [true, false] => 5,
                [false, true] => 3,
                [false, false] => 0,
            };
            assert_eq!(program.tangent_elements(), 2 * expected_nodes);
            let mut work = p.workspace();
            let mut scratch = vec![f64::NAN; program.tangent_elements()];
            let (mut out, mut jvp) = ([f64::NAN; 2], [f64::NAN; 2]);
            let (dx, dy) = ([0.25, -2.], [0.75, 0.2]);
            let directions = [
                mask[0].then_some(dx.as_slice()),
                mask[1].then_some(dy.as_slice()),
            ];
            let allocations = allocation_counter::measure(|| {
                for (binding, shift) in [(&p, [1., 2.]), (&rebound, [3., 4.]), (&p, [1., 2.])] {
                    program
                        .value_and_jvp(
                            binding,
                            &[2., 4., 3., 5.],
                            &directions,
                            (&mut out, &mut jvp),
                            &mut work,
                            &mut scratch,
                        )
                        .unwrap();
                    for (i, (x, y)) in [(2., 3.), (4., 5.)].into_iter().enumerate() {
                        let a = x + shift[i];
                        let dx = if mask[0] { dx[i] } else { 0. };
                        let dy = if mask[1] { dy[i] } else { 0. };
                        near(out[i], a * a / y - 1.);
                        near(jvp[i], 2. * a * dx / y - a * a * dy / (y * y));
                    }
                }
            });
            assert_eq!(allocations.count_total, 0);
        }
    }

    #[test]
    fn zero_directions_preserve_primal_admission_and_errors_are_transactional() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let y = b.unary(UnaryOp::Sqrt, x).unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let zero = Program::compile(&p.plan, &[false]).unwrap();
        let active = Program::compile(&p.plan, &[true]).unwrap();
        let mut scratch = [f64::NAN; 2];
        let mut work = p.workspace();
        let (mut out, mut jvp) = ([17.], [19.]);
        for (program, input, direction, scratch) in [
            (&zero, -1., None, &mut [][..]),
            (&active, -1., Some(&[0.][..]), &mut scratch[..]),
        ] {
            assert!(matches!(
                program.value_and_jvp(
                    &p,
                    &[input],
                    &[direction],
                    (&mut out, &mut jvp),
                    &mut work,
                    scratch
                ),
                Err(E::Numerical)
            ));
            assert_eq!((out, jvp), ([17.], [19.]));
        }

        for direction in [1., f64::NAN] {
            assert!(matches!(
                active.value_and_jvp(
                    &p,
                    &[0.],
                    &[Some(&[direction])],
                    (&mut out, &mut jvp),
                    &mut work,
                    &mut scratch
                ),
                Err(E::Numerical)
            ));
            assert_eq!((out, jvp), ([17.], [19.]));
            assert!(!work.prepared);
        }
        assert!(matches!(
            active.value_and_jvp(
                &p,
                &[1.],
                &[None],
                (&mut out, &mut jvp),
                &mut work,
                &mut scratch
            ),
            Err(E::Shape)
        ));
        assert_eq!((out, jvp), ([17.], [19.]));
        for (input, direction, expected) in [(0., 0., 0.), (4., 2., 0.5)] {
            active
                .value_and_jvp(
                    &p,
                    &[input],
                    &[Some(&[direction])],
                    (&mut out, &mut jvp),
                    &mut work,
                    &mut scratch,
                )
                .unwrap();
            assert_eq!((out, jvp), ([input.sqrt()], [expected]));
        }
        zero.value_and_jvp(&p, &[0.], &[None], (&mut out, &mut jvp), &mut work, &mut [])
            .unwrap();
        assert_eq!((out, jvp), ([0.], [0.]));
    }

    #[test]
    fn empty_result_does_not_skip_admission_of_its_scalar_operand() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let y = b.unary(UnaryOp::Log, x).unwrap();
        let empty = b
            .constant(Shape::new(vec![0, usize::MAX]).unwrap(), vec![])
            .unwrap();
        let output = b.binary(BinaryOp::Multiply, y, empty).unwrap();
        let p = b.compile(output).unwrap().bind(&[]).unwrap();
        let program = Program::compile(&p.plan, &[false]).unwrap();
        assert_eq!(program.tangent_elements(), 0);
        let mut work = p.workspace();
        assert!(matches!(
            program.value_and_jvp(&p, &[-1.], &[None], (&mut [], &mut []), &mut work, &mut []),
            Err(E::Numerical)
        ));
        program
            .value_and_jvp(&p, &[2.], &[None], (&mut [], &mut []), &mut work, &mut [])
            .unwrap();
    }

    #[test]
    fn private_capability_declines_registered_vjp_calls() {
        let shape = Shape::new(vec![1, 1]).unwrap();
        let mut b = RegionBuilder::new();
        let x = b.input("x", shape.clone()).unwrap();
        let block = b
            .register_block(
                &[x],
                (),
                BlockRules {
                    inputs: vec![BlockType {
                        dtype: ElementType::Real,
                        shape: shape.clone(),
                    }],
                    output: shape,
                    workspace: |_| panic!("JVP resolution must not allocate a callback frame"),
                    heap_bytes: 0,
                    preparation: None,
                    forward: |_, _, _, _: &mut ()| unreachable!(),
                    vjp: Some(|_, _, _, seed, scores, _| {
                        scores.copy_from_slice(seed);
                        Ok(())
                    }),
                },
            )
            .unwrap();
        let p = b.compile(block).unwrap().bind(&[]).unwrap();
        assert!(matches!(
            p.jvp(&[false]),
            Err(E::Factor(error)) if error.downcast_ref::<MissingJvp>().is_some_and(|e| e.node == block.node)
        ));
    }

    #[test]
    fn broadcast_and_repeated_gather_reuse_checked_layout() {
        let mut b = RegionBuilder::new();
        let a = b.input("a", Shape::new(vec![2, 1]).unwrap()).unwrap();
        let c = b.input("b", Shape::new(vec![1, 3]).unwrap()).unwrap();
        let product = b.binary(BinaryOp::Multiply, a, c).unwrap();
        let rows = b
            .gather(product, Shape::new(vec![4]).unwrap(), vec![5, 0, 5, 3])
            .unwrap();
        let p = b.compile(rows).unwrap().bind(&[]).unwrap();
        let program = Program::compile(&p.plan, &[true, true]).unwrap();
        let (mut out, mut jvp) = ([0.; 4], [0.; 4]);
        program
            .value_and_jvp(
                &p,
                &[2., 3., 5., 7., 11.],
                &[Some(&[1., 2.]), Some(&[3., 4., 5.])],
                (&mut out, &mut jvp),
                &mut p.workspace(),
                &mut vec![f64::NAN; program.tangent_elements()],
            )
            .unwrap();
        assert_eq!(out, [33., 10., 33., 15.]);
        assert_eq!(jvp, [37., 11., 37., 19.]);
    }

    #[test]
    fn ordered_sum_keeps_cancellation_and_empty_reductions_have_no_tangent() {
        for (width, direction, expected) in [(3, &[1e16, 1., -1e16][..], 1.), (0, &[][..], 0.)] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![width]).unwrap()).unwrap();
            let y = b.sum(x).unwrap();
            let p = b.compile(y).unwrap().bind(&[]).unwrap();
            let program = Program::compile(&p.plan, &[true]).unwrap();
            if width == 0 {
                assert_eq!(program.tangent_elements(), 0);
            }
            let (mut out, mut jvp) = ([f64::NAN], [f64::NAN]);
            program
                .value_and_jvp(
                    &p,
                    &vec![0.; width],
                    &[Some(direction)],
                    (&mut out, &mut jvp),
                    &mut p.workspace(),
                    &mut vec![f64::NAN; program.tangent_elements()],
                )
                .unwrap();
            assert_eq!((out, jvp), ([0.], [expected]));
        }
    }

    #[test]
    fn rectangular_matrix_directions_preserve_the_resolved_provider() {
        let mut requests = vec![MatrixRequest::default()];
        for (provider, enabled) in [
            (MatrixProvider::Faer, cfg!(feature = "faer")),
            (MatrixProvider::Blas, cfg!(feature = "openblas")),
        ] {
            let request = MatrixRequest::standard(provider);
            match request.resolve() {
                Ok(resolved) if enabled => {
                    assert_eq!(resolved.provider(), provider);
                    requests.push(request);
                }
                Err(MatrixPolicyError::Unavailable(rejected)) if !enabled => {
                    assert_eq!(rejected, provider);
                    let mut b = RegionBuilder::new();
                    let x = b.input("x", Shape::new(vec![1, 1]).unwrap()).unwrap();
                    assert!(matches!(
                        b.matmul_with_request(x, x, request),
                        Err(E::MatrixPolicy(MatrixPolicyError::Unavailable(p))) if p == provider
                    ));
                }
                other => panic!("unexpected matrix resolution: {other:?}"),
            }
        }
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![1, 1]).unwrap()).unwrap();
        let incompatible = MatrixRequest {
            cpu: MatrixCpu::Scalar,
            ..MatrixRequest::standard(MatrixProvider::Faer)
        };
        assert!(matches!(
            b.matmul_with_request(x, x, incompatible),
            Err(E::MatrixPolicy(MatrixPolicyError::Incompatible(_)))
        ));

        for request in requests {
            let mut b = RegionBuilder::new();
            let a = b.input("a", Shape::new(vec![2, 3]).unwrap()).unwrap();
            let c = b.input("b", Shape::new(vec![3, 2]).unwrap()).unwrap();
            let y = b.matmul_with_request(a, c, request).unwrap();
            let p = b.compile(y).unwrap().bind(&[]).unwrap();
            let (da, db) = ([1., 0., 0., 0., 0., 2.], [0., 1., 0., 0., 3., 0.]);
            for (mask, expected) in [
                ([true, true], [16., 9., 40., 28.]),
                ([true, false], [7., 8., 22., 24.]),
                ([false, true], [9., 1., 18., 4.]),
                ([false, false], [0.; 4]),
            ] {
                let program = Program::compile(&p.plan, &mask).unwrap();
                if let Some(step) = program.steps.first() {
                    assert!(
                        matches!(&step.rule, Rule::Matrix { calls, .. } if calls.kernel.requested() == request)
                    );
                }
                let (mut out, mut jvp) = ([17.; 4], [19.; 4]);
                let x = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
                let mut work = p.workspace();
                let mut scratch = vec![f64::NAN; program.tangent_elements()];
                if mask == [true, true] {
                    assert!(matches!(
                        program.value_and_jvp(
                            &p,
                            &x,
                            &[Some(&da), Some(&[f64::MAX; 6])],
                            (&mut out, &mut jvp),
                            &mut work,
                            &mut scratch,
                        ),
                        Err(E::Numerical)
                    ));
                    assert_eq!((out, jvp), ([17.; 4], [19.; 4]));
                }
                program
                    .value_and_jvp(
                        &p,
                        &x,
                        &[
                            mask[0].then_some(da.as_slice()),
                            mask[1].then_some(db.as_slice()),
                        ],
                        (&mut out, &mut jvp),
                        &mut work,
                        &mut scratch,
                    )
                    .unwrap();
                assert_eq!(out, [58., 64., 139., 154.]);
                assert_eq!(jvp, expected);
            }
        }
    }

    #[test]
    fn matrix_compensation_spans_both_terms_and_zero_contractions_are_free() {
        let mut b = RegionBuilder::new();
        let a = b.input("a", Shape::new(vec![1, 2]).unwrap()).unwrap();
        let c = b.input("b", Shape::new(vec![2, 1]).unwrap()).unwrap();
        let y = b.matmul(a, c).unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let program = Program::compile(&p.plan, &[true, true]).unwrap();
        let (mut out, mut jvp) = ([0.], [0.]);
        program
            .value_and_jvp(
                &p,
                &[1., 1e16, 1., 1.],
                &[Some(&[1e16, 0.]), Some(&[1., -1.])],
                (&mut out, &mut jvp),
                &mut p.workspace(),
                &mut vec![f64::NAN; program.tangent_elements()],
            )
            .unwrap();
        assert_eq!((out, jvp), ([1e16], [1.]));

        for (rows, cols) in [(2, 3), (usize::MAX, 0)] {
            let mut b = RegionBuilder::new();
            let a = b.input("a", Shape::new(vec![rows, 0]).unwrap()).unwrap();
            let c = b.input("b", Shape::new(vec![0, cols]).unwrap()).unwrap();
            let y = b.matmul(a, c).unwrap();
            let p = b.compile(y).unwrap().bind(&[]).unwrap();
            let program = Program::compile(&p.plan, &[true, true]).unwrap();
            assert_eq!(program.tangent_elements(), 0);
            let (mut out, mut jvp) = (vec![f64::NAN; rows * cols], vec![f64::NAN; rows * cols]);
            program
                .value_and_jvp(
                    &p,
                    &[],
                    &[Some(&[]), Some(&[])],
                    (&mut out, &mut jvp),
                    &mut p.workspace(),
                    &mut [],
                )
                .unwrap();
            assert!(out.iter().chain(&jvp).all(|&value| value == 0.));
        }
    }

    #[test]
    fn native_transpose_and_jvp_match_independent_shared_matrix_and_map_answers() {
        let mut b = RegionBuilder::new();
        let a = b.input("a", Shape::new(vec![2, 2]).unwrap()).unwrap();
        let x = b.input("x", Shape::new(vec![2, 1]).unwrap()).unwrap();
        let bias = b.input("bias", Shape::scalar()).unwrap();
        let product = b.matmul(a, x).unwrap();
        let alias = b.reshape(product, Shape::new(vec![2]).unwrap()).unwrap();
        let squares = b.binary(BinaryOp::Multiply, alias, alias).unwrap();
        let rows = b
            .gather(squares, Shape::new(vec![3]).unwrap(), vec![1, 0, 1])
            .unwrap();
        let y = b.binary(BinaryOp::Add, rows, bias).unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let plan = Program::compile(&p.plan, &[true, true, true]).unwrap();
        let mut work = p.workspace();
        let mut tangent = vec![f64::NAN; plan.tangent_elements()];
        let (mut out, mut jvp, mut vjp) = ([f64::NAN; 3], [f64::NAN; 3], [f64::NAN; 7]);
        let point = [1., 2., 3., 4., 2., -1., 7.];
        let direction = [1., 0., -1., 2., 0.5, -1., -2.];
        let seed = [2., -1., 3.];
        plan.value_and_jvp(
            &p,
            &point,
            &[
                Some(&direction[..4]),
                Some(&direction[4..6]),
                Some(&direction[6..]),
            ],
            (&mut out, &mut jvp),
            &mut work,
            &mut tangent,
        )
        .unwrap();
        // A x = [0, 2], d(A x) = [1/2, -13/2]. The repeated second
        // row receives cotangent 2 + 3 before the square's transpose.
        assert_eq!(out, [11., 7., 11.]);
        assert_eq!(jvp, [-28., -2., -28.]);
        p.value_and_vjp(&point, &seed, &mut out, &mut vjp, &mut work)
            .unwrap();
        assert_eq!(out, [11., 7., 11.]);
        assert_eq!(vjp, [0., 0., 40., -20., 60., 80., 4.]);
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        assert_eq!(dot(&jvp, &seed), -138.);
        assert_eq!(dot(&vjp, &direction), -138.);
    }

    #[test]
    fn explicit_jvp_reads_one_forward_and_never_supplies_a_vjp() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let count = b.count_data(Shape::scalar(), vec![2]).unwrap();
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let y = b
            .register_block_with_jvp(
                &[x, count],
                Arc::clone(&runs),
                BlockRules {
                    inputs: vec![
                        BlockType {
                            dtype: ElementType::Real,
                            shape: Shape::scalar(),
                        },
                        BlockType {
                            dtype: ElementType::Count,
                            shape: Shape::scalar(),
                        },
                    ],
                    output: Shape::scalar(),
                    workspace: |_| 0_f64,
                    heap_bytes: 0,
                    forward: |count, inputs, out, saved| {
                        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        assert_eq!(inputs.counts(1)?, &[2]);
                        *saved = inputs.real(0)?[0];
                        out[0] = 2. * *saved;
                        Ok(())
                    },
                    preparation: None,
                    vjp: None,
                },
                |_, _, tangent, _, out, saved| {
                    assert!(matches!(tangent.real(1), Err(E::Shape)));
                    if *saved != 1. {
                        out[0] = 2. * tangent.real(0)?.unwrap()[0];
                    }
                    Ok(())
                },
            )
            .unwrap();
        let bound = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut jvp = bound.jvp(&[true]).unwrap();
        let (mut out, mut tangent) = ([17.], [19.]);
        jvp.value_and_jvp(&[2.], &[Some(&[3.])], &mut out, &mut tangent)
            .unwrap();
        assert_eq!((out, tangent), ([4.], [6.]));
        assert!(matches!(
            jvp.value_and_jvp(&[1.], &[Some(&[3.])], &mut out, &mut tangent),
            Err(E::Numerical)
        ));
        assert_eq!((out, tangent), ([4.], [6.]));
        jvp.value_and_jvp(&[3.], &[Some(&[-1.])], &mut out, &mut tangent)
            .unwrap();
        assert_eq!((out, tangent), ([6.], [-2.]));
        assert_eq!(runs.load(std::sync::atomic::Ordering::Relaxed), 3);
        let mut gradient = [23.];
        assert!(matches!(
            bound.value_and_vjp(
                &[4.],
                &[1.],
                &mut out,
                &mut gradient,
                &mut bound.workspace()
            ),
            Err(E::Factor(_))
        ));
        assert_eq!((out, gradient), ([6.], [23.]));
    }
}
