//! Forward tangents and differentiated adjoints for the existing reverse sweep.
use super::*;

#[derive(Debug, thiserror::Error)]
#[error("tensor node {node} has no native second derivative rule")]
pub(in crate::model::tensor::execute) struct MissingSecondRule {
    pub(super) node: usize,
}

pub(in crate::model::tensor::execute) struct Frame {
    pub(in crate::model::tensor::execute) tangents: Vec<f64>,
    pub(in crate::model::tensor::execute) adjoints: Vec<f64>,
    pub(in crate::model::tensor::execute) gradient: Vec<f64>,
    pub(in crate::model::tensor::execute) regions: Vec<Option<Box<Region>>>,
}

pub(in crate::model::tensor::execute) struct Region {
    pub(in crate::model::tensor::execute) frames: Vec<Frame>,
    pub(in crate::model::tensor::execute) seed: Vec<f64>,
    pub(super) boundaries: Vec<f64>,
    pub(super) carry: Vec<f64>,
}

impl Frame {
    pub(in crate::model::tensor::execute) fn new(
        program: &Program<'_>,
        work: &TensorWorkspace,
    ) -> Self {
        let mut regions: Vec<_> = (0..program.plan.nodes.len()).map(|_| None).collect();
        for step in &program.steps {
            if let Some(control) = &step.control {
                let region = work.regions[step.node].as_ref().expect("control frames");
                let frames = region
                    .frames
                    .iter()
                    .enumerate()
                    .map(|(i, frame)| Self::new(control.frame_program(i), frame))
                    .collect();
                regions[step.node] = Some(Box::new(Region {
                    frames,
                    seed: vec![0.; step.primal.len()],
                    boundaries: region
                        .checkpoint
                        .as_ref()
                        .map_or_else(Vec::new, |c| vec![0.; c.carries.len()]),
                    carry: region
                        .checkpoint
                        .as_ref()
                        .map_or_else(Vec::new, |_| vec![0.; step.primal.len()]),
                }));
            }
        }
        Self {
            tangents: vec![0.; program.slots],
            adjoints: vec![0.; program.plan.slots],
            gradient: vec![0.; program.plan.dimension],
            regions,
        }
    }
}

impl<'a> Program<'a> {
    pub(in crate::model::tensor::execute) fn compile_curvature(plan: &'a Plan) -> Result<Self, E> {
        Self::compile_curvature_with_activity(plan, &vec![true; plan.inputs.len()])
    }

    pub(in crate::model::tensor::execute) fn compile_curvature_with_activity(
        plan: &'a Plan,
        activity: &[bool],
    ) -> Result<Self, E> {
        Self::compile_mode(plan, activity, true)
    }

    pub(in crate::model::tensor::execute) fn direction<'d>(
        &self,
        v: Value,
        tangents: &'d [f64],
    ) -> Option<&'d [f64]> {
        let Slot(start) = self.node_slots[v.node]?;
        Some(&tangents[start..start + self.plan.nodes[v.node].shape.elements])
    }

    pub(in crate::model::tensor::execute) fn control(&self, node: usize) -> &Control<'a> {
        let i = self
            .steps
            .binary_search_by_key(&node, |step| step.node)
            .expect("compiled control step");
        self.steps[i].control.as_deref().expect("compiled control")
    }

    pub(in crate::model::tensor::execute) fn directional_frame(
        &self,
        binding: &BoundProgram,
        work: &mut TensorWorkspace,
        frame: &mut Frame,
    ) -> Result<(), E> {
        binding.check_residual(work, work.prepared)?;
        if !std::ptr::eq(self.plan, binding.plan.as_ref()) {
            return Err(E::Shape);
        }
        for step in &self.steps {
            step.apply(
                self.plan,
                work,
                &mut frame.tangents,
                &mut [],
                frame.regions[step.node].as_deref_mut(),
            )?;
        }
        Ok(())
    }

    pub(in crate::model::tensor::execute) fn load_flat(&self, v: &[f64], frame: &mut Frame) {
        for (i, input) in self.plan.inputs.iter().enumerate() {
            self.load_direction(
                i,
                Some(&v[input.start..input.start + input.shape.elements]),
                &mut frame.tangents,
            );
        }
    }

    pub(in crate::model::tensor::execute) fn copy_curvature_result(
        &self,
        output: &mut [f64],
        frame: &Frame,
    ) {
        self.copy_result(output, &frame.tangents);
    }

    pub(in crate::model::tensor::execute) fn seeded_directional_vjp(
        &self,
        binding: &BoundProgram,
        work: &mut TensorWorkspace,
        frame: &mut Frame,
        [x, v]: [&[f64]; 2],
        [seed, seed_direction]: [&[f64]; 2],
    ) -> Result<(), E> {
        work.prepared = false;
        binding.validate(x, work)?;
        if v.len() != x.len() || seed_direction.len() != seed.len() {
            return Err(E::Shape);
        }
        // Admit the actual first-order evaluator before evaluating any
        // second-order coefficient (even for a zero direction).
        binding.execute_forward(x, work, None)?;
        binding.reverse(seed, work)?;
        if v.iter().chain(seed_direction).any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        binding.execute_forward(x, work, None)?;
        self.load_flat(v, frame);
        self.directional_frame(binding, work, frame)?;
        binding.reverse_directional(seed, seed_direction, work, self, frame)
    }
}

/// A differentiated transpose receives both output seeds. First-order
/// contributions are accumulated by the ordinary transpose in the same sweep.
pub(in crate::model::tensor::execute) struct ReversePair<'a, 'p> {
    pub(in crate::model::tensor::execute) program: &'a Program<'p>,
    pub(in crate::model::tensor::execute) frame: &'a mut Frame,
}

impl Rule<'_> {
    pub(in crate::model::tensor::execute) fn transpose_direction(
        self,
        node: usize,
        pair: ReversePair<'_, '_>,
        values: &[f64],
        active: &[bool],
        seed: &[f64],
    ) {
        let p = pair.program.plan;
        let Frame {
            tangents, adjoints, ..
        } = pair.frame;
        let read = |v: Value| &values[p.nodes[v.node].primal_range()];
        let direction = |v: Value| pair.program.direction(v, tangents);
        let start = p.nodes[node].start;
        let (before, after) = adjoints.split_at_mut(start);
        let ds = &after[..seed.len()];
        match self {
            Rule::Unary(kind, a) => {
                let x = read(a);
                let y = &values[p.nodes[node].primal_range()];
                let dx = direction(a);
                let dy = direction(Value {
                    owner: p.owner,
                    node,
                });
                let dest = &mut before[p.nodes[a.node].range()];
                for i in 0..seed.len() {
                    let slope = || derivative(kind, x[i], y[i]);
                    let coefficient = |d: Option<&[f64]>| d.map_or(0., |t| t[i]);
                    dest[i] += scaled(ds[i], slope)
                        + scaled(seed[i], || {
                            slope_direction(kind, x[i], y[i], coefficient(dx), coefficient(dy))
                        });
                }
            }
            Rule::Binary(b) => {
                let a = read(b.a);
                let r = read(b.b);
                let da = direction(b.a);
                let dr = direction(b.b);
                let lhs = p.nodes[b.a.node].start;
                let rhs = p.nodes[b.b.node].start;
                kernels::for_each_index(b, 0, seed.len(), |j, l, r_index| {
                    let (x, y) = (a[l], r[r_index]);
                    let dx = da.map_or(0., |d| d[l]);
                    let dy = dr.map_or(0., |d| d[r_index]);
                    if active[b.a.node] {
                        before[lhs + l] += scaled(ds[j], || match b.kind {
                            BinaryOp::Add | BinaryOp::Subtract => 1.,
                            BinaryOp::Multiply => y,
                            BinaryOp::Divide => kernels::divide_left(y),
                        }) + scaled(seed[j], || match b.kind {
                            BinaryOp::Add | BinaryOp::Subtract => 0.,
                            BinaryOp::Multiply => dy,
                            BinaryOp::Divide => -(dy / y) / y,
                        });
                    }
                    if active[b.b.node] {
                        before[rhs + r_index] += scaled(ds[j], || match b.kind {
                            BinaryOp::Add => 1.,
                            BinaryOp::Subtract => -1.,
                            BinaryOp::Multiply => x,
                            BinaryOp::Divide => kernels::divide_right(x, y),
                        }) + scaled(seed[j], || match b.kind {
                            BinaryOp::Add | BinaryOp::Subtract => 0.,
                            BinaryOp::Multiply => dx,
                            BinaryOp::Divide => {
                                -(dx / y) / y
                                    - 2. * (scaled(dy, || kernels::divide_right(x, y)) / y)
                            }
                        });
                    }
                });
            }
            Rule::Matrix { a, b, calls } => {
                let (av, bv) = (read(a), read(b));
                let starts = (p.nodes[a.node].start, p.nodes[b.node].start);
                let apply = |a: &[f64], r: &[f64], seeds: &[f64], adjoints: &mut [f64], active| {
                    kernels::matmul_reverse(
                        kernels::Pullback {
                            a,
                            r,
                            seeds,
                            adjoints,
                            starts,
                            active,
                        },
                        calls,
                    )
                };
                let (left, right) = (active[a.node], active[b.node]);
                apply(av, bv, ds, before, (left, right));
                if let Some(da) = direction(a).filter(|_| right) {
                    apply(da, bv, seed, before, (false, true));
                }
                if let Some(db) = direction(b).filter(|_| left) {
                    apply(av, db, seed, before, (true, false));
                }
            }
            Rule::Fixed | Rule::Alias(_) | Rule::Input(_) => {}
            Rule::Gather(..) | Rule::Sum(..) => self.transpose::<false, false>(
                node,
                Reverse {
                    plan: p,
                    primals: values,
                    adjoints,
                    active,
                    starts: &[],
                },
                0..seed.len(),
                None,
                lowering::AdjointWrite::Add,
            ),
            Rule::Native(_) | Rule::Cond(_) | Rule::Scan(_) => {
                unreachable!("second-order builtin rule")
            }
        }
    }
}

fn slope_direction(kind: UnaryOp, x: f64, y: f64, dx: f64, dy: f64) -> f64 {
    match kind {
        UnaryOp::Neg => 0.,
        UnaryOp::Exp => dy,
        UnaryOp::Expm1 => scaled(dx, || derivative(kind, x, y)),
        UnaryOp::Log | UnaryOp::Log1p if dx != 0. => {
            let slope = derivative(kind, x, y);
            -(dx * slope) * slope
        }
        UnaryOp::Log | UnaryOp::Log1p => 0.,
        UnaryOp::Tanh => (-2. * y) * dy,
        UnaryOp::Sin => -y * dx,
        UnaryOp::Cos => -y * dx,
        UnaryOp::Sqrt if dy != 0. => -(dy * derivative(kind, x, y) / y),
        UnaryOp::Sqrt => 0.,
    }
}

impl<'a, 'p> ReversePair<'a, 'p> {
    pub(in crate::model::tensor::execute) fn native(
        &mut self,
        i: usize,
        call: &registration::Call,
        (values, active): (&[f64], &[bool]),
        seed: &[f64],
        work: &mut registration::Work,
    ) -> Result<(), E> {
        let Ok(step) = self
            .program
            .steps
            .binary_search_by_key(&i, |step| step.node)
        else {
            return Ok(());
        };
        let step = &self.program.steps[step];
        let p = self.program.plan;
        let (adjoints, current) = self.frame.adjoints.split_at_mut(p.nodes[step.node].start);
        call.reverse_direction(
            (&p.nodes, values),
            (step.inputs.as_slice(), &self.frame.tangents),
            &values[step.primal.clone()],
            [seed, &current[..seed.len()]],
            work,
        )?;
        call.accumulate(work, |input, scores| {
            if active[input.node] {
                add_at(adjoints, p.nodes[input.node].start, scores);
            }
        });
        Ok(())
    }

    pub(in crate::model::tensor::execute) fn reborrow(&mut self) -> ReversePair<'_, 'p> {
        ReversePair {
            program: self.program,
            frame: self.frame,
        }
    }
}

pub(in crate::model::tensor::execute) struct ScanPair<'a, 'p> {
    program: &'a Program<'p>,
    tangents: &'a [f64],
    adjoints: &'a mut [f64],
    initial_seed: &'a [f64],
    region: Option<&'a mut Region>,
    control: Option<&'a Control<'p>>,
}

impl<'a, 'p> ScanPair<'a, 'p> {
    pub(in crate::model::tensor::execute) fn new(
        program: &'a Program<'p>,
        tangents: &'a [f64],
        adjoints: &'a mut [f64],
        seed: &'a [f64],
        region: Option<&'a mut Region>,
        node: usize,
    ) -> Self {
        let control = region.as_ref().map(|_| program.control(node));
        let mut pair = Self {
            program,
            tangents,
            adjoints,
            initial_seed: seed,
            region,
            control,
        };
        if let Some(tape) = &mut pair.region {
            tape.seed.copy_from_slice(seed);
        }
        pair
    }

    pub(in crate::model::tensor::execute) fn restore_boundary(&mut self, offset: usize) {
        if let Some(tape) = &mut self.region {
            let len = tape.carry.len();
            tape.carry
                .copy_from_slice(&tape.boundaries[offset..offset + len]);
        }
    }

    pub(in crate::model::tensor::execute) fn replay(
        &mut self,
        plan: &Plan,
        s: &Scan,
        work: &mut RegionWork,
        step: usize,
        index: usize,
    ) -> Result<(), E> {
        if let Some(tape) = &mut self.region {
            let argument = |i| {
                let value = if i == 1 { s.xs } else { s.captures[i - 2] };
                self.program.direction(value, self.tangents)
            };
            self.control.expect("compiled scan").scan_direction(
                (plan, s),
                work,
                step,
                argument,
                (&mut tape.carry, &mut []),
                Some(&mut tape.frames[index]),
            )?;
        }
        Ok(())
    }

    pub(in crate::model::tensor::execute) fn reverse(
        &mut self,
        s: &Scan,
        work: &mut RegionWork,
        guard: ResidualGuard,
        index: usize,
    ) -> Result<(), E> {
        if let Some(tape) = &mut self.region {
            let program = self.control.expect("compiled scan").frame_program(index);
            work.residual(guard, index)?;
            let frame = &mut tape.frames[index];
            s.body.reverse_directional(
                &work.seed,
                &tape.seed,
                &mut work.frames[index],
                program,
                frame,
            )?;
            let carry = tape.seed.len();
            tape.seed.copy_from_slice(&frame.gradient[..carry]);
        } else {
            work.reverse_residual(guard, index, &s.body)?;
        }
        Ok(())
    }

    pub(in crate::model::tensor::execute) fn accumulate(
        &mut self,
        start: usize,
        index: usize,
        input: Range<usize>,
    ) {
        if let Some(tape) = &self.region {
            add_at(self.adjoints, start, &tape.frames[index].gradient[input]);
        }
    }

    pub(in crate::model::tensor::execute) fn accumulate_initial(&mut self, plan: &Plan, s: &Scan) {
        let seed = self.region.as_ref().map_or(self.initial_seed, |r| &r.seed);
        add_at(self.adjoints, plan.nodes[s.initial.node].start, seed);
    }
}

fn add_at(adjoints: &mut [f64], start: usize, values: &[f64]) {
    for (adjoint, direction) in adjoints[start..start + values.len()].iter_mut().zip(values) {
        *adjoint += direction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differentiated_transpose_includes_a_seed_direction_at_a_zero_seed() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let square = b.binary(BinaryOp::Multiply, x, x).unwrap();
        let p = b.compile(square).unwrap().bind(&[]).unwrap();
        let plan = Program::compile_curvature(&p.plan).unwrap();
        let mut work = p.workspace();
        let mut frame = Frame::new(&plan, &work);
        for (seed, derivative) in [(0., 30.), (4., 46.)] {
            // (s * 2x)' = s' * 2x + s * 2x'.
            plan.seeded_directional_vjp(
                &p,
                &mut work,
                &mut frame,
                [&[3.], &[2.]],
                [&[seed], &[5.]],
            )
            .unwrap();
            assert_eq!(frame.gradient, [derivative]);
        }
    }

    #[test]
    fn registered_curvature_reads_forward_state_and_admits_complete_contributions() {
        use crate::capability::PotentialHessianVector;

        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let square = b
            .register_block_with_vjp_jvp(
                &[x],
                (),
                BlockRules {
                    inputs: vec![BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar(),
                    }],
                    output: Shape::scalar(),
                    workspace: |_| 0.,
                    heap_bytes: 0,
                    forward: |_, inputs, out, saved| {
                        *saved = inputs.real(0)?[0];
                        out[0] = *saved * *saved;
                        Ok(())
                    },
                    preparation: None,
                    vjp: Some(|_, _, _, seed, score, saved| {
                        score[0] = 2. * *saved * seed[0];
                        // The ordinary VJP may mutate private scratch.
                        *saved = f64::NAN;
                        Ok(())
                    }),
                },
                |_, _, tangent, _, out, saved| {
                    out[0] = 2. * *saved * tangent.real(0)?.unwrap()[0];
                    Ok(())
                },
                |_, inputs, tangent, _, [seed, seed_direction], out, saved| {
                    assert_eq!(*saved, inputs.real(0)?[0]);
                    if *saved != 1. {
                        out[0] = 2.
                            * (*saved * seed_direction[0] + seed[0] * tangent.real(0)?.unwrap()[0]);
                    }
                    Ok(())
                },
            )
            .unwrap();
        let p = b.compile(square).unwrap().bind(&[]).unwrap();
        let target = p.potential_hvp_target().unwrap();
        let mut hvp = [19.];
        assert!(matches!(
            target.potential_hvp(&[1.], &[3.], &mut hvp),
            Err(E::Numerical)
        ));
        assert_eq!(hvp, [19.]);
        target.potential_hvp(&[2.], &[3.], &mut hvp).unwrap();
        assert_eq!(hvp, [-6.]);
    }
}
