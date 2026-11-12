//! Dense block execution and reverse propagation, with preallocated region frames.
use super::*;
use alea_math::numeric::CompensatedSum;

struct RegionWork {
    frames: Vec<TensorWorkspace>,
    args: Vec<f64>,
    seed: Vec<f64>,
    checkpoint: Option<Box<Checkpoint>>,
}
// Checkpoint-only metadata must not enlarge every region slot by a Vec plus
// an interval. The single cold allocation is absent for retained scans/branches.
struct Checkpoint {
    carries: Vec<f64>,
    interval: usize,
}
// Segment checkpointing only for substantial retained tapes. Short scans keep
// their original one-forward-pass VJP. Integer sqrt avoids rounded size decisions.
// The extra frame preserves the first iteration's distinct derivative activity.
fn scan_interval(s: &Scan, carry: usize) -> usize {
    if !s.checkpointed || s.steps < 64 {
        return 0;
    }
    // Minimize the continuous byte model B*K + (T/K)*C, not just the number
    // of frames. A carry is often much smaller than a complete body workspace.
    // u128 covers the product of two addressable usize-sized byte extents.
    let carry_bytes = carry.saturating_mul(size_of::<f64>());
    let square = (s.steps as u128 * carry_bytes as u128) / s.body.plan.workspace_bytes as u128;
    let interval =
        usize::try_from(square.isqrt()).map_or(s.steps, |interval| interval.clamp(1, s.steps));
    let retained = s.steps.saturating_mul(s.body.plan.workspace_bytes);
    let checkpointed = (interval + 1)
        .saturating_mul(s.body.plan.workspace_bytes)
        .saturating_add(size_of::<Checkpoint>())
        .saturating_add(
            s.steps
                .div_ceil(interval)
                .saturating_mul(carry)
                .saturating_mul(size_of::<f64>()),
        );
    if retained >= 1024 * 1024 && checkpointed < retained / 2 {
        interval
    } else {
        0
    }
}
// Account for recursively retained frames BEFORE any execution workspace is
// allocated. Includes metadata, not just child value slots: even empty-output
// loops can otherwise overflow the frame vector.
pub(super) fn workspace_bytes(p: &Plan) -> Result<usize, E> {
    let add = |a: usize, b: usize| {
        a.checked_add(b)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(E::Shape)
    };
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(E::Shape)
    };
    let mut bytes = add(
        add(size_of::<TensorWorkspace>(), size_of::<AdjointPlan>())?,
        mul(
            add(
                add(p.slots, super::pullback::adjoint_layout(p, &p.active)?.1)?,
                p.dimension,
            )?,
            size_of::<f64>(),
        )?,
    )?;
    bytes = add(bytes, mul(p.nodes.len(), size_of::<usize>())?)?;
    bytes = add(bytes, mul(p.nodes.len(), size_of::<Option<RegionWork>>())?)?;
    bytes = add(bytes, p.nodes.len())?;
    bytes = add(bytes, p.nodes.len())?; // Cached reverse schedule eligibility.
    bytes = add(bytes, mul(p.nodes.len(), size_of::<Range<usize>>())?)?;
    bytes = add(bytes, mul(p.data.len(), size_of::<Arc<[f64]>>())?)?;
    if let Some(last) = p.last_native_block() {
        bytes = add(
            bytes,
            mul(last + 1, size_of::<Option<registration::Work>>())?,
        )?;
        for (i, node) in p.nodes[..=last].iter().enumerate() {
            if p.active[i]
                && let Some(call) = node.native_block()
            {
                bytes = add(bytes, call.workspace_bytes())?;
            }
        }
    }
    for (index, node) in p.nodes.iter().enumerate() {
        if !p.active[index] {
            continue;
        }
        let (frames, args) = match &node.op {
            Op::Cond(c) => (
                add(c.yes.plan.workspace_bytes, c.no.plan.workspace_bytes)?,
                c.yes.dimension(),
            ),
            Op::Scan(s) => {
                let interval = scan_interval(s, node.shape.elements);
                let frames = if interval == 0 {
                    mul(s.steps, s.body.plan.workspace_bytes)?
                } else {
                    add(
                        mul(interval + 1, s.body.plan.workspace_bytes)?,
                        add(
                            size_of::<Checkpoint>(),
                            mul(
                                mul(s.steps.div_ceil(interval), node.shape.elements)?,
                                size_of::<f64>(),
                            )?,
                        )?,
                    )?
                };
                (frames, s.body.dimension())
            }
            _ => continue,
        };
        bytes = add(
            bytes,
            add(
                frames,
                mul(add(args, node.shape.elements)?, size_of::<f64>())?,
            )?,
        )?;
    }
    Ok(bytes)
}
// Keep cold layout and optional reverse metadata outside retained scan-frame
// headers. A Vec per optional feature taxes every trip even when it is unused.
#[inline]
fn adjoint_start<const PLANNED: bool>(p: &Plan, starts: &[usize], i: usize) -> usize {
    if PLANNED { starts[i] } else { p.nodes[i].start }
}
struct AdjointPlan {
    starts: Box<[usize]>,
    seed: Option<Box<[f64]>>,
    groups: Box<[bool]>,
}
/// Execution-local values, planned adjoints and structured-control-flow frames.
/// Reusable for same-plan immutable data bindings. Outputs are scratch on failure;
/// public evaluation methods stage caller-visible values and gradients.
pub struct TensorWorkspace {
    owner: u64,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    adjoint_plan: Option<Box<AdjointPlan>>,
    gradient: Vec<f64>,
    differential: Vec<bool>,
    adjoint_ranges: Vec<Range<usize>>,
    // Own the cached binding arrays: address-only identity could be reused after
    // the old snapshot is dropped. Arc identity remains valid until refreshed.
    data: Vec<Arc<[f64]>>,
    regions: Vec<Option<RegionWork>>,
    blocks: Vec<Option<registration::Work>>,
}
impl BoundProgram {
    pub(in crate::model) fn reverse_point(
        &self,
        seed: &[f64],
        scores: &mut [f64],
        w: &mut TensorWorkspace,
    ) -> Result<(), E> {
        if seed.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        self.reverse(seed, w)?;
        scores.copy_from_slice(&w.gradient);
        Ok(())
    }

    /// Native per-execution workspace estimate. This is a tensor-target budget,
    /// separate from session sampler/storage limits and immutable bindings.
    pub fn workspace_bytes(&self) -> usize {
        self.plan.workspace_bytes
    }
    /// Checks a caller-owned byte budget before allocating any execution frames.
    /// Use one budget per simultaneous evaluation/chain. This is not an OOM
    /// guarantee: allocator metadata and unrelated process storage are excluded.
    /// # Errors
    /// Returns required/allowed bytes without allocating tensor execution scratch.
    pub fn workspace_with_limit(
        &self,
        limit: usize,
    ) -> Result<TensorWorkspace, WorkspaceBudgetError> {
        self.check_workspace_limit(limit)?;
        Ok(self.workspace())
    }
    /// Allocates all value/adjoint/control-flow storage once, outside evaluation.
    pub fn workspace(&self) -> TensorWorkspace {
        self.workspace_with_activity(&vec![true; self.plan.inputs.len()])
    }
    // Specialize derivative activity through region operands, not predicates.
    // Closed data and constant operands must not acquire spurious derivatives.
    pub(in crate::model) fn workspace_with_activity(&self, inputs: &[bool]) -> TensorWorkspace {
        let p = &self.plan;
        let mut differential = vec![false; p.nodes.len()];
        let mut regions = Vec::with_capacity(p.nodes.len());
        for (i, node) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                regions.push(None);
                continue;
            }
            let region = match &node.op {
                Op::Cond(c) => {
                    let inputs: Vec<_> = c.args.iter().map(|a| differential[a.node]).collect();
                    let yes = c.yes.workspace_with_activity(&inputs);
                    let no = c.no.workspace_with_activity(&inputs);
                    differential[i] = yes.differential[c.yes.plan.output.node]
                        || no.differential[c.no.plan.output.node];
                    Some(RegionWork {
                        frames: vec![yes, no],
                        args: vec![0.; c.yes.dimension()],
                        seed: vec![0.; node.shape.elements],
                        checkpoint: None,
                    })
                }
                Op::Scan(s) => {
                    let mut inputs = vec![differential[s.initial.node], differential[s.xs.node]];
                    inputs.extend(s.captures.iter().map(|a| differential[a.node]));
                    let interval = scan_interval(s, node.shape.elements);
                    let count = if interval == 0 { s.steps } else { interval + 1 };
                    let mut frames = Vec::with_capacity(count);
                    // Body output activity has form (depends_on_carry && carry)
                    // || other_operands, so it reaches a fixed point after one
                    // step. Frame zero is first-step-specific; all later slots
                    // have steady-state activity, including nested regions.
                    for _ in 0..count {
                        let frame = s.body.workspace_with_activity(&inputs);
                        inputs[0] = frame.differential[s.body.plan.output.node];
                        frames.push(frame);
                    }
                    differential[i] = inputs[0];
                    Some(RegionWork {
                        frames,
                        args: vec![0.; s.body.dimension()],
                        seed: vec![0.; node.shape.elements],
                        checkpoint: (interval != 0).then(|| {
                            Box::new(Checkpoint {
                                carries: vec![0.; s.steps.div_ceil(interval) * node.shape.elements],
                                interval,
                            })
                        }),
                    })
                }
                Op::Input(index) => {
                    differential[i] = inputs[*index];
                    None
                }
                _ => {
                    differential[i] = node.inputs().iter().any(|v| differential[v.node]);
                    None
                }
            };
            regions.push(region);
        }
        let mut adjoint_ranges: Vec<Range<usize>> = Vec::with_capacity(p.nodes.len());
        let (mut adjoint_starts, mut adjoint_slots) =
            super::pullback::adjoint_layout(p, &differential)
                .expect("compiled tensor adjoint layout is addressable");
        let reuse = super::pullback::reused_adjoint_layout(p, &differential, adjoint_slots)
            .expect("compiled tensor live intervals are addressable");
        let reuse_adjoints = reuse.is_some();
        let mut seed_slots = 0;
        if let Some((offsets, slots, scratch)) = reuse {
            adjoint_starts = offsets;
            adjoint_slots = slots;
            seed_slots = scratch;
        }
        // A scalar scan body can save less storage than the address table costs.
        // Keep its original dense layout and avoid a per-trip metadata object.
        let layout_bytes = size_of::<AdjointPlan>() + p.nodes.len() * size_of::<usize>();
        let planned = p.execution.is_some()
            || reuse_adjoints
            || (p.slots - adjoint_slots) * size_of::<f64>() > layout_bytes;
        if !planned {
            adjoint_slots = p.slots;
            for (i, n) in p.nodes.iter().enumerate() {
                adjoint_starts[i] = n.start;
            }
        }
        let mut values = vec![0.; p.slots];
        for (i, node) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                continue;
            }
            match &node.op {
                Op::Data(index) => values[node.range()].copy_from_slice(&self.data[*index]),
                Op::Constant(data) => values[node.range()].copy_from_slice(data),
                _ => {}
            }
            if !reuse_adjoints
                && differential[i]
                && !matches!(node.op, Op::Alias(_))
                && node.shape.elements != 0
            {
                if let Some(last) = adjoint_ranges
                    .last_mut()
                    .filter(|last| last.end == adjoint_starts[i])
                {
                    last.end = adjoint_starts[i] + node.shape.elements;
                } else {
                    adjoint_ranges.push(adjoint_starts[i]..adjoint_starts[i] + node.shape.elements);
                }
            }
        }
        if reuse_adjoints {
            adjoint_ranges.push(0..adjoint_slots);
        }
        let reverse_groups = p.execution.as_ref().map_or_else(Vec::new, |schedule| {
            schedule
                .0
                .iter()
                .map(|step| match step {
                    fusion::Step::Fused(group) => {
                        fusion::reverse_compatible(p, group, &differential)
                    }
                    fusion::Step::Nodes(_) => false,
                })
                .collect()
        });
        TensorWorkspace {
            owner: p.owner,
            values,
            adjoints: vec![0.; adjoint_slots],
            adjoint_plan: planned.then(|| {
                Box::new(AdjointPlan {
                    starts: adjoint_starts.into_boxed_slice(),
                    seed: reuse_adjoints.then(|| vec![0.; seed_slots].into_boxed_slice()),
                    groups: reverse_groups.into_boxed_slice(),
                })
            }),
            gradient: vec![0.; self.dimension()],
            differential,
            adjoint_ranges,
            data: self.data.clone(),
            regions,
            blocks: p.last_native_block().map_or_else(Vec::new, |last| {
                p.nodes[..=last]
                    .iter()
                    .zip(&p.active)
                    .map(|(node, &active)| {
                        active
                            .then(|| node.native_block().map(registration::Call::workspace))
                            .flatten()
                    })
                    .collect()
            }),
        }
    }
    fn validate(&self, x: &[f64], w: &TensorWorkspace) -> Result<(), E> {
        if x.len() != self.dimension()
            || w.owner != self.plan.owner
            || w.values.len() != self.plan.slots
        {
            return Err(E::Shape);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(())
    }
    /// Evaluates all result elements without computing derivatives.
    /// # Errors
    /// Shape/workspace/domain errors preserve the caller's entire output buffer.
    pub fn evaluate(&self, x: &[f64], out: &mut [f64], w: &mut TensorWorkspace) -> Result<(), E> {
        if out.len() != self.output_shape().elements {
            return Err(E::Shape);
        }
        self.validate(x, w)?;
        self.execute_forward(x, w)?;
        out.copy_from_slice(&w.values[self.plan.nodes[self.plan.output.node].range()]);
        Ok(())
    }
    /// Fused value and vector-Jacobian product, without materializing a Jacobian.
    /// All supplied seeds must be finite. Data slots are not differentiable.
    /// # Errors
    /// Errors preserve both caller buffers; scratch is reset on the next call.
    pub fn value_and_vjp(
        &self,
        x: &[f64],
        seed: &[f64],
        out: &mut [f64],
        g: &mut [f64],
        w: &mut TensorWorkspace,
    ) -> Result<(), E> {
        if out.len() != self.output_shape().elements
            || seed.len() != out.len()
            || g.len() != self.dimension()
        {
            return Err(E::Shape);
        }
        self.validate(x, w)?;
        if seed.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        self.execute_forward(x, w)?;
        self.reverse(seed, w)?;
        out.copy_from_slice(&w.values[self.plan.nodes[self.plan.output.node].range()]);
        g.copy_from_slice(&w.gradient);
        Ok(())
    }
    fn execute_forward(&self, x: &[f64], w: &mut TensorWorkspace) -> Result<(), E> {
        let Some(schedule) = &self.plan.execution else {
            return self.forward(x, w);
        };
        for step in &schedule.0 {
            match step {
                fusion::Step::Nodes(range) => self.forward_nodes::<false>(x, w, range.clone())?,
                fusion::Step::Fused(group) => fusion::forward(&self.plan, group, &mut w.values)?,
            }
        }
        Ok(())
    }
    // Keep the unscheduled interpreter as one outlined call, as before fusion.
    // In particular, scalar scan bodies must not inline a complete interpreter
    // or re-check the planner on every trip.
    fn forward(&self, x: &[f64], w: &mut TensorWorkspace) -> Result<(), E> {
        self.forward_nodes::<true>(x, w, 0..0)
    }
    #[inline(always)]
    fn forward_nodes<const FULL: bool>(
        &self,
        x: &[f64],
        w: &mut TensorWorkspace,
        range: Range<usize>,
    ) -> Result<(), E> {
        let p = &self.plan;
        let nodes = if FULL {
            &p.nodes[..]
        } else {
            &p.nodes[range.clone()]
        };
        for (offset, node) in nodes.iter().enumerate() {
            let i = if FULL { offset } else { range.start + offset };
            if !p.active[i] {
                continue;
            }
            match &node.op {
                Op::Registered(op) => match &**op {
                    registration::Operation::Data(_) => continue,
                    registration::Operation::Native(call) => call.forward(
                        &p.nodes,
                        node,
                        &mut w.values,
                        w.blocks[i].as_mut().expect("registered block workspace"),
                        #[cfg(test)]
                        self.preparation(i),
                    )?,
                },
                Op::Input(input) => {
                    let input = &p.inputs[*input];
                    w.values[node.range()]
                        .copy_from_slice(&x[input.start..input.start + input.shape.elements]);
                    // The complete input was validated before entering forward.
                    continue;
                }
                Op::Data(data) => {
                    if !Arc::ptr_eq(&w.data[*data], &self.data[*data]) {
                        w.values[node.range()].copy_from_slice(&self.data[*data]);
                        w.data[*data] = Arc::clone(&self.data[*data]);
                    }
                    // Both construction and immutable rebinding admit finiteness.
                    continue;
                }
                Op::Constant(_) | Op::Alias(_) => continue,
                Op::Unary(op, a) => {
                    let (before, after) = w.values.split_at_mut(node.start);
                    for (out, &a) in after[..node.shape.elements]
                        .iter_mut()
                        .zip(&before[p.nodes[a.node].range()])
                    {
                        *out = unary(*op, a);
                    }
                }
                Op::Binary(b) => {
                    let (before, after) = w.values.split_at_mut(node.start);
                    kernels::forward(
                        b,
                        &before[p.nodes[b.a.node].range()],
                        &before[p.nodes[b.b.node].range()],
                        &mut after[..node.shape.elements],
                    );
                }
                Op::Sum(a) => {
                    let mut sum = CompensatedSum::default();
                    for &value in &w.values[p.nodes[a.node].range()] {
                        sum.add(value);
                    }
                    w.values[node.start] = sum.total();
                }
                Op::Gather(a, indices) => {
                    let (before, after) = w.values.split_at_mut(node.start);
                    let start = p.nodes[a.node].start;
                    for (out, &index) in after[..node.shape.elements].iter_mut().zip(indices.iter())
                    {
                        *out = before[start + index];
                    }
                }
                Op::Matmul {
                    a,
                    b,
                    m,
                    k,
                    n,
                    kernel,
                } => {
                    if node.shape.elements == 0 {
                        continue;
                    }
                    let (before, after) = w.values.split_at_mut(node.start);
                    kernels::matmul_forward(
                        &before[p.nodes[a.node].range()],
                        &before[p.nodes[b.node].range()],
                        &mut after[..node.shape.elements],
                        (*m, *k, *n),
                        *kernel,
                    );
                }
                Op::Cond(c) => {
                    let choice = if compare(
                        c.comparison,
                        w.values[p.nodes[c.lhs.node].start],
                        w.values[p.nodes[c.rhs.node].start],
                    ) {
                        0
                    } else {
                        1
                    };
                    let selected = if choice == 0 { &c.yes } else { &c.no };
                    let region = w.regions[i].as_mut().expect("condition frame");
                    let mut offset = 0;
                    for arg in &c.args {
                        let values = &w.values[p.nodes[arg.node].range()];
                        region.args[offset..offset + values.len()].copy_from_slice(values);
                        offset += values.len();
                    }
                    selected.execute_forward(&region.args, &mut region.frames[choice])?;
                    let output = selected.plan.nodes[selected.plan.output.node].range();
                    w.values[node.range()].copy_from_slice(&region.frames[choice].values[output]);
                }
                Op::Scan(s) => {
                    let region = w.regions[i].as_mut().expect("scan frames");
                    let (before, after) = w.values.split_at_mut(node.start);
                    scan_forward(s, p, before, &mut after[..node.shape.elements], region)?;
                }
            }
            if alea_math::numeric::first_nonfinite(&w.values[node.range()]).is_some() {
                return Err(E::Numerical);
            }
        }
        Ok(())
    }
    fn reverse(&self, seed: &[f64], w: &mut TensorWorkspace) -> Result<(), E> {
        let p = &self.plan;
        for range in &w.adjoint_ranges {
            w.adjoints[range.clone()].fill(0.);
        }
        w.gradient.fill(0.);
        if w.differential[p.output.node] {
            let start = w
                .adjoint_plan
                .as_ref()
                .map_or(p.nodes[p.output.node].start, |a| a.starts[p.output.node]);
            w.adjoints[start..start + seed.len()].copy_from_slice(seed);
        }
        if let Some(schedule) = &p.execution {
            for (index, step) in schedule.0.iter().enumerate().rev() {
                match step {
                    fusion::Step::Fused(group)
                        if w.adjoint_plan.as_ref().expect("fused layout").groups[index] =>
                    {
                        fusion::reverse(
                            p,
                            group,
                            &w.values,
                            &mut w.adjoints,
                            &w.differential,
                            &w.adjoint_plan.as_ref().expect("fused layout").starts,
                        );
                    }
                    fusion::Step::Fused(group) => {
                        self.reverse_nodes::<false, true>(w, group.maps.start..group.end())?
                    }
                    fusion::Step::Nodes(range) => {
                        self.reverse_nodes::<false, true>(w, range.clone())?
                    }
                }
            }
        } else {
            if let Some(layout) = &w.adjoint_plan {
                if layout.seed.is_some() {
                    self.reverse_nodes::<true, true>(w, 0..p.nodes.len())?;
                } else {
                    self.reverse_nodes::<false, true>(w, 0..p.nodes.len())?;
                }
            } else {
                self.reverse_nodes::<false, false>(w, 0..p.nodes.len())?;
            }
        }
        if alea_math::numeric::first_nonfinite(&w.gradient).is_some()
            || w.adjoint_ranges.iter().any(|range| {
                alea_math::numeric::first_nonfinite(&w.adjoints[range.clone()]).is_some()
            })
        {
            return Err(E::Numerical);
        }
        Ok(())
    }
    fn reverse_nodes<const REUSE: bool, const PLANNED: bool>(
        &self,
        w: &mut TensorWorkspace,
        range: Range<usize>,
    ) -> Result<(), E> {
        let p = &self.plan;
        let (starts, mut seed_buffer): (&[usize], Option<&mut [f64]>) =
            match w.adjoint_plan.as_mut() {
                Some(a) => (&a.starts, a.seed.as_deref_mut()),
                None => (&[], None),
            };
        for i in range.rev() {
            let node = &p.nodes[i];
            if !p.active[i] || !w.differential[i] {
                continue;
            }
            if REUSE && matches!(node.op, Op::Binary(_) | Op::Matmul { .. }) {
                let start = adjoint_start::<PLANNED>(p, starts, i);
                seed_buffer.as_deref_mut().expect("reused adjoint seed")[..node.shape.elements]
                    .copy_from_slice(&w.adjoints[start..start + node.shape.elements]);
            }
            match &node.op {
                Op::Registered(op) => match &**op {
                    registration::Operation::Data(_) => {}
                    registration::Operation::Native(call) => {
                        let base = adjoint_start::<PLANNED>(p, starts, i);
                        let work = w.blocks[i].as_mut().expect("registered block workspace");
                        call.reverse(
                            &p.nodes,
                            &w.values,
                            &w.adjoints[base..base + node.shape.elements],
                            &w.values[node.range()],
                            work,
                            #[cfg(test)]
                            self.preparation(i),
                        )?;
                        call.accumulate(work, |input, scores| {
                            if w.differential[input.node] {
                                let base = adjoint_start::<PLANNED>(p, starts, input.node);
                                for (adjoint, &score) in
                                    w.adjoints[base..base + scores.len()].iter_mut().zip(scores)
                                {
                                    *adjoint += score;
                                }
                            }
                        });
                    }
                },
                Op::Input(input) => {
                    let input = &p.inputs[*input];
                    w.gradient[input.start..input.start + input.shape.elements].copy_from_slice(
                        &w.adjoints[adjoint_start::<PLANNED>(p, starts, i)
                            ..adjoint_start::<PLANNED>(p, starts, i) + node.shape.elements],
                    );
                }
                Op::Data(_) | Op::Constant(_) | Op::Alias(_) => {}
                Op::Unary(op, a) => {
                    kernels::unary_reverse_range(
                        *op,
                        &w.values[p.nodes[a.node].range()],
                        &w.values[node.range()],
                        &mut w.adjoints,
                        (
                            adjoint_start::<PLANNED>(p, starts, a.node),
                            adjoint_start::<PLANNED>(p, starts, i),
                        ),
                    );
                }
                Op::Binary(b) => {
                    let (adjoints, after): (&mut [f64], &[f64]) = if REUSE {
                        (
                            &mut w.adjoints,
                            &seed_buffer.as_deref().expect("reused adjoint seed")
                                [..node.shape.elements],
                        )
                    } else {
                        let (before, after) = w
                            .adjoints
                            .split_at_mut(adjoint_start::<PLANNED>(p, starts, i));
                        (before, &after[..node.shape.elements])
                    };
                    kernels::reverse(
                        b,
                        kernels::Pullback {
                            a: &w.values[p.nodes[b.a.node].range()],
                            r: &w.values[p.nodes[b.b.node].range()],
                            seeds: &after[..node.shape.elements],
                            adjoints,
                            starts: (
                                adjoint_start::<PLANNED>(p, starts, b.a.node),
                                adjoint_start::<PLANNED>(p, starts, b.b.node),
                            ),
                            active: (w.differential[b.a.node], w.differential[b.b.node]),
                        },
                    );
                }
                Op::Sum(a) => {
                    let seed = w.adjoints[adjoint_start::<PLANNED>(p, starts, i)];
                    for j in adjoint_start::<PLANNED>(p, starts, a.node)
                        ..adjoint_start::<PLANNED>(p, starts, a.node)
                            + p.nodes[a.node].shape.elements
                    {
                        w.adjoints[j] += seed;
                    }
                }
                Op::Gather(a, indices) => {
                    let start = adjoint_start::<PLANNED>(p, starts, a.node);
                    for (j, &index) in indices.iter().enumerate() {
                        w.adjoints[start + index] +=
                            w.adjoints[adjoint_start::<PLANNED>(p, starts, i) + j];
                    }
                }
                Op::Matmul {
                    a,
                    b,
                    m,
                    k,
                    n,
                    kernel,
                } => {
                    if node.shape.elements == 0 {
                        continue;
                    }
                    let (adjoints, seeds): (&mut [f64], &[f64]) = if REUSE {
                        (
                            &mut w.adjoints,
                            &seed_buffer.as_deref().expect("reused adjoint seed")
                                [..node.shape.elements],
                        )
                    } else {
                        let (before, after) = w
                            .adjoints
                            .split_at_mut(adjoint_start::<PLANNED>(p, starts, i));
                        (before, &after[..node.shape.elements])
                    };
                    kernels::matmul_reverse(
                        (*m, *k, *n),
                        kernels::Pullback {
                            a: &w.values[p.nodes[a.node].range()],
                            r: &w.values[p.nodes[b.node].range()],
                            seeds: &seeds[..node.shape.elements],
                            adjoints,
                            starts: (
                                adjoint_start::<PLANNED>(p, starts, a.node),
                                adjoint_start::<PLANNED>(p, starts, b.node),
                            ),
                            active: (w.differential[a.node], w.differential[b.node]),
                        },
                        *kernel,
                    );
                }
                Op::Cond(c) => {
                    let choice = if compare(
                        c.comparison,
                        w.values[p.nodes[c.lhs.node].start],
                        w.values[p.nodes[c.rhs.node].start],
                    ) {
                        0
                    } else {
                        1
                    };
                    let selected = if choice == 0 { &c.yes } else { &c.no };
                    let region = w.regions[i].as_mut().expect("condition frame");
                    region.seed.copy_from_slice(
                        &w.adjoints[adjoint_start::<PLANNED>(p, starts, i)
                            ..adjoint_start::<PLANNED>(p, starts, i) + node.shape.elements],
                    );
                    selected.reverse(&region.seed, &mut region.frames[choice])?;
                    let mut offset = 0;
                    for arg in &c.args {
                        for j in 0..p.nodes[arg.node].shape.elements {
                            if w.differential[arg.node] {
                                w.adjoints[adjoint_start::<PLANNED>(p, starts, arg.node) + j] +=
                                    region.frames[choice].gradient[offset];
                            }
                            offset += 1;
                        }
                    }
                }
                Op::Scan(s) => {
                    let region = w.regions[i].as_mut().expect("scan frames");
                    region.seed.copy_from_slice(
                        &w.adjoints[adjoint_start::<PLANNED>(p, starts, i)
                            ..adjoint_start::<PLANNED>(p, starts, i) + node.shape.elements],
                    );
                    scan_reverse(
                        s,
                        p,
                        &w.values,
                        &mut w.adjoints,
                        region,
                        (starts, &w.differential),
                    )?;
                }
            }
            if REUSE && !matches!(node.op, Op::Alias(_)) {
                let start = adjoint_start::<PLANNED>(p, starts, i);
                let seed = &mut w.adjoints[start..start + node.shape.elements];
                // Admit every historical seed before recycling: clearing a dead
                // nonfinite intermediate must never turn failure into success.
                if alea_math::numeric::first_nonfinite(seed).is_some() {
                    return Err(E::Numerical);
                }
                seed.fill(0.);
            }
        }
        Ok(())
    }
}
// Select storage policy once per scan, not per ordinary trip. Keeping the loop
// bodies out of the shared interpreter also bounds its code-size cost.
#[inline]
fn scan_forward(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    out: &mut [f64],
    region: &mut RegionWork,
) -> Result<(), E> {
    if s.body.plan.execution.is_none() {
        scan_forward_with::<false>(s, p, parent, out, region)
    } else {
        scan_forward_with::<true>(s, p, parent, out, region)
    }
}
#[inline(never)]
fn scan_forward_with<const FUSED: bool>(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    out: &mut [f64],
    region: &mut RegionWork,
) -> Result<(), E> {
    let carry = out.len();
    if region.checkpoint.is_none() {
        // Original retained-frame path: no checkpoint index/modulo/lookup per trip.
        for step in 0..s.steps {
            let previous = if step == 0 {
                &parent[p.nodes[s.initial.node].range()]
            } else {
                &region.frames[step - 1].values[s.body.plan.nodes[s.body.plan.output.node].range()]
            };
            region.args[..carry].copy_from_slice(previous);
            scan_arguments(s, p, parent, &mut region.args, step);
            scan_body_forward::<FUSED>(&s.body, &region.args, &mut region.frames[step])?;
        }
        if s.steps == 0 {
            out.copy_from_slice(&parent[p.nodes[s.initial.node].range()]);
        } else {
            out.copy_from_slice(
                &region.frames[s.steps - 1].values
                    [s.body.plan.nodes[s.body.plan.output.node].range()],
            );
        }
    } else {
        region.args[..carry].copy_from_slice(&parent[p.nodes[s.initial.node].range()]);
        for step in 0..s.steps {
            let checkpoint = region.checkpoint.as_mut().expect("checkpointed forward");
            if step % checkpoint.interval == 0 {
                let start = step / checkpoint.interval * carry;
                checkpoint.carries[start..start + carry].copy_from_slice(&region.args[..carry]);
            }
            scan_step::<FUSED>(s, p, parent, region, step)?;
        }
        out.copy_from_slice(&region.args[..carry]);
    }
    Ok(())
}

#[inline]
fn scan_reverse(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    adjoints: &mut [f64],
    region: &mut RegionWork,
    layout: (&[usize], &[bool]),
) -> Result<(), E> {
    if s.body.plan.execution.is_none() {
        scan_reverse_with::<false>(s, p, parent, adjoints, region, layout)
    } else {
        scan_reverse_with::<true>(s, p, parent, adjoints, region, layout)
    }
}
#[inline(never)]
fn scan_reverse_with<const FUSED: bool>(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    adjoints: &mut [f64],
    region: &mut RegionWork,
    layout: (&[usize], &[bool]),
) -> Result<(), E> {
    if let Some(checkpoint) = region.checkpoint.as_ref() {
        let interval = checkpoint.interval;
        let carry = region.seed.len();
        let mut end = s.steps;
        while end != 0 {
            let segment = (end - 1) / interval;
            let start = segment * interval;
            let saved = segment * carry;
            region.args[..carry].copy_from_slice(
                &region.checkpoint.as_ref().expect("replay boundary").carries[saved..saved + carry],
            );
            for step in start..end {
                scan_step::<FUSED>(s, p, parent, region, step)?;
            }
            for step in (start..end).rev() {
                reverse_scan_step(
                    s,
                    p,
                    adjoints,
                    region,
                    step,
                    scan_frame(interval, step),
                    layout,
                )?;
            }
            end = start;
        }
    } else {
        for step in (0..s.steps).rev() {
            reverse_scan_step(s, p, adjoints, region, step, step, layout)?;
        }
    }
    if layout.1[s.initial.node] {
        let base = layout
            .0
            .get(s.initial.node)
            .copied()
            .unwrap_or(p.nodes[s.initial.node].start);
        for (g, &seed) in adjoints[base..base + region.seed.len()]
            .iter_mut()
            .zip(&region.seed)
        {
            *g += seed;
        }
    }
    Ok(())
}

#[inline]
fn reverse_scan_step(
    s: &Scan,
    p: &Plan,
    adjoints: &mut [f64],
    region: &mut RegionWork,
    step: usize,
    index: usize,
    layout: (&[usize], &[bool]),
) -> Result<(), E> {
    let carry = region.seed.len();
    let frame = &mut region.frames[index];
    s.body.reverse(&region.seed, frame)?;
    region.seed.copy_from_slice(&frame.gradient[..carry]);
    if layout.1[s.xs.node] {
        let start = layout
            .0
            .get(s.xs.node)
            .copied()
            .unwrap_or(p.nodes[s.xs.node].start)
            + step * s.row;
        for j in 0..s.row {
            adjoints[start + j] += frame.gradient[carry + j];
        }
    }
    let mut offset = carry + s.row;
    for arg in &s.captures {
        for j in 0..p.nodes[arg.node].shape.elements {
            if layout.1[arg.node] {
                adjoints[layout
                    .0
                    .get(arg.node)
                    .copied()
                    .unwrap_or(p.nodes[arg.node].start)
                    + j] += frame.gradient[offset];
            }
            offset += 1;
        }
    }
    Ok(())
}

fn scan_frame(interval: usize, step: usize) -> usize {
    if interval == 0 || step == 0 {
        step
    } else {
        1 + step % interval
    }
}
// Args' carry is both the input and returned carry. No heap work occurs here,
// including segment replay. Body bindings are immutable and operations pure.
fn scan_step<const FUSED: bool>(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    region: &mut RegionWork,
    step: usize,
) -> Result<(), E> {
    let carry = s.body.output_shape().elements;
    scan_arguments(s, p, parent, &mut region.args, step);
    let index = scan_frame(region.checkpoint.as_ref().map_or(0, |c| c.interval), step);
    let frame = &mut region.frames[index];
    scan_body_forward::<FUSED>(&s.body, &region.args, frame)?;
    region.args[..carry]
        .copy_from_slice(&frame.values[s.body.plan.nodes[s.body.plan.output.node].range()]);
    Ok(())
}
#[inline]
fn scan_body_forward<const FUSED: bool>(
    body: &BoundProgram,
    args: &[f64],
    frame: &mut TensorWorkspace,
) -> Result<(), E> {
    if FUSED {
        body.execute_forward(args, frame)
    } else {
        body.forward(args, frame)
    }
}
#[inline]
fn scan_arguments(s: &Scan, p: &Plan, parent: &[f64], args: &mut [f64], step: usize) {
    let carry = s.body.output_shape().elements;
    let start = p.nodes[s.xs.node].start + step * s.row;
    args[carry..carry + s.row].copy_from_slice(&parent[start..start + s.row]);
    let mut offset = carry + s.row;
    for arg in &s.captures {
        let values = &parent[p.nodes[arg.node].range()];
        args[offset..offset + values.len()].copy_from_slice(values);
        offset += values.len();
    }
}
fn compare(op: Comparison, a: f64, b: f64) -> bool {
    match op {
        Comparison::Less => a < b,
        Comparison::LessEqual => a <= b,
        Comparison::Greater => a > b,
        Comparison::GreaterEqual => a >= b,
    }
}
pub(super) fn unary(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Sqrt => x.sqrt(),
    }
}
pub(super) fn derivative(op: UnaryOp, x: f64, y: f64) -> f64 {
    match op {
        UnaryOp::Neg => -1.,
        UnaryOp::Exp => y,
        UnaryOp::Log => 1. / x,
        UnaryOp::Log1p => 1. / (1. + x),
        UnaryOp::Expm1 => x.exp(),
        UnaryOp::Tanh => 1. - y * y,
        UnaryOp::Sin => x.cos(),
        UnaryOp::Cos => -x.sin(),
        UnaryOp::Sqrt => 0.5 / y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_regions_decline_layouts_larger_than_the_storage_they_save() {
        for len in [1, 10_000] {
            let mut b = RegionBuilder::new();
            let shape = Shape::new(vec![len]).unwrap();
            let x = b.input("x", shape.clone()).unwrap();
            let data = b.constant(shape, vec![0.5; len]).unwrap();
            let out = b.binary(BinaryOp::Add, x, data).unwrap();
            let bound = b.compile(out).unwrap().bind(&[]).unwrap();
            let mut workspace = bound.workspace();
            assert_eq!(workspace.adjoint_plan.is_some(), len > 1);
            let (mut out, mut gradient) = (vec![0.; len], vec![0.; len]);
            bound
                .value_and_vjp(
                    &vec![0.5; len],
                    &vec![1.; len],
                    &mut out,
                    &mut gradient,
                    &mut workspace,
                )
                .unwrap();
            assert!(out.iter().all(|&x| x == 1.));
            assert!(gradient.iter().all(|&x| x == 1.));
        }
    }

    #[test]
    fn lexical_matrix_policy_resolves_once_without_changing_existing_nodes() {
        for kernel in [
            MatrixRequest::default(),
            #[cfg(any(feature = "faer", feature = "openblas"))]
            MatrixRequest::standard(MatrixProvider::Auto),
            #[cfg(feature = "faer")]
            MatrixRequest::standard(MatrixProvider::Faer),
            #[cfg(feature = "openblas")]
            MatrixRequest::standard(MatrixProvider::Blas),
        ] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![2, 2]).unwrap()).unwrap();
            let old = b.matmul(x, x).unwrap();
            let mut b = b.with_matrix_request(kernel).unwrap();
            let inherited = b.matmul(x, x).unwrap();
            let explicit = b
                .matmul_with_request(x, x, MatrixRequest::default())
                .unwrap();
            for (v, expected) in [
                (old, ResolvedMatrix::COMPENSATED),
                (inherited, kernel.resolve().unwrap()),
                (explicit, ResolvedMatrix::COMPENSATED),
            ] {
                let Op::Matmul { kernel, .. } = b.nodes[v.node].op else {
                    panic!("matrix node")
                };
                assert_eq!(kernel, expected);
            }
        }
    }

    #[test]
    fn backward_live_intervals_reuse_alias_classes_without_changing_scores_or_errors() {
        let n = 8193;
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![n, 1]).unwrap()).unwrap();
        let scale = b
            .constant(Shape::new(vec![1, 1]).unwrap(), vec![0.5])
            .unwrap();
        let projected = b.matmul(x, scale).unwrap();
        let projected = b.reshape(projected, Shape::new(vec![n]).unwrap()).unwrap();
        let gathered = b
            .gather(
                projected,
                Shape::new(vec![n]).unwrap(),
                (0..n).map(|i| i / 2).collect(),
            )
            .unwrap();
        let squared = b.binary(BinaryOp::Multiply, gathered, gathered).unwrap();
        let mut value = b.unary(UnaryOp::Sqrt, squared).unwrap();
        for _ in 0..64 {
            value = b.unary(UnaryOp::Tanh, value).unwrap();
            value = b.reshape(value, Shape::new(vec![n]).unwrap()).unwrap();
        }
        let program = b.compile(value).unwrap().bind(&[]).unwrap();
        assert!(program.plan.execution.is_none());
        let mut reused = program.workspace();
        assert!(reused.adjoint_plan.as_ref().unwrap().seed.is_some());
        let (starts, slots) =
            super::super::pullback::adjoint_layout(&program.plan, &reused.differential).unwrap();
        assert!(
            reused.adjoints.len()
                + reused
                    .adjoint_plan
                    .as_ref()
                    .unwrap()
                    .seed
                    .as_ref()
                    .unwrap()
                    .len()
                < slots / 8
        );
        let mut dense = program.workspace();
        dense.adjoint_plan.as_mut().unwrap().seed = None;
        dense.adjoint_plan.as_mut().unwrap().starts = starts.into_boxed_slice();
        dense.adjoints = vec![0.; slots];
        dense.adjoint_ranges.clear();
        dense.adjoint_ranges.push(0..slots);
        let x: Vec<_> = (0..n).map(|i| 0.5 + (i % 29) as f64 / 17.).collect();
        let seed: Vec<_> = (0..n).map(|i| if i % 7 == 0 { 0. } else { 0.3 }).collect();
        let (mut out, mut expected, mut score, mut reference) =
            (vec![0.; n], vec![0.; n], vec![0.; n], vec![0.; n]);
        program
            .value_and_vjp(&x, &seed, &mut out, &mut score, &mut reused)
            .unwrap();
        program
            .value_and_vjp(&x, &seed, &mut expected, &mut reference, &mut dense)
            .unwrap();
        assert!(
            out.iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(
            score
                .iter()
                .zip(reference)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(
            allocation_counter::measure(|| {
                program
                    .value_and_vjp(&x, &seed, &mut out, &mut score, &mut reused)
                    .unwrap();
            })
            .count_total,
            0
        );
        out.fill(77.);
        score.fill(88.);
        assert!(
            program
                .value_and_vjp(
                    &vec![0.; n],
                    &vec![1.; n],
                    &mut out,
                    &mut score,
                    &mut reused
                )
                .is_err()
        );
        assert!(out.iter().all(|v| *v == 77.) && score.iter().all(|v| *v == 88.));
        program
            .value_and_vjp(&x, &seed, &mut out, &mut score, &mut reused)
            .unwrap();
    }
}
