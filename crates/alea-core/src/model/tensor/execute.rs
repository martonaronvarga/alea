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
        size_of::<TensorWorkspace>(),
        mul(add(mul(p.slots, 2)?, p.dimension)?, size_of::<f64>())?,
    )?;
    bytes = add(bytes, mul(p.nodes.len(), size_of::<Option<RegionWork>>())?)?;
    bytes = add(bytes, p.nodes.len())?;
    bytes = add(bytes, mul(p.nodes.len(), size_of::<Range<usize>>())?)?;
    bytes = add(bytes, mul(p.data.len(), size_of::<Arc<[f64]>>())?)?;
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
/// Execution-local dense values/adjoints and structured-control-flow frames.
/// Reusable for same-plan immutable data bindings. Outputs are scratch on failure;
/// public evaluation methods stage caller-visible values and gradients.
pub struct TensorWorkspace {
    owner: u64,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    gradient: Vec<f64>,
    differential: Vec<bool>,
    adjoint_ranges: Vec<Range<usize>>,
    // Own the cached binding arrays: address-only identity could be reused after
    // the old snapshot is dropped. Arc identity remains valid until refreshed.
    data: Vec<Arc<[f64]>>,
    regions: Vec<Option<RegionWork>>,
}
impl BoundProgram {
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
        if self.workspace_bytes() > limit {
            return Err(WorkspaceBudgetError {
                required: self.workspace_bytes(),
                limit,
            });
        }
        Ok(self.workspace())
    }
    /// Allocates all value/adjoint/control-flow storage once, outside evaluation.
    pub fn workspace(&self) -> TensorWorkspace {
        self.workspace_with_activity(&vec![true; self.plan.inputs.len()])
    }
    // Specialize derivative activity through region operands, not predicates.
    // Closed data and constant operands must not acquire spurious derivatives.
    fn workspace_with_activity(&self, inputs: &[bool]) -> TensorWorkspace {
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
            if differential[i] && !matches!(node.op, Op::Alias(_)) && node.shape.elements != 0 {
                if let Some(last) = adjoint_ranges
                    .last_mut()
                    .filter(|last| last.end == node.start)
                {
                    last.end = node.range().end;
                } else {
                    adjoint_ranges.push(node.range());
                }
            }
        }
        TensorWorkspace {
            owner: p.owner,
            values,
            adjoints: vec![0.; p.slots],
            gradient: vec![0.; self.dimension()],
            differential,
            adjoint_ranges,
            data: self.data.clone(),
            regions,
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
        self.forward(x, w)?;
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
        self.forward(x, w)?;
        self.reverse(seed, w)?;
        out.copy_from_slice(&w.values[self.plan.nodes[self.plan.output.node].range()]);
        g.copy_from_slice(&w.gradient);
        Ok(())
    }
    fn forward(&self, x: &[f64], w: &mut TensorWorkspace) -> Result<(), E> {
        let p = &self.plan;
        for (i, node) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                continue;
            }
            match &node.op {
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
                Op::Matmul { a, b, m, k, n } => {
                    if node.shape.elements == 0 {
                        continue;
                    }
                    let (before, after) = w.values.split_at_mut(node.start);
                    kernels::matmul_forward(
                        &before[p.nodes[a.node].range()],
                        &before[p.nodes[b.node].range()],
                        &mut after[..node.shape.elements],
                        (*m, *k, *n),
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
                    selected.forward(&region.args, &mut region.frames[choice])?;
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
        w.adjoints[p.nodes[p.output.node].range()].copy_from_slice(seed);
        for (i, node) in p.nodes.iter().enumerate().rev() {
            if !p.active[i] || !w.differential[i] {
                continue;
            }
            match &node.op {
                Op::Input(input) => {
                    let input = &p.inputs[*input];
                    w.gradient[input.start..input.start + input.shape.elements]
                        .copy_from_slice(&w.adjoints[node.range()]);
                }
                Op::Data(_) | Op::Constant(_) | Op::Alias(_) => {}
                Op::Unary(op, a) => {
                    let start = p.nodes[a.node].start;
                    for j in 0..node.shape.elements {
                        let seed = w.adjoints[node.start + j];
                        if seed != 0. {
                            w.adjoints[start + j] += seed
                                * derivative(*op, w.values[start + j], w.values[node.start + j]);
                        }
                    }
                }
                Op::Binary(b) => {
                    let (adjoints, after) = w.adjoints.split_at_mut(node.start);
                    kernels::reverse(
                        b,
                        kernels::Pullback {
                            a: &w.values[p.nodes[b.a.node].range()],
                            r: &w.values[p.nodes[b.b.node].range()],
                            seeds: &after[..node.shape.elements],
                            adjoints,
                            starts: (p.nodes[b.a.node].start, p.nodes[b.b.node].start),
                            active: (w.differential[b.a.node], w.differential[b.b.node]),
                        },
                    );
                }
                Op::Sum(a) => {
                    let seed = w.adjoints[node.start];
                    for j in p.nodes[a.node].range() {
                        w.adjoints[j] += seed;
                    }
                }
                Op::Gather(a, indices) => {
                    let start = p.nodes[a.node].start;
                    for (j, &index) in indices.iter().enumerate() {
                        w.adjoints[start + index] += w.adjoints[node.start + j];
                    }
                }
                Op::Matmul { a, b, m, k, n } => {
                    if node.shape.elements == 0 {
                        continue;
                    }
                    let (adjoints, seeds) = w.adjoints.split_at_mut(node.start);
                    kernels::matmul_reverse(
                        (*m, *k, *n),
                        kernels::Pullback {
                            a: &w.values[p.nodes[a.node].range()],
                            r: &w.values[p.nodes[b.node].range()],
                            seeds: &seeds[..node.shape.elements],
                            adjoints,
                            starts: (p.nodes[a.node].start, p.nodes[b.node].start),
                            active: (w.differential[a.node], w.differential[b.node]),
                        },
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
                    region.seed.copy_from_slice(&w.adjoints[node.range()]);
                    selected.reverse(&region.seed, &mut region.frames[choice])?;
                    let mut offset = 0;
                    for arg in &c.args {
                        for index in p.nodes[arg.node].range() {
                            w.adjoints[index] += region.frames[choice].gradient[offset];
                            offset += 1;
                        }
                    }
                }
                Op::Scan(s) => {
                    let region = w.regions[i].as_mut().expect("scan frames");
                    region.seed.copy_from_slice(&w.adjoints[node.range()]);
                    scan_reverse(s, p, &w.values, &mut w.adjoints, region)?;
                }
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
}
// Select storage policy once per scan, not per ordinary trip. Keeping the loop
// bodies out of the shared interpreter also bounds its code-size cost.
#[inline(never)]
fn scan_forward(
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
            s.body.forward(&region.args, &mut region.frames[step])?;
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
            scan_step(s, p, parent, region, step)?;
        }
        out.copy_from_slice(&region.args[..carry]);
    }
    Ok(())
}

#[inline(never)]
fn scan_reverse(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    adjoints: &mut [f64],
    region: &mut RegionWork,
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
                scan_step(s, p, parent, region, step)?;
            }
            for step in (start..end).rev() {
                reverse_scan_step(s, p, adjoints, region, step, scan_frame(interval, step))?;
            }
            end = start;
        }
    } else {
        for step in (0..s.steps).rev() {
            reverse_scan_step(s, p, adjoints, region, step, step)?;
        }
    }
    for (index, &seed) in p.nodes[s.initial.node].range().zip(&region.seed) {
        adjoints[index] += seed;
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
) -> Result<(), E> {
    let carry = region.seed.len();
    let frame = &mut region.frames[index];
    s.body.reverse(&region.seed, frame)?;
    region.seed.copy_from_slice(&frame.gradient[..carry]);
    let start = p.nodes[s.xs.node].start + step * s.row;
    for j in 0..s.row {
        adjoints[start + j] += frame.gradient[carry + j];
    }
    let mut offset = carry + s.row;
    for arg in &s.captures {
        for index in p.nodes[arg.node].range() {
            adjoints[index] += frame.gradient[offset];
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
fn scan_step(
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
    s.body.forward(&region.args, frame)?;
    region.args[..carry]
        .copy_from_slice(&frame.values[s.body.plan.nodes[s.body.plan.output.node].range()]);
    Ok(())
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
fn unary(op: UnaryOp, x: f64) -> f64 {
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
fn derivative(op: UnaryOp, x: f64, y: f64) -> f64 {
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
