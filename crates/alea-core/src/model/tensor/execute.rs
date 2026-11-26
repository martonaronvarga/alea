//! Dense block execution and reverse propagation, with preallocated region frames.
use super::*;

use super::analysis::{FrameInstance, ResidualGuard};
pub(super) mod linear;
use linear::curvature::{Frame, ReversePair, ScanPair};

// A lexical call borrows its operands from the parent frame. Flat public
// entry points and the joint executor keep their existing input layout.
#[derive(Clone, Copy)]
enum ForwardInputs<'a, 'frame> {
    Flat(&'frame [f64]),
    Lexical(&'a dyn Fn(usize) -> &'frame [f64]),
}

/// Reusable native directional evaluator for one bound tensor region.
/// The structural direction mask is selected at construction.
pub struct TensorJvp<'a> {
    binding: &'a BoundProgram,
    program: linear::Program<'a>,
    work: TensorWorkspace,
    tangents: Vec<f64>,
}

impl TensorJvp<'_> {
    /// Evaluates a primal and its directional derivative. Each direction is
    /// flat row-major in its formal input's shape. Use `None` only for inputs
    /// disabled in the structural mask. Both outputs publish on success.
    /// # Errors
    /// Rejects shape, direction, domain or nonfinite result errors.
    pub fn value_and_jvp(
        &mut self,
        x: &[f64],
        directions: &[Option<&[f64]>],
        out: &mut [f64],
        tangent: &mut [f64],
    ) -> Result<(), E> {
        self.program.value_and_jvp(
            self.binding,
            x,
            directions,
            (out, tangent),
            &mut self.work,
            &mut self.tangents,
        )
    }
}

/// Work counted from the selected plan for a steady, successful VJP.
#[cfg(alea_l42_joint)]
#[doc(hidden)]
pub struct QualificationWork {
    /// Explicit adjoint and gradient initialization and retirement bytes.
    pub initialized_bytes: usize,
    /// Input, seed and staged gradient copy bytes.
    pub copied_bytes: usize,
    /// Pointwise forward and reverse vector passes, excluding reduction.
    pub map_passes: usize,
    /// Additional forward advances from checkpoint replay.
    pub replay_advances: usize,
}

impl BoundProgram {
    pub(in crate::model) fn is_input_view(&self) -> bool {
        // A one-operand lexical call can be erased only when every admitted
        // operation is a formal input or an alias. An unused checked node
        // can still carry a domain error or a native effect.
        self.plan.inputs.len() == 1
            && self.plan.nodes.iter().enumerate().all(|(index, node)| {
                !self.plan.facts.admission(index) || matches!(node.op, Op::Input(0) | Op::Alias(_))
            })
    }

    /// Builds a directional evaluator for the given formal-input mask.
    /// Primal admission runs even if all directions are disabled. Registered
    /// calls require an explicit JVP rule. Conditions follow the admitted
    /// branch, and fixed-trip scans propagate directions through the carry.
    /// # Errors
    /// Rejects an incorrect mask arity or a reachable unsupported operation.
    pub fn jvp(&self, directions: &[bool]) -> Result<TensorJvp<'_>, E> {
        let program = linear::Program::compile(&self.plan, directions)?;
        let tangents = vec![0.; program.tangent_elements()];
        Ok(TensorJvp {
            binding: self,
            program,
            #[cfg(alea_l42_joint)]
            work: self.dense_workspace(),
            #[cfg(not(alea_l42_joint))]
            work: self.workspace(),
            tangents,
        })
    }
}

/// Native potential curvature for a single scalar tensor region.
/// The region result is interpreted as log density in its flattened inputs.
pub struct TensorPotential<'a> {
    binding: &'a BoundProgram,
    program: linear::Program<'a>,
    work: std::cell::RefCell<(TensorWorkspace, Frame)>,
}

/// Differentiates a model call through its exclusive tensor workspace.
pub(in crate::model) struct RegionCurvature<'a> {
    program: linear::Program<'a>,
    frame: Frame,
}

impl<'a> RegionCurvature<'a> {
    pub(in crate::model) fn new(
        binding: &'a BoundProgram,
        activity: &[bool],
        work: &TensorWorkspace,
    ) -> Result<Self, E> {
        let program = linear::Program::compile_curvature_with_activity(&binding.plan, activity)?;
        let frame = Frame::new(&program, work);
        Ok(Self { program, frame })
    }

    pub(in crate::model) fn forward<'input>(
        &mut self,
        binding: &BoundProgram,
        work: &mut TensorWorkspace,
        inputs: &dyn Fn(usize) -> &'input [f64],
        direction: &[f64],
        preparation: Option<&(dyn std::any::Any + Send + Sync)>,
        output: &mut [f64],
    ) -> Result<(), E> {
        // The assembly adapter already admitted value and score at this point.
        // Refresh the one-use tensor residuals with the same binding and recipe.
        binding.forward_region(inputs, work, preparation)?;
        self.program.load_flat(direction, &mut self.frame);
        self.program
            .directional_frame(binding, work, &mut self.frame)?;
        self.program.copy_curvature_result(output, &self.frame);
        Ok(())
    }

    pub(in crate::model) fn reverse(
        &mut self,
        binding: &BoundProgram,
        work: &mut TensorWorkspace,
        seed: &[f64],
        seed_direction: &[f64],
    ) -> Result<&[f64], E> {
        binding.reverse_directional(seed, seed_direction, work, &self.program, &mut self.frame)?;
        Ok(&self.frame.gradient)
    }
}

impl crate::target::TargetSpace for TensorPotential<'_> {
    type Error = E;

    fn dimension(&self) -> usize {
        self.binding.dimension()
    }
}

impl crate::target::LogDensityGradient for TensorPotential<'_> {
    fn is_numerical_error(&self, error: &E) -> bool {
        matches!(error, E::Numerical)
    }

    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, E> {
        let mut value = [0.];
        let mut work = self.work.try_borrow_mut().map_err(|_| E::WorkspaceBusy)?;
        self.binding
            .value_and_vjp(position, &[1.], &mut value, gradient, &mut work.0)?;
        Ok(value[0])
    }
}

impl crate::capability::PotentialHessianVector for TensorPotential<'_> {
    fn potential_hvp(&self, position: &[f64], vector: &[f64], output: &mut [f64]) -> Result<(), E> {
        if output.len() != self.binding.dimension() {
            return Err(E::Shape);
        }
        let mut storage = self.work.try_borrow_mut().map_err(|_| E::WorkspaceBusy)?;
        let (work, frame) = &mut *storage;
        self.program.seeded_directional_vjp(
            self.binding,
            work,
            frame,
            [position, vector],
            [&[1.], &[0.]],
        )?;
        for (out, direction) in output.iter_mut().zip(&frame.gradient) {
            *out = -*direction;
        }
        Ok(())
    }
}

impl BoundProgram {
    /// Creates a reusable native curvature target for a scalar tensor region.
    /// Control flow is differentiated along the selected path. Inputs are the
    /// target coordinates, with no implicit transforms or Jacobian terms.
    /// # Errors
    /// Rejects a nonscalar output or a reachable call without a second rule.
    pub fn potential_hvp_target(&self) -> Result<TensorPotential<'_>, E> {
        if !self.output_shape().axes().is_empty() {
            return Err(E::Shape);
        }
        let program = linear::Program::compile_curvature(&self.plan)?;
        #[cfg(alea_l42_joint)]
        let work = self.dense_workspace();
        #[cfg(not(alea_l42_joint))]
        let work = self.workspace();
        let frame = Frame::new(&program, &work);
        Ok(TensorPotential {
            binding: self,
            program,
            work: std::cell::RefCell::new((work, frame)),
        })
    }
}

struct RegionWork {
    site: Value,
    branch: Option<usize>,
    frames: Vec<TensorWorkspace>,
    args: Vec<f64>,
    seed: Vec<f64>,
    checkpoint: Option<Box<Checkpoint>>,
}
// Checkpoint-only metadata must not enlarge every region slot by a Vec plus
// an interval. The single cold allocation is absent for retained scans/branches.
struct Checkpoint {
    carries: Vec<f64>,
    plan: SegmentPlan,
    // Reused slots must refer to the replayed iteration, not their last occupant.
    iterations: Vec<usize>,
}
impl RegionWork {
    // The exclusive borrow is the context identity: residuals cannot escape
    // this region or survive the next mutable evaluation. Site and instance
    // retain the original call's SSA provenance across checkpoint slot reuse.
    fn residual(&mut self, guard: ResidualGuard, index: usize) -> Result<&mut TensorWorkspace, E> {
        if guard.site != self.site {
            return Err(E::Shape);
        }
        let matches = match guard.instance {
            FrameInstance::Branch(choice) => self.branch == Some(choice) && index == choice,
            FrameInstance::Iteration(step) => self
                .checkpoint
                .as_ref()
                .map_or(index == step, |c| c.iterations.get(index) == Some(&step)),
        };
        if !matches {
            return Err(E::Shape);
        }
        let frame = self.frames.get_mut(index).ok_or(E::Shape)?;
        if !frame.prepared || frame.owner != guard.child {
            return Err(E::Shape);
        }
        Ok(frame)
    }
    fn reverse_residual(
        &mut self,
        guard: ResidualGuard,
        index: usize,
        binding: &BoundProgram,
    ) -> Result<(), E> {
        self.residual(guard, index)?;
        binding.reverse(&self.seed, &mut self.frames[index])
    }
}
// A uniform segment schedule stores one width. The two action streams are
// derived lazily, including the partial last segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SegmentPlan(std::num::NonZeroUsize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayAction {
    Save(usize),
    Restore(usize),
    Advance(usize),
    Reverse(usize),
}

#[cfg(test)]
std::thread_local! {
    static REPLAY_ADVANCES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl SegmentPlan {
    fn new(width: usize) -> Self {
        Self(std::num::NonZeroUsize::new(width).expect("positive segment width"))
    }

    fn width(self) -> usize {
        self.0.get()
    }

    fn frames(self) -> usize {
        // Slot zero preserves the first iteration's separate derivative activity.
        self.width() + 1
    }

    fn segments(self, steps: usize) -> usize {
        steps.div_ceil(self.width())
    }

    fn ranges(self, steps: usize) -> impl DoubleEndedIterator<Item = (usize, Range<usize>)> {
        (0..self.segments(steps)).map(move |slot| {
            let start = slot * self.width();
            (slot, start..start + self.width().min(steps - start))
        })
    }

    fn forward(self, steps: usize) -> impl Iterator<Item = ReplayAction> {
        self.ranges(steps).flat_map(|(slot, range)| {
            std::iter::once(ReplayAction::Save(slot)).chain(range.map(ReplayAction::Advance))
        })
    }

    fn backward(self, steps: usize) -> impl Iterator<Item = ReplayAction> {
        self.ranges(steps).rev().flat_map(|(slot, range)| {
            std::iter::once(ReplayAction::Restore(slot))
                .chain(range.clone().map(ReplayAction::Advance))
                .chain(range.rev().map(ReplayAction::Reverse))
        })
    }
}

fn scan_tape_bytes(
    steps: usize,
    carry: usize,
    body_bytes: usize,
    plan: Option<SegmentPlan>,
) -> Option<usize> {
    let Some(plan) = plan else {
        return steps.checked_mul(body_bytes);
    };
    let saved = plan
        .segments(steps)
        .checked_mul(carry)?
        .checked_mul(size_of::<f64>())?;
    plan.frames()
        .checked_mul(body_bytes.checked_add(size_of::<usize>())?)?
        .checked_add(size_of::<Checkpoint>())?
        .checked_add(saved)
}

fn estimated_width(steps: usize, carry: usize, body_bytes: usize) -> usize {
    let carry = carry as u128 * size_of::<f64>() as u128;
    let square = (steps as u128 * carry) / body_bytes as u128;
    usize::try_from(square.isqrt()).map_or(steps, |width| width.clamp(1, steps))
}

// Preserve automatic checkpointing's admission and pre-reuse sizing basis.
pub(super) fn automatic_replay(
    steps: usize,
    carry: usize,
    body_bytes: usize,
) -> Option<SegmentPlan> {
    if steps < 64 {
        return None;
    }
    let retained = steps.saturating_mul(body_bytes);
    let plan = SegmentPlan::new(estimated_width(steps, carry, body_bytes));
    (retained >= 1024 * 1024
        && scan_tape_bytes(steps, carry, body_bytes, Some(plan))
            .is_some_and(|bytes| bytes < retained / 2))
    .then_some(plan)
}

fn scan_execution_bytes(
    body: &Plan,
    steps: usize,
    carry: usize,
    plan: Option<SegmentPlan>,
) -> Option<usize> {
    let scratch = body
        .dimension
        .checked_add(carry)?
        .checked_mul(size_of::<f64>())?;
    scan_tape_bytes(steps, carry, body.workspace_bytes, plan)?.checked_add(scratch)
}

pub(super) fn bounded_replay(
    body: &Plan,
    steps: usize,
    carry: usize,
    resident: usize,
    max_bytes: usize,
    max_replay_steps: usize,
) -> Result<Option<SegmentPlan>, E> {
    let budget_error = E::ScanReplayBudget {
        max_bytes,
        max_replay_steps,
    };
    let resident_bytes = resident
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(size_of::<RegionWork>()))
        .ok_or(E::Shape)?;
    let limit = max_bytes
        .min(isize::MAX as usize)
        .checked_sub(resident_bytes)
        .ok_or(budget_error)?;
    let fits =
        |plan| scan_execution_bytes(body, steps, carry, plan).filter(|bytes| *bytes <= limit);
    if fits(None).is_some() {
        return Ok(None);
    }
    if steps != 0 && steps <= max_replay_steps {
        let preferred = estimated_width(steps, carry, body.workspace_bytes);
        // Exhaust short chains. Limit the cost of planning long chains by trying
        // small widths plus both neighboring integer widths of the byte estimate.
        if let Some((_, _, plan)) = (1..=steps.min(64))
            .chain([preferred, preferred.saturating_add(1).min(steps)])
            .filter_map(|width| {
                let plan = SegmentPlan::new(width);
                Some((fits(Some(plan))?, width, plan))
            })
            .min_by_key(|&(bytes, width, _)| (bytes, width))
        {
            return Ok(Some(plan));
        }
    }
    Err(E::ScanReplayBudget {
        max_bytes,
        max_replay_steps,
    })
}

// Account for recursively retained frames BEFORE any execution workspace is
// allocated. Includes metadata, not just child value slots: even empty-output
// loops can otherwise overflow the frame vector.
pub(super) fn workspace_bytes(p: &Plan) -> Result<(usize, usize), E> {
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
                add(
                    p.primal_slots,
                    super::storage::adjoint_layout(p, &p.active)?.1,
                )?,
                p.dimension,
            )?,
            size_of::<f64>(),
        )?,
    )?;
    let mut saved = mul(p.slots - p.primal_slots, size_of::<f64>())?;
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
            Op::Cond(c) => {
                for child in [&c.yes.plan, &c.no.plan] {
                    saved = add(saved, child.checkpoint_bytes - child.workspace_bytes)?;
                }
                (
                    add(c.yes.plan.workspace_bytes, c.no.plan.workspace_bytes)?,
                    c.yes.dimension(),
                )
            }
            Op::Scan(s) => {
                let count = s.replay.map_or(s.steps, SegmentPlan::frames);
                saved = add(
                    saved,
                    mul(
                        count,
                        s.body.plan.checkpoint_bytes - s.body.plan.workspace_bytes,
                    )?,
                )?;
                bytes = add(
                    bytes,
                    scan_execution_bytes(&s.body.plan, s.steps, node.shape.elements, s.replay)
                        .ok_or(E::Shape)?,
                )?;
                continue;
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
    Ok((bytes, add(bytes, saved)?))
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkspaceMode {
    Materialized,
    #[cfg(any(alea_l42_joint, test))]
    Ordered,
    #[cfg(any(alea_l42_joint, test))]
    ForwardTiles,
    #[cfg(any(alea_l42_joint, test))]
    Joint(fusion::JointTile),
}
impl WorkspaceMode {
    fn joint(self) -> bool {
        #[cfg(any(alea_l42_joint, test))]
        {
            matches!(self, Self::Joint(_))
        }
        #[cfg(not(any(alea_l42_joint, test)))]
        {
            false
        }
    }
    fn scheduled(self) -> bool {
        #[cfg(any(alea_l42_joint, test))]
        {
            self != Self::Ordered
        }
        #[cfg(not(any(alea_l42_joint, test)))]
        {
            true
        }
    }
    fn tiled_reverse(self) -> bool {
        self == Self::Materialized
    }
}
/// Execution-local values, planned adjoints and structured-control-flow frames.
/// Reusable for same-plan immutable data bindings. Outputs are scratch on failure;
/// public evaluation methods stage caller-visible values and gradients.
pub struct TensorWorkspace {
    owner: u64,
    mode: WorkspaceMode,
    // A residual is usable only after a complete admitted forward. Clear this
    // before any fallible entry, including shape checks and callback unwinds.
    prepared: bool,
    values: Vec<f64>,
    adjoints: Vec<f64>,
    adjoint_plan: Option<Box<AdjointPlan>>,
    gradient: Vec<f64>,
    differential: Vec<bool>,
    adjoint_ranges: Vec<Range<usize>>,
    // Own the cached binding arrays: address-only identity could be reused after
    // the old snapshot is dropped. Arc identity remains valid until refreshed.
    data: Vec<Arc<[f64]>>,
    snapshot: Option<Arc<registration::BindingState>>,
    regions: Vec<Option<RegionWork>>,
    blocks: Vec<Option<registration::Work>>,
}

#[cfg(alea_l42_joint)]
impl TensorWorkspace {
    /// Bytes allocated in the principal primal, adjoint, seed and gradient arenas.
    #[doc(hidden)]
    pub fn qualification_arena_bytes(&self) -> usize {
        (self.values.capacity()
            + self.adjoints.capacity()
            + self.gradient.capacity()
            + self
                .adjoint_plan
                .as_ref()
                .and_then(|plan| plan.seed.as_ref())
                .map_or(0, |seed| seed.len()))
            * size_of::<f64>()
    }
}

impl BoundProgram {
    pub(in crate::model) fn reverse_region<'work>(
        &self,
        seed: &[f64],
        w: &'work mut TensorWorkspace,
    ) -> Result<&'work [f64], E> {
        self.reverse(seed, w)?;
        Ok(&w.gradient)
    }

    #[cfg(test)]
    pub(in crate::model) fn reverse_point(
        &self,
        seed: &[f64],
        scores: &mut [f64],
        w: &mut TensorWorkspace,
    ) -> Result<(), E> {
        if scores.len() != self.dimension() {
            w.prepared = false;
            return Err(E::Shape);
        }
        scores.copy_from_slice(self.reverse_region(seed, w)?);
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
        #[cfg(alea_l42_joint)]
        {
            self.workspace_for(
                &vec![true; self.plan.inputs.len()],
                self.qualification_mode(),
            )
        }
        #[cfg(not(alea_l42_joint))]
        self.workspace_with_activity(&vec![true; self.plan.inputs.len()])
    }
    #[cfg(alea_l42_joint)]
    fn qualification_mode(&self) -> WorkspaceMode {
        let p = &self.plan;
        let Some(joint) = p
            .execution
            .as_ref()
            .and_then(|schedule| schedule.joint(fusion::JointTile::Reference))
        else {
            return WorkspaceMode::Materialized;
        };
        let active = p.facts.activity(&vec![true; p.inputs.len()]);
        let metadata = size_of::<AdjointPlan>() + p.nodes.len() * size_of::<usize>();
        let spans = [false, true].map(|tiled| {
            storage::materialized_adjoints(p, &active, tiled, metadata).map(|a| {
                p.primal_slots
                    .saturating_add(a.slots)
                    .saturating_add(a.seed_slots.unwrap_or(0))
            })
        });
        // The qualification policy is memory-first on this pure, replay-free
        // subset. The stable reference remains selected when it cannot improve
        // that declared objective.
        let compact = joint.primal_slots.saturating_add(joint.adjoints.slots);
        if spans
            .into_iter()
            .all(|span| span.is_ok_and(|span| compact < span))
        {
            WorkspaceMode::Joint(fusion::JointTile::Reference)
        } else {
            WorkspaceMode::Materialized
        }
    }

    /// Construct legal placements for the same compiled graph for qualification.
    #[cfg(any(alea_l42_joint, test))]
    #[doc(hidden)]
    pub fn qualification_workspaces(
        &self,
    ) -> impl Iterator<Item = (&'static str, TensorWorkspace)> + '_ {
        let schedule = self.plan.execution.as_ref();
        let reference = schedule.and_then(|s| s.joint(fusion::JointTile::Reference));
        let modes = [
            ("ordered", reference.map(|_| WorkspaceMode::Ordered)),
            (
                "forward_tiles",
                reference.map(|_| WorkspaceMode::ForwardTiles),
            ),
            ("materialized", Some(WorkspaceMode::Materialized)),
        ];
        let joint = fusion::JointTile::ALL.into_iter().filter_map(move |tile| {
            schedule?.joint(tile)?;
            Some((tile.label(), WorkspaceMode::Joint(tile)))
        });
        modes
            .into_iter()
            .filter_map(|(name, mode)| Some((name, mode?)))
            .chain(joint)
            .map(move |(name, mode)| {
                (
                    name,
                    self.workspace_for(&vec![true; self.plan.inputs.len()], mode),
                )
            })
    }

    /// Read the work of a qualified terminal group without executing it.
    #[cfg(alea_l42_joint)]
    #[doc(hidden)]
    pub fn qualification_work(&self, w: &TensorWorkspace) -> Option<QualificationWork> {
        let p = &self.plan;
        let schedule = p.execution.as_ref().filter(|_| w.owner == p.owner)?;
        let tile = match w.mode {
            WorkspaceMode::Joint(tile) => tile,
            _ => fusion::JointTile::Reference,
        };
        let joint = schedule.joint(tile)?;
        let group = joint.group(schedule);
        let domain = if w.mode.joint() {
            &joint.domain
        } else {
            &group.domain
        };
        let width = domain.tile_width();
        let full = domain.end / width;
        let tail = !domain.end.is_multiple_of(width);
        let tiles = full + usize::from(tail);
        let maps = group.maps().count();
        let transposes = group.maps().filter(|&i| w.differential[i]).count();
        let mut map_passes = if w.mode.scheduled() {
            maps * tiles
        } else {
            maps
        };
        let (initialized, copied) = if w.mode.joint() {
            if w.differential[p.output.node] {
                map_passes += transposes * tiles;
                (
                    joint.adjoints.clear.iter().map(Range::len).sum::<usize>() * full
                        + if tail { joint.adjoints.slots } else { 0 }
                        + group.domain.end,
                    p.dimension
                        + p.nodes
                            .iter()
                            .enumerate()
                            .filter(|(i, node)| {
                                w.differential[*i] && matches!(node.op, Op::Input(_))
                            })
                            .map(|(_, node)| node.shape.elements)
                            .sum::<usize>(),
                )
            } else {
                (0, p.dimension)
            }
        } else {
            let recycled = w.adjoint_plan.as_ref().is_some_and(|a| a.seed.is_some());
            let mut retired = 0;
            let mut copied_seeds = 0;
            let mut loads = 0;
            for (i, node) in p.nodes.iter().enumerate().filter(|(i, _)| p.active[*i]) {
                if matches!(node.op, Op::Input(_)) {
                    loads += node.shape.elements;
                }
                if recycled && w.differential[i] && !matches!(node.op, Op::Alias(_)) {
                    retired += node.shape.elements;
                    if matches!(node.op, Op::Binary(_)) {
                        copied_seeds += node.shape.elements;
                    }
                }
            }
            map_passes += transposes * if w.mode.tiled_reverse() { tiles } else { 1 };
            (
                w.adjoint_ranges.iter().map(Range::len).sum::<usize>() + retired + p.dimension,
                loads + copied_seeds + p.dimension + usize::from(w.differential[p.output.node]),
            )
        };
        Some(QualificationWork {
            initialized_bytes: initialized * size_of::<f64>(),
            copied_bytes: copied * size_of::<f64>(),
            map_passes,
            replay_advances: 0,
        })
    }
    #[cfg(alea_l42_joint)]
    fn dense_workspace(&self) -> TensorWorkspace {
        self.workspace_with_activity(&vec![true; self.plan.inputs.len()])
    }
    // Specialize derivative activity through region operands, not predicates.
    // Closed data and constant operands must not acquire spurious derivatives.
    pub(in crate::model) fn workspace_with_activity(&self, inputs: &[bool]) -> TensorWorkspace {
        self.workspace_for(inputs, WorkspaceMode::Materialized)
    }

    fn workspace_for(&self, inputs: &[bool], mode: WorkspaceMode) -> TensorWorkspace {
        let p = &self.plan;
        let differential = p.facts.activity(inputs);
        let mut regions = Vec::new();
        if !mode.joint() {
            regions.reserve_exact(p.nodes.len());
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
                        Some(RegionWork {
                            site: Value {
                                owner: p.owner,
                                node: i,
                            },
                            branch: None,
                            frames: vec![yes, no],
                            args: vec![0.; c.yes.dimension()],
                            seed: vec![0.; node.shape.elements],
                            checkpoint: None,
                        })
                    }
                    Op::Scan(s) => {
                        let mut inputs =
                            vec![differential[s.initial.node], differential[s.xs.node]];
                        inputs.extend(s.captures.iter().map(|a| differential[a.node]));
                        let count = s.replay.map_or(s.steps, SegmentPlan::frames);
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
                        Some(RegionWork {
                            site: Value {
                                owner: p.owner,
                                node: i,
                            },
                            branch: None,
                            frames,
                            args: vec![0.; s.body.dimension()],
                            seed: vec![0.; node.shape.elements],
                            checkpoint: s.replay.map(|plan| {
                                Box::new(Checkpoint {
                                    carries: vec![0.; plan.segments(s.steps) * node.shape.elements],
                                    plan,
                                    iterations: vec![usize::MAX; count],
                                })
                            }),
                        })
                    }
                    _ => None,
                };
                regions.push(region);
            }
        }
        let schedule = p.execution.as_ref().filter(|_| mode.scheduled());
        let layout_bytes = size_of::<AdjointPlan>() + p.nodes.len() * size_of::<usize>();
        let place_materialized = || {
            storage::materialized_adjoints(
                p,
                &differential,
                schedule.is_some() && mode.tiled_reverse(),
                layout_bytes,
            )
            .expect("compiled tensor adjoint placement is addressable")
        };
        let (value_slots, placement) = match mode {
            #[cfg(any(alea_l42_joint, test))]
            WorkspaceMode::Joint(tile) => {
                let step = schedule
                    .and_then(|s| s.joint(tile))
                    .expect("checked joint step");
                (
                    step.primal_slots,
                    storage::AdjointPlacement {
                        starts: Vec::new(),
                        slots: step.adjoints.slots,
                        seed_slots: None,
                        clear: Vec::new(),
                        planned: false,
                    },
                )
            }
            _ => (p.primal_slots, place_materialized()),
        };
        let mut values = vec![0.; value_slots];
        if !mode.joint() {
            for (i, node) in p.nodes.iter().enumerate() {
                if !p.facts.admission(i) {
                    continue;
                }
                match &node.op {
                    Op::Data(index) => {
                        values[node.primal_range()].copy_from_slice(&self.data[*index])
                    }
                    Op::Constant(data) => values[node.primal_range()].copy_from_slice(data),
                    _ => {}
                }
            }
        }
        let reverse_groups =
            schedule
                .filter(|_| mode.tiled_reverse())
                .map_or_else(Vec::new, |schedule| {
                    schedule
                        .materialized
                        .iter()
                        .map(|step| match step {
                            fusion::Step::Fused(group) => {
                                fusion::reverse_compatible(p, group, &differential)
                            }
                            _ => false,
                        })
                        .collect()
                });
        let mut work = TensorWorkspace {
            owner: p.owner,
            mode,
            prepared: false,
            values,
            adjoints: vec![0.; placement.slots],
            adjoint_plan: placement.planned.then(|| {
                Box::new(AdjointPlan {
                    starts: placement.starts.into_boxed_slice(),
                    seed: placement.seed_slots.map(|n| vec![0.; n].into_boxed_slice()),
                    groups: reverse_groups.into_boxed_slice(),
                })
            }),
            gradient: vec![0.; self.dimension()],
            differential,
            adjoint_ranges: placement.clear,
            data: if mode.joint() {
                Vec::new()
            } else {
                self.data.clone()
            },
            snapshot: None,
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
        };
        if !mode.joint() {
            self.refresh_fixed(&mut work);
        }
        work
    }
    fn same_snapshot(&self, w: &TensorWorkspace) -> bool {
        match (&self.snapshot, &w.snapshot) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    fn refresh_fixed(&self, w: &mut TensorWorkspace) {
        if self.same_snapshot(w) {
            return;
        }
        if let Some(snapshot) = &self.snapshot {
            for (i, fixed) in snapshot.fixed.iter().enumerate() {
                if let Some(binding::Fixed::Value(values)) = fixed {
                    w.values[self.plan.nodes[i].primal_range()].copy_from_slice(values);
                }
            }
        }
        w.snapshot.clone_from(&self.snapshot);
    }
    fn validate(&self, x: &[f64], w: &TensorWorkspace) -> Result<(), E> {
        if x.len() != self.dimension()
            || w.owner != self.plan.owner
            || (!w.mode.joint() && w.values.len() != self.plan.primal_slots)
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
        self.evaluate_request(x, out, w, None, false)
    }
    pub(in crate::model) fn region_result<'work>(&self, w: &'work TensorWorkspace) -> &'work [f64] {
        &w.values[self.result_range(w.mode)]
    }

    pub(in crate::model) fn forward_region<'input>(
        &self,
        inputs: &dyn Fn(usize) -> &'input [f64],
        w: &mut TensorWorkspace,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<(), E> {
        w.prepared = false;
        if w.owner != self.plan.owner || w.mode.joint() {
            return Err(E::Shape);
        }
        // Admit every formal operand, including unused formals, before any
        // operation runs or a callback can observe the call.
        for (index, input) in self.plan.inputs.iter().enumerate() {
            let values = inputs(index);
            if values.len() != input.shape.elements {
                return Err(E::Shape);
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(E::Numerical);
            }
        }
        self.execute_steps(ForwardInputs::Lexical(inputs), w, evaluation_prepared, None)
    }

    #[cfg(test)]
    pub(in crate::model) fn evaluate_with_preparation(
        &self,
        x: &[f64],
        out: &mut [f64],
        w: &mut TensorWorkspace,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<(), E> {
        self.evaluate_request(x, out, w, evaluation_prepared, true)
    }
    fn evaluate_request(
        &self,
        x: &[f64],
        out: &mut [f64],
        w: &mut TensorWorkspace,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
        split: bool,
    ) -> Result<(), E> {
        w.prepared = false;
        if out.len() != self.output_shape().elements {
            return Err(E::Shape);
        }
        self.validate(x, w)?;
        if split && w.mode.joint() {
            // A split query must retain the residuals until its seed arrives.
            *w = self.workspace_with_activity(&vec![true; self.plan.inputs.len()]);
        }
        self.execute_forward(x, w, evaluation_prepared)?;
        out.copy_from_slice(&w.values[self.result_range(w.mode)]);
        Ok(())
    }
    fn result_range(&self, mode: WorkspaceMode) -> Range<usize> {
        if mode.joint() {
            0..1
        } else {
            self.plan.nodes[self.plan.output.node].primal_range()
        }
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
        w.prepared = false;
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
        self.execute_steps(ForwardInputs::Flat(x), w, None, Some(seed))?;
        if !w.mode.joint() {
            self.reverse(seed, w)?;
        }
        out.copy_from_slice(&w.values[self.result_range(w.mode)]);
        g.copy_from_slice(&w.gradient);
        Ok(())
    }
    fn execute_forward(
        &self,
        x: &[f64],
        w: &mut TensorWorkspace,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<(), E> {
        self.execute_steps(ForwardInputs::Flat(x), w, evaluation_prepared, None)
    }

    fn execute_steps(
        &self,
        x: ForwardInputs<'_, '_>,
        w: &mut TensorWorkspace,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
        _seed: Option<&[f64]>,
    ) -> Result<(), E> {
        w.prepared = false;
        if !w.mode.joint() {
            self.refresh_fixed(w);
        }
        let Some(schedule) = self.plan.execution.as_ref().filter(|_| w.mode.scheduled()) else {
            return self.forward(x, w, evaluation_prepared);
        };
        let steps = schedule.materialized.as_slice();
        #[cfg(any(alea_l42_joint, test))]
        let steps = if let WorkspaceMode::Joint(tile) = w.mode {
            std::slice::from_ref(schedule.joint_step(tile).ok_or(E::Shape)?)
        } else {
            steps
        };
        for step in steps {
            match step {
                fusion::Step::Nodes(range) => {
                    if self.has_fixed() {
                        self.forward_nodes::<false, true>(
                            x,
                            w,
                            range.clone(),
                            evaluation_prepared,
                        )?;
                    } else {
                        self.forward_nodes::<false, false>(
                            x,
                            w,
                            range.clone(),
                            evaluation_prepared,
                        )?;
                    }
                }
                fusion::Step::Fused(group) => {
                    // Input finiteness and data bindings were checked before
                    // dispatch. These leaf copies add no admission boundary.
                    for &i in &group.loads {
                        self.forward_nodes::<false, false>(x, w, i..i + 1, None)?;
                    }
                    fusion::forward(&self.plan, group, &mut w.values)?;
                }
                #[cfg(any(alea_l42_joint, test))]
                fusion::Step::Joint(step) => {
                    let ForwardInputs::Flat(x) = x else {
                        return Err(E::Shape);
                    };
                    step.run(
                        self,
                        x,
                        _seed.map(|seed| seed[0]),
                        &mut w.values,
                        &mut w.adjoints,
                        &mut w.gradient,
                    )?;
                }
            }
        }
        w.prepared = !w.mode.joint();
        Ok(())
    }
    // Keep the unscheduled interpreter as one outlined call, as before fusion.
    // In particular, scalar scan bodies must not inline a complete interpreter
    // or re-check the planner on every trip.
    fn forward(
        &self,
        x: ForwardInputs<'_, '_>,
        w: &mut TensorWorkspace,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<(), E> {
        w.prepared = false;
        self.refresh_fixed(w);
        if self.has_fixed() {
            self.forward_nodes::<true, true>(x, w, 0..0, evaluation_prepared)?;
        } else {
            self.forward_nodes::<true, false>(x, w, 0..0, evaluation_prepared)?;
        }
        w.prepared = true;
        Ok(())
    }
    #[inline(always)]
    fn forward_nodes<const FULL: bool, const FIXED: bool>(
        &self,
        x: ForwardInputs<'_, '_>,
        w: &mut TensorWorkspace,
        range: Range<usize>,
        evaluation_prepared: Option<&(dyn std::any::Any + Send + Sync)>,
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
            if FIXED && let Some(fixed) = self.fixed(i) {
                match fixed {
                    binding::Fixed::Value(_) => continue,
                    binding::Fixed::Numerical => return Err(E::Numerical),
                }
            }
            match &node.op {
                Op::Registered(op) => match &**op {
                    registration::Operation::Data(_) => continue,
                    registration::Operation::Native(call) => call.forward(
                        &p.nodes,
                        node,
                        &mut w.values,
                        w.blocks[i].as_mut().expect("registered block workspace"),
                        self.preparation(i),
                        evaluation_prepared.filter(|_| i == p.output.node),
                    )?,
                },
                Op::Input(input) => {
                    let values = match x {
                        ForwardInputs::Flat(x) => {
                            let input = &p.inputs[*input];
                            &x[input.start..input.start + input.shape.elements]
                        }
                        ForwardInputs::Lexical(inputs) => inputs(*input),
                    };
                    w.values[node.primal_range()].copy_from_slice(values);
                    // The complete input was validated before entering forward.
                    continue;
                }
                Op::Data(data) => {
                    if !Arc::ptr_eq(&w.data[*data], &self.data[*data]) {
                        w.values[node.primal_range()].copy_from_slice(&self.data[*data]);
                        w.data[*data] = Arc::clone(&self.data[*data]);
                    }
                    // Both construction and immutable rebinding admit finiteness.
                    continue;
                }
                Op::Constant(_) | Op::Alias(_) => continue,
                Op::Unary(op, a) => {
                    let (inputs, output) =
                        lowering::PrimalView::split(&mut w.values, node.primal_range());
                    kernels::unary_forward(
                        *op,
                        inputs.read(p.nodes[a.node].primal_range()),
                        output,
                    );
                }
                Op::Binary(b) => {
                    let (inputs, output) =
                        lowering::PrimalView::split(&mut w.values, node.primal_range());
                    kernels::forward(
                        b,
                        inputs.read(p.nodes[b.a.node].primal_range()),
                        inputs.read(p.nodes[b.b.node].primal_range()),
                        output,
                    );
                }
                Op::Sum(a) => {
                    w.values[node.primal_start] =
                        kernels::sum_forward(&w.values[p.nodes[a.node].primal_range()]);
                }
                Op::Gather(a, indices) => {
                    let (inputs, output) =
                        lowering::PrimalView::split(&mut w.values, node.primal_range());
                    kernels::gather_forward(
                        inputs.read(p.nodes[a.node].primal_range()),
                        indices,
                        output,
                    );
                }
                Op::Matmul { a, b, .. } => {
                    if node.shape.elements == 0 {
                        continue;
                    }
                    let (inputs, output) =
                        lowering::PrimalView::split(&mut w.values, node.primal_range());
                    kernels::matmul_forward(
                        inputs.read(p.nodes[a.node].primal_range()),
                        inputs.read(p.nodes[b.node].primal_range()),
                        output,
                        p.facts.lowered.matrix(i),
                    );
                }
                Op::Cond(c) => {
                    let choice = if compare(
                        c.comparison,
                        w.values[p.nodes[c.lhs.node].primal_start],
                        w.values[p.nodes[c.rhs.node].primal_start],
                    ) {
                        0
                    } else {
                        1
                    };
                    let selected = if choice == 0 { &c.yes } else { &c.no };
                    let region = w.regions[i].as_mut().expect("condition frame");
                    let mut offset = 0;
                    for arg in &c.args {
                        let values = &w.values[p.nodes[arg.node].primal_range()];
                        region.args[offset..offset + values.len()].copy_from_slice(values);
                        offset += values.len();
                    }
                    region.branch = None;
                    selected.execute_forward(&region.args, &mut region.frames[choice], None)?;
                    region.branch = Some(choice);
                    let output = selected.plan.nodes[selected.plan.output.node].primal_range();
                    w.values[node.primal_range()]
                        .copy_from_slice(&region.frames[choice].values[output]);
                }
                Op::Scan(s) => {
                    let region = w.regions[i].as_mut().expect("scan frames");
                    let (inputs, output) =
                        lowering::PrimalView::split(&mut w.values, node.primal_range());
                    scan_forward(s, p, inputs, output, region)?;
                }
            }
            if alea_math::numeric::first_nonfinite(&w.values[node.primal_range()]).is_some() {
                return Err(E::Numerical);
            }
        }
        Ok(())
    }
    fn check_residual(&self, w: &TensorWorkspace, admitted: bool) -> Result<(), E> {
        let p = &self.plan;
        if !admitted
            || w.owner != p.owner
            || !self.same_snapshot(w)
            || p.facts
                .binding_data
                .iter()
                .any(|&j| !Arc::ptr_eq(&self.data[j], &w.data[j]))
        {
            return Err(E::Shape);
        }
        Ok(())
    }

    fn seed_reverse(&self, seed: &[f64], w: &mut TensorWorkspace) -> Result<(), E> {
        let admitted = std::mem::replace(&mut w.prepared, false);
        self.check_residual(w, admitted)?;
        if seed.len() != self.output_shape().elements {
            return Err(E::Shape);
        }
        if seed.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
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
        Ok(())
    }

    fn reverse(&self, seed: &[f64], w: &mut TensorWorkspace) -> Result<(), E> {
        self.seed_reverse(seed, w)?;
        let p = &self.plan;
        if let Some(schedule) = p.execution.as_ref().filter(|_| w.mode.tiled_reverse()) {
            for (index, step) in schedule.materialized.iter().enumerate().rev() {
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
                        // Leaves contribute no adjoint writes. Publish their
                        // completed adjoints after all tiles have run.
                        for &i in group.loads.iter().rev() {
                            self.reverse_nodes::<false, true, false>(w, i..i + 1, None)?;
                        }
                    }
                    fusion::Step::Fused(group) => self.reverse_nodes::<false, true, false>(
                        w,
                        group.start()..group.end(),
                        None,
                    )?,
                    fusion::Step::Nodes(range) => {
                        self.reverse_nodes::<false, true, false>(w, range.clone(), None)?
                    }
                    #[cfg(any(alea_l42_joint, test))]
                    fusion::Step::Joint(_) => unreachable!("joint step consumes the seed"),
                }
            }
        } else {
            if let Some(layout) = &w.adjoint_plan {
                if layout.seed.is_some() {
                    self.reverse_nodes::<true, true, false>(w, 0..p.nodes.len(), None)?;
                } else {
                    self.reverse_nodes::<false, true, false>(w, 0..p.nodes.len(), None)?;
                }
            } else {
                self.reverse_nodes::<false, false, false>(w, 0..p.nodes.len(), None)?;
            }
        }
        self.check_reverse(w)
    }

    fn reverse_directional(
        &self,
        seed: &[f64],
        seed_direction: &[f64],
        work: &mut TensorWorkspace,
        program: &linear::Program<'_>,
        frame: &mut Frame,
    ) -> Result<(), E> {
        self.seed_reverse(seed, work)?;
        let p = &self.plan;
        frame.adjoints.fill(0.);
        frame.gradient.fill(0.);
        if work.differential[p.output.node] {
            frame.adjoints[p.nodes[p.output.node].range()].copy_from_slice(seed_direction);
        }
        let pair = ReversePair { program, frame };
        let range = 0..p.nodes.len();
        match &work.adjoint_plan {
            Some(layout) if layout.seed.is_some() => {
                self.reverse_nodes::<true, true, true>(work, range, Some(pair))?
            }
            Some(_) => self.reverse_nodes::<false, true, true>(work, range, Some(pair))?,
            None => self.reverse_nodes::<false, false, true>(work, range, Some(pair))?,
        }
        self.check_reverse(work)?;
        if alea_math::numeric::first_nonfinite(&frame.adjoints).is_some() {
            return Err(E::Numerical);
        }
        Ok(())
    }

    fn check_reverse(&self, w: &TensorWorkspace) -> Result<(), E> {
        if alea_math::numeric::first_nonfinite(&w.gradient).is_some()
            || w.adjoint_ranges.iter().any(|range| {
                alea_math::numeric::first_nonfinite(&w.adjoints[range.clone()]).is_some()
            })
        {
            return Err(E::Numerical);
        }
        if let Some(writes) = &self.plan.adjoint_writes
            && w.adjoint_plan.as_ref().is_none_or(|a| a.seed.is_none())
        {
            for i in writes.replaced(&w.differential) {
                let node = &self.plan.nodes[i];
                let start = w.adjoint_plan.as_ref().map_or(node.start, |a| a.starts[i]);
                if alea_math::numeric::first_nonfinite(
                    &w.adjoints[start..start + node.shape.elements],
                )
                .is_some()
                {
                    return Err(E::Numerical);
                }
            }
        }
        Ok(())
    }
    fn reverse_nodes<const REUSE: bool, const PLANNED: bool, const CURVATURE: bool>(
        &self,
        w: &mut TensorWorkspace,
        range: Range<usize>,
        mut pair: Option<ReversePair<'_, '_>>,
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
            // Transpose the builtin rule at its position in the existing reverse
            // schedule. Keep the adjoint layout and seed admission independent.
            use linear::Rule;
            let rule = Rule::at(p, i);
            let copied_seed = if REUSE && matches!(rule, Rule::Binary(_) | Rule::Matrix { .. }) {
                let start = adjoint_start::<PLANNED>(p, starts, i);
                let seed = &mut seed_buffer.as_deref_mut().expect("reused adjoint seed")
                    [..node.shape.elements];
                seed.copy_from_slice(&w.adjoints[start..start + node.shape.elements]);
                Some(&*seed)
            } else {
                None
            };
            let base = adjoint_start::<PLANNED>(p, starts, i);
            let curvature = if CURVATURE {
                pair.as_mut().filter(|_| node.shape.elements != 0)
            } else {
                None
            };
            match rule {
                Rule::Native(call) => {
                    let base = adjoint_start::<PLANNED>(p, starts, i);
                    let work = w.blocks[i].as_mut().expect("registered block workspace");
                    let seed = &w.adjoints[base..base + node.shape.elements];
                    if let Some(pair) = curvature {
                        pair.native(i, call, (&w.values, &w.differential), seed, work)?;
                    }
                    let residuals = p
                        .facts
                        .lowered
                        .call(i)
                        .expect("checked callback")
                        .native_primals(&w.values);
                    call.reverse(&p.nodes, residuals, seed, work, self.preparation(i))?;
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
                Rule::Input(input) => {
                    let input = &p.inputs[input];
                    if let Some(pair) = curvature {
                        pair.frame.gradient[input.start..input.start + input.shape.elements]
                            .copy_from_slice(&pair.frame.adjoints[node.range()]);
                    }
                    w.gradient[input.start..input.start + input.shape.elements].copy_from_slice(
                        &w.adjoints[adjoint_start::<PLANNED>(p, starts, i)
                            ..adjoint_start::<PLANNED>(p, starts, i) + node.shape.elements],
                    );
                }
                Rule::Cond(c) => {
                    let region = w.regions[i].as_mut().expect("condition frame");
                    let choice = region.branch.ok_or(E::Shape)?;
                    let selected = if choice == 0 { &c.yes } else { &c.no };
                    region.seed.copy_from_slice(
                        &w.adjoints[adjoint_start::<PLANNED>(p, starts, i)
                            ..adjoint_start::<PLANNED>(p, starts, i) + node.shape.elements],
                    );
                    let guard = p.facts.guard(
                        Value {
                            owner: p.owner,
                            node: i,
                        },
                        FrameInstance::Branch(choice),
                    )?;
                    let mut curvature_adjoints = if let Some(pair) = curvature
                        && let Some(tape) = pair.frame.regions[i].as_mut()
                    {
                        let seed = &pair.frame.adjoints[node.range()];
                        let program = pair.program.control(i).frame_program(choice);
                        region.residual(guard, choice)?;
                        selected.reverse_directional(
                            &region.seed,
                            seed,
                            &mut region.frames[choice],
                            program,
                            &mut tape.frames[choice],
                        )?;
                        Some((&mut pair.frame.adjoints, &tape.frames[choice].gradient))
                    } else {
                        region.reverse_residual(guard, choice, selected)?;
                        None
                    };
                    let mut offset = 0;
                    for arg in &c.args {
                        for j in 0..p.nodes[arg.node].shape.elements {
                            if w.differential[arg.node] {
                                w.adjoints[adjoint_start::<PLANNED>(p, starts, arg.node) + j] +=
                                    region.frames[choice].gradient[offset];
                                if let Some((target, scores)) = &mut curvature_adjoints {
                                    target[p.nodes[arg.node].start + j] += scores[offset];
                                }
                            }
                            offset += 1;
                        }
                    }
                }
                Rule::Scan(s) => {
                    let region = w.regions[i].as_mut().expect("scan frames");
                    region.seed.copy_from_slice(
                        &w.adjoints[adjoint_start::<PLANNED>(p, starts, i)
                            ..adjoint_start::<PLANNED>(p, starts, i) + node.shape.elements],
                    );
                    let scan_pair = curvature.map(|pair| {
                        let tape = pair.frame.regions[i].as_deref_mut();
                        let (parent, seed) = pair.frame.adjoints.split_at_mut(node.start);
                        ScanPair::new(
                            pair.program,
                            &pair.frame.tangents,
                            parent,
                            &seed[..node.shape.elements],
                            tape,
                            i,
                        )
                    });
                    scan_reverse::<CURVATURE>(
                        s,
                        p,
                        &w.values,
                        &mut w.adjoints,
                        region,
                        (starts, &w.differential),
                        scan_pair,
                    )?;
                }
                builtin => {
                    if let Some(pair) = curvature {
                        let seeds =
                            copied_seed.unwrap_or(&w.adjoints[base..base + node.shape.elements]);
                        builtin.transpose_direction(
                            i,
                            pair.reborrow(),
                            &w.values,
                            &w.differential,
                            seeds,
                        );
                    }
                    builtin.transpose::<PLANNED, false>(
                        i,
                        linear::Reverse {
                            plan: p,
                            primals: &w.values,
                            adjoints: &mut w.adjoints,
                            active: &w.differential,
                            starts,
                        },
                        0..node.shape.elements,
                        copied_seed,
                        p.adjoint_writes
                            .as_ref()
                            .map_or(lowering::AdjointWrite::Add, |writes| {
                                writes.mode(i, &w.differential)
                            }),
                    );
                }
            }
            if REUSE && !matches!(rule, Rule::Alias(_)) {
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
    parent: lowering::PrimalView<'_>,
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
    parent: lowering::PrimalView<'_>,
    out: &mut [f64],
    region: &mut RegionWork,
) -> Result<(), E> {
    let carry = out.len();
    let inputs = p
        .facts
        .lowered
        .call(region.site.node)
        .expect("checked scan")
        .scan_inputs(parent);
    if region.checkpoint.is_none() {
        // Original retained-frame path: no checkpoint index/modulo/lookup per trip.
        for step in 0..s.steps {
            let previous = if step == 0 {
                parent.read(p.nodes[s.initial.node].primal_range())
            } else {
                &region.frames[step - 1].values
                    [s.body.plan.nodes[s.body.plan.output.node].primal_range()]
            };
            region.args[..carry].copy_from_slice(previous);
            scan_arguments(s, inputs, &mut region.args, step);
            scan_body_forward::<FUSED>(&s.body, &region.args, &mut region.frames[step])?;
        }
        if s.steps == 0 {
            out.copy_from_slice(parent.read(p.nodes[s.initial.node].primal_range()));
        } else {
            out.copy_from_slice(
                &region.frames[s.steps - 1].values
                    [s.body.plan.nodes[s.body.plan.output.node].primal_range()],
            );
        }
    } else {
        region.args[..carry].copy_from_slice(parent.read(p.nodes[s.initial.node].primal_range()));
        let plan = region
            .checkpoint
            .as_ref()
            .expect("checkpointed forward")
            .plan;
        for action in plan.forward(s.steps) {
            match action {
                ReplayAction::Save(slot) => {
                    let checkpoint = region.checkpoint.as_mut().expect("checkpointed forward");
                    let start = slot * carry;
                    checkpoint.carries[start..start + carry].copy_from_slice(&region.args[..carry]);
                }
                ReplayAction::Advance(step) => scan_step::<FUSED>(s, inputs, region, step)?,
                _ => unreachable!("forward scan action"),
            }
        }
        out.copy_from_slice(&region.args[..carry]);
    }
    Ok(())
}

#[inline]
fn scan_reverse<const CURVATURE: bool>(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    adjoints: &mut [f64],
    region: &mut RegionWork,
    layout: (&[usize], &[bool]),
    pair: Option<ScanPair<'_, '_>>,
) -> Result<(), E> {
    if s.body.plan.execution.is_none() {
        scan_reverse_with::<false, CURVATURE>(s, p, parent, adjoints, region, layout, pair)
    } else {
        scan_reverse_with::<true, CURVATURE>(s, p, parent, adjoints, region, layout, pair)
    }
}
#[inline(never)]
fn scan_reverse_with<const FUSED: bool, const CURVATURE: bool>(
    s: &Scan,
    p: &Plan,
    parent: &[f64],
    adjoints: &mut [f64],
    region: &mut RegionWork,
    layout: (&[usize], &[bool]),
    pair: Option<ScanPair<'_, '_>>,
) -> Result<(), E> {
    let mut pair = pair;
    if let Some(checkpoint) = region.checkpoint.as_ref() {
        let inputs = p
            .facts
            .lowered
            .call(region.site.node)
            .expect("checked replay")
            .native_primals(parent)
            .inputs;
        let plan = checkpoint.plan;
        let carry = region.seed.len();
        for action in plan.backward(s.steps) {
            match action {
                ReplayAction::Restore(slot) => {
                    let saved = slot * carry;
                    let checkpoint = region.checkpoint.as_ref().expect("replay boundary");
                    region.args[..carry].copy_from_slice(&checkpoint.carries[saved..saved + carry]);
                    if CURVATURE && let Some(pair) = &mut pair {
                        pair.restore_boundary(saved);
                    }
                }
                ReplayAction::Advance(step) => {
                    #[cfg(test)]
                    REPLAY_ADVANCES.with(|count| count.set(count.get() + 1));
                    scan_step::<FUSED>(s, inputs, region, step)?;
                    if CURVATURE && let Some(pair) = &mut pair {
                        pair.replay(p, s, region, step, scan_frame(plan.width(), step))?;
                    }
                }
                ReplayAction::Reverse(step) => reverse_scan_step::<CURVATURE>(
                    s,
                    p,
                    adjoints,
                    region,
                    (step, scan_frame(plan.width(), step)),
                    layout,
                    pair.as_mut(),
                )?,
                _ => unreachable!("backward scan action"),
            }
        }
    } else {
        for step in (0..s.steps).rev() {
            reverse_scan_step::<CURVATURE>(
                s,
                p,
                adjoints,
                region,
                (step, step),
                layout,
                pair.as_mut(),
            )?;
        }
    }
    if layout.1[s.initial.node] {
        if CURVATURE && let Some(pair) = &mut pair {
            pair.accumulate_initial(p, s);
        }
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
fn reverse_scan_step<const CURVATURE: bool>(
    s: &Scan,
    p: &Plan,
    adjoints: &mut [f64],
    region: &mut RegionWork,
    (step, index): (usize, usize),
    layout: (&[usize], &[bool]),
    mut pair: Option<&mut ScanPair<'_, '_>>,
) -> Result<(), E> {
    let carry = region.seed.len();
    let guard = p.facts.guard(region.site, FrameInstance::Iteration(step))?;
    if CURVATURE && let Some(pair) = &mut pair {
        pair.reverse(s, region, guard, index)?;
    } else {
        region.reverse_residual(guard, index, &s.body)?;
    }
    let frame = &region.frames[index];
    region.seed.copy_from_slice(&frame.gradient[..carry]);
    if layout.1[s.xs.node] {
        let start = layout
            .0
            .get(s.xs.node)
            .copied()
            .unwrap_or(p.nodes[s.xs.node].start)
            + step * s.row;
        if CURVATURE && let Some(pair) = &mut pair {
            pair.accumulate(
                p.nodes[s.xs.node].start + step * s.row,
                index,
                carry..carry + s.row,
            );
        }
        for j in 0..s.row {
            adjoints[start + j] += frame.gradient[carry + j];
        }
    }
    let mut offset = carry + s.row;
    for arg in &s.captures {
        let len = p.nodes[arg.node].shape.elements;
        if CURVATURE
            && layout.1[arg.node]
            && let Some(pair) = &mut pair
        {
            pair.accumulate(p.nodes[arg.node].start, index, offset..offset + len);
        }
        for j in 0..len {
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
    inputs: lowering::PrimalInputs<'_>,
    region: &mut RegionWork,
    step: usize,
) -> Result<(), E> {
    let carry = s.body.output_shape().elements;
    scan_arguments(s, inputs, &mut region.args, step);
    let index = scan_frame(
        region.checkpoint.as_ref().map_or(0, |c| c.plan.width()),
        step,
    );
    let frame = &mut region.frames[index];
    if let Some(c) = &mut region.checkpoint {
        c.iterations[index] = usize::MAX;
    }
    scan_body_forward::<FUSED>(&s.body, &region.args, frame)?;
    if let Some(c) = &mut region.checkpoint {
        c.iterations[index] = step;
    }
    region.args[..carry]
        .copy_from_slice(&frame.values[s.body.plan.nodes[s.body.plan.output.node].primal_range()]);
    Ok(())
}
#[inline]
fn scan_body_forward<const FUSED: bool>(
    body: &BoundProgram,
    args: &[f64],
    frame: &mut TensorWorkspace,
) -> Result<(), E> {
    if FUSED {
        body.execute_forward(args, frame, None)
    } else {
        body.forward(ForwardInputs::Flat(args), frame, None)
    }
}
#[inline]
fn scan_arguments(s: &Scan, inputs: lowering::PrimalInputs<'_>, args: &mut [f64], step: usize) {
    let carry = s.body.output_shape().elements;
    let rows = inputs.get(0).expect("checked scan rows");
    let start = step * s.row;
    args[carry..carry + s.row].copy_from_slice(&rows[start..start + s.row]);
    let mut offset = carry + s.row;
    for i in 0..s.captures.len() {
        let values = inputs.get(i + 1).expect("checked scan capture");
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
    fn replay_actions_match_full_small_chain_version_oracle() {
        use ReplayAction::*;
        for steps in 0..=24 {
            for width in 1..=steps.max(1) {
                let plan = SegmentPlan::new(width);
                let mut saved = vec![None; plan.segments(steps)];
                let mut carry = 3_u64;
                let next = |x: u64, i| x.wrapping_mul(7).wrapping_add(i as u64);
                let mut retained = vec![carry];
                for step in 0..steps {
                    retained.push(next(retained[step], step));
                }
                for action in plan.forward(steps) {
                    match action {
                        Save(slot) => {
                            assert_eq!(carry, retained[slot * width]);
                            assert!(saved[slot].replace(carry).is_none());
                        }
                        Advance(step) => {
                            assert_eq!(carry, retained[step]);
                            carry = next(carry, step);
                        }
                        _ => panic!("forward action"),
                    }
                }
                assert_eq!(carry, retained[steps]);
                let mut frames = vec![None; plan.frames()];
                let mut reversed = Vec::new();
                let mut advances = 0;
                for action in plan.backward(steps) {
                    match action {
                        Restore(slot) => carry = saved[slot].unwrap(),
                        Advance(step) => {
                            assert_eq!(carry, retained[step]);
                            let slot = scan_frame(width, step);
                            assert!(frames[slot].replace((step, carry)).is_none());
                            carry = next(carry, step);
                            advances += 1;
                        }
                        Reverse(step) => {
                            assert_eq!(
                                frames[scan_frame(width, step)].take(),
                                Some((step, retained[step]))
                            );
                            reversed.push(step);
                        }
                        _ => panic!("backward action"),
                    }
                }
                assert_eq!(advances, steps);
                assert_eq!(reversed, (0..steps).rev().collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn explicit_scan_policy_matches_small_uniform_budget_oracle() {
        let mut b = RegionBuilder::new();
        let x = b.input("carry", Shape::scalar()).unwrap();
        let row = b.input("row", Shape::scalar()).unwrap();
        let out = b.binary(BinaryOp::Add, x, row).unwrap();
        let p = b.compile(out).unwrap();
        for steps in 1_usize..=24 {
            for carry in [0, 1, 19, 256] {
                let resident = steps + 2 * carry;
                let fixed = size_of::<RegionWork>() + (resident + 2 + carry) * size_of::<f64>();
                let retained = fixed + steps * p.workspace_bytes();
                let (compact, width) = (1..=steps)
                    .map(|width| {
                        let stored = (0..steps).step_by(width).count() * carry * size_of::<f64>();
                        (
                            fixed
                                + size_of::<Checkpoint>()
                                + stored
                                + (width + 1) * (p.workspace_bytes() + size_of::<usize>()),
                            width,
                        )
                    })
                    .min()
                    .unwrap();
                let limited = |bytes, replays| {
                    bounded_replay(&p.plan, steps, carry, resident, bytes, replays)
                };
                assert_eq!(limited(retained, 0).unwrap(), None);
                assert!(matches!(
                    limited(retained - 1, steps - 1),
                    Err(E::ScanReplayBudget { .. })
                ));
                if compact < retained {
                    assert_eq!(
                        limited(compact, steps).unwrap(),
                        Some(SegmentPlan::new(width))
                    );
                }
                assert!(matches!(
                    limited(compact.min(retained) - 1, steps),
                    Err(E::ScanReplayBudget { .. })
                ));
            }
        }
    }

    #[test]
    fn explicit_scan_policy_rejects_executed_nested_scans_but_admits_zero_trips() {
        let mut leaf = RegionBuilder::new();
        let carry = leaf.input("carry", Shape::scalar()).unwrap();
        let _row = leaf.input("row", Shape::scalar()).unwrap();
        let leaf = leaf.compile(carry).unwrap().bind(&[]).unwrap();
        let mut body = RegionBuilder::new();
        let carry = body.input("carry", Shape::scalar()).unwrap();
        let _row = body.input("row", Shape::scalar()).unwrap();
        let xs = body
            .constant(Shape::new(vec![1]).unwrap(), vec![0.])
            .unwrap();
        let out = body.scan(carry, xs, &[], leaf).unwrap();
        let body = body.compile(out).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let initial = b.input("initial", Shape::scalar()).unwrap();
        let xs = b.constant(Shape::new(vec![1]).unwrap(), vec![0.]).unwrap();
        let empty = b.constant(Shape::new(vec![0]).unwrap(), vec![]).unwrap();
        let before = b.nodes.len();
        assert!(matches!(
            b.scan_with_replay_budget(initial, xs, &[], body.clone(), usize::MAX, 1),
            Err(E::ScanReplayUnavailable)
        ));
        assert_eq!(b.nodes.len(), before);
        let out = b
            .scan_with_replay_budget(initial, empty, &[], body, usize::MAX, 0)
            .unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let (mut y, mut g) = ([0.], [0.]);
        p.value_and_vjp(&[2.], &[3.], &mut y, &mut g, &mut p.workspace())
            .unwrap();
        assert_eq!((y, g), ([2.], [3.]));
    }

    #[test]
    fn budgeted_scan_measures_owned_peak_and_replay_and_rejects_stale_or_failed_replay() {
        const TRIPS: usize = 32;
        let mut body = RegionBuilder::new();
        let carry = body.input("carry", Shape::scalar()).unwrap();
        let row = body.input("row", Shape::scalar()).unwrap();
        let gain = body.input("gain", Shape::scalar()).unwrap();
        let log = body.unary(UnaryOp::Log, row).unwrap();
        let product = body.binary(BinaryOp::Multiply, log, gain).unwrap();
        let next = body.binary(BinaryOp::Add, carry, product).unwrap();
        let body = body.compile(next).unwrap().bind(&[]).unwrap();
        let plan = SegmentPlan::new(1);
        let budget = scan_execution_bytes(&body.plan, TRIPS, 1, Some(plan)).unwrap()
            + size_of::<RegionWork>()
            + (TRIPS + 3) * size_of::<f64>();
        let fixture = |retained: bool| {
            let mut b = RegionBuilder::new();
            let initial = b.input("initial", Shape::scalar()).unwrap();
            let gain = b.input("gain", Shape::scalar()).unwrap();
            let rows = b.data("rows", Shape::new(vec![TRIPS]).unwrap()).unwrap();
            let result = if retained {
                b.scan(initial, rows, &[gain], body.clone()).unwrap()
            } else {
                let count = b.nodes.len();
                assert!(matches!(
                    b.scan_with_replay_budget(
                        initial,
                        rows,
                        &[gain],
                        body.clone(),
                        budget,
                        TRIPS - 1
                    ),
                    Err(E::ScanReplayBudget { .. })
                ));
                assert!(matches!(
                    b.scan_with_replay_budget(
                        initial,
                        rows,
                        &[gain],
                        body.clone(),
                        budget - 1,
                        TRIPS
                    ),
                    Err(E::ScanReplayBudget { .. })
                ));
                assert_eq!(b.nodes.len(), count);
                b.scan_with_replay_budget(initial, rows, &[gain], body.clone(), budget, TRIPS)
                    .unwrap()
            };
            (
                b.compile(result)
                    .unwrap()
                    .bind(&[("rows", &[1.; TRIPS])])
                    .unwrap(),
                rows,
                result,
            )
        };
        let (retained, ..) = fixture(true);
        let (compact, rows, output) = fixture(false);
        let measure = |p: &BoundProgram, expected_advances| {
            let mut actual = None;
            REPLAY_ADVANCES.with(|count| count.set(0));
            let stats = allocation_counter::measure(|| {
                let mut w = p.workspace();
                let mut y = [0.];
                let mut g = [0.; 2];
                p.value_and_vjp(&[2., 3.], &[1.], &mut y, &mut g, &mut w)
                    .unwrap();
                assert_eq!((y, g), ([2.], [1., 0.]));
                actual = Some(w);
            });
            REPLAY_ADVANCES.with(|count| assert_eq!(count.get(), expected_advances));
            (actual.unwrap(), stats.bytes_max)
        };
        let (_, retained_peak) = measure(&retained, 0);
        let (mut work, compact_peak) = measure(&compact, TRIPS);
        assert!(compact_peak < retained_peak);
        assert_eq!(
            work.regions[output.node]
                .as_ref()
                .unwrap()
                .checkpoint
                .as_ref()
                .unwrap()
                .plan,
            plan
        );
        eprintln!(
            "scan local budget {budget} B, workspace peak {retained_peak} -> {compact_peak} B, replay advances 0 -> {TRIPS}"
        );
        assert!(matches!(
            compact.jvp(&[true, true]),
            Err(E::ScanReplayUnavailable)
        ));
        assert!(matches!(
            compact.potential_hvp_target(),
            Err(E::ScanReplayUnavailable)
        ));

        let q = [2., 3.];
        let mut value = [0.];
        let mut scores = [17.; 2];
        compact.evaluate(&q, &mut value, &mut work).unwrap();
        let rebound = compact.rebind("rows", &[2.; TRIPS]).unwrap();
        assert!(matches!(
            rebound.reverse_point(&[1.], &mut scores, &mut work),
            Err(E::Shape)
        ));
        assert_eq!(scores, [17.; 2]);
        rebound.evaluate(&q, &mut value, &mut work).unwrap();
        let last_row = rebound.plan.nodes[rows.node].primal_range().end - 1;
        work.values[last_row] = -1.;
        assert!(matches!(
            rebound.reverse_point(&[1.], &mut scores, &mut work),
            Err(E::Numerical)
        ));
        assert_eq!(scores, [17.; 2]);
        work.values[last_row] = 2.;
        rebound
            .value_and_vjp(&q, &[1.], &mut value, &mut scores, &mut work)
            .unwrap();
        let summed = TRIPS as f64 * 2_f64.ln();
        assert!((value[0] - (2. + 3. * summed)).abs() < 1e-12);
        assert_eq!(scores[0], 1.);
        assert!((scores[1] - summed).abs() < 1e-12);
    }

    #[test]
    fn first_writer_replaces_whole_alias_but_keeps_shared_scalar_and_scatter_order() {
        use lowering::AdjointWrite::{Add, Replace};

        for unary_first in [false, true] {
            let fixture = |replacement: bool| {
                let mut b = RegionBuilder::new();
                let shape = Shape::new(vec![3]).unwrap();
                let x = b.input("x", shape.clone()).unwrap();
                let s = b.input("s", Shape::scalar()).unwrap();
                let alias = b.reshape(x, shape).unwrap();
                let unary = |b: &mut RegionBuilder| b.unary(UnaryOp::Neg, alias).unwrap();
                let scatter = |b: &mut RegionBuilder| {
                    let selected = b
                        .gather(x, Shape::new(vec![2]).unwrap(), vec![1, 1])
                        .unwrap();
                    let broadcast = b.binary(BinaryOp::Add, selected, s).unwrap();
                    (selected, broadcast)
                };
                let (neg, gather, broadcast) = if unary_first {
                    let (gather, broadcast) = scatter(&mut b);
                    (unary(&mut b), gather, broadcast)
                } else {
                    let neg = unary(&mut b);
                    let (gather, broadcast) = scatter(&mut b);
                    (neg, gather, broadcast)
                };
                let reduced = b.sum(neg).unwrap();
                let shared = b.sum(broadcast).unwrap();
                let out = b.binary(BinaryOp::Add, reduced, shared).unwrap();
                let mut p = b.compile(out).unwrap();
                if !replacement {
                    Arc::get_mut(&mut p.plan).unwrap().adjoint_writes = None;
                }
                (p.bind(&[]).unwrap(), alias, neg, gather, broadcast, reduced)
            };
            let (p, alias, neg, gather, broadcast, sum) = fixture(true);
            let (reference, ..) = fixture(false);
            let mut work = p.workspace();
            let mut baseline = reference.workspace();
            let modes = p.plan.adjoint_writes.as_ref().unwrap();
            let mode = |v: Value| modes.mode(v.node, &work.differential);
            assert_eq!(mode(neg), if unary_first { Replace } else { Add });
            assert_eq!(mode(sum), Replace);
            assert_eq!(mode(gather), Add);
            assert_eq!(mode(broadcast), Add);
            let initialized =
                |work: &TensorWorkspace| work.adjoint_ranges.iter().map(Range::len).sum::<usize>();
            assert_eq!(
                initialized(&baseline) - initialized(&work),
                5 + 3 * usize::from(unary_first)
            );
            let q = [2., 4., 9., 0.5];
            let (mut value, mut expected) = ([0.], [0.]);
            let (mut score, mut expected_score) = ([0.; 4], [0.; 4]);
            for seed in [1., -3., -0.] {
                work.adjoints.fill(f64::NAN);
                p.value_and_vjp(&q, &[seed], &mut value, &mut score, &mut work)
                    .unwrap();
                reference
                    .value_and_vjp(
                        &q,
                        &[seed],
                        &mut expected,
                        &mut expected_score,
                        &mut baseline,
                    )
                    .unwrap();
                assert_eq!(value[0].to_bits(), expected[0].to_bits());
                assert_eq!(value, [-6.]);
                assert_eq!(score.map(f64::to_bits), expected_score.map(f64::to_bits));
                assert_eq!(score, [-seed, seed, -seed, 2. * seed]);
                assert_eq!(work.values[p.plan.nodes[alias.node].primal_range()], q[..3]);
            }
        }
    }

    #[test]
    fn replacement_normalizes_zero_checks_singular_scores_and_recovers() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let y = b.unary(UnaryOp::Sqrt, x).unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut work = p.workspace();
        assert!(p.plan.adjoint_writes.is_some());
        let (mut value, mut score) = ([17.], [19.]);
        assert!(matches!(
            p.value_and_vjp(&[0.], &[1.], &mut value, &mut score, &mut work),
            Err(E::Numerical)
        ));
        assert_eq!((value, score), ([17.], [19.]));
        assert!(matches!(p.reverse(&[1.], &mut work), Err(E::Shape)));
        for (x, seed) in [(0., -0.), (4., -f64::from_bits(1))] {
            p.value_and_vjp(&[x], &[seed], &mut value, &mut score, &mut work)
                .unwrap();
            assert_eq!(score[0].to_bits(), 0);
        }
        p.value_and_vjp(&[4.], &[2.], &mut value, &mut score, &mut work)
            .unwrap();
        assert_eq!((value, score), ([2.], [0.5]));
    }

    #[test]
    fn primal_workspace_reuses_dead_gathers_but_keeps_live_reshape_views() {
        use crate::capability::PotentialHessianVector;
        const N: usize = 256;
        fn fixture(dense: bool) -> (BoundProgram, Value, Value, Value, Value) {
            let mut b = RegionBuilder::new();
            let shape = Shape::new(vec![N]).unwrap();
            let x = b.input("x", shape.clone()).unwrap();
            let reversed = b.gather(x, shape.clone(), (0..N).rev().collect()).unwrap();
            let alias = b
                .reshape(reversed, Shape::new(vec![N, 1]).unwrap())
                .unwrap();
            let mut gathers = (0..7).scan(reversed, |previous, _| {
                *previous = b
                    .gather(*previous, shape.clone(), (0..N).rev().collect())
                    .unwrap();
                Some(*previous)
            });
            let second = gathers.next().unwrap();
            let fourth = gathers.nth(1).unwrap();
            let last = gathers.last().unwrap();
            let final_sum = b.sum(last).unwrap();
            let earlier_sum = b.sum(alias).unwrap();
            let total = b.binary(BinaryOp::Add, final_sum, earlier_sum).unwrap();
            let mut p = b.compile(total).unwrap();
            if dense {
                let plan = Arc::get_mut(&mut p.plan).unwrap();
                for node in &mut plan.nodes {
                    node.primal_start = node.start;
                }
                plan.primal_slots = plan.slots;
                plan.facts.lowered.place_primals(&plan.nodes);
                (plan.workspace_bytes, plan.checkpoint_bytes) = workspace_bytes(plan).unwrap();
            }
            (p.bind(&[]).unwrap(), alias, reversed, second, fourth)
        }
        let (compact, alias, original, recycled_from, recycled_to) = fixture(false);
        let (dense, ..) = fixture(true);
        let p = &compact.plan;
        assert!(p.execution.is_none());
        let range = |v: Value| p.nodes[v.node].primal_range();
        assert_eq!(range(alias), range(original));
        assert_ne!(range(alias), range(recycled_to));
        assert_eq!(range(recycled_from), range(recycled_to));
        let allocate = |program: &BoundProgram| {
            let mut arena = None;
            let bytes = allocation_counter::measure(|| {
                arena = Some((
                    program
                        .workspace_with_limit(program.workspace_bytes())
                        .unwrap(),
                    (0..N).map(|i| i as f64).collect::<Vec<_>>(),
                    vec![0.],
                ));
            })
            .bytes_current;
            let (mut work, x, mut output) = arena.unwrap();
            let additional = allocation_counter::measure(|| {
                program.evaluate(&x, &mut output, &mut work).unwrap();
            })
            .bytes_max;
            assert_eq!(additional, 0);
            ((work, x, output), bytes)
        };
        let ((mut work, x, mut y), compact_bytes) = allocate(&compact);
        let ((mut reference, _, mut expected), dense_bytes) = allocate(&dense);
        let saved = (reference.values.capacity() - work.values.capacity()) * size_of::<f64>();
        assert!(saved >= 4 * N * size_of::<f64>());
        assert_eq!(dense.workspace_bytes() - compact.workspace_bytes(), saved);
        assert_eq!(dense_bytes - compact_bytes, saved as i64);
        assert!(
            compact
                .workspace_with_limit(compact.workspace_bytes() - 1)
                .is_err()
        );
        assert_eq!(work.adjoints.len(), reference.adjoints.len());
        eprintln!(
            "primal capacity: {} -> {} f64, workspace budget: {} -> {} B, live heap with input and output: {dense_bytes} -> {compact_bytes} B",
            reference.values.capacity(),
            work.values.capacity(),
            dense.workspace_bytes(),
            compact.workspace_bytes()
        );
        let mut g = [0.; N];
        let mut expected_g = [0.; N];
        compact
            .value_and_vjp(&x, &[1.], &mut y, &mut g, &mut work)
            .unwrap();
        dense
            .value_and_vjp(&x, &[1.], &mut expected, &mut expected_g, &mut reference)
            .unwrap();
        assert_eq!(y[0].to_bits(), expected[0].to_bits());
        assert_eq!(y[0], (N * (N - 1)) as f64);
        assert_eq!(g.map(f64::to_bits), expected_g.map(f64::to_bits));
        assert_eq!(g, [2.; N]);
        let mut tangent = [0.];
        compact
            .jvp(&[true])
            .unwrap()
            .value_and_jvp(&x, &[Some(&[1.; N])], &mut y, &mut tangent)
            .unwrap();
        assert_eq!(tangent, [2. * N as f64]);
        compact
            .potential_hvp_target()
            .unwrap()
            .potential_hvp(&x, &[1.; N], &mut g)
            .unwrap();
        assert_eq!(g, [0.; N]);
    }

    #[test]
    fn compact_checkpoint_frames_are_private_and_recover_after_late_admission_failure() {
        const WIDTH: usize = 16;
        const TRIPS: usize = 512;
        let shape = Shape::new(vec![WIDTH]).unwrap();
        let mut body = RegionBuilder::new();
        let mut carry = body.input("carry", shape.clone()).unwrap();
        let row = body.input("row", shape.clone()).unwrap();
        for _ in 0..8 {
            carry = body
                .gather(carry, shape.clone(), (0..WIDTH).rev().collect())
                .unwrap();
        }
        let log = body.unary(UnaryOp::Log, row).unwrap();
        let next = body.binary(BinaryOp::Add, carry, log).unwrap();
        let body = body.compile(next).unwrap().bind(&[]).unwrap();
        assert!(body.plan.primal_slots < body.plan.slots);

        let mut b = RegionBuilder::new();
        let initial = b.input("initial", shape).unwrap();
        let rows = b
            .input("rows", Shape::new(vec![TRIPS, WIDTH]).unwrap())
            .unwrap();
        let scan = b
            .checkpointed_scan(initial, rows, &[], body.clone())
            .unwrap();
        let total = b.sum(scan).unwrap();
        let p = b.compile(total).unwrap().bind(&[]).unwrap();
        let mut work = p.workspace();
        let region = work.regions[scan.node].as_ref().unwrap();
        assert!(region.checkpoint.is_some());
        assert!(
            region
                .frames
                .iter()
                .all(|frame| frame.values.len() == body.plan.primal_slots)
        );
        assert_ne!(
            region.frames[0].values.as_ptr(),
            region.frames[1].values.as_ptr()
        );
        assert_eq!(
            p.plan.checkpoint_bytes - p.workspace_bytes(),
            (p.plan.slots - p.plan.primal_slots) * size_of::<f64>()
                + region.frames.len() * (body.plan.checkpoint_bytes - body.workspace_bytes())
        );
        let mut x = vec![1.; p.dimension()];
        let mut scores = vec![31.; p.dimension()];
        let mut result = [17.];
        *x.last_mut().unwrap() = -1.;
        assert!(matches!(
            p.value_and_vjp(&x, &[1.], &mut result, &mut scores, &mut work),
            Err(E::Numerical)
        ));
        assert_eq!(result, [17.]);
        assert!(scores.iter().all(|&x| x == 31.));
        assert!(matches!(p.reverse(&[1.], &mut work), Err(E::Shape)));
        *x.last_mut().unwrap() = 1.;
        p.value_and_vjp(&x, &[1.], &mut result, &mut scores, &mut work)
            .unwrap();
        assert_eq!(result, [WIDTH as f64]);
        assert!(scores.iter().all(|&x| x == 1.));
    }

    #[test]
    fn residuals_require_complete_same_binding_forward_and_are_consumed_once() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let data = b.data("data", Shape::scalar()).unwrap();
        let log = b.unary(UnaryOp::Log, x).unwrap();
        let out = b.binary(BinaryOp::Multiply, log, data).unwrap();
        let definition = b.compile(out).unwrap();
        let p = definition.bind(&[("data", &[2.])]).unwrap();
        let rebound = definition.bind(&[("data", &[3.])]).unwrap();
        let mut w = p.workspace();
        let mut output = [17.];
        let mut scores = [19.];
        assert!(matches!(
            p.reverse_point(&[1.], &mut scores, &mut w),
            Err(E::Shape)
        ));
        p.evaluate(&[2.], &mut output, &mut w).unwrap();
        assert!(matches!(
            rebound.reverse_point(&[1.], &mut scores, &mut w),
            Err(E::Shape)
        ));
        assert_eq!(scores, [19.]);
        p.evaluate(&[2.], &mut output, &mut w).unwrap();
        p.reverse_point(&[1.], &mut scores, &mut w).unwrap();
        assert_eq!(scores, [1.]);
        assert!(matches!(
            p.reverse_point(&[1.], &mut scores, &mut w),
            Err(E::Shape)
        ));
        // Domain, shape, nonfinite seed and foreign-workspace failures invalidate
        // earlier residuals before any result or score may be published.
        for failure in 0..4 {
            p.evaluate(&[2.], &mut output, &mut w).unwrap();
            let failed = match failure {
                0 => p.evaluate(&[-1.], &mut output, &mut w),
                1 => p.evaluate(&[2.], &mut [], &mut w),
                2 => p.value_and_vjp(&[2.], &[f64::NAN], &mut output, &mut scores, &mut w),
                _ => {
                    let mut b = RegionBuilder::new();
                    let x = b.input("x", Shape::scalar()).unwrap();
                    b.compile(x)
                        .unwrap()
                        .bind(&[])
                        .unwrap()
                        .evaluate(&[2.], &mut output, &mut w)
                }
            };
            assert!(failed.is_err());
            assert!(matches!(
                p.reverse_point(&[1.], &mut scores, &mut w),
                Err(E::Shape)
            ));
            rebound
                .value_and_vjp(&[2.], &[1.], &mut output, &mut scores, &mut w)
                .unwrap();
            assert_eq!(scores, [1.5]);
        }
    }

    #[test]
    fn recorded_branch_residual_rejects_foreign_site_choice_and_kind() {
        let mut yes = RegionBuilder::new();
        let x = yes.input("x", Shape::scalar()).unwrap();
        let log = yes.unary(UnaryOp::Log, x).unwrap();
        let yes = yes.compile(log).unwrap().bind(&[]).unwrap();
        let mut no = RegionBuilder::new();
        let x = no.input("x", Shape::scalar()).unwrap();
        let neg = no.unary(UnaryOp::Neg, x).unwrap();
        let no = no.compile(neg).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
        let out = b.cond(x, zero, Comparison::Greater, &[x], yes, no).unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        p.evaluate(&[2.], &mut [0.], &mut w).unwrap();
        let guard = p.plan.facts.guard(out, FrameInstance::Branch(0)).unwrap();
        let region = w.regions[out.node].as_mut().unwrap();
        assert!(region.residual(guard, 0).is_ok());
        assert!(
            region
                .residual(
                    p.plan.facts.guard(out, FrameInstance::Branch(1)).unwrap(),
                    1
                )
                .is_err()
        );
        let foreign = ResidualGuard {
            site: Value {
                owner: out.owner.wrapping_add(1),
                node: out.node,
            },
            ..guard
        };
        assert!(region.residual(foreign, 0).is_err());
        assert!(
            p.plan
                .facts
                .guard(out, FrameInstance::Iteration(0))
                .is_err()
        );
        // Reverse follows the recorded guard rather than rediscovering it from
        // unrelated parent primal storage. Child residuals retain the old point.
        w.values[p.plan.nodes[x.node].primal_start] = -2.;
        let mut scores = [0.];
        p.reverse_point(&[1.], &mut scores, &mut w).unwrap();
        assert_eq!(scores, [0.5]);
        p.value_and_vjp(&[-2.], &[1.], &mut [0.], &mut scores, &mut w)
            .unwrap();
        assert_eq!(scores, [-1.]);
    }

    #[test]
    fn checkpoint_slots_reject_residuals_from_a_different_iteration() {
        let shape = Shape::new(vec![128]).unwrap();
        let mut body = RegionBuilder::new();
        let carry = body.input("carry", shape.clone()).unwrap();
        let row = body.input("row", shape.clone()).unwrap();
        let out = body.binary(BinaryOp::Add, carry, row).unwrap();
        let body = body.compile(out).unwrap().bind(&[]).unwrap();
        let mut b = RegionBuilder::new();
        let initial = b.input("initial", shape).unwrap();
        let xs = b
            .constant(Shape::new(vec![257, 128]).unwrap(), vec![0.; 257 * 128])
            .unwrap();
        let out = b.checkpointed_scan(initial, xs, &[], body).unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        p.evaluate(&[2.; 128], &mut [0.; 128], &mut w).unwrap();
        let region = w.regions[out.node].as_mut().unwrap();
        let checkpoint = region
            .checkpoint
            .as_ref()
            .expect("substantial checkpointed tape");
        let (slot, &step) = checkpoint
            .iterations
            .iter()
            .enumerate()
            .find(|(_, step)| **step != usize::MAX)
            .unwrap();
        let guard = p
            .plan
            .facts
            .guard(out, FrameInstance::Iteration(step))
            .unwrap();
        assert!(region.residual(guard, slot).is_ok());
        let wrong = p
            .plan
            .facts
            .guard(out, FrameInstance::Iteration(if step == 0 { 1 } else { 0 }))
            .unwrap();
        assert!(region.residual(wrong, slot).is_err());
        assert!(
            p.plan
                .facts
                .guard(out, FrameInstance::Iteration(257))
                .is_err()
        );
        // The failed lookup did not poison the valid next evaluation/replay.
        let mut scores = [0.; 128];
        p.value_and_vjp(&[2.; 128], &[1.; 128], &mut [0.; 128], &mut scores, &mut w)
            .unwrap();
        assert_eq!(scores, [1.; 128]);
    }

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
        let sqrt = b.unary(UnaryOp::Sqrt, squared).unwrap();
        let mut value = sqrt;
        for step in 0..64 {
            value = b.unary(UnaryOp::Tanh, value).unwrap();
            value = b.reshape(value, Shape::new(vec![n]).unwrap()).unwrap();
            // Reshapes now participate in fusion. A live gather keeps this
            // fixture on the ordinary reused-adjoint route.
            if step % 2 == 0 {
                value = b
                    .gather(value, Shape::new(vec![n]).unwrap(), (0..n).collect())
                    .unwrap();
            }
        }
        let program = b.compile(value).unwrap().bind(&[]).unwrap();
        assert!(program.plan.execution.is_none());
        let mut reused = program.workspace();
        assert!(reused.adjoint_plan.as_ref().unwrap().seed.is_some());
        assert_eq!(
            program
                .plan
                .adjoint_writes
                .as_ref()
                .unwrap()
                .mode(sqrt.node, &reused.differential),
            lowering::AdjointWrite::Replace
        );
        let (starts, slots) =
            super::super::storage::adjoint_layout(&program.plan, &reused.differential).unwrap();
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

    #[test]
    fn unreachable_values_own_no_primal_storage_and_live_aliases_keep_their_owner() {
        let n = 2049;
        let build = |with_dead_ops| {
            let mut b = RegionBuilder::new();
            let shape = Shape::new(vec![n]).unwrap();
            let x = b.input("x", shape.clone()).unwrap();
            if with_dead_ops {
                for _ in 0..16 {
                    let _unused = b.unary(UnaryOp::Log, x).unwrap();
                }
            }
            let neg = b.unary(UnaryOp::Neg, x).unwrap();
            let alias = b.reshape(neg, Shape::new(vec![1, n]).unwrap()).unwrap();
            let sum = b.sum(alias).unwrap();
            let p = b.compile(sum).unwrap().bind(&[]).unwrap();
            assert_eq!(
                p.plan.nodes[neg.node].range(),
                p.plan.nodes[alias.node].range()
            );
            p
        };
        let compact = build(true);
        let control = build(false);
        assert!(compact.plan.nodes.len() > control.plan.nodes.len());
        let (mut a, mut b) = (compact.workspace(), control.workspace());
        assert_eq!(a.values.len(), 2 * n + 1);
        assert_eq!(a.values.len(), b.values.len());
        let q = vec![-2.; n];
        let (mut v, mut u) = ([0.], [0.]);
        let (mut g, mut h) = (vec![0.; n], vec![0.; n]);
        compact
            .value_and_vjp(&q, &[1.], &mut v, &mut g, &mut a)
            .unwrap();
        control
            .value_and_vjp(&q, &[1.], &mut u, &mut h, &mut b)
            .unwrap();
        assert_eq!(v, [2. * n as f64]);
        assert_eq!((v, g), (u, h));
    }
}
