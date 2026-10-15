//! Allocation-bounded multinomial NUTS with fixed Euclidean geometry.
//!
//! Binary carries replace recursive subtree construction. The enhanced generalized
//! U-turn checks and biased outer doubling follow Stan 2.38's multinomial NUTS.
//! Every edge is one complete integrator macrostep; internal stages are never
//! candidates. Adaptation belongs to a separate warmup, not this stationary kernel.
#![forbid(unsafe_code)]

use alea_core::target::{EvaluationError, LogDensityGradient, PointState};
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use rand::{Rng, RngExt};

use crate::{
    Hmc,
    config::{SamplerConfigError, TreeDepth},
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
    hmc::{HmcConfigError, HmcError, StepSize},
    integrator::{Integrator, Leapfrog},
    trajectory_weight::Weight,
};

/// Invalid immutable NUTS settings.
#[derive(Debug, thiserror::Error)]
pub enum NutsConfigError {
    /// Invalid step magnitude.
    #[error(transparent)]
    Step(#[from] HmcConfigError),
    /// Unrepresentable doubling depth.
    #[error(transparent)]
    Depth(#[from] SamplerConfigError),
}

/// Fixed step, maximum doubling depth and one-sided energy-increase limit.
#[derive(Debug, Clone, Copy)]
pub struct NutsOptions {
    step_size: StepSize,
    max_depth: TreeDepth,
    max_energy_error: f64,
}

impl NutsOptions {
    /// Validates settings, with energy-increase limit 1000.
    /// # Errors
    /// Rejects nonpositive/nonfinite steps and unrepresentable tree depths.
    pub fn new(step_size: f64, max_depth: usize) -> Result<Self, NutsConfigError> {
        Ok(Self {
            step_size: step_size.try_into()?,
            max_depth: max_depth.try_into()?,
            max_energy_error: 1000.0,
        })
    }
    /// Sets the one-sided `H - H_initial` cutoff (unlike fixed HMC's absolute cutoff).
    /// # Errors
    /// Rejects nonpositive or nonfinite limits.
    pub fn with_max_energy_error(mut self, limit: f64) -> Result<Self, HmcConfigError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(HmcConfigError::EnergyErrorLimit);
        }
        self.max_energy_error = limit;
        Ok(self)
    }
    /// Positive step used for each complete macrostep.
    pub fn step_size(self) -> StepSize {
        self.step_size
    }
    /// Maximum number of doubling attempts.
    pub fn max_depth(self) -> TreeDepth {
        self.max_depth
    }
    /// One-sided energy-increase cutoff.
    pub fn max_energy_error(self) -> f64 {
        self.max_energy_error
    }
}

/// Why trajectory expansion stopped; no implicit numerical/backend conflation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NutsTermination {
    /// A subtree or the merged trajectory turned.
    UTurn,
    /// All configured doublings completed without a turn.
    MaxDepth,
    /// A numerical failure or excessive energy increase stopped expansion.
    Divergence(Divergence),
}

/// Statistics include attempted edges in discarded subtrees.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct NutsTransition {
    /// A non-origin phase was selected (not a second Metropolis test).
    pub accepted: bool,
    /// Average min(1, exp(H_initial-H)) over attempted edges. Unrepresentable
    /// edges contribute zero; finite energy-cutoff edges retain their probability.
    pub acceptance_probability: f64,
    /// Attempted complete macrosteps, including a failing edge.
    pub integration_steps: usize,
    /// Completed valid outer doublings, including an outer merge that turns.
    pub tree_depth: usize,
    /// Candidate states in admitted subtrees, including the origin; zero if
    /// initial momentum/energy could not be represented.
    pub valid_states: usize,
    /// Attempt ordinal of the selected edge; zero denotes the origin.
    pub selected_index: usize,
    /// Initial energy, absent if momentum/energy initialization failed.
    pub initial_energy: Option<f64>,
    /// Energy of the selected phase, not the final attempted endpoint.
    pub proposal_energy: Option<f64>,
    /// Largest finite signed energy increase observed, including origin zero.
    pub max_energy_error: Option<f64>,
    /// Explicit termination classification.
    pub termination: NutsTermination,
}

impl NutsTransition {
    fn new() -> Self {
        Self {
            accepted: false,
            acceptance_probability: 0.0,
            integration_steps: 0,
            tree_depth: 0,
            valid_states: 0,
            selected_index: 0,
            initial_energy: None,
            proposal_energy: None,
            max_energy_error: None,
            termination: NutsTermination::MaxDepth,
        }
    }
}

/// Construction and concrete backend failures. Live state survives errors/panics;
/// RNG consumption and target-side effects are not rolled back.
#[derive(Debug, thiserror::Error)]
pub enum NutsError<E: std::error::Error + 'static> {
    /// Invalid initial state or dimension contract.
    #[error(transparent)]
    Initialization(#[from] HmcError<E>),
    /// Checked aggregate workspace size cannot be represented.
    #[error("NUTS workspace size exceeds addressable storage")]
    StorageOverflow,
    /// Original typed metric/model failure, with the attempted macrostep ordinal.
    #[error("phase failure at integration step {integration_step}: {source}")]
    Phase {
        /// One-based macrostep, or zero during initialization.
        integration_step: usize,
        /// Original phase failure and its source chain.
        #[source]
        source: PhaseError<E>,
    },
}

#[derive(Debug)]
struct Tree<'a, T: LogDensityGradient + ?Sized> {
    left: PhaseState<'a, T>,
    right: PhaseState<'a, T>,
    candidate: PhaseState<'a, T>,
    rho: OwnedBuffer,
    weight: Weight,
    energy: f64,
    index: usize,
}
impl<'a, T: LogDensityGradient + ?Sized> Tree<'a, T> {
    fn new(state: &PhaseState<'a, T>) -> Self {
        Self {
            left: state.clone(),
            right: state.clone(),
            candidate: state.clone(),
            rho: OwnedBuffer::new(state.dimension()),
            weight: Weight::leaf(0.0),
            energy: 0.0,
            index: 0,
        }
    }
    fn leaf(&mut self, state: &PhaseState<'a, T>, energy: f64, log: f64, index: usize) {
        self.left.clone_from(state);
        self.right.clone_from(state);
        self.candidate.clone_from(state);
        self.rho.copy_from_slice(state.momentum());
        self.weight = Weight::leaf(log);
        self.energy = energy;
        self.index = index;
    }
}

#[derive(Debug)]
struct TurnScratch {
    rho: OwnedBuffer,
    left: OwnedBuffer,
    right: OwnedBuffer,
}
mod turning;
impl TurnScratch {
    fn new(d: usize) -> Self {
        Self {
            rho: OwnedBuffer::new(d),
            left: OwnedBuffer::new(d),
            right: OwnedBuffer::new(d),
        }
    }
    fn continues<T: LogDensityGradient + ?Sized, M: EuclideanMetric>(
        &mut self,
        left: &PhaseState<'_, T>,
        right: &PhaseState<'_, T>,
        a: &[f64],
        b: &[f64],
        metric: &M,
    ) -> Result<bool, PhaseError<T::Error>> {
        metric
            .velocity(left.momentum(), &mut self.left)
            .map_err(PhaseError::Metric)?;
        metric
            .velocity(right.momentum(), &mut self.right)
            .map_err(PhaseError::Metric)?;
        turning::continues(&mut self.rho, a, b, &self.left, &self.right)
            .ok_or(PhaseError::Divergence(Divergence::Energy))
    }
}

// Merge two already-valid trees. The left/right endpoint convention is always
// chronological, including backwards construction. Selection precedes turning.
// `new` is consumed scratch: exchange owned snapshots instead of copying their
// coordinate buffers. Its endpoints/candidate must not be read after this merge.
fn merge<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric, R: Rng + ?Sized>(
    old: &mut Tree<'a, T>,
    new: &mut Tree<'a, T>,
    forward: bool,
    biased: bool,
    metric: &M,
    scratch: &mut TurnScratch,
    rng: &mut R,
) -> Result<bool, PhaseError<T::Error>> {
    let (weight, probability) = old.weight.merge_select(new.weight, biased);
    let selected = rng.random::<f64>() < probability;
    let (left, right) = if forward {
        (&*old, &*new)
    } else {
        (&*new, &*old)
    };
    let mut valid = scratch.continues(&left.left, &right.right, &left.rho, &right.rho, metric)?;
    valid &= scratch.continues(
        &left.left,
        &right.left,
        &left.rho,
        right.left.momentum(),
        metric,
    )?;
    valid &= scratch.continues(
        &left.right,
        &right.right,
        &right.rho,
        left.right.momentum(),
        metric,
    )?;
    if selected {
        std::mem::swap(&mut old.candidate, &mut new.candidate);
        old.energy = new.energy;
        old.index = new.index;
    }
    if forward {
        std::mem::swap(&mut old.right, &mut new.right);
    } else {
        std::mem::swap(&mut old.left, &mut new.left);
    }
    for (a, b) in old.rho.iter_mut().zip(new.rho.iter()) {
        *a += b;
    }
    old.weight = weight;
    Ok(valid)
}

/// Fixed-geometry multinomial NUTS with O(dimension * max_depth) owned workspace.
/// Construction allocates; transitions reuse all buffers without heap allocation.
/// Target, metric and integrator remain statically dispatched and fixed throughout
/// each trajectory. Numerical failure discards the unfinished subtree, preserving
/// candidates from earlier valid doublings. Concrete backend errors commit nothing.
///
/// # Examples
/// ```
/// use alea_distributions::Gaussian;
/// use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
/// use alea_mcmc::{Nuts, NutsOptions};
/// use rand::{SeedableRng, rngs::SmallRng};
/// let target = Gaussian::new(2);
/// let mut chain = Nuts::new(&target, OwnedBuffer::new(2),
///     IdentityMetric::new(2), NutsOptions::new(0.2, 8)?)?;
/// let info = chain.step(&mut SmallRng::seed_from_u64(42))?;
/// assert!(info.integration_steps <= 255);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Nuts<'a, T: LogDensityGradient + ?Sized, M, I = Leapfrog> {
    point: PointState<'a, T>,
    workspace: PhaseWorkspace<'a, T>,
    metric: M,
    integrator: I,
    options: NutsOptions,
    root: Tree<'a, T>,
    pending: Tree<'a, T>,
    stack: Vec<Tree<'a, T>>,
    edge: PhaseState<'a, T>,
    scratch: TurnScratch,
}
impl<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric> Nuts<'a, T, M> {
    /// Construct with leapfrog and one initial target evaluation.
    /// # Errors
    /// Rejects empty/mismatched dimensions, invalid initial evaluations or size overflow.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: NutsOptions,
    ) -> Result<Self, NutsError<T::Error>> {
        Self::new_with_integrator(target, position, metric, options, Leapfrog)
    }
}
impl<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric, I: Integrator> Nuts<'a, T, M, I> {
    /// Construct with a sealed symmetric integrator; only macrostep endpoints enter trees.
    /// # Errors
    /// Rejects shape/evaluation failures or unaddressable workspace size.
    pub fn new_with_integrator(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: NutsOptions,
        integrator: I,
    ) -> Result<Self, NutsError<T::Error>> {
        let hmc = Hmc::new_with_integrator(
            target,
            position,
            metric,
            crate::HmcOptions::new(options.step_size.value(), 1).expect("validated step"),
            integrator,
        )?;
        Self::from_parts(hmc.into_parts(), options)
    }
    /// Consume a frozen HMC chain, retaining its cache, metric, step and integrator.
    /// No target reevaluation or RNG draw occurs. This supports both reference M4
    /// and explicitly experimental M4F warmup, without promoting Fisher defaults.
    /// The HMC cutoff is retained in magnitude but becomes one-sided for NUTS.
    /// # Errors
    /// Rejects unaddressable workspace or a changed target/metric contract.
    pub fn from_hmc(
        chain: Hmc<'a, T, M, I>,
        max_depth: TreeDepth,
    ) -> Result<Self, NutsError<T::Error>> {
        let settings = chain.options();
        let options = NutsOptions {
            step_size: settings.step_size(),
            max_depth,
            max_energy_error: settings.max_energy_error(),
        };
        Self::from_parts(chain.into_parts(), options)
    }
    fn from_parts(
        (point, metric, integrator): (PointState<'a, T>, M, I),
        options: NutsOptions,
    ) -> Result<Self, NutsError<T::Error>> {
        let d = point.dimension();
        d.checked_mul(10 * (options.max_depth.value() + 2) + 14)
            .and_then(|n| n.checked_mul(size_of::<f64>()))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(NutsError::StorageOverflow)?;
        let edge = PhaseState::new(point.clone(), OwnedBuffer::new(d)).map_err(|source| {
            NutsError::Phase {
                integration_step: 0,
                source,
            }
        })?;
        if metric.dimension() != d {
            return Err(HmcError::MetricDimension {
                target: d,
                metric: metric.dimension(),
            }
            .into());
        }
        Ok(Self {
            workspace: PhaseWorkspace::new(&point),
            root: Tree::new(&edge),
            pending: Tree::new(&edge),
            stack: (0..options.max_depth.value())
                .map(|_| Tree::new(&edge))
                .collect(),
            scratch: TurnScratch::new(d),
            edge,
            point,
            metric,
            integrator,
            options,
        })
    }
    /// Current coherent point.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }
    /// Read-only stationary geometry.
    pub fn metric(&self) -> &M {
        &self.metric
    }
    /// Immutable trajectory settings.
    pub fn options(&self) -> NutsOptions {
        self.options
    }
    /// Transactional reset with one target evaluation.
    /// # Errors
    /// Invalid shapes/evaluations preserve the old cache.
    pub fn set_position(&mut self, position: &[f64]) -> Result<(), EvaluationError<T::Error>> {
        self.workspace.update_point(&mut self.point, position)
    }

    /// Attempt one random forward/backward doubling trajectory.
    /// # Errors
    /// Concrete metric/model errors preserve the live cache; numerical failures
    /// are reported in the transition. Panics propagate without committing state.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<NutsTransition, NutsError<T::Error>> {
        let mut info = NutsTransition::new();
        let result = self.trajectory(rng, &mut info);
        if let Err(source) = result {
            match source {
                PhaseError::Divergence(reason) => {
                    info.termination = NutsTermination::Divergence(reason)
                }
                source => {
                    return Err(NutsError::Phase {
                        integration_step: info.integration_steps,
                        source,
                    });
                }
            }
        }
        if info.initial_energy.is_some() {
            self.point.clone_from(self.root.candidate.point());
            info.selected_index = self.root.index;
            info.accepted = self.root.index != 0;
            info.proposal_energy = Some(self.root.energy);
        }
        Ok(info)
    }
    fn trajectory<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        info: &mut NutsTransition,
    ) -> Result<(), PhaseError<T::Error>> {
        let mut phase = self.workspace.start(&self.point, &self.metric, rng)?;
        let initial = phase.energy()?;
        phase.save_into(&mut self.edge)?;
        self.root.leaf(&self.edge, initial, 0.0, 0);
        info.valid_states = 1;
        info.initial_energy = Some(initial);
        info.max_energy_error = Some(0.0);
        for depth in 0..self.options.max_depth.value() {
            let forward = rng.random::<bool>();
            self.edge.clone_from(if forward {
                &self.root.right
            } else {
                &self.root.left
            });
            let signed = SignedStep::new(if forward {
                self.options.step_size.value()
            } else {
                -self.options.step_size.value()
            })
            .expect("validated step");
            for leaf in 0..(1_usize << depth) {
                info.integration_steps += 1;
                // Include a failed edge as zero in the acceptance average.
                let count = info.integration_steps as f64;
                info.acceptance_probability *= (count - 1.0) / count;
                let phase = self.workspace.start_from_phase(&self.edge, &self.metric)?;
                let mut phase = phase.step_with(signed, &self.integrator)?;
                let energy = phase.energy()?;
                let delta = energy - initial;
                if !delta.is_finite() {
                    return Err(PhaseError::Divergence(Divergence::Energy));
                }
                info.max_energy_error = Some(info.max_energy_error.unwrap_or(0.0).max(delta));
                info.acceptance_probability += (-delta).min(0.0).exp() / count;
                if delta > self.options.max_energy_error {
                    return Err(PhaseError::Divergence(Divergence::EnergyErrorLimit));
                }
                phase.save_into(&mut self.edge)?;
                self.pending
                    .leaf(&self.edge, energy, -delta, info.integration_steps);
                // Completed siblings are stored by height. Binary carries visit
                // exactly the same internal subtrees as depth-first recursion.
                let height = (leaf + 1).trailing_zeros() as usize;
                for level in 0..height {
                    let valid = merge(
                        &mut self.stack[level],
                        &mut self.pending,
                        forward,
                        false,
                        &self.metric,
                        &mut self.scratch,
                        rng,
                    )?;
                    std::mem::swap(&mut self.pending, &mut self.stack[level]);
                    if !valid {
                        info.termination = NutsTermination::UTurn;
                        return Ok(());
                    }
                }
                if leaf + 1 < (1_usize << depth) {
                    std::mem::swap(&mut self.pending, &mut self.stack[height]);
                }
            }
            // Invalid new subtrees never reach this biased outer selection.
            let valid = merge(
                &mut self.root,
                &mut self.pending,
                forward,
                true,
                &self.metric,
                &mut self.scratch,
                rng,
            )?;
            info.tree_depth += 1;
            info.valid_states += 1_usize << depth;
            if !valid {
                info.termination = NutsTermination::UTurn;
                return Ok(());
            }
        }
        Ok(())
    }
}

impl<T: LogDensityGradient + ?Sized, M: EuclideanMetric, I: Integrator> crate::MarkovChain
    for Nuts<'_, T, M, I>
{
    type Error = NutsError<T::Error>;
    type Transition = NutsTransition;
    fn position(&self) -> &[f64] {
        self.point.position()
    }
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Nuts::step(self, rng)
    }
}

#[cfg(test)]
mod tests;
