//! Allocation-bounded multinomial NUTS with fixed Euclidean geometry.
//!
//! Binary carries replace recursive subtree construction. The enhanced generalized
//! U-turn checks and biased outer doubling follow Stan 2.38's multinomial NUTS.
//! Every edge is one complete integrator macrostep; internal stages are never
//! candidates. Adaptation belongs to a separate warmup, not this stationary kernel.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use alea_core::target::{EvaluationError, LogDensityGradient, PointState};
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use rand::{Rng, RngExt};

use crate::{
    Hmc,
    config::{SamplerConfigError, TreeDepth},
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
    hmc::{HmcConfigError, HmcError, StepSize},
    integrator::{Integrator, Leapfrog},
};

/// Invalid immutable NUTS settings.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NutsConfigError {
    /// Invalid step magnitude.
    #[error(transparent)]
    Step(#[from] HmcConfigError),
    /// Unrepresentable doubling depth.
    #[error(transparent)]
    Depth(#[from] SamplerConfigError),
    /// Relative trajectory step jitter must be finite and lie in `[0, 1)`.
    #[error("NUTS step jitter must be finite and in [0, 1)")]
    StepJitter,
}

/// Base step, optional trajectory jitter, depth and one-sided energy limit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct NutsOptions {
    step_size: StepSize,
    max_depth: TreeDepth,
    max_energy_error: f64,
    step_jitter: f64,
    position_turning: bool,
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
            step_jitter: 0.0,
            position_turning: false,
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
    /// Independently randomize the step once per trajectory, including retention.
    /// The multiplier is uniform in `[1-relative, 1+relative)`. Each complete
    /// trajectory uses one fixed step; this is a fixed mixture of NUTS kernels,
    /// not retained adaptation. Zero preserves the original RNG consumption.
    /// Floating-point extremes clamp to the smallest positive value or `MAX`.
    /// # Errors
    /// Rejects nonfinite relative widths or values outside `[0, 1)`.
    pub fn with_step_size_jitter(mut self, relative: f64) -> Result<Self, NutsConfigError> {
        if !relative.is_finite() || !(0.0..1.0).contains(&relative) {
            return Err(NutsConfigError::StepJitter);
        }
        self.step_jitter = relative;
        Ok(self)
    }
    /// Relative width of the independent trajectory step jitter; default zero.
    pub fn step_size_jitter(self) -> f64 {
        self.step_jitter
    }
    /// Select the endpoint-position U-turn criterion for constant geometry.
    ///
    /// Tests `(q_right - q_left) · p_endpoint` at both ends, including the two
    /// cross-subtree intervals. This is the original NUTS criterion expressed
    /// in mass-whitened coordinates. It avoids maintaining momentum sums.
    /// The default uses Stan-style generalized momentum-sum turning instead.
    /// Changing this setting changes the sampled trajectories and RNG stream.
    pub fn with_position_turning(mut self, enabled: bool) -> Self {
        self.position_turning = enabled;
        self
    }
    /// Whether endpoint-position turning replaces generalized momentum sums.
    pub fn position_turning(self) -> bool {
        self.position_turning
    }
    fn draw_step<R: Rng + ?Sized>(self, rng: &mut R) -> StepSize {
        if self.step_jitter == 0.0 {
            return self.step_size;
        }
        let factor = rng.random_range((1.0 - self.step_jitter)..(1.0 + self.step_jitter));
        (self.step_size.value() * factor)
            .clamp(f64::from_bits(1), f64::MAX)
            .try_into()
            .expect("finite positive jittered step")
    }
    /// Base positive step before optional trajectory jitter.
    pub fn step_size(self) -> StepSize {
        self.step_size
    }
    /// Maximum number of doubling attempts.
    pub fn max_depth(self) -> TreeDepth {
        self.max_depth
    }
    /// One-sided energy-increase cutoff.
    #[must_use]
    pub fn max_energy_error(self) -> f64 {
        self.max_energy_error
    }
}

/// Why trajectory expansion stopped; no implicit numerical/backend conflation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NutsTermination {
    /// A subtree or the merged trajectory turned.
    UTurn,
    /// All configured doublings completed without a turn.
    MaxDepth,
    /// A numerical failure or excessive energy increase stopped expansion.
    Divergence(Divergence),
}

/// Statistics include attempted edges in discarded subtrees.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
pub struct NutsTransition {
    /// Actual positive step used throughout this trajectory, including jitter.
    pub step_size: StepSize,
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
    fn new(step_size: StepSize) -> Self {
        Self {
            step_size,
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
#[non_exhaustive]
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

mod state;
use state::StatePool;
mod weight;
#[cfg(test)]
use crate::trajectory_weight::Weight;
use weight::CachedWeight;

#[derive(Debug)]
struct Tree {
    left: usize,
    right: usize,
    candidate: usize,
    active: bool,
    rho: OwnedBuffer,
    rho_bound: f64,
    weight: CachedWeight,
    energy: f64,
    index: usize,
}
impl Tree {
    fn new(dimension: usize) -> Self {
        Self {
            left: 0,
            right: 0,
            candidate: 0,
            active: false,
            rho: OwnedBuffer::new(dimension),
            rho_bound: 0.0,
            weight: CachedWeight::leaf(0.0),
            energy: 0.0,
            index: 0,
        }
    }
    fn leaf<const POSITION: bool, T: LogDensityGradient + ?Sized>(
        &mut self,
        state: usize,
        pool: &mut StatePool<'_, T>,
        energy: f64,
        weight: CachedWeight,
        index: usize,
    ) {
        debug_assert!(!self.active);
        self.left = state;
        self.right = state;
        self.candidate = state;
        for _ in 0..3 {
            pool.retain(state);
        }
        self.active = true;
        if !POSITION {
            self.rho.copy_from_slice(pool.get(state).phase.momentum());
            self.rho_bound = pool.get(state).momentum_scale;
        }
        self.weight = weight;
        self.energy = energy;
        self.index = index;
    }
    fn release<T: LogDensityGradient + ?Sized>(&mut self, pool: &mut StatePool<'_, T>) {
        debug_assert!(self.active);
        for index in [self.left, self.right, self.candidate] {
            pool.release(index);
        }
        self.active = false;
    }
}

#[derive(Debug)]
struct TurnScratch {
    rho: OwnedBuffer,
}
mod turning;
impl TurnScratch {
    fn new(d: usize) -> Self {
        Self {
            rho: OwnedBuffer::new(d),
        }
    }
    fn continues<T: LogDensityGradient + ?Sized>(
        &mut self,
        left: usize,
        right: usize,
        a: &[f64],
        b: &[f64],
        bound: f64,
        pool: &StatePool<'_, T>,
    ) -> Result<bool, PhaseError<T::Error>> {
        let (l, r) = (pool.get(left), pool.get(right));
        if let (Some(ls), Some(rs)) = (l.turning_scale, r.turning_scale) {
            if let Some(result) =
                turning::bounded::<false>(a, b, &l.velocity, &r.velocity, [bound, ls, rs])
            {
                return Ok(result);
            }
            return turning::cached(&mut self.rho, a, b, &l.velocity, &r.velocity, ls, rs)
                .ok_or(PhaseError::Divergence(Divergence::Energy));
        }
        turning::continues(
            &mut self.rho,
            a,
            b,
            &pool.get(left).velocity,
            &pool.get(right).velocity,
        )
        .ok_or(PhaseError::Divergence(Divergence::Energy))
    }
    fn position_continues<T: LogDensityGradient + ?Sized>(
        &mut self,
        left: usize,
        right: usize,
        pool: &StatePool<'_, T>,
    ) -> Result<bool, PhaseError<T::Error>> {
        let (l, r) = (pool.get(left), pool.get(right));
        if let (Some(ls), Some(rs)) = (l.turning_scale, r.turning_scale) {
            if let Some(result) = turning::bounded::<true>(
                l.phase.point().position(),
                r.phase.point().position(),
                l.phase.momentum(),
                r.phase.momentum(),
                [2.0 * l.position_scale.max(r.position_scale), ls, rs],
            ) {
                return Ok(result);
            }
            return turning::cached_position(
                &mut self.rho,
                l.phase.point().position(),
                r.phase.point().position(),
                l.phase.momentum(),
                r.phase.momentum(),
                ls,
                rs,
            )
            .ok_or(PhaseError::Divergence(Divergence::Energy));
        }
        turning::position(
            &mut self.rho,
            l.phase.point().position(),
            r.phase.point().position(),
            l.phase.momentum(),
            r.phase.momentum(),
        )
        .ok_or(PhaseError::Divergence(Divergence::Energy))
    }

    fn certified_three<const POSITION: bool, T: LogDensityGradient + ?Sized>(
        &self,
        left: &Tree,
        right: &Tree,
        pool: &StatePool<'_, T>,
    ) -> Option<bool> {
        if self.rho.len() < 16 || (left.left == left.right && right.left == right.right) {
            return None;
        }
        let slots = [
            pool.get(left.left),
            pool.get(left.right),
            pool.get(right.left),
            pool.get(right.right),
        ];
        let [l, m, n, r] = slots;
        let vs = [
            l.turning_scale?,
            m.turning_scale?,
            n.turning_scale?,
            r.turning_scale?,
        ];
        let (x, v, bounds) = if POSITION {
            (
                slots.map(|s| s.phase.point().position()),
                slots.map(|s| s.phase.momentum()),
                [
                    2.0 * l.position_scale.max(r.position_scale),
                    2.0 * l.position_scale.max(n.position_scale),
                    2.0 * m.position_scale.max(r.position_scale),
                ],
            )
        } else {
            (
                [
                    &*left.rho,
                    &*right.rho,
                    n.phase.momentum(),
                    m.phase.momentum(),
                ],
                slots.map(|s| &*s.velocity),
                [
                    2.0 * left.rho_bound.max(right.rho_bound),
                    2.0 * left.rho_bound.max(n.momentum_scale),
                    2.0 * right.rho_bound.max(m.momentum_scale),
                ],
            )
        };
        turning::bounded_three::<POSITION>(
            x,
            v,
            [
                [bounds[0], vs[0], vs[3]],
                [bounds[1], vs[0], vs[2]],
                [bounds[2], vs[1], vs[3]],
            ],
        )
    }
}

// Chronological endpoints, including backwards construction. Merge consumes new's
// handles; only index/reference-count updates move endpoints and candidates.
fn merge<const POSITION: bool, T: LogDensityGradient + ?Sized, R: Rng + ?Sized>(
    old: &mut Tree,
    new: &mut Tree,
    forward: bool,
    biased: bool,
    pool: &mut StatePool<'_, T>,
    scratch: &mut TurnScratch,
    rng: &mut R,
) -> Result<bool, PhaseError<T::Error>> {
    debug_assert!(old.active && new.active);
    let (weight, probability) = old.weight.merge_select(new.weight, biased);
    let selected = probability >= 1.0 || (probability > 0.0 && rng.random::<f64>() < probability);
    let (left, right) = if forward {
        (&*old, &*new)
    } else {
        (&*new, &*old)
    };
    let certified = scratch.certified_three::<POSITION, T>(left, right, pool);
    let mut valid = if let Some(valid) = certified {
        valid
    } else if POSITION {
        scratch.position_continues(left.left, right.right, pool)?
    } else {
        scratch.continues(
            left.left,
            right.right,
            &left.rho,
            &right.rho,
            2.0 * left.rho_bound.max(right.rho_bound),
            pool,
        )?
    };
    // Leaf pairs have three identical tests. For larger trees keep eager numeric
    // admission: a turn must not conceal overflow in a different cross criterion.
    if certified.is_none() && (left.left != left.right || right.left != right.right) {
        if POSITION {
            valid &= scratch.position_continues(left.left, right.left, pool)?;
            valid &= scratch.position_continues(left.right, right.right, pool)?;
        } else {
            valid &= scratch.continues(
                left.left,
                right.left,
                &left.rho,
                pool.get(right.left).phase.momentum(),
                2.0 * left.rho_bound.max(pool.get(right.left).momentum_scale),
                pool,
            )?;
            valid &= scratch.continues(
                left.right,
                right.right,
                &right.rho,
                pool.get(left.right).phase.momentum(),
                2.0 * right.rho_bound.max(pool.get(left.right).momentum_scale),
                pool,
            )?;
        }
    }
    if selected {
        pool.replace(&mut old.candidate, new.candidate);
        old.energy = new.energy;
        old.index = new.index;
    }
    if forward {
        pool.replace(&mut old.right, new.right);
    } else {
        pool.replace(&mut old.left, new.left);
    }
    if !POSITION {
        for (a, b) in old.rho.iter_mut().zip(new.rho.iter()) {
            *a += b;
        }
        old.rho_bound = 2.0 * old.rho_bound.max(new.rho_bound);
    }
    old.weight = weight;
    new.release(pool);
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
/// use alea::distributions::Gaussian;
/// use alea::math::{buffer::OwnedBuffer, metric::IdentityMetric};
/// use alea::sampling::{Nuts, NutsOptions};
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
    root: Box<Tree>,
    pending: Box<Tree>,
    // Binary carries swap pointers; moving each buffer-owning Tree costs more.
    #[allow(
        clippy::vec_box,
        reason = "binary-carry tree swaps must move only a pointer"
    )]
    stack: Vec<Box<Tree>>,
    pool: StatePool<'a, T>,
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
    /// No target reevaluation or RNG draw occurs. This supports both covariance
    /// and Fisher warmup, preserving the selected geometry.
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
            step_jitter: 0.0,
            position_turning: false,
        };
        Self::from_parts(chain.into_parts(), options)
    }
    fn from_parts(
        (point, metric, integrator): (PointState<'a, T>, M, I),
        options: NutsOptions,
    ) -> Result<Self, NutsError<T::Error>> {
        let d = point.dimension();
        d.checked_mul(13 * options.max_depth.value() + 24)
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
            root: Box::new(Tree::new(d)),
            pending: Box::new(Tree::new(d)),
            stack: (0..options.max_depth.value())
                .map(|_| Box::new(Tree::new(d)))
                .collect(),
            scratch: TurnScratch::new(d),
            // Before allocating a leaf, pending is inactive and at most depth-1
            // stack trees plus the root own three handles each. One edge and one
            // writable slot need two more. Sharing usually lowers the live count.
            pool: StatePool::new(&edge, 3 * options.max_depth.value() + 2),
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
    #[must_use]
    pub fn metric(&self) -> &M {
        &self.metric
    }
    /// Immutable trajectory settings.
    pub fn options(&self) -> NutsOptions {
        self.options
    }

    pub(crate) fn set_step_size(&mut self, step: StepSize) {
        self.options.step_size = step;
    }

    pub(crate) fn metric_mut(&mut self) -> &mut M {
        &mut self.metric
    }

    pub(crate) fn integrator_mut(&mut self) -> &mut I {
        &mut self.integrator
    }

    /// Consume frozen NUTS geometry and its coherent cache to initialize static
    /// HMC without target reevaluation or RNG draws. The energy-limit magnitude
    /// is preserved and becomes HMC's symmetric absolute-error cutoff.
    /// Static HMC uses the base step; optional NUTS trajectory jitter is dropped.
    pub fn into_hmc(self, integration_steps: std::num::NonZeroUsize) -> Hmc<'a, T, M, I> {
        let options =
            crate::HmcOptions::new(self.options.step_size.value(), integration_steps.get())
                .expect("validated NUTS step and nonzero count")
                .with_max_energy_error(self.options.max_energy_error)
                .expect("validated NUTS cutoff");
        Hmc::from_cached(self.point, self.metric, options, self.integrator)
    }

    pub(crate) fn replace_metric(&mut self, metric: M) -> Result<(), NutsError<T::Error>> {
        if metric.dimension() != self.point.dimension() {
            return Err(HmcError::MetricDimension {
                target: self.point.dimension(),
                metric: metric.dimension(),
            }
            .into());
        }
        self.metric = metric;
        Ok(())
    }

    /// A one-macrostep warmup probe with the NUTS one-sided energy policy.
    /// Only scratch and RNG change; the live cache and settings never change.
    pub(crate) fn probe<R: Rng + ?Sized>(
        &mut self,
        step: StepSize,
        rng: &mut R,
    ) -> Result<f64, NutsError<T::Error>> {
        let result = (|| {
            let mut phase = self.workspace.start(&self.point, &self.metric, rng)?;
            let initial = phase.energy()?;
            let mut phase = phase.step_with(
                SignedStep::new(step.value()).expect("validated positive step"),
                &self.integrator,
            )?;
            let delta = phase.energy()? - initial;
            Ok::<_, PhaseError<T::Error>>(
                if delta.is_finite() && delta <= self.options.max_energy_error {
                    (-delta).min(0.0).exp()
                } else {
                    0.0
                },
            )
        })();
        match result {
            Ok(probability) => Ok(probability),
            Err(PhaseError::Divergence(_)) => Ok(0.0),
            Err(source) => Err(NutsError::Phase {
                integration_step: 1,
                source,
            }),
        }
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
        self.step_impl::<false, R>(rng)
            .map(|(transition, _)| transition)
    }
    pub(crate) fn step_with_symmetric_acceptance<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<(NutsTransition, f64), NutsError<T::Error>> {
        self.step_impl::<true, R>(rng)
    }
    fn step_impl<const SYMMETRIC: bool, R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<(NutsTransition, f64), NutsError<T::Error>> {
        let mut info = NutsTransition::new(self.options.draw_step(rng));
        let mut sum_acceptance = 0.0;
        let mut symmetric = 0.0;
        let result = if self.options.position_turning {
            self.trajectory::<SYMMETRIC, true, R>(
                rng,
                &mut info,
                &mut sum_acceptance,
                &mut symmetric,
            )
        } else {
            self.trajectory::<SYMMETRIC, false, R>(
                rng,
                &mut info,
                &mut sum_acceptance,
                &mut symmetric,
            )
        };
        if info.integration_steps > 0 {
            info.acceptance_probability = sum_acceptance / info.integration_steps as f64;
            if SYMMETRIC {
                symmetric /= info.integration_steps as f64;
            }
        }
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
            self.point
                .clone_from(self.pool.get(self.root.candidate).phase.point());
            info.selected_index = self.root.index;
            info.accepted = self.root.index != 0;
            info.proposal_energy = Some(self.root.energy);
        }
        Ok((info, symmetric))
    }
    fn trajectory<const SYMMETRIC: bool, const POSITION: bool, R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        info: &mut NutsTransition,
        sum_acceptance: &mut f64,
        sum_symmetric: &mut f64,
    ) -> Result<(), PhaseError<T::Error>> {
        // Abandon every scratch handle, including an interrupted trajectory's
        // edge, before rebuilding counts. The externally visible point is separate.
        self.root.active = false;
        self.pending.active = false;
        for tree in &mut self.stack {
            tree.active = false;
        }
        self.pool.reset();
        let mut edge = self.pool.allocate();
        let slot = self.pool.vacant(edge);
        let phase = self.workspace.start(&self.point, &self.metric, rng)?;
        let admission =
            phase.finish_cached_into::<POSITION>(&mut slot.phase, &mut slot.velocity)?;
        let initial = admission.energy;
        self.pool.admit_turning::<POSITION>(edge, admission.scales);
        self.root
            .leaf::<POSITION, T>(edge, &mut self.pool, initial, CachedWeight::leaf(0.0), 0);
        info.valid_states = 1;
        info.initial_energy = Some(initial);
        info.max_energy_error = Some(0.0);
        for depth in 0..self.options.max_depth.value() {
            let forward = rng.random::<bool>();
            let endpoint = if forward {
                self.root.right
            } else {
                self.root.left
            };
            self.pool.replace(&mut edge, endpoint);
            let signed = SignedStep::new(if forward {
                info.step_size.value()
            } else {
                -info.step_size.value()
            })
            .expect("validated step");
            for leaf in 0..(1_usize << depth) {
                info.integration_steps += 1;
                // Increment before attempting the edge: failures contribute zero.
                let phase = self.workspace.step_from_phase(
                    &self.pool.get(edge).phase,
                    &self.metric,
                    signed,
                    &self.integrator,
                )?;
                self.pool.release(edge);
                edge = self.pool.allocate();
                let slot = self.pool.vacant(edge);
                let admission =
                    phase.finish_cached_into::<POSITION>(&mut slot.phase, &mut slot.velocity)?;
                let energy = admission.energy;
                self.pool.admit_turning::<POSITION>(edge, admission.scales);
                let delta = energy - initial;
                if !delta.is_finite() {
                    return Err(PhaseError::Divergence(Divergence::Energy));
                }
                info.max_energy_error = Some(info.max_energy_error.unwrap_or(0.0).max(delta));
                let weight = CachedWeight::leaf(-delta);
                *sum_acceptance += weight.acceptance();
                if SYMMETRIC {
                    let weight = (-delta.abs()).exp();
                    *sum_symmetric += 2.0 * weight / (1.0 + weight);
                }
                if delta > self.options.max_energy_error {
                    return Err(PhaseError::Divergence(Divergence::EnergyErrorLimit));
                }
                self.pending.leaf::<POSITION, T>(
                    edge,
                    &mut self.pool,
                    energy,
                    weight,
                    info.integration_steps,
                );
                // Completed siblings are stored by height. Binary carries visit
                // exactly the same internal subtrees as depth-first recursion.
                let height = (leaf + 1).trailing_zeros() as usize;
                for level in 0..height {
                    let valid = merge::<POSITION, T, R>(
                        &mut self.stack[level],
                        &mut self.pending,
                        forward,
                        false,
                        &mut self.pool,
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
            let valid = merge::<POSITION, T, R>(
                &mut self.root,
                &mut self.pending,
                forward,
                true,
                &mut self.pool,
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
