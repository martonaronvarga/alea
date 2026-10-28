//! Within-orbit adaptive leapfrog NUTS (WALNUTS).
//!
//! Candidate states lie on a fixed macro-time grid. Each macro interval is
//! integrated from the coarsest configured dyadic microstep count whose full
//! Hamiltonian range is within tolerance. The deterministic WALNUTS-D and
//! randomized two-point WALNUTS-R2P local distributions both use the paper's
//! forward/reverse probability correction.
//!
//! Local refinement is part of trajectory construction, not an `SymmetricSplitting`:
//! only complete, reversibly selected macrostep endpoints can enter a tree.
#![forbid(unsafe_code)]

use crate::config::StepSize;
use alea_core::target::{EvaluationError, LogDensityGradient, PointState};
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use rand::{Rng, RngExt};

use crate::{
    Hmc,
    config::{SamplerConfigError, TreeDepth},
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
    hmc::{HmcConfigError, HmcError},
    integrator::Leapfrog,
    trajectory_weight::Weight,
};

/// Invalid immutable WALNUTS settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WalnutsConfigError {
    /// Invalid macro-time interval.
    #[error(transparent)]
    MacroStep(#[from] HmcConfigError),
    /// Invalid doubling depth.
    #[error(transparent)]
    Depth(#[from] SamplerConfigError),
    /// A local budget is zero or its dyadic schedule cannot be represented.
    #[error("local refinement budget must be nonzero and representable")]
    RefinementBudget,
    /// The Hamiltonian-range tolerance is nonpositive or nonfinite.
    #[error("local Hamiltonian-range tolerance must be finite and positive")]
    EnergyTolerance,
    /// The relative orbit-level macro-step jitter is outside `[0, 1)`.
    #[error("relative macro-step jitter must be finite and in [0, 1)")]
    MacroStepJitter,
    /// The configured worst-case microstep work cannot be represented.
    #[error("worst-case WALNUTS work exceeds the counter range")]
    WorkOverflow,
}

/// Distribution over the dyadic microstep count after finding its critical level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WalnutsMicrostepDistribution {
    /// WALNUTS-D: always use the critical (coarsest passing) count.
    #[default]
    Deterministic,
    /// WALNUTS-R2P: use the critical count with probability 2/3 and the next
    /// finer count with probability 1/3.
    RandomizedTwoPoint,
}

impl WalnutsMicrostepDistribution {
    fn sample_level<R: Rng + ?Sized>(self, critical_level: usize, rng: &mut R) -> usize {
        match self {
            Self::Deterministic => critical_level,
            Self::RandomizedTwoPoint => critical_level
                .checked_add(usize::from(rng.random_range(0_u8..3) == 0))
                .expect("validated WALNUTS refinement level"),
        }
    }

    fn log_probability(self, level: usize, critical_level: usize) -> Option<f64> {
        match self {
            Self::Deterministic => (level == critical_level).then_some(0.0),
            Self::RandomizedTwoPoint if level == critical_level => Some((2.0_f64 / 3.0).ln()),
            Self::RandomizedTwoPoint if level == critical_level.checked_add(1)? => {
                Some((1.0_f64 / 3.0).ln())
            }
            Self::RandomizedTwoPoint => None,
        }
    }

    fn extra_level(self) -> u32 {
        u32::from(matches!(self, Self::RandomizedTwoPoint))
    }
}

/// Fixed macro-time and bounded local-refinement policy.
#[derive(Debug, Clone, Copy)]
pub struct WalnutsOptions {
    macro_step: StepSize,
    max_depth: TreeDepth,
    max_refinement_attempts: usize,
    min_microsteps: usize,
    max_energy_range: f64,
    microstep_distribution: WalnutsMicrostepDistribution,
    macro_step_jitter: f64,
}

impl WalnutsOptions {
    /// Construct with Walnutpie-compatible local defaults: five dyadic attempts,
    /// one initial microstep and Hamiltonian-range tolerance 0.5.
    ///
    /// # Errors
    ///
    /// Rejects a nonpositive/nonfinite macro step, an invalid depth, or a
    /// configuration whose worst-case work cannot be represented.
    pub fn new(macro_step: f64, max_depth: usize) -> Result<Self, WalnutsConfigError> {
        let options = Self {
            macro_step: macro_step.try_into()?,
            max_depth: max_depth.try_into()?,
            max_refinement_attempts: 5,
            min_microsteps: 1,
            max_energy_range: 0.5,
            microstep_distribution: WalnutsMicrostepDistribution::Deterministic,
            macro_step_jitter: 0.0,
        };
        options.validate_work()?;
        Ok(options)
    }

    /// Replace the bounded dyadic local-refinement settings.
    ///
    /// The attempted microstep counts are `min_microsteps * 2^j` for
    /// `j = 0..max_refinement_attempts`.
    ///
    /// # Errors
    ///
    /// Rejects zero or overflowing budgets and nonpositive/nonfinite tolerance.
    #[must_use = "this returns validated WALNUTS options"]
    pub fn with_local_refinement(
        mut self,
        max_refinement_attempts: usize,
        min_microsteps: usize,
        max_energy_range: f64,
    ) -> Result<Self, WalnutsConfigError> {
        if max_refinement_attempts == 0 || min_microsteps == 0 {
            return Err(WalnutsConfigError::RefinementBudget);
        }
        if !max_energy_range.is_finite() || max_energy_range <= 0.0 {
            return Err(WalnutsConfigError::EnergyTolerance);
        }
        self.max_refinement_attempts = max_refinement_attempts;
        self.min_microsteps = min_microsteps;
        self.max_energy_range = max_energy_range;
        self.validate_work()?;
        Ok(self)
    }

    /// Select deterministic WALNUTS-D or randomized two-point WALNUTS-R2P.
    ///
    /// R2P may integrate one dyadic level finer than the critical-search budget;
    /// this successor level is included in work validation.
    ///
    /// # Errors
    ///
    /// Rejects a distribution whose worst-case work cannot be represented.
    #[must_use = "this returns validated WALNUTS options"]
    pub fn with_microstep_distribution(
        mut self,
        distribution: WalnutsMicrostepDistribution,
    ) -> Result<Self, WalnutsConfigError> {
        self.microstep_distribution = distribution;
        self.validate_work()?;
        Ok(self)
    }

    /// Uniformly jitter the macro interval once per orbit by this relative amount.
    ///
    /// For example, `0.2` draws one interval from `[0.8 h, 1.2 h)` and then
    /// keeps that interval fixed over the complete doubling trajectory. The
    /// state-independent auxiliary density cancels under reversal while the
    /// jitter breaks resonances with the U-turn schedule.
    ///
    /// # Errors
    ///
    /// Rejects nonfinite values and values outside `[0, 1)`.
    #[must_use = "this returns validated WALNUTS options"]
    pub fn with_macro_step_jitter(mut self, relative: f64) -> Result<Self, WalnutsConfigError> {
        if !relative.is_finite()
            || !(0.0..1.0).contains(&relative)
            || !(self.macro_step() * (1.0 + relative)).is_finite()
        {
            return Err(WalnutsConfigError::MacroStepJitter);
        }
        self.macro_step_jitter = relative;
        Ok(self)
    }

    fn validate_work(self) -> Result<(), WalnutsConfigError> {
        let attempts = u32::try_from(self.max_refinement_attempts)
            .map_err(|_| WalnutsConfigError::RefinementBudget)?;
        let powers = 1_usize
            .checked_shl(attempts)
            .ok_or(WalnutsConfigError::RefinementBudget)?;
        let critical_search = self
            .min_microsteps
            .checked_mul(powers - 1)
            .ok_or(WalnutsConfigError::WorkOverflow)?;
        let selected_level = attempts
            .checked_sub(1)
            .and_then(|level| level.checked_add(self.microstep_distribution.extra_level()))
            .ok_or(WalnutsConfigError::RefinementBudget)?;
        let maximum_microsteps = self
            .min_microsteps
            .checked_mul(
                1_usize
                    .checked_shl(selected_level)
                    .ok_or(WalnutsConfigError::RefinementBudget)?,
            )
            .ok_or(WalnutsConfigError::WorkOverflow)?;
        // The quotient h / microsteps must use the exact integer count. Larger
        // values also cannot be practical bounded trajectory work.
        if u64::try_from(maximum_microsteps)
            .map_or(true, |count| count > (1_u64 << f64::MANTISSA_DIGITS))
        {
            return Err(WalnutsConfigError::RefinementBudget);
        }
        let selected = if matches!(
            self.microstep_distribution,
            WalnutsMicrostepDistribution::RandomizedTwoPoint
        ) {
            maximum_microsteps
        } else {
            0
        };
        // Reverse critical-level discovery may check every count through the
        // selected level. This also replays a selected critical count rather than
        // assuming exact-arithmetic reversibility at an f64 tolerance boundary.
        let reverse_powers = 1_usize
            .checked_shl(
                selected_level
                    .checked_add(1)
                    .ok_or(WalnutsConfigError::RefinementBudget)?,
            )
            .ok_or(WalnutsConfigError::RefinementBudget)?;
        let reverse = self
            .min_microsteps
            .checked_mul(reverse_powers - 1)
            .ok_or(WalnutsConfigError::WorkOverflow)?;
        self.max_depth
            .max_integration_steps()
            .checked_mul(
                critical_search
                    .checked_add(selected)
                    .ok_or(WalnutsConfigError::WorkOverflow)?
                    .checked_add(reverse)
                    .ok_or(WalnutsConfigError::WorkOverflow)?,
            )
            .ok_or(WalnutsConfigError::WorkOverflow)?;
        Ok(())
    }

    /// Fixed simulated time covered by every successful macrostep.
    pub fn macro_step(self) -> f64 {
        self.macro_step.value()
    }

    /// Maximum number of orbit doublings.
    pub fn max_depth(self) -> TreeDepth {
        self.max_depth
    }

    /// Number of dyadic microstep counts that may be attempted per macrostep.
    pub fn max_refinement_attempts(self) -> usize {
        self.max_refinement_attempts
    }

    /// Smallest microstep count in the dyadic schedule.
    pub fn min_microsteps(self) -> usize {
        self.min_microsteps
    }

    /// Maximum Hamiltonian range over one accepted macrostep.
    pub fn max_energy_range(self) -> f64 {
        self.max_energy_range
    }

    /// Local microstep-count distribution.
    pub fn microstep_distribution(self) -> WalnutsMicrostepDistribution {
        self.microstep_distribution
    }

    /// Relative orbit-level uniform macro-step jitter.
    pub fn macro_step_jitter(self) -> f64 {
        self.macro_step_jitter
    }

    fn macro_step_for_orbit<R: Rng + ?Sized>(self, rng: &mut R) -> f64 {
        if self.macro_step_jitter == 0.0 {
            return self.macro_step();
        }
        let centered = 2.0 * rng.random::<f64>() - 1.0;
        self.macro_step() * centered.mul_add(self.macro_step_jitter, 1.0)
    }
}

/// Why WALNUTS stopped expanding its orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WalnutsTermination {
    /// A completed subtree or merged orbit made a U-turn.
    UTurn,
    /// Every configured doubling completed.
    MaxDepth,
    /// No configured dyadic refinement produced a finite in-tolerance edge.
    RefinementLimit,
    /// The selected forward edge had zero probability under reverse local selection.
    ReverseSelection,
    /// Initial momentum or energy was numerically invalid.
    Divergence(Divergence),
}

/// Work and selection statistics for one WALNUTS transition.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct WalnutsTransition {
    /// Whether a non-origin macro-grid state was selected.
    pub accepted: bool,
    /// Attempted macrostep edges, including the edge that stopped refinement.
    pub macro_steps: usize,
    /// Attempted leapfrog microsteps, including discarded refinements and reverse checks.
    pub microsteps: usize,
    /// Forward and reverse dyadic integration attempts.
    pub refinement_attempts: usize,
    /// Largest refinement level selected by a successful forward construction.
    pub max_refinement_level: usize,
    /// Completed valid outer doublings, including a final merged orbit that turns.
    pub tree_depth: usize,
    /// Candidate macro-grid states admitted to completed subtrees, including origin.
    pub valid_states: usize,
    /// One-based attempted macrostep ordinal of the selected state; zero is origin.
    pub selected_index: usize,
    /// Initial Hamiltonian, absent if phase initialization failed.
    pub initial_energy: Option<f64>,
    /// Hamiltonian at the selected macro-grid state.
    pub proposal_energy: Option<f64>,
    /// Largest finite Hamiltonian range seen in a completed refinement attempt.
    pub max_local_energy_range: Option<f64>,
    /// Explicit orbit termination.
    pub termination: WalnutsTermination,
}

impl WalnutsTransition {
    fn new() -> Self {
        Self {
            accepted: false,
            macro_steps: 0,
            microsteps: 0,
            refinement_attempts: 0,
            max_refinement_level: 0,
            tree_depth: 0,
            valid_states: 0,
            selected_index: 0,
            initial_energy: None,
            proposal_energy: None,
            max_local_energy_range: None,
            termination: WalnutsTermination::MaxDepth,
        }
    }
}

/// Construction and typed target/metric failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WalnutsError<E: std::error::Error + 'static> {
    /// Invalid initial state or dimension contract.
    #[error(transparent)]
    Initialization(#[from] HmcError<E>),
    /// Aggregate trajectory workspace is not addressable.
    #[error("WALNUTS workspace size exceeds addressable storage")]
    StorageOverflow,
    /// A concrete metric or model failure occurred during trajectory construction.
    #[error("phase failure at macrostep {macro_step}: {source}")]
    Phase {
        /// One-based attempted macrostep, or zero during initialization.
        macro_step: usize,
        /// Original failure and source chain.
        #[source]
        source: PhaseError<E>,
    },
}

#[derive(Debug)]
struct Tree<'a, T: LogDensityGradient + ?Sized> {
    left: PhaseState<'a, T>,
    right: PhaseState<'a, T>,
    candidate: PhaseState<'a, T>,
    weight: Weight,
    energy: f64,
    index: usize,
    left_log_weight: f64,
    right_log_weight: f64,
}

impl<'a, T: LogDensityGradient + ?Sized> Tree<'a, T> {
    fn new(state: &PhaseState<'a, T>) -> Self {
        Self {
            left: state.clone(),
            right: state.clone(),
            candidate: state.clone(),
            weight: Weight::leaf(0.0),
            energy: 0.0,
            index: 0,
            left_log_weight: 0.0,
            right_log_weight: 0.0,
        }
    }

    fn leaf(&mut self, state: &PhaseState<'a, T>, energy: f64, log_weight: f64, index: usize) {
        self.left.clone_from(state);
        self.right.clone_from(state);
        self.candidate.clone_from(state);
        self.weight = Weight::leaf(log_weight);
        self.energy = energy;
        self.index = index;
        self.left_log_weight = log_weight;
        self.right_log_weight = log_weight;
    }
}

#[derive(Debug)]
struct TurnScratch {
    delta: OwnedBuffer,
    left_velocity: OwnedBuffer,
    right_velocity: OwnedBuffer,
}

impl TurnScratch {
    fn new(dimension: usize) -> Self {
        Self {
            delta: OwnedBuffer::new(dimension),
            left_velocity: OwnedBuffer::new(dimension),
            right_velocity: OwnedBuffer::new(dimension),
        }
    }

    fn continues<T: LogDensityGradient + ?Sized, M: EuclideanMetric>(
        &mut self,
        left: &PhaseState<'_, T>,
        right: &PhaseState<'_, T>,
        metric: &M,
    ) -> Result<bool, PhaseError<T::Error>> {
        for ((delta, right), left) in self
            .delta
            .iter_mut()
            .zip(right.point().position())
            .zip(left.point().position())
        {
            *delta = right - left;
        }
        metric
            .velocity(left.momentum(), &mut self.left_velocity)
            .map_err(PhaseError::Metric)?;
        metric
            .velocity(right.momentum(), &mut self.right_velocity)
            .map_err(PhaseError::Metric)?;
        let left_dot = self
            .left_velocity
            .iter()
            .zip(self.delta.iter())
            .map(|(velocity, delta)| velocity * delta)
            .sum::<f64>();
        let right_dot = self
            .right_velocity
            .iter()
            .zip(self.delta.iter())
            .map(|(velocity, delta)| velocity * delta)
            .sum::<f64>();
        if !left_dot.is_finite() || !right_dot.is_finite() {
            return Err(PhaseError::Divergence(Divergence::Energy));
        }
        Ok(left_dot >= 0.0 && right_dot >= 0.0)
    }
}

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
    let valid = scratch.continues(&left.left, &right.right, metric)?;
    if selected {
        std::mem::swap(&mut old.candidate, &mut new.candidate);
        old.energy = new.energy;
        old.index = new.index;
    }
    if forward {
        std::mem::swap(&mut old.right, &mut new.right);
        std::mem::swap(&mut old.right_log_weight, &mut new.right_log_weight);
    } else {
        std::mem::swap(&mut old.left, &mut new.left);
        std::mem::swap(&mut old.left_log_weight, &mut new.left_log_weight);
    }
    old.weight = weight;
    Ok(valid)
}

#[derive(Debug, Clone, Copy)]
struct Attempt {
    start_energy: f64,
    energy: f64,
    range: f64,
}

fn integrate_attempt<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric>(
    workspace: &mut PhaseWorkspace<'a, T>,
    start: &PhaseState<'a, T>,
    metric: &M,
    macro_step: f64,
    microsteps: usize,
    save: Option<&mut PhaseState<'a, T>>,
    info: &mut WalnutsTransition,
) -> Result<Attempt, PhaseError<T::Error>> {
    debug_assert!(microsteps > 0);
    let step = SignedStep::new(macro_step / microsteps as f64)
        .ok_or(PhaseError::Divergence(Divergence::Energy))?;
    let mut phase = workspace.start_from_phase(start, metric)?;
    let initial = phase.energy()?;
    let mut minimum = initial;
    let mut maximum = initial;
    let mut energy = initial;
    for _ in 0..microsteps {
        info.microsteps = info
            .microsteps
            .checked_add(1)
            .expect("validated WALNUTS microstep budget");
        phase = phase.step(step)?;
        energy = phase.energy()?;
        minimum = minimum.min(energy);
        maximum = maximum.max(energy);
    }
    if let Some(destination) = save {
        phase.save_into(destination)?;
    }
    let range = maximum - minimum;
    if !range.is_finite() {
        return Err(PhaseError::Divergence(Divergence::Energy));
    }
    info.max_local_energy_range = Some(info.max_local_energy_range.unwrap_or(0.0).max(range));
    Ok(Attempt {
        start_energy: initial,
        energy,
        range,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeFailure {
    RefinementLimit,
    ReverseSelection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EdgeBuild {
    Accepted {
        energy: f64,
        refinement_level: usize,
        log_weight_delta: f64,
    },
    Failed(EdgeFailure),
}

/// Fixed-geometry WALNUTS using the paper's Hamiltonian-range criterion.
///
/// The chain owns `O(dimension * max_depth)` reusable storage. Transition work
/// is bounded by [`WalnutsOptions`], and successful transitions allocate no heap
/// memory in library-owned code. The target and metric are statically dispatched.
#[derive(Debug)]
pub struct Walnuts<'a, T: LogDensityGradient + ?Sized, M> {
    point: PointState<'a, T>,
    workspace: PhaseWorkspace<'a, T>,
    metric: M,
    options: WalnutsOptions,
    root: Tree<'a, T>,
    pending: Tree<'a, T>,
    stack: Vec<Tree<'a, T>>,
    edge: PhaseState<'a, T>,
    edge_start: PhaseState<'a, T>,
    scratch: TurnScratch,
}

impl<'a, T: LogDensityGradient + ?Sized, M: EuclideanMetric> Walnuts<'a, T, M> {
    /// Construct a chain with one target evaluation.
    ///
    /// # Errors
    ///
    /// Rejects empty/mismatched dimensions, invalid initial evaluation, or
    /// unaddressable workspace.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        metric: M,
        options: WalnutsOptions,
    ) -> Result<Self, WalnutsError<T::Error>> {
        let hmc = Hmc::new(
            target,
            position,
            metric,
            crate::HmcOptions::new(options.macro_step(), 1).expect("validated macro step"),
        )?;
        Self::from_parts(hmc.into_parts(), options)
    }

    /// Consume a leapfrog HMC chain after covariance or experimental Fisher
    /// warmup without reevaluating the target or consuming randomness.
    ///
    /// The supplied WALNUTS macro step is explicit; it need not equal the HMC
    /// step retained by warmup.
    ///
    /// # Errors
    ///
    /// Rejects an incompatible metric/target dimension or workspace overflow.
    pub fn from_hmc(
        chain: Hmc<'a, T, M, Leapfrog>,
        options: WalnutsOptions,
    ) -> Result<Self, WalnutsError<T::Error>> {
        Self::from_parts(chain.into_parts(), options)
    }

    fn from_parts(
        (point, metric, _): (PointState<'a, T>, M, Leapfrog),
        options: WalnutsOptions,
    ) -> Result<Self, WalnutsError<T::Error>> {
        let dimension = point.dimension();
        dimension
            .checked_mul(10 * (options.max_depth.value() + 2) + 14)
            .and_then(|count| count.checked_mul(size_of::<f64>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(WalnutsError::StorageOverflow)?;
        let edge =
            PhaseState::new(point.clone(), OwnedBuffer::new(dimension)).map_err(|source| {
                WalnutsError::Phase {
                    macro_step: 0,
                    source,
                }
            })?;
        let edge_start =
            PhaseState::new(point.clone(), OwnedBuffer::new(dimension)).map_err(|source| {
                WalnutsError::Phase {
                    macro_step: 0,
                    source,
                }
            })?;
        if metric.dimension() != dimension {
            return Err(HmcError::MetricDimension {
                target: dimension,
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
            edge,
            edge_start,
            scratch: TurnScratch::new(dimension),
            point,
            metric,
            options,
        })
    }

    /// Current coherent cached point.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }

    /// Read-only stationary Euclidean geometry.
    pub fn metric(&self) -> &M {
        &self.metric
    }

    /// Immutable trajectory/refinement settings.
    pub fn options(&self) -> WalnutsOptions {
        self.options
    }

    /// Transactionally reset the position with one target evaluation.
    ///
    /// # Errors
    ///
    /// Invalid shape/evaluation preserves the old point cache.
    pub fn set_position(&mut self, position: &[f64]) -> Result<(), EvaluationError<T::Error>> {
        self.workspace.update_point(&mut self.point, position)
    }

    /// Execute one bounded forward/backward WALNUTS trajectory.
    ///
    /// # Errors
    ///
    /// Concrete metric/model failures preserve the live point. Local numerical
    /// failures exhaust refinement and become transition termination reasons.
    /// Panics propagate without committing a partial state.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<WalnutsTransition, WalnutsError<T::Error>> {
        let mut info = WalnutsTransition::new();
        let result = self.trajectory(rng, &mut info);
        if let Err(source) = result {
            match source {
                PhaseError::Divergence(reason) => {
                    info.termination = WalnutsTermination::Divergence(reason);
                }
                source => {
                    return Err(WalnutsError::Phase {
                        macro_step: info.macro_steps,
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
        info: &mut WalnutsTransition,
    ) -> Result<(), PhaseError<T::Error>> {
        let mut phase = self.workspace.start(&self.point, &self.metric, rng)?;
        let initial = phase.energy()?;
        phase.save_into(&mut self.edge)?;
        self.root.leaf(&self.edge, initial, 0.0, 0);
        info.valid_states = 1;
        info.initial_energy = Some(initial);
        info.max_local_energy_range = Some(0.0);
        let macro_step = self.options.macro_step_for_orbit(rng);

        for depth in 0..self.options.max_depth.value() {
            let forward = rng.random::<bool>();
            let mut edge_log_weight;
            self.edge.clone_from(if forward {
                edge_log_weight = self.root.right_log_weight;
                &self.root.right
            } else {
                edge_log_weight = self.root.left_log_weight;
                &self.root.left
            });
            for leaf in 0..(1_usize << depth) {
                info.macro_steps = info
                    .macro_steps
                    .checked_add(1)
                    .expect("validated WALNUTS macrostep budget");
                let direction = if forward { 1.0 } else { -1.0 };
                let edge = self.build_edge(direction, macro_step, rng, info)?;
                let (energy, refinement_level, log_weight_delta) = match edge {
                    EdgeBuild::Accepted {
                        energy,
                        refinement_level,
                        log_weight_delta,
                    } => (energy, refinement_level, log_weight_delta),
                    EdgeBuild::Failed(EdgeFailure::RefinementLimit) => {
                        info.termination = WalnutsTermination::RefinementLimit;
                        return Ok(());
                    }
                    EdgeBuild::Failed(EdgeFailure::ReverseSelection) => {
                        info.termination = WalnutsTermination::ReverseSelection;
                        return Ok(());
                    }
                };
                info.max_refinement_level = info.max_refinement_level.max(refinement_level);
                edge_log_weight += log_weight_delta;
                if !edge_log_weight.is_finite() {
                    return Err(PhaseError::Divergence(Divergence::Energy));
                }
                self.pending
                    .leaf(&self.edge, energy, edge_log_weight, info.macro_steps);
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
                        info.termination = WalnutsTermination::UTurn;
                        return Ok(());
                    }
                }
                if leaf + 1 < (1_usize << depth) {
                    std::mem::swap(&mut self.pending, &mut self.stack[height]);
                }
            }
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
                info.termination = WalnutsTermination::UTurn;
                return Ok(());
            }
        }
        Ok(())
    }

    fn build_edge<R: Rng + ?Sized>(
        &mut self,
        direction: f64,
        macro_step: f64,
        rng: &mut R,
        info: &mut WalnutsTransition,
    ) -> Result<EdgeBuild, PhaseError<T::Error>> {
        self.edge_start.clone_from(&self.edge);
        let mut accepted = None;
        for level in 0..self.options.max_refinement_attempts {
            info.refinement_attempts = info
                .refinement_attempts
                .checked_add(1)
                .expect("validated WALNUTS refinement budget");
            let microsteps = self.options.min_microsteps << level;
            let attempt = integrate_attempt(
                &mut self.workspace,
                &self.edge_start,
                &self.metric,
                direction * macro_step,
                microsteps,
                Some(&mut self.edge),
                info,
            );
            match attempt {
                Ok(attempt) if attempt.range <= self.options.max_energy_range => {
                    accepted = Some((attempt.start_energy, attempt.energy, level));
                    break;
                }
                Ok(_) | Err(PhaseError::Divergence(_)) => {}
                Err(error) => return Err(error),
            }
        }
        let Some((start_energy, mut energy, critical_level)) = accepted else {
            self.edge.clone_from(&self.edge_start);
            return Ok(EdgeBuild::Failed(EdgeFailure::RefinementLimit));
        };

        let selected_level = self
            .options
            .microstep_distribution
            .sample_level(critical_level, rng);
        let log_forward_probability = self
            .options
            .microstep_distribution
            .log_probability(selected_level, critical_level)
            .expect("selected WALNUTS level has positive forward probability");
        if selected_level != critical_level {
            info.refinement_attempts = info
                .refinement_attempts
                .checked_add(1)
                .expect("validated WALNUTS refinement budget");
            let microsteps = self.options.min_microsteps << selected_level;
            match integrate_attempt(
                &mut self.workspace,
                &self.edge_start,
                &self.metric,
                direction * macro_step,
                microsteps,
                Some(&mut self.edge),
                info,
            ) {
                Ok(attempt) => energy = attempt.energy,
                Err(PhaseError::Divergence(_)) => {
                    self.edge.clone_from(&self.edge_start);
                    return Ok(EdgeBuild::Failed(EdgeFailure::RefinementLimit));
                }
                Err(error) => return Err(error),
            }
        }

        // Discover the reverse critical count explicitly. Testing through the
        // selected count handles floating-point threshold asymmetry and detects
        // zero reverse support for either local distribution.
        let mut reverse_critical_level = None;
        for reverse_level in 0..=selected_level {
            info.refinement_attempts = info
                .refinement_attempts
                .checked_add(1)
                .expect("validated WALNUTS refinement budget");
            let microsteps = self.options.min_microsteps << reverse_level;
            let attempt = integrate_attempt(
                &mut self.workspace,
                &self.edge,
                &self.metric,
                -direction * macro_step,
                microsteps,
                None,
                info,
            );
            match attempt {
                Ok(attempt) if attempt.range <= self.options.max_energy_range => {
                    reverse_critical_level = Some(reverse_level);
                    break;
                }
                Ok(_) | Err(PhaseError::Divergence(_)) => {}
                Err(error) => {
                    self.edge.clone_from(&self.edge_start);
                    return Err(error);
                }
            }
        }
        let Some(log_reverse_probability) = reverse_critical_level.and_then(|reverse_level| {
            self.options
                .microstep_distribution
                .log_probability(selected_level, reverse_level)
        }) else {
            self.edge.clone_from(&self.edge_start);
            return Ok(EdgeBuild::Failed(EdgeFailure::ReverseSelection));
        };
        let log_weight_delta =
            start_energy - energy + log_reverse_probability - log_forward_probability;
        if !log_weight_delta.is_finite() {
            self.edge.clone_from(&self.edge_start);
            return Err(PhaseError::Divergence(Divergence::Energy));
        }
        Ok(EdgeBuild::Accepted {
            energy,
            refinement_level: selected_level,
            log_weight_delta,
        })
    }
}

impl<T: LogDensityGradient + ?Sized, M: EuclideanMetric> crate::MarkovChain for Walnuts<'_, T, M> {
    type Error = WalnutsError<T::Error>;
    type Transition = WalnutsTransition;

    fn position(&self) -> &[f64] {
        self.point.position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Walnuts::step(self, rng)
    }
}

#[cfg(test)]
mod tests;

impl From<crate::config::InvalidScale> for WalnutsConfigError {
    fn from(error: crate::config::InvalidScale) -> Self {
        Self::MacroStep(error.into())
    }
}
