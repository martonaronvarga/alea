//! Shared, warmup-only ChEES trajectory adaptation.
//!
//! The controller is deliberately independent of a particular static-HMC
//! implementation. Every chain reports the initial point, the *unselected*
//! trajectory endpoint and its endpoint velocity. One controller then advances
//! from the complete, stable-order ensemble. This keeps the cross-chain update
//! deterministic under sequential or parallel trajectory execution.

use alea_math::buffer::OwnedBuffer;
use rand::{Rng, RngExt};

use super::{AdaptationError, StepObservation, StepSizeController};
use crate::config::{AcceptanceTarget, StepSize};

const CHEES_ACCEPTANCE_TARGET: f64 = 0.651;
const TRAJECTORY_RATE: f64 = 0.025;
const TRAJECTORY_BETA2: f64 = 0.95;
const TRAJECTORY_EPSILON: f64 = 1e-12;

/// Choice of squared-distance statistic in the ChEES objective.
#[derive(Debug, Clone)]
pub enum CheesCriterion {
    /// The published global squared Euclidean distance.
    Global,
    /// A separately selected unit direction for a difficult-direction experiment.
    DifficultDirection(OwnedBuffer),
}

impl CheesCriterion {
    /// Constructs the optional projected criterion.
    ///
    /// # Errors
    /// Requires a nonempty, finite, nonzero direction. It is normalized once.
    pub fn difficult_direction(direction: &[f64]) -> Result<Self, CheesConfigError> {
        if direction.is_empty() || direction.iter().any(|x| !x.is_finite()) {
            return Err(CheesConfigError::Direction);
        }
        let scale = direction
            .iter()
            .fold(0.0_f64, |scale, x| scale.max(x.abs()));
        if scale == 0.0 {
            return Err(CheesConfigError::Direction);
        }
        let norm = direction
            .iter()
            .map(|x| (x / scale) * (x / scale))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(CheesConfigError::Direction);
        }
        Ok(Self::DifficultDirection(OwnedBuffer::from_fn(
            direction.len(),
            |i| direction[i] / scale / norm,
        )))
    }
}

/// Invalid ensemble-adaptation configuration or observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CheesConfigError {
    /// A step-size or acceptance-target setting is invalid.
    #[error(transparent)]
    Adaptation(#[from] AdaptationError),
    /// A trajectory-time bound is nonpositive, nonfinite, or reversed.
    #[error("trajectory times must be finite, positive, and ordered")]
    TrajectoryTime,
    /// The difficult direction is empty, nonfinite, or zero.
    #[error("difficult direction must be finite and nonzero")]
    Direction,
    /// The ensemble is empty or observation dimensions disagree.
    #[error("ChEES observations must form a nonempty equal-dimensional ensemble")]
    EnsembleShape,
    /// An observation contains a nonfinite value or invalid acceptance probability.
    #[error("ChEES observation is not finite and valid")]
    Observation,
    /// The integration count cannot be represented.
    #[error("trajectory time and step imply an unrepresentable integration count")]
    IntegrationCount,
}

/// One chain's pathwise data for a shared ChEES update.
#[derive(Debug, Clone, Copy)]
pub struct CheesObservation<'a> {
    /// Position before the static trajectory.
    pub initial: &'a [f64],
    /// Endpoint before the Metropolis choice, including when it was rejected.
    pub proposal: &'a [f64],
    /// `M^-1 p` at that same endpoint.
    pub endpoint_velocity: &'a [f64],
    /// Endpoint Metropolis probability; a divergence is represented by zero.
    pub acceptance_probability: f64,
    /// Actually simulated time, `step_size * integration_steps`.
    pub trajectory_time: f64,
}

/// One shared static-HMC plan used by every chain in an ensemble iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheesPlan {
    /// Current jointly adapted integration step.
    pub step_size: StepSize,
    /// Shared randomized number of complete macrosteps.
    pub integration_steps: usize,
    /// Product of step size and integration count.
    pub trajectory_time: f64,
}

/// Summary of one transactional ensemble update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheesUpdate {
    /// Harmonic mean acceptance used by the step-size controller.
    pub harmonic_acceptance: f64,
    /// Mean ChEES objective over chains.
    pub objective: f64,
    /// Acceptance-weighted pathwise derivative in log trajectory time.
    pub trajectory_gradient: f64,
    /// Step size for the next ensemble iteration.
    pub step_size: StepSize,
    /// Maximum trajectory time for the next ensemble iteration.
    pub max_trajectory_time: f64,
}

/// Frozen parameters returned across the warmup/sampling type boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrozenCheesParameters {
    step_size: StepSize,
    max_trajectory_time: f64,
}

impl FrozenCheesParameters {
    /// Frozen integration step.
    pub fn step_size(self) -> StepSize {
        self.step_size
    }

    /// Frozen upper trajectory-time bound.
    pub fn max_trajectory_time(self) -> f64 {
        self.max_trajectory_time
    }

    /// Uniformly randomizes retained integration length from one through
    /// `ceil(max_trajectory_time / step_size)`, following ChEES-HMC.
    pub fn sample_plan<R: Rng + ?Sized>(self, rng: &mut R) -> CheesPlan {
        let maximum = integration_count(self.max_trajectory_time, self.step_size)
            .expect("validated frozen ChEES parameters have a representable count");
        let integration_steps = rng.random_range(1..=maximum);
        CheesPlan {
            step_size: self.step_size,
            integration_steps,
            trajectory_time: self.step_size.value() * integration_steps as f64,
        }
    }
}

/// Shared chain-ensemble warmup state.
///
/// `finish` consumes all adaptation state, so retained sampling can only receive
/// fixed parameters. The optional difficult-direction variant is intentionally a
/// separate criterion and should be benchmarked under its own label.
#[derive(Debug, Clone)]
#[must_use = "consume the adapter with finish before retaining samples"]
pub struct CheesEnsembleAdapter {
    dimension: usize,
    step: StepSize,
    step_controller: StepSizeController,
    log_trajectory: f64,
    log_trajectory_average: f64,
    min_log_trajectory: f64,
    max_log_trajectory: f64,
    trajectory_rms: f64,
    iterations: usize,
    halton_index: usize,
    initial_center: OwnedBuffer,
    proposal_center: OwnedBuffer,
    criterion: CheesCriterion,
}

impl CheesEnsembleAdapter {
    /// Constructs joint step/trajectory adaptation with the paper's acceptance
    /// target (0.651) and trajectory Adam constants (0.025, beta1=0, beta2=0.95).
    ///
    /// # Errors
    /// Rejects invalid dimensions, bounds, step, or criterion dimensions.
    pub fn new(
        dimension: usize,
        initial_step: StepSize,
        initial_trajectory_time: f64,
        min_trajectory_time: f64,
        max_trajectory_time: f64,
        criterion: CheesCriterion,
    ) -> Result<Self, CheesConfigError> {
        if dimension == 0
            || !initial_trajectory_time.is_finite()
            || !min_trajectory_time.is_finite()
            || !max_trajectory_time.is_finite()
            || min_trajectory_time <= 0.0
            || initial_trajectory_time < min_trajectory_time
            || initial_trajectory_time > max_trajectory_time
            || max_trajectory_time <= 0.0
        {
            return Err(CheesConfigError::TrajectoryTime);
        }
        if let CheesCriterion::DifficultDirection(direction) = &criterion
            && direction.len() != dimension
        {
            return Err(CheesConfigError::Direction);
        }
        integration_count(max_trajectory_time, initial_step)?;
        let target: AcceptanceTarget = CHEES_ACCEPTANCE_TARGET
            .try_into()
            .expect("published target is strictly between zero and one");
        Ok(Self {
            dimension,
            step: initial_step,
            step_controller: StepSizeController::new(initial_step, target, Default::default()),
            log_trajectory: initial_trajectory_time.ln(),
            log_trajectory_average: initial_trajectory_time.ln(),
            min_log_trajectory: min_trajectory_time.ln(),
            max_log_trajectory: max_trajectory_time.ln(),
            trajectory_rms: 0.0,
            iterations: 0,
            halton_index: 0,
            initial_center: OwnedBuffer::new(dimension),
            proposal_center: OwnedBuffer::new(dimension),
            criterion,
        })
    }

    /// Produces the next deterministic shared warmup plan using a base-two
    /// Halton variate. The schedule advances on an attempted ensemble iteration.
    pub fn next_plan(&mut self) -> Result<CheesPlan, CheesConfigError> {
        self.halton_index = self
            .halton_index
            .checked_add(1)
            .ok_or(CheesConfigError::IntegrationCount)?;
        let maximum = integration_count(self.log_trajectory.exp(), self.step)?;
        let integration_steps = 1
            + (halton_base_two(self.halton_index) * maximum as f64)
                .floor()
                .min((maximum - 1) as f64) as usize;
        Ok(CheesPlan {
            step_size: self.step,
            integration_steps,
            trajectory_time: self.step.value() * integration_steps as f64,
        })
    }

    /// Advances both controllers from one complete ensemble iteration without
    /// allocation. Invalid input leaves all persistent optimizer parameters
    /// unchanged; numerical failure may overwrite only reusable centering scratch.
    pub fn update(
        &mut self,
        observations: &[CheesObservation<'_>],
    ) -> Result<CheesUpdate, CheesConfigError> {
        validate_observations(self.dimension, observations)?;
        self.initial_center.fill(0.0);
        self.proposal_center.fill(0.0);
        for observation in observations {
            for ((initial_center, proposal_center), (initial, proposal)) in self
                .initial_center
                .iter_mut()
                .zip(self.proposal_center.iter_mut())
                .zip(observation.initial.iter().zip(observation.proposal))
            {
                *initial_center += initial;
                *proposal_center += proposal;
            }
        }
        let denominator = observations.len() as f64;
        for (initial_center, proposal_center) in self
            .initial_center
            .iter_mut()
            .zip(self.proposal_center.iter_mut())
        {
            *initial_center /= denominator;
            *proposal_center /= denominator;
        }

        let harmonic_acceptance = if observations
            .iter()
            .any(|observation| observation.acceptance_probability == 0.0)
        {
            0.0
        } else {
            observations.len() as f64
                / observations
                    .iter()
                    .map(|observation| 1.0 / observation.acceptance_probability)
                    .sum::<f64>()
        };
        let mut objective = 0.0;
        let mut gradient = 0.0;
        let mut total_weight = 0.0;
        for observation in observations {
            let (initial_distance, proposal_distance, endpoint_derivative) = criterion_terms(
                &self.criterion,
                &self.initial_center,
                &self.proposal_center,
                *observation,
            );
            let difference = proposal_distance - initial_distance;
            objective += 0.25 * difference * difference;
            let weight = observation.acceptance_probability;
            gradient += weight * observation.trajectory_time * difference * endpoint_derivative;
            total_weight += weight;
        }
        objective /= observations.len() as f64;
        if total_weight > 0.0 {
            gradient /= total_weight;
        } else {
            gradient = 0.0;
        }
        if !objective.is_finite() || !gradient.is_finite() {
            return Err(CheesConfigError::Observation);
        }

        let mut step_controller = self.step_controller.clone();
        let step = step_controller.update(StepObservation::Acceptance(harmonic_acceptance))?;
        let iterations = self
            .iterations
            .checked_add(1)
            .ok_or(AdaptationError::IterationOverflow)?;
        let rms = (TRAJECTORY_BETA2.sqrt() * self.trajectory_rms)
            .hypot((1.0 - TRAJECTORY_BETA2).sqrt() * gradient);
        let correction = -(iterations as f64 * TRAJECTORY_BETA2.ln()).exp_m1();
        let normalized = gradient / (rms / correction.sqrt() + TRAJECTORY_EPSILON);
        let log_trajectory = (self.log_trajectory + TRAJECTORY_RATE * normalized)
            .clamp(self.min_log_trajectory, self.max_log_trajectory);
        if !rms.is_finite() || !log_trajectory.is_finite() {
            return Err(CheesConfigError::Observation);
        }
        integration_count(log_trajectory.exp(), step)?;
        self.step_controller = step_controller;
        self.step = step;
        self.trajectory_rms = rms;
        self.log_trajectory = log_trajectory;
        self.log_trajectory_average = 0.9 * self.log_trajectory_average + 0.1 * log_trajectory;
        self.iterations = iterations;
        Ok(CheesUpdate {
            harmonic_acceptance,
            objective,
            trajectory_gradient: gradient,
            step_size: step,
            max_trajectory_time: log_trajectory.exp(),
        })
    }

    /// Consumes warmup state and returns only fixed retained-sampling settings.
    pub fn finish(self) -> Result<FrozenCheesParameters, CheesConfigError> {
        let step_size = if self.iterations == 0 {
            self.step
        } else {
            self.step_controller.finish()
        };
        let max_trajectory_time = if self.iterations == 0 {
            self.log_trajectory.exp()
        } else {
            self.log_trajectory_average.exp()
        };
        integration_count(max_trajectory_time, step_size)?;
        Ok(FrozenCheesParameters {
            step_size,
            max_trajectory_time,
        })
    }
}

fn validate_observations(
    dimension: usize,
    observations: &[CheesObservation<'_>],
) -> Result<(), CheesConfigError> {
    if observations.is_empty()
        || observations.iter().any(|observation| {
            observation.initial.len() != dimension
                || observation.proposal.len() != dimension
                || observation.endpoint_velocity.len() != dimension
        })
    {
        return Err(CheesConfigError::EnsembleShape);
    }
    if observations.iter().any(|observation| {
        !observation.acceptance_probability.is_finite()
            || !(0.0..=1.0).contains(&observation.acceptance_probability)
            || !observation.trajectory_time.is_finite()
            || observation.trajectory_time <= 0.0
            || observation
                .initial
                .iter()
                .chain(observation.proposal)
                .chain(observation.endpoint_velocity)
                .any(|x| !x.is_finite())
    }) {
        return Err(CheesConfigError::Observation);
    }
    Ok(())
}

fn criterion_terms(
    criterion: &CheesCriterion,
    initial_center: &[f64],
    proposal_center: &[f64],
    observation: CheesObservation<'_>,
) -> (f64, f64, f64) {
    match criterion {
        CheesCriterion::Global => {
            let mut initial = 0.0;
            let mut proposal = 0.0;
            let mut derivative = 0.0;
            for ((((&q0, &q1), &velocity), &initial_mean), &proposal_mean) in observation
                .initial
                .iter()
                .zip(observation.proposal)
                .zip(observation.endpoint_velocity)
                .zip(initial_center)
                .zip(proposal_center)
            {
                initial += (q0 - initial_mean) * (q0 - initial_mean);
                proposal += (q1 - proposal_mean) * (q1 - proposal_mean);
                derivative += (q1 - proposal_mean) * velocity;
            }
            (initial, proposal, derivative)
        }
        CheesCriterion::DifficultDirection(direction) => {
            let mut initial = 0.0;
            let mut proposal = 0.0;
            let mut velocity = 0.0;
            for (((((&q0, &q1), &v), &initial_mean), &proposal_mean), &axis) in observation
                .initial
                .iter()
                .zip(observation.proposal)
                .zip(observation.endpoint_velocity)
                .zip(initial_center)
                .zip(proposal_center)
                .zip(direction.iter())
            {
                initial += axis * (q0 - initial_mean);
                proposal += axis * (q1 - proposal_mean);
                velocity += axis * v;
            }
            (initial * initial, proposal * proposal, proposal * velocity)
        }
    }
}

fn integration_count(time: f64, step: StepSize) -> Result<usize, CheesConfigError> {
    let ratio = time / step.value();
    if !ratio.is_finite() || ratio <= 0.0 || ratio > usize::MAX as f64 {
        return Err(CheesConfigError::IntegrationCount);
    }
    Ok(ratio.ceil().max(1.0) as usize)
}

fn halton_base_two(mut index: usize) -> f64 {
    let mut result = 0.0;
    let mut factor = 0.5;
    while index != 0 {
        result += factor * (index & 1) as f64;
        index >>= 1;
        factor *= 0.5;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    fn adapter() -> CheesEnsembleAdapter {
        CheesEnsembleAdapter::new(
            2,
            0.1.try_into().unwrap(),
            1.0,
            0.1,
            10.0,
            CheesCriterion::Global,
        )
        .unwrap()
    }

    #[test]
    fn shared_halton_plans_are_deterministic_and_frozen_plans_are_bounded() {
        let mut left = adapter();
        let mut right = adapter();
        for _ in 0..32 {
            assert_eq!(left.next_plan().unwrap(), right.next_plan().unwrap());
        }
        let frozen = left.finish().unwrap();
        let mut rng = SmallRng::seed_from_u64(19);
        for _ in 0..100 {
            let plan = frozen.sample_plan(&mut rng);
            assert!(plan.integration_steps >= 1);
            assert!(
                plan.trajectory_time <= frozen.max_trajectory_time() + frozen.step_size().value()
            );
        }
    }

    #[test]
    fn ensemble_update_matches_hand_computed_objective_and_is_atomic() {
        let mut adapter = adapter();
        let observations = [
            CheesObservation {
                initial: &[-1.0, 0.0],
                proposal: &[1.0, 0.0],
                endpoint_velocity: &[0.5, 0.0],
                acceptance_probability: 0.5,
                trajectory_time: 1.0,
            },
            CheesObservation {
                initial: &[1.0, 0.0],
                proposal: &[1.0, 0.0],
                endpoint_velocity: &[-0.25, 0.0],
                acceptance_probability: 1.0,
                trajectory_time: 1.0,
            },
        ];
        let update = adapter.update(&observations).unwrap();
        assert!((update.harmonic_acceptance - 2.0 / 3.0).abs() < 1e-15);
        // Initial mean is zero and proposal mean is one; both differences are -1.
        assert!((update.objective - 0.25).abs() < 1e-15);

        let before = adapter.clone();
        let invalid = [CheesObservation {
            acceptance_probability: f64::NAN,
            ..observations[0]
        }];
        assert_eq!(adapter.update(&invalid), Err(CheesConfigError::Observation));
        assert_eq!(adapter.iterations, before.iterations);
        assert_eq!(
            adapter.step.value().to_bits(),
            before.step.value().to_bits()
        );
        assert_eq!(
            adapter.log_trajectory.to_bits(),
            before.log_trajectory.to_bits()
        );
        assert_eq!(
            adapter.log_trajectory_average.to_bits(),
            before.log_trajectory_average.to_bits()
        );
    }

    #[test]
    fn difficult_direction_is_normalized_and_separate() {
        let criterion = CheesCriterion::difficult_direction(&[3.0, 4.0]).unwrap();
        let CheesCriterion::DifficultDirection(axis) = criterion else {
            unreachable!()
        };
        assert!((axis.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(CheesCriterion::difficult_direction(&[0.0, 0.0]).is_err());
    }
}
