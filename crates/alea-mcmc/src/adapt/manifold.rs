//! Warmup-only Bayesian optimization for fixed RMHMC trajectory settings.
//!
//! Wang et al. optimize expected squared jumping distance divided by the square
//! root of trajectory cost. This implementation keeps that objective, but uses
//! a bounded, caller-supplied candidate set and a deterministic Gaussian-process
//! upper-confidence rule. [`ManifoldBayesTuner::finish`] consumes every adaptive
//! statistic and returns only fixed [`RmhmcOptions`] for retained sampling.

use std::error::Error;

use alea_core::{capability::PositionMetric, target::LogDensityGradient};
use alea_math::buffer::OwnedBuffer;
use rand::Rng;

use crate::{Rmhmc, RmhmcError, RmhmcOptions, RmhmcTransition};

const MAX_CANDIDATES: usize = 64;

/// Validated Gaussian-process controls for manifold warmup.
#[derive(Debug, Clone, Copy)]
pub struct ManifoldTunerOptions {
    log_step_length_scale: f64,
    log_time_length_scale: f64,
    observation_noise: f64,
    exploration: f64,
}

impl ManifoldTunerOptions {
    /// Constructs positive finite kernel, noise, and exploration controls.
    ///
    /// # Errors
    /// Rejects non-finite or non-positive values.
    pub fn new(
        log_step_length_scale: f64,
        log_time_length_scale: f64,
        observation_noise: f64,
        exploration: f64,
    ) -> Result<Self, ManifoldTuningError> {
        if [
            log_step_length_scale,
            log_time_length_scale,
            observation_noise,
            exploration,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ManifoldTuningError::Configuration);
        }
        Ok(Self {
            log_step_length_scale,
            log_time_length_scale,
            observation_noise,
            exploration,
        })
    }
}

impl Default for ManifoldTunerOptions {
    fn default() -> Self {
        Self {
            log_step_length_scale: 0.7,
            log_time_length_scale: 0.7,
            observation_noise: 1.0e-6,
            exploration: 1.5,
        }
    }
}

/// Invalid candidate set, observation, or optimizer arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ManifoldTuningError {
    #[error("manifold tuner configuration must be finite and positive")]
    Configuration,
    #[error("manifold tuner requires 1 through 64 distinct candidates")]
    Candidates,
    #[error("manifold observation has mismatched, empty, or non-finite positions")]
    Observation,
    #[error("manifold observation requires a nonzero dynamics cost")]
    DynamicsCost,
    #[error("manifold warmup observation count overflow")]
    ObservationCount,
    #[error("manifold Gaussian-process arithmetic is not finite positive definite")]
    Numerical,
    #[error("manifold tuner has no observation to freeze")]
    NoObservations,
}

/// One measured Wang-style warmup objective.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldObservation {
    pub squared_jump: f64,
    pub dynamics_cost: usize,
    pub objective: f64,
}

/// Fixed RMHMC settings returned across the warmup/sampling boundary.
#[derive(Debug, Clone, Copy)]
pub struct FrozenManifoldParameters {
    candidate_index: usize,
    options: RmhmcOptions,
    mean_objective: f64,
    observations: usize,
}

/// An RMHMC chain whose trajectory controls may change only during warmup.
///
/// Consuming [`Self::finish`] installs one frozen choice and returns the ordinary
/// retained-sampling kernel, so adaptation state cannot accompany retained draws.
#[derive(Debug)]
#[must_use = "consume manifold warmup with finish before retaining samples"]
pub struct RmhmcManifoldWarmup<'a, T: LogDensityGradient + ?Sized, M: PositionMetric + ?Sized> {
    chain: Rmhmc<'a, T, M>,
}

impl<'a, T, M> RmhmcManifoldWarmup<'a, T, M>
where
    T: LogDensityGradient + ?Sized,
    M: PositionMetric + ?Sized,
    T::Error: Error + 'static,
    M::Error: Error + 'static,
{
    /// Constructs one reusable warmup chain without per-candidate refactorization.
    ///
    /// # Errors
    /// Returns the same initial target, metric, shape, and factorization errors as
    /// [`Rmhmc::new`].
    pub fn new(
        target: &'a T,
        metric: &'a M,
        position: OwnedBuffer,
        initial: RmhmcOptions,
    ) -> Result<Self, RmhmcError<T::Error, M::Error>> {
        Ok(Self {
            chain: Rmhmc::new(target, metric, position, initial)?,
        })
    }

    pub fn position(&self) -> &[f64] {
        self.chain.point().position()
    }

    /// Executes one non-retained transition with the selected candidate.
    ///
    /// # Errors
    /// Propagates the concrete RMHMC transition error.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        candidate: RmhmcOptions,
        rng: &mut R,
    ) -> Result<RmhmcTransition, RmhmcError<T::Error, M::Error>> {
        self.chain.install_options(candidate);
        self.chain.step(rng)
    }

    /// Consumes the warmup boundary and returns a fixed retained kernel.
    pub fn finish(mut self, frozen: FrozenManifoldParameters) -> Rmhmc<'a, T, M> {
        self.chain.install_options(frozen.options());
        self.chain
    }
}

impl FrozenManifoldParameters {
    pub fn candidate_index(self) -> usize {
        self.candidate_index
    }

    pub fn options(self) -> RmhmcOptions {
        self.options
    }

    pub fn mean_objective(self) -> f64 {
        self.mean_objective
    }

    pub fn observations(self) -> usize {
        self.observations
    }
}

/// Bounded warmup-only GP optimizer for RMHMC step and trajectory settings.
#[derive(Debug)]
#[must_use = "consume the manifold tuner with finish before retained sampling"]
pub struct ManifoldBayesTuner {
    candidates: Vec<RmhmcOptions>,
    counts: Vec<usize>,
    means: Vec<f64>,
    controls: ManifoldTunerOptions,
    covariance: OwnedBuffer,
    lower: OwnedBuffer,
    rhs: OwnedBuffer,
    alpha: OwnedBuffer,
    kernel_vector: OwnedBuffer,
    solve: OwnedBuffer,
}

impl ManifoldBayesTuner {
    /// Copies and validates a finite candidate grid. Candidates must differ in
    /// step size or integration count; implicit-solve controls are fixed policy,
    /// not coordinates in the Wang objective search.
    ///
    /// # Errors
    /// Rejects empty, oversized, or duplicate trajectory candidates.
    pub fn new(
        candidates: &[RmhmcOptions],
        controls: ManifoldTunerOptions,
    ) -> Result<Self, ManifoldTuningError> {
        if candidates.is_empty() || candidates.len() > MAX_CANDIDATES {
            return Err(ManifoldTuningError::Candidates);
        }
        for (index, candidate) in candidates.iter().enumerate() {
            if candidates[..index].iter().any(|other| {
                other.step_size().to_bits() == candidate.step_size().to_bits()
                    && other.integration_steps() == candidate.integration_steps()
            }) {
                return Err(ManifoldTuningError::Candidates);
            }
        }
        let count = candidates.len();
        let matrix_len = count
            .checked_mul(count)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f64>())
            .ok_or(ManifoldTuningError::Candidates)?;
        Ok(Self {
            candidates: candidates.to_vec(),
            counts: vec![0; count],
            means: vec![0.0; count],
            controls,
            covariance: OwnedBuffer::new(matrix_len),
            lower: OwnedBuffer::new(matrix_len),
            rhs: OwnedBuffer::new(count),
            alpha: OwnedBuffer::new(count),
            kernel_vector: OwnedBuffer::new(count),
            solve: OwnedBuffer::new(count),
        })
    }

    pub fn candidates(&self) -> &[RmhmcOptions] {
        &self.candidates
    }

    /// Selects an unobserved candidate first, then maximizes a deterministic GP
    /// upper-confidence bound over the finite grid.
    ///
    /// # Errors
    /// Returns a numerical error if the bounded GP factorization fails.
    pub fn suggest(&mut self) -> Result<usize, ManifoldTuningError> {
        if let Some(index) = self.counts.iter().position(|count| *count == 0) {
            return Ok(index);
        }
        self.build_posterior()?;
        let mut best_index = 0;
        let mut best_score = f64::NEG_INFINITY;
        for candidate in 0..self.candidates.len() {
            for observed in 0..self.candidates.len() {
                self.kernel_vector[observed] = self.kernel(candidate, observed);
            }
            solve_lower(
                &self.lower,
                &self.kernel_vector,
                &mut self.solve,
                self.candidates.len(),
            )?;
            let mean = dot(&self.kernel_vector, &self.alpha);
            let variance = (1.0 - dot(&self.solve, &self.solve)).max(0.0);
            let score = mean + self.controls.exploration * variance.sqrt();
            if !score.is_finite() {
                return Err(ManifoldTuningError::Numerical);
            }
            if score.total_cmp(&best_score).is_gt() {
                best_score = score;
                best_index = candidate;
            }
        }
        Ok(best_index)
    }

    /// Records squared Euclidean displacement divided by square-root dynamics
    /// cost, the efficiency objective used by Wang et al. Invalid input leaves
    /// persistent statistics unchanged.
    ///
    /// # Errors
    /// Rejects an invalid candidate, positions, cost, overflow, or non-finite
    /// objective.
    pub fn observe(
        &mut self,
        candidate: usize,
        initial: &[f64],
        proposal: &[f64],
        dynamics_cost: usize,
    ) -> Result<ManifoldObservation, ManifoldTuningError> {
        if candidate >= self.candidates.len()
            || initial.is_empty()
            || initial.len() != proposal.len()
            || initial
                .iter()
                .chain(proposal)
                .any(|value| !value.is_finite())
        {
            return Err(ManifoldTuningError::Observation);
        }
        if dynamics_cost == 0 {
            return Err(ManifoldTuningError::DynamicsCost);
        }
        let squared_jump = initial
            .iter()
            .zip(proposal)
            .try_fold(0.0, |sum, (before, after)| {
                let difference = after - before;
                let next = difference.mul_add(difference, sum);
                next.is_finite().then_some(next)
            })
            .ok_or(ManifoldTuningError::Observation)?;
        let objective = squared_jump / (dynamics_cost as f64).sqrt();
        if !objective.is_finite() {
            return Err(ManifoldTuningError::Observation);
        }
        let next_count = self.counts[candidate]
            .checked_add(1)
            .ok_or(ManifoldTuningError::ObservationCount)?;
        let next_mean =
            self.means[candidate] + (objective - self.means[candidate]) / next_count as f64;
        if !next_mean.is_finite() {
            return Err(ManifoldTuningError::Observation);
        }
        self.counts[candidate] = next_count;
        self.means[candidate] = next_mean;
        Ok(ManifoldObservation {
            squared_jump,
            dynamics_cost,
            objective,
        })
    }

    /// Consumes all adaptive state and freezes the best observed empirical mean.
    ///
    /// # Errors
    /// Returns an error if warmup supplied no valid observation.
    pub fn finish(self) -> Result<FrozenManifoldParameters, ManifoldTuningError> {
        let candidate_index = self
            .counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .max_by(|(left, _), (right, _)| self.means[*left].total_cmp(&self.means[*right]))
            .map(|(index, _)| index)
            .ok_or(ManifoldTuningError::NoObservations)?;
        Ok(FrozenManifoldParameters {
            candidate_index,
            options: self.candidates[candidate_index],
            mean_objective: self.means[candidate_index],
            observations: self.counts[candidate_index],
        })
    }

    fn build_posterior(&mut self) -> Result<(), ManifoldTuningError> {
        let count = self.candidates.len();
        self.covariance.fill(0.0);
        self.lower.fill(0.0);
        for row in 0..count {
            self.rhs[row] = self.means[row];
            for column in 0..=row {
                let mut value = self.kernel(row, column);
                if row == column {
                    value += self.controls.observation_noise / self.counts[row] as f64;
                }
                self.covariance[row * count + column] = value;
                self.covariance[column * count + row] = value;
            }
        }
        cholesky(&self.covariance, &mut self.lower, count)?;
        solve_lower(&self.lower, &self.rhs, &mut self.solve, count)?;
        solve_upper_from_lower(&self.lower, &self.solve, &mut self.alpha, count)?;
        Ok(())
    }

    fn kernel(&self, left: usize, right: usize) -> f64 {
        let left = self.candidates[left];
        let right = self.candidates[right];
        let step_difference =
            (left.step_size().ln() - right.step_size().ln()) / self.controls.log_step_length_scale;
        let left_time = left.step_size() * left.integration_steps().get() as f64;
        let right_time = right.step_size() * right.integration_steps().get() as f64;
        let time_difference =
            (left_time.ln() - right_time.ln()) / self.controls.log_time_length_scale;
        (-0.5 * (step_difference * step_difference + time_difference * time_difference)).exp()
    }
}

/// Diminishing probability from Wang et al.'s continuing-adaptation schedule.
///
/// Alea exposes this for reproducibility but does not use it during retained
/// sampling: [`ManifoldBayesTuner`] is always consumed at the warmup boundary.
pub fn wang_diminishing_probability(iteration: usize, delay: usize) -> f64 {
    let effective = iteration.saturating_sub(delay).saturating_add(1);
    (effective as f64).sqrt().recip()
}

fn cholesky(
    matrix: &[f64],
    lower: &mut [f64],
    dimension: usize,
) -> Result<(), ManifoldTuningError> {
    lower.fill(0.0);
    for row in 0..dimension {
        for column in 0..=row {
            let correction = (0..column)
                .map(|index| lower[row * dimension + index] * lower[column * dimension + index])
                .sum::<f64>();
            let value = matrix[row * dimension + column] - correction;
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(ManifoldTuningError::Numerical);
                }
                lower[row * dimension + column] = value.sqrt();
            } else {
                lower[row * dimension + column] = value / lower[column * dimension + column];
            }
        }
    }
    Ok(())
}

fn solve_lower(
    lower: &[f64],
    rhs: &[f64],
    output: &mut [f64],
    dimension: usize,
) -> Result<(), ManifoldTuningError> {
    for row in 0..dimension {
        let correction = (0..row)
            .map(|column| lower[row * dimension + column] * output[column])
            .sum::<f64>();
        output[row] = (rhs[row] - correction) / lower[row * dimension + row];
        if !output[row].is_finite() {
            return Err(ManifoldTuningError::Numerical);
        }
    }
    Ok(())
}

fn solve_upper_from_lower(
    lower: &[f64],
    rhs: &[f64],
    output: &mut [f64],
    dimension: usize,
) -> Result<(), ManifoldTuningError> {
    for row in (0..dimension).rev() {
        let correction = ((row + 1)..dimension)
            .map(|column| lower[column * dimension + row] * output[column])
            .sum::<f64>();
        output[row] = (rhs[row] - correction) / lower[row * dimension + row];
        if !output[row].is_finite() {
            return Err(ManifoldTuningError::Numerical);
        }
    }
    Ok(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(step: f64, integration_steps: usize) -> RmhmcOptions {
        RmhmcOptions::new(step, integration_steps, 8, 1.0e-9).unwrap()
    }

    #[test]
    fn published_objective_and_diminishing_schedule_are_reproduced() {
        let candidates = [candidate(0.1, 4), candidate(0.2, 4)];
        let mut tuner =
            ManifoldBayesTuner::new(&candidates, ManifoldTunerOptions::default()).unwrap();
        assert_eq!(tuner.suggest().unwrap(), 0);
        let observation = tuner.observe(0, &[1.0, 2.0], &[4.0, 6.0], 25).unwrap();
        assert_eq!(observation.squared_jump, 25.0);
        assert_eq!(observation.objective, 5.0);
        assert_eq!(tuner.suggest().unwrap(), 1);
        assert_eq!(wang_diminishing_probability(0, 10), 1.0);
        assert_eq!(wang_diminishing_probability(10, 10), 1.0);
        assert!((wang_diminishing_probability(13, 10) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn finish_consumes_adaptation_and_selects_best_empirical_candidate() {
        let candidates = [candidate(0.1, 2), candidate(0.2, 3), candidate(0.3, 4)];
        let mut tuner =
            ManifoldBayesTuner::new(&candidates, ManifoldTunerOptions::default()).unwrap();
        tuner.observe(0, &[0.0], &[1.0], 1).unwrap();
        tuner.observe(1, &[0.0], &[2.0], 1).unwrap();
        tuner.observe(2, &[0.0], &[1.5], 1).unwrap();
        assert!(tuner.suggest().unwrap() < 3);
        let frozen = tuner.finish().unwrap();
        assert_eq!(frozen.candidate_index(), 1);
        assert_eq!(frozen.options().step_size(), 0.2);
        assert_eq!(frozen.mean_objective(), 4.0);
        assert_eq!(frozen.observations(), 1);
    }

    #[test]
    fn invalid_observation_is_transactional() {
        let candidates = [candidate(0.1, 2)];
        let mut tuner =
            ManifoldBayesTuner::new(&candidates, ManifoldTunerOptions::default()).unwrap();
        assert_eq!(
            tuner.observe(0, &[0.0], &[f64::NAN], 1).unwrap_err(),
            ManifoldTuningError::Observation
        );
        assert_eq!(
            tuner.finish().unwrap_err(),
            ManifoldTuningError::NoObservations
        );
    }
}
