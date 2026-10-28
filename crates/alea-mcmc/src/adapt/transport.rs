//! Consuming warmup learner for frozen affine transport maps.

use std::{error::Error, num::NonZeroUsize};

use alea_core::target::LogDensityGradient;
use alea_math::{
    affine::{AffineMapError, LowRankAffineMap},
    buffer::OwnedBuffer,
    metric::EuclideanMetric,
};
use rand::Rng;
use thiserror::Error;

use crate::{Hmc, hmc::HmcError, integrator::SymmetricSplitting};

/// Validated bounded-memory transport fitting policy.
#[derive(Debug, Clone, Copy)]
pub struct TransportLearningOptions {
    history: NonZeroUsize,
    rank: usize,
    power_iterations: NonZeroUsize,
    ridge: f64,
}

impl TransportLearningOptions {
    /// Stores at most `history` warmup states and fits at most `rank` directions.
    ///
    /// # Errors
    /// Rejects zero history/iterations and non-positive or non-finite ridge.
    pub fn new(history: usize, rank: usize) -> Result<Self, TransportLearningError> {
        Ok(Self {
            history: NonZeroUsize::new(history).ok_or(TransportLearningError::Configuration)?,
            rank,
            power_iterations: NonZeroUsize::new(12).expect("twelve is nonzero"),
            ridge: 1e-6,
        })
    }

    /// Changes the deterministic covariance power-iteration budget.
    pub fn with_power_iterations(
        mut self,
        iterations: usize,
    ) -> Result<Self, TransportLearningError> {
        self.power_iterations =
            NonZeroUsize::new(iterations).ok_or(TransportLearningError::Configuration)?;
        Ok(self)
    }

    /// Changes the positive variance ridge in target coordinates.
    pub fn with_ridge(mut self, ridge: f64) -> Result<Self, TransportLearningError> {
        if !ridge.is_finite() || ridge <= 0.0 {
            return Err(TransportLearningError::Configuration);
        }
        self.ridge = ridge;
        Ok(self)
    }
}

/// Invalid observations or a failed cold fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TransportLearningError {
    /// Options, dimensions or storage sizes are invalid.
    #[error("invalid affine transport learning configuration")]
    Configuration,
    /// An observation has the wrong dimension or a non-finite coordinate.
    #[error("invalid affine transport warmup observation")]
    Observation,
    /// At least two observations are required to estimate scale.
    #[error("insufficient affine transport warmup observations")]
    InsufficientObservations,
    /// Counting or floating-point fitting failed.
    #[error("affine transport fit failed numerically")]
    Numerical,
    /// The fitted frozen map did not satisfy its structural contract.
    #[error(transparent)]
    Affine(#[from] AffineMapError),
}

/// Bounded ring of warmup states. `finish` consumes all trainable state.
#[derive(Debug)]
pub struct AffineTransportLearner {
    dimension: usize,
    options: TransportLearningOptions,
    history: OwnedBuffer,
    observations: usize,
}

impl AffineTransportLearner {
    /// Allocates the fixed history budget once.
    pub fn new(
        dimension: usize,
        options: TransportLearningOptions,
    ) -> Result<Self, TransportLearningError> {
        if options.rank > dimension {
            return Err(TransportLearningError::Configuration);
        }
        let length = dimension
            .checked_mul(options.history.get())
            .ok_or(TransportLearningError::Configuration)?;
        Ok(Self {
            dimension,
            options,
            history: OwnedBuffer::new(length),
            observations: 0,
        })
    }

    /// Number of successfully stored warmup states, including overwritten ones.
    pub fn observations(&self) -> usize {
        self.observations
    }

    /// Stores one complete state without allocating. Invalid input commits nothing.
    pub fn observe(&mut self, position: &[f64]) -> Result<(), TransportLearningError> {
        if position.len() != self.dimension || position.iter().any(|value| !value.is_finite()) {
            return Err(TransportLearningError::Observation);
        }
        let next = self
            .observations
            .checked_add(1)
            .ok_or(TransportLearningError::Numerical)?;
        let slot = self.observations % self.options.history.get();
        let start = slot * self.dimension;
        self.history[start..start + self.dimension].copy_from_slice(position);
        self.observations = next;
        Ok(())
    }

    /// Fits and returns an immutable map, destroying the learning interface.
    pub fn finish(self) -> Result<LowRankAffineMap, TransportLearningError> {
        let count = self.observations.min(self.options.history.get());
        if count < 2 {
            return Err(TransportLearningError::InsufficientObservations);
        }
        let dimension = self.dimension;
        let mut mean = vec![0.0; dimension];
        for sample in self.history[..count * dimension].chunks_exact(dimension) {
            for (average, &value) in mean.iter_mut().zip(sample) {
                *average += value / count as f64;
            }
        }
        let mut variance = vec![0.0; dimension];
        for sample in self.history[..count * dimension].chunks_exact(dimension) {
            for ((sum, &value), &average) in variance.iter_mut().zip(sample).zip(&mean) {
                let residual = value - average;
                *sum += residual * residual / (count - 1) as f64;
            }
        }
        let mut scales = vec![0.0; dimension];
        for (scale, value) in scales.iter_mut().zip(variance) {
            *scale = (value + self.options.ridge).sqrt();
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(TransportLearningError::Numerical);
            }
        }

        let mut basis = Vec::with_capacity(dimension * self.options.rank);
        let mut eigenvalues = Vec::with_capacity(self.options.rank);
        let mut direction = vec![0.0; dimension];
        let mut action = vec![0.0; dimension];
        for rank in 0..self.options.rank {
            for (index, value) in direction.iter_mut().enumerate() {
                let phase = (index + 1).saturating_mul(rank + 1) as f64;
                *value = (phase * 1.618_033_988_749_895).sin() + (phase * 0.5).cos();
            }
            if !orthonormalize(&mut direction, &basis, dimension)
                && !coordinate_seed(&mut direction, &basis, dimension)
            {
                break;
            }
            for _ in 0..self.options.power_iterations.get() {
                covariance_action(
                    &self.history[..count * dimension],
                    count,
                    dimension,
                    &mean,
                    &scales,
                    &direction,
                    &mut action,
                );
                if !orthonormalize(&mut action, &basis, dimension) {
                    break;
                }
                direction.copy_from_slice(&action);
            }
            covariance_action(
                &self.history[..count * dimension],
                count,
                dimension,
                &mean,
                &scales,
                &direction,
                &mut action,
            );
            let eigenvalue = direction
                .iter()
                .zip(&action)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if !eigenvalue.is_finite() {
                return Err(TransportLearningError::Numerical);
            }
            // Rank-deficient empirical directions are represented by the positive
            // diagonal ridge rather than an invalid zero spectral multiplier.
            if eigenvalue <= f64::EPSILON.sqrt() {
                break;
            }
            basis.extend_from_slice(&direction);
            eigenvalues.push(eigenvalue);
        }
        LowRankAffineMap::new(
            OwnedBuffer::from_fn(dimension, |i| mean[i]),
            OwnedBuffer::from_fn(dimension, |i| scales[i]),
            OwnedBuffer::from_fn(basis.len(), |i| basis[i]),
            OwnedBuffer::from_fn(eigenvalues.len(), |i| eigenvalues[i]),
        )
        .map_err(TransportLearningError::from)
    }
}

fn covariance_action(
    samples: &[f64],
    count: usize,
    dimension: usize,
    mean: &[f64],
    scales: &[f64],
    direction: &[f64],
    output: &mut [f64],
) {
    output.fill(0.0);
    for sample in samples.chunks_exact(dimension) {
        let projection = sample
            .iter()
            .zip(mean)
            .zip(scales)
            .zip(direction)
            .map(|(((&value, &average), &scale), &axis)| ((value - average) / scale) * axis)
            .sum::<f64>();
        for (((out, &value), &average), &scale) in
            output.iter_mut().zip(sample).zip(mean).zip(scales)
        {
            *out += projection * ((value - average) / scale) / (count - 1) as f64;
        }
    }
}

fn orthonormalize(direction: &mut [f64], basis: &[f64], dimension: usize) -> bool {
    for _ in 0..2 {
        for axis in basis.chunks_exact(dimension) {
            let projection = direction.iter().zip(axis).map(|(a, b)| a * b).sum::<f64>();
            for (value, &component) in direction.iter_mut().zip(axis) {
                *value -= projection * component;
            }
        }
    }
    let norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 64.0 * f64::EPSILON.sqrt() {
        return false;
    }
    for value in direction {
        *value /= norm;
    }
    true
}

fn coordinate_seed(direction: &mut [f64], basis: &[f64], dimension: usize) -> bool {
    for coordinate in 0..dimension {
        direction.fill(0.0);
        direction[coordinate] = 1.0;
        if orthonormalize(direction, basis, dimension) {
            return true;
        }
    }
    false
}

/// Completed warmup counts; retained sampling has not started.
#[derive(Debug, Clone, Copy)]
pub struct TransportWarmupReport {
    /// Completed HMC transitions.
    pub iterations: usize,
    /// Accepted warmup transitions.
    pub accepted: usize,
    /// Numerically divergent warmup transitions.
    pub divergences: usize,
    /// States supplied to the learner, including the initial state.
    pub observations: usize,
}

/// Frozen handoff from target-coordinate warmup to transported sampling.
#[derive(Debug)]
pub struct TransportWarmupResult {
    /// Immutable learned affine map.
    pub map: LowRankAffineMap,
    /// Last warmup state expressed in the map's latent coordinates.
    pub latent_position: OwnedBuffer,
    /// Warmup-only accounting.
    pub report: TransportWarmupReport,
}

/// A warmup transition or transport fit failed; no retained sampler is returned.
#[derive(Debug, Error)]
pub enum TransportWarmupError<E: Error + 'static> {
    /// The HMC warmup chain failed.
    #[error(transparent)]
    Hmc(#[from] HmcError<E>),
    /// Learning or the final coordinate conversion failed.
    #[error(transparent)]
    Learning(#[from] TransportLearningError),
}

/// Runs fixed-geometry warmup, consumes the learner, and returns only frozen state.
///
/// The initial state and every completed transition state are observed. Rejected
/// states are included because they are the actual chain states. The returned map
/// has no training method, so retained draws cannot update it.
pub fn learn_affine_transport<T, M, I, R>(
    chain: &mut Hmc<'_, T, M, I>,
    mut learner: AffineTransportLearner,
    iterations: usize,
    rng: &mut R,
) -> Result<TransportWarmupResult, TransportWarmupError<T::Error>>
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
    I: SymmetricSplitting,
    R: Rng + ?Sized,
{
    learner.observe(chain.point().position())?;
    let mut accepted = 0;
    let mut divergences = 0;
    for _ in 0..iterations {
        let transition = chain.step(rng)?;
        accepted += usize::from(transition.accepted);
        divergences += usize::from(transition.divergence.is_some());
        learner.observe(chain.point().position())?;
    }
    let observations = learner.observations();
    let map = learner.finish()?;
    let mut latent_position = OwnedBuffer::new(map.dimension());
    map.inverse(chain.point().position(), &mut latent_position)
        .map_err(TransportLearningError::from)?;
    Ok(TransportWarmupResult {
        map,
        latent_position,
        report: TransportWarmupReport {
            iterations,
            accepted,
            divergences,
            observations,
        },
    })
}
