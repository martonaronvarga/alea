//! Transactional batched leapfrog finite-state execution for explicit layouts.

use std::error::Error;

use alea_core::{
    capability::{
        BatchError, BatchLane, BatchLayout, BatchLogDensityGradient, BatchShape, BatchShapeError,
        evaluate_batch,
    },
    target::EvaluationError,
};
use alea_math::buffer::OwnedBuffer;

use crate::hamiltonian::SignedStep;

/// Failure to construct or execute a batched phase machine. A failed run leaves
/// the last published batch unchanged; proposal scratch is discarded on restart.
#[derive(Debug, thiserror::Error)]
pub enum BatchMachineError<E: Error + 'static> {
    #[error(transparent)]
    Shape(#[from] BatchShapeError),
    #[error(transparent)]
    Batch(#[from] BatchError),
    #[error("batch lane {lane} failed: {source}")]
    Lane {
        lane: usize,
        #[source]
        source: EvaluationError<E>,
    },
    #[error("batched leapfrog requires at least one integration step")]
    IntegrationCount,
    #[error("batched integration produced a non-finite momentum in storage entry {index}")]
    NonFiniteMomentum { index: usize },
}

/// Exact work performed by a successful batched run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchExecutionReport {
    pub chains: usize,
    pub integration_steps: usize,
    /// Number of batched target launches, each covering all lanes.
    pub batch_launches: usize,
    pub scalar_equivalent_evaluations: usize,
}

/// Identity-mass leapfrog over a logical chain batch. Both chain-major and
/// coordinate-major layouts are first-class; no implicit transpose occurs.
#[derive(Debug)]
pub struct BatchedLeapfrog<'a, T: BatchLogDensityGradient + ?Sized> {
    target: &'a T,
    shape: BatchShape,
    position: OwnedBuffer,
    momentum: OwnedBuffer,
    gradient: OwnedBuffer,
    log_density: Vec<f64>,
    proposal_position: OwnedBuffer,
    proposal_momentum: OwnedBuffer,
    proposal_gradient: OwnedBuffer,
    proposal_log_density: Vec<f64>,
    next_position: OwnedBuffer,
    results: Vec<BatchLane<T::Error>>,
}

impl<'a, T: BatchLogDensityGradient + ?Sized> BatchedLeapfrog<'a, T> {
    /// Evaluate and publish the supplied initial batch.
    pub fn new(
        target: &'a T,
        chains: usize,
        layout: BatchLayout,
        position: OwnedBuffer,
        momentum: OwnedBuffer,
    ) -> Result<Self, BatchMachineError<T::Error>> {
        let shape = BatchShape::new(target.dimension(), chains, layout)?;
        if position.len() != shape.storage_len() || momentum.len() != shape.storage_len() {
            return Err(BatchShapeError.into());
        }
        if let Some(index) = momentum.iter().position(|value| !value.is_finite()) {
            return Err(BatchMachineError::NonFiniteMomentum { index });
        }
        let mut gradient = OwnedBuffer::new(shape.storage_len());
        let mut results = (0..chains).map(|_| BatchLane::Pending).collect::<Vec<_>>();
        evaluate_batch(target, shape, &position, &mut gradient, &mut results)?;
        let mut log_density = vec![0.0; chains];
        take_results_into(&mut results, &mut log_density)?;
        let proposal_position = position.clone();
        let proposal_momentum = momentum.clone();
        let proposal_gradient = gradient.clone();
        Ok(Self {
            target,
            shape,
            position,
            momentum,
            gradient,
            proposal_log_density: log_density.clone(),
            log_density,
            proposal_position,
            proposal_momentum,
            proposal_gradient,
            next_position: OwnedBuffer::new(shape.storage_len()),
            results,
        })
    }

    pub fn shape(&self) -> BatchShape {
        self.shape
    }

    pub fn positions(&self) -> &[f64] {
        &self.position
    }

    pub fn momenta(&self) -> &[f64] {
        &self.momentum
    }

    pub fn gradients(&self) -> &[f64] {
        &self.gradient
    }

    pub fn log_densities(&self) -> &[f64] {
        &self.log_density
    }

    /// Run complete kick/drift/evaluate/kick states, then atomically publish the
    /// whole batch. The target receives exactly `steps` batched launches.
    pub fn advance(
        &mut self,
        step: SignedStep,
        steps: usize,
    ) -> Result<BatchExecutionReport, BatchMachineError<T::Error>> {
        if steps == 0 {
            return Err(BatchMachineError::IntegrationCount);
        }
        self.proposal_position.copy_from_slice(&self.position);
        self.proposal_momentum.copy_from_slice(&self.momentum);
        self.proposal_gradient.copy_from_slice(&self.gradient);
        self.proposal_log_density.copy_from_slice(&self.log_density);
        let h = step.value();
        for (momentum, score) in self
            .proposal_momentum
            .iter_mut()
            .zip(self.proposal_gradient.iter())
        {
            *momentum += 0.5 * h * score;
        }
        for integration_step in 0..steps {
            for ((next, position), momentum) in self
                .next_position
                .iter_mut()
                .zip(self.proposal_position.iter())
                .zip(self.proposal_momentum.iter())
            {
                *next = *position + h * momentum;
            }
            evaluate_batch(
                self.target,
                self.shape,
                &self.next_position,
                &mut self.proposal_gradient,
                &mut self.results,
            )?;
            take_results_into(&mut self.results, &mut self.proposal_log_density)?;
            self.proposal_position.copy_from_slice(&self.next_position);
            let kick = if integration_step + 1 == steps {
                0.5
            } else {
                1.0
            };
            for (momentum, score) in self
                .proposal_momentum
                .iter_mut()
                .zip(self.proposal_gradient.iter())
            {
                *momentum += kick * h * score;
            }
            if let Some(index) = self
                .proposal_momentum
                .iter()
                .position(|value| !value.is_finite())
            {
                return Err(BatchMachineError::NonFiniteMomentum { index });
            }
        }
        std::mem::swap(&mut self.position, &mut self.proposal_position);
        std::mem::swap(&mut self.momentum, &mut self.proposal_momentum);
        std::mem::swap(&mut self.gradient, &mut self.proposal_gradient);
        std::mem::swap(&mut self.log_density, &mut self.proposal_log_density);
        Ok(BatchExecutionReport {
            chains: self.shape.chains(),
            integration_steps: steps,
            batch_launches: steps,
            scalar_equivalent_evaluations: steps
                .checked_mul(self.shape.chains())
                .expect("validated storage bounds imply representable evaluation count"),
        })
    }
}

fn take_results_into<E: Error + 'static>(
    results: &mut [BatchLane<E>],
    log_density: &mut [f64],
) -> Result<(), BatchMachineError<E>> {
    debug_assert_eq!(results.len(), log_density.len());
    for (lane, result) in results.iter_mut().enumerate() {
        match std::mem::replace(result, BatchLane::Pending) {
            BatchLane::Complete(Ok(value)) => log_density[lane] = value,
            BatchLane::Complete(Err(source)) => {
                return Err(BatchMachineError::Lane { lane, source });
            }
            BatchLane::Pending => return Err(BatchError::Incomplete { lane }.into()),
        }
    }
    Ok(())
}
