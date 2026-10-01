//! Optional target capabilities; neither AD nor a batch runtime is mandatory.
use crate::target::{DimensionError, EvaluationError, LogDensityGradient};
use std::error::Error;

/// Optional Hessian-vector product of **potential** `U = -log pi`.
/// Implementations overwrite all output entries on success, reuse caller storage,
/// and obey the same semantic purity contract as `LogDensityGradient`.
pub trait HessianVector: LogDensityGradient {
    fn potential_hvp(
        &self,
        position: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error>;
}
#[derive(Debug, thiserror::Error)]
pub enum HvpError<E: Error + 'static> {
    #[error(transparent)]
    Evaluation(#[from] EvaluationError<E>),
    #[error("HVP vector dimension {actual} differs from target dimension {expected}")]
    VectorDimension { expected: usize, actual: usize },
    #[error("HVP vector entry {index} is not finite")]
    NonFiniteVector { index: usize },
    #[error("HVP output entry {index} is missing or not finite")]
    NonFiniteOutput { index: usize },
}
/// Checked, allocation-free HVP. Shape/input failures preserve the output;
/// backend failures may leave partial output. No finite-difference fallback.
/// Target panics propagate, as with [`crate::target::evaluate`]. After a backend
/// error or unwinding, discard output contents; this wrapper is not a cache.
/// # Errors
/// Returns shape, input, missing/non-finite output or original model errors.
pub fn evaluate_hvp<T: HessianVector + ?Sized>(
    target: &T,
    q: &[f64],
    v: &[f64],
    out: &mut [f64],
) -> Result<(), HvpError<T::Error>> {
    let d = target.dimension();
    if q.len() != d || out.len() != d {
        return Err(EvaluationError::Dimension(DimensionError {
            expected: d,
            position: q.len(),
            gradient: out.len(),
        })
        .into());
    }
    if v.len() != d {
        return Err(HvpError::VectorDimension {
            expected: d,
            actual: v.len(),
        });
    }
    if let Some(index) = q.iter().position(|x| !x.is_finite()) {
        return Err(EvaluationError::NonFinitePosition { index }.into());
    }
    if let Some(index) = v.iter().position(|x| !x.is_finite()) {
        return Err(HvpError::NonFiniteVector { index });
    }
    out.fill(f64::NAN);
    target
        .potential_hvp(q, v, out)
        .map_err(EvaluationError::Model)?;
    if let Some(index) = out.iter().position(|x| !x.is_finite()) {
        return Err(HvpError::NonFiniteOutput { index });
    }
    Ok(())
}

/// Physical layout is a backend choice, independent of logical chain identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchLayout {
    ChainMajor,
    CoordinateMajor,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid batch shape, buffer length or target dimension")]
pub struct BatchShapeError;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchShape {
    dimension: usize,
    chains: usize,
    layout: BatchLayout,
    len: usize,
}
impl BatchShape {
    /// Checked storage shape, including empty batches/zero-dimensional targets.
    /// # Errors
    /// Rejects storage sizes not representable as Rust f64 slices.
    pub fn new(
        dimension: usize,
        chains: usize,
        layout: BatchLayout,
    ) -> Result<Self, BatchShapeError> {
        let len = dimension
            .checked_mul(chains)
            .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
            .ok_or(BatchShapeError)?;
        Ok(Self {
            dimension,
            chains,
            layout,
            len,
        })
    }
    pub fn dimension(self) -> usize {
        self.dimension
    }
    pub fn chains(self) -> usize {
        self.chains
    }
    pub fn layout(self) -> BatchLayout {
        self.layout
    }
    pub fn storage_len(self) -> usize {
        self.len
    }
    pub fn index(self, chain: usize, coordinate: usize) -> Option<usize> {
        if chain >= self.chains || coordinate >= self.dimension {
            return None;
        }
        Some(match self.layout {
            BatchLayout::ChainMajor => chain * self.dimension + coordinate,
            BatchLayout::CoordinateMajor => coordinate * self.chains + chain,
        })
    }
}
/// One independently reported lane. Pending is caught by the checked wrapper.
#[derive(Debug)]
pub enum BatchLane<E: Error + 'static> {
    Pending,
    Complete(Result<f64, EvaluationError<E>>),
}
/// Optional fused batched evaluation. Caller supplies validated shape/storage and
/// finite inputs. Each lane must be completed; a failed lane must not poison other
/// lanes. Success writes all of that lane's gradient. No scheduling assumptions.
pub trait BatchLogDensityGradient: LogDensityGradient {
    fn batch_logp_grad(
        &self,
        shape: BatchShape,
        positions: &[f64],
        gradients: &mut [f64],
        results: &mut [BatchLane<Self::Error>],
    );
}
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error(transparent)]
    Shape(#[from] BatchShapeError),
    #[error("batch position entry {index} is not finite")]
    NonFiniteInput { index: usize },
    #[error("batch target did not complete lane {lane}")]
    Incomplete { lane: usize },
}
/// Check buffers, call once and validate every successful lane's finite outputs.
/// Numerical/backend failures are per-lane results. Shape/non-finite input errors
/// occur before any output mutation. Missing lane status is a protocol error.
/// A target panic propagates and skips output validation: even lanes marked
/// `Complete(Ok(_))` must not be consumed after unwinding. This is not per-lane
/// panic isolation. Ordinary returned lane errors do not invalidate other lanes.
/// # Errors
/// Returns shape/input/incomplete-lane errors; completed lanes remain inspectable.
pub fn evaluate_batch<T: BatchLogDensityGradient + ?Sized>(
    target: &T,
    shape: BatchShape,
    q: &[f64],
    grad: &mut [f64],
    results: &mut [BatchLane<T::Error>],
) -> Result<(), BatchError> {
    if shape.dimension() != target.dimension()
        || q.len() != shape.storage_len()
        || grad.len() != q.len()
        || results.len() != shape.chains()
    {
        return Err(BatchShapeError.into());
    }
    if let Some(index) = q.iter().position(|x| !x.is_finite()) {
        return Err(BatchError::NonFiniteInput { index });
    }
    grad.fill(f64::NAN);
    for result in results.iter_mut() {
        *result = BatchLane::Pending;
    }
    target.batch_logp_grad(shape, q, grad, results);
    let mut missing = None;
    for (lane, result) in results.iter_mut().enumerate() {
        match result {
            BatchLane::Pending => {
                missing.get_or_insert(lane);
            }
            BatchLane::Complete(Ok(logp)) => {
                if !logp.is_finite() {
                    *result = BatchLane::Complete(Err(EvaluationError::NonFiniteLogDensity));
                } else if let Some(index) = (0..shape.dimension()).find(|&i| {
                    !grad[shape.index(lane, i).expect("validated lane coordinate")].is_finite()
                }) {
                    *result =
                        BatchLane::Complete(Err(EvaluationError::NonFiniteGradient { index }));
                }
            }
            BatchLane::Complete(Err(_)) => {}
        }
    }
    if let Some(lane) = missing {
        return Err(BatchError::Incomplete { lane });
    }
    Ok(())
}
