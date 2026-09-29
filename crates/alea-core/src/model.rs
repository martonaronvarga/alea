//! Constrained model boundary with chain-local, aligned, reusable scratch.
#![forbid(unsafe_code)]

use alea_math::buffer::OwnedBuffer;

mod primitive;
pub use primitive::{OpaquePrimitive, PrimitiveError, PrimitiveModel};

use crate::{
    target::{self, EvaluationError, LogDensityGradient},
    transform::{ParameterLayout, TransformError},
};
use std::{cell::RefCell, error::Error};
use thiserror::Error;

/// The same fused protocol as a target, but derivatives are in **constrained**
/// coordinates. For symmetric matrices, supply gradients of the full row-major
/// matrix: the transform sums the two off-diagonal contributions automatically.
pub trait ConstrainedModel {
    type Error: Error + 'static;
    fn dimension(&self) -> usize;
    fn logp_grad(&self, parameters: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error>;
}

/// Statically dispatched adapter for analytic or compiler-generated derivatives.
/// The callback must overwrite every gradient component on success.
pub struct AnalyticModel<F> {
    dimension: usize,
    callback: F,
}
impl<F> AnalyticModel<F> {
    pub fn new(dimension: usize, callback: F) -> Self {
        Self {
            dimension,
            callback,
        }
    }
}
impl<F, E> ConstrainedModel for AnalyticModel<F>
where
    F: Fn(&[f64], &mut [f64]) -> Result<f64, E>,
    E: Error + 'static,
{
    type Error = E;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, parameters: &[f64], gradient: &mut [f64]) -> Result<f64, E> {
        (self.callback)(parameters, gradient)
    }
}

#[derive(Debug, Error)]
pub enum ModelError<E: Error + 'static> {
    #[error(transparent)]
    Transform(#[from] TransformError),
    #[error("constrained model dimension changed or does not match the layout")]
    Dimension,
    #[error("recursive evaluation attempted to borrow the same model workspace")]
    WorkspaceBusy,
    #[error("constrained model failed: {0}")]
    Evaluation(#[source] EvaluationError<E>),
}

struct ModelView<'a, M>(&'a M);
impl<M: ConstrainedModel> LogDensityGradient for ModelView<'_, M> {
    type Error = M::Error;
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn logp_grad(&self, x: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.0.logp_grad(x, g)
    }
}

struct Workspace {
    parameters: OwnedBuffer,
    constrained_gradient: OwnedBuffer,
    gradient: OwnedBuffer,
    scratch: OwnedBuffer,
}

/// Fused constrained model + transform Jacobian + reverse pullback.
///
/// Allocate one adapter per chain. This type is intentionally not `Sync`: a
/// shared scratch lock is not placed on the sampler hot path. Errors leave the
/// caller's gradient untouched. No evaluation allocates unless the model does.
pub struct TransformedTarget<M> {
    model: M,
    layout: ParameterLayout,
    workspace: RefCell<Workspace>,
}
impl<M: ConstrainedModel> TransformedTarget<M> {
    pub fn new(model: M, layout: ParameterLayout) -> Result<Self, ModelError<M::Error>> {
        if model.dimension() != layout.constrained_dimension() {
            return Err(ModelError::Dimension);
        }
        let workspace = Workspace {
            parameters: OwnedBuffer::new(layout.constrained_dimension()),
            constrained_gradient: OwnedBuffer::new(layout.constrained_dimension()),
            gradient: OwnedBuffer::new(layout.unconstrained_dimension()),
            scratch: OwnedBuffer::new(layout.scratch_dimension()),
        };
        Ok(Self {
            model,
            layout,
            workspace: RefCell::new(workspace),
        })
    }
    pub fn layout(&self) -> &ParameterLayout {
        &self.layout
    }
    pub fn model(&self) -> &M {
        &self.model
    }
}
impl<M: ConstrainedModel> LogDensityGradient for TransformedTarget<M> {
    type Error = ModelError<M::Error>;
    fn dimension(&self) -> usize {
        self.layout.unconstrained_dimension()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        if q.len() != self.dimension()
            || g.len() != self.dimension()
            || self.model.dimension() != self.layout.constrained_dimension()
        {
            return Err(ModelError::Dimension);
        }
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| ModelError::WorkspaceBusy)?;
        let Workspace {
            parameters,
            constrained_gradient,
            gradient,
            scratch,
        } = &mut *workspace;
        let jac = self.layout.constrain(q, parameters, scratch)?;
        let logp = target::evaluate(&ModelView(&self.model), parameters, constrained_gradient)
            .map_err(ModelError::Evaluation)?;
        let total = logp + jac;
        if !total.is_finite() {
            return Err(ModelError::Evaluation(EvaluationError::NonFiniteLogDensity));
        }
        self.layout
            .pullback(q, constrained_gradient, gradient, scratch)?;
        g.copy_from_slice(gradient);
        Ok(total)
    }
}

/// Sum two same-coordinate models, e.g. an Enzyme prior and an opaque likelihood.
/// Scratch belongs to this chain; no allocations or dynamic dispatch per call.
pub struct SumModel<A, B> {
    first: A,
    second: B,
    dimension: usize,
    scratch: RefCell<OwnedBuffer>,
}
#[derive(Debug, Error)]
pub enum SumModelError<A: Error + 'static, B: Error + 'static> {
    #[error("summed model dimensions disagree or changed")]
    Dimension,
    #[error("summed model workspace is already in use")]
    WorkspaceBusy,
    #[error("first component failed: {0}")]
    First(#[source] EvaluationError<A>),
    #[error("second component failed: {0}")]
    Second(#[source] EvaluationError<B>),
    #[error("summed model value or gradient overflowed")]
    NonFinite,
}
impl<A: ConstrainedModel, B: ConstrainedModel> SumModel<A, B> {
    pub fn new(first: A, second: B) -> Result<Self, SumModelError<A::Error, B::Error>> {
        let dimension = first.dimension();
        if second.dimension() != dimension {
            return Err(SumModelError::Dimension);
        }
        Ok(Self {
            first,
            second,
            dimension,
            scratch: RefCell::new(OwnedBuffer::new(dimension)),
        })
    }
}
impl<A: ConstrainedModel, B: ConstrainedModel> ConstrainedModel for SumModel<A, B> {
    type Error = SumModelError<A::Error, B::Error>;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        if q.len() != self.dimension
            || g.len() != self.dimension
            || self.first.dimension() != self.dimension
            || self.second.dimension() != self.dimension
        {
            return Err(SumModelError::Dimension);
        }
        let mut scratch = self
            .scratch
            .try_borrow_mut()
            .map_err(|_| SumModelError::WorkspaceBusy)?;
        let a = target::evaluate(&ModelView(&self.first), q, g).map_err(SumModelError::First)?;
        let b = target::evaluate(&ModelView(&self.second), q, &mut scratch)
            .map_err(SumModelError::Second)?;
        for (g, b) in g.iter_mut().zip(scratch.iter()) {
            *g += b;
        }
        if !(a + b).is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err(SumModelError::NonFinite);
        }
        Ok(a + b)
    }
}
