use crate::density::LogDensity;

pub trait MeasurableSpace {}

pub trait Measure {
    type Space: ?Sized;
    type Point;
}

/// A normalized probability measure that admits a density in the chosen reference measure.
/// Pointwise `log_prob` is only meaningful when a density w.r.t. a base measure exists.
pub trait Distribution: Measure {
    fn log_prob(&self, x: &Self::Point) -> f64;
}

trait Sampleable {
    type Sample;

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::Sample;
}

/// Statistical model / parametric family:
/// maps parameters and data to a log density or log likelihood
pub trait Family {
    type Params;
    type Data;

    /// .
    fn log_prob(params: &Self::Params, data: &Self::Data) -> f64;
}

pub struct WithParams<F: Family> {
    pub family: F,
    pub params: F::Params,
}

impl<F: Family> WithParams<F> {
    pub fn new(family: F, params: F::Params) -> Self {
        Self { family, params }
    }
}

pub struct Target<F, D> {
    pub family: F,
    pub data: D,
}

impl<F, D> Target<F, D> {
    pub fn new(family: F, data: D) -> Self {
        Self { family, data }
    }
}

impl<F, D> LogDensity for Target<F, D>
where
    F: Family<Data = D>,
{
    type Point = F::Params;
    fn log_prob(&self, theta: &Self::Point) -> f64 {
        F::log_prob(theta, &self.data)
    }
}

pub trait TransitionModel<S> {
    fn log_transition(&self, prev: &S, next: &S, t: usize) -> f64;
    fn sample_next<R: rand::Rng + ?Sized>(&self, prev: &S, rng: &mut R) -> S;
}

pub trait ObservationModel<S, Y> {
    fn log_likelihood(&self, state: &S, obs: &Y) -> f64;
}

/// Generalized parametrization / link / reparametrization.
/// Broader than a link function because it may depend on
/// context and state.
pub trait ParameterMap {
    type GlobalParams;
    type Context: ?Sized;
    type State;
    type LocalParams;

    fn init_state(&self, global: &Self::GlobalParams, ctx: &Self::Context) -> Self::State;

    fn map(
        &self,
        global: &Self::GlobalParams,
        ctx: &Self::Context,
        state: &Self::State,
        t: usize,
    ) -> Self::LocalParams;

    fn update_state(&self, state: &mut Self::State, ctx: &Self::Context, t: usize);
}

pub trait Parameter {
    type Constrained;
    type Unconstrained;

    fn to_unconstrained(value: &Self::Constrained) -> Self::Unconstrained;
    fn from_unconstrained(value: &Self::Unconstrained) -> Self::Constrained;
    fn log_abs_det_jacobian(unconstrained: &Self::Unconstrained) -> f64;
}
