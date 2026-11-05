//! Distribution-backed factors for typed model assembly.
mod builder;
mod normal;
pub mod regression;
use crate::{
    scalar::Bernoulli,
    wiener::{Wiener4, Wiener4Params, WienerObservation, WienerOptions},
};
use alea_core::{
    distribution::{ParameterScore, ScalarEventScore},
    model::assembly::{AssemblyError as E, Expr, ModelBuilder, Parameter, Positive, ScalarInput},
};
pub use builder::ModelBuilderExt;
pub use normal::{NormalBatch, NormalBatchWorkspace, NormalFactor};
fn error(e: impl std::error::Error + Send + Sync + 'static) -> E {
    E::Factor(Box::new(e))
}
// Common source bridge for built-in analytic primitives. Each emitted helper
// owns a cache; native setup/heap storage stays outside the float-arena receipt.
fn cached_primitive_source(native: &str, constructor: &str) -> String {
    format!(
        r#"use ::alea::model::assembly::Primitive as _;
type Native={native};
::std::thread_local! {{static CACHE: ::std::cell::RefCell<Option<(Native,<Native as ::alea::model::assembly::Primitive>::Workspace)>> = const {{::std::cell::RefCell::new(None)}};}}
fn prepare()->Result<Native,::alea::model::assembly::AssemblyError> {{{constructor}}}
CACHE.try_with(|cache| {{
    let Ok(mut cache)=cache.try_borrow_mut() else {{return f64::NAN}};
    if cache.is_none() {{let Ok(kernel)=prepare() else {{return f64::NAN}};let workspace=kernel.workspace();*cache=Some((kernel,workspace));}}
    let Some((kernel,workspace))=cache.as_mut() else {{return f64::NAN}};
    kernel.evaluate(x,g,workspace).unwrap_or(f64::NAN)
}}).unwrap_or(f64::NAN)"#
    )
}
/// A continuous scalar law as a model factor, including the event derivative.
/// `construct` receives the parameter values in the order supplied by `parameters`.
/// The law's parameter-score order must match that order.
///
/// This is an opaque analytic node; Enzyme source export requires a custom primal
/// instead. Constructed laws are local to each evaluation, not shared across chains.
/// # Errors
/// Rejects foreign expressions or dimension overflow. Construction, support and
/// derivative failures are preserved at evaluation time.
pub fn continuous<D, F>(
    builder: &mut ModelBuilder,
    event: Expr,
    parameters: &[Expr],
    construct: F,
) -> Result<Expr, E>
where
    D: ScalarEventScore + ParameterScore + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
{
    let mut inputs = Vec::with_capacity(parameters.len().checked_add(1).ok_or(E::Shape)?);
    inputs.push(event);
    inputs.extend_from_slice(parameters);
    builder.custom(&inputs, move |x, g| {
        let law = construct(&x[1..]).map_err(error)?;
        if law.parameter_count() != g.len() - 1 {
            return Err(error(alea_core::distribution::DistributionError::Shape));
        }
        let (value, event_score) = law
            .log_prob_event_parameter_score(x[0], &mut g[1..])
            .map_err(error)?;
        g[0] = event_score;
        Ok(value)
    })
}

/// A fixed observation with differentiable distribution parameters. Works with
/// scalar, count, vector and matrix events, using their existing event types.
/// For vector and matrix laws an owned `Vec` borrows the event slice.
///
/// `construct` must return scores in the order of `parameters`. This analytic
/// node has no Enzyme-exportable scalar primal.
/// # Errors
/// Rejects foreign expressions; parameter, observation, shape and score failures
/// remain visible when evaluating the compiled model.
pub fn observation<D, F, O>(
    builder: &mut ModelBuilder,
    event: O,
    parameters: &[Expr],
    construct: F,
) -> Result<Expr, E>
where
    D: ParameterScore + 'static,
    O: std::borrow::Borrow<D::Event> + Send + Sync + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
{
    builder.custom(parameters, move |x, g| {
        let law = construct(x).map_err(error)?;
        if law.parameter_count() != g.len() {
            return Err(error(alea_core::distribution::DistributionError::Shape));
        }
        law.log_prob_parameter_score(event.borrow(), g)
            .map_err(error)
    })
}
/// Normalized Normal log density node, with event/location/scale reverse scores.
/// # Errors
/// Rejects foreign expressions; runtime parameters are checked by Normal.
pub fn normal(
    builder: &mut ModelBuilder,
    event: impl Into<ScalarInput>,
    location: impl Into<ScalarInput>,
    scale: impl Into<ScalarInput>,
) -> Result<Expr, E> {
    let [event, location, scale] = builder.inputs([event.into(), location.into(), scale.into()])?;
    builder.primitive(NormalFactor::new(event, location, scale))
}
/// Sum of Normal log densities with scalar broadcasting along one observation
/// axis. Each nonempty argument has length one or the common batch length.
/// Events may be differentiable expressions. This is one joint factor; explicit
/// conditional row reporting remains available through its native descriptor.
/// # Errors
/// Rejects incompatible/empty shapes and foreign handles before registration;
/// all scalar law checks and local-score failures remain visible at evaluation.
pub fn normal_batch(
    builder: &mut ModelBuilder,
    events: &[Expr],
    locations: &[Expr],
    scales: &[Expr],
) -> Result<Expr, E> {
    builder.primitive(NormalBatch::new(events, locations, scales)?)
}
/// Convenience observation with a support-typed positive scale handle.
/// # Errors
/// Rejects a non-finite observation or foreign handle.
pub fn normal_observation(
    builder: &mut ModelBuilder,
    event: f64,
    location: impl Into<ScalarInput>,
    scale: &Parameter<Positive>,
) -> Result<Expr, E> {
    let event = builder.constant(event)?;
    normal(builder, event, location, scale.expr())
}
/// Bernoulli likelihood in stable log-odds coordinates. Discrete events are data,
/// never continuous model parameters.
/// # Errors
/// Rejects a foreign log-odds expression; runtime non-finite logits are errors.
pub fn bernoulli_logit(
    builder: &mut ModelBuilder,
    event: bool,
    logit: impl Into<ScalarInput>,
) -> Result<Expr, E> {
    let logit = builder.expression(logit)?;
    let primal = if event {
        "let z = x[0]; if z >= 0.0 { -(-z).exp().ln_1p() } else { z - z.exp().ln_1p() }"
    } else {
        "let z = x[0]; if z >= 0.0 { -z - (-z).exp().ln_1p() } else { -z.exp().ln_1p() }"
    };
    builder.custom_with_primal(&[logit], primal, move |x, g| {
        let z = x[0];
        let (value, score) = Bernoulli::logit_log_prob_score(z, event).map_err(error)?;
        g[0] = score;
        Ok(value)
    })
}
/// Checked Wiener4 likelihood node; parameter order is alpha, nondecision time,
/// relative start, drift. The existing analytic series supplies its local VJP.
/// # Errors
/// Rejects foreign expressions; Wiener parameter/backend failures stay errors.
pub fn wiener4(
    builder: &mut ModelBuilder,
    event: WienerObservation,
    parameters: [Expr; 4],
    options: WienerOptions,
) -> Result<Expr, E> {
    builder.custom(&parameters, move |x, g| {
        let p = Wiener4Params::with_params(x[0], x[1], x[2], x[3]).map_err(error)?;
        let result = Wiener4.try_fused(&event, &p, options).map_err(error)?;
        g.copy_from_slice(&result.grad_array());
        Ok(result.log_prob)
    })
}
/// Probability of the upper DDM boundary, diffusion variance one.
/// # Errors
/// Rejects invalid separation/start/drift or unrepresentable output.
pub fn ddm_upper_probability(
    alpha: f64,
    beta: f64,
    drift: f64,
) -> Result<f64, alea_core::distribution::DistributionError> {
    use alea_core::distribution::DistributionError as D;
    if !alpha.is_finite()
        || alpha <= 0.0
        || !beta.is_finite()
        || !(0.0..1.0).contains(&beta)
        || !drift.is_finite()
    {
        return Err(D::Parameter("DDM prediction"));
    }
    let z = 2.0 * drift * alpha;
    let p = if z.abs() < 1e-8 {
        beta + 0.5 * z * beta * (1.0 - beta)
    } else if z > 0.0 {
        (-z * beta).exp_m1() / (-z).exp_m1()
    } else {
        (z * (1.0 - beta)).exp() * (z * beta).exp_m1() / z.exp_m1()
    };
    if p.is_finite() && (0.0..=1.0).contains(&p) {
        Ok(p)
    } else {
        Err(D::Numerical)
    }
}
