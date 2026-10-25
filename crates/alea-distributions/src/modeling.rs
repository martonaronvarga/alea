//! Distribution-backed scalar factors for typed model assembly.
use crate::{
    scalar::{Bernoulli, Normal},
    wiener::{Wiener4, Wiener4Params, WienerObservation, WienerOptions},
};
use alea_core::{
    distribution::{ContinuousDistribution, Distribution, ParameterScore},
    model::assembly::{AssemblyError as E, Expr, ModelBuilder, Parameter, Positive},
};
fn error(e: impl std::error::Error + Send + Sync + 'static) -> E {
    E::Factor(Box::new(e))
}
/// Normalized Normal log density node, with event/location/scale reverse scores.
/// # Errors
/// Rejects foreign expressions; runtime parameters are checked by Normal.
pub fn normal(
    builder: &mut ModelBuilder,
    event: Expr,
    location: Expr,
    scale: Expr,
) -> Result<Expr, E> {
    builder.custom_with_primal(&[event, location, scale],
        "if x[2] <= 0.0 { return f64::NAN; }\nlet z = (x[0] - x[1]) / x[2];\n-0.5 * z * z - x[2].ln() - 0.5 * ::std::f64::consts::TAU.ln()",
        |x, g| {
        let law = Normal::new(x[1], x[2]).map_err(error)?;
        let value = law.log_prob(&x[0]).map_err(error)?;
        g[0] = law.event_score(x[0]).map_err(error)?;
        law.parameter_score(&x[0], &mut g[1..]).map_err(error)?;
        Ok(value)
    })
}
/// Convenience observation with a support-typed positive scale handle.
/// # Errors
/// Rejects a non-finite observation or foreign handle.
pub fn normal_observation(
    builder: &mut ModelBuilder,
    event: f64,
    location: Expr,
    scale: &Parameter<Positive>,
) -> Result<Expr, E> {
    let event = builder.constant(event)?;
    normal(builder, event, location, scale.expr())
}
/// Bernoulli likelihood in stable log-odds coordinates. Discrete events are data,
/// never continuous model parameters.
/// # Errors
/// Rejects a foreign log-odds expression; runtime non-finite logits are errors.
pub fn bernoulli_logit(builder: &mut ModelBuilder, event: bool, logit: Expr) -> Result<Expr, E> {
    let primal = if event {
        "let z = x[0]; if z >= 0.0 { -(-z).exp().ln_1p() } else { z - z.exp().ln_1p() }"
    } else {
        "let z = x[0]; if z >= 0.0 { -z - (-z).exp().ln_1p() } else { -z.exp().ln_1p() }"
    };
    builder.custom_with_primal(&[logit], primal, move |x, g| {
        let z = x[0];
        let p = if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            z.exp() / (1.0 + z.exp())
        };
        g[0] = f64::from(event) - p;
        Bernoulli::logit_log_prob(z, event).map_err(error)
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
