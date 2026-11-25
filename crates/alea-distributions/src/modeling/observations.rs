//! One native observed-law map on the production whole-value registration path.
use super::{E, error};
use crate::scalar::{BernoulliLogit, Normal, PoissonLog};
use alea_core::{
    distribution::{DistributionError, ParameterScore},
    model::{
        assembly::{ModelBuilder, Value},
        tensor::{
            BlockInputs, BlockRules, BlockTangents, BlockType, ElementType, RegionBuilder, Shape,
        },
    },
};
use alea_math::numeric::CompensatedSum;
use std::borrow::Borrow;

type RowScore<D, P> = fn(
    &D,
    &<D as alea_core::distribution::Distribution>::Event,
    &mut [f64],
    &P,
) -> Result<f64, <D as alea_core::distribution::Distribution>::Error>;

struct Observations<D: ParameterScore, F, O, P> {
    events: Vec<O>,
    construct: F,
    lengths: Vec<usize>,
    scores: usize,
    score: Option<RowScore<D, P>>,
}
struct Workspace {
    point: Vec<f64>,
    scores: Vec<f64>,
}

/// Maps fixed typed observations through existing combined value/score entries.
/// Laws own whether that entry is fused; reverse always uses the cached scores.
/// Parameters are whole scalar or rank-one values of length `events.len()`;
/// scalar parameters broadcast. The result is a rank-one value of full log
/// probabilities. Use `ModelBuilder::sum_value` for a joint likelihood factor.
///
/// `construct` receives parameters in the supplied order. All local parameter
/// scores are admitted during forward, including inactive parameters and zero
/// reverse seeds, and cached in exclusive typed workspace. Reverse does not
/// reconstruct laws. Event derivatives are deliberately not requested: this is
/// the same parameter-only contract as `observation`, not a replacement for
/// `normal_batch`'s event-score admission or its qualified source specialization.
/// Empty batches invoke no law. No source, JVP, curvature, replay or separate
/// row-reporting capability is implied; explicit factor roles remain caller-owned.
/// # Errors
/// Rejects foreign values, incompatible shapes or overflowing workspace sizes.
/// Construction, support, shape, missing/nonfinite score and accumulation errors
/// reject evaluation without publishing partial outputs; later valid calls recover.
pub fn observations<D, F, O>(
    builder: &mut ModelBuilder,
    events: Vec<O>,
    parameters: &[Value],
    construct: F,
) -> Result<Value, E>
where
    D: ParameterScore + 'static,
    O: Borrow<D::Event> + Send + Sync + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
{
    register::<D, F, O, (), fn(f64)>(builder, events, parameters, construct, None)
}

#[derive(Clone, Copy)]
struct NormalScale {
    log: f64,
}
fn prepare_normal_scale(scale: f64) -> NormalScale {
    NormalScale { log: scale.ln() }
}

/// Maps Normal observations to their full log probabilities, using the same
/// local parameter-score admission as [observations]. A scalar scale shares
/// its prepared logarithm with other calls on the same scale value during one
/// evaluation. Row-specific scales use the ordinary per-row Normal kernel.
/// # Errors
/// Propagates the shape, domain and score errors of [observations].
pub fn normal_observation_block<O>(
    builder: &mut ModelBuilder,
    events: Vec<O>,
    location: Value,
    scale: Value,
) -> Result<Value, E>
where
    O: Borrow<f64> + Send + Sync + 'static,
{
    let prepare = (!events.is_empty() && builder.value_shape(scale)? == Shape::scalar())
        .then_some(prepare_normal_scale);
    register(
        builder,
        events,
        &[location, scale],
        |p| Normal::new(p[0], p[1]),
        prepare.map(|recipe| Preparation {
            argument: 1,
            recipe,
            score: |law: &Normal, event, scores, scale: &NormalScale| {
                let (value, local) =
                    law.log_prob_event_parameter_scores_prepared(*event, scale.log)?;
                scores.copy_from_slice(&local[1..]);
                Ok(value)
            },
        }),
    )
}

struct Preparation<D: ParameterScore, P, R> {
    argument: usize,
    recipe: R,
    score: RowScore<D, P>,
}

fn register<D, F, O, P, R>(
    builder: &mut ModelBuilder,
    events: Vec<O>,
    parameters: &[Value],
    construct: F,
    preparation: Option<Preparation<D, P, R>>,
) -> Result<Value, E>
where
    D: ParameterScore + 'static,
    O: Borrow<D::Event> + Send + Sync + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
    P: Copy + Send + Sync + 'static,
    R: Fn(f64) -> P + Copy + Send + Sync + 'static,
{
    let (descriptor, rules) = observation_rules(
        events,
        parameters.iter().map(|&value| builder.value_shape(value)),
        construct,
        preparation.as_ref().map(|p| p.score),
    )?;
    match preparation {
        Some(Preparation {
            argument, recipe, ..
        }) => {
            builder.register_block_with_preparation(parameters, descriptor, rules, argument, recipe)
        }
        None => builder.register_block(parameters, descriptor, rules),
    }
}

type ObservationRules<D, F, O, P> = (
    Observations<D, F, O, P>,
    BlockRules<Observations<D, F, O, P>, Workspace>,
);

fn observation_rules<D, F, O, P>(
    events: Vec<O>,
    shapes: impl ExactSizeIterator<Item = Result<Shape, E>>,
    construct: F,
    score: Option<RowScore<D, P>>,
) -> Result<ObservationRules<D, F, O, P>, E>
where
    D: ParameterScore + 'static,
    O: Borrow<D::Event> + Send + Sync + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
    P: 'static,
{
    let count = shapes.len();
    let shape = Shape::new(vec![events.len()])?;
    let mut inputs = Vec::with_capacity(count);
    let mut lengths = Vec::with_capacity(count);
    for input in shapes {
        let input = input?;
        if input != Shape::scalar() && input != shape {
            return Err(E::Shape);
        }
        lengths.push(input.elements());
        inputs.push(BlockType {
            dtype: ElementType::Real,
            shape: input,
        });
    }
    let scores = events.len().checked_mul(count).ok_or(E::Shape)?;
    let heap_bytes = scores
        .checked_add(count)
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .ok_or(E::Shape)?;
    let descriptor = Observations {
        events,
        construct,
        lengths,
        scores,
        score,
    };
    let rules = BlockRules {
        preparation: None,
        inputs,
        output: shape,
        workspace: |d: &Observations<D, F, O, P>| Workspace {
            point: vec![0.; d.lengths.len()],
            scores: vec![0.; d.scores],
        },
        heap_bytes,
        forward: forward::<D, F, O, P>,
        vjp: Some(reverse::<D, F, O, P>),
    };
    Ok((descriptor, rules))
}

/// Normal log probabilities in a tensor region with explicit directional and
/// reverse rules. Scalars broadcast, and vectors match the observation count.
/// The local scores contain location and scale derivatives.
/// # Errors
/// Rejects foreign values or incompatible shapes. Forward execution admits all
/// row values and parameter scores before either derivative rule runs.
pub fn normal_region_observations<O>(
    region: &mut RegionBuilder,
    events: Vec<O>,
    location: Value,
    scale: Value,
) -> Result<Value, E>
where
    O: Borrow<f64> + Send + Sync + 'static,
{
    register_region(region, events, &[location, scale], |p| {
        Normal::new(p[0], p[1])
    })
}

/// Bernoulli-logit log probabilities in a tensor region with explicit
/// directional and reverse rules for the logits.
/// # Errors
/// Rejects foreign values or shapes other than scalar or one logit per event.
pub fn bernoulli_logit_region_observations<O>(
    region: &mut RegionBuilder,
    events: Vec<O>,
    logits: Value,
) -> Result<Value, E>
where
    O: Borrow<bool> + Send + Sync + 'static,
{
    register_region(region, events, &[logits], |p| BernoulliLogit::new(p[0]))
}

/// Poisson-log log probabilities in a tensor region with explicit directional
/// and reverse rules for the log rates. Observations remain integer counts.
/// # Errors
/// Rejects foreign values or shapes other than scalar or one log rate per event.
pub fn poisson_log_region_observations<O>(
    region: &mut RegionBuilder,
    events: Vec<O>,
    log_rates: Value,
) -> Result<Value, E>
where
    O: Borrow<u64> + Send + Sync + 'static,
{
    register_region(region, events, &[log_rates], |p| PoissonLog::new(p[0]))
}

fn register_region<D, F, O>(
    region: &mut RegionBuilder,
    events: Vec<O>,
    parameters: &[Value],
    construct: F,
) -> Result<Value, E>
where
    D: ParameterScore + 'static,
    O: Borrow<D::Event> + Send + Sync + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
{
    let (descriptor, rules) = observation_rules::<D, F, O, ()>(
        events,
        parameters
            .iter()
            .map(|&value| region.value_shape(value).cloned()),
        construct,
        None,
    )?;
    region.register_block_with_jvp(parameters, descriptor, rules, directional::<D, F, O>)
}

fn directional<D: ParameterScore, F, O>(
    d: &Observations<D, F, O, ()>,
    _: BlockInputs<'_>,
    directions: BlockTangents<'_>,
    _: &[f64],
    out: &mut [f64],
    work: &Workspace,
) -> Result<(), E> {
    let count = d.lengths.len();
    for (i, result) in out.iter_mut().enumerate() {
        let mut sum = CompensatedSum::default();
        for (j, &length) in d.lengths.iter().enumerate() {
            if let Some(direction) = directions.real(j)? {
                let value = direction[if length == 1 { 0 } else { i }];
                sum.add(value * work.scores[i * count + j]);
            }
        }
        *result = sum.total();
    }
    Ok(())
}

fn forward<D, F, O, P>(
    d: &Observations<D, F, O, P>,
    inputs: BlockInputs<'_>,
    out: &mut [f64],
    work: &mut Workspace,
) -> Result<(), E>
where
    D: ParameterScore,
    O: Borrow<D::Event>,
    F: Fn(&[f64]) -> Result<D, D::Error>,
    P: 'static,
{
    let prepared = if let Some(score) = d.score {
        Some((score, inputs.evaluation_prepared::<P>()?))
    } else {
        None
    };
    let count = d.lengths.len();
    for (i, event) in d.events.iter().enumerate() {
        for (j, point) in work.point.iter_mut().enumerate() {
            let input = inputs.real(j)?;
            *point = input[if d.lengths[j] == 1 { 0 } else { i }];
        }
        let law = (d.construct)(&work.point).map_err(error)?;
        if law.parameter_count() != count {
            return Err(error(DistributionError::Shape));
        }
        let scores = &mut work.scores[i * count..(i + 1) * count];
        scores.fill(f64::NAN);
        let value = match prepared {
            Some((score, prepared)) => score(&law, event.borrow(), scores, prepared),
            None => law.log_prob_parameter_score(event.borrow(), scores),
        }
        .map_err(error)?;
        if !value.is_finite() || scores.iter().any(|x| !x.is_finite()) {
            return Err(E::Numerical);
        }
        out[i] = value;
    }
    Ok(())
}
fn reverse<D: ParameterScore, F, O, P>(
    d: &Observations<D, F, O, P>,
    _: BlockInputs<'_>,
    _: &[f64],
    seed: &[f64],
    out: &mut [f64],
    work: &mut Workspace,
) -> Result<(), E> {
    let count = d.lengths.len();
    let mut start = 0;
    for (j, &length) in d.lengths.iter().enumerate() {
        if length == 1 {
            let mut sum = CompensatedSum::default();
            for (i, &seed) in seed.iter().enumerate() {
                sum.add(seed * work.scores[i * count + j]);
            }
            out[start] = sum.total();
        } else {
            for (i, &seed) in seed.iter().enumerate() {
                out[start + i] = seed * work.scores[i * count + j];
            }
        }
        start += length;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_observation_constructor_keeps_vjp_and_rejects_directional_mode() {
        let mut region = ModelBuilder::new().region();
        let logit = region.input("logit", Shape::scalar()).unwrap();
        let (descriptor, rules) = observation_rules::<_, _, _, ()>(
            vec![true],
            std::iter::once(Ok(Shape::scalar())),
            |p| BernoulliLogit::new(p[0]),
            None,
        )
        .unwrap();
        let y = region.register_block(&[logit], descriptor, rules).unwrap();
        let bound = region.compile(y).unwrap().bind(&[]).unwrap();
        for active in [false, true] {
            assert!(matches!(bound.jvp(&[active]), Err(E::Factor(_))));
        }
        let (mut value, mut score) = ([0.], [0.]);
        bound
            .value_and_vjp(&[0.], &[1.], &mut value, &mut score, &mut bound.workspace())
            .unwrap();
        assert_eq!((value, score), ([-std::f64::consts::LN_2], [0.5]));
    }
}
