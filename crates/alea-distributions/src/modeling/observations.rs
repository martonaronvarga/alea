//! Observed-law maps on the production whole-value registration path.
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
use std::fmt::Write;

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

// Only these explicit law adapters supply qualified row curvature.
trait DirectionalScore: ParameterScore {
    const SOURCE_CONSTRUCTOR: &'static str;
    const PARAMETERS: &'static str;
    fn source_event(event: &Self::Event) -> String;

    fn score_direction(
        event: &Self::Event,
        point: [f64; 2],
        scores: &[f64],
        direction: [f64; 2],
        parameter: usize,
    ) -> f64;
}

impl DirectionalScore for Normal {
    const SOURCE_CONSTRUCTOR: &'static str = "Normal";
    const PARAMETERS: &'static str = "p[0],p[1]";
    fn source_event(event: &f64) -> String {
        format!("f64::from_bits({}_u64)", event.to_bits())
    }

    fn score_direction(
        event: &f64,
        point: [f64; 2],
        scores: &[f64],
        direction: [f64; 2],
        parameter: usize,
    ) -> f64 {
        Self::parameter_score_direction(*event, point, scores, direction, parameter)
    }
}

impl DirectionalScore for BernoulliLogit {
    const SOURCE_CONSTRUCTOR: &'static str = "BernoulliLogit";
    const PARAMETERS: &'static str = "p[0]";
    fn source_event(event: &bool) -> String {
        event.to_string()
    }

    fn score_direction(_: &bool, point: [f64; 2], _: &[f64], direction: [f64; 2], _: usize) -> f64 {
        -Self::curvature_weight(point[0]) * direction[0]
    }
}

impl DirectionalScore for PoissonLog {
    const SOURCE_CONSTRUCTOR: &'static str = "PoissonLog";
    const PARAMETERS: &'static str = "p[0]";
    fn source_event(event: &u64) -> String {
        format!("{event}_u64")
    }

    fn score_direction(_: &u64, point: [f64; 2], _: &[f64], direction: [f64; 2], _: usize) -> f64 {
        -Self::curvature_weight(point[0]) * direction[0]
    }
}

/// Normal log probabilities in a tensor region with native directional, reverse,
/// and curvature rules. Scalars broadcast across the observations.
/// The local scores contain location and scale derivatives.
/// # Errors
/// Rejects foreign values or incompatible shapes. Forward execution admits all
/// row values and parameter scores before the derivative rules run.
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
/// directional, reverse and curvature rules for the logits.
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

/// Poisson-log log probabilities in a tensor region with native directional,
/// reverse, and curvature rules for the log rates. Observations are integer counts.
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
    D: DirectionalScore + 'static,
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
    region.register_block_with_source(
        parameters,
        descriptor,
        rules,
        directional::<D, F, O>,
        reverse_direction::<D, F, O>,
        observation_source::<D, F, O>,
    )
}

fn observation_source<D, F, O>(d: &Observations<D, F, O, ()>) -> String
where
    D: DirectionalScore,
    O: Borrow<D::Event>,
{
    let parameters = d.lengths.len();
    let mut body = String::from("let events=[");
    for event in &d.events {
        write!(body, "{},", D::source_event(event.borrow())).unwrap();
    }
    write!(
        body,
        "];::alea::distributions::modeling::__observation_block::<_,{parameters}>(&events,args,out,seed,scores,|p|::alea::distributions::scalar::{}::new({}))",
        D::SOURCE_CONSTRUCTOR,
        D::PARAMETERS
    )
    .unwrap();
    body
}

/// Runs one emitted observed-law map over frozen events and checked operands.
/// Generated callers stage both outputs in private scratch.
#[doc(hidden)]
pub fn __observation_block<D: ParameterScore, const P: usize>(
    events: &[D::Event],
    args: &[&[f64]],
    out: &mut [f64],
    seed: Option<&[f64]>,
    scores: &mut [f64],
    construct: impl Fn([f64; P]) -> Result<D, D::Error>,
) -> bool
where
    D::Event: Sized,
{
    let Ok(args) = <[&[f64]; P]>::try_from(args) else {
        return false;
    };
    if args
        .iter()
        .any(|arg| arg.len() != 1 && arg.len() != events.len())
    {
        return false;
    }
    if let Some(seed) = seed {
        if seed.len() != events.len()
            || args
                .iter()
                .try_fold(0_usize, |n, arg| n.checked_add(arg.len()))
                != Some(scores.len())
        {
            return false;
        }
    } else if out.len() != events.len() {
        return false;
    }

    let mut broadcast = [CompensatedSum::default(); P];
    for (row, event) in events.iter().enumerate() {
        let point = args.map(|arg| arg[if arg.len() == 1 { 0 } else { row }]);
        let Ok(law) = construct(point) else {
            return false;
        };
        let mut local = [f64::NAN; P];
        let Ok(value) = law.log_prob_parameter_score(event, &mut local) else {
            return false;
        };
        if law.parameter_count() != P || !value.is_finite() || local.iter().any(|v| !v.is_finite())
        {
            return false;
        }
        if let Some(seed) = seed {
            let mut offset = 0;
            for (j, arg) in args.iter().enumerate() {
                let contribution = seed[row] * local[j];
                if arg.len() == 1 {
                    // Interleaved arguments have independent sums. Each sees
                    // the identical row order used by the native pullback.
                    broadcast[j].add(contribution);
                } else {
                    scores[offset + row] = contribution;
                }
                offset += arg.len();
            }
        } else {
            out[row] = value;
        }
    }
    if seed.is_some() {
        let mut offset = 0;
        for (arg, sum) in args.iter().zip(broadcast) {
            if arg.len() == 1 {
                scores[offset] = sum.total();
            }
            offset += arg.len();
        }
        return scores.iter().all(|v| v.is_finite());
    }
    true
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

fn reverse_direction<D: DirectionalScore, F, O: Borrow<D::Event>>(
    d: &Observations<D, F, O, ()>,
    inputs: BlockInputs<'_>,
    tangents: BlockTangents<'_>,
    _: &[f64],
    [seed, seed_direction]: [&[f64]; 2],
    out: &mut [f64],
    work: &Workspace,
) -> Result<(), E> {
    let count = d.lengths.len();
    let mut point = [&[][..]; 2];
    let mut direction = [None; 2];
    for j in 0..count {
        point[j] = inputs.real(j)?;
        direction[j] = tangents.real(j)?;
    }
    accumulate_scores(&d.lengths, seed.len(), out, |i, j| {
        // The region's curvature request enables every differentiable formal
        // input. A symbolic zero here denotes an inactive argument occurrence.
        if direction[j].is_none() {
            return 0.;
        }
        let scores = &work.scores[i * count..(i + 1) * count];
        let mut contribution = seed_direction[i] * scores[j];
        if seed[i] != 0. {
            let mut row = [0.; 2];
            let mut row_direction = [0.; 2];
            for (k, &length) in d.lengths.iter().enumerate() {
                let index = if length == 1 { 0 } else { i };
                row[k] = point[k][index];
                row_direction[k] = direction[k].map_or(0., |v| v[index]);
            }
            if row_direction.iter().any(|&v| v != 0.) {
                contribution += seed[i]
                    * D::score_direction(d.events[i].borrow(), row, scores, row_direction, j);
            }
        }
        contribution
    });
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
    accumulate_scores(&d.lengths, seed.len(), out, |i, j| {
        seed[i] * work.scores[i * count + j]
    });
    Ok(())
}

fn accumulate_scores(
    lengths: &[usize],
    rows: usize,
    out: &mut [f64],
    contribution: impl Fn(usize, usize) -> f64,
) {
    let mut start = 0;
    for (j, &length) in lengths.iter().enumerate() {
        if length == 1 {
            let mut sum = CompensatedSum::default();
            for i in 0..rows {
                sum.add(contribution(i, j));
            }
            out[start] = sum.total();
        } else {
            for i in 0..length {
                out[start + i] = contribution(i, j);
            }
        }
        start += length;
    }
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
