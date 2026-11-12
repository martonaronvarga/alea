//! One native observed-law map on the production whole-value registration path.
use super::{E, error};
use alea_core::{
    distribution::ParameterScore,
    model::{
        assembly::{ModelBuilder, Value},
        tensor::{BlockInputs, BlockRules, BlockType, ElementType, Shape},
    },
};
use alea_math::numeric::CompensatedSum;
use std::{borrow::Borrow, marker::PhantomData};

struct Observations<D, F, O> {
    events: Vec<O>,
    construct: F,
    lengths: Vec<usize>,
    scores: usize,
    law: PhantomData<fn() -> D>,
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
    let shape = Shape::new(vec![events.len()])?;
    let mut inputs = Vec::with_capacity(parameters.len());
    let mut lengths = Vec::with_capacity(parameters.len());
    for &value in parameters {
        let input = builder.value_shape(value)?;
        if input != Shape::scalar() && input != shape {
            return Err(E::Shape);
        }
        lengths.push(input.elements());
        inputs.push(BlockType {
            dtype: ElementType::Real,
            shape: input,
        });
    }
    let scores = events.len().checked_mul(parameters.len()).ok_or(E::Shape)?;
    let heap_bytes = scores
        .checked_add(parameters.len())
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .ok_or(E::Shape)?;
    builder.register_block(
        parameters,
        Observations::<D, F, O> {
            events,
            construct,
            lengths,
            scores,
            law: PhantomData,
        },
        BlockRules {
            inputs,
            output: shape,
            workspace: |d| Workspace {
                point: vec![0.; d.lengths.len()],
                scores: vec![0.; d.scores],
            },
            heap_bytes,
            forward: forward::<D, F, O>,
            vjp: Some(reverse::<D, F, O>),
        },
    )
}
fn forward<D, F, O>(
    d: &Observations<D, F, O>,
    inputs: BlockInputs<'_>,
    out: &mut [f64],
    work: &mut Workspace,
) -> Result<(), E>
where
    D: ParameterScore,
    O: Borrow<D::Event>,
    F: Fn(&[f64]) -> Result<D, D::Error>,
{
    let count = d.lengths.len();
    for (i, event) in d.events.iter().enumerate() {
        for (j, point) in work.point.iter_mut().enumerate() {
            let input = inputs.real(j)?;
            *point = input[if d.lengths[j] == 1 { 0 } else { i }];
        }
        let law = (d.construct)(&work.point).map_err(error)?;
        if law.parameter_count() != count {
            return Err(error(alea_core::distribution::DistributionError::Shape));
        }
        let scores = &mut work.scores[i * count..(i + 1) * count];
        scores.fill(f64::NAN);
        let value = law
            .log_prob_parameter_score(event.borrow(), scores)
            .map_err(error)?;
        if !value.is_finite() || scores.iter().any(|x| !x.is_finite()) {
            return Err(E::Numerical);
        }
        out[i] = value;
    }
    Ok(())
}
fn reverse<D, F, O>(
    d: &Observations<D, F, O>,
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
