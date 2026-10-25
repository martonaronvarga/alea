//! Generated quantities and predictive sampling outside the fused gradient loop.
use super::assembly::Coordinate;
use crate::transform::{ParameterLayout, TransformError};
use rand::Rng;
use std::sync::Arc;
/// A schema-bearing output wrapper; T may be runtime draws, predictions or diagnostics.
#[derive(Debug, Clone)]
pub struct NamedOutput<T> {
    coordinates: Arc<[Coordinate]>,
    layout: ParameterLayout,
    pub values: T,
}
impl<T> NamedOutput<T> {
    /// Attaches constrained parameter names and their transform to an output.
    /// Names describe model provenance; generated values have their own event shape.
    /// # Errors
    /// Rejects mismatched dimensions, empty/duplicate names, or labels that split
    /// one indivisible transform block across different geometry groups.
    pub fn new(
        coordinates: Arc<[Coordinate]>,
        layout: ParameterLayout,
        values: T,
    ) -> Result<Self, TransformError> {
        if coordinates.len() != layout.constrained_dimension() {
            return Err(TransformError::Dimension {
                expected: layout.constrained_dimension(),
                actual: coordinates.len(),
            });
        }
        if coordinates.iter().enumerate().any(|(i, c)| {
            c.name.is_empty() || coordinates[..i].iter().any(|old| old.name == c.name)
        }) {
            return Err(TransformError::InvalidDimension);
        }
        for block in layout.blocks() {
            let entries = &coordinates[block.constrained_range()];
            if entries.iter().any(|c| c.block != entries[0].block) {
                return Err(TransformError::InvalidDimension);
            }
        }
        Ok(Self {
            coordinates,
            layout,
            values,
        })
    }
    pub fn coordinates(&self) -> &[Coordinate] {
        &self.coordinates
    }
    pub fn layout(&self) -> &ParameterLayout {
        &self.layout
    }
    /// Returns **unconstrained** slots grouped by declared geometry block.
    /// This cold operation does not alter numerical layout or infer a metric.
    pub fn blocks(&self) -> Vec<(String, Vec<usize>)> {
        let mut result: Vec<(String, Vec<usize>)> = Vec::new();
        for parameter in self.layout.blocks() {
            let constrained = parameter.constrained_range();
            if constrained.is_empty() {
                continue;
            }
            if let Some(name) = &self.coordinates[constrained.start].block {
                if let Some((_, slots)) = result.iter_mut().find(|(n, _)| n == name) {
                    slots.extend(parameter.unconstrained_range());
                } else {
                    result.push((name.clone(), parameter.unconstrained_range().collect()));
                }
            }
        }
        result
    }
}
/// Predictions preserve whether they came from prior or posterior coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveOrigin {
    Prior,
    Posterior,
}
/// Prediction output with explicit provenance, separate from retained draws.
#[derive(Debug, Clone)]
pub struct Predictions<T> {
    pub origin: PredictiveOrigin,
    pub generated: Vec<T>,
}
/// Checked predictive workflow failure with the original generator error retained.
#[derive(Debug, thiserror::Error)]
pub enum PredictionError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Transform(#[from] TransformError),
    #[error("predictive generator failed: {0}")]
    Generator(#[source] E),
}
/// Constrains posterior draws without adding a Jacobian to generated quantities.
/// `generate` can sample observations or compute deterministic summaries.
///
/// # Errors
/// Propagates layout and generator failures. RNG consumption is not rolled back.
pub fn posterior_predict<'a, T, E, R, F>(
    layout: &ParameterLayout,
    draws: impl IntoIterator<Item = &'a [f64]>,
    rng: &mut R,
    mut generate: F,
) -> Result<Predictions<T>, PredictionError<E>>
where
    E: std::error::Error + 'static,
    R: Rng + ?Sized,
    F: FnMut(&[f64], &mut R) -> Result<T, E>,
{
    let mut parameters = vec![0.0; layout.constrained_dimension()];
    let mut scratch = vec![0.0; layout.scratch_dimension()];
    let mut generated = Vec::new();
    for q in draws {
        layout.constrain(q, &mut parameters, &mut scratch)?;
        generated.push(generate(&parameters, rng).map_err(PredictionError::Generator)?);
    }
    Ok(Predictions {
        origin: PredictiveOrigin::Posterior,
        generated,
    })
}
/// Runs a caller's generative prior workflow, with no sampler or gradient evaluation.
/// # Errors
/// Propagates the prior generator failure; RNG consumption is not rolled back.
pub fn prior_predict<T, E, R, F>(
    count: usize,
    rng: &mut R,
    mut generate: F,
) -> Result<Predictions<T>, E>
where
    R: Rng + ?Sized,
    F: FnMut(&mut R) -> Result<T, E>,
{
    let generated = (0..count)
        .map(|_| generate(rng))
        .collect::<Result<_, _>>()?;
    Ok(Predictions {
        origin: PredictiveOrigin::Prior,
        generated,
    })
}
