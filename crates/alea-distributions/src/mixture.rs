//! Finite latent-category marginalization using stable log-sum-exp.
use crate::multivariate::Categorical;
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, ScalarEventScore,
    Support,
};
use rand::Rng;
/// Normalized homogeneous finite mixture. Zero-weight components are never evaluated.
#[derive(Debug)]
#[must_use]
pub struct FiniteMixture<D> {
    weights: Vec<f64>,
    components: Vec<D>,
    selector: Categorical,
}
impl<D: Distribution> FiniteMixture<D> {
    /// # Errors
    /// Rejects invalid weights, empty/mismatched components or differing event shapes or reference measures.
    pub fn new(weights: Vec<f64>, components: Vec<D>) -> Result<Self, E> {
        if components.is_empty() || components.len() != weights.len() {
            return Err(E::Shape);
        }
        let first = &components[0];
        if components.iter().any(|x| {
            x.event_shape() != first.event_shape()
                || x.reference_measure() != first.reference_measure()
        }) {
            return Err(E::Shape);
        }
        let selector = Categorical::new(weights.clone())?;
        Ok(Self {
            weights,
            components,
            selector,
        })
    }
}
impl<D: Distribution> Distribution for FiniteMixture<D> {
    type Error = D::Error;
    type Event = D::Event;
    fn event_shape(&self) -> EventShape {
        self.components[0].event_shape()
    }
    fn support(&self) -> Support {
        let first = self.components[0].support();
        if self.components.iter().all(|law| law.support() == first) {
            first
        } else {
            Support::Union
        }
    }
    fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
        self.components[0].reference_measure()
    }
    fn log_prob(&self, x: &Self::Event) -> Result<f64, D::Error> {
        let mut max = f64::NEG_INFINITY;
        for (w, d) in self.weights.iter().zip(&self.components) {
            if *w > 0.0 {
                let logp = d.log_prob(x)?;
                if logp.is_nan() {
                    return Err(E::Numerical.into());
                }
                max = max.max(w.ln() + logp);
            }
        }
        if max.is_infinite() {
            return Ok(max);
        }
        let mut sum = 0.0;
        for (w, d) in self.weights.iter().zip(&self.components) {
            if *w > 0.0 {
                let logp = d.log_prob(x)?;
                if logp.is_nan() {
                    return Err(E::Numerical.into());
                }
                sum += (w.ln() + logp - max).exp();
            }
        }
        Ok(max + sum.ln())
    }
}
impl<D: ScalarEventScore> ScalarEventScore for FiniteMixture<D> {
    fn event_score(&self, x: f64) -> Result<f64, D::Error> {
        let logp = self.log_prob(&x)?;
        if !logp.is_finite() {
            return Err(E::Numerical.into());
        }
        let mut value = 0.0;
        for (w, d) in self.weights.iter().zip(&self.components) {
            if *w > 0.0 {
                let r = (w.ln() + d.log_prob(&x)? - logp).exp();
                if r > 0.0 {
                    value += r * d.event_score(x)?;
                }
            }
        }
        if value.is_finite() {
            Ok(value)
        } else {
            Err(E::Numerical.into())
        }
    }
}
impl<D: Sample> Sample for FiniteMixture<D> {
    type Draw = D::Draw;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Self::Draw, D::Error> {
        self.components[self.selector.sample(rng)?].sample(rng)
    }
}
/// Ambient weight scores followed by each component's documented parameter scores.
/// Weights are simplex coordinates; their transform supplies the tangent pullback.
impl<D: ParameterScore> ParameterScore for FiniteMixture<D> {
    fn parameter_count(&self) -> usize {
        self.weights.len()
            + self
                .components
                .iter()
                .map(ParameterScore::parameter_count)
                .sum::<usize>()
    }
    fn parameter_score(&self, event: &Self::Event, output: &mut [f64]) -> Result<(), D::Error> {
        if output.len() != self.parameter_count() {
            return Err(E::Shape.into());
        }
        if self.weights.contains(&0.0) {
            return Err(E::Observation.into());
        }
        let logp = self.log_prob(event)?;
        if !logp.is_finite() {
            return Err(E::Numerical.into());
        }
        output.fill(0.0);
        let mut offset = self.weights.len();
        for (i, (weight, component)) in self.weights.iter().zip(&self.components).enumerate() {
            let n = component.parameter_count();
            if *weight > 0.0 {
                let responsibility = (weight.ln() + component.log_prob(event)? - logp).exp();
                output[i] = responsibility / weight;
                if responsibility > 0.0 {
                    component.parameter_score(event, &mut output[offset..offset + n])?;
                    for g in &mut output[offset..offset + n] {
                        *g *= responsibility;
                    }
                }
            }
            offset += n;
        }
        if output.iter().all(|g| g.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical.into())
        }
    }
}
