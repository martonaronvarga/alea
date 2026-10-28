//! Validated, parameter-bound joint reaction-time/response laws.
use super::*;
use alea_core::distribution::{
    Distribution, EventShape, ParameterScore, ReferenceMeasure, Sample, Support,
};
use rand::Rng;

macro_rules! law {
    ($name:ident, $kernel:ident, $params:ident, $n:literal, $valid:expr, $tau:expr) => {
        /// A joint law of response boundary and reaction time. Density is relative
        /// to time Lebesgue measure times boundary counting measure. Parameters
        /// and numerical options are validated once and cannot be mutated.
        #[derive(Debug, Clone, Copy)]
        #[must_use]
        pub struct $name {
            parameters: $params,
            options: WienerOptions,
        }
        impl $name {
            /// # Errors
            /// Rejects invalid parameters, variability intervals or numerical options.
            pub fn new(parameters: $params, options: WienerOptions) -> Result<Self> {
                if !($valid)(&parameters) {
                    return Err(ProbError::InvalidParameters(
                        "invalid bound Wiener law parameters".into(),
                    ));
                }
                Ok(Self {
                    parameters,
                    options: options.validate()?,
                })
            }
            pub fn parameters(&self) -> &$params {
                &self.parameters
            }
            pub fn options(&self) -> WienerOptions {
                self.options
            }
        }
        impl Distribution for $name {
            type Error = ProbError;
            type Event = WienerObservation;
            fn event_shape(&self) -> EventShape {
                EventShape::Mixed {
                    continuous: 1,
                    discrete: 1,
                }
            }
            fn support(&self) -> Support {
                Support::LowerBounded
            }
            fn reference_measure(&self) -> ReferenceMeasure {
                ReferenceMeasure::LebesgueCounting { dimension: 1 }
            }
            fn log_prob(&self, event: &Self::Event) -> Result<f64> {
                if event.rt.is_nan() {
                    return Err(alea_core::distribution::DistributionError::Observation.into());
                }
                if event.rt <= ($tau)(&self.parameters) || event.rt == f64::INFINITY {
                    return Ok(f64::NEG_INFINITY);
                }
                $kernel
                    .try_log_prob(event, &self.parameters, self.options)
                    .map(|e| e.log_prob)
            }
        }
        impl ParameterScore for $name {
            fn parameter_count(&self) -> usize {
                $n
            }
            fn parameter_score(&self, event: &Self::Event, output: &mut [f64]) -> Result<()> {
                if output.len() != $n {
                    return Err(alea_core::distribution::DistributionError::Shape.into());
                }
                let eval = $kernel.try_fused(event, &self.parameters, self.options)?;
                output.copy_from_slice(&eval.grad_array());
                Ok(())
            }
        }
        impl Sample for $name {
            type Draw = WienerObservation;
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Self::Draw> {
                $kernel.sample(&self.parameters, rng)
            }
        }
    };
}
law!(
    Wiener4Law,
    Wiener4,
    Wiener4Params,
    4,
    |p: &Wiener4Params| p.valid(),
    |p: &Wiener4Params| p.tau
);
law!(
    Wiener5Law,
    Wiener5,
    Wiener5Params,
    5,
    |p: &Wiener5Params| p.base.valid() && p.s_delta.is_finite() && p.s_delta >= 0.,
    |p: &Wiener5Params| p.base.tau
);
law!(
    Wiener7Law,
    Wiener7,
    Wiener7Params,
    7,
    |p: &Wiener7Params| p.base.base.valid()
        && p.base.s_delta.is_finite()
        && p.base.s_delta >= 0.
        && p.s_beta.is_finite()
        && p.s_beta >= 0.
        && p.s_tau.is_finite()
        && p.s_tau >= 0.
        && p.base.base.beta - p.s_beta / 2. > 0.
        && p.base.base.beta + p.s_beta / 2. < 1.,
    |p: &Wiener7Params| p.base.base.tau
);
