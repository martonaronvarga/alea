use alea::distribution::{
    Distribution, DistributionError, EventShape, ReferenceMeasure, Sample, Support,
};
use rand::Rng;
struct Law;
impl Distribution for Law {
    type Error = DistributionError;
    type Event = f64;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn reference_measure(&self) -> ReferenceMeasure {
        ReferenceMeasure::Lebesgue { dimension: 1 }
    }
    fn log_prob(&self, x: &f64) -> Result<f64, Self::Error> {
        alea::distributions::scalar::Normal::new(0., 1.)?.log_prob(x)
    }
}
impl Sample for Law {
    type Draw = bool;
    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> Result<bool, Self::Error> {
        Ok(true)
    }
}
fn main() {}
