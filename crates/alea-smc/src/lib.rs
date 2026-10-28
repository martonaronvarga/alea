//! Particle-operation contracts. Filtering and PMCMC are not implemented.
use rand::Rng;

/// Incremental importance weight, including target/proposal corrections.
/// This is not generally an observation likelihood. `prev` is absent at
/// initialization; the implementation owns the proposal/target specification.
pub trait ParticleWeightModel<S, Y> {
    type Error: std::error::Error + 'static;
    fn log_weight(
        &self,
        prev: Option<&S>,
        state: &S,
        obs: &Y,
        t: usize,
    ) -> Result<f64, Self::Error>;
}
/// Proposal propagation, distinct from the model's generative transition.
/// Output is scratch after a failure; callers commit only successful proposals.
pub trait ParticleMutationKernel<S> {
    type Error: std::error::Error + 'static;
    fn propagate<R: Rng + ?Sized>(
        &self,
        prev: &S,
        t: usize,
        rng: &mut R,
        out: &mut S,
    ) -> Result<(), Self::Error>;
}
/// Resampling from normalized nonnegative weights. Implementations must reject
/// invalid weights and output dimensions rather than invent valid ancestors.
pub trait ParticleResampler {
    type Error: std::error::Error + 'static;
    fn resample<R: Rng + ?Sized>(
        &mut self,
        normalized_weights: &[f64],
        rng: &mut R,
        out_index: &mut [usize],
    ) -> Result<(), Self::Error>;
}
