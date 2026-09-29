//! State-space and particle-kernel contracts. Particle filtering and PMCMC are not yet implemented.
use rand::Rng;

pub trait ParticleWeightModel<S, Y> {
    fn log_weight(&self, state: &S, obs: &Y, t: usize) -> f64;
}

pub trait ParticleMutationKernel<S> {
    fn propagate<R: Rng + ?Sized>(&self, prev: &S, t: usize, rng: &mut R, out: &mut S);
}

pub trait ParticleResampler {
    fn resample<R: Rng + ?Sized>(
        &mut self,
        normalized_weights: &[f64],
        rng: &mut R,
        out_index: &mut [usize],
    );
}
