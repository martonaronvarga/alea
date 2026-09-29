use rand::Rng;

pub trait BatchLogDensity {
    fn log_prob_batch(&self, batch: &[&[f64]], out: &mut [f64]);
}

pub trait FactorEnergy<S> {
    fn unary_energy(&self, node: usize, state: &S) -> f64;
    fn pairwise_energy(&self, edge: (usize, usize), left: &S, right: &S) -> f64;
}

pub trait DiscreteTransitionKernel<S> {
    fn log_transition_prob(&self, prev: &S, next: &S, t: usize) -> f64;
    fn sample_transition<R: Rng + ?Sized>(&self, prev: &S, t: usize, rng: &mut R, out: &mut S);
}
