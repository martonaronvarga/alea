//! Conditional state-space contracts, independent of particle algorithms.
/// A generative transition with consistent time conditioning for both operations.
pub trait TransitionModel<S> {
    type Error: std::error::Error + 'static;
    /// Log density/mass in the model's declared state coordinates and measure.
    fn log_transition(&self, prev: &S, next: &S, t: usize) -> Result<f64, Self::Error>;
    fn sample_next<R: rand::Rng + ?Sized>(
        &self,
        prev: &S,
        t: usize,
        rng: &mut R,
    ) -> Result<S, Self::Error>;
}
/// Observation likelihood conditioned on state and observation time.
pub trait ObservationModel<S, Y> {
    type Error: std::error::Error + 'static;
    fn log_likelihood(&self, state: &S, obs: &Y, t: usize) -> Result<f64, Self::Error>;
}
