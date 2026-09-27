use kernels::{density::FusedLogDensity, metric::Metric, state::GradientState};

#[inline]
pub fn leapfrog_step<D, M, S>(
    metric: &M,
    step_size: f64,
    state: &mut S,
    target: &D,
    velocity: &mut [f64],
) where
    D: FusedLogDensity<Point = [f64], Gradient = [f64]>,
    M: Metric,
    S: GradientState,
{
    assert_eq!(state.dim(), velocity.len(), "velocity dimension mismatch");
    assert_eq!(
        state.momentum().len(),
        velocity.len(),
        "momentum dimension mismatch"
    );
    velocity.copy_from_slice(state.gradient());
    for (momentum, gradient) in state.momentum_mut().iter_mut().zip(velocity.iter()) {
        *momentum += 0.5 * step_size * gradient;
    }
    metric.apply_inverse(state.momentum(), velocity);
    for (q, v) in state.position_mut().iter_mut().zip(velocity.iter()) {
        *q += step_size * v;
    }
    let log_prob = state.with_position_and_gradient_mut(|position, gradient| {
        target.log_prob_and_grad(position, gradient)
    });
    state.set_log_prob(log_prob);
    velocity.copy_from_slice(state.gradient());
    for (momentum, gradient) in state.momentum_mut().iter_mut().zip(velocity.iter()) {
        *momentum += 0.5 * step_size * gradient;
    }
}
