use kernels::{
    buffer::OwnedBuffer,
    density::{GradLogDensity, LogDensity},
    kernel::Kernel,
    metric::Metric,
    state::{GradientState, LogProbState},
};
use rand::{Rng, RngExt};

use crate::integrator::leapfrog_step;

#[derive(Debug, Clone, Copy)]
pub struct HmcConfig {
    pub step_size: f64,
    pub n_leapfrog: usize,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            n_leapfrog: 10,
        }
    }
}

pub struct Hmc<M, S> {
    pub config: HmcConfig,
    metric: M,
    dim: usize,
    velocity: OwnedBuffer,
    proposal_velocity: OwnedBuffer,
    proposal_position: OwnedBuffer,
    proposal_momentum: OwnedBuffer,
    proposal_gradient: OwnedBuffer,
    _marker: std::marker::PhantomData<S>,
}

impl<M, S> Hmc<M, S>
where
    M: Metric,
    S: GradientState,
{
    #[inline]
    fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        loop {
            let u = 2.0 * rng.random::<f64>() - 1.0;
            let v = 2.0 * rng.random::<f64>() - 1.0;
            let s = u * u + v * v;
            if (0.0..1.0).contains(&s) {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }

    pub fn new(config: HmcConfig, metric: M) -> Self {
        let dim = metric.dim();
        Self {
            config,
            metric,
            dim,
            velocity: OwnedBuffer::new(dim),
            proposal_velocity: OwnedBuffer::new(dim),
            proposal_position: OwnedBuffer::new(dim),
            proposal_momentum: OwnedBuffer::new(dim),
            proposal_gradient: OwnedBuffer::new(dim),
            _marker: std::marker::PhantomData,
        }
    }

    fn kinetic_energy_with(metric: &M, momentum: &[f64], velocity: &mut [f64]) -> f64 {
        metric.apply_inverse(momentum, velocity);
        0.5 * momentum
            .iter()
            .zip(velocity.iter())
            .map(|(p, v)| p * v)
            .sum::<f64>()
    }
}

impl<M, S, D> Kernel<D> for Hmc<M, S>
where
    M: Metric,
    S: GradientState,
    D: GradLogDensity + LogDensity,
{
    type State = S;

    fn initialize(&mut self, state: &mut Self::State, target: &D) {
        if !state.log_prob().is_finite() {
            state.initialize_log_prob(target);
        }
        state.initialize_gradient(target);
    }

    fn step<R: Rng + ?Sized>(&mut self, state: &mut Self::State, target: &D, rng: &mut R) -> bool {
        if !state.log_prob().is_finite() {
            self.initialize(state, target);
        }

        for p in state.momentum_mut().iter_mut() {
            *p = Self::standard_normal(rng);
        }
        let current_h = -state.log_prob()
            + Self::kinetic_energy_with(
                &self.metric,
                state.momentum(),
                self.velocity.as_mut_slice(),
            );

        self.proposal_position.copy_from_slice(state.position());
        self.proposal_momentum.copy_from_slice(state.momentum());
        self.proposal_gradient.copy_from_slice(state.gradient());

        let proposal_h = {
            let mut proposal = kernels::state::State::with_aux(
                self.proposal_position.as_mut_slice(),
                kernels::state::GradientBuffers {
                    gradient: self.proposal_gradient.as_mut_slice(),
                    momentum: self.proposal_momentum.as_mut_slice(),
                },
            );
            proposal.set_log_prob(state.log_prob());

            trace!(
                n_leapfrog = self.config.n_leapfrog,
                "Startig leapfrog integration"
            );
            for i in 0..self.config.n_leapfrog {
                leapfrog_step(
                    &self.metric,
                    self.config.step_size,
                    &mut proposal,
                    target,
                    self.proposal_velocity.as_mut_slice(),
                );
            }

            let proposal_lp = target.log_prob(proposal.position());
            proposal.set_log_prob(proposal_lp);
            if !proposal_lp.is_finite() {
                debug!("Divergence encountered: proposal log_prob is not finite");
            }
            -proposal.log_prob()
                + Self::kinetic_energy_with(
                    &self.metric,
                    proposal.momentum(),
                    self.proposal_velocity.as_mut_slice(),
                )
        };

        let accept_prob = (current_h - proposal_h).min(0.0).exp();
        let accepted = rng.random::<f64>() < accept_prob;

        trace!(
            current_h = current_h,
            proposal_h = proposal_h,
            accept_prob = accept_prob,
            accepted = accepted,
            "HMC step completed"
        );

        if accepted {
            state
                .position_mut()
                .copy_from_slice(self.proposal_position.as_slice());
            state
                .gradient_mut()
                .copy_from_slice(self.proposal_gradient.as_slice());
            state.set_log_prob(target.log_prob(self.proposal_position.as_slice()));
        }

        accepted
    }
}
