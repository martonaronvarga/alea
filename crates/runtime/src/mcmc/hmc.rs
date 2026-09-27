use kernels::{
    buffer::OwnedBuffer,
    density::FusedLogDensity,
    kernel::Kernel,
    metric::Metric,
    state::{GradientState, LogProbState},
};
use rand::{Rng, RngExt};
use tracing::{debug, trace};

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

/// Experimental fixed-length Euclidean HMC with reusable proposal storage.
///
/// The metric represents the momentum mass matrix `M`: momenta have covariance
/// `M`, and velocity is `M^-1 p`. Targets must be pure fused evaluations.
/// Non-finite trajectories are rejected; this legacy API has no typed failure
/// reporting. See [`super::hmc_chain::HmcChain`] for the target-bound, fallible API.
/// Mutable public chain state requires refreshing caches at the
/// start of every transition (one extra fused evaluation per transition).
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
    D: FusedLogDensity<Point = [f64], Gradient = [f64]>,
{
    type State = S;

    fn initialize(&mut self, state: &mut Self::State, target: &D) {
        let log_prob = state.with_position_and_gradient_mut(|position, gradient| {
            target.log_prob_and_grad(position, gradient)
        });
        state.set_log_prob(log_prob);
    }

    fn step<R: Rng + ?Sized>(&mut self, state: &mut Self::State, target: &D, rng: &mut R) -> bool {
        // State is currently publicly mutable, so refresh both caches together.
        assert_eq!(state.dim(), self.dim, "HMC position dimension mismatch");
        assert_eq!(
            state.gradient().len(),
            self.dim,
            "HMC gradient dimension mismatch"
        );
        assert_eq!(
            state.momentum().len(),
            self.dim,
            "HMC momentum dimension mismatch"
        );
        self.initialize(state, target);
        if !state.log_prob().is_finite()
            || !state.position().iter().all(|q| q.is_finite())
            || !state.gradient().iter().all(|g| g.is_finite())
            || !self.config.step_size.is_finite()
            || self.config.step_size <= 0.0
            || self.config.n_leapfrog == 0
        {
            return false;
        }

        for z in self.velocity.iter_mut() {
            *z = crate::random::standard_normal(rng);
        }
        self.metric
            .apply_sqrt(self.velocity.as_slice(), state.momentum_mut());
        let current_h = -state.log_prob()
            + Self::kinetic_energy_with(
                &self.metric,
                state.momentum(),
                self.velocity.as_mut_slice(),
            );
        if !current_h.is_finite() {
            return false;
        }

        self.proposal_position.copy_from_slice(state.position());
        self.proposal_momentum.copy_from_slice(state.momentum());
        self.proposal_gradient.copy_from_slice(state.gradient());

        let (proposal_h, proposal_lp) = {
            let mut proposal = kernels::state::ChainState::with_aux(
                self.proposal_position.as_mut_slice(),
                kernels::state::GradientBuffers {
                    gradient: self.proposal_gradient.as_mut_slice(),
                    momentum: self.proposal_momentum.as_mut_slice(),
                },
            );
            proposal.set_log_prob(state.log_prob());

            trace!(
                n_leapfrog = self.config.n_leapfrog,
                "Starting leapfrog integration"
            );
            for _ in 0..self.config.n_leapfrog {
                leapfrog_step(
                    &self.metric,
                    self.config.step_size,
                    &mut proposal,
                    target,
                    self.proposal_velocity.as_mut_slice(),
                );
                if !proposal.log_prob().is_finite()
                    || !proposal.position().iter().all(|q| q.is_finite())
                    || !proposal.gradient().iter().all(|g| g.is_finite())
                    || !proposal.momentum().iter().all(|p| p.is_finite())
                {
                    debug!("Rejecting non-finite HMC trajectory");
                    return false;
                }
            }

            let proposal_lp = proposal.log_prob();
            let energy = -proposal_lp
                + Self::kinetic_energy_with(
                    &self.metric,
                    proposal.momentum(),
                    self.proposal_velocity.as_mut_slice(),
                );
            (energy, proposal_lp)
        };

        if !proposal_h.is_finite() {
            return false;
        }
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
            state.set_log_prob(proposal_lp);
        }

        accepted
    }
}
