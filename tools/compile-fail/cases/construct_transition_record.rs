use alea_mcmc::prelude::HmcTransition;

fn transition() -> HmcTransition {
    HmcTransition {
        accepted: false,
        acceptance_probability: 0.0,
        integration_steps: 0,
        initial_energy: None,
        proposal_energy: None,
        energy_error: None,
        divergence: None,
    }
}

fn main() {
    let _ = transition;
}
