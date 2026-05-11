use kernels::kernel::Kernel;
use rand::Rng;
use tracing::{info, instrument, trace};

pub struct Chain<K, D>
where
    K: Kernel<D>,
{
    pub kernel: K,
    pub target: D,
    pub state: K::State,
}
impl<K, D> Chain<K, D>
where
    K: Kernel<D>,
{
    pub fn new(kernel: K, target: D, state: K::State) -> Self {
        Self {
            kernel,
            target,
            state,
        }
    }

    pub fn initialize(&mut self) {
        self.kernel.initialize(&mut self.state, &self.target);
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        self.kernel.step(&mut self.state, &self.target, rng)
    }
}

#[instrument(skip(chain, rng), fields(n_steps = n_steps), level = "info")]
pub fn run_chain<K, D, R>(chain: &mut Chain<K, D>, n_steps: usize, rng: &mut R)
where
    K: Kernel<D>,
    R: Rng + ?Sized,
{
    info!("Initializing MCMC chain");
    chain.initialize();

    info!("Starting sampling loop");
    let mut accepted_count = 0;

    for i in 0..n_steps {
        let accepted = chain.step(rng);
        if accepted {
            accepted_count += 1;
        }
        trace!(step = i, accepted = accepted, "Completed MCMC step");
    }
    let acceptance_rate = accepted_count as f64 / n_steps as f64;
    info!(acceptance_rate = acceptance_rate, "Finished sampling");
}
