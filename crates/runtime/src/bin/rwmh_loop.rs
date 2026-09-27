use rand::SeedableRng;
use rand::rngs::SmallRng;

use kernels::dist::Gaussian;
use kernels::{
    kernel::Kernel,
    metric::{CholeskyFactor, DenseMetric, IdentityMetric, Metric},
    state::{ChainState, LogProbState},
};
use runtime::mcmc::rwmh::{Rwmh, RwmhConfig};
use std::env;

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    for arg in env::args() {
        if let Some(val) = arg.strip_prefix(&format!("--{}=", name))
            && let Ok(parsed) = val.parse()
        {
            return parsed;
        }
    }
    default
}
fn has_flag(name: &str) -> bool {
    env::args().any(|arg| arg == format!("--{}", name))
}

fn base_config(dim: usize) -> RwmhConfig {
    let step_size = 2.38 / (dim as f64).sqrt();
    RwmhConfig::default()
        .with_warmup(0)
        .with_draws(0)
        .with_adapt_step_size(false)
        .with_step_size(step_size)
}

fn make_dense_factor(dim: usize) -> CholeskyFactor {
    let mut chol = kernels::buffer::OwnedBuffer::new(dim * dim);
    for j in 0..dim {
        for i in 0..dim {
            chol.as_mut_slice()[i + j * dim] = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                0.001 * (((i + 1) as f64) * ((j + 1) as f64)).sin()
            };
        }
    }
    CholeskyFactor::new_lower(dim, chol).expect("generated unit-diagonal lower factor is valid")
}

fn run<M: Metric>(metric: M, dim: usize, n_steps: usize) {
    let target = Gaussian;
    let config = base_config(dim);

    let mut kernel = Rwmh::new(config, metric);
    let mut state = ChainState::new(dim);
    state.position.fill(0.1);

    let mut rng = SmallRng::seed_from_u64(42);

    let mut accepted = 0usize;
    for _ in 0..n_steps {
        if kernel.step(&mut state, &target, &mut rng) {
            accepted += 1;
        }
    }

    let checksum = state.position().iter().copied().sum::<f64>() + state.log_prob();
    println!(
        "dim={} steps={} accept_rate={} checksum={}",
        dim,
        n_steps,
        accepted as f64 / n_steps as f64,
        checksum
    );
}

fn main() {
    let dim: usize = parse_arg("dim", 256);
    let n_steps: usize = parse_arg("steps", 1_000_000);

    if has_flag("dense") {
        run(DenseMetric::new(make_dense_factor(dim)), dim, n_steps);
    } else {
        run(IdentityMetric::new(dim), dim, n_steps);
    }
}
