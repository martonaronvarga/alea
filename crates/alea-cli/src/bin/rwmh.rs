use rand::SeedableRng;
use rand::rngs::SmallRng;

use alea_distributions::Gaussian;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{CholeskyFactor, DenseMetric, EuclideanMetric, IdentityMetric},
};
use alea_mcmc::{Rwmh, RwmhOptions};
use std::env;

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::error::Error + 'static,
{
    for arg in env::args() {
        if let Some(val) = arg.strip_prefix(&format!("--{}=", name)) {
            return Ok(val.parse()?);
        }
    }
    Ok(default)
}
fn has_flag(name: &str) -> bool {
    env::args().any(|arg| arg == format!("--{}", name))
}

fn base_config(dim: usize) -> RwmhOptions {
    RwmhOptions::new(2.38 / (dim as f64).sqrt()).expect("positive dimension")
}

fn make_dense_factor(dim: usize) -> CholeskyFactor {
    let mut chol = OwnedBuffer::new(dim * dim);
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

fn run<M: EuclideanMetric>(
    metric: M,
    dim: usize,
    n_steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let target = Gaussian::new(dim);
    let config = base_config(dim);

    let mut chain = Rwmh::new(&target, OwnedBuffer::from_fn(dim, |_| 0.1), metric, config)?;

    let mut rng = SmallRng::seed_from_u64(42);

    let mut accepted = 0usize;
    for _ in 0..n_steps {
        if chain.step(&mut rng)?.accepted {
            accepted += 1;
        }
    }

    let checksum =
        chain.point().position().iter().copied().sum::<f64>() + chain.point().log_density();
    println!(
        "dim={} steps={} accept_rate={} checksum={}",
        dim,
        n_steps,
        alea_runtime::diagnostics::acceptance_rate(accepted, n_steps),
        checksum
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim: usize = parse_arg("dim", 256)?;
    let n_steps: usize = parse_arg("steps", 1_000_000)?;
    if dim == 0
        || dim
            .checked_mul(dim)
            .is_none_or(|n| n > (isize::MAX as usize - 64) / 8)
    {
        return Err("dimension must be positive and fit aligned storage".into());
    }

    if has_flag("dense") {
        run(DenseMetric::new(make_dense_factor(dim)), dim, n_steps)?;
    } else {
        run(IdentityMetric::new(dim), dim, n_steps)?;
    }
    Ok(())
}
