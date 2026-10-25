use std::{cell::Cell, convert::Infallible};

use alea_core::target::LogDensityGradient;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric, MetricError},
};
use alea_mcmc::{Nuts, NutsOptions};
use rand::{SeedableRng, rngs::SmallRng};

struct Gaussian {
    dimension: usize,
    calls: Cell<usize>,
}

impl LogDensityGradient for Gaussian {
    type Error = Infallible;

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        let mut square_sum = 0.0;
        for (q, g) in q.iter().zip(g) {
            *g = -*q;
            square_sum += q * q;
        }
        Ok(-0.5 * square_sum)
    }
}

struct CountingMetric {
    inner: IdentityMetric,
    velocities: Cell<usize>,
    momentum_samples: Cell<usize>,
}

impl EuclideanMetric for CountingMetric {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn velocity(&self, momentum: &[f64], velocity: &mut [f64]) -> Result<(), MetricError> {
        self.velocities.set(self.velocities.get() + 1);
        self.inner.velocity(momentum, velocity)
    }

    fn sample_momentum(&self, z: &[f64], p: &mut [f64]) -> Result<(), MetricError> {
        self.momentum_samples.set(self.momentum_samples.get() + 1);
        self.inner.sample_momentum(z, p)
    }

    fn log_det(&self) -> f64 {
        self.inner.log_det()
    }
}

fn main() {
    println!(
        "dimension,max_depth,integration_steps,gradient_calls,velocity_calls,momentum_samples,termination"
    );
    for dimension in [2, 16, 64] {
        for max_depth in [1, 3, 8] {
            let target = Gaussian {
                dimension,
                calls: Cell::new(0),
            };
            let metric = CountingMetric {
                inner: IdentityMetric::new(dimension),
                velocities: Cell::new(0),
                momentum_samples: Cell::new(0),
            };
            let mut chain = Nuts::new(
                &target,
                OwnedBuffer::new(dimension),
                metric,
                NutsOptions::new(1e-4, max_depth).unwrap(),
            )
            .unwrap();
            target.calls.set(0);
            chain.metric().velocities.set(0);
            chain.metric().momentum_samples.set(0);
            let mut rng = SmallRng::seed_from_u64(20261022);
            let outcome = chain.step(&mut rng).unwrap();
            assert_eq!(outcome.integration_steps, (1 << max_depth) - 1);
            assert_eq!(target.calls.get(), outcome.integration_steps);
            assert_eq!(
                chain.metric().velocities.get(),
                1 + 2 * outcome.integration_steps
            );
            println!(
                "{dimension},{max_depth},{},{},{},{},{:?}",
                outcome.integration_steps,
                target.calls.get(),
                chain.metric().velocities.get(),
                chain.metric().momentum_samples.get(),
                outcome.termination
            );
        }
    }
}
