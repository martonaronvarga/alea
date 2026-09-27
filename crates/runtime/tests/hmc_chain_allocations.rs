#![cfg(not(miri))]

use kernels::{
    buffer::OwnedBuffer,
    dist::Gaussian,
    metric::{CholeskyFactor, DenseMetric, DiagonalMetric, IdentityMetric, Metric},
    target::FusedAdapter,
};
use rand::{SeedableRng, rngs::SmallRng};
use runtime::{HmcChain, HmcOptions};
use std::hint::black_box;

fn check<M: Metric>(metric: M, options: HmcOptions) {
    check_count(metric, options, 100);
}

fn check_count<M: Metric>(metric: M, options: HmcOptions, transitions: usize) {
    let dimension = metric.dim();
    let target = FusedAdapter::new(&Gaussian, dimension);
    let mut chain = HmcChain::new(&target, OwnedBuffer::new(dimension), metric, options).unwrap();
    let mut rng = SmallRng::seed_from_u64(89);
    let count = allocation_counter::measure(|| {
        for _ in 0..transitions {
            let info = black_box(chain.step(&mut rng).unwrap());
            // An early numerical exit must not masquerade as a full-trajectory
            // allocation measurement. The intentional divergence is endpoint-only.
            assert_eq!(info.leapfrog_steps, options.leapfrog_steps().get());
            black_box(chain.point().position());
        }
    });
    assert_eq!(count.count_total, 0);
    assert_eq!(count.bytes_total, 0);
}

#[test]
fn larger_metrics_and_simd_remainders_allocate_nothing() {
    for dimension in [7, 33, 129] {
        check_count(IdentityMetric::new(dimension), HmcOptions::default(), 8);
        check_count(
            DiagonalMetric::new(OwnedBuffer::from_fn(dimension, |i| 1.0 + (i % 3) as f64)).unwrap(),
            HmcOptions::default(),
            8,
        );
        // A nontrivial lower bidiagonal Cholesky factor in column-major storage.
        let lower = OwnedBuffer::from_fn(dimension * dimension, |i| {
            let (row, col) = (i % dimension, i / dimension);
            if row == col {
                1.5
            } else if row == col + 1 {
                0.2
            } else {
                0.0
            }
        });
        check_count(
            DenseMetric::new(CholeskyFactor::new_lower(dimension, lower).unwrap()),
            HmcOptions::default(),
            8,
        );
    }
}

#[test]
fn steady_state_hmc_and_divergent_trajectories_allocate_nothing() {
    let probe = allocation_counter::measure(|| {
        black_box(Box::new(42));
    });
    assert!(probe.count_total >= 1);
    check(IdentityMetric::new(2), HmcOptions::default());
    check(
        DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| [4.0, 0.5][i])).unwrap(),
        HmcOptions::default(),
    );
    let lower = OwnedBuffer::from_fn(4, |i| [2.0, 0.5, 0.0, 1.3][i]);
    check(
        DenseMetric::new(CholeskyFactor::new_lower(2, lower).unwrap()),
        HmcOptions::default(),
    );
    check(
        IdentityMetric::new(2),
        HmcOptions::new(1.0, 3)
            .unwrap()
            .with_max_energy_error(1e-30)
            .unwrap(),
    );
}
