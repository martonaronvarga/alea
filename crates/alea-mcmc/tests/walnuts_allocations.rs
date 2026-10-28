#![cfg(not(miri))]

use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::walnuts::{
    Walnuts, WalnutsMicrostepDistribution, WalnutsOptions, WalnutsTermination,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{convert::Infallible, hint::black_box};

struct Flat;

impl alea_core::target::TargetSpace for Flat {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        7
    }
}
impl LogDensityGradient for Flat {
    fn logp_grad(&self, _: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        gradient.fill(0.0);
        Ok(0.0)
    }
}

#[test]
fn transitions_allocate_nothing_at_the_full_tree_budget() {
    for distribution in [
        WalnutsMicrostepDistribution::Deterministic,
        WalnutsMicrostepDistribution::RandomizedTwoPoint,
    ] {
        for depth in [1, 5, 10] {
            let options = WalnutsOptions::new(0.1, depth)
                .unwrap()
                .with_microstep_distribution(distribution)
                .unwrap();
            let mut chain =
                Walnuts::new(&Flat, OwnedBuffer::new(7), IdentityMetric::new(7), options).unwrap();
            let mut rng = SmallRng::seed_from_u64(19);
            let allocation = allocation_counter::measure(|| {
                for _ in 0..8 {
                    let info = black_box(chain.step(&mut rng).unwrap());
                    assert_eq!(info.termination, WalnutsTermination::MaxDepth);
                    assert_eq!(info.macro_steps, (1 << depth) - 1);
                }
            });
            assert_eq!(allocation.count_total, 0);
            assert_eq!(allocation.bytes_total, 0);
        }
    }
}

#[test]
fn construction_storage_is_linear_in_depth() {
    let measure = |depth| {
        allocation_counter::measure(|| {
            black_box(
                Walnuts::new(
                    &Flat,
                    OwnedBuffer::new(7),
                    IdentityMetric::new(7),
                    WalnutsOptions::new(0.1, depth).unwrap(),
                )
                .unwrap(),
            );
        })
    };
    let a = measure(4);
    let b = measure(8);
    let c = measure(12);
    assert!(a.bytes_total > 0);
    assert_eq!(b.bytes_total - a.bytes_total, c.bytes_total - b.bytes_total);
    assert_eq!(b.count_total - a.count_total, c.count_total - b.count_total);
}
