#![cfg(not(miri))]

use std::{convert::Infallible, hint::black_box};

use alea_core::{capability::PositionMetric, target::LogDensityGradient};
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::riemannian::{ExactnessPendingRmhmc, RmhmcOptions};
use rand::{SeedableRng, rngs::SmallRng};

struct Normal;
impl LogDensityGradient for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        4
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let mut squared = 0.0;
        for (gradient, value) in gradient.iter_mut().zip(q) {
            *gradient = -*value;
            squared += value * value;
        }
        Ok(-0.5 * squared)
    }
}

struct Identity;
impl PositionMetric for Identity {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        4
    }
    fn metric_and_derivatives(
        &self,
        _: &[f64],
        metric: &mut [f64],
        derivatives: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), Self::Error> {
        metric.fill(0.0);
        derivatives.fill(0.0);
        for index in 0..4 {
            metric[index * 4 + index] = 1.0;
        }
        Ok(())
    }
}

#[test]
fn retained_rmhmc_transition_allocates_nothing() {
    let target = Normal;
    let metric = Identity;
    let options = RmhmcOptions::new(0.1, 4, 2, 1.0e-14).unwrap();
    let mut chain =
        ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::new(4), options).unwrap();
    let mut rng = SmallRng::seed_from_u64(11_011);
    let allocations = allocation_counter::measure(|| {
        for _ in 0..64 {
            let _ = black_box(chain.step(&mut rng).unwrap());
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}
