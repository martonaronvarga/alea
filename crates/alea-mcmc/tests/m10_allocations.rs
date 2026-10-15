#![cfg(not(miri))]

use std::{convert::Infallible, hint::black_box};

use alea_core::{
    capability::{
        BatchLane, BatchLayout, BatchLogDensityGradient, BatchShape, GaussianResidual,
        HessianVector, SplitLogDensityGradient,
    },
    target::LogDensityGradient,
};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    AdvancedHmcOptions, BatchedLeapfrog, ForceGradientHmc, ForceGradientMode, GaussianSplitHmc,
    SplitForceHmc, hamiltonian::SignedStep,
};
use rand::{SeedableRng, rngs::SmallRng};

struct Normal;
impl LogDensityGradient for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        4
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let mut squared = 0.0;
        for (gradient, q) in gradient.iter_mut().zip(q) {
            *gradient = -*q;
            squared += q * q;
        }
        Ok(-0.5 * squared)
    }
}
impl HessianVector for Normal {
    fn potential_hvp(
        &self,
        _: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        output.copy_from_slice(vector);
        Ok(())
    }
}
impl SplitLogDensityGradient for Normal {
    fn cheap_score(&self, q: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        for (output, q) in output.iter_mut().zip(q) {
            *output = -0.75 * q;
        }
        Ok(())
    }
    fn expensive_score(&self, q: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        for (output, q) in output.iter_mut().zip(q) {
            *output = -0.25 * q;
        }
        Ok(())
    }
}
impl GaussianResidual for Normal {
    fn gaussian_center(&self) -> &[f64] {
        &[0.0; 4]
    }
    fn gaussian_precision(&self) -> &[f64] {
        &[1.0; 4]
    }
    fn residual_score(&self, _: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        output.fill(0.0);
        Ok(())
    }
}
impl BatchLogDensityGradient for Normal {
    fn batch_logp_grad(
        &self,
        shape: BatchShape,
        positions: &[f64],
        gradients: &mut [f64],
        results: &mut [BatchLane<Self::Error>],
    ) {
        for (chain, result) in results.iter_mut().enumerate() {
            let mut squared = 0.0;
            for coordinate in 0..shape.dimension() {
                let index = shape.index(chain, coordinate).unwrap();
                gradients[index] = -positions[index];
                squared += positions[index] * positions[index];
            }
            *result = BatchLane::Complete(Ok(-0.5 * squared));
        }
    }
}

#[test]
fn retained_advanced_kernels_and_batch_fsm_allocate_nothing() {
    let target = Normal;
    let options = AdvancedHmcOptions::new(0.1, 3).unwrap();
    let mut gaussian = GaussianSplitHmc::new(&target, OwnedBuffer::new(4), options).unwrap();
    let mut split = SplitForceHmc::new(
        &target,
        OwnedBuffer::new(4),
        IdentityMetric::new(4),
        options,
        2,
    )
    .unwrap();
    let mut exact = ForceGradientHmc::new(
        &target,
        OwnedBuffer::new(4),
        IdentityMetric::new(4),
        options,
        ForceGradientMode::AnalyticHvp,
    )
    .unwrap();
    let mut hessian_free = ForceGradientHmc::new(
        &target,
        OwnedBuffer::new(4),
        IdentityMetric::new(4),
        options,
        ForceGradientMode::HessianFree,
    )
    .unwrap();
    let mut batch = BatchedLeapfrog::new(
        &target,
        8,
        BatchLayout::CoordinateMajor,
        OwnedBuffer::new(32),
        OwnedBuffer::new(32),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(10_010);
    let allocations = allocation_counter::measure(|| {
        for _ in 0..32 {
            let _ = black_box(gaussian.step(&mut rng).unwrap());
            let _ = black_box(split.step(&mut rng).unwrap());
            let _ = black_box(exact.step(&mut rng).unwrap());
            let _ = black_box(hessian_free.step(&mut rng).unwrap());
            black_box(batch.advance(SignedStep::new(0.03).unwrap(), 4).unwrap());
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}
