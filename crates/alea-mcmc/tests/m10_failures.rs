use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

use alea_core::{
    capability::{
        BatchLane, BatchLayout, BatchLogDensityGradient, BatchShape, GaussianResidual,
        HessianVector, SplitLogDensityGradient,
    },
    target::LogDensityGradient,
};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    advanced::{
        AdvancedHmcOptions, ForceGradientHmc, ForceGradientMode, GaussianSplitHmc, SplitForceHmc,
    },
    batch::BatchedLeapfrog,
    hamiltonian::SignedStep,
};
use rand::{SeedableRng, rngs::SmallRng};

#[derive(Debug, thiserror::Error)]
#[error("injected M10 backend failure")]
struct Failure;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Split,
    NonFiniteSplit,
    Residual,
    NonFiniteResidual,
    Hvp,
    NonFiniteHvp,
    Batch,
    Panic,
}

struct Target {
    fault: Cell<Fault>,
}

impl LogDensityGradient for Target {
    type Error = Failure;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        if self.fault.get() == Fault::Panic {
            panic!("injected target panic");
        }
        gradient[0] = -q[0];
        gradient[1] = -q[1];
        Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
    }
}

impl SplitLogDensityGradient for Target {
    fn cheap_score(&self, q: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        output[0] = -0.5 * q[0];
        if self.fault.get() == Fault::Split {
            return Err(Failure);
        }
        output[1] = if self.fault.get() == Fault::NonFiniteSplit {
            f64::NAN
        } else {
            -0.5 * q[1]
        };
        Ok(())
    }
    fn expensive_score(&self, q: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        output[0] = -0.5 * q[0];
        output[1] = -0.5 * q[1];
        Ok(())
    }
}

impl GaussianResidual for Target {
    fn gaussian_center(&self) -> &[f64] {
        &[0.0; 2]
    }
    fn gaussian_precision(&self) -> &[f64] {
        &[1.0; 2]
    }
    fn residual_score(&self, _: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        output[0] = 0.0;
        if self.fault.get() == Fault::Residual {
            return Err(Failure);
        }
        output[1] = if self.fault.get() == Fault::NonFiniteResidual {
            f64::NAN
        } else {
            0.0
        };
        Ok(())
    }
}

impl HessianVector for Target {
    fn potential_hvp(
        &self,
        _: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        output[0] = vector[0];
        if self.fault.get() == Fault::Hvp {
            return Err(Failure);
        }
        output[1] = if self.fault.get() == Fault::NonFiniteHvp {
            f64::NAN
        } else {
            vector[1]
        };
        Ok(())
    }
}

impl BatchLogDensityGradient for Target {
    fn batch_logp_grad(
        &self,
        shape: BatchShape,
        positions: &[f64],
        gradients: &mut [f64],
        results: &mut [BatchLane<Self::Error>],
    ) {
        for (chain, result) in results.iter_mut().enumerate() {
            let a = shape.index(chain, 0).unwrap();
            let b = shape.index(chain, 1).unwrap();
            gradients[a] = -positions[a];
            gradients[b] = -positions[b];
            *result = if self.fault.get() == Fault::Batch && chain == 1 {
                BatchLane::Complete(Err(alea_core::target::EvaluationError::Model(Failure)))
            } else {
                BatchLane::Complete(Ok(
                    -0.5 * (positions[a] * positions[a] + positions[b] * positions[b])
                ))
            };
        }
    }
}

fn position() -> OwnedBuffer {
    OwnedBuffer::from_fn(2, |index| [0.3, -0.2][index])
}

#[test]
fn advanced_backend_errors_and_panics_never_publish_partial_state() {
    let target = Target {
        fault: Cell::new(Fault::None),
    };
    let options = AdvancedHmcOptions::new(0.1, 2).unwrap();
    let mut split =
        SplitForceHmc::new(&target, position(), IdentityMetric::new(2), options, 2).unwrap();
    let mut gaussian = GaussianSplitHmc::new(&target, position(), options).unwrap();
    let mut force_gradient = ForceGradientHmc::new(
        &target,
        position(),
        IdentityMetric::new(2),
        options,
        ForceGradientMode::AnalyticHvp,
    )
    .unwrap();
    let mut batch = BatchedLeapfrog::new(
        &target,
        2,
        BatchLayout::ChainMajor,
        OwnedBuffer::from_fn(4, |index| [0.3, -0.2, 0.1, 0.4][index]),
        OwnedBuffer::new(4),
    )
    .unwrap();
    let split_before = split.point().position().to_vec();
    let gaussian_before = gaussian.point().position().to_vec();
    let force_gradient_before = force_gradient.point().position().to_vec();
    let batch_before = batch.positions().to_vec();
    let mut rng = SmallRng::seed_from_u64(10_020);

    target.fault.set(Fault::Split);
    assert!(split.step(&mut rng).is_err());
    assert_eq!(split.point().position(), split_before);
    target.fault.set(Fault::Residual);
    assert!(gaussian.step(&mut rng).is_err());
    assert_eq!(gaussian.point().position(), gaussian_before);
    target.fault.set(Fault::Hvp);
    assert!(force_gradient.step(&mut rng).is_err());
    assert_eq!(force_gradient.point().position(), force_gradient_before);
    target.fault.set(Fault::Batch);
    assert!(batch.advance(SignedStep::new(0.1).unwrap(), 2).is_err());
    assert_eq!(batch.positions(), batch_before);

    target.fault.set(Fault::NonFiniteSplit);
    let transition = split.step(&mut rng).unwrap();
    assert!(transition.divergence.is_some());
    assert_eq!(split.point().position(), split_before);
    target.fault.set(Fault::NonFiniteResidual);
    let transition = gaussian.step(&mut rng).unwrap();
    assert!(transition.divergence.is_some());
    assert_eq!(gaussian.point().position(), gaussian_before);
    target.fault.set(Fault::NonFiniteHvp);
    let transition = force_gradient.step(&mut rng).unwrap();
    assert!(transition.divergence.is_some());
    assert_eq!(force_gradient.point().position(), force_gradient_before);

    target.fault.set(Fault::Panic);
    assert!(catch_unwind(AssertUnwindSafe(|| gaussian.step(&mut rng))).is_err());
    assert_eq!(gaussian.point().position(), gaussian_before);

    target.fault.set(Fault::None);
    assert!(split.step(&mut rng).is_ok());
    assert!(gaussian.step(&mut rng).is_ok());
    assert!(force_gradient.step(&mut rng).is_ok());
    assert!(batch.advance(SignedStep::new(0.1).unwrap(), 2).is_ok());
}
