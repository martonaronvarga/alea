use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

use alea_core::{capability::PositionMetric, target::LogDensityGradient};
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{Rmhmc, RmhmcError, RmhmcOptions};
use rand::{SeedableRng, rngs::SmallRng};

#[derive(Debug, thiserror::Error)]
#[error("injected M11 backend failure")]
struct Failure;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Target,
    NonFiniteTarget,
    Metric,
    NonFiniteMetric,
    PanicTarget,
    PanicMetric,
}

struct Target<'a> {
    fault: &'a Cell<Fault>,
}

impl LogDensityGradient for Target<'_> {
    type Error = Failure;

    fn dimension(&self) -> usize {
        2
    }

    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        match self.fault.get() {
            Fault::Target => return Err(Failure),
            Fault::PanicTarget => panic!("injected M11 target panic"),
            _ => {}
        }
        gradient[0] = -q[0];
        gradient[1] = if self.fault.get() == Fault::NonFiniteTarget {
            f64::NAN
        } else {
            -q[1]
        };
        Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
    }
}

struct Metric<'a> {
    fault: &'a Cell<Fault>,
}

impl PositionMetric for Metric<'_> {
    type Error = Failure;

    fn dimension(&self) -> usize {
        2
    }

    fn metric_and_derivatives(
        &self,
        _: &[f64],
        metric: &mut [f64],
        derivatives: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), Self::Error> {
        match self.fault.get() {
            Fault::Metric => return Err(Failure),
            Fault::PanicMetric => panic!("injected M11 metric panic"),
            _ => {}
        }
        metric.fill(0.0);
        derivatives.fill(0.0);
        metric[0] = 1.0;
        metric[3] = if self.fault.get() == Fault::NonFiniteMetric {
            f64::NAN
        } else {
            1.0
        };
        Ok(())
    }
}

#[test]
fn backend_errors_nonfinite_outputs_and_panics_never_publish() {
    let fault = Cell::new(Fault::None);
    let target = Target { fault: &fault };
    let metric = Metric { fault: &fault };
    let options = RmhmcOptions::new(0.1, 2, 4, 1.0e-12).unwrap();
    let mut chain = Rmhmc::new(
        &target,
        &metric,
        OwnedBuffer::from_fn(2, |index| [0.3, -0.2][index]),
        options,
    )
    .unwrap();
    let before = chain.point().position().to_vec();
    let mut rng = SmallRng::seed_from_u64(11_300);

    fault.set(Fault::Target);
    assert!(matches!(chain.step(&mut rng), Err(RmhmcError::Target(_))));
    assert_eq!(chain.point().position(), before);

    fault.set(Fault::Metric);
    assert!(matches!(chain.step(&mut rng), Err(RmhmcError::Metric(_))));
    assert_eq!(chain.point().position(), before);

    fault.set(Fault::NonFiniteTarget);
    assert!(chain.step(&mut rng).unwrap().termination.is_some());
    assert_eq!(chain.point().position(), before);

    fault.set(Fault::NonFiniteMetric);
    assert!(matches!(chain.step(&mut rng), Err(RmhmcError::Metric(_))));
    assert_eq!(chain.point().position(), before);

    fault.set(Fault::PanicTarget);
    assert!(catch_unwind(AssertUnwindSafe(|| chain.step(&mut rng))).is_err());
    assert_eq!(chain.point().position(), before);

    fault.set(Fault::PanicMetric);
    assert!(catch_unwind(AssertUnwindSafe(|| chain.step(&mut rng))).is_err());
    assert_eq!(chain.point().position(), before);

    fault.set(Fault::None);
    assert!(chain.step(&mut rng).is_ok());
}
