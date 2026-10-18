use alea_core::target::LogDensityGradient;
use alea_math::metric::EuclideanMetric;
use alea_mcmc::hamiltonian::{Phase, SignedStep};

fn advance<T, M>(phase: Phase<'_, '_, T, M>)
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
{
    let _advanced = phase.step(SignedStep::new(0.1).unwrap());
    let _invalid_reuse = phase.point();
}

fn main() {}
