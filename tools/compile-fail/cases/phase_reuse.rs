use alea::math::metric::EuclideanMetric;
use alea::sampling::hamiltonian::{Phase, SignedStep};
use alea::target::LogDensityGradient;

fn advance<T, M>(phase: Phase<'_, '_, T, M>)
where
    T: LogDensityGradient + ?Sized,
    M: EuclideanMetric,
{
    let _advanced = phase.step(SignedStep::new(0.1).unwrap());
    let _invalid_reuse = phase.point();
}

fn main() {}
