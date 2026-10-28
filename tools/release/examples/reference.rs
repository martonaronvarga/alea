//! A fresh project's complete analytic-target workflow, with no native backend.
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::target::LogDensityGradient;
use alea::{
    ensemble::Execution,
    session::SessionBuilder,
    session::{Kernel, Status, Warmup, storage::Discard},
};
use std::convert::Infallible;

struct Normal;
impl alea::target::TargetSpace for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for Normal {
    fn logp_grad(&self, q: &[f64], score: &mut [f64]) -> Result<f64, Self::Error> {
        score[0] = -q[0];
        score[1] = -q[1];
        Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = SessionBuilder::new("analytic-normal-v1", 2, vec![vec![0.0; 2]; 4])
        .seed([17; 32])
        .draws(256)
        .kernel(Kernel::Nuts(NutsOptions::new(0.2, 6)?))
        .warmup(Warmup::Covariance {
            iterations: 150,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into()?,
            search: None,
        })
        .build()?;
    let result = session.run(|_| Ok(Normal), Execution::Sequential, &Discard)?;
    assert!(matches!(
        result.status,
        Status::Success | Status::DiagnosticWarning
    ));
    assert_eq!(result.chains.len(), 4);
    assert!(result.chains.iter().all(|chain| chain.completed == 256));
    println!("{:?}: four chains, 256 retained draws each", result.status);
    Ok(())
}
