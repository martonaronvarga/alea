//! Estimate a normal mean with four NUTS chains.
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::target::LogDensityGradient;
use alea::{
    ensemble::Execution,
    session::{SamplerConfig, SessionBuilder, Warmup, storage::Discard},
};
use std::convert::Infallible;

struct MeanModel {
    observations: Vec<f64>,
}

impl alea::target::TargetSpace for MeanModel {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for MeanModel {
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let mean = q[0];
        let mut logp = -0.5 * (mean / 2.0).powi(2);
        gradient[0] = -mean / 4.0;
        for &observation in &self.observations {
            let residual = observation - mean;
            logp -= 0.5 * residual * residual;
            gradient[0] += residual;
        }
        Ok(logp)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = SessionBuilder::new(
        "normal-mean;data=1.2,0.9,1.4,1.1;model=1",
        1,
        vec![vec![-1.0], vec![0.0], vec![1.0], vec![2.0]],
    )
    .seed([17; 32])
    .draws(1_000)
    .sampler(SamplerConfig::Nuts(NutsOptions::new(0.2, 8)?))
    .warmup(Warmup::Covariance {
        iterations: 500,
        metric: MetricKind::Diagonal,
        target: 0.8.try_into()?,
        search: None,
    })
    .build()?;

    let result = session.run(
        |_| {
            Ok(MeanModel {
                observations: vec![1.2, 0.9, 1.4, 1.1],
            })
        },
        Execution::Sequential,
        &Discard,
    )?;

    println!("Run status: {:?}", result.status);
    for chain in &result.chains {
        if let Some(failure) = &chain.failure {
            eprintln!("Chain {:?}: {failure:?}", chain.chain);
        }
    }
    if result
        .chains
        .iter()
        .any(|chain| chain.failure.is_some() || chain.completed != 1_000)
    {
        return Err(
            "Sampling did not complete; inspect the chain outcomes before summarizing".into(),
        );
    }
    let draws: Vec<f64> = result
        .chains
        .iter()
        .flat_map(|chain| chain.draws.iter().map(|draw| draw.position[0]))
        .collect();
    if !draws.is_empty() {
        println!(
            "Posterior mean: {:.3}",
            draws.iter().sum::<f64>() / draws.len() as f64
        );
    }
    Ok(())
}
