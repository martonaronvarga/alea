# Your first model

Suppose four measurements share an unknown mean. We model each measurement with
a normal distribution of standard deviation 1 and put a normal prior with mean 0
and standard deviation 2 on the unknown mean:

\[
 y_i \mid \mu \sim \operatorname{Normal}(\mu, 1), \qquad
 \mu \sim \operatorname{Normal}(0, 2).
\]

This example writes the log density and gradient directly. The
[modeling guide](modeling.md) builds models from distribution factors instead.
Both approaches use the same sampling interface.

## Run the example

Follow [Installation](installation.md), then place this complete program in
`src/main.rs`:

```rust
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::target::LogDensityGradient;
use alea::{
    ensemble::Execution,
    session::{Kernel, SessionBuilder, Warmup, storage::Discard},
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
    .kernel(Kernel::Nuts(NutsOptions::new(0.2, 8)?))
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
```

Run `cargo run --release`. This model has an analytic posterior: its mean is about
1.082 and its standard deviation is about 0.485. A finite Monte Carlo run will
vary around those values. Agreement with an analytic answer is useful here, but
you should also inspect the [chain diagnostics](diagnostics.md).

## What the program does

`LogDensityGradient` supplies the log posterior and its derivatives in a single
call. `q[0]` is the current mean. The method writes its derivative into
`gradient[0]`. The omitted normalizing constants do not depend on the unknown mean,
so they do not affect this posterior. If a scale were unknown, its `-log(scale)`
term would have to remain in the model.

`SessionBuilder` receives a model identity, the number of unconstrained
coordinates, and one initial point per chain. This example runs four chains from
different starting values. Give the identity a new value when the data or model
changes, particularly if you save checkpoints.

Warmup adjusts the step size and diagonal geometry using 500 discarded
transitions. The sampler then keeps 1,000 draws per chain with adaptation frozen.
A session does **no warmup unless you request it**.

`Discard` means there is no external output sink. The session still collects the
draws in memory because collection is enabled by default. For larger fits, use a
[streaming sink](sessions.md) and turn collection off.

## Inspect before summarizing

A run can return a diagnostic warning, a cancellation, or a partial chain rather
than a complete fit. Check `result.status`, chain failures, and the number of
completed draws before using a posterior summary. Rejected proposals remain in
the chain as repeated draws and must not be removed.

Next, learn to [declare parameters and likelihoods](modeling.md), or read about
[choosing a sampler and warmup](sampling.md).
