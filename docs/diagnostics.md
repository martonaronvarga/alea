# Diagnosing a fit

A program finishing its requested number of iterations does not establish that
its samples describe the posterior well. Check the chains, sampler warnings, and
the Monte Carlo uncertainty of the quantities you intend to report.

## Start with the run outcome

A session returns a status and a result for each chain. Check both before pooling
draws. The main outcomes are:

| Status | What to do |
|---|---|
| `Success` | Review the diagnostics and scientific plausibility of the fit |
| `DiagnosticWarning` | Inspect the diagnostic report and sampler events |
| `Cancelled` | Decide whether to use the partial output or resume it |
| `NumericalExhaustion` | Inspect initial values, scales, gradients, and warmup |
| `BackendFailure` | Inspect the original error and any undelivered output |

A single chain cannot provide between-chain R-hat. A short or constant series may
have undefined diagnostics. Alea reports these cases rather than replacing them
with reassuring numbers.

## Summarize one observable

`summarize` accepts one scalar series per chain. In this example the series are
independent normal draws, so they illustrate the API rather than a fitted model:

```rust
use alea::distribution::Sample;
use alea::distributions::scalar::Normal;
use alea::diagnostics::modern::summarize;
use rand::{rngs::SmallRng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let normal = Normal::new(0.0, 1.0)?;
let mut rng = SmallRng::seed_from_u64(42);
let mut chains = vec![Vec::new(); 4];
for chain in &mut chains {
    for _ in 0..1_000 {
        chain.push(normal.sample(&mut rng)?);
    }
}
let series: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
let summary = summarize(&series)?;
println!("mean = {:.3}", summary.mean);
println!("R-hat = {:.3}", summary.rhat);
println!("bulk ESS = {:.0}; tail ESS = {:.0}", summary.bulk_ess, summary.tail_ess);
println!("Monte Carlo SE of the mean = {:.3}", summary.mcse_mean);
# Ok(())
# }
```

For a fitted model, extract the same parameter or derived quantity from every
chain. Keep the chain boundaries. Each series must have the same length, and all
values must be finite. The function does not silently truncate ragged chains or
drop NaNs.

## Read the numbers together

**R-hat** compares variation within and between chains. Alea reports the larger
of rank-normalized split R-hat and folded R-hat. Values near 1 are desirable;
1.01 is the session's default warning threshold. A good R-hat alone does not
rule out a shared failure to explore another mode.

**Bulk ESS** estimates information about the central part of the distribution.
**Tail ESS** uses the lower and upper tails and is particularly relevant to
interval endpoints and rare-event summaries. ESS is not the number of stored
rows: a long correlated chain can contain much less information than its length
suggests.

**Monte Carlo standard error** describes the sampling uncertainty of an estimated
mean. For a scientifically useful fit, this uncertainty should be small relative
to the precision you need. It does not include model uncertainty or data errors.

Plot traces and compare distributions across chains. Also inspect important
predictions and derived quantities. Parameters can look well behaved while a
nonlinear function of them still has poor effective sample size.

## Sampler warnings

Divergences signal difficulty following the target's geometry. Review the model,
parameterization, and gradients before simply increasing the number of draws.
Tree-depth saturation means NUTS reached its work limit. Increasing the limit can
help a fit explore, but it also increases the possible cost of every transition.

Acceptance rate is a tuning diagnostic, not a posterior accuracy score. A very
small step can give high acceptance while moving too slowly to be useful.

## Session diagnostic windows

Sessions use a bounded window of retained draws for their final diagnostic report.
The default window is 512 draws per chain, with warning thresholds R-hat 1.01 and
bulk and tail ESS 100. These are screening defaults, not a stopping theorem.

For full-run summaries, collect the draws or read your durable stream and call
`summarize` on the series you want to analyze. After checkpoint recovery, session
diagnostics describe the new segment only. Combine earlier and later segments by
chain and draw index when computing a diagnostic for the entire run.
