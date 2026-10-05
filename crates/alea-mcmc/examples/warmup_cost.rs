//! Actual evaluation counts for the fixed Criterion warmup workload (CSV v1).
//! Run with `--features faer` to include the low-rank policy. No timing or ESS claim.
#[path = "../benches/support/warmup.rs"]
mod workload;

use std::{
    cell::Cell,
    io::{self, Write},
};
use workload::*;

fn main() -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(
        out,
        "schema,shape,dimension,method,seed,iterations,steps,initialization,target_calls,search_probes,integration_attempts,metric_updates,accepted,divergences,final_rank,step_size,final_logp"
    )?;
    for shape in SHAPES {
        for dimension in DIMENSIONS {
            for &method in METHODS {
                for seed in SEEDS {
                    let target = Counted {
                        target: Target::new(dimension, shape),
                        calls: Cell::new(0),
                    };
                    let result = run(&target, method, seed);
                    let r = result.report;
                    writeln!(
                        out,
                        "1,{},{},{},{seed},{ITERATIONS},{STEPS},identity,{},{},{},{},{},{},{},{},{}",
                        shape.name(),
                        dimension,
                        method.name(),
                        target.calls.get(),
                        r.search_probes,
                        r.integration_attempts,
                        r.metric_updates,
                        r.accepted,
                        r.divergences,
                        result.rank,
                        r.step_size.value(),
                        result.logp
                    )?;
                }
            }
        }
    }
    out.flush()
}
