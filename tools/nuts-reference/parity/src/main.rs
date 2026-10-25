//! Fixed-geometry attribution; production samplers remain separate dependencies.
use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Nuts, NutsOptions};
use nuts_rs::{
    Chain as _, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    StepSizeAdaptMethod,
};
use rand::{SeedableRng, rngs::SmallRng};
use serde_json::json;
use std::{cell::Cell, collections::HashMap, error::Error, hint::black_box, rc::Rc, time::Instant};

#[derive(Debug)]
struct Fault;
impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("nonfinite normal")
    }
}
impl Error for Fault {}
impl LogpError for Fault {
    fn is_recoverable(&self) -> bool {
        true
    }
}
struct Normal {
    d: usize,
    calls: Rc<Cell<usize>>,
}
impl LogDensityGradient for Normal {
    type Error = Fault;
    fn dimension(&self) -> usize {
        self.d
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
        self.calls.set(self.calls.get() + 1);
        let mut sum = 0.;
        for (&q, g) in q.iter().zip(g) {
            *g = -q;
            sum += q * q;
        }
        Ok(-0.5 * sum)
    }
}
impl HasDims for Normal {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("q".into(), self.d as u64)])
    }
}
impl CpuLogpFunc for Normal {
    type LogpError = Fault;
    type ExpandedVector = Vec<f64>;
    type FlowParameters = ();
    fn dim(&self) -> usize {
        self.d
    }
    fn logp(&mut self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
        self.logp_grad(q, g)
    }
    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _: &mut R,
        q: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(q.to_vec())
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let engine = args.first().ok_or("engine required")?;
    let d: usize = args.get(1).ok_or("dimension required")?.parse()?;
    let n: usize = args.get(2).map_or(Ok(20000), |x| x.parse())?;
    if ![2, 16, 64].contains(&d) || n < 1024 {
        return Err("unsupported bounded workload".into());
    }
    let calls = Rc::new(Cell::new(0));
    let target = Normal {
        d,
        calls: Rc::clone(&calls),
    };
    let mut sum = 0.;
    let mut square = 0.;
    let mut steps = 0usize;
    let seconds;
    if engine == "alea" || engine == "alea-position" {
        let mut chain = Nuts::new(
            &target,
            OwnedBuffer::from_fn(d, |_| 1.),
            IdentityMetric::new(d),
            NutsOptions::new(0.3, 8)?.with_position_turning(engine == "alea-position"),
        )?;
        let mut rng = SmallRng::seed_from_u64(43001);
        for _ in 0..33 {
            let _ = chain.step(&mut rng)?;
        }
        calls.set(0);
        let start = Instant::now();
        for _ in 0..n {
            let info = black_box(chain.step(&mut rng)?);
            let q = black_box(chain.point().position().to_vec());
            steps += info.integration_steps;
            sum += q[0];
            square += q[0] * q[0];
        }
        seconds = start.elapsed().as_secs_f64();
    } else if engine == "nuts-rs" {
        let mut options = DiagNutsSettings {
            num_tune: 1,
            num_draws: n as u64,
            maxdepth: 8,
            ..Default::default()
        };
        options.adapt_options.step_size_settings.jitter = None;
        options
            .adapt_options
            .step_size_settings
            .adapt_options
            .method = StepSizeAdaptMethod::Fixed(0.3);
        let mut chain =
            options.new_chain(0, CpuMath::new(target), &mut SmallRng::seed_from_u64(43001));
        chain.set_position(&vec![1.; d])?;
        for _ in 0..33 {
            let _ = chain.draw()?;
        }
        calls.set(0);
        let start = Instant::now();
        for _ in 0..n {
            let (q, _, _, info) = black_box(chain.expanded_draw()?);
            let q = black_box(q.to_vec());
            steps += info.num_steps as usize;
            sum += q[0];
            square += q[0] * q[0];
        }
        seconds = start.elapsed().as_secs_f64();
    } else {
        return Err("unknown engine".into());
    }
    println!(
        "{}",
        json!({"engine":engine,"dimension":d,"draws":n,"seconds":seconds,"gradient_calls":calls.get(),"steps":steps,"mean":sum/n as f64,"second_moment":square/n as f64})
    );
    Ok(())
}
