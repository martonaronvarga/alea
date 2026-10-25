//! Fixed physical time, frozen-step posterior refinement experiment.
//! Execute on the benchmark host; raw trials include failures and repeated states.
use alea_core::target::LogDensityGradient;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::mams::{
    ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget, Mams, MamsOptions,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    cell::Cell,
    convert::Infallible,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::PathBuf,
};

struct Banana {
    bend: f64,
    calls: Cell<usize>,
}
impl LogDensityGradient for Banana {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        self.calls.set(self.calls.get() + 1);
        let r = q[1] - self.bend * (q[0] * q[0] - 1.);
        g[0] = -q[0] + 2. * self.bend * q[0] * r;
        g[1] = -r;
        Ok(-0.5 * (q[0] * q[0] + r * r))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("missing output directory")?,
    );
    std::fs::create_dir(&directory)?;
    let create = |name| {
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.join(name))
    };
    let mut raw = BufWriter::new(create("draws.csv")?);
    let mut trials = BufWriter::new(create("trials.csv")?);
    writeln!(
        raw,
        "target,method,step,seed,draw,x,y,correction_error,accepted"
    )?;
    writeln!(
        trials,
        "target,method,step,seed,warmup,planned_draws,completed_draws,gradient_calls,failure"
    )?;
    for (name, bend) in [("gaussian", 0.), ("banana", 0.5)] {
        for method in ["mclmc", "mams"] {
            for h in [1., 0.5, 0.25, 0.125] {
                let warmup = (1024. / (8. * h)) as usize;
                let draws = (8192. / (8. * h)) as usize;
                for seed in 19001..19009 {
                    let t = Banana {
                        bend,
                        calls: Cell::new(0),
                    };
                    let mut rng = SmallRng::seed_from_u64(seed);
                    let mut completed = 0;
                    let mut failure = String::new();
                    if method == "mclmc" {
                        let mut s = ApproximateMclmc::new(
                            ApproximateTarget::acknowledge_bias(),
                            &t,
                            OwnedBuffer::new(2),
                            ApproximateMclmcOptions::new(h, 8, Some(2_f64.sqrt()))?,
                            &mut rng,
                        )?;
                        for i in 0..warmup + draws {
                            match s.step(&mut rng) {
                                Ok(o) => {
                                    if i >= warmup {
                                        let q = s.point().position();
                                        writeln!(
                                            raw,
                                            "{name},{method},{h},{seed},{completed},{:.17e},{:.17e},{:.17e},1",
                                            q[0], q[1], o.correction_error
                                        )?;
                                        completed += 1;
                                    }
                                }
                                Err(e) => {
                                    failure = e.to_string().replace([',', '\n'], " ");
                                    break;
                                }
                            }
                        }
                    } else {
                        let mut s = Mams::new(&t, OwnedBuffer::new(2), MamsOptions::new(h, 8)?)?;
                        for i in 0..warmup + draws {
                            match s.step(&mut rng) {
                                Ok(o) => {
                                    if i >= warmup {
                                        let q = s.point().position();
                                        writeln!(
                                            raw,
                                            "{name},{method},{h},{seed},{completed},{:.17e},{:.17e},{:.17e},{}",
                                            q[0],
                                            q[1],
                                            o.correction_error.unwrap_or(f64::NAN),
                                            u8::from(o.accepted)
                                        )?;
                                        completed += 1;
                                    }
                                }
                                Err(e) => {
                                    failure = e.to_string().replace([',', '\n'], " ");
                                    break;
                                }
                            }
                        }
                    }
                    writeln!(
                        trials,
                        "{name},{method},{h},{seed},{warmup},{draws},{completed},{},{failure}",
                        t.calls.get()
                    )?;
                }
            }
        }
    }
    Ok(())
}
