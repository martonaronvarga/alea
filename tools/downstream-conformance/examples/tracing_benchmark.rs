//! Fixed-work tracing overhead probe. Timing and numerical fingerprints are separate.
use alea::{
    ensemble::Execution,
    math::{buffer::OwnedBuffer, metric::IdentityMetric},
    optimize::{LbfgsOptions, ObjectiveDescription, ObjectiveKind, maximize},
    prelude::*,
    session::{SamplerConfig, SessionBuilder, storage::Discard},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{convert::Infallible, hint::black_box, ops::ControlFlow, time::Instant};

struct Gaussian(usize);
impl TargetSpace for Gaussian {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.0
    }
}
impl LogDensityGradient for Gaussian {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        let mut value = 0.;
        g.fill(0.);
        for i in 0..q.len() {
            let residual = q[i] - if i > 0 { 0.5 * q[i - 1] } else { 0. };
            value -= 0.5 * residual * residual;
            g[i] -= residual;
            if i > 0 {
                g[i - 1] += 0.5 * residual;
            }
        }
        Ok(value)
    }
}
impl alea::density::LogDensity for Gaussian {
    fn logp(&self, q: &[f64]) -> Result<f64, Infallible> {
        let mut value = 0.;
        for i in 0..q.len() {
            let d = q[i] - if i > 0 { 0.5 * q[i - 1] } else { 0. };
            value -= 0.5 * d * d;
        }
        Ok(value)
    }
}
fn mix(hash: &mut u64, n: u64) {
    *hash = (*hash ^ n).wrapping_mul(0x100000001b3);
}
fn workload(case: &str) -> Result<(u64, String), Box<dyn std::error::Error>> {
    if ![
        "stream",
        "hmc",
        "nuts",
        "rwmh",
        "parallel50",
        "bernoulli",
        "poisson",
        "cone",
        "svd",
        "wiener4_density",
        "wiener4_fused",
        "wiener5_density",
        "wiener5_fused",
        "wiener7_density",
        "wiener7_fused",
        "wiener7_rare_density",
        "wiener7_rare_fused",
    ]
    .contains(&case)
    {
        return Err("unknown workload".into());
    }
    let mut hash = 1;
    let mut outcome = "Completed".to_owned();
    if case == "stream" {
        let target = Gaussian(2);
        let mut rng = SmallRng::seed_from_u64(17);
        let mut chain = alea::sampling::Hmc::new(
            &target,
            OwnedBuffer::new(2),
            IdentityMetric::new(2),
            alea::sampling::HmcOptions::new(0.2, 8)?,
        )?;
        let n = alea::run_chain(&mut chain, 100_000, &mut rng, |q, _| {
            for x in q {
                mix(&mut hash, x.to_bits());
            }
            ControlFlow::Continue(())
        })?;
        mix(&mut hash, n as u64);
        use rand::Rng;
        mix(&mut hash, rng.next_u64());
    } else if ["hmc", "nuts", "rwmh", "parallel50"].contains(&case) {
        let d = if case == "parallel50" { 50 } else { 2 };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let kernel = match case {
            "hmc" => SamplerConfig::Hmc(alea::sampling::HmcOptions::new(0.2, 8)?),
            "rwmh" => SamplerConfig::Rwmh(alea::sampling::RwmhOptions::new(0.4)?),
            _ => SamplerConfig::Nuts(alea::sampling::NutsOptions::new(0.2, 7)?),
        };
        let s = SessionBuilder::new("tracing-benchmark", d, vec![vec![0.; d]; 4])
            .sampler(kernel)
            .draws(if d == 50 { 2000 } else { 10_000 })
            .collect(true)
            .seed([17; 32])
            .build()?;
        let execution = if d == 50 {
            Execution::Parallel(&pool)
        } else {
            Execution::Sequential
        };
        let outcome = if case == "rwmh" {
            s.run_density(|_| Ok(Gaussian(d)), execution, &Discard)?
        } else {
            s.run(|_| Ok(Gaussian(d)), execution, &Discard)?
        };
        for chain in outcome.chains {
            if chain.failure.is_some() {
                return Err("benchmark chain failed".into());
            }
            mix(&mut hash, chain.completed as u64);
            for draw in chain.draws {
                for x in draw.position {
                    mix(&mut hash, x.to_bits());
                }
            }
        }
    } else if case.starts_with("wiener") {
        use alea::wiener::{
            Boundary, Wiener4, Wiener4Params, Wiener5, Wiener5Params, Wiener7, Wiener7Params,
            WienerObservation, WienerOptions,
        };
        let p4 = Wiener4Params::with_params(1.6, 0.2, 0.45, 0.7)?;
        let p5 = Wiener5Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1)?;
        let rare = case.contains("rare");
        let p7 = if rare {
            Wiener7Params::with_params(0.1, 0., 0.1, -10., 0.0001, 0.001, 0.1)?
        } else {
            Wiener7Params::with_params(1.6, 0.2, 0.45, 0.7, 0.05, 0.1, 0.1)?
        };
        let density = case.ends_with("density");
        let count = if case.starts_with("wiener7") {
            200
        } else {
            300_000
        };
        for i in 0..count {
            let obs = WienerObservation {
                rt: if rare {
                    0.000001
                } else {
                    0.4 + (i % 101) as f64 / 100.
                },
                boundary: Boundary::Upper,
            };
            let obs = black_box(&obs);
            let options = WienerOptions::new(if case.starts_with("wiener7") {
                1e-4
            } else {
                1e-12
            });
            let value = if case.starts_with("wiener4") {
                if density {
                    Wiener4.try_log_prob(obs, black_box(&p4), options)?.log_prob
                } else {
                    let e = Wiener4.try_fused(obs, black_box(&p4), options)?;
                    for g in e.grad_array() {
                        mix(&mut hash, g.to_bits());
                    }
                    e.log_prob
                }
            } else if case.starts_with("wiener5") {
                if density {
                    Wiener5.try_log_prob(obs, black_box(&p5), options)?.log_prob
                } else {
                    let e = Wiener5.try_fused(obs, black_box(&p5), options)?;
                    for g in e.grad_array() {
                        mix(&mut hash, g.to_bits());
                    }
                    e.log_prob
                }
            } else {
                if density {
                    Wiener7.try_log_prob(obs, black_box(&p7), options)?.log_prob
                } else {
                    let e = Wiener7.try_fused(obs, black_box(&p7), options)?;
                    for g in e.grad_array() {
                        mix(&mut hash, g.to_bits());
                    }
                    e.log_prob
                }
            };
            mix(&mut hash, value.to_bits());
        }
    } else if case == "cone" {
        for _ in 0..20_000 {
            use alea::math::exact::{ConeSearch, ConeSearchOptions, strict_cone_direction};
            let result = strict_cone_direction(
                2,
                black_box(&[1., -0.5, -0.5, 1.]),
                &[],
                ConeSearchOptions::default(),
            )?;
            match result {
                ConeSearch::Direction(d) => {
                    for n in d.coefficients() {
                        for b in n.to_signed_bytes_le() {
                            mix(&mut hash, u64::from(b));
                        }
                    }
                }
                _ => return Err("cone unexpectedly unresolved".into()),
            }
        }
    } else if case == "svd" {
        #[cfg(feature = "faer")]
        for _ in 0..2000 {
            let fit = alea::math::least_squares::solve(
                4,
                2,
                black_box(&[1., -1., 1., 0., 1., 1., 1., 2.]),
                &[0., 1., 2., 2.5],
                1e-12,
            )?;
            for x in fit.coefficients {
                mix(&mut hash, x.to_bits());
            }
            mix(&mut hash, fit.residual_norm.to_bits());
        }
        #[cfg(not(feature = "faer"))]
        return Err("svd requires the faer feature".into());
    } else if case == "bernoulli" || case == "poisson" {
        use alea::distributions::modeling::regression::{
            RegressionData, bernoulli_logit_regression, poisson_log_regression,
        };
        let mut b = alea::model::assembly::ModelBuilder::new();
        let beta = [b.real("intercept")?.expr(), b.real("slope")?.expr()];
        let design: Vec<_> = (0..100_000)
            .flat_map(|i| [1., (i % 101) as f64 / 50. - 1.])
            .collect();
        let f = if case == "bernoulli" {
            bernoulli_logit_regression(
                &mut b,
                &beta,
                RegressionData::new(2, design, (0..100_000).map(|i| i % 3 == 0).collect())?,
            )?
        } else {
            poisson_log_regression(
                &mut b,
                &beta,
                RegressionData::new(2, design, (0..100_000).map(|i| (i % 7) as u64).collect())?,
            )?
        };
        b.likelihood("observations", f)?;
        let model = b.compile()?.select_likelihood()?.into_objective()?;
        let fit = maximize(
            &model,
            &[0., 0.],
            ObjectiveDescription {
                kind: ObjectiveKind::Likelihood,
                measure: "coefficients".into(),
                model: case.into(),
                data: "synthetic100k".into(),
                source: "tracing-benchmark".into(),
            },
            LbfgsOptions::new(100, 500)?,
        )?;
        outcome = format!("{:?}", fit.termination);
        for byte in outcome.bytes() {
            mix(&mut hash, u64::from(byte));
        }
        mix(&mut hash, fit.evaluations as u64);
        let c = fit.candidate.ok_or("missing candidate")?;
        for x in c.position.into_iter().chain(c.gradient).chain([c.value]) {
            mix(&mut hash, x.to_bits());
        }
    } else {
        return Err("unknown workload".into());
    }
    Ok((black_box(hash), outcome))
}

#[cfg(feature = "tracing")]
mod collection {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    use tracing::{Event, Subscriber};
    use tracing_subscriber::{Layer, Registry, prelude::*};
    #[derive(Clone, Default)]
    pub struct Count(pub Arc<AtomicU64>);
    impl<S: Subscriber> Layer<S> for Count {
        fn on_event(&self, _: &Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn dispatch(mode: &str, count: Count) -> tracing::Dispatch {
        let level = match mode {
            "filtered" => tracing::level_filters::LevelFilter::OFF,
            "debug" => tracing::level_filters::LevelFilter::DEBUG,
            "trace" => tracing::level_filters::LevelFilter::TRACE,
            _ => tracing::level_filters::LevelFilter::INFO,
        };
        tracing::Dispatch::new(Registry::default().with(count.with_filter(level)))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: tracing_benchmark workload off|none|filtered|info|debug|trace repetitions"
                .into(),
        );
    }
    let repetitions: usize = args[3].parse()?;
    if repetitions == 0
        || !["off", "none", "filtered", "info", "debug", "trace"].contains(&args[2].as_str())
    {
        return Err("invalid benchmark configuration".into());
    }
    #[cfg(not(feature = "tracing"))]
    if args[2] != "off" {
        return Err("this binary was compiled without tracing".into());
    }
    #[cfg(feature = "tracing")]
    let count = collection::Count::default();
    #[cfg(feature = "tracing")]
    let dispatch = if args[2] == "none" {
        None
    } else {
        Some(collection::dispatch(&args[2], count.clone()))
    };
    for repetition in 0..repetitions {
        let start = Instant::now();
        #[cfg(feature = "tracing")]
        let result = if let Some(d) = &dispatch {
            tracing::dispatcher::with_default(d, || workload(&args[1]))
        } else {
            workload(&args[1])
        };
        #[cfg(not(feature = "tracing"))]
        let result = workload(&args[1]);
        let (fingerprint, outcome) = result?;
        let nanos = start.elapsed().as_nanos();
        #[cfg(feature = "tracing")]
        let events = count.0.swap(0, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(feature = "tracing"))]
        let events = 0;
        println!(
            "{{\"repetition\":{repetition},\"nanoseconds\":{nanos},\"fingerprint\":\"{fingerprint:016x}\",\"events\":{events},\"outcome\":\"{outcome}\"}}"
        );
    }
    Ok(())
}
