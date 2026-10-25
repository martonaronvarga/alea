//! Bounded CPU profiling workload; reports phase time, actual calls and draw bits.
//! Run saved before/after executables serially with identical arguments.
use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Nuts, NutsOptions,
    adapt::{NutsWarmup, WarmupOptions},
};
use alea_runtime::diagnostics::modern;
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible, hint::black_box, time::Instant};

struct Normal {
    dimension: usize,
    calls: Cell<usize>,
}
impl LogDensityGradient for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        let mut square = 0.0;
        for (g, &x) in gradient.iter_mut().zip(q) {
            *g = -x;
            square += x * x;
        }
        Ok(-0.5 * square)
    }
}

fn hash(values: impl IntoIterator<Item = f64>) -> u64 {
    values.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
        (h ^ x.to_bits()).wrapping_mul(0x0100_0000_01b3)
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("kernel");
    let dimension: usize = args.get(2).map_or(Ok(16), |s| s.parse())?;
    let draws: usize = args.get(3).map_or(Ok(8192), |s| s.parse())?;
    let repeats: usize = args.get(4).map_or(Ok(4), |s| s.parse())?;
    if !(1..=128).contains(&dimension)
        || !(4..=100_000).contains(&draws)
        || !(1..=64).contains(&repeats)
    {
        return Err("bounded dimension/draw/repeat workload required".into());
    }
    if mode == "diagnostics" {
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                (0..draws)
                    .map(|i| ((i * 17 + c * 71) as f64 * 0.071).sin() + c as f64 * 0.001)
                    .collect()
            })
            .collect();
        let slices: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        let start = Instant::now();
        let mut checksum = 0;
        for _ in 0..repeats {
            let s = modern::summarize(black_box(&slices))?;
            checksum ^= hash([s.rhat, s.bulk_ess, s.tail_ess, s.mean_ess, s.mcse_mean]);
        }
        println!(
            "{{\"mode\":\"diagnostics\",\"seconds\":{},\"checksum\":{checksum}}}",
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    if !matches!(mode, "kernel" | "warmup") {
        return Err("mode must be kernel, warmup or diagnostics".into());
    }
    for seed in 0..repeats {
        let target = Normal {
            dimension,
            calls: Cell::new(0),
        };
        let mut rng = SmallRng::seed_from_u64(21001 + u64::try_from(seed)?);
        let mut positions = Vec::with_capacity(draws * dimension);
        let mut statistics = Vec::with_capacity(draws * 3);
        let start = Instant::now();
        let (mut chain, warmup) = if mode == "warmup" {
            let (chain, _) = NutsWarmup::new(
                &target,
                OwnedBuffer::from_fn(dimension, |_| 0.1),
                NutsOptions::new(0.3, 8)?,
                WarmupOptions::new(512),
            )?
            .run(&mut rng)?;
            (chain, start.elapsed().as_secs_f64())
        } else {
            let identity = IdentityMetric::new(dimension);
            let mut chain = Nuts::new(
                &target,
                OwnedBuffer::from_fn(dimension, |_| 0.1),
                identity,
                NutsOptions::new(0.3, 8)?,
            )?;
            run(&mut chain, &mut rng, draws, &mut positions, &mut statistics)?;
            println!(
                "{{\"mode\":\"kernel\",\"dimension\":{dimension},\"seed\":{seed},\"seconds\":{},\"calls\":{},\"draw_hash\":{},\"stats_hash\":{}}}",
                start.elapsed().as_secs_f64(),
                target.calls.get(),
                hash(positions),
                hash(statistics)
            );
            continue;
        };
        let before = target.calls.get();
        let start = Instant::now();
        run(&mut chain, &mut rng, draws, &mut positions, &mut statistics)?;
        let retained = start.elapsed().as_secs_f64();
        println!(
            "{{\"mode\":\"warmup\",\"dimension\":{dimension},\"seed\":{seed},\"warmup_seconds\":{warmup},\"retained_seconds\":{retained},\"warmup_calls\":{before},\"retained_calls\":{},\"draw_hash\":{},\"stats_hash\":{}}}",
            target.calls.get() - before,
            hash(positions),
            hash(statistics)
        );
    }
    Ok(())
}

fn run<M: alea_math::metric::EuclideanMetric>(
    chain: &mut Nuts<'_, Normal, M>,
    rng: &mut SmallRng,
    draws: usize,
    positions: &mut Vec<f64>,
    statistics: &mut Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..draws {
        let t = chain.step(rng)?;
        positions.extend_from_slice(chain.point().position());
        statistics.extend([
            t.acceptance_probability,
            t.integration_steps as f64,
            t.selected_index as f64,
        ]);
    }
    Ok(())
}
