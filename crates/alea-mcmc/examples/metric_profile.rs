//! Isolated metric-action workload; use one backend feature at a time.
use alea_math::{
    buffer::OwnedBuffer,
    metric::{CholeskyFactor, DenseMetric, EuclideanMetric, LowRankDiagonalMetric},
};
use std::{error::Error, hint::black_box, time::Instant};

fn measure(metric: impl EuclideanMetric, repetitions: usize) -> Result<(), Box<dyn Error>> {
    let n = metric.dimension();
    let input = OwnedBuffer::from_fn(n, |i| (i as f64 + 1.0).sin());
    let mut output = OwnedBuffer::new(n);
    let start = Instant::now();
    for _ in 0..repetitions {
        metric.velocity(black_box(&input), black_box(&mut output))?;
        black_box(&output);
        metric.sample_momentum(black_box(&input), black_box(&mut output))?;
        black_box(&output);
    }
    println!("{:.9}", start.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let kind = args.next().ok_or("missing dense/lowrank")?;
    let n: usize = args.next().ok_or("missing dimension")?.parse()?;
    let repetitions: usize = args.next().ok_or("missing repetitions")?.parse()?;
    if !(4..=4096).contains(&n) || repetitions == 0 || args.next().is_some() {
        return Err("invalid workload dimensions or arguments".into());
    }
    match kind.as_str() {
        "dense" => {
            // M = L L^T with unit diagonal L and 0.5/n below the diagonal.
            let lower = OwnedBuffer::from_fn(n * n, |k| {
                let (i, j) = (k % n, k / n);
                if i == j {
                    1.0
                } else if i > j {
                    0.5 / n as f64
                } else {
                    0.0
                }
            });
            measure(
                DenseMetric::new(CholeskyFactor::new_lower(n, lower)?),
                repetitions,
            )
        }
        "lowrank" => {
            // Four orthonormal DCT columns, spread across all coordinates.
            let basis = OwnedBuffer::from_fn(n * 4, |k| {
                let (i, j) = (k % n, k / n);
                if j == 0 {
                    1.0 / (n as f64).sqrt()
                } else {
                    (2.0 / n as f64).sqrt()
                        * (std::f64::consts::PI * (i as f64 + 0.5) * j as f64 / n as f64).cos()
                }
            });
            measure(
                LowRankDiagonalMetric::new(
                    OwnedBuffer::from_fn(n, |i| 1.0 + (i % 3) as f64 * 0.1),
                    basis,
                    OwnedBuffer::from_fn(4, |i| [0.25, 0.5, 2.0, 4.0][i]),
                )?,
                repetitions,
            )
        }
        _ => Err("unknown metric".into()),
    }
}
