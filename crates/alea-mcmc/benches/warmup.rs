use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    HmcOptions,
    adapt::{HmcWarmup, MetricKind, OnlineCovariance, WarmupOptions},
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::SmallRng};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("warmup");
    for kind in [MetricKind::Diagonal, MetricKind::Dense] {
        for dimension in [8, 64] {
            let values: Vec<_> = (0..64)
                .map(|row| OwnedBuffer::from_fn(dimension, |i| ((row * 7 + i * 13) % 31) as f64))
                .collect();
            let mut covariance = OnlineCovariance::new(dimension, kind).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("updates64/{kind:?}"), dimension),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        covariance.reset();
                        for point in &values {
                            covariance.update(black_box(point)).unwrap();
                        }
                        black_box(covariance.mean());
                    })
                },
            );
            covariance.reset();
            for point in &values {
                covariance.update(point).unwrap();
            }
            group.bench_with_input(
                BenchmarkId::new(format!("window_metric/{kind:?}"), dimension),
                &dimension,
                |b, _| b.iter(|| black_box(covariance.metric().unwrap())),
            );
        }
        let target = Gaussian::new(8);
        group.bench_function(format!("hmc200/{kind:?}/8"), |b| {
            b.iter(|| {
                let controller = HmcWarmup::new(
                    &target,
                    OwnedBuffer::new(8),
                    HmcOptions::new(0.1, 5).unwrap(),
                    WarmupOptions::new(200).with_metric(kind),
                )
                .unwrap();
                black_box(controller.run(&mut SmallRng::seed_from_u64(7)).unwrap());
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("fisher");
    let data: Vec<_> = (0..64)
        .map(|row| OwnedBuffer::from_fn(64, |i| ((row * 7 + i * 13) % 31) as f64 / 31.0))
        .collect();
    let scores: Vec<_> = data
        .iter()
        .map(|q| OwnedBuffer::from_fn(64, |i| -q[i] / (1.0 + i as f64)))
        .collect();
    let mut estimator = alea_mcmc::adapt::FisherMetricAdapter::new(64, 80, 0).unwrap();
    group.bench_function("paired_updates64/d64", |b| {
        b.iter(|| {
            estimator.reset(80).unwrap();
            for (q, s) in data.iter().zip(&scores) {
                estimator.observe(black_box(q), black_box(s)).unwrap();
            }
            black_box(estimator.count());
        })
    });
    for low_rank in [false, true] {
        if low_rank && !cfg!(feature = "faer") {
            continue;
        }
        let target = Gaussian::new(8);
        group.bench_function(
            if low_rank {
                "warmup200/low_rank/d8"
            } else {
                "warmup200/diagonal/d8"
            },
            |b| {
                b.iter(|| {
                    let options = alea_mcmc::adapt::FisherOptions::new(200);
                    #[cfg(feature = "faer")]
                    let options = if low_rank {
                        options.with_max_rank(4)
                    } else {
                        options
                    };
                    let warmup = alea_mcmc::adapt::FisherHmcWarmup::new(
                        &target,
                        OwnedBuffer::new(8),
                        HmcOptions::new(0.1, 5).unwrap(),
                        options,
                    )
                    .unwrap();
                    black_box(warmup.run(&mut SmallRng::seed_from_u64(7)).unwrap());
                })
            },
        );
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
