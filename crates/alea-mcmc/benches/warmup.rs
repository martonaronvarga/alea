use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    HmcOptions,
    adapt::{HmcWarmup, MetricKind, OnlineCovariance, WarmupOptions},
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::SmallRng};
use std::{hint::black_box, time::Duration};

// The counter wrapper and extra seeds are used by the CSV example/tests, not timing.
#[allow(dead_code)]
#[path = "support/warmup.rs"]
mod workload;

fn matched_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("warmup_cost_v1");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for shape in workload::SHAPES {
        for dimension in workload::DIMENSIONS {
            let target = workload::Target::new(dimension, shape);
            for &method in workload::METHODS {
                group.bench_function(
                    format!("{}/{}/{dimension}", shape.name(), method.name()),
                    |b| b.iter(|| black_box(workload::run(black_box(&target), method, 7))),
                );
            }
        }
    }
    group.finish();
}

#[cfg(feature = "faer")]
fn fitting_cost(c: &mut Criterion) {
    use alea_core::target::LogDensityGradient;
    use alea_mcmc::adapt::FisherMetricAdapter;
    use rand::RngExt;

    let mut group = c.benchmark_group("fisher_fit_cost_v1");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    for dimension in [8, 64, 256] {
        for history in [16, 64] {
            let target = workload::Target::new(dimension, workload::Shape::Correlated);
            let mut rng = SmallRng::seed_from_u64(7);
            let mut adapter = FisherMetricAdapter::new(dimension, history, history).unwrap();
            let mut q = vec![0.0; dimension];
            let mut score = vec![0.0; dimension];
            for _ in 0..history {
                for x in &mut q {
                    *x = rng.random_range(-1.0..1.0);
                }
                target.logp_grad(&q, &mut score).unwrap();
                adapter.observe(&q, &score).unwrap();
            }
            let mut scales = vec![0.0; dimension];
            adapter
                .scales_into(&vec![1.0; dimension], 1e-5, &mut scales)
                .unwrap();
            // Observation generation/diagonal estimation excluded. Rank four
            // caps only output: the joint history subspace still gets factored.
            group.bench_function(format!("rank4/d{dimension}/history{history}"), |b| {
                b.iter(|| {
                    black_box(
                        adapter
                            .fit_low_rank(black_box(&scales), 1e-5, 2.0, 4)
                            .unwrap(),
                    )
                })
            });
        }
    }
    group.finish();
}

#[cfg(not(feature = "faer"))]
fn fitting_cost(_: &mut Criterion) {}

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
criterion_group!(benches, bench, matched_cost, fitting_cost);
criterion_main!(benches);
