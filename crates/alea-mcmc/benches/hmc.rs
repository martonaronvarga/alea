//! Cheap-gradient steady-state baseline: no construction or allocation in timing.
use alea_distributions::Gaussian;
use alea_distributions::wiener::WienerPrimitive;
use alea_distributions::wiener::{Boundary, WienerObservation};
use alea_math::buffer::OwnedBuffer;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use alea_core::model::{AnalyticModel, PrimitiveModel, TransformedTarget};
use alea_math::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
};

use alea_core::target::LogDensityGradient;
use alea_core::transform::{ParameterLayout, Transform};
use alea_mcmc::{Hmc, HmcOptions};
use rand::{SeedableRng, rngs::SmallRng};
use std::{hint::black_box, time::Duration};

fn workload<T: LogDensityGradient, M: EuclideanMetric>(
    c: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    target: &T,
    metric: M,
) {
    let mut chain = Hmc::new(
        target,
        OwnedBuffer::new(target.dimension()),
        metric,
        HmcOptions::new(0.02, 8).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(912);
    c.bench_function(name, |b| {
        b.iter(|| {
            let info = chain.step(&mut rng).unwrap();
            assert_eq!(info.integration_steps, 8);
            assert!(info.divergence.is_none());
            let _ = black_box(info);
        })
    });
}

fn workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_workloads");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    let gaussian = Gaussian::new(64);
    workload(
        &mut group,
        "gaussian64_identity",
        &gaussian,
        IdentityMetric::new(64),
    );
    workload(
        &mut group,
        "gaussian64_diagonal",
        &gaussian,
        DiagonalMetric::new(OwnedBuffer::from_fn(64, |i| 1.0 + (i % 3) as f64)).unwrap(),
    );
    let factor = OwnedBuffer::from_fn(64 * 64, |i| {
        let (row, col) = (i % 64, i / 64);
        if row == col {
            1.5
        } else if row == col + 1 {
            0.2
        } else {
            0.0
        }
    });
    workload(
        &mut group,
        "gaussian64_dense",
        &gaussian,
        DenseMetric::new(CholeskyFactor::new_lower(64, factor).unwrap()),
    );

    // Immutable data are constructed outside timing; the model computes 1024
    // logistic terms and all 16 derivatives in a single allocation-free pass.
    let data = (0..1024 * 16)
        .map(|i| ((i * 17 % 101) as f64 - 50.0) / 50.0)
        .collect::<Vec<_>>();
    let logistic = AnalyticModel::new(16, move |q: &[f64], g: &mut [f64]| {
        let mut lp = 0.0;
        for (&q, g) in q.iter().zip(g.iter_mut()) {
            *g = -q;
            lp -= 0.5 * q * q;
        }
        for (i, row) in data.chunks_exact(16).enumerate() {
            let eta: f64 = row.iter().zip(q).map(|(x, q)| x * q).sum();
            let t = (-eta.abs()).exp();
            let p = if eta >= 0.0 {
                1.0 / (1.0 + t)
            } else {
                t / (1.0 + t)
            };
            let y = (i % 2) as f64;
            lp += y * eta - eta.max(0.0) - t.ln_1p();
            for (g, x) in g.iter_mut().zip(row) {
                *g += (y - p) * x;
            }
        }
        Ok::<_, std::convert::Infallible>(lp)
    });
    let logistic = TransformedTarget::new(
        logistic,
        ParameterLayout::new([Transform::identity(16).unwrap()]).unwrap(),
    )
    .unwrap();
    workload(
        &mut group,
        "logistic1024x16",
        &logistic,
        IdentityMetric::new(16),
    );

    let wiener = PrimitiveModel(
        WienerPrimitive::new(
            WienerObservation {
                rt: 0.8,
                boundary: Boundary::Upper,
            },
            1e-12,
        )
        .unwrap(),
    );
    // Bound otherwise improper single-observation tails so this benchmark is a
    // fixed valid target throughout Criterion's unbounded number of iterations.
    let layout = ParameterLayout::new([
        Transform::interval(1, 0.5, 2.0).unwrap(),
        Transform::interval(1, 0.0, 0.4).unwrap(),
        Transform::interval(1, 0.1, 0.9).unwrap(),
        Transform::interval(1, -3.0, 3.0).unwrap(),
    ])
    .unwrap();
    let wiener = TransformedTarget::new(wiener, layout).unwrap();
    workload(
        &mut group,
        "wiener4_transformed",
        &wiener,
        IdentityMetric::new(4),
    );
    group.finish();
}

fn hmc(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_protocol");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for dim in [8, 64, 1024] {
        let target = Gaussian::new(dim);
        let mut chain = Hmc::new(
            &target,
            OwnedBuffer::new(dim),
            IdentityMetric::new(dim),
            HmcOptions::new(0.1, 8).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(912);
        group.bench_with_input(BenchmarkId::new("hmc", dim), &dim, |b, _| {
            b.iter(|| {
                let _ = black_box(chain.step(&mut rng).unwrap());
                black_box((
                    chain.point().position(),
                    chain.point().gradient(),
                    chain.point().log_density(),
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, hmc, workloads);
criterion_main!(benches);
