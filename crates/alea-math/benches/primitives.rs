//! Atomic native math/metric/RNG operations; data and geometry prepared outside loops.
use alea_math::{
    buffer::OwnedBuffer,
    metric::{DiagonalMetric, EuclideanMetric},
    numeric::{CompensatedSum, first_nonfinite, scaled_norm, softplus_and_sigmoid},
    random::NormalGenerator,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::SmallRng};
use std::hint::black_box;

fn primitives(c: &mut Criterion) {
    for n in [4, 64, 4096] {
        let x: Vec<_> = (0..n).map(|i| (i % 31) as f64 / 16. - 1.).collect();
        let metric =
            DiagonalMetric::new(OwnedBuffer::from_fn(n, |i| 1. + i as f64 / n as f64)).unwrap();
        let mut out = vec![0.; n];
        let mut group = c.benchmark_group(format!("math_primitives/{n}"));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function("finite_scan", |b| {
            b.iter(|| black_box(first_nonfinite(black_box(&x))))
        });
        group.bench_function("scaled_norm", |b| {
            b.iter(|| black_box(scaled_norm(black_box(&x))))
        });
        group.bench_function("compensated_sum", |b| {
            b.iter(|| {
                let mut sum = CompensatedSum::default();
                for &v in black_box(&x) {
                    sum.add(v);
                }
                black_box(sum.total())
            })
        });
        group.bench_function("softplus_sigmoid_map", |b| {
            b.iter(|| {
                for (&x, y) in black_box(&x).iter().zip(black_box(&mut out)) {
                    let (a, b) = softplus_and_sigmoid(x);
                    *y = a + b;
                }
                black_box(&out);
            })
        });
        group.bench_function("metric_velocity", |b| {
            b.iter(|| {
                metric.velocity(black_box(&x), black_box(&mut out)).unwrap();
                black_box(&out);
            })
        });
        group.bench_function("metric_kinetic", |b| {
            b.iter(|| {
                black_box(
                    metric
                        .kinetic_energy(black_box(&x), black_box(&mut out))
                        .unwrap(),
                )
            })
        });
        let mut rng = SmallRng::seed_from_u64(912);
        let mut normal = NormalGenerator::default();
        group.bench_function("cached_normal_fill", |b| {
            b.iter(|| {
                for v in black_box(&mut out) {
                    *v = normal.sample(&mut rng);
                }
                black_box(&out);
            })
        });
        group.bench_function("exact_dot_sign", |b| {
            b.iter(|| black_box(alea_math::exact::dot_sign(black_box(&x), black_box(&x)).unwrap()))
        });
        let affine = alea_math::affine::LowRankAffineMap::new(
            OwnedBuffer::new(n),
            OwnedBuffer::from_fn(n, |_| 1.3),
            OwnedBuffer::new(0),
            OwnedBuffer::new(0),
        )
        .unwrap();
        group.bench_function("affine_forward_rank0", |b| {
            b.iter(|| {
                affine.forward(black_box(&x), black_box(&mut out)).unwrap();
                black_box(&out);
            })
        });
        group.bench_function("affine_pullback_rank0", |b| {
            b.iter(|| {
                affine.pullback(black_box(&x), black_box(&mut out)).unwrap();
                black_box(&out);
            })
        });
        let spline = alea_math::interpolation::LinearSpline::new(
            (0..n).map(|i| i as f64).collect(),
            x.clone(),
        )
        .unwrap();
        group.throughput(Throughput::Elements(1)); // One query, not n visited knots.
        group.bench_function("interpolation_query", |b| {
            b.iter(|| black_box(spline.evaluate(black_box(n as f64 * 0.37)).unwrap()))
        });
        group.finish();
    }
    for n in [4, 32, 128] {
        let matrix: Vec<_> = (0..n * n)
            .map(|i| if i / n == i % n { 1.3 } else { 0. })
            .collect();
        c.bench_function(&format!("math_primitives/cholesky/{n}"), |b| {
            b.iter_batched(
                || matrix.clone(),
                |mut a| {
                    alea_math::matrix::cholesky_in_place::<alea_math::matrix::RowMajor>(
                        n,
                        black_box(&mut a),
                    )
                    .unwrap();
                    black_box(a)
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=primitives}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
