use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use kernels::density::FusedLogDensity;
use kernels::dist::traits::Target;
use kernels::dist::wiener::{
    Boundary, Quadrature, SeriesBranch, Wiener4, Wiener4Params, Wiener5, Wiener5Params, Wiener7,
    Wiener7Params, WienerObservation, WienerObservations, WienerOptions,
};
use pprof::criterion::{Output, PProfProfiler};

const EPS4: f64 = 1e-12;
const EPS5: f64 = 1e-12;
const EPS7: f64 = 1e-4;

fn observations(n: usize) -> Vec<WienerObservation> {
    (0..n)
        .map(|i| {
            let u = ((i * 37) % 1000) as f64 / 1000.0;
            WienerObservation {
                rt: 0.35 + 1.5 * u,
                boundary: if i % 3 == 0 {
                    Boundary::Lower
                } else {
                    Boundary::Upper
                },
            }
        })
        .collect()
}

fn wiener4_params() -> Wiener4Params {
    Wiener4Params::with_params(1.6, 0.2, 0.45, 0.7).unwrap()
}

fn wiener5_params() -> Wiener5Params {
    Wiener5Params::with_params(1.6, 0.2, 0.45, 0.7, 0.1).unwrap()
}

fn wiener7_params() -> Wiener7Params {
    Wiener7Params::builder()
        .alpha(1.6)
        .tau(0.2)
        .beta(0.45)
        .delta(0.7)
        .s_delta(0.1)
        .s_beta(0.05)
        .s_tau(0.1)
        .build()
        .unwrap()
}

fn bench_single_observation(c: &mut Criterion) {
    let obs = WienerObservation {
        rt: 1.2,
        boundary: Boundary::Upper,
    };

    let p4 = wiener4_params();
    c.bench_function("wiener4/single/fused", |b| {
        b.iter(|| {
            let eval = Wiener4.fused(black_box(&obs), black_box(&p4), EPS4);
            black_box((eval.log_prob, eval.grad.to_array()));
        });
    });

    let p5 = wiener5_params();
    c.bench_function("wiener5/single/fused", |b| {
        b.iter(|| {
            let eval = Wiener5.fused(black_box(&obs), black_box(&p5), EPS5);
            black_box((eval.log_prob, eval.grad.to_array()));
        });
    });

    let p7 = wiener7_params();
    c.bench_function("wiener7/single/fused/adaptive", |b| {
        b.iter(|| {
            let eval = Wiener7.fused(black_box(&obs), black_box(&p7), EPS7);
            black_box((eval.log_prob, eval.grad.to_array()));
        });
    });

    let fixed = WienerOptions::new(EPS7)
        .with_inner_precision(EPS5)
        .with_quadrature(Quadrature::FixedGaussLegendre { order: 25 });
    c.bench_function("wiener7/single/fused/fixed_gl25", |b| {
        b.iter(|| {
            let eval = Wiener7
                .try_fused(black_box(&obs), black_box(&p7), black_box(fixed))
                .unwrap();
            black_box((eval.log_prob, eval.grad.to_array()));
        });
    });
}

fn bench_wiener4_batch(c: &mut Criterion) {
    let params = wiener4_params();
    let mut group = c.benchmark_group("wiener4/batch_fused");
    for &n in &[128, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        let data = observations(n);
        let target_aos = Target::new(Wiener4, data.clone());
        let target_soa = Target::new(Wiener4, WienerObservations::from(data));

        group.bench_function(BenchmarkId::new("aos", n), |b| {
            let mut grad = [0.0; 4];
            b.iter(|| {
                let lp = target_aos.log_prob_and_grad(black_box(&params), black_box(&mut grad));
                black_box((lp, grad));
            });
        });
        group.bench_function(BenchmarkId::new("soa", n), |b| {
            let mut grad = [0.0; 4];
            b.iter(|| {
                let lp = target_soa.log_prob_and_grad(black_box(&params), black_box(&mut grad));
                black_box((lp, grad));
            });
        });
    }
    group.finish();
}

fn bench_wiener5_batch(c: &mut Criterion) {
    let params = wiener5_params();
    let mut group = c.benchmark_group("wiener5/batch_fused");
    for &n in &[128, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        let data = observations(n);
        let target_aos = Target::new(Wiener5, data.clone());
        let target_soa = Target::new(Wiener5, WienerObservations::from(data));

        group.bench_function(BenchmarkId::new("aos", n), |b| {
            let mut grad = [0.0; 5];
            b.iter(|| {
                let lp = target_aos.log_prob_and_grad(black_box(&params), black_box(&mut grad));
                black_box((lp, grad));
            });
        });
        group.bench_function(BenchmarkId::new("soa", n), |b| {
            let mut grad = [0.0; 5];
            b.iter(|| {
                let lp = target_soa.log_prob_and_grad(black_box(&params), black_box(&mut grad));
                black_box((lp, grad));
            });
        });
    }
    group.finish();
}

fn bench_wiener7_batch(c: &mut Criterion) {
    let params = wiener7_params();
    let mut group = c.benchmark_group("wiener7/batch_fused");
    for &n in &[32, 128] {
        group.throughput(Throughput::Elements(n as u64));
        let data = observations(n);
        let target_aos = Target::new(Wiener7, data.clone());
        let target_soa = Target::new(Wiener7, WienerObservations::from(data));

        group.bench_function(BenchmarkId::new("aos", n), |b| {
            let mut grad = [0.0; 7];
            b.iter(|| {
                let lp = target_aos.log_prob_and_grad(black_box(&params), black_box(&mut grad));
                black_box((lp, grad));
            });
        });
        group.bench_function(BenchmarkId::new("soa", n), |b| {
            let mut grad = [0.0; 7];
            b.iter(|| {
                let lp = target_soa.log_prob_and_grad(black_box(&params), black_box(&mut grad));
                black_box((lp, grad));
            });
        });
    }
    group.finish();
}

fn branch_counts_wiener4(data: &[WienerObservation], params: &Wiener4Params) -> [usize; 6] {
    let mut counts = [0; 6];
    for obs in data {
        if let Some(branch) = Wiener4.branch_counts(obs, params, EPS4) {
            match branch.branch {
                SeriesBranch::SmallTime => counts[0] += 1,
                SeriesBranch::LargeTime => counts[1] += 1,
            }
            counts[2] += branch.k_small_used;
            counts[3] += branch.k_large_used;
            counts[4] = counts[4].max(branch.k_small_used);
            counts[5] = counts[5].max(branch.k_large_used);
        }
    }
    counts
}

fn branch_counts_wiener5(data: &[WienerObservation], params: &Wiener5Params) -> [usize; 6] {
    let mut counts = [0; 6];
    for obs in data {
        if let Some(branch) = Wiener5.branch_counts(obs, params, EPS5) {
            match branch.branch {
                SeriesBranch::SmallTime => counts[0] += 1,
                SeriesBranch::LargeTime => counts[1] += 1,
            }
            counts[2] += branch.k_small_used;
            counts[3] += branch.k_large_used;
            counts[4] = counts[4].max(branch.k_small_used);
            counts[5] = counts[5].max(branch.k_large_used);
        }
    }
    counts
}

fn bench_branch_counts(c: &mut Criterion) {
    let data = observations(10_000);
    let p4 = wiener4_params();
    let p5 = wiener5_params();
    let mut group = c.benchmark_group("wiener/branch_counts");
    group.throughput(Throughput::Elements(data.len() as u64));
    group.bench_function("wiener4", |b| {
        b.iter(|| black_box(branch_counts_wiener4(black_box(&data), black_box(&p4))));
    });
    group.bench_function("wiener5", |b| {
        b.iter(|| black_box(branch_counts_wiener5(black_box(&data), black_box(&p5))));
    });
    group.finish();
}

criterion_group! {
    name = wiener_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_single_observation, bench_wiener4_batch, bench_wiener5_batch, bench_wiener7_batch, bench_branch_counts,
}

criterion_main!(wiener_benches);
