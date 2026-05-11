use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kernels::dist::wiener::{
    Boundary, Wiener4, Wiener5, Wiener5Params, Wiener7, Wiener7Params, WienerObservation,
};
use pprof::criterion::{Output, PProfProfiler};

fn common_params() -> (Wiener7Params, WienerObservation) {
    let params = Wiener7Params::with_params(
        1.5, 0.3, 0.55, 0.4, // alpha, tau, beta, delta
        0.05, 0.15, 0.1, // sw, st0, sv
    )
    .unwrap();
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    (params, obs)
}

// Print one evaluation for visual comparison with Stan output
fn print_example() {
    let (params, obs) = common_params();
    let eval = Wiener7.fused(&obs, &params, 1e-4);
    let grad_arr = eval.grad.to_array();

    let p5 = Wiener5Params::with_params(1.5, 0.3, 0.55, 0.4, 0.1).unwrap();
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let eval5 = Wiener5.log_prob(&obs, &p5, 1e-12);
    println!("Rust Wiener5 log_prob: {:.10}", eval5.log_prob);

    // The one chosen by trunc_counts is:
    eprintln!("=== Rust Wiener7 example ===");
    eprintln!("log_prob: {:.10}", eval.log_prob);
    eprintln!("grad: {:.10?}", grad_arr);
    eprintln!("grad (structured): {:.10?}", eval.grad);

    // Also print on stdout for easy capture
    println!("Rust log_prob: {:.10}", eval.log_prob);
    println!("Rust grad: {:+.10?}", grad_arr);
}

// Benchmarks

fn bench_fused(c: &mut Criterion) {
    print_example();
    let (params, obs) = common_params();
    c.bench_function("wiener7_fused_baseline", |b| {
        b.iter(|| {
            let eval = Wiener7.fused(black_box(&obs), black_box(&params), 1e-4);
            black_box((eval.log_prob, eval.grad.to_array()));
        })
    });
}

fn bench_log_prob_only(c: &mut Criterion) {
    let (params, obs) = common_params();
    c.bench_function("wiener7_log_prob", |b| {
        b.iter(|| {
            let eval = Wiener7.log_prob(black_box(&obs), black_box(&params), 1e-4);
            black_box(eval.log_prob);
        })
    });
}

fn bench_fused_zero_variability(c: &mut Criterion) {
    let params = Wiener7Params::with_params(2.0, 0.25, 0.5, -0.3, 0.0, 0.0, 0.2).unwrap();
    let obs = WienerObservation {
        rt: 1.2,
        boundary: Boundary::Lower,
    };
    c.bench_function("wiener7_fused_zero_var", |b| {
        b.iter(|| {
            let eval = Wiener7.fused(black_box(&obs), black_box(&params), 1e-4);
            black_box((eval.log_prob, eval.grad.to_array()));
        })
    });
}

fn bench_fused_high_variability(c: &mut Criterion) {
    let params = Wiener7Params::with_params(3.0, 0.4, 0.25, 0.8, 0.35, 0.3, 0.4).unwrap();
    let obs = WienerObservation {
        rt: 1.5,
        boundary: Boundary::Upper,
    };
    c.bench_function("wiener7_fused_high_var", |b| {
        b.iter(|| {
            let eval = Wiener7.fused(black_box(&obs), black_box(&params), 1e-4);
            black_box((eval.log_prob, eval.grad.to_array()));
        })
    });
}

fn bench_batch_10(c: &mut Criterion) {
    let params = Wiener7Params::with_params(1.5, 0.3, 0.55, 0.4, 0.05, 0.15, 0.1).unwrap();
    let obs_batch: Vec<_> = (0..10)
        .map(|i| WienerObservation {
            rt: 0.7 + i as f64 * 0.05,
            boundary: if i % 2 == 0 {
                Boundary::Upper
            } else {
                Boundary::Lower
            },
        })
        .collect();
    c.bench_function("wiener7_batch_10", |b| {
        b.iter(|| {
            let mut total_lp = 0.0;
            let mut total_grad = [0.0; 7];
            for obs in obs_batch.iter() {
                let eval = Wiener7.fused(black_box(obs), black_box(&params), 1e-4);
                total_lp += eval.log_prob;
                let g = eval.grad.to_array();
                for i in 0..7 {
                    total_grad[i] += g[i];
                }
            }
            black_box((total_lp, total_grad));
        })
    });
}

// Criterion requires these macros when harness = false
criterion_group! {
    name = wiener7_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_fused, bench_log_prob_only, bench_fused_zero_variability,
               bench_fused_high_variability, bench_batch_10,
}

// Main entry point: print example before running benchmarks
criterion_main!(wiener7_benches);
