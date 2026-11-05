//! Real checked C integration; reused Rust outputs do not remove C scratch allocation.
use alea_ffi::{Bounds, Options, hcubature, hcubature_into};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn integral(x: &[f64], out: &mut [f64]) -> std::ffi::c_int {
    out[0] = x.iter().map(|x| x * x).sum();
    0
}
fn cubature(c: &mut Criterion) {
    for dimension in [1, 3] {
        let lower = vec![0.; dimension];
        let upper = vec![1.; dimension];
        let bounds = Bounds::try_new(&lower, &upper).unwrap();
        let options = Options::default();
        let (mut value, mut error) = hcubature(1, bounds, options, integral).unwrap();
        assert!((value[0] - dimension as f64 / 3.).abs() < 1e-8);
        let mut group = c.benchmark_group(format!("cubature/{dimension}"));
        group.bench_function("allocate_results_integrate_drop", |b| {
            b.iter(|| black_box(hcubature(1, black_box(bounds), options, integral).unwrap()))
        });
        group.bench_function("reused_results_integrate", |b| {
            b.iter(|| {
                hcubature_into(
                    1,
                    black_box(bounds),
                    options,
                    black_box(&mut value),
                    black_box(&mut error),
                    integral,
                )
                .unwrap();
                black_box((&value, &error));
            })
        });
        group.finish();
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=cubature}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
