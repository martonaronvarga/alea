//! Same mathematical models through native kernels and actual generated AD.
//! Requires pinned nightly/autodiff flags and release fat LTO, as do the tests.
#[allow(dead_code)]
#[path = "../models.rs"]
mod models;
use alea::{
    capability::PotentialHessianDerivative,
    model::assembly::CompiledModel,
    target::{LogDensityGradient, TargetSpace},
};
use alea_enzyme_conformance as generated;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn compare<T: PotentialHessianDerivative>(
    c: &mut Criterion,
    name: &str,
    native: CompiledModel,
    compiled: T,
) {
    let native = native.into_target().unwrap();
    let n = native.dimension();
    assert_eq!(n, compiled.dimension());
    let q: Vec<_> = (0..n).map(|i| 0.1 + i as f64 * 0.01).collect();
    let mut a = vec![0.; n];
    let mut b = a.clone();
    let va = native.logp_grad(&q, &mut a).unwrap();
    let vb = compiled.logp_grad(&q, &mut b).unwrap();
    for (x, y) in std::iter::once((&va, &vb)).chain(a.iter().zip(&b)) {
        assert!((x - y).abs() <= 2e-10 * (1. + x.abs().max(y.abs())));
    }
    let mut group = c.benchmark_group(format!("native_compiled/{name}"));
    group.throughput(Throughput::Elements(1)); // Complete value AND gradient.
    group.bench_function("native_reused", |b| {
        b.iter(|| black_box(native.logp_grad(black_box(&q), black_box(&mut a)).unwrap()))
    });
    group.bench_function("compiled_reused", |b| {
        b.iter(|| {
            black_box(
                compiled
                    .logp_grad(black_box(&q), black_box(&mut a))
                    .unwrap(),
            )
        })
    });
    let direction = vec![0.2; n];
    compiled.potential_hvp(&q, &direction, &mut a).unwrap();
    group.bench_function("compiled_potential_hvp", |b| {
        b.iter(|| {
            compiled
                .potential_hvp(black_box(&q), black_box(&direction), black_box(&mut a))
                .unwrap();
            black_box(&a);
        })
    });
    let mut h = vec![0.; n * n];
    let mut d = vec![0.; n * n * n];
    compiled
        .potential_hessian_and_derivatives(&q, &mut h, &mut d)
        .unwrap();
    group.bench_function("compiled_potential_hessian_third", |b| {
        b.iter(|| {
            compiled
                .potential_hessian_and_derivatives(
                    black_box(&q),
                    black_box(&mut h),
                    black_box(&mut d),
                )
                .unwrap();
            black_box((&h, &d));
        })
    });
    group.finish();
}
fn native_compiled(c: &mut Criterion) {
    // Do not label native-backed source pullbacks as pure compiler AD.
    macro_rules! route {
        ($($module:ident),* $(,)?) => {$(
            eprintln!("ALEA_SOURCE_ROUTE {} explicit_first_order={} checkpointed={}",
                stringify!($module), generated::$module::SOURCE_EXPLICIT_FIRST_ORDER,
                generated::$module::SOURCE_CHECKPOINTED);
        )*};
    }
    route!(
        polynomial,
        native_normal_large,
        native_logit_large,
        native_poisson_large,
        normal_vector,
        normal_vector_observed,
        tensor_checkpoint_long,
        tensor_retained_long
    );
    assert!(!generated::polynomial::SOURCE_EXPLICIT_FIRST_ORDER);
    compare(
        c,
        "polynomial_compiler_ad",
        models::polynomial().unwrap(),
        generated::polynomial::target(),
    );
    compare(
        c,
        "normal129",
        models::native_batch_rows(0, 129).unwrap(),
        generated::native_normal_large::target(),
    );
    compare(
        c,
        "logit129",
        models::native_batch_rows(1, 129).unwrap(),
        generated::native_logit_large::target(),
    );
    compare(
        c,
        "poisson129",
        models::native_batch_rows(2, 129).unwrap(),
        generated::native_poisson_large::target(),
    );
    compare(
        c,
        "normal_vector",
        models::normal_vector(false).unwrap(),
        generated::normal_vector::target(),
    );
    compare(
        c,
        "normal_vector_observed",
        models::normal_vector(true).unwrap(),
        generated::normal_vector_observed::target(),
    );
    let mut all = models::all().unwrap();
    let name = "tensor_checkpoint_long";
    let i = all.iter().position(|(n, _)| *n == name).unwrap();
    compare(
        c,
        name,
        all.swap_remove(i).1,
        generated::tensor_checkpoint_long::target(),
    );
    let name = "tensor_retained_long";
    let i = all.iter().position(|(n, _)| *n == name).unwrap();
    compare(
        c,
        name,
        all.swap_remove(i).1,
        generated::tensor_retained_long::target(),
    );
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=native_compiled}
#[path = "../../../crates/benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
