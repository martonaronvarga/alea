//! Measures semantic registration against explicit opaque lowering of the SAME
//! native primitive/kernel. No reimplemented probability or sampler arithmetic.
use alea_core::model::{
    ConstrainedModel,
    assembly::{ModelBuilder, NativeModel},
};
use alea_distributions::modeling::regression::{NormalRegression, RegressionData};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

fn build(data: RegressionData<f64>, semantic: bool) -> NativeModel {
    let mut b = ModelBuilder::new();
    let beta = [
        b.real("intercept").unwrap().expr(),
        b.real("slope").unwrap().expr(),
    ];
    let sigma = b.positive("sigma").unwrap();
    let primitive = NormalRegression::new(data).unwrap();
    let inputs = [&beta[..], &[sigma.expr()]].concat();
    let f = if semantic {
        b.primitive(&inputs, primitive).unwrap()
    } else {
        let primitive = std::sync::Arc::new(primitive);
        let binding = std::sync::Arc::clone(&primitive);
        b.custom_with_scratch(
            &inputs,
            move || binding.workspace(),
            move |x, g, scratch| primitive.evaluate(x, g, scratch),
        )
        .unwrap()
    };
    b.likelihood("data", f).unwrap();
    b.compile().unwrap()
}
fn benchmark(c: &mut Criterion) {
    let variants = if std::env::var_os("ALEA_SEMANTICS_REVERSE").is_some() {
        [("semantic", true), ("opaque", false)]
    } else {
        [("opaque", false), ("semantic", true)]
    };
    for rows in [1_000, 10_000] {
        let data = RegressionData::new(
            2,
            (0..rows)
                .flat_map(|i| [1., (i % 17) as f64 / 17.])
                .collect(),
            (0..rows).map(|i| (i % 11) as f64 / 11.).collect(),
        )
        .unwrap();
        let mut construction = c.benchmark_group(format!("model_construction_shared_data/{rows}"));
        for (name, semantic) in variants {
            construction.bench_function(name, |b| {
                b.iter(|| build(black_box(data.clone()), semantic))
            });
        }
        construction.finish();
        let mut evaluation = c.benchmark_group(format!("model_evaluation/{rows}"));
        evaluation.throughput(Throughput::Elements(1)); // One fused value/gradient call.
        for (name, semantic) in variants {
            let model = build(data.clone(), semantic);
            let mut g = [0.; 3];
            evaluation.bench_with_input(BenchmarkId::new(name, rows), &model, |b, model| {
                b.iter(|| {
                    black_box(
                        model
                            .logp_grad(black_box(&[0.2, -0.4, 1.1]), black_box(&mut g))
                            .unwrap(),
                    )
                })
            });
        }
        evaluation.finish();
    }
}
criterion_group! {
    name = benches;
    config = benchmark_support::criterion().sample_size(30).warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = benchmark
}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
