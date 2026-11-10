//! Source emission is cold work, not compilation or an Enzyme execution timing.
use alea_autodiff::{
    codegen::enzyme_source,
    gradient_check::{GradientCheckOptions, check_gradient},
};
use alea_core::model::assembly::ModelBuilder;
use alea_distributions::modeling::{NormalBatch, normal};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn derivatives(c: &mut Criterion) {
    for rows in [4, 128, 4096] {
        for (name, batch) in [("scalar", false), ("batch", true)] {
            let mut b = ModelBuilder::new();
            let mu = b.real("mu").unwrap().expr();
            let sigma = b.positive("sigma").unwrap().expr();
            if batch {
                let f = b
                    .primitive(
                        &[&[mu][..], &[sigma][..]].concat(),
                        NormalBatch::observed(vec![0.3; rows], 1, 1).unwrap(),
                    )
                    .unwrap();
                b.factor(f).unwrap();
            } else {
                for _ in 0..rows {
                    let f = normal(&mut b, 0.3, mu, sigma).unwrap();
                    b.factor(f).unwrap();
                }
            }
            let model = b.compile().unwrap();
            let mut group = c.benchmark_group(format!("derivatives/{name}/{rows}"));
            assert!(!enzyme_source(&model).unwrap().is_empty());
            group.bench_function("emit_rust_and_drop", |b| {
                b.iter(|| black_box(enzyme_source(black_box(&model)).unwrap()))
            });
            let target = model.into_target().unwrap();
            assert!(
                check_gradient(&target, &[0.2, 0.1], GradientCheckOptions::default())
                    .unwrap()
                    .passed()
            );
            group.bench_function("central_difference_diagnostic", |b| {
                b.iter(|| {
                    black_box(
                        check_gradient(
                            black_box(&target),
                            black_box(&[0.2, 0.1]),
                            GradientCheckOptions::default(),
                        )
                        .unwrap(),
                    )
                })
            });
            group.finish();
        }
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=derivatives}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
