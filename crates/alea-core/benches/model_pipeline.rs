//! Separate registration, compilation, target conversion, and reusable execution.
use alea_core::{
    model::{ConstrainedModel, assembly::ModelBuilder},
    target::LogDensityGradient,
};
use alea_distributions::modeling::{NormalBatch, normal};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn builder(rows: usize, batch: bool) -> ModelBuilder {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap().expr();
    let sigma = b.positive("sigma").unwrap().expr();
    let y: Vec<_> = (0..rows).map(|i| (i % 17) as f64 / 8. - 1.).collect();
    if batch {
        let f = b
            .primitive(
                &[&[mu][..], &[sigma][..]].concat(),
                NormalBatch::observed(y, 1, 1).unwrap(),
            )
            .unwrap();
        b.likelihood("observations", f).unwrap();
    } else {
        for value in y {
            let f = normal(&mut b, value, mu, sigma).unwrap();
            b.factor(f).unwrap();
        }
    }
    b
}

fn pipeline(c: &mut Criterion) {
    // The actual centered funnel surface, not a handwritten density/gradient.
    // Keep scalar versus explicit native batch assembly as separate controls.
    for latents in [8, 32, 128] {
        for batch in [false, true] {
            let build = || {
                let mut b = ModelBuilder::new();
                let v = b.real("v").unwrap().expr();
                let prior = normal(&mut b, v, 0., 3.).unwrap();
                b.prior("v prior", prior).unwrap();
                let half = b.constant(0.5).unwrap();
                let log_scale = b.multiply(v, half).unwrap();
                let scale = b.exp(log_scale).unwrap();
                let zero = b.constant(0.).unwrap();
                let mut events = Vec::with_capacity(latents);
                for j in 0..latents {
                    let event = b.real(&format!("z{j}")).unwrap().expr();
                    events.push(event);
                    if !batch {
                        let factor = normal(&mut b, event, zero, scale).unwrap();
                        b.prior(&format!("conditional{j}"), factor).unwrap();
                    }
                }
                if batch {
                    let factor = b
                        .primitive(
                            &[&events[..], &[zero][..], &[scale][..]].concat(),
                            NormalBatch::new([events.len(), 1, 1]).unwrap(),
                        )
                        .unwrap();
                    b.prior("conditionals", factor).unwrap();
                }
                b.compile().unwrap()
            };
            let name = if batch { "batch" } else { "scalar" };
            let mut group = c.benchmark_group(format!("funnel_pipeline/{name}/{latents}"));
            group.bench_function("construction_and_drop", |b| b.iter(|| black_box(build())));
            let model = build();
            let q: Vec<_> = (0..=latents)
                .map(|j| 0.2 * ((j + 1) as f64).sin())
                .collect();
            let mut gradient = vec![0.; q.len()];
            assert!(model.logp_grad(&q, &mut gradient).unwrap().is_finite());
            group.bench_function("constrained_value_gradient_reused", |b| {
                b.iter(|| {
                    black_box(
                        model
                            .logp_grad(black_box(&q), black_box(&mut gradient))
                            .unwrap(),
                    )
                })
            });
            group.finish();
        }
    }
    for n in [4, 64, 4096] {
        use alea_core::density;
        use alea_core::target::{EvaluationWorkspace, PointState, evaluate};
        use alea_distributions::StandardNormalTarget;
        let target = StandardNormalTarget::new(n);
        let q = vec![0.2; n];
        let mut g = vec![0.; n];
        let mut group = c.benchmark_group(format!("target_boundary/{n}"));
        group.bench_function("raw_gradient", |b| {
            b.iter(|| black_box(target.logp_grad(black_box(&q), black_box(&mut g)).unwrap()))
        });
        group.bench_function("checked_gradient", |b| {
            b.iter(|| black_box(evaluate(&target, black_box(&q), black_box(&mut g)).unwrap()))
        });
        group.bench_function("checked_density_only", |b| {
            b.iter(|| black_box(density::evaluate(&target, black_box(&q)).unwrap()))
        });
        let mut point = PointState::new(&target, alea_math::buffer::OwnedBuffer::new(n)).unwrap();
        let mut work = EvaluationWorkspace::new(n);
        group.bench_function("transactional_update_reused", |b| {
            b.iter(|| {
                point.try_update(black_box(&q), &mut work).unwrap();
                black_box((point.log_density(), point.gradient()));
            })
        });
        group.finish();
    }
    for rows in [4, 128, 4096] {
        for (name, batch) in [("scalar", false), ("batch", true)] {
            let mut group = c.benchmark_group(format!("model_pipeline/{name}/{rows}"));
            group.bench_function("register_and_drop", |b| {
                b.iter(|| black_box(builder(black_box(rows), batch)))
            });
            group.bench_function("compile", |b| {
                b.iter_batched(
                    || builder(rows, batch),
                    |b| black_box(b.compile().unwrap()),
                    BatchSize::SmallInput,
                )
            });
            group.bench_function("into_target", |b| {
                b.iter_batched(
                    || builder(rows, batch).compile().unwrap(),
                    |m| black_box(m.into_target().unwrap()),
                    BatchSize::SmallInput,
                )
            });
            let model = builder(rows, batch).compile().unwrap();
            if batch {
                group.bench_function("rebind_existing_shape", |b| {
                    b.iter_batched(
                        || {
                            let model = builder(rows, true).compile().unwrap();
                            let expression = model
                                .graph()
                                .find(|n| n.primitive.is_some())
                                .unwrap()
                                .expression;
                            (model, expression, vec![0.1; rows])
                        },
                        |(model, expression, y)| {
                            black_box(
                                model
                                    .rebind::<NormalBatch>(expression, |p| p.with_observations(y))
                                    .unwrap(),
                            )
                        },
                        BatchSize::SmallInput,
                    )
                });
                let reporting = builder(rows, true)
                    .compile()
                    .unwrap()
                    .batch_likelihood()
                    .unwrap();
                let mut values = vec![0.; rows];
                group.bench_function("conditional_row_reporting_reused", |b| {
                    b.iter(|| {
                        black_box(
                            reporting
                                .evaluate(black_box(&[0.2, 0.1]), black_box(&mut values))
                                .unwrap(),
                        )
                    })
                });
            }
            let mut g = [0.; 2];
            let q = [0.2, 1.3]; // Constrained scale for model-only evaluation.
            assert!(model.logp_grad(&q, &mut g).unwrap().is_finite());
            group.bench_function("constrained_value_gradient_reused", |b| {
                b.iter(|| black_box(model.logp_grad(black_box(&q), black_box(&mut g)).unwrap()))
            });
            let target = model.into_target().unwrap();
            let q = [0.2, 1.3_f64.ln()];
            assert!(target.logp_grad(&q, &mut g).unwrap().is_finite());
            group.bench_function("unconstrained_value_gradient_reused", |b| {
                b.iter(|| black_box(target.logp_grad(black_box(&q), black_box(&mut g)).unwrap()))
            });
            group.finish();
        }
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=pipeline}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
