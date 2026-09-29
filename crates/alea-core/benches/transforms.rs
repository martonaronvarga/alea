use alea_core::model::{AnalyticModel, TransformedTarget};
use alea_core::target::evaluate;
use alea_core::transform::{ParameterLayout, Transform};
use alea_math::buffer::OwnedBuffer;
use criterion::{Criterion, criterion_group, criterion_main};
use std::{convert::Infallible, hint::black_box};

fn transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("m2_fused_boundary");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(200));
    for (name, transform) in [
        ("identity_64", Transform::identity(64).unwrap()),
        ("positive_64", Transform::positive(64).unwrap()),
        ("simplex_64", Transform::simplex(64).unwrap()),
        (
            "cholesky_covariance_8",
            Transform::cholesky_covariance(8).unwrap(),
        ),
        ("covariance_8", Transform::covariance(8).unwrap()),
        (
            "cholesky_correlation_8",
            Transform::cholesky_correlation(8).unwrap(),
        ),
        ("correlation_8", Transform::correlation(8).unwrap()),
    ] {
        let model = AnalyticModel::new(
            transform.constrained_dimension(),
            |q: &[f64], g: &mut [f64]| {
                let mut lp = 0.0;
                for (&x, g) in q.iter().zip(g) {
                    lp -= 0.5 * x * x;
                    *g = -x;
                }
                Ok::<_, Infallible>(lp)
            },
        );
        let target =
            TransformedTarget::new(model, ParameterLayout::new([transform]).unwrap()).unwrap();
        let q = OwnedBuffer::from_fn(transform.unconstrained_dimension(), |_| 0.1);
        let mut gradient = OwnedBuffer::new(q.len());
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(evaluate(&target, black_box(&q), black_box(&mut gradient)).unwrap())
            })
        });
    }
    group.finish();
}
criterion_group!(benches, transforms);
criterion_main!(benches);
