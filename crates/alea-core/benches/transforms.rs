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
        // Native chart stages, isolated from target/model arithmetic.
        let q0 = vec![0.1; transform.unconstrained_dimension()];
        let mut x = vec![0.; transform.constrained_dimension()];
        let mut work = vec![0.; transform.scratch_dimension()];
        let gx = vec![0.2; x.len()];
        let mut gq = vec![0.; q0.len()];
        transform.constrain(&q0, &mut x, &mut work).unwrap();
        group.bench_function(format!("constrain/{name}"), |b| {
            b.iter(|| {
                black_box(
                    transform
                        .constrain(black_box(&q0), black_box(&mut x), &mut work)
                        .unwrap(),
                )
            })
        });
        group.bench_function(format!("pullback/{name}"), |b| {
            b.iter(|| {
                transform
                    .pullback(
                        black_box(&q0),
                        black_box(&gx),
                        black_box(&mut gq),
                        &mut work,
                    )
                    .unwrap();
                black_box(&gq);
            })
        });
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
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=transforms}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
