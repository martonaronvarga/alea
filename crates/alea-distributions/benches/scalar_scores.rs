//! Public native scalar laws: density, each score, separate tuple, fused tuple.
use alea_core::distribution::{ParameterScore, ScalarEventScore};
use alea_distributions::scalar::{
    Beta, Cauchy, Exponential, Gamma, Laplace, LogNormal, Logistic, Normal, StudentT, Weibull,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn law<D>(c: &mut Criterion, name: &str, make: impl Fn() -> D, x: f64)
where
    D: ParameterScore<Event = f64> + ScalarEventScore,
{
    let law = make();
    let mut scores = vec![0.; law.parameter_count()];
    let (value, event) = law.log_prob_event_parameter_score(x, &mut scores).unwrap();
    assert!(value.is_finite() && event.is_finite() && scores.iter().all(|s| s.is_finite()));
    let mut separate = scores.clone();
    law.parameter_score(&x, &mut separate).unwrap();
    assert_eq!(scores, separate);
    assert_eq!(event, law.event_score(x).unwrap());
    assert!((value - law.log_prob(&x).unwrap()).abs() <= 1e-12 * (1. + value.abs()));
    let mut group = c.benchmark_group(format!("scalar_scores/{name}"));
    group.bench_function("construct_drop", |b| b.iter(|| black_box(make())));
    group.bench_function("density", |b| {
        b.iter(|| black_box(law.log_prob(black_box(&x)).unwrap()))
    });
    group.bench_function("event", |b| {
        b.iter(|| black_box(law.event_score(black_box(x)).unwrap()))
    });
    group.bench_function("parameters", |b| {
        b.iter(|| {
            law.parameter_score(black_box(&x), black_box(&mut scores))
                .unwrap();
            black_box(&scores);
        })
    });
    group.bench_function("separate_value_event_parameters", |b| {
        b.iter(|| {
            let v = law.log_prob(black_box(&x)).unwrap();
            let e = law.event_score(black_box(x)).unwrap();
            law.parameter_score(black_box(&x), black_box(&mut scores))
                .unwrap();
            black_box((v, e, &scores));
        })
    });
    group.bench_function("combined_value_event_parameters", |b| {
        b.iter(|| {
            let value = law
                .log_prob_event_parameter_score(black_box(x), black_box(&mut scores))
                .unwrap();
            black_box((value, &scores));
        })
    });
    group.finish();
}
fn scalar_scores(c: &mut Criterion) {
    law(
        c,
        "normal",
        || Normal::new(black_box(0.2), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "cauchy_jet",
        || Cauchy::new(black_box(0.2), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "student_t",
        || StudentT::new(black_box(5.), black_box(0.2), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "log_normal",
        || LogNormal::new(black_box(0.2), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "gamma",
        || Gamma::new(black_box(2.3), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "beta",
        || Beta::new(black_box(2.3), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "exponential",
        || Exponential::new(black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "laplace_jet",
        || Laplace::new(black_box(0.2), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "logistic_jet",
        || Logistic::new(black_box(0.2), black_box(1.3)).unwrap(),
        0.7,
    );
    law(
        c,
        "weibull_jet",
        || Weibull::new(black_box(2.3), black_box(1.3)).unwrap(),
        0.7,
    );
    for k in [2, 8, 64] {
        let mixture = || {
            alea_distributions::mixture::FiniteMixture::new(
                vec![1. / k as f64; k],
                (0..k)
                    .map(|i| Normal::new(i as f64 / k as f64, 1.3).unwrap())
                    .collect(),
            )
            .unwrap()
        };
        law(c, &format!("normal_mixture/{k}"), mixture, 0.7);
    }
}
fn vector_scores(c: &mut Criterion) {
    use alea_core::distribution::Distribution;
    use alea_distributions::{directional::UniformSphere, multivariate::MultivariateNormal};
    for n in [2, 8, 32] {
        let make = || {
            MultivariateNormal::new(
                vec![0.2; n],
                (0..n * n)
                    .map(|i| if i / n == i % n { 1.3 } else { 0. })
                    .collect(),
            )
            .unwrap()
        };
        let distribution = make();
        let x = vec![0.1; n];
        let mut g = vec![0.; n];
        let mut p = vec![0.; distribution.parameter_count()];
        let mut group = c.benchmark_group(format!("vector_scores/{n}"));
        group.bench_function("normal_construct_drop", |b| b.iter(|| black_box(make())));
        group.bench_function("normal_density", |b| {
            b.iter(|| black_box(distribution.log_prob(black_box(&x)).unwrap()))
        });
        group.bench_function("normal_event", |b| {
            b.iter(|| {
                distribution
                    .event_gradient(black_box(&x), black_box(&mut g))
                    .unwrap();
                black_box(&g);
            })
        });
        group.bench_function("normal_parameters", |b| {
            b.iter(|| {
                distribution
                    .parameter_score(black_box(&x), black_box(&mut p))
                    .unwrap();
                black_box(&p);
            })
        });
        let sphere = UniformSphere::new(n).unwrap();
        let mut unit = vec![0.; n];
        unit[0] = 1.;
        group.bench_function("sphere_density", |b| {
            b.iter(|| black_box(sphere.log_prob(black_box(&unit)).unwrap()))
        });
        group.finish();
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=scalar_scores, vector_scores}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
