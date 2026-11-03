//! Forced legal CPU candidates, not a sampler/Stan benchmark. Construction and
//! retained fused evaluations are separate; mathematical kernels are not copied.
use alea_core::model::assembly::{ModelBuilder, Primitive};
use alea_distributions::modeling::regression::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};

fn family<P: Primitive>(
    c: &mut Criterion,
    name: &str,
    rows: usize,
    columns: usize,
    prepare: impl Fn(RegressionExecution) -> P,
    q: &[f64],
) {
    eprintln!("ALEA_CPU_PLAN_BEGIN {name} {rows} {columns}");
    let variants = if std::env::var_os("ALEA_REGRESSION_PLAN_REVERSE").is_some() {
        [RegressionExecution::Auto, RegressionExecution::Scalar]
    } else {
        [RegressionExecution::Scalar, RegressionExecution::Auto]
    };
    let mut construction =
        c.benchmark_group(format!("regression_plan/build/{name}/{rows}/{columns}"));
    for mode in variants {
        construction.bench_function(format!("{mode:?}"), |b| {
            b.iter(|| black_box(prepare(black_box(mode))))
        });
    }
    construction.finish();
    let mut evaluation =
        c.benchmark_group(format!("regression_plan/evaluate/{name}/{rows}/{columns}"));
    evaluation.throughput(Throughput::Elements(1)); // One fused value/gradient call.
    let mut reference: Option<(f64, Vec<f64>)> = None;
    for mode in variants {
        let primitive = prepare(mode);
        let mut work = primitive.workspace();
        let mut gradient = vec![0.; q.len()];
        let value = primitive.evaluate(q, &mut gradient, &mut work).unwrap();
        if let Some((expected, score)) = &reference {
            assert!((value - expected).abs() <= 3e-12 * (1. + expected.abs()));
            for (actual, expected) in gradient.iter().zip(score) {
                assert!((actual - expected).abs() <= 3e-12 * (1. + expected.abs()));
            }
        } else {
            reference = Some((value, gradient.clone()));
        }
        evaluation.bench_function(format!("{mode:?}"), |b| {
            b.iter(|| {
                black_box(
                    primitive
                        .evaluate(black_box(q), black_box(&mut gradient), &mut work)
                        .unwrap(),
                )
            })
        });
        // A narrow machine-readable receipt, emitted only after this candidate
        // completes. Fixed family/mode names and admitted finite numbers need no
        // string escaping. The campaign independently checks BOTH candidates;
        // an incomplete/mismatched pair never acquires qualified timing.
        eprintln!("ALEA_CPU_PLAN_VALUE {name} {rows} {columns} {mode:?} {value:?} {gradient:?}");
    }
    evaluation.finish();
}
fn benchmark(c: &mut Criterion) {
    // Cloned RegressionData now shares lazy prepared predictors. Construction
    // below measures repeated descriptor binding to those shared data, not fresh
    // input allocation/packing. Cross-version campaign readers must not compare
    // this scope with the older private-pack-per-descriptor construction cost.
    eprintln!("ALEA_CPU_PLAN_BUILD_SCOPE shared-prepared-data-v2");
    // An explicitly smaller smoke campaign; the default retains the full matrix.
    let quick = std::env::var_os("ALEA_REGRESSION_PLAN_QUICK").is_some();
    let held_out = std::env::var_os("ALEA_REGRESSION_PLAN_HELD_OUT").is_some();
    assert!(!(quick && held_out), "choose only one CPU-plan shape grid");
    let (sizes, dimensions): (&[usize], &[usize]) = if quick {
        (&[1_000], &[4])
    } else if held_out {
        (&[1, 7, 127, 2_049], &[2, 7, 31])
    } else {
        (&[3, 1_000, 10_000], &[0, 1, 4, 16, 64])
    };
    for &rows in sizes {
        for &columns in dimensions {
            let mut builder = ModelBuilder::new();
            let beta: Vec<_> = (0..columns)
                .map(|j| builder.real(&format!("b{j}")).unwrap().expr())
                .collect();
            let sigma = builder.positive("sigma").unwrap();
            let design: Vec<_> = (0..rows)
                .flat_map(|i| (0..columns).map(move |j| ((i + 3 * j) % 13) as f64 / 16. - 0.375))
                .collect();
            let normal = RegressionData::new(
                columns,
                design.clone(),
                (0..rows).map(|i| (i % 7) as f64 / 4. - 0.5).collect(),
            )
            .unwrap();
            let logit = RegressionData::new(
                columns,
                design.clone(),
                (0..rows).map(|i| i % 3 != 0).collect(),
            )
            .unwrap();
            let poisson =
                RegressionData::new(columns, design, (0..rows).map(|i| (i % 4) as u64).collect())
                    .unwrap();
            let q: Vec<_> = (0..columns).map(|j| j as f64 / 256. - 0.125).collect();
            let mut nq = q.clone();
            nq.push(1.25);
            let plan = NormalRegression::new(&beta, &sigma, normal.clone()).unwrap();
            eprintln!("native regression plan: {:?}", plan.execution_plan());
            family(
                c,
                "normal",
                rows,
                columns,
                |mode| {
                    NormalRegression::new(&beta, &sigma, normal.clone())
                        .unwrap()
                        .with_execution(mode)
                },
                &nq,
            );
            family(
                c,
                "logit",
                rows,
                columns,
                |mode| {
                    BernoulliLogitRegression::new(&beta, logit.clone())
                        .unwrap()
                        .with_execution(mode)
                },
                &q,
            );
            family(
                c,
                "poisson",
                rows,
                columns,
                |mode| {
                    PoissonLogRegression::new(&beta, poisson.clone())
                        .unwrap()
                        .with_execution(mode)
                },
                &q,
            );
        }
    }
}
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20).warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_secs(1));
    targets = benchmark
}
criterion_main!(benches);
