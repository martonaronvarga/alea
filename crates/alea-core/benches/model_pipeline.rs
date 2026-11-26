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
                                    .binding()
                                    .rebind::<NormalBatch>(expression, |p| p.with_observations(y))
                                    .unwrap()
                                    .compile()
                                    .unwrap()
                                    .into_native_model(),
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
            use benchmark_support::fixed_work::{self, Fixture};
            let bank = fixed_work::states(2);
            let model_id = fixed_work::identity(model.rust_primal().unwrap().as_bytes());
            let coordinates = fixed_work::identity(format!("{:?}", model.coordinates()).as_bytes());
            let fixture_name = format!("pipeline_{name}_{rows}_constrained");
            let fixture = Fixture {
                name: &fixture_name,
                model: model_id,
                coordinates,
                query: "constrained-density",
                algorithm: "fused-logp-grad",
                arithmetic: "fixture-default",
                provider: "rust",
            };
            fixed_work::run(&fixture, "N", &bank, None, |q, g| model.logp_grad(q, g)).unwrap();
            let mut g = [0.; 2];
            let q = [0.2, 1.3]; // Constrained scale for model-only evaluation.
            assert!(model.logp_grad(&q, &mut g).unwrap().is_finite());
            group.bench_function("constrained_value_gradient_reused", |b| {
                b.iter(|| black_box(model.logp_grad(black_box(&q), black_box(&mut g)).unwrap()))
            });
            let target = model.into_target().unwrap();
            let fixture_name = format!("pipeline_{name}_{rows}_unconstrained");
            let fixture = Fixture {
                name: &fixture_name,
                query: "density/current-factor-selection",
                ..fixture
            };
            fixed_work::run(&fixture, "N", &bank, None, |q, g| target.logp_grad(q, g)).unwrap();
            let q = [0.2, 1.3_f64.ln()];
            assert!(target.logp_grad(&q, &mut g).unwrap().is_finite());
            group.bench_function("unconstrained_value_gradient_reused", |b| {
                b.iter(|| black_box(target.logp_grad(black_box(&q), black_box(&mut g)).unwrap()))
            });
            group.finish();
        }
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=pipeline, compositions}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);

// Reuse the qualified constructors rather than copying models or law formulas.
#[allow(dead_code, unexpected_cfgs)]
#[path = "../../../tools/native-benchmark/src/conformance.rs"]
mod composition_models;

fn compositions(c: &mut Criterion) {
    use alea::model::assembly::{
        CurvatureRoute, DerivativeOrder, DerivativeSource, SourceMode, SourceRequest,
    };
    use benchmark_support::fixed_work::{self, Fixture};
    for name in ["covariance", "schools", "schools_nc", "radon"] {
        let (n, k, x, y, county) = match name {
            "covariance" => {
                let (x, y) = composition_models::covariance_data();
                (32, 2, x, y, Vec::new())
            }
            "radon" => {
                if let Some(path) = std::env::var_os("ALEA_RADON_DATA") {
                    let data: serde_json::Value = serde_json::from_str(
                        &std::fs::read_to_string(path).expect("declared Radon input must exist"),
                    )
                    .unwrap();
                    let j = usize::try_from(data["J"].as_u64().unwrap()).unwrap();
                    let n = usize::try_from(data["N"].as_u64().unwrap()).unwrap();
                    let county = data["county_idx"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|i| {
                            usize::try_from(i.as_u64().unwrap())
                                .unwrap()
                                .checked_sub(1)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let values = |name: &str| {
                        data[name]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect::<Vec<_>>()
                    };
                    let y = values("log_radon");
                    assert_eq!(n, y.len());
                    (n, j, values("floor_measure"), y, county)
                } else {
                    (
                        5,
                        3,
                        vec![0., 1., 1., 0., 0.],
                        vec![0.5, 1., -0.25, 0., 0.75],
                        vec![2, 0, 2, 1, 0],
                    )
                }
            }
            _ => (
                8,
                1,
                vec![15., 10., 16., 11., 9., 11., 10., 18.],
                vec![28., 8., -3., 7., -1., 1., 18., 12.],
                Vec::new(),
            ),
        };
        // File parsing is outside the model stages. The row count and input bits
        // distinguish the tiny default control from an explicitly pinned corpus.
        let build = |block: bool| {
            if name == "radon" {
                if block {
                    composition_models::radon(k, &county, &x, &y)
                } else {
                    composition_models::radon_scalar(k, &county, &x, &y)
                }
            } else if block {
                composition_models::native(name, n, k, &x, &y)
            } else {
                composition_models::build(name, n, k, &x, &y)
            }
        };
        let dimension = build(false).unwrap().dimension();
        let bank = fixed_work::states(dimension);
        // Canonical Stan coordinate order explicitly maps scalar effect0..N to
        // theta[1..N]/z[1..N]. Names are aliases; ordering/charts are qualified.
        let coordinates =
            fixed_work::identity(format!("{name}:Stan-free-order:{dimension}").as_bytes());
        let model_id =
            fixed_work::identity(format!("{name}:full:{n}:{k}:{x:?}:{y:?}:{county:?}").as_bytes());
        let mut reference: Option<Vec<f64>> = None;
        let mut representations = [("scalar", false), ("block", true)];
        if std::env::var_os("ALEA_COMPOSITION_REVERSE").is_some() {
            representations.reverse();
        }
        for (representation, block) in representations {
            let model = build(block).unwrap();
            let fixture_name = format!("composition_{name}_{n}_{representation}");
            let fixture = Fixture {
                name: &fixture_name,
                model: model_id,
                coordinates,
                query: "density/current-factor-selection",
                algorithm: "fused-logp-grad",
                arithmetic: "fixture-default",
                provider: "rust",
            };
            let source = model.rust_primal().is_ok();
            let first = if source {
                if matches!(
                    model
                        .source_derivatives(SourceRequest::first(SourceMode::Density))
                        .unwrap(),
                    DerivativeSource::Composed(_)
                ) {
                    "H"
                } else {
                    "C"
                }
            } else {
                "none"
            };
            for requested in ["N", "C", "L", "H"] {
                for (order, derivative) in [
                    ("value", DerivativeOrder::Value),
                    ("first", DerivativeOrder::First),
                    ("hvp", DerivativeOrder::Hvp),
                    ("third", DerivativeOrder::Third),
                ] {
                    let (resolved, admission, status, reason) = if requested == "N" {
                        if matches!(derivative, DerivativeOrder::Hvp | DerivativeOrder::Third) {
                            (
                                "none",
                                "N",
                                "unsupported",
                                "native target has no curvature trait",
                            )
                        } else if order == "first" {
                            ("N", "N", "not-run", "awaiting fixed-work fused evaluation")
                        } else {
                            ("N", "N", "not-run", "value-only execution not isolated")
                        }
                    } else if !source {
                        (
                            "none",
                            "none",
                            "unsupported",
                            "MissingPrimal; no source route resolved",
                        )
                    } else {
                        let eligible = model
                            .source_capabilities()
                            .resolve(SourceRequest {
                                mode: SourceMode::Density,
                                order: derivative,
                                curvature: CurvatureRoute::OriginalCompilerReverse,
                            })
                            .is_ok();
                        let resolved = if order == "first" { first } else { "C" };
                        let admission = if order == "value" { "none" } else { first };
                        let selected = if order == "value" {
                            requested == "C"
                        } else {
                            requested == first
                        };
                        let status = if !eligible || !selected {
                            "unsupported"
                        } else {
                            "not-run"
                        };
                        (
                            resolved,
                            admission,
                            status,
                            "emission eligibility only; pinned compiled execution not run by native bench",
                        )
                    };
                    eprintln!(
                        "ALEA_BASELINE_ROUTE_V1\t{fixture_name}\t{requested}\t{resolved}\t{order}\tfixture-default\trust\t{admission}\tfalse\t{status}\t{reason}"
                    );
                }
            }
            let target = model.into_target().unwrap();
            let answers =
                fixed_work::run(&fixture, "N", &bank, None, |q, g| target.logp_grad(q, g)).unwrap();
            if let Some(expected) = &reference {
                for (actual, expected) in answers.iter().zip(expected) {
                    assert!((actual - expected).abs() <= 3e-10 * (1. + f64::abs(*expected)));
                }
            } else {
                reference = Some(answers);
            }
            eprintln!(
                "ALEA_BASELINE_ROUTE_V1\t{fixture_name}\tN\tN\tfirst\tfixture-default\trust\tN\tfalse\tsupported\tfixed-work fused evaluation and matched representation preflight passed"
            );
            let model = build(block).unwrap();
            let definition = model.binding().definition();
            let executable = model.binding().compile().unwrap();
            let mut group =
                c.benchmark_group(format!("composition_pipeline/{name}/{n}/{representation}"));
            group.bench_function("construct_compile_context_and_drop", |b| {
                b.iter(|| black_box(build(block).unwrap()))
            });
            group.bench_function("bind_snapshot", |b| b.iter(|| black_box(definition.bind())));
            group.bench_function("compile_executable", |b| {
                b.iter(|| black_box(model.binding().compile().unwrap()))
            });
            group.bench_function("context_create", |b| {
                b.iter(|| black_box(executable.context()))
            });
            group.bench_function("first_value_gradient", |b| {
                b.iter_batched(
                    || {
                        (
                            build(block).unwrap().into_target().unwrap(),
                            vec![0.; dimension],
                        )
                    },
                    |(target, mut g)| {
                        black_box(target.logp_grad(black_box(&bank[0]), &mut g).unwrap())
                    },
                    BatchSize::SmallInput,
                )
            });
            let mut gradient = vec![0.; dimension];
            group.bench_function("steady_value_gradient", |b| {
                b.iter(|| {
                    black_box(
                        target
                            .logp_grad(black_box(&bank[0]), black_box(&mut gradient))
                            .unwrap(),
                    )
                })
            });
            if source {
                group.bench_function("primal_source_generation", |b| {
                    b.iter(|| black_box(model.rust_primal().unwrap()))
                });
            }
            group.finish();
        }
    }
}
