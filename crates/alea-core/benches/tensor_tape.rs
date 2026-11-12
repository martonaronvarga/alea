//! Identical dyadic deterministic objective through block and scalar tapes.
//! No probability distribution/derivative/sampler is reimplemented here.
use alea_core::model::{
    ConstrainedModel,
    assembly::{ModelBuilder, NativeModel},
    tensor::{BinaryOp as B, BoundProgram, MatrixRequest, Shape},
};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

fn region(n: usize) -> BoundProgram {
    let mut b = ModelBuilder::new().region();
    let slope = b.input("slope", Shape::scalar()).unwrap();
    let weights = b.data("weights", Shape::new(vec![n]).unwrap()).unwrap();
    let product = b.binary(B::Multiply, slope, weights).unwrap();
    let offset = b.constant(Shape::scalar(), vec![0.125]).unwrap();
    let shifted = b.binary(B::Add, product, offset).unwrap();
    let square = b.binary(B::Multiply, shifted, shifted).unwrap();
    let result = b.sum(square).unwrap();
    let data: Vec<_> = (0..n).map(|i| (i % 17) as f64 / 16.).collect();
    b.compile(result)
        .unwrap()
        .bind(&[("weights", &data)])
        .unwrap()
}
fn assembled(n: usize, block: bool) -> NativeModel {
    let mut b = ModelBuilder::new();
    let slope = b.real("slope").unwrap().expr();
    let result = if block {
        b.apply_region(&[slope], region(n)).unwrap()
    } else {
        let offset = b.constant(0.125).unwrap();
        let terms: Vec<_> = (0..n)
            .map(|i| {
                let p = b.multiply(slope, (i % 17) as f64 / 16.).unwrap();
                let shifted = b.add(p, offset).unwrap();
                b.multiply(shifted, shifted).unwrap()
            })
            .collect();
        b.sum(&terms).unwrap()
    };
    b.potential("objective", result).unwrap();
    b.compile().unwrap()
}
fn matrix_kernels(c: &mut Criterion) {
    for n in [8, 64, 128] {
        let mut group = c.benchmark_group(format!("tensor_gemm/{n}"));
        let x: Vec<_> = (0..2 * n * n)
            .map(|i| ((i % 31) as f64 - 15.) / 32.)
            .collect();
        let seed = vec![0.5; n * n];
        let mut reference: Option<(Vec<f64>, Vec<f64>)> = None;
        for kernel in [
            MatrixRequest::default(),
            #[cfg(feature = "faer")]
            MatrixRequest::standard(alea_core::model::tensor::MatrixProvider::Faer),
            #[cfg(feature = "openblas")]
            MatrixRequest::standard(alea_core::model::tensor::MatrixProvider::Blas),
        ] {
            let mut b = ModelBuilder::new().region();
            let a = b.input("a", Shape::new(vec![n, n]).unwrap()).unwrap();
            let r = b.input("r", Shape::new(vec![n, n]).unwrap()).unwrap();
            let out = b.matmul_with_request(a, r, kernel).unwrap();
            let p = b.compile(out).unwrap().bind(&[]).unwrap();
            let mut w = p.workspace();
            let mut out = vec![0.; n * n];
            let mut g = vec![0.; x.len()];
            p.value_and_vjp(&x, &seed, &mut out, &mut g, &mut w)
                .unwrap();
            if let Some((values, scores)) = &reference {
                for (got, want) in out.iter().zip(values).chain(g.iter().zip(scores)) {
                    assert!((got - want).abs() < 1e-10);
                }
            } else {
                reference = Some((out.clone(), g.clone()));
            }
            group.bench_function(format!("{kernel:?}"), |b| {
                b.iter(|| {
                    p.value_and_vjp(
                        black_box(&x),
                        black_box(&seed),
                        black_box(&mut out),
                        black_box(&mut g),
                        &mut w,
                    )
                    .unwrap();
                    black_box(&out);
                })
            });
        }
        group.finish();
    }
}
fn benchmark(c: &mut Criterion) {
    for n in [1_000, 10_000] {
        let scalar = assembled(n, false);
        let block = assembled(n, true);
        let direct = region(n);
        let mut scratch = None;
        let allocation = allocation_counter::measure(|| {
            scratch = Some(direct.workspace());
        });
        let workspace_owned_bytes = usize::try_from(allocation.bytes_current).unwrap();
        let mut scratch = scratch.unwrap();
        use benchmark_support::fixed_work::{self, Fixture};
        let bank = fixed_work::states(1);
        let coordinate_id = fixed_work::identity(b"constrained/slope");
        for (kind, model) in [("scalar", &scalar), ("block", &block)] {
            let name = format!("tensor_{kind}_{n}");
            let fixture = Fixture {
                name: &name,
                model: fixed_work::identity(model.rust_objective_primal().unwrap().as_bytes()),
                coordinates: coordinate_id,
                query: "constrained-objective",
                algorithm: "fused-logp-grad",
                arithmetic: "fixture-default",
                provider: "rust",
            };
            fixed_work::run(&fixture, "N", &bank, None, |q, g| model.logp_grad(q, g)).unwrap();
        }
        let name = format!("tensor_direct_{n}");
        let fixture = Fixture {
            name: &name,
            model: fixed_work::identity(direct.rust_function().unwrap().as_bytes()),
            coordinates: coordinate_id,
            query: "constrained-objective",
            algorithm: "fused-logp-grad",
            arithmetic: "fixture-default",
            provider: "rust",
        };
        fixed_work::run(&fixture, "N", &bank, Some(workspace_owned_bytes), |q, g| {
            let mut value = [0.];
            direct.value_and_vjp(q, &[1.], &mut value, g, &mut scratch)?;
            Ok::<_, alea_core::model::assembly::AssemblyError>(value[0])
        })
        .unwrap();
        let mut out = [0.];
        let mut g = [0.];
        let expected = scalar.logp_grad(&[0.5], &mut g).unwrap();
        let expected_g = g;
        assert_eq!(block.logp_grad(&[0.5], &mut g).unwrap(), expected);
        assert_eq!(g, expected_g);
        direct
            .value_and_vjp(&[0.5], &[1.], &mut out, &mut g, &mut scratch)
            .unwrap();
        assert_eq!(out[0], expected);
        assert_eq!(g, expected_g);
        let mut group = c.benchmark_group(format!("tensor_tape/{n}"));
        group.throughput(Throughput::Elements(n as u64));
        let mut variants = [("scalar", &scalar), ("tensor_factor", &block)];
        if std::env::var_os("ALEA_TENSOR_REVERSE").is_some() {
            variants.reverse();
        }
        for (name, model) in variants {
            group.bench_function(name, |b| {
                b.iter(|| {
                    black_box(
                        model
                            .logp_grad(black_box(&[0.5]), black_box(&mut g))
                            .unwrap(),
                    )
                })
            });
        }
        group.bench_function("tensor_direct", |b| {
            b.iter(|| {
                direct
                    .value_and_vjp(
                        black_box(&[0.5]),
                        &[1.],
                        black_box(&mut out),
                        black_box(&mut g),
                        &mut scratch,
                    )
                    .unwrap();
                black_box(out);
            })
        });
        group.finish();
    }
}
fn broadcast_region(n: usize, width: usize, gather: bool, frozen: bool) -> BoundProgram {
    let mut b = ModelBuilder::new().region();
    let column = b.input("column", Shape::new(vec![n, 1]).unwrap()).unwrap();
    let row_shape = Shape::new(vec![width]).unwrap();
    let row = if frozen {
        b.data("row", row_shape).unwrap()
    } else {
        b.input("row", row_shape).unwrap()
    };
    let (column, row) = if gather {
        let shape = Shape::new(vec![n, width]).unwrap();
        (
            b.gather(
                column,
                shape.clone(),
                (0..n * width).map(|i| i / width).collect(),
            )
            .unwrap(),
            b.gather(row, shape, (0..n * width).map(|i| i % width).collect())
                .unwrap(),
        )
    } else {
        (column, row)
    };
    let product = b.binary(B::Multiply, column, row).unwrap();
    let square = b.binary(B::Multiply, product, product).unwrap();
    let out = b.sum(square).unwrap();
    let program = b.compile(out).unwrap();
    if frozen {
        program.bind(&[("row", &vec![0.25; width])]).unwrap()
    } else {
        program.bind(&[]).unwrap()
    }
}
// Retain the original width-eight controls and names. Added cells distinguish
// compact read maps from explicit (potentially repeated) gather storage.
fn broadcast_maps(c: &mut Criterion) {
    let shapes = [128, 4096, 65_536].into_iter().map(|n| (n, 8)).chain(
        [1, 3, 64, 1024, 65_536]
            .into_iter()
            .flat_map(|n| [1, 3, 16].into_iter().map(move |width| (n, width))),
    );
    for (n, width) in shapes {
        for gather in [false, true] {
            let name = if gather {
                format!("tensor_broadcast_gather/{n}x{width}")
            } else if width == 8 {
                format!("tensor_broadcast/{n}")
            } else {
                format!("tensor_broadcast/{n}x{width}")
            };
            let mut program = None;
            let plan = allocation_counter::measure(|| {
                program = Some(broadcast_region(n, width, gather, false));
            });
            let p = program.unwrap();
            let source = p.rust_function().unwrap();
            let frozen = broadcast_region(n, width, gather, true);
            let frozen_source = frozen.rust_function().unwrap();
            let mut workspace = None;
            let workspace_alloc = allocation_counter::measure(|| workspace = Some(p.workspace()));
            let mut w = workspace.unwrap();
            let x = vec![0.25; p.dimension()];
            let mut out = [0.];
            let mut g = vec![0.; x.len()];
            let first = allocation_counter::measure(|| {
                p.value_and_vjp(&x, &[1.], &mut out, &mut g, &mut w)
                    .unwrap();
            });
            // Dense coordinate oracle, independent of map/gather planning.
            assert_eq!(out[0], (n * width) as f64 / 256.);
            assert!(g[..n].iter().all(|&v| v == width as f64 / 32.));
            assert!(g[n..].iter().all(|&v| v == n as f64 / 32.));
            let mut frozen_out = [0.];
            let mut frozen_g = vec![0.; n];
            frozen
                .value_and_vjp(
                    &x[..n],
                    &[1.],
                    &mut frozen_out,
                    &mut frozen_g,
                    &mut frozen.workspace(),
                )
                .unwrap();
            assert_eq!(frozen_out, out);
            assert_eq!(frozen_g, g[..n]);
            let steady = allocation_counter::measure(|| {
                p.value_and_vjp(&x, &[1.], &mut out, &mut g, &mut w)
                    .unwrap();
            });
            assert_eq!(steady.count_total, 0);
            eprintln!(
                "ALEA_BROADCAST_STORAGE_V1\t{name}\tplan_owned={}\tsource_runtime={}\tsource_frozen={}\tfrozen_payload={}\tworkspace_budget={}\tworkspace_owned={}\tfirst_allocations={}\tfirst_bytes={}\tsteady_allocations={}\tsteady_bytes={}",
                plan.bytes_current,
                source.len(),
                frozen_source.len(),
                width * size_of::<f64>(),
                p.workspace_bytes(),
                workspace_alloc.bytes_current,
                first.count_total,
                first.bytes_total,
                steady.count_total,
                steady.bytes_total,
            );
            // Existing fixed-work receipt owns optional first/steady timing.
            // Use a fresh workspace so the first phase includes binding refresh.
            let mut fresh = p.workspace();
            let receipt_name = name.replace('/', "_");
            let fixture = benchmark_support::fixed_work::Fixture {
                name: &receipt_name,
                model: benchmark_support::fixed_work::identity(source.as_bytes()),
                coordinates: benchmark_support::fixed_work::identity(b"column/row-row-major"),
                query: "constrained-objective",
                algorithm: "fused-logp-grad",
                arithmetic: "fixture-default",
                provider: "rust",
            };
            benchmark_support::fixed_work::run(
                &fixture,
                "N",
                &benchmark_support::fixed_work::states(x.len()),
                Some(usize::try_from(workspace_alloc.bytes_current).unwrap()),
                |q, g| {
                    p.value_and_vjp(q, &[1.], &mut out, g, &mut fresh)?;
                    Ok::<_, alea_core::model::assembly::AssemblyError>(out[0])
                },
            )
            .unwrap();
            let mut group = c.benchmark_group(name);
            group.bench_function("plan_bind", |b| {
                b.iter(|| black_box(broadcast_region(black_box(n), width, gather, false)))
            });
            group.bench_function("primal_source", |b| {
                b.iter(|| black_box(p.rust_function().unwrap()))
            });
            group.bench_function("frozen_source", |b| {
                b.iter(|| black_box(frozen.rust_function().unwrap()))
            });
            group.bench_function("workspace", |b| b.iter(|| black_box(p.workspace())));
            group.throughput(Throughput::Elements((width * n) as u64));
            group.bench_function("first_value_vjp", |b| {
                b.iter_batched(
                    || p.workspace(),
                    |mut fresh| {
                        p.value_and_vjp(black_box(&x), &[1.], &mut out, &mut g, &mut fresh)
                            .unwrap();
                        black_box(&g);
                    },
                    criterion::BatchSize::LargeInput,
                )
            });
            group.bench_function("forward", |b| {
                b.iter(|| {
                    p.evaluate(black_box(&x), black_box(&mut out), &mut w)
                        .unwrap();
                    black_box(out);
                })
            });
            group.bench_function("value_vjp", |b| {
                b.iter(|| {
                    p.value_and_vjp(
                        black_box(&x),
                        &[1.],
                        black_box(&mut out),
                        black_box(&mut g),
                        &mut w,
                    )
                    .unwrap();
                    black_box(&g);
                })
            });
            group.finish();
        }
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion().sample_size(30).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2)); targets=benchmark,matrix_kernels,broadcast_maps}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
