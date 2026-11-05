//! Identical dyadic deterministic objective through block and scalar tapes.
//! No probability distribution/derivative/sampler is reimplemented here.
use alea_core::model::{
    ConstrainedModel,
    assembly::{CompiledModel, ModelBuilder},
    tensor::{BinaryOp as B, BoundProgram, MatrixKernel, Shape, TensorBuilder, TensorFactor},
};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

fn region(n: usize) -> BoundProgram {
    let mut b = TensorBuilder::new();
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
fn assembled(n: usize, block: bool) -> CompiledModel {
    let mut b = ModelBuilder::new();
    let slope = b.real("slope").unwrap().expr();
    let result = if block {
        b.primitive(TensorFactor::new(region(n), &[slope]).unwrap())
            .unwrap()
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
            MatrixKernel::Compensated,
            #[cfg(feature = "faer")]
            MatrixKernel::Faer,
            #[cfg(feature = "openblas")]
            MatrixKernel::Blas,
        ] {
            let mut b = TensorBuilder::new();
            let a = b.input("a", Shape::new(vec![n, n]).unwrap()).unwrap();
            let r = b.input("r", Shape::new(vec![n, n]).unwrap()).unwrap();
            let out = b.matmul_with_kernel(a, r, kernel).unwrap();
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
        let mut scratch = direct.workspace();
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
fn broadcast_region(n: usize) -> BoundProgram {
    let mut b = TensorBuilder::new();
    let column = b.input("column", Shape::new(vec![n, 1]).unwrap()).unwrap();
    let row = b.input("row", Shape::new(vec![8]).unwrap()).unwrap();
    let product = b.binary(B::Multiply, column, row).unwrap();
    let square = b.binary(B::Multiply, product, product).unwrap();
    let out = b.sum(square).unwrap();
    b.compile(out).unwrap().bind(&[]).unwrap()
}
// Separate cold planning/emission/allocation from repeated forward and VJP work.
// Compact metadata reduces storage, but integer indexing may cost hot time;
// these controls must accompany remote before/after measurements.
fn broadcast_maps(c: &mut Criterion) {
    for n in [128, 4096, 65_536] {
        let p = broadcast_region(n);
        let mut group = c.benchmark_group(format!("tensor_broadcast/{n}"));
        group.bench_function("plan_bind", |b| {
            b.iter(|| black_box(broadcast_region(black_box(n))))
        });
        group.bench_function("primal_source", |b| b.iter(|| black_box(p.rust_function())));
        group.bench_function("workspace", |b| b.iter(|| black_box(p.workspace())));
        let x = vec![0.25; p.dimension()];
        let mut w = p.workspace();
        let mut out = [0.];
        let mut g = vec![0.; x.len()];
        p.value_and_vjp(&x, &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        assert_eq!(out[0], n as f64 / 32.);
        assert!(g[..n].iter().all(|&v| v == 0.25));
        assert!(g[n..].iter().all(|&v| v == n as f64 / 32.));
        group.throughput(Throughput::Elements((8 * n) as u64));
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
criterion_group! {name=benches; config=benchmark_support::criterion().sample_size(30).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2)); targets=benchmark,matrix_kernels,broadcast_maps}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
