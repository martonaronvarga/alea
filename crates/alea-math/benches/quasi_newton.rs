use alea_math::quasi_newton::{lbfgs_inverse_product, lbfgs_inverse_product_with_scratch};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn actions(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse_bfgs_action");
    for dim in [4, 64, 1024] {
        let s: Vec<Vec<f64>> = (0..8)
            .map(|k| (0..dim).map(|i| ((i + k) as f64 + 1.).sin()).collect())
            .collect();
        let y: Vec<Vec<f64>> = s
            .iter()
            .map(|v| v.iter().map(|x| 2. * x).collect())
            .collect();
        let v = vec![0.5; dim];
        let mut output = vec![0.; dim];
        let mut scratch = [0.; 8];
        lbfgs_inverse_product(&v, &s, &y, &mut output).unwrap();
        let expected = output.clone();
        lbfgs_inverse_product_with_scratch(&v, &s, &y, &mut output, &mut scratch).unwrap();
        assert_eq!(output, expected);
        group.bench_with_input(BenchmarkId::new("allocated", dim), &dim, |b, _| {
            b.iter(|| {
                lbfgs_inverse_product(black_box(&v), &s, &y, black_box(&mut output)).unwrap()
            });
        });
        group.bench_with_input(BenchmarkId::new("reused", dim), &dim, |b, _| {
            b.iter(|| {
                lbfgs_inverse_product_with_scratch(
                    black_box(&v),
                    &s,
                    &y,
                    black_box(&mut output),
                    &mut scratch,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}
criterion_group!(benches, actions);
criterion_main!(benches);
