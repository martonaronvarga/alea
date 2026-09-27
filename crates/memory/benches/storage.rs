#![cfg_attr(feature = "simd", feature(portable_simd))]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memory::OwnedBuffer;
use std::{hint::black_box, time::Duration};

#[inline(never)]
fn axpy(x: &[f64], y: &mut [f64]) {
    for (x, y) in x.iter().zip(y) {
        *y += 0.125 * x;
    }
}

#[cfg(feature = "simd")]
#[inline(never)]
fn axpy_simd(x: &[f64], y: &mut [f64]) {
    use std::simd::Simd;
    type V = Simd<f64, 4>;
    let n = x.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        let out = V::from_slice(&y[i..]) + V::splat(0.125) * V::from_slice(&x[i..]);
        out.copy_to_slice(&mut y[i..]);
    }
    axpy(&x[n..], &mut y[n..]);
}

#[cfg(feature = "simd")]
#[inline(never)]
fn axpy_aligned(x: &[f64], y: &mut [f64]) {
    use std::simd::Simd;
    let (xp, xv, xt) = x.as_simd::<4>();
    let (yp, yv, yt) = y.as_simd_mut::<4>();
    assert!(xp.is_empty() && yp.is_empty());
    for (x, y) in xv.iter().zip(yv) {
        *y += Simd::splat(0.125) * x;
    }
    axpy(xt, yt);
}

fn storage(c: &mut Criterion) {
    // Allocation/initialization is separate from reused numerical working sets.
    // The old unsound allocator is deliberately not a benchmark candidate.
    for n in [31, 32, 1024, 32_768, 1_048_576] {
        let x: Vec<_> = (0..n).map(|i| i as f64 / n as f64).collect();
        let ax = OwnedBuffer::from_fn(n, |i| x[i]);
        let mut y = vec![1.0; n];
        let mut ay = OwnedBuffer::from_fn(n, |_| 1.0);
        eprintln!(
            "n={n}: Vec x/y mod64={}/{}, aligned x/y mod64={}/{}",
            x.as_ptr().addr() % 64,
            y.as_ptr().addr() % 64,
            ax.as_ptr().addr() % 64,
            ay.as_ptr().addr() % 64
        );

        // Check each implementation before timing. Odd sizes exercise the tail.
        axpy(&x, &mut y);
        axpy(&ax, &mut ay);
        assert_eq!(y.as_slice(), ay.as_slice());
        #[cfg(feature = "simd")]
        {
            ay.fill(1.0);
            axpy_simd(&ax, &mut ay);
            assert_eq!(y.as_slice(), ay.as_slice());
            ay.fill(1.0);
            axpy_aligned(&ax, &mut ay);
            assert_eq!(y.as_slice(), ay.as_slice());
        }

        let mut group = c.benchmark_group("reused_axpy");
        group.throughput(Throughput::Bytes((3 * n * size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, _| {
            b.iter(|| axpy(black_box(&x), black_box(&mut y)));
        });
        group.bench_with_input(BenchmarkId::new("aligned64", n), &n, |b, _| {
            b.iter(|| axpy(black_box(&ax), black_box(&mut ay)));
        });
        #[cfg(feature = "simd")]
        {
            group.bench_with_input(BenchmarkId::new("vec_simd", n), &n, |b, _| {
                b.iter(|| axpy_simd(black_box(&x), black_box(&mut y)));
            });
            group.bench_with_input(BenchmarkId::new("aligned64_simd", n), &n, |b, _| {
                b.iter(|| axpy_simd(black_box(&ax), black_box(&mut ay)));
            });
            group.bench_with_input(BenchmarkId::new("aligned64_blocks", n), &n, |b, _| {
                b.iter(|| axpy_aligned(black_box(&ax), black_box(&mut ay)));
            });
        }
        group.finish();

        let mut group = c.benchmark_group("allocate_initialize_drop");
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(|i| i as f64).collect::<Vec<_>>()));
        });
        group.bench_with_input(BenchmarkId::new("aligned64", n), &n, |b, &n| {
            b.iter(|| black_box(OwnedBuffer::from_fn(n, |i| i as f64)));
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30)
        .warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));
    targets = storage
}
criterion_main!(benches);
