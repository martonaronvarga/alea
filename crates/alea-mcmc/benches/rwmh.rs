use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::hint::black_box;

use alea_core::density::LogDensity;
use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{CholeskyFactor, DenseMetric, EuclideanMetric, IdentityMetric};
use alea_mcmc::{Rwmh, RwmhOptions};

const DIMS: &[usize; 6] = &[32, 64, 128, 256, 512, 1024];

fn base_config(dim: usize) -> RwmhOptions {
    RwmhOptions::new(2.38 / (dim as f64).sqrt()).unwrap()
}

fn make_dense_factor(dim: usize) -> CholeskyFactor {
    let mut chol = OwnedBuffer::new(dim * dim);

    for j in 0..dim {
        for i in 0..dim {
            let idx = i + j * dim;
            chol.as_mut_slice()[idx] = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                0.001 * (((i + 1) as f64) * ((j + 1) as f64)).sin()
            };
        }
    }

    CholeskyFactor::new_lower(dim, chol).expect("generated unit-diagonal lower factor is valid")
}

fn bench_rwmh_isotropic(c: &mut Criterion) {
    for &dim in DIMS {
        let name = format!("rwmh/isotropic/step/dim={}", dim);

        c.bench_function(&name, move |b| {
            b.iter_custom(|iters| {
                let density = Gaussian::new(dim);
                let mut kernel = Rwmh::new(
                    &density,
                    OwnedBuffer::from_fn(dim, |_| 0.1),
                    IdentityMetric::new(dim),
                    base_config(dim),
                )
                .unwrap();
                let mut rng = SmallRng::seed_from_u64(42 ^ dim as u64);

                let start = Instant::now();
                for _ in 0..iters {
                    let _ = black_box(kernel.step(black_box(&mut rng)).unwrap());
                }
                start.elapsed()
            })
        });
    }
}

fn bench_rwmh_dense(c: &mut Criterion) {
    for &dim in DIMS {
        let name = format!("rwmh/dense/step/dim={}", dim);
        c.bench_function(&name, move |b| {
            b.iter_custom(|iters| {
                let density = Gaussian::new(dim);
                let factor = make_dense_factor(dim);
                let mut kernel = Rwmh::new(
                    &density,
                    OwnedBuffer::from_fn(dim, |_| 0.1),
                    DenseMetric::new(factor),
                    base_config(dim),
                )
                .unwrap();
                let mut rng = SmallRng::seed_from_u64(1337 ^ dim as u64);

                let start = Instant::now();
                for _ in 0..iters {
                    let _ = black_box(kernel.step(black_box(&mut rng)).unwrap());
                }
                start.elapsed()
            })
        });
    }
}

fn bench_step_no_density(c: &mut Criterion) {
    let dim = 256;

    struct Dummy;
    impl LogDensity for Dummy {
        type Error = std::convert::Infallible;
        fn dimension(&self) -> usize {
            256
        }
        #[inline(always)]
        fn logp(&self, _: &[f64]) -> Result<f64, Self::Error> {
            Ok(0.0)
        }
    }

    let density = Dummy;

    c.bench_function("rwmh/isotropic/step_no_density", |b| {
        b.iter_custom(|iters| {
            let mut kernel = Rwmh::new(
                &density,
                OwnedBuffer::from_fn(dim, |_| 0.1),
                IdentityMetric::new(dim),
                base_config(dim),
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(42);

            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(kernel.step(black_box(&mut rng)).unwrap());
            }
            start.elapsed()
        })
    });
}

fn bench_density(c: &mut Criterion) {
    for &dim in DIMS {
        let name = format!("density/gaussian/dim={}", dim);
        let density = Gaussian::new(dim);
        let mut x = OwnedBuffer::new(dim);
        x.fill(0.1);

        c.bench_function(&name, move |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(density.logp(black_box(&x)).unwrap());
                }
                start.elapsed()
            })
        });
    }
}

fn bench_copy(c: &mut Criterion) {
    for &dim in DIMS {
        let name = format!("copy/from_slice//dim={}", dim);
        let mut src = OwnedBuffer::new(dim);
        src.fill(0.1);
        let mut dst = OwnedBuffer::new(dim);
        dst.fill(0.0);

        c.bench_function(&name, move |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    dst.copy_from_slice(black_box(&src));
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });
    }
}

fn bench_metric_identity(c: &mut Criterion) {
    for &dim in DIMS {
        let mut src = OwnedBuffer::new(dim);
        src.fill(0.1);
        let mut dst = OwnedBuffer::new(dim);
        dst.fill(0.0);

        let id = IdentityMetric::new(dim);

        c.bench_function(&format!("metric/identity/velocity/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    id.velocity(black_box(&src), &mut dst).unwrap();
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });

        c.bench_function(
            &format!("metric/identity/sample_momentum/dim={}", dim),
            |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        id.sample_momentum(black_box(&src), &mut dst).unwrap();
                        black_box(dst[0]);
                    }
                    start.elapsed()
                })
            },
        );
    }
}

fn bench_metric_dense(c: &mut Criterion) {
    for &dim in DIMS {
        let factor = make_dense_factor(dim);
        let metric = DenseMetric::new(factor);

        let src = vec![0.1; dim];
        let mut dst = vec![0.0; dim];

        c.bench_function(&format!("metric/dense/sample_momentum/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    metric.sample_momentum(black_box(&src), &mut dst).unwrap();
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });

        c.bench_function(&format!("metric/dense/velocity/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    metric.velocity(black_box(&src), &mut dst).unwrap();
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });
    }
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(80)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets =
        bench_rwmh_isotropic,
        bench_rwmh_dense,
        bench_step_no_density,
        bench_density,
        bench_copy,
        bench_metric_identity,
        bench_metric_dense,
);

criterion_main!(benches);
