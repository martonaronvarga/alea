use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;

use ffi::gaussian::Gaussian;
use kernels::buffer::OwnedBuffer;
use kernels::density::LogDensity;
use kernels::kernel::Kernel;
use kernels::metric::{CholeskyFactor, DenseMetric, IdentityMetric, Metric};
use kernels::state::State;
use runtime::mcmc::rwmh::{Rwmh, RwmhConfig};

const DIMS: &[usize; 6] = &[32, 64, 128, 256, 512, 1024];

fn base_config(dim: usize) -> RwmhConfig {
    let step_size = 2.38 / (dim as f64).sqrt();

    RwmhConfig::default()
        .with_warmup(0)
        .with_draws(0)
        .with_adapt_step_size(false)
        .with_step_size(step_size)
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

    CholeskyFactor::new_lower(dim, chol)
}

fn bench_rwmh_isotropic(c: &mut Criterion) {
    for &dim in DIMS {
        let name = format!("rwmh/isotropic/step/dim={}", dim);

        c.bench_function(&name, move |b| {
            b.iter_custom(|iters| {
                let density = Gaussian;
                let mut kernel = Rwmh::isotropic(base_config(dim), dim);
                let mut state = State::new(dim);
                state.position.fill(0.1);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(42 ^ dim as u64);

                let start = Instant::now();
                for _ in 0..iters {
                    black_box(kernel.step(
                        black_box(&mut state),
                        black_box(&density),
                        black_box(&mut rng),
                    ));
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
                let density = Gaussian;
                let factor = make_dense_factor(dim);
                let mut kernel = Rwmh::dense_cholesky(base_config(dim), factor);
                let mut state = State::new(dim);
                state.position.fill(0.1);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337 ^ dim as u64);

                let start = Instant::now();
                for _ in 0..iters {
                    black_box(kernel.step(
                        black_box(&mut state),
                        black_box(&density),
                        black_box(&mut rng),
                    ));
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
        #[inline(always)]
        fn log_prob(&self, _: &[f64]) -> f64 {
            0.0
        }
    }

    let density = Dummy;
    let mut kernel = Rwmh::isotropic(base_config(dim), dim);

    c.bench_function("rwmh/isotropic/step_no_density", |b| {
        b.iter_custom(|iters| {
            let mut state = State::new(dim);
            state.position.fill(0.1);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

            let start = Instant::now();
            for _ in 0..iters {
                black_box(kernel.step(
                    black_box(&mut state),
                    black_box(&density),
                    black_box(&mut rng),
                ));
            }
            start.elapsed()
        })
    });
}

fn bench_density(c: &mut Criterion) {
    for &dim in DIMS {
        let name = format!("density/gaussian/dim={}", dim);
        let density = Gaussian;
        let mut x = OwnedBuffer::new(dim);
        x.fill(0.1);

        c.bench_function(&name, move |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(density.log_prob(black_box(&x)));
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

        c.bench_function(&format!("metric/identity/apply_inverse/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    id.apply_inverse(black_box(&src), &mut dst);
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });

        c.bench_function(&format!("metric/identity/apply_sqrt/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    id.apply_sqrt(black_box(&src), &mut dst);
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });
    }
}

fn bench_metric_dense(c: &mut Criterion) {
    for &dim in DIMS {
        let factor = make_dense_factor(dim);
        let metric = DenseMetric::new(factor);

        let src = vec![0.1; dim];
        let mut dst = vec![0.0; dim];

        c.bench_function(&format!("metric/dense/apply_sqrt/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    metric.apply_sqrt(black_box(&src), &mut dst);
                    black_box(dst[0]);
                }
                start.elapsed()
            })
        });

        c.bench_function(&format!("metric/dense/apply_inverse/dim={}", dim), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    metric.apply_inverse(black_box(&src), &mut dst);
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
