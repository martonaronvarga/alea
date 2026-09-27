//! Cheap-gradient steady-state baseline: no construction or allocation in timing.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kernels::{
    buffer::OwnedBuffer,
    dist::Gaussian,
    kernel::Kernel,
    metric::IdentityMetric,
    state::{ChainState, GradientBuffers},
    target::FusedAdapter,
};
use rand::{SeedableRng, rngs::SmallRng};
use runtime::{Hmc, HmcChain, HmcConfig, HmcOptions};
use std::{hint::black_box, time::Duration};

fn hmc(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_protocol");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for dim in [8, 64, 1024] {
        let mut legacy = Hmc::new(
            HmcConfig {
                step_size: 0.1,
                n_leapfrog: 8,
            },
            IdentityMetric::new(dim),
        );
        let mut state = ChainState::with_aux(
            OwnedBuffer::new(dim),
            GradientBuffers {
                gradient: OwnedBuffer::new(dim),
                momentum: OwnedBuffer::new(dim),
            },
        );
        let mut legacy_rng = SmallRng::seed_from_u64(912);
        group.bench_with_input(BenchmarkId::new("legacy", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(legacy.step(&mut state, &Gaussian, &mut legacy_rng));
                black_box((&state.position, &state.aux.gradient, state.log_prob));
            });
        });
        let target = FusedAdapter::new(&Gaussian, dim);
        let mut chain = HmcChain::new(
            &target,
            OwnedBuffer::new(dim),
            IdentityMetric::new(dim),
            HmcOptions::new(0.1, 8).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(912);
        group.bench_with_input(BenchmarkId::new("checked", dim), &dim, |b, _| {
            b.iter(|| {
                let _ = black_box(chain.step(&mut rng).unwrap());
                black_box((
                    chain.point().position(),
                    chain.point().gradient(),
                    chain.point().log_density(),
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, hmc);
criterion_main!(benches);
