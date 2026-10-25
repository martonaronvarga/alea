#![cfg(not(miri))]
use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Nuts, NutsOptions, nuts::NutsTermination};
use rand::{SeedableRng, rngs::SmallRng};
use std::{convert::Infallible, hint::black_box};
struct Flat(usize);
impl LogDensityGradient for Flat {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.0
    }
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g.fill(0.0);
        Ok(0.0)
    }
}
#[test]
fn transitions_allocate_nothing_even_when_every_tree_edge_is_visited() {
    for (depth, dimension) in [1, 5, 10]
        .into_iter()
        .flat_map(|d| [(d, 7), (d, 17), (d, 64)])
    {
        for (jitter, position) in [(0.0, false), (0.1, false), (0.0, true), (0.1, true)] {
            let target = Flat(dimension);
            let mut chain = Nuts::new(
                &target,
                OwnedBuffer::new(dimension),
                IdentityMetric::new(dimension),
                NutsOptions::new(0.1, depth)
                    .unwrap()
                    .with_step_size_jitter(jitter)
                    .unwrap()
                    .with_position_turning(position),
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(19);
            let count = allocation_counter::measure(|| {
                for _ in 0..8 {
                    let info = black_box(chain.step(&mut rng).unwrap());
                    assert_eq!(info.integration_steps, (1 << depth) - 1);
                    assert_eq!(info.termination, NutsTermination::MaxDepth);
                }
            });
            assert_eq!(count.count_total, 0);
            assert_eq!(count.bytes_total, 0);
        }
    }
}
#[test]
fn construction_storage_grows_linearly_with_depth_not_tree_size() {
    let measure = |depth| {
        allocation_counter::measure(|| {
            black_box(
                Nuts::new(
                    &Flat(7),
                    OwnedBuffer::new(7),
                    IdentityMetric::new(7),
                    NutsOptions::new(0.1, depth).unwrap(),
                )
                .unwrap(),
            );
        })
    };
    let a = measure(4);
    let b = measure(8);
    let c = measure(12);
    assert!(a.bytes_total > 0);
    assert_eq!(b.bytes_total - a.bytes_total, c.bytes_total - b.bytes_total);
    assert_eq!(b.count_total - a.count_total, c.count_total - b.count_total);
}
