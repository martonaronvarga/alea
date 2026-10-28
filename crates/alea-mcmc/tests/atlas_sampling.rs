use alea_distributions::StandardNormalTarget;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Hmc, HmcOptions,
    atlas_sampler::{Atlas, AtlasOptions, AtlasStage},
};
use rand::{SeedableRng, rngs::SmallRng};
#[test]
fn full_atlas_exercises_all_branches_with_frozen_geometry_and_bounded_work() {
    let t = StandardNormalTarget::new(3);
    let mut observed = [0_usize; 3];
    for baseline in [0.8, 1.5, 4.] {
        let mut rng = SmallRng::seed_from_u64(18001);
        let h = Hmc::new(
            &t,
            OwnedBuffer::from_fn(3, |_| 0.3),
            IdentityMetric::new(3),
            HmcOptions::new(baseline, 8).unwrap(),
        )
        .unwrap();
        let mut s = Atlas::from_hmc(
            h,
            AtlasOptions::new(baseline, 128).unwrap(),
            &[4, 6, 8, 10, 12],
        )
        .unwrap();
        let mut sums = [0.; 3];
        let mut squares = [0.; 3];
        for index in 0..7000 {
            let o = s.step(&mut rng).unwrap();
            assert!(o.integration_steps <= 5 * 128);
            assert_eq!(
                o.integration_steps,
                o.forward_integration_steps
                    + o.reverse_integration_steps
                    + o.ghost_integration_steps
            );
            assert!(o.hvp_calls <= 28);
            observed[match o.stage {
                AtlasStage::Preliminary => 0,
                AtlasStage::Delayed => 1,
                AtlasStage::Failure => 2,
            }] += 1;
            assert_eq!(s.options().baseline_step().value(), baseline);
            assert_eq!(s.trajectory_lengths(), &[4, 6, 8, 10, 12]);
            if index >= 1000 {
                for i in 0..3 {
                    let q = s.point().position()[i];
                    sums[i] += q;
                    squares[i] += q * q;
                }
            }
        }
        for i in 0..3 {
            assert!(
                (sums[i] / 6000.).abs() < 0.10,
                "baseline={baseline}, mean={}",
                sums[i] / 6000.
            );
            assert!(
                (squares[i] / 6000. - 1.).abs() < 0.16,
                "baseline={baseline}, second moment={}",
                squares[i] / 6000.
            );
        }
        let measured = allocation_counter::measure(|| {
            for _ in 0..20 {
                let _ = s.step(&mut rng).unwrap();
            }
        });
        assert_eq!(measured.count_total, 0);
    }
    assert!(observed.iter().all(|n| *n > 0), "{observed:?}");
}
