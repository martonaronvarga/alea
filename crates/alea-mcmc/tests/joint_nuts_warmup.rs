use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    NutsOptions,
    adapt::{CurvatureProbeOptions, JointNutsWarmup, JointWarmupOptions, WarmupOptions},
};
use rand::{SeedableRng, rngs::SmallRng};
#[test]
fn coefficients_freeze_before_final_fast_and_retained_steps_keep_geometry() {
    let target = Gaussian::new(3);
    let mut rng = SmallRng::seed_from_u64(781);
    let options = JointWarmupOptions::new(
        WarmupOptions::new(150),
        CurvatureProbeOptions::new(2, 3).unwrap(),
        2.,
    )
    .unwrap();
    let (mut s, r) = JointNutsWarmup::new(
        &target,
        OwnedBuffer::from_fn(3, |_| 2.),
        NutsOptions::new(10., 7).unwrap(),
        options,
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(r.iterations, 150);
    assert_eq!(r.coefficient_updates, r.metric_updates + 1);
    assert!(r.final_fast_iterations >= 10);
    assert!(r.selection_dimensionless_limit <= 3.8);
    assert!(r.step_size.value() * r.curvature_upper_bound.sqrt() <= 3.8 + 1e-14);
    let step = s.options().step_size();
    let mass = s.metric().clone();
    for _ in 0..50 {
        let _ = s.step(&mut rng).unwrap();
        assert_eq!(s.options().step_size(), step);
        assert_eq!(s.metric(), &mass);
    }
}
