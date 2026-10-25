use alea::capability::{PositionMetric, evaluate_position_metric};
use alea::math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea::sampling::{
    NutsOptions,
    adapt::{
        CurvatureProbeOptions, JointNutsWarmup, JointWarmupOptions, NutsWarmup, WarmupOptions,
    },
    advanced::{AdvancedHmcOptions, ForceGradientHmc, ForceGradientMode},
    riemannian::{ExactnessPendingRmhmc, RmhmcOptions, SoftAbsMetric, SoftAbsOptions},
};
use alea_enzyme_conformance as generated;
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn generated_gradient_drives_native_nuts_warmup_and_frozen_sampling() {
    let target = generated::regression::target();
    let mut rng = SmallRng::seed_from_u64(7011);
    let (mut sampler, report) = NutsWarmup::new(
        &target,
        OwnedBuffer::new(2),
        NutsOptions::new(0.1, 7).unwrap(),
        WarmupOptions::new(150),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(report.iterations, 150);
    assert!(report.metric_updates > 0);
    let frozen = sampler.options().step_size();
    for _ in 0..128 {
        let _ = sampler.step(&mut rng).unwrap();
        assert_eq!(sampler.options().step_size(), frozen);
        assert!(sampler.point().position().iter().all(|x| x.is_finite()));
    }
}

#[test]
fn generated_hvp_drives_joint_curvature_warmup_and_force_gradient_hmc() {
    let target = generated::polynomial::target();
    let mut rng = SmallRng::seed_from_u64(7012);
    let options = JointWarmupOptions::new(
        WarmupOptions::new(150),
        CurvatureProbeOptions::new(2, 3).unwrap(),
        2.0,
    )
    .unwrap();
    let (mut sampler, report) = JointNutsWarmup::new(
        &target,
        OwnedBuffer::new(2),
        NutsOptions::new(0.1, 7).unwrap(),
        options,
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert!(report.hvp_calls > 0);
    for _ in 0..32 {
        let _ = sampler.step(&mut rng).unwrap();
    }
    let mut sampler = ForceGradientHmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        AdvancedHmcOptions::new(0.1, 4).unwrap(),
        ForceGradientMode::AnalyticHvp,
    )
    .unwrap();
    for _ in 0..32 {
        let _ = sampler.step(&mut rng).unwrap();
    }
    assert!(sampler.point().position().iter().all(|x| x.is_finite()));
}

#[test]
fn generated_nonzero_third_derivatives_drive_softabs_and_rmhmc() {
    let target = generated::polynomial::target();
    let metric = SoftAbsMetric::new(&target, SoftAbsOptions::new(1.0, 64, 1e-13).unwrap()).unwrap();
    let mut work = vec![0.0; metric.workspace_len()];
    let mut m = [0.0; 4];
    let mut d = [0.0; 8];
    evaluate_position_metric(&metric, &[0.3, -0.7], &mut m, &mut d, &mut work).unwrap();
    assert!(d.iter().any(|x| x.abs() > 0.01));
    for k in 0..2 {
        let mut plus = [0.3, -0.7];
        let mut minus = plus;
        plus[k] += 1e-5;
        minus[k] -= 1e-5;
        let mut mp = [0.0; 4];
        let mut mm = mp;
        let mut unused = [0.0; 8];
        evaluate_position_metric(&metric, &plus, &mut mp, &mut unused, &mut work).unwrap();
        evaluate_position_metric(&metric, &minus, &mut mm, &mut unused, &mut work).unwrap();
        for i in 0..4 {
            assert!((d[k * 4 + i] - (mp[i] - mm[i]) / 2e-5).abs() < 1e-7);
        }
    }
    let options = RmhmcOptions::new(0.05, 2, 50, 1e-11)
        .unwrap()
        .with_newton_solver()
        .with_reverse_check(1e-7)
        .unwrap();
    let mut sampler = ExactnessPendingRmhmc::new(
        &target,
        &metric,
        OwnedBuffer::from_fn(2, |i| if i == 0 { 0.3 } else { -0.7 }),
        options,
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(7013);
    let measured = allocation_counter::measure(|| {
        for _ in 0..32 {
            let result = sampler.step(&mut rng).unwrap();
            assert!(result.costs.metric_derivative_evaluations > 0);
            assert_eq!(result.reverse_passed, Some(true));
        }
    });
    assert_eq!(measured.count_total, 0);
}
