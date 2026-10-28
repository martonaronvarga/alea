use super::*;
use alea_distributions::StandardNormalTarget;
use alea_math::metric::IdentityMetric;
use rand::{SeedableRng, rngs::SmallRng};
use std::convert::Infallible;

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |index| values[index])
}

#[test]
fn configuration_rejects_invalid_and_overflowing_budgets() {
    for macro_step in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(WalnutsOptions::new(macro_step, 5).is_err());
    }
    for depth in [0, usize::BITS as usize, usize::MAX] {
        assert!(WalnutsOptions::new(0.1, depth).is_err());
    }
    for attempts in [0, usize::BITS as usize, usize::MAX] {
        assert!(
            WalnutsOptions::new(0.1, 3)
                .unwrap()
                .with_local_refinement(attempts, 1, 0.5)
                .is_err()
        );
    }
    for minimum in [0, usize::MAX] {
        assert!(
            WalnutsOptions::new(0.1, 3)
                .unwrap()
                .with_local_refinement(5, minimum, 0.5)
                .is_err()
        );
    }
    for tolerance in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(
            WalnutsOptions::new(0.1, 3)
                .unwrap()
                .with_local_refinement(5, 1, tolerance)
                .is_err()
        );
    }
    for jitter in [-0.1, 1.0, f64::INFINITY, f64::NAN] {
        assert!(
            WalnutsOptions::new(0.1, 3)
                .unwrap()
                .with_macro_step_jitter(jitter)
                .is_err()
        );
    }
    assert!(
        WalnutsOptions::new(f64::MAX, 3)
            .unwrap()
            .with_macro_step_jitter(0.5)
            .is_err()
    );
}

#[test]
fn randomized_two_point_distribution_has_the_published_support_and_ratio() {
    let distribution = WalnutsMicrostepDistribution::RandomizedTwoPoint;
    let critical = distribution.log_probability(2, 2).unwrap();
    let finer = distribution.log_probability(3, 2).unwrap();
    assert!((critical - (2.0_f64 / 3.0).ln()).abs() <= 4.0 * f64::EPSILON);
    assert!((finer - (1.0_f64 / 3.0).ln()).abs() <= 4.0 * f64::EPSILON);
    assert_eq!(distribution.log_probability(1, 2), None);
    assert_eq!(distribution.log_probability(4, 2), None);

    let forward = distribution.log_probability(3, 3).unwrap();
    let reverse = distribution.log_probability(3, 2).unwrap();
    assert!((reverse - forward + 2.0_f64.ln()).abs() <= 4.0 * f64::EPSILON);
}

#[test]
fn microtrajectory_matches_independent_scalar_leapfrog_and_range() {
    let target = StandardNormalTarget::new(1);
    let metric = IdentityMetric::new(1);
    let point = PointState::new(&target, buffer(&[0.7])).unwrap();
    let start = PhaseState::new(point.clone(), buffer(&[-0.3])).unwrap();
    let mut end = PhaseState::new(point.clone(), OwnedBuffer::new(1)).unwrap();
    let mut workspace = PhaseWorkspace::new(&point);
    let mut info = WalnutsTransition::new();
    let attempt = integrate_attempt(
        &mut workspace,
        &start,
        &metric,
        0.8,
        4,
        Some(&mut end),
        &mut info,
    )
    .unwrap();

    let mut q = 0.7_f64;
    let mut p = -0.3_f64;
    let mut minimum = 0.5 * (q * q + p * p);
    let mut maximum = minimum;
    for _ in 0..4 {
        p -= 0.1 * q;
        q += 0.2 * p;
        p -= 0.1 * q;
        let energy = 0.5 * (q * q + p * p);
        minimum = minimum.min(energy);
        maximum = maximum.max(energy);
    }
    assert!((end.point().position()[0] - q).abs() < 1e-15);
    assert!((end.momentum()[0] - p).abs() < 1e-15);
    assert!((attempt.energy - 0.5 * (q * q + p * p)).abs() < 1e-15);
    assert!((attempt.range - (maximum - minimum)).abs() < 1e-15);
    assert_eq!(info.microsteps, 4);
}

struct Flat;
impl alea_core::target::TargetSpace for Flat {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        3
    }
}
impl LogDensityGradient for Flat {
    fn logp_grad(&self, _: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        gradient.fill(0.0);
        Ok(0.0)
    }
}

#[test]
fn fixed_macro_grid_has_bounded_work_on_flat_target() {
    let options = WalnutsOptions::new(0.2, 6).unwrap();
    let mut chain =
        Walnuts::new(&Flat, OwnedBuffer::new(3), IdentityMetric::new(3), options).unwrap();
    let info = chain.step(&mut SmallRng::seed_from_u64(91)).unwrap();
    assert_eq!(info.termination, WalnutsTermination::MaxDepth);
    assert_eq!(info.macro_steps, 63);
    assert_eq!(info.microsteps, 126);
    assert_eq!(info.refinement_attempts, 126);
    assert_eq!(info.max_refinement_level, 0);
    assert_eq!(info.tree_depth, 6);
    assert_eq!(info.valid_states, 64);
}

#[test]
fn local_refinement_uses_a_finer_dyadic_grid_when_required() {
    let target = StandardNormalTarget::new(1);
    let options = WalnutsOptions::new(2.0, 2)
        .unwrap()
        .with_local_refinement(8, 1, 0.05)
        .unwrap();
    let mut chain = Walnuts::new(&target, buffer(&[1.0]), IdentityMetric::new(1), options).unwrap();
    let info = chain.step(&mut SmallRng::seed_from_u64(7)).unwrap();
    assert!(info.max_refinement_level > 0);
    assert!(info.microsteps > info.macro_steps);
    assert!(info.max_local_energy_range.unwrap().is_finite());
}

#[test]
fn deterministic_reverse_probability_excludes_mismatched_refinement() {
    // Independent scalar search: this edge first passes forward at eight
    // microsteps, while the reversed endpoint first passes at four. For
    // deterministic p_micro the forward/reverse probability ratio is zero.
    let target = StandardNormalTarget::new(1);
    let options = WalnutsOptions::new(3.0, 1)
        .unwrap()
        .with_local_refinement(8, 1, 1.0)
        .unwrap();
    let mut chain =
        Walnuts::new(&target, buffer(&[-4.0]), IdentityMetric::new(1), options).unwrap();
    let point = PointState::new(&target, buffer(&[-4.0])).unwrap();
    chain.edge = PhaseState::new(point, buffer(&[-1.0])).unwrap();
    let original = chain.edge.point().position()[0];
    let mut info = WalnutsTransition::new();
    let result = chain
        .build_edge(1.0, 3.0, &mut SmallRng::seed_from_u64(1), &mut info)
        .unwrap();
    assert_eq!(result, EdgeBuild::Failed(EdgeFailure::ReverseSelection));
    assert_eq!(info.refinement_attempts, 7);
    assert_eq!(info.microsteps, 22);
    assert_eq!(chain.edge.point().position(), &[original]);
    assert_eq!(chain.edge.momentum(), &[-1.0]);
}

#[test]
fn randomized_two_point_weights_an_asymmetric_reverse_critical_level() {
    // The same independently identified edge has forward critical level three
    // and reverse critical level two. R2P gives the selected critical count a
    // reverse/forward probability ratio (1/3)/(2/3) = 1/2.
    let target = StandardNormalTarget::new(1);
    let options = WalnutsOptions::new(3.0, 1)
        .unwrap()
        .with_local_refinement(8, 1, 1.0)
        .unwrap()
        .with_microstep_distribution(WalnutsMicrostepDistribution::RandomizedTwoPoint)
        .unwrap();
    let mut observed = None;
    for seed in 0..256 {
        let mut chain =
            Walnuts::new(&target, buffer(&[-4.0]), IdentityMetric::new(1), options).unwrap();
        let point = PointState::new(&target, buffer(&[-4.0])).unwrap();
        chain.edge = PhaseState::new(point, buffer(&[-1.0])).unwrap();
        let mut info = WalnutsTransition::new();
        let result = chain
            .build_edge(1.0, 3.0, &mut SmallRng::seed_from_u64(seed), &mut info)
            .unwrap();
        if let EdgeBuild::Accepted {
            energy,
            refinement_level: 3,
            log_weight_delta,
        } = result
        {
            observed = Some((energy, log_weight_delta, chain.edge.clone()));
            break;
        }
    }
    let (energy, log_weight_delta, endpoint) = observed.expect("R2P selected the critical level");
    let expected = 8.5 - energy - 2.0_f64.ln();
    assert!((log_weight_delta - expected).abs() < 1e-14);

    // From the endpoint, the same level is the finer R2P choice above reverse
    // critical level two. Its probability ratio is reciprocal and so is the
    // complete directed edge weight.
    let mut reverse = None;
    for seed in 0..256 {
        let mut chain =
            Walnuts::new(&target, buffer(&[-4.0]), IdentityMetric::new(1), options).unwrap();
        chain.edge.clone_from(&endpoint);
        let mut info = WalnutsTransition::new();
        let result = chain
            .build_edge(-1.0, 3.0, &mut SmallRng::seed_from_u64(seed), &mut info)
            .unwrap();
        if let EdgeBuild::Accepted {
            refinement_level: 3,
            log_weight_delta,
            ..
        } = result
        {
            reverse = Some(log_weight_delta);
            break;
        }
    }
    let reverse_log_weight_delta = reverse.expect("R2P selected the finer reverse level");
    assert!((log_weight_delta + reverse_log_weight_delta).abs() < 1e-12);
}

#[test]
fn selected_reverse_level_is_replayed_at_floating_point_thresholds() {
    let target = StandardNormalTarget::new(1);
    let metric = IdentityMetric::new(1);
    let mut mismatch = None;
    'search: for q in [0.1, 0.3, 0.7, 1.1, 3.0, 10.0] {
        for p in [-3.0, -0.9, -0.2, 0.4, 1.7] {
            for macro_step in [0.01, 0.1, 0.3, 0.9, 1.7] {
                let point = PointState::new(&target, buffer(&[q])).unwrap();
                let start = PhaseState::new(point.clone(), buffer(&[p])).unwrap();
                let mut end = PhaseState::new(point.clone(), OwnedBuffer::new(1)).unwrap();
                let mut workspace = PhaseWorkspace::new(&point);
                let mut info = WalnutsTransition::new();
                let forward = integrate_attempt(
                    &mut workspace,
                    &start,
                    &metric,
                    macro_step,
                    1,
                    Some(&mut end),
                    &mut info,
                )
                .unwrap();
                let reverse = integrate_attempt(
                    &mut workspace,
                    &end,
                    &metric,
                    -macro_step,
                    1,
                    None,
                    &mut info,
                )
                .unwrap();
                let tolerance = forward.range.midpoint(reverse.range);
                if forward.range > 0.0 && forward.range < tolerance && tolerance < reverse.range {
                    mismatch = Some((q, p, macro_step, tolerance));
                    break 'search;
                }
            }
        }
    }
    let (q, p, macro_step, tolerance) = mismatch.expect("search grid has an asymmetric range");
    let options = WalnutsOptions::new(macro_step, 1)
        .unwrap()
        .with_local_refinement(1, 1, tolerance)
        .unwrap();
    let mut chain = Walnuts::new(&target, buffer(&[q]), metric, options).unwrap();
    let point = PointState::new(&target, buffer(&[q])).unwrap();
    chain.edge = PhaseState::new(point, buffer(&[p])).unwrap();
    let mut info = WalnutsTransition::new();
    let result = chain
        .build_edge(1.0, macro_step, &mut SmallRng::seed_from_u64(1), &mut info)
        .unwrap();
    assert_eq!(
        result,
        EdgeBuild::Failed(EdgeFailure::ReverseSelection),
        "q={q} p={p} h={macro_step} tolerance={tolerance:e} range={:?}",
        info.max_local_energy_range
    );
    assert_eq!(info.refinement_attempts, 2);
    assert_eq!(info.microsteps, 2);
    assert_eq!(chain.edge.point().position(), &[q]);
    assert_eq!(chain.edge.momentum(), &[p]);
}
