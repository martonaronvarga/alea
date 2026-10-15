use super::*;
use alea_distributions::Gaussian;
use alea_math::metric::{DiagonalMetric, IdentityMetric};
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn shifted_weights_keep_multiplicity_and_extreme_ratios() {
    for offset in [0.0, 1e300, -1e300] {
        let one = Weight::leaf(offset);
        let two = one.merge(one);
        assert!((two.probability(one, false) - 2.0 / 3.0).abs() < 1e-15);
        assert_eq!(one.probability(one, false), 0.5);
        assert_eq!(two.probability(one, true), 1.0);
        // ln/exp are independently perturbed under strict Miri; this is a
        // numerical identity, not a bitwise state-preservation assertion.
        assert!((one.probability(two, true) - 0.5).abs() < 2e-14);
    }
    assert_eq!(
        Weight::leaf(f64::MAX).probability(Weight::leaf(-f64::MAX), false),
        1.0
    );
    assert_eq!(
        Weight::leaf(-f64::MAX).probability(Weight::leaf(f64::MAX), false),
        0.0
    );
}

#[derive(Clone, Copy, Debug)]
struct State {
    q: [f64; 2],
    p: [f64; 2],
    index: usize,
}

#[test]
fn merged_weight_selection_matches_log_oracle_across_fast_path_boundary() {
    for gap in [
        -1500., -750., -501., -500., -499., -1., 0., 1., 499., 500., 501., 750., 1500.,
    ] {
        for a in [1., 32., 1e18] {
            for b in [1., 32., 1e18] {
                let old = Weight { max: 0., sum: a };
                let new = Weight { max: gap, sum: b };
                let log_ratio = gap + b.ln() - a.ln();
                for biased in [false, true] {
                    let expected = if biased {
                        log_ratio.min(0.).exp()
                    } else if log_ratio >= 0. {
                        1. / (1. + (-log_ratio).exp())
                    } else {
                        let r = log_ratio.exp();
                        r / (1. + r)
                    };
                    let (weight, actual) = old.merge_select(new, biased);
                    assert!(weight.sum.is_finite() && weight.sum >= 1.);
                    assert!(
                        (actual - expected).abs() <= 2e-13 * expected + f64::from_bits(4),
                        "gap={gap}, a={a}, b={b}, biased={biased}, actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }
    // Premature exp(-750) underflows, but the multiplicity rescues a nonzero
    // representable selection probability. Keep the logarithmic tail path.
    let (_, probability) = Weight::leaf(0.).merge_select(
        Weight {
            max: -750.,
            sum: 1e18,
        },
        false,
    );
    assert!(probability > 0.);
}
const MASS: [f64; 2] = [0.7, 2.3];
fn energy(s: State) -> f64 {
    0.5 * (0..2)
        .map(|i| s.q[i] * s.q[i] + s.p[i] * s.p[i] / MASS[i])
        .sum::<f64>()
}
fn advance(mut s: State, h: f64, index: usize) -> State {
    for (i, mass) in MASS.iter().enumerate() {
        s.p[i] -= 0.5 * h * s.q[i];
        s.q[i] += h * s.p[i] / mass;
        s.p[i] -= 0.5 * h * s.q[i];
    }
    s.index = index;
    s
}
#[derive(Clone)]
struct ReferenceTree {
    left: State,
    right: State,
    chosen: State,
    rho: [f64; 2],
    weight: f64,
}
fn criterion(a: State, b: State, rho: [f64; 2]) -> bool {
    [a, b]
        .iter()
        .all(|s| (0..2).map(|i| s.p[i] / MASS[i] * rho[i]).sum::<f64>() > 0.0)
}
fn join(
    a: ReferenceTree,
    b: ReferenceTree,
    forward: bool,
    biased: bool,
    rng: &mut SmallRng,
) -> (ReferenceTree, bool) {
    let chosen = if rng.random::<f64>()
        < b.weight
            / if biased {
                a.weight
            } else {
                a.weight + b.weight
            } {
        b.chosen
    } else {
        a.chosen
    };
    let (left, right) = if forward { (&a, &b) } else { (&b, &a) };
    let rho = std::array::from_fn(|i| a.rho[i] + b.rho[i]);
    let valid = criterion(left.left, right.right, rho)
        && criterion(
            left.left,
            right.left,
            std::array::from_fn(|i| left.rho[i] + right.left.p[i]),
        )
        && criterion(
            left.right,
            right.right,
            std::array::from_fn(|i| right.rho[i] + left.right.p[i]),
        );
    (
        ReferenceTree {
            left: left.left,
            right: right.right,
            chosen,
            rho,
            weight: a.weight + b.weight,
        },
        valid,
    )
}
struct Reference {
    initial: f64,
    steps: usize,
    sum_acceptance: f64,
    termination: NutsTermination,
    max_delta: f64,
}
fn build(
    state: &mut State,
    depth: usize,
    h: f64,
    r: &mut Reference,
    rng: &mut SmallRng,
) -> Option<ReferenceTree> {
    if depth == 0 {
        r.steps += 1;
        *state = advance(*state, h, r.steps);
        let delta = energy(*state) - r.initial;
        r.max_delta = r.max_delta.max(delta);
        r.sum_acceptance += (-delta).exp().min(1.0);
        if delta > 10.0 {
            r.termination = NutsTermination::Divergence(Divergence::EnergyErrorLimit);
            return None;
        }
        return Some(ReferenceTree {
            left: *state,
            right: *state,
            chosen: *state,
            rho: state.p,
            weight: (-delta).exp(),
        });
    }
    let a = build(state, depth - 1, h, r, rng)?;
    let b = build(state, depth - 1, h, r, rng)?;
    let (tree, valid) = join(a, b, h > 0.0, false, rng);
    if valid {
        Some(tree)
    } else {
        r.termination = NutsTermination::UTurn;
        None
    }
}

#[test]
fn iterative_carries_match_independent_recursive_scalar_oracle() {
    let target = Gaussian::new(2);
    for eps in [0.01, 0.3, 1.4, 3.0] {
        for depth in 1..=6 {
            for seed in 0..if cfg!(miri) { 2 } else { 32 } {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mut replay = rng.clone();
                let start = State {
                    q: [0.7, -0.2],
                    p: std::array::from_fn(|i| {
                        MASS[i].sqrt() * alea_math::random::standard_normal(&mut replay)
                    }),
                    index: 0,
                };
                let mut root = ReferenceTree {
                    left: start,
                    right: start,
                    chosen: start,
                    rho: start.p,
                    weight: 1.0,
                };
                let mut r = Reference {
                    initial: energy(start),
                    steps: 0,
                    sum_acceptance: 0.0,
                    termination: NutsTermination::MaxDepth,
                    max_delta: 0.0,
                };
                let mut completed = 0;
                for d in 0..depth {
                    let forward = replay.random::<bool>();
                    let mut edge = if forward { root.right } else { root.left };
                    let Some(subtree) = build(
                        &mut edge,
                        d,
                        if forward { eps } else { -eps },
                        &mut r,
                        &mut replay,
                    ) else {
                        break;
                    };
                    let (next, valid) = join(root, subtree, forward, true, &mut replay);
                    root = next;
                    completed += 1;
                    if !valid {
                        r.termination = NutsTermination::UTurn;
                        break;
                    }
                }
                let mut chain = Nuts::new(
                    &target,
                    OwnedBuffer::from_fn(2, |i| start.q[i]),
                    DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| MASS[i])).unwrap(),
                    NutsOptions::new(eps, depth)
                        .unwrap()
                        .with_max_energy_error(10.0)
                        .unwrap(),
                )
                .unwrap();
                let info = chain.step(&mut rng).unwrap();
                assert_eq!(
                    info.integration_steps, r.steps,
                    "eps={eps} depth={depth} seed={seed}"
                );
                assert_eq!(info.tree_depth, completed);
                assert_eq!(info.selected_index, root.chosen.index);
                assert_eq!(info.termination, r.termination);
                assert_eq!(info.valid_states, 1_usize << completed);
                for (actual, expected) in chain.point.position().iter().zip(root.chosen.q) {
                    assert!((actual - expected).abs() < 1e-11);
                }
                assert!(
                    (info.acceptance_probability - r.sum_acceptance / r.steps as f64).abs() < 1e-11
                );
                assert_eq!(rng.random::<u64>(), replay.random::<u64>());
            }
        }
    }
}

#[test]
fn configuration_rejects_invalid_boundaries() {
    for step in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(NutsOptions::new(step, 5).is_err());
    }
    for depth in [0, usize::BITS as usize, usize::MAX] {
        assert!(NutsOptions::new(0.1, depth).is_err());
    }
    for limit in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(
            NutsOptions::new(0.1, 5)
                .unwrap()
                .with_max_energy_error(limit)
                .is_err()
        );
    }
    assert!(
        Nuts::new(
            &Gaussian::new(0),
            OwnedBuffer::new(0),
            IdentityMetric::new(0),
            NutsOptions::new(0.1, 3).unwrap()
        )
        .is_err()
    );
}

#[test]
fn turning_uses_velocity_and_scaled_dots_not_raw_momentum() {
    let target = Gaussian::new(2);
    let point = PointState::new(&target, OwnedBuffer::new(2)).unwrap();
    let state = PhaseState::new(point.clone(), OwnedBuffer::from_fn(2, |_| 1.0)).unwrap();
    let mut scratch = TurnScratch::new(2);
    assert!(
        scratch
            .continues(
                &state,
                &state,
                &[-1.0, 2.0],
                &[0.0, 0.0],
                &IdentityMetric::new(2)
            )
            .unwrap()
    );
    let metric = DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| [0.1, 10.0][i])).unwrap();
    assert!(
        !scratch
            .continues(&state, &state, &[-1.0, 2.0], &[0.0, 0.0], &metric)
            .unwrap()
    );
    let huge = PhaseState::new(point, OwnedBuffer::from_fn(2, |_| f64::MAX)).unwrap();
    assert!(
        !scratch
            .continues(
                &huge,
                &huge,
                &[f64::MAX, -f64::MAX],
                &[0.0, 0.0],
                &IdentityMetric::new(2)
            )
            .unwrap()
    );
    assert!(
        scratch
            .continues(
                &huge,
                &huge,
                &[f64::MAX, f64::MAX],
                &[0.0, 0.0],
                &IdentityMetric::new(2)
            )
            .unwrap()
    );
    assert!(matches!(
        scratch.continues(
            &huge,
            &huge,
            &[f64::MAX, 0.0],
            &[f64::MAX, 0.0],
            &IdentityMetric::new(2)
        ),
        Err(PhaseError::Divergence(Divergence::Energy))
    ));
}

#[test]
fn failed_initial_momentum_never_commits_a_previous_tree() {
    use alea_math::metric::MetricError;
    use std::cell::Cell;
    struct Metric<'a>(&'a Cell<bool>);
    impl EuclideanMetric for Metric<'_> {
        fn dimension(&self) -> usize {
            2
        }
        fn velocity(&self, p: &[f64], v: &mut [f64]) -> Result<(), MetricError> {
            v.copy_from_slice(p);
            Ok(())
        }
        fn sample_momentum(&self, z: &[f64], p: &mut [f64]) -> Result<(), MetricError> {
            p.copy_from_slice(z);
            if self.0.get() {
                p[0] = f64::NAN;
            }
            Ok(())
        }
        fn log_det(&self) -> f64 {
            0.0
        }
    }
    let target = Gaussian::new(2);
    let fail = Cell::new(false);
    let mut chain = Nuts::new(
        &target,
        OwnedBuffer::new(2),
        Metric(&fail),
        NutsOptions::new(0.2, 4).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let _ = chain.step(&mut rng).unwrap();
    let before = chain.point.position().to_vec();
    fail.set(true);
    let info = chain.step(&mut rng).unwrap();
    assert_eq!(info.valid_states, 0);
    assert_eq!(info.integration_steps, 0);
    assert!(info.initial_energy.is_none());
    assert!(info.proposal_energy.is_none());
    assert!(!info.accepted);
    assert_eq!(chain.point.position(), before);
    fail.set(false);
    assert!(chain.step(&mut rng).unwrap().initial_energy.is_some());
}
