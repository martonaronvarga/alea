//! Fixed-budget non-Gaussian retained-draw gates, not an efficiency benchmark.
#![cfg(not(miri))]

#[path = "support/targets.rs"]
mod targets;

use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use alea_mcmc::{
    Hmc, HmcOptions,
    adapt::{FisherHmcWarmup, FisherOptions, HmcWarmup, HmcWarmupReport, WarmupOptions},
    config::AcceptanceTarget,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible};
use targets::Target;

const WARMUP: usize = 1000;
const DRAWS: usize = 8192;
const STEPS: usize = 5;
const STARTS: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]];

#[derive(Clone, Copy, Debug)]
enum SamplingTarget {
    Reference(Target),
    // q = (z0, 3*z0 + z1) for z distributed as the existing banana.
    // The shear has determinant one; no log-Jacobian term is omitted.
    ShearedBanana,
}
impl SamplingTarget {
    fn observables(self, q: &[f64]) -> [f64; 5] {
        match self {
            Self::Reference(target) => target.observables(q),
            Self::ShearedBanana => Target::Banana.observables(&[q[0], q[1] - 3.0 * q[0]]),
        }
    }
}
impl LogDensityGradient for SamplingTarget {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        match self {
            Self::Reference(target) => target.logp_grad(q, g),
            Self::ShearedBanana => {
                let logp = Target::Banana.logp_grad(&[q[0], q[1] - 3.0 * q[0]], g)?;
                // Pull back the score by the transpose of the inverse shear.
                g[0] -= 3.0 * g[1];
                Ok(logp)
            }
        }
    }
}

#[test]
fn sheared_banana_score_and_observables_match_independent_generative_formula() {
    // Independent scalar density, not a call through the wrapper/base target.
    let density = |q: [f64; 2]| {
        let residual = q[1] - 3.0 * q[0] - 0.4 * (q[0].powi(2) - 1.0);
        -0.5 * (q[0].powi(2) + residual.powi(2))
    };
    for x in [-2.0_f64, -0.5, 0.0, 1.5, 3.0] {
        for y in [-2.0_f64, 0.25, 1.0] {
            let q = [x, 3.0 * x + y + 0.4 * (x.powi(2) - 1.0)];
            let mut score = [0.0; 2];
            let logp = SamplingTarget::ShearedBanana
                .logp_grad(&q, &mut score)
                .unwrap();
            assert!((logp + 0.5 * (x * x + y * y)).abs() < 1e-12);
            let actual = SamplingTarget::ShearedBanana.observables(&q);
            for (value, expected) in actual.into_iter().zip([x, y, x * x, y * y, x * y]) {
                assert!((value - expected).abs() < 1e-12);
            }
            for axis in 0..2 {
                let h = 1e-5;
                let mut plus = q;
                let mut minus = q;
                plus[axis] += h;
                minus[axis] -= h;
                let numerical = (density(plus) - density(minus)) / (2.0 * h);
                assert!((score[axis] - numerical).abs() < 1e-8 * (1.0 + score[axis].abs()));
            }
        }
    }
}

struct Counted {
    target: SamplingTarget,
    calls: Cell<usize>,
}
impl LogDensityGradient for Counted {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.target.dimension()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        self.target.logp_grad(q, g)
    }
}

#[derive(Clone, Copy, Debug)]
enum Adaptation {
    Covariance,
    FisherDiagonal,
    #[cfg(feature = "faer")]
    FisherLowRank,
}

// Flatten only complete within-chain batches. Never construct a batch across
// chain boundaries; equal chain lengths give each chain the same weight.
fn moments(chains: &[Vec<[f64; 5]>], batches: &[usize]) -> ([f64; 5], [f64; 5]) {
    assert!(chains.len() > 1 && !batches.is_empty());
    let length = chains[0].len();
    assert!(length > 0 && chains.iter().all(|chain| chain.len() == length));
    let mean = std::array::from_fn(|i| {
        chains.iter().flatten().map(|x| x[i]).sum::<f64>() / (chains.len() * length) as f64
    });
    let mut mcse = [0.0_f64; 5];
    for &batch in batches {
        assert!(batch > 0 && length.is_multiple_of(batch));
        let count = chains.len() * (length / batch);
        for i in 0..5 {
            let squared_deviations = chains
                .iter()
                .flat_map(|chain| chain.chunks_exact(batch))
                .map(|xs| (xs.iter().map(|x| x[i]).sum::<f64>() / batch as f64 - mean[i]).powi(2))
                .sum::<f64>();
            mcse[i] = mcse[i].max((squared_deviations / ((count - 1) * count) as f64).sqrt());
        }
    }
    (mean, mcse)
}

#[test]
fn batch_means_keep_chain_boundaries_and_take_larger_uncertainty() {
    let chains = [vec![[0.0; 5]; 4], vec![[2.0; 5]; 4]];
    let (mean, mcse) = moments(&chains, &[2, 4]);
    assert_eq!(mean, [1.0; 5]);
    // Two chain-sized batch means 0 and 2: variance=2, SE=sqrt(2/2).
    assert_eq!(mcse, [1.0; 5]);
    let (_, small_batch_mcse) = moments(&chains, &[2]);
    assert!((small_batch_mcse[0] - (1.0_f64 / 3.0).sqrt()).abs() < 1e-15);
}

#[test]
#[should_panic]
fn batch_means_reject_incomplete_batches_instead_of_dropping_draws() {
    moments(&[vec![[0.0; 5]; 3], vec![[1.0; 5]; 3]], &[2]);
}

fn metric_signature(metric: &impl EuclideanMetric) -> [u64; 5] {
    assert_eq!(metric.dimension(), 2);
    let mut first = [0.0; 2];
    let mut second = [0.0; 2];
    metric.velocity(&[1.0, 0.0], &mut first).unwrap();
    metric.velocity(&[0.0, 1.0], &mut second).unwrap();
    [first[0], first[1], second[0], second[1], metric.log_det()].map(f64::to_bits)
}

fn snapshot<M: EuclideanMetric>(chain: &Hmc<'_, Counted, M>) -> [u64; 5] {
    let point = chain.point();
    [
        point.position()[0],
        point.position()[1],
        point.gradient()[0],
        point.gradient()[1],
        point.log_density(),
    ]
    .map(f64::to_bits)
}

fn collect<M: EuclideanMetric>(
    mut chain: Hmc<'_, Counted, M>,
    report: HmcWarmupReport,
    rng: &mut SmallRng,
    adaptation: Adaptation,
    seed: u64,
    rank: Option<usize>,
) -> (Vec<[f64; 5]>, usize) {
    let target = chain.point().target();
    assert_eq!(report.iterations, WARMUP);
    assert!(report.metric_updates > 0);
    let warmup_calls = target.calls.get(); // Includes construction and step-size search.
    let frozen_metric = metric_signature(chain.metric());
    let frozen_step = chain.options().step_size();
    assert_eq!(frozen_step, report.step_size);
    let mut samples = Vec::with_capacity(DRAWS);
    let (mut divergences, mut accepted, mut attempts) = (0, 0, 0);
    for _ in 0..DRAWS {
        let before = snapshot(&chain);
        let calls_before = target.calls.get();
        let transition = chain.step(rng).unwrap();
        let calls = target.calls.get() - calls_before;
        assert!(transition.integration_steps <= STEPS);
        assert!(calls <= transition.integration_steps);
        if transition.divergence.is_none() {
            assert_eq!(transition.integration_steps, STEPS);
            assert_eq!(calls, STEPS); // One new fused evaluation per leapfrog step.
        } else {
            divergences += 1;
            assert!(!transition.accepted);
            assert_eq!(transition.acceptance_probability, 0.0);
        }
        if transition.accepted {
            accepted += 1;
        } else {
            assert_eq!(snapshot(&chain), before);
        }
        attempts += transition.integration_steps;
        let point = chain.point();
        let mut gradient = [0.0; 2];
        // Bypass the counter: validation evaluations are not sampler work.
        let logp = target
            .target
            .logp_grad(point.position(), &mut gradient)
            .unwrap();
        assert!(logp.is_finite() && gradient.iter().all(|g| g.is_finite()));
        assert_eq!(logp.to_bits(), point.log_density().to_bits());
        assert_eq!(
            gradient.map(f64::to_bits),
            [point.gradient()[0], point.gradient()[1]].map(f64::to_bits)
        );
        samples.push(target.target.observables(point.position()));
    }
    assert_eq!(metric_signature(chain.metric()), frozen_metric);
    assert_eq!(chain.options().step_size(), frozen_step);
    assert_eq!(samples.len(), DRAWS);
    eprintln!(
        "{:?},{adaptation:?},seed={seed},rank={rank:?},epsilon={},warmup_calls={warmup_calls},warmup_divergences={},sample_calls={},sample_attempts={attempts},accepted={accepted},divergences={divergences}",
        target.target,
        frozen_step.value(),
        report.divergences,
        target.calls.get() - warmup_calls,
    );
    (samples, divergences)
}

fn reference(target_name: &str) -> [(f64, f64); 5] {
    let mut result = [None; 5];
    for line in include_str!("fixtures/blackjax-sampling.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
    {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 14);
        if fields[0] != target_name || fields[1] != "identity" {
            continue;
        }
        assert_eq!(&fields[7..10], &["4", "1024", "8192"]);
        let observable: usize = fields[10].parse().unwrap();
        if observable < 5 {
            assert!(result[observable].is_none());
            result[observable] = Some((fields[11].parse().unwrap(), fields[12].parse().unwrap()));
        }
    }
    result.map(|entry| entry.expect("one pinned identity reference per observable"))
}

fn validate(target: SamplingTarget, name: &str) {
    validate_with_target(target, name, 0.8.try_into().unwrap());
}

fn validate_with_target(
    target: SamplingTarget,
    name: &str,
    acceptance: AcceptanceTarget,
) -> Vec<usize> {
    let reference = reference(name);
    let mut divergence_counts = Vec::new();
    for adaptation in [
        Adaptation::Covariance,
        Adaptation::FisherDiagonal,
        #[cfg(feature = "faer")]
        Adaptation::FisherLowRank,
    ] {
        eprintln!(
            "{target:?},{adaptation:?},target_acceptance={}",
            acceptance.value()
        );
        let mut chains = Vec::new();
        let mut divergences = 0;
        for (index, start) in STARTS.into_iter().enumerate() {
            let seed = 1001 + index as u64;
            let counted = Counted {
                target,
                calls: Cell::new(0),
            };
            let mut rng = SmallRng::seed_from_u64(seed);
            let position = OwnedBuffer::from_fn(2, |i| start[i]);
            let options = HmcOptions::new(0.1, STEPS).unwrap();
            let (samples, count) = match adaptation {
                Adaptation::Covariance => {
                    let (chain, report) = HmcWarmup::new(
                        &counted,
                        position,
                        options,
                        WarmupOptions::new(WARMUP).with_target(acceptance),
                    )
                    .unwrap()
                    .run(&mut rng)
                    .unwrap();
                    collect(chain, report, &mut rng, adaptation, seed, None)
                }
                _ => {
                    let fisher = FisherOptions::new(WARMUP).with_target(acceptance);
                    #[cfg(feature = "faer")]
                    let fisher = if matches!(adaptation, Adaptation::FisherLowRank) {
                        fisher.with_max_rank(2)
                    } else {
                        fisher
                    };
                    let (chain, report) = FisherHmcWarmup::new(&counted, position, options, fisher)
                        .unwrap()
                        .run(&mut rng)
                        .unwrap();
                    let rank = chain.metric().rank();
                    #[cfg(feature = "faer")]
                    if matches!(target, SamplingTarget::ShearedBanana)
                        && matches!(adaptation, Adaptation::FisherLowRank)
                    {
                        assert!(
                            rank > 0,
                            "seed {seed}: nonzero frozen-rank coverage required"
                        );
                    }
                    collect(chain, report, &mut rng, adaptation, seed, Some(rank))
                }
            };
            chains.push(samples);
            divergences += count;
        }
        let (mean, mcse) = moments(&chains, &[256, 512]);
        eprintln!("{target:?},{adaptation:?},mean={mean:?},mcse={mcse:?}");
        for i in 0..5 {
            let context = format!(
                "{target:?}/{adaptation:?} observable {i}: {} +/- {}",
                mean[i], mcse[i]
            );
            assert!(mcse[i] > 0.0 && mcse[i] < 0.12, "{context}");
            assert!(
                (mean[i] - reference[i].0).abs() < 6.0 * mcse[i].hypot(reference[i].1) + 0.01,
                "BlackJAX mismatch: {context}"
            );
            if !matches!(target, SamplingTarget::Reference(Target::Logistic)) {
                let expected = [0.0, 0.0, 1.0, 1.0, 0.0][i];
                assert!(
                    (mean[i] - expected).abs() < 6.0 * mcse[i] + 0.01,
                    "analytic mismatch: {context}"
                );
            }
        }
        divergence_counts.push(divergences);
    }
    divergence_counts
}

#[test]
fn conservative_acceptance_resolves_banana_divergences_without_changing_moment_gates() {
    for (target, name) in [
        (SamplingTarget::Reference(Target::Banana), "banana"),
        (SamplingTarget::ShearedBanana, "banana"),
    ] {
        let divergences = validate_with_target(target, name, 0.95.try_into().unwrap());
        eprintln!("conservative {target:?}, retained divergences={divergences:?}");
        assert!(
            divergences.iter().all(|&count| count == 0),
            "{target:?}: {divergences:?}"
        );
    }
}

#[test]
fn conservative_funnel_acceptance_checks_moments_but_is_not_divergence_free() {
    // The initial zero-divergence experiment failed here: at target .95 the
    // stable/Faer run retained 2 diagonal and 3 rank-capped Fisher divergences.
    // Keep this target and unchanged moment/cache gates in CI, but do not turn
    // a model-dependent tuning control into a false divergence-free contract.
    let counts = validate_with_target(
        SamplingTarget::Reference(Target::Funnel),
        "funnel",
        0.95.try_into().unwrap(),
    );
    eprintln!("conservative funnel remaining divergences={counts:?}");
}

#[test]
fn banana_retained_moments_match_analytic_and_blackjax_references() {
    validate(SamplingTarget::Reference(Target::Banana), "banana");
}

#[test]
fn logistic_retained_moments_match_blackjax_reference() {
    validate(SamplingTarget::Reference(Target::Logistic), "logistic");
}

#[test]
fn width_one_funnel_retained_moments_match_analytic_and_blackjax_references() {
    validate(SamplingTarget::Reference(Target::Funnel), "funnel");
}

#[test]
fn sheared_banana_retained_moments_match_references_with_nonzero_fisher_rank() {
    // Inverting the unit-determinant shear gives the original banana law, so
    // its transformed observables can use the existing independent reference.
    validate(SamplingTarget::ShearedBanana, "banana");
}

#[cfg(feature = "faer")]
#[test]
fn divergent_fisher_proposals_recover_with_fixed_time_step_refinement() {
    use alea_math::metric::LowRankDiagonalMetric;
    use alea_mcmc::hmc::Divergence;

    let target = SamplingTarget::ShearedBanana;
    let mut failures = 0;
    for (index, start) in STARTS.into_iter().enumerate() {
        let seed = 1001 + index as u64;
        let mut rng = SmallRng::seed_from_u64(seed);
        let (mut chain, _) = FisherHmcWarmup::new(
            &target,
            OwnedBuffer::from_fn(2, |i| start[i]),
            HmcOptions::new(0.1, STEPS).unwrap(),
            FisherOptions::new(WARMUP).with_max_rank(2),
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        for draw in 0..DRAWS {
            let before: [f64; 2] = chain.point().position().try_into().unwrap();
            let initial_rng = rng.clone();
            let transition = chain.step(&mut rng).unwrap();
            if transition.divergence.is_none() {
                continue;
            }
            failures += 1;
            assert_eq!(transition.divergence, Some(Divergence::EnergyErrorLimit));
            let mut errors = Vec::new();
            for refinement in [2, 4] {
                let metric = chain.metric();
                let copy = |xs: &[f64]| OwnedBuffer::from_fn(xs.len(), |i| xs[i]);
                let metric = LowRankDiagonalMetric::new(
                    copy(metric.scales()),
                    copy(metric.basis()),
                    copy(metric.eigenvalues()),
                )
                .unwrap();
                let mut replay = Hmc::new(
                    &target,
                    copy(&before),
                    metric,
                    HmcOptions::new(
                        chain.options().step_size().value() / refinement as f64,
                        STEPS * refinement,
                    )
                    .unwrap(),
                )
                .unwrap();
                // Same momentum draw and same physical integration time. This
                // diagnostic never modifies the original chain or its RNG.
                let refined = replay.step(&mut initial_rng.clone()).unwrap();
                assert_eq!(refined.initial_energy, transition.initial_energy);
                assert!(
                    refined.divergence.is_none(),
                    "seed={seed}, draw={draw}, refinement={refinement}: {refined:?}"
                );
                assert_eq!(refined.integration_steps, STEPS * refinement);
                errors.push(refined.energy_error.unwrap());
            }
            eprintln!(
                "refinement seed={seed},draw={draw},original={:?},half_quarter={errors:?}",
                transition.energy_error
            );
        }
    }
    assert!(
        failures > 0,
        "must exercise the known default-policy instability"
    );
    eprintln!("refined {failures} divergent proposals without changing live sampling");
}

// Hessian of U=-logp for the width-one funnel, in the original coordinates.
// This starting-point diagnostic is not a trajectory-wide stability bound.
fn funnel_curvature(q: [f64; 2]) -> [[f64; 2]; 2] {
    let precision = (-q[0]).exp();
    [
        [1.0 + 0.5 * q[1].powi(2) * precision, -q[1] * precision],
        [-q[1] * precision, precision],
    ]
}

#[test]
fn funnel_potential_curvature_matches_score_differences() {
    for x in [-8.0_f64, -3.0, 0.0, 3.0] {
        for standardized_y in [-2.0, 0.0, 1.5] {
            let q = [x, standardized_y * (0.5 * x).exp()];
            let hessian = funnel_curvature(q);
            for axis in 0..2 {
                let mut plus = q;
                let mut minus = q;
                let h = 1e-5;
                plus[axis] += h;
                minus[axis] -= h;
                let mut plus_score = [0.0; 2];
                let mut minus_score = [0.0; 2];
                Target::Funnel.logp_grad(&plus, &mut plus_score).unwrap();
                Target::Funnel.logp_grad(&minus, &mut minus_score).unwrap();
                for i in 0..2 {
                    let numerical = -(plus_score[i] - minus_score[i]) / (2.0 * h);
                    assert!(
                        (numerical - hessian[i][axis]).abs()
                            < 1e-8 * (1.0 + hessian[i][axis].abs())
                    );
                }
            }
        }
    }
}

#[cfg(feature = "faer")]
#[test]
fn funnel_divergence_replay_preserves_live_chains_and_reports_fixed_time_refinement() {
    use alea_math::metric::LowRankDiagonalMetric;
    use rand::Rng;

    // Declared before execution: original ensemble plus a disjoint seed ensemble,
    // both acceptance targets, both Fisher geometries, and no adaptive retries.
    for acceptance in [0.8, 0.95] {
        for cap in [0, 2] {
            let mut failures = 0;
            let mut remaining = [0_usize; 4];
            for seed_base in [1001, 2001] {
                for (index, start) in STARTS.into_iter().enumerate() {
                    let seed = seed_base + index as u64;
                    let target = Counted {
                        target: SamplingTarget::Reference(Target::Funnel),
                        calls: Cell::new(0),
                    };
                    let mut rng = SmallRng::seed_from_u64(seed);
                    let (mut chain, _) = FisherHmcWarmup::new(
                        &target,
                        OwnedBuffer::from_fn(2, |i| start[i]),
                        HmcOptions::new(0.1, STEPS).unwrap(),
                        FisherOptions::new(WARMUP)
                            .with_target(acceptance.try_into().unwrap())
                            .with_max_rank(cap),
                    )
                    .unwrap()
                    .run(&mut rng)
                    .unwrap();
                    for draw in 0..DRAWS {
                        let before = snapshot(&chain);
                        let position = [f64::from_bits(before[0]), f64::from_bits(before[1])];
                        let initial_rng = rng.clone();
                        let transition = chain.step(&mut rng).unwrap();
                        if transition.divergence.is_none() {
                            continue;
                        }
                        failures += 1;
                        assert!(!transition.accepted);
                        assert_eq!(snapshot(&chain), before);
                        let live_rng = rng.clone().next_u64();
                        let live_calls = target.calls.get();
                        let frozen_metric = metric_signature(chain.metric());
                        let hessian = funnel_curvature([position[0], position[1]]);
                        let curvature_max = 0.5
                            * (hessian[0][0]
                                + hessian[1][1]
                                + (hessian[0][0] - hessian[1][1]).hypot(2.0 * hessian[0][1]));
                        for (slot, refinement) in [1, 2, 4, 8].into_iter().enumerate() {
                            let metric = chain.metric();
                            let copy = |xs: &[f64]| OwnedBuffer::from_fn(xs.len(), |i| xs[i]);
                            let copied_metric = LowRankDiagonalMetric::new(
                                copy(metric.scales()),
                                copy(metric.basis()),
                                copy(metric.eigenvalues()),
                            )
                            .unwrap();
                            let replay_target = Counted {
                                target: target.target,
                                calls: Cell::new(0),
                            };
                            let mut replay = Hmc::new(
                                &replay_target,
                                copy(&position),
                                copied_metric,
                                HmcOptions::new(
                                    chain.options().step_size().value() / refinement as f64,
                                    STEPS * refinement,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                            let refined = replay.step(&mut initial_rng.clone()).unwrap();
                            assert_eq!(refined.initial_energy, transition.initial_energy);
                            assert!(replay_target.calls.get() <= 1 + STEPS * refinement);
                            if refinement == 1 {
                                assert_eq!(refined.divergence, transition.divergence);
                                assert_eq!(
                                    refined.energy_error.map(f64::to_bits),
                                    transition.energy_error.map(f64::to_bits)
                                );
                                assert_eq!(refined.integration_steps, transition.integration_steps);
                            }
                            if refined.divergence.is_some() {
                                remaining[slot] += 1;
                                assert!(!refined.accepted);
                                assert_eq!(snapshot(&replay), before);
                            } else {
                                assert_eq!(refined.integration_steps, STEPS * refinement);
                                assert_eq!(replay_target.calls.get(), 1 + STEPS * refinement);
                            }
                            eprintln!(
                                "funnel_replay,target={acceptance},cap={cap},seed={seed},draw={draw},x={},y={},start_curvature_max={curvature_max},step={},rank={},refinement={refinement},calls={},divergence={:?},energy_error={:?}",
                                position[0],
                                position[1],
                                chain.options().step_size().value(),
                                chain.metric().rank(),
                                replay_target.calls.get(),
                                refined.divergence,
                                refined.energy_error
                            );
                        }
                        assert_eq!(snapshot(&chain), before);
                        assert_eq!(metric_signature(chain.metric()), frozen_metric);
                        assert_eq!(target.calls.get(), live_calls);
                        assert_eq!(rng.clone().next_u64(), live_rng);
                    }
                }
            }
            assert!(
                failures > 0,
                "exercise divergence replay for target={acceptance}, cap={cap}"
            );
            assert_eq!(remaining[0], failures);
            eprintln!(
                "funnel_replay_summary,target={acceptance},cap={cap},failures={failures},remaining={remaining:?}"
            );
        }
    }
}
