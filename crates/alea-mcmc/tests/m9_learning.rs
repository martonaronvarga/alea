use std::{cell::Cell, convert::Infallible};

use alea_core::{target::LogDensityGradient, transport::FrozenTransportTarget};
use alea_distributions::Gaussian;
use alea_math::{affine::LowRankAffineMap, buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Hmc, HmcOptions,
    adapt::{
        AffineTransportLearner, MultiPathfinderOptions, PathfinderError,
        ScoreInitializationOptions, TransportLearningOptions, learn_affine_transport,
        multi_pathfinder, score_initialize,
    },
};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn learner_fits_bounded_low_rank_map_and_updates_without_allocating() {
    let options = TransportLearningOptions::new(16, 1)
        .unwrap()
        .with_power_iterations(24)
        .unwrap()
        .with_ridge(1e-8)
        .unwrap();
    let mut learner = AffineTransportLearner::new(2, options).unwrap();
    let root = 0.5_f64.sqrt();
    let samples = (0..16)
        .map(|i| {
            let x = (i as f64 - 7.5) / 3.0;
            [1.0 + x, -2.0 + 0.9 * x + 0.1 * root * (-1.0_f64).powi(i)]
        })
        .collect::<Vec<_>>();
    let measured = allocation_counter::measure(|| {
        for sample in &samples {
            learner.observe(sample).unwrap();
        }
    });
    assert_eq!(measured.count_total, 0);
    let map = learner.finish().unwrap();
    assert_eq!(map.rank(), 1);
    assert!((map.shift()[0] - 1.0).abs() < 1e-14);
    assert!((map.shift()[1] + 2.0).abs() < 1e-14);
    let latent = [0.7, -1.2];
    let mut target = [0.0; 2];
    let mut recovered = [0.0; 2];
    map.forward(&latent, &mut target).unwrap();
    map.inverse(&target, &mut recovered).unwrap();
    for (&actual, expected) in recovered.iter().zip(latent) {
        assert!((actual - expected).abs() < 1e-11);
    }
}

#[test]
fn warmup_consumes_learning_and_retained_draws_cannot_change_map() {
    let base = Gaussian::new(2);
    let mut warmup = Hmc::new(
        &base,
        OwnedBuffer::from_fn(2, |i| [0.8, -0.6][i]),
        IdentityMetric::new(2),
        HmcOptions::default(),
    )
    .unwrap();
    let learner =
        AffineTransportLearner::new(2, TransportLearningOptions::new(64, 1).unwrap()).unwrap();
    let mut rng = StdRng::seed_from_u64(0x4d39);
    let learned = learn_affine_transport(&mut warmup, learner, 48, &mut rng).unwrap();
    assert_eq!(learned.report.observations, 49);
    let target = FrozenTransportTarget::new(&base, learned.map).unwrap();
    let before = (
        target.map().shift().to_vec(),
        target.map().scales().to_vec(),
        target.map().basis().to_vec(),
        target.map().eigenvalues().to_vec(),
    );
    let mut retained = Hmc::new(
        &target,
        learned.latent_position,
        IdentityMetric::new(2),
        HmcOptions::default(),
    )
    .unwrap();
    for _ in 0..64 {
        let _ = retained.step(&mut rng).unwrap();
    }
    assert_eq!(target.map().shift(), before.0);
    assert_eq!(target.map().scales(), before.1);
    assert_eq!(target.map().basis(), before.2);
    assert_eq!(target.map().eigenvalues(), before.3);
}

struct ShiftedGaussian;
impl LogDensityGradient for ShiftedGaussian {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Infallible> {
        let x = q[0] - 2.0;
        let y = q[1] + 1.0;
        gradient[0] = -x / 9.0;
        gradient[1] = -4.0 * y;
        Ok(-0.5 * (x * x / 9.0 + 4.0 * y * y))
    }
}

#[test]
fn multipath_pathfinder_is_deterministic_calibrated_and_costed() {
    let options = MultiPathfinderOptions::new(4, 12)
        .unwrap()
        .with_history(5)
        .unwrap()
        .with_elbo_draws(8)
        .unwrap()
        .with_draws_per_path(8)
        .unwrap();
    let initials = [-8.0, 7.0, 8.0, -7.0, -6.0, -8.0, 9.0, 8.0];
    let mut first_rng = StdRng::seed_from_u64(91);
    let first = multi_pathfinder(&ShiftedGaussian, &initials, options, &mut first_rng).unwrap();
    let mut replay_rng = StdRng::seed_from_u64(91);
    let replay = multi_pathfinder(&ShiftedGaussian, &initials, options, &mut replay_rng).unwrap();
    assert_eq!(first.position.as_slice(), replay.position.as_slice());
    assert_eq!(first.report.paths, 4);
    assert_eq!(first.report.proposal_draws, 32);
    assert!(first.report.raw_importance_resampling);
    let squared = (first.position[0] - 2.0).powi(2) / 9.0 + 4.0 * (first.position[1] + 1.0).powi(2);
    assert!(
        squared < 16.0,
        "selected initialization is outside a broad typical set"
    );

    let score = score_initialize(
        &ShiftedGaussian,
        &initials[..2],
        ScoreInitializationOptions::new(12, 1.0, 20).unwrap(),
    )
    .unwrap();
    assert!(first.report.target_evaluations > score.report.target_evaluations);
    assert_eq!(score.report.proposal_draws, 0);
}

#[derive(Debug, thiserror::Error)]
#[error("planned failure")]
struct PlannedFailure;

struct FailingTarget(Cell<usize>);
impl LogDensityGradient for FailingTarget {
    type Error = PlannedFailure;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, PlannedFailure> {
        let call = self.0.get() + 1;
        self.0.set(call);
        if call == 2 {
            return Err(PlannedFailure);
        }
        gradient[0] = -q[0];
        Ok(-0.5 * q[0] * q[0])
    }
}

#[test]
fn pathfinder_propagates_elbo_backend_failures() {
    let target = FailingTarget(Cell::new(0));
    let mut rng = StdRng::seed_from_u64(12);
    let error = multi_pathfinder(
        &target,
        &[0.0],
        MultiPathfinderOptions::new(1, 1).unwrap(),
        &mut rng,
    )
    .unwrap_err();
    assert!(matches!(error, PathfinderError::Evaluation { path: 0, .. }));
}

#[test]
#[cfg_attr(
    miri,
    ignore = "statistical workload is covered by bounded algebra tests under Miri"
)]
fn exact_frozen_transport_preserves_target_measure() {
    let map = LowRankAffineMap::new(
        OwnedBuffer::from_fn(2, |i| [2.0, -1.0][i]),
        OwnedBuffer::from_fn(2, |i| [3.0, 0.5][i]),
        OwnedBuffer::new(0),
        OwnedBuffer::new(0),
    )
    .unwrap();
    let target = FrozenTransportTarget::new(&ShiftedGaussian, map).unwrap();
    let mut chain = Hmc::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        HmcOptions::new(0.35, 5).unwrap(),
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(0x4d395f4d45415355);
    let mut target_position = [0.0; 2];
    let mut sum = [0.0; 2];
    let mut squares = [0.0; 2];
    for iteration in 0..12_000 {
        let _ = chain.step(&mut rng).unwrap();
        if iteration >= 2_000 {
            target
                .to_target(chain.point().position(), &mut target_position)
                .unwrap();
            for coordinate in 0..2 {
                sum[coordinate] += target_position[coordinate];
                squares[coordinate] += target_position[coordinate].powi(2);
            }
        }
    }
    let expected_mean = [2.0, -1.0];
    let expected_variance = [9.0, 0.25];
    for coordinate in 0..2 {
        let mean = sum[coordinate] / 10_000.0;
        let variance = squares[coordinate] / 10_000.0 - mean * mean;
        assert!((mean - expected_mean[coordinate]).abs() < 0.08);
        assert!((variance - expected_variance[coordinate]).abs() < 0.3);
    }
}
