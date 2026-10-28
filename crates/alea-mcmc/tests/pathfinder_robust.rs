use alea_core::target::{EvaluationError, LogDensityGradient};
use alea_mcmc::adapt::{
    ImportanceResampling, MultiPathfinderOptions, PathfinderError, PathfinderPathStatus,
    multi_pathfinder,
};
use rand::{SeedableRng, rngs::StdRng};
use std::{cell::Cell, convert::Infallible};
struct Supported {
    calls: Cell<usize>,
}
impl alea_core::target::TargetSpace for Supported {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Supported {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        self.calls.set(self.calls.get() + 1);
        if q[0] <= 0. {
            return Ok(f64::NEG_INFINITY);
        }
        g[0] = -100. * (q[0] - 1.);
        Ok(-50. * (q[0] - 1.).powi(2))
    }
}
#[test]
fn support_zero_paths_and_proposals_are_visible_and_backtracking_survives() {
    for policy in [
        ImportanceResampling::RawMixture,
        ImportanceResampling::PsisPathwise,
    ] {
        let t = Supported {
            calls: Cell::new(0),
        };
        let result = multi_pathfinder(
            &t,
            &[-1., 0.1, 3., 1.],
            MultiPathfinderOptions::new(4, 40)
                .unwrap()
                .with_draws_per_path(64)
                .unwrap()
                .with_resampling(policy),
            &mut StdRng::seed_from_u64(881),
        )
        .unwrap();
        assert_eq!(
            result.paths[0].status,
            PathfinderPathStatus::NoApproximation
        );
        assert_eq!(result.report.requested_paths, 4);
        assert_eq!(result.report.target_evaluations, t.calls.get());
        assert_eq!(result.report.paths, 3);
        assert!(result.position[0] > 0.);
        assert!(result.report.optimization_steps > 0);
        assert!(result.report.importance_ess.unwrap() > 10.);
        assert_eq!(
            result.report.raw_importance_resampling,
            policy == ImportanceResampling::RawMixture
        );
    }
}
#[derive(Debug, thiserror::Error)]
#[error("backend failed")]
struct Fault;
struct Broken;
impl alea_core::target::TargetSpace for Broken {
    type Error = Fault;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Broken {
    fn logp_grad(&self, _: &[f64], _: &mut [f64]) -> Result<f64, Fault> {
        Err(Fault)
    }
}
#[test]
fn robust_mode_does_not_hide_concrete_backend_errors() {
    let error = multi_pathfinder(
        &Broken,
        &[0., 1.],
        MultiPathfinderOptions::new(2, 4)
            .unwrap()
            .with_resampling(ImportanceResampling::PsisPathwise),
        &mut StdRng::seed_from_u64(77),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        PathfinderError::Evaluation {
            source: EvaluationError::Model(Fault),
            ..
        }
    ));
}

struct Sharp;
impl alea_core::target::TargetSpace for Sharp {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Sharp {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = 1e10 * (3. - q[0]);
        Ok(-0.5e10 * (q[0] - 3.).powi(2))
    }
}
#[test]
fn first_lbfgs_trial_is_scaled_before_curvature_exists() {
    let result = multi_pathfinder(
        &Sharp,
        &[0.1],
        MultiPathfinderOptions::new(1, 40)
            .unwrap()
            .with_draws_per_path(64)
            .unwrap()
            .with_resampling(ImportanceResampling::PsisPathwise),
        &mut StdRng::seed_from_u64(4401),
    )
    .unwrap();
    assert!(result.report.optimization_steps >= 2);
    assert!((result.position[0] - 3.).abs() < 5e-5);
    assert!(result.report.target_evaluations < 500);
    assert_eq!(
        result.report.optimization_steps,
        result.paths[0].optimization_steps
    );
}

struct OneInvalidPath;
impl alea_core::target::TargetSpace for OneInvalidPath {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for OneInvalidPath {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        if q[0] == -99. {
            g[0] = f64::NAN;
            return Ok(-1.);
        }
        g[0] = -q[0];
        Ok(-0.5 * q[0] * q[0])
    }
}
#[test]
fn numerical_output_failure_is_recorded_without_losing_other_paths() {
    let result = multi_pathfinder(
        &OneInvalidPath,
        &[0.1, -99., 0.2],
        MultiPathfinderOptions::new(3, 20)
            .unwrap()
            .with_draws_per_path(64)
            .unwrap()
            .with_resampling(ImportanceResampling::PsisPathwise),
        &mut StdRng::seed_from_u64(4402),
    )
    .unwrap();
    assert_eq!(result.report.paths, 2);
    assert_eq!(result.report.requested_paths, 3);
    assert_eq!(
        result.paths[1].status,
        PathfinderPathStatus::NumericalFailure
    );
    assert_eq!(result.paths[1].target_evaluations, 1);
    assert!(result.position[0].is_finite());
    let legacy = multi_pathfinder(
        &OneInvalidPath,
        &[-99.],
        MultiPathfinderOptions::new(1, 20).unwrap(),
        &mut StdRng::seed_from_u64(4402),
    )
    .unwrap_err();
    assert!(matches!(
        legacy,
        PathfinderError::Evaluation {
            source: EvaluationError::NonFiniteGradient { .. },
            ..
        }
    ));
}
