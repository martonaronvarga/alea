use alea_core::target::{EvaluationError, LogDensityGradient, TargetSpace};
use alea_runtime::optimize::*;
use std::{cell::Cell, convert::Infallible};

fn identity() -> ObjectiveDescription {
    ObjectiveDescription {
        kind: ObjectiveKind::Other,
        measure: "Lebesgue in q".into(),
        model: "analytic test".into(),
        data: "none".into(),
        source: "test".into(),
    }
}
struct Gaussian;
impl TargetSpace for Gaussian {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for Gaussian {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = -(4. * (q[0] - 2.) + (q[1] + 1.));
        g[1] = -((q[0] - 2.) + 2. * (q[1] + 1.));
        Ok(-0.5
            * (4. * (q[0] - 2.).powi(2)
                + 2. * (q[0] - 2.) * (q[1] + 1.)
                + 2. * (q[1] + 1.).powi(2)))
    }
}
#[test]
fn correlated_quadratic_with_scales_and_offset_invariant_stopping() {
    let options = LbfgsOptions::new(200, 1000)
        .unwrap()
        .with_scales(vec![2., 0.5])
        .unwrap();
    let r = maximize(&Gaussian, &[-3., 4.], identity(), options).unwrap();
    assert_eq!(r.termination, OptimizationTermination::ScoreTolerance);
    let c = r.candidate.unwrap();
    assert!((c.position[0] - 2.).abs() < 1e-8 && (c.position[1] + 1.).abs() < 1e-8);
    assert!(c.gradient.iter().all(|v| v.abs() < 2e-8));
    assert!(r.iterations > 1 && r.evaluations <= 1000);
    assert_eq!(r.options.scales(), Some([2., 0.5].as_slice()));
}
#[test]
fn budgets_and_cancellation_are_not_convergence() {
    let r = maximize(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(200, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::EvaluationLimit);
    assert_eq!(r.evaluations, 1);
    assert_eq!(r.iterations, 0);
    let r = maximize(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(1, 100).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::IterationLimit);
    let calls = Cell::new(0);
    let r = maximize_with_cancel(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(200, 100).unwrap(),
        || {
            calls.set(calls.get() + 1);
            calls.get() > 2
        },
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::Cancelled);
    assert_eq!(r.evaluations, 1);
    assert_eq!(r.candidate.unwrap().position, [0., 0.]);
    let r = maximize_with_cancel(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(200, 100).unwrap(),
        || true,
    )
    .unwrap();
    assert_eq!(r.evaluations, 0);
    assert!(r.candidate.is_none());
}
#[derive(Debug, thiserror::Error)]
#[error("backend deliberately failed")]
struct Broken;
struct Fault {
    calls: Cell<usize>,
    mode: u8,
}
impl TargetSpace for Fault {
    type Error = Broken;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Fault {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Broken> {
        let count = self.calls.get();
        self.calls.set(count + 1);
        if count > 0 {
            match self.mode {
                0 => {
                    g[0] = 99.;
                    return Err(Broken);
                }
                1 => return Ok(0.), // incomplete gradient
                2 => return Ok(f64::NAN),
                3 => return Ok(f64::NEG_INFINITY),
                _ => {}
            }
        }
        g[0] = 2. - q[0];
        Ok(-0.5 * (q[0] - 2.).powi(2))
    }
}
#[test]
fn backend_failures_keep_the_last_candidate_and_all_calls() {
    for mode in 0..3 {
        let target = Fault {
            calls: Cell::new(0),
            mode,
        };
        let error = maximize(
            &target,
            &[0.],
            identity(),
            LbfgsOptions::new(10, 100).unwrap(),
        )
        .unwrap_err();
        let MaximizeError::Evaluation(error) = error else {
            panic!("wrong error");
        };
        assert_eq!(
            error.result.termination,
            OptimizationTermination::EvaluationFailed
        );
        assert_eq!(error.result.evaluations, 2);
        assert_eq!(error.result.candidate.as_ref().unwrap().position, [0.]);
        assert_eq!(error.result.candidate.as_ref().unwrap().gradient, [2.]);
        match mode {
            0 => assert!(matches!(error.source, EvaluationError::Model(Broken))),
            1 => assert!(matches!(
                error.source,
                EvaluationError::NonFiniteGradient { index: 0 }
            )),
            _ => assert!(matches!(error.source, EvaluationError::NonFiniteLogDensity)),
        }
    }
}
#[test]
fn negative_infinity_backtracks_but_never_corrupts_the_candidate() {
    let target = Fault {
        calls: Cell::new(0),
        mode: 3,
    };
    let r = maximize(
        &target,
        &[0.],
        identity(),
        LbfgsOptions::new(10, 100)
            .unwrap()
            .with_line_search(4)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::LineSearchFailed);
    assert_eq!(
        r.line_search_failure,
        Some(LineSearchFailure::BacktrackLimit)
    );
    assert_eq!(r.domain_trials, 4);
    assert_eq!(r.evaluations, 5);
    assert_eq!(r.candidate.unwrap().position, [0.]);
}
#[test]
fn invalid_admission_never_invokes_backend() {
    let target = Fault {
        calls: Cell::new(0),
        mode: 0,
    };
    for q in [vec![], vec![f64::NAN], vec![0., 0.]] {
        assert!(matches!(
            maximize(&target, &q, identity(), LbfgsOptions::new(10, 10).unwrap()),
            Err(MaximizeError::Invalid(_))
        ));
    }
    assert!(matches!(
        maximize(
            &target,
            &[0.],
            identity(),
            LbfgsOptions::new(1, 1)
                .unwrap()
                .with_history(usize::MAX)
                .unwrap()
        ),
        Err(MaximizeError::Invalid(_))
    ));
    assert_eq!(target.calls.get(), 0);
    assert!(LbfgsOptions::new(0, 1).is_err());
    assert!(
        LbfgsOptions::new(1, 1)
            .unwrap()
            .with_scales(vec![0.])
            .is_err()
    );
    assert!(
        LbfgsOptions::new(1, 1)
            .unwrap()
            .with_score_tolerance(f64::NAN)
            .is_err()
    );
}

struct Linear;
impl TargetSpace for Linear {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Linear {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = 1.;
        Ok(q[0])
    }
}
#[test]
fn unbounded_linear_objective_exhausts_work_instead_of_converging() {
    let r = maximize(
        &Linear,
        &[0.],
        identity(),
        LbfgsOptions::new(5, 20).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::IterationLimit);
    assert_eq!(r.skipped_updates, 5);
    assert_eq!(r.candidate.unwrap().position, [5.]);
}
struct Offset;
impl TargetSpace for Offset {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for Offset {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        Ok(Gaussian.logp_grad(q, g)? + 1e20)
    }
}
#[test]
fn huge_constant_offset_cannot_trigger_score_convergence_or_accept_no_progress() {
    let r = maximize(
        &Offset,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(5, 100).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::LineSearchFailed);
    assert_eq!(r.iterations, 0);
    assert_eq!(r.candidate.unwrap().position, [0., 0.]);
}
#[test]
fn invalid_initial_objective_has_no_candidate() {
    let target = Fault {
        calls: Cell::new(1),
        mode: 3,
    };
    let MaximizeError::Evaluation(e) = maximize(
        &target,
        &[0.],
        identity(),
        LbfgsOptions::new(2, 10).unwrap(),
    )
    .unwrap_err() else {
        panic!("wrong error")
    };
    assert!(e.result.candidate.is_none());
    assert_eq!(e.result.evaluations, 1);
}

struct Supported;
impl TargetSpace for Supported {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for Supported {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        if !(0. ..1.).contains(&q[0]) {
            return Ok(f64::NEG_INFINITY);
        }
        g[0] = 20. * (0.8 - q[0]);
        Ok(-10. * (q[0] - 0.8).powi(2))
    }
}
#[test]
fn valid_domain_backtracking_recovers_and_initial_support_zero_is_an_error() {
    let r = maximize(
        &Supported,
        &[0.1],
        identity(),
        LbfgsOptions::new(100, 1000).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::ScoreTolerance);
    assert!(r.domain_trials > 0);
    assert!((r.candidate.unwrap().position[0] - 0.8).abs() < 1e-9);
    assert!(matches!(
        maximize(
            &Supported,
            &[2.],
            identity(),
            LbfgsOptions::new(10, 100).unwrap()
        ),
        Err(MaximizeError::Evaluation(_))
    ));
}
struct ChangingDimension(Cell<usize>);
impl TargetSpace for ChangingDimension {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.0.get()
    }
}
impl LogDensityGradient for ChangingDimension {
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = 1.;
        self.0.set(2);
        Ok(0.)
    }
}
#[test]
fn changed_target_dimension_does_not_invoke_or_count_a_second_backend_call() {
    let MaximizeError::Evaluation(e) = maximize(
        &ChangingDimension(Cell::new(1)),
        &[0.],
        identity(),
        LbfgsOptions::new(10, 100).unwrap(),
    )
    .unwrap_err() else {
        panic!("wrong error")
    };
    assert!(matches!(
        e.source,
        EvaluationError::TargetDimensionChanged {
            expected: 1,
            actual: 2
        }
    ));
    assert_eq!(e.result.evaluations, 1);
    assert_eq!(e.result.candidate.as_ref().unwrap().position, [0.]);
}

struct Rosenbrock;
impl TargetSpace for Rosenbrock {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
}
impl LogDensityGradient for Rosenbrock {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        let residual = q[1] - q[0] * q[0];
        g[0] = 2. * (1. - q[0]) + 400. * q[0] * residual;
        g[1] = -200. * residual;
        Ok(-(1. - q[0]).powi(2) - 100. * residual * residual)
    }
}
#[test]
fn curved_rosenbrock_valley_reaches_the_known_stationary_point() {
    let r = maximize(
        &Rosenbrock,
        &[-1.2, 1.],
        identity(),
        LbfgsOptions::new(2000, 20000)
            .unwrap()
            .with_score_tolerance(1e-6)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        r.termination,
        OptimizationTermination::ScoreTolerance,
        "{r:?}"
    );
    let c = r.candidate.unwrap();
    assert!(c.position.iter().all(|x| (x - 1.).abs() < 1e-5));
    assert!(c.value.abs() < 1e-12);
}

struct OverflowDirection;
impl TargetSpace for OverflowDirection {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for OverflowDirection {
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = f64::MAX;
        Ok(0.)
    }
}
#[test]
fn unrepresentable_direction_is_explicit_and_keeps_finite_initial_data() {
    let r = maximize(
        &OverflowDirection,
        &[0.],
        identity(),
        LbfgsOptions::new(10, 100).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::NumericalDirection);
    assert_eq!(r.evaluations, 1);
    assert_eq!(r.iterations, 0);
    assert_eq!(r.candidate.unwrap().gradient, [f64::MAX]);
    assert!(matches!(
        maximize(
            &Gaussian,
            &[0., 0.],
            identity(),
            LbfgsOptions::new(10, 100)
                .unwrap()
                .with_scales(vec![1.])
                .unwrap()
        ),
        Err(MaximizeError::Invalid(_))
    ));
}

#[test]
fn named_projection_keeps_free_scores_and_checked_constrained_values() {
    use alea_core::{
        model::{assembly::Coordinate, prediction::NamedOutput},
        transform::{ParameterLayout, Transform},
    };
    let schema = || {
        NamedOutput::new(
            vec![
                Coordinate {
                    name: "mean".into(),
                    dimension: None,
                    coordinate: None,
                    block: None,
                },
                Coordinate {
                    name: "scale".into(),
                    dimension: None,
                    coordinate: None,
                    block: None,
                },
            ]
            .into(),
            ParameterLayout::new([
                Transform::identity(1).unwrap(),
                Transform::positive(1).unwrap(),
            ])
            .unwrap(),
            (),
        )
        .unwrap()
    };
    let r = maximize(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(10, 1).unwrap(),
    )
    .unwrap();
    let g = r.candidate.as_ref().unwrap().gradient.clone();
    let named = NamedFit::new(schema(), r).unwrap();
    assert_eq!(named.estimate("mean"), Some(0.));
    assert_eq!(named.estimate("scale"), Some(1.));
    assert_eq!(named.estimate("missing"), None);
    assert_eq!(named.estimates(), Some([0., 1.].as_slice()));
    assert_eq!(named.result().candidate.as_ref().unwrap().gradient, g);
    assert_eq!(
        named.result().termination,
        OptimizationTermination::EvaluationLimit
    );
    let empty = maximize_with_cancel(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(1, 1).unwrap(),
        || true,
    )
    .unwrap();
    assert!(
        NamedFit::new(schema(), empty)
            .unwrap()
            .estimates()
            .is_none()
    );
    let mut malformed = named.into_result();
    malformed.candidate.as_mut().unwrap().position[1] = 1000.;
    let e = NamedFit::new(schema(), malformed).unwrap_err();
    assert_eq!(e.result.candidate.as_ref().unwrap().position[1], 1000.);
}

#[test]
fn named_projection_respects_dimension_reducing_simplex_coordinates() {
    use alea_core::{
        model::{assembly::Coordinate, prediction::NamedOutput},
        transform::{ParameterLayout, Transform},
    };
    let r = maximize(
        &Gaussian,
        &[0., 0.],
        identity(),
        LbfgsOptions::new(1, 1).unwrap(),
    )
    .unwrap();
    let coordinates = (0..3)
        .map(|i| Coordinate {
            name: format!("weight[{i}]"),
            dimension: Some("component".into()),
            coordinate: Some(i.to_string()),
            block: Some("weights".into()),
        })
        .collect::<Vec<_>>()
        .into();
    let schema = NamedOutput::new(
        coordinates,
        ParameterLayout::new([Transform::simplex(3).unwrap()]).unwrap(),
        (),
    )
    .unwrap();
    let fit = NamedFit::new(schema, r).unwrap();
    assert_eq!(fit.estimates().unwrap().len(), 3);
    assert!((fit.estimates().unwrap().iter().sum::<f64>() - 1.).abs() < 1e-15);
    assert_eq!(fit.result().candidate.as_ref().unwrap().gradient.len(), 2);
}

struct RoundedPosition;
impl TargetSpace for RoundedPosition {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
}
impl LogDensityGradient for RoundedPosition {
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = 1.;
        Ok(0.)
    }
}
#[test]
fn unrepresentable_step_retains_candidate_without_an_extra_evaluation() {
    let r = maximize(
        &RoundedPosition,
        &[1e30],
        identity(),
        LbfgsOptions::new(10, 100).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::LineSearchFailed);
    assert_eq!(
        r.line_search_failure,
        Some(LineSearchFailure::NoRepresentableStep)
    );
    assert_eq!(r.evaluations, 1);
    assert_eq!(r.iterations, 0);
    assert_eq!(r.candidate.unwrap().gradient, [1.]);
    let r = maximize(
        &Gaussian,
        &[2., -1.],
        identity(),
        LbfgsOptions::new(10, 100).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::ScoreTolerance);
    assert_eq!(r.line_search_failure, None);
}
