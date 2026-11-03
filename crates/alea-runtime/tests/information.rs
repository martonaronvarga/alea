#![cfg(feature = "faer")]
use alea_core::{
    capability::{HvpError, PotentialHessianVector},
    model::{assembly::Coordinate, prediction::NamedOutput},
    target::{LogDensityGradient, TargetSpace},
    transform::{ParameterLayout, Transform},
};
use alea_math::information::{InformationError, InformationOptions};
use alea_runtime::optimize::{
    LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination,
    information::{ObservedInformationError, observed_information},
    maximize,
};
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};
#[derive(Debug, thiserror::Error, PartialEq)]
#[error("original curvature fault")]
struct Fault;
struct Quadratic {
    mode: Cell<usize>,
    evaluations: Cell<usize>,
    hvps: Cell<usize>,
}
impl Quadratic {
    fn new() -> Self {
        Self {
            mode: Cell::new(0),
            evaluations: Cell::new(0),
            hvps: Cell::new(0),
        }
    }
}
impl TargetSpace for Quadratic {
    type Error = Fault;
    fn dimension(&self) -> usize {
        if self.mode.get() == 9 && self.hvps.get() > 0 {
            1
        } else {
            2
        }
    }
}
impl LogDensityGradient for Quadratic {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
        self.evaluations.set(self.evaluations.get() + 1);
        if self.mode.get() == 7 {
            return Err(Fault);
        }
        g[0] = -2. * q[0] - q[1];
        g[1] = -q[0] - 2. * q[1];
        Ok(-q[0] * q[0] - q[0] * q[1] - q[1] * q[1] + if self.mode.get() == 8 { 1. } else { 0. })
    }
}
impl PotentialHessianVector for Quadratic {
    fn potential_hvp(&self, _: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Fault> {
        let call = self.hvps.get();
        self.hvps.set(call + 1);
        match self.mode.get() {
            1 if call == 1 => {
                out[0] = 99.;
                return Err(Fault);
            }
            2 => {
                out[0] = 1.;
                return Ok(());
            }
            3 => {
                out.fill(f64::INFINITY);
                return Ok(());
            }
            4 => {
                out[0] = v[1];
                out[1] = -v[0];
                return Ok(());
            }
            5 => {
                out[0] = -v[0];
                out[1] = v[1];
                return Ok(());
            }
            6 => panic!("curvature panic"),
            _ => {}
        }
        out[0] = 2. * v[0] + v[1];
        out[1] = v[0] + 2. * v[1];
        Ok(())
    }
}
fn options() -> InformationOptions {
    InformationOptions {
        relative_eigenvalue_cutoff: 1e-12,
        relative_symmetry_tolerance: 64. * f64::EPSILON,
    }
}
fn fit(target: &Quadratic, transform: Transform) -> NamedFit {
    let result = maximize(
        target,
        &[0., 0.],
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "declared physical likelihood".into(),
            model: "quadratic".into(),
            data: "test".into(),
            source: "information.rs".into(),
        },
        LbfgsOptions::new(5, 20)
            .unwrap()
            .with_scales(vec![2., 3.])
            .unwrap(),
    )
    .unwrap();
    let n = transform.constrained_dimension();
    let names = (0..n)
        .map(|i| Coordinate {
            name: format!("value[{i}]"),
            dimension: None,
            coordinate: None,
            block: None,
        })
        .collect::<Vec<_>>()
        .into();
    NamedFit::new(
        NamedOutput::new(names, ParameterLayout::new([transform]).unwrap(), ()).unwrap(),
        result,
    )
    .unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-12, "{a} != {b}");
}
#[test]
fn checked_hvps_preserve_basis_counts_coordinates_and_original_fit() {
    let target = Quadratic::new();
    let fit = fit(&target, Transform::identity(2).unwrap());
    let before = fit.result().candidate.clone();
    let old_evals = target.evaluations.get();
    let information = observed_information(&target, &fit, options()).unwrap();
    assert_eq!(target.evaluations.get() - old_evals, 1);
    assert_eq!(target.hvps.get(), 2);
    for (&actual, expected) in
        information
            .free_covariance()
            .as_slice()
            .iter()
            .zip([2. / 3., -1. / 3., -1. / 3., 2. / 3.])
    {
        close(actual, expected);
    }
    assert_eq!(
        information.constrained_covariance().values,
        information.free_covariance().as_slice()
    );
    assert_eq!(information.objective(), &fit.result().objective);
    assert_eq!(information.position(), [0., 0.]);
    assert_eq!(fit.result().candidate, before);
    close(information.factor().reciprocal_condition(), 1. / 3.);
}
#[test]
fn nonlinear_and_dimension_reducing_charts_use_composition_not_density_pullback() {
    let target = Quadratic::new();
    for transform in [
        Transform::positive(2).unwrap(),
        Transform::simplex(3).unwrap(),
    ] {
        let f = fit(&target, transform);
        let information = observed_information(&target, &f, options()).unwrap();
        let n = f.estimates().unwrap().len();
        let d = 2;
        let mut j = vec![0.; n * d];
        let mut scratch = vec![0.; f.parameters().layout().scratch_dimension()];
        // Independent central differences of constrained values, not the production pullback.
        for k in 0..d {
            let mut qp = [0.; 2];
            let mut qm = [0.; 2];
            qp[k] = 1e-5;
            qm[k] = -1e-5;
            let mut xp = vec![0.; n];
            let mut xm = vec![0.; n];
            f.parameters()
                .layout()
                .constrain_values(&qp, &mut xp, &mut scratch)
                .unwrap();
            f.parameters()
                .layout()
                .constrain_values(&qm, &mut xm, &mut scratch)
                .unwrap();
            for i in 0..n {
                j[i * d + k] = (xp[i] - xm[i]) / 2e-5;
            }
        }
        let c = information.free_covariance();
        let physical = &information.constrained_covariance().values;
        for i in 0..n {
            for k in 0..n {
                let mut expected = 0.;
                for p in 0..d {
                    for q in 0..d {
                        expected += j[i * d + p] * c.as_slice()[p * d + q] * j[k * d + q];
                    }
                }
                assert!((physical[i * n + k] - expected).abs() < 1e-10);
            }
        }
        if n == 3 {
            for row in physical.chunks_exact(3) {
                close(row.iter().sum(), 0.);
            }
        }
    }
}
#[test]
fn invalid_fit_and_policy_fail_before_target_or_derivative_calls() {
    let target = Quadratic::new();
    let f = fit(&target, Transform::identity(2).unwrap());
    for mode in 0..3 {
        let mut r = f.result().clone();
        match mode {
            0 => r.objective.kind = ObjectiveKind::Posterior,
            1 => r.termination = OptimizationTermination::EvaluationLimit,
            _ => r.candidate = None,
        }
        let other = NamedFit::new(f.parameters().clone(), r).unwrap();
        let count = target.evaluations.get();
        assert!(matches!(
            observed_information(&target, &other, options()),
            Err(ObservedInformationError::InvalidFit)
        ));
        assert_eq!(count, target.evaluations.get());
    }
    let mut invalid = options();
    invalid.relative_eigenvalue_cutoff = f64::NAN;
    let count = target.evaluations.get();
    assert!(matches!(
        observed_information(&target, &f, invalid),
        Err(ObservedInformationError::Matrix(InformationError::Invalid))
    ));
    assert_eq!(count, target.evaluations.get());
    assert_eq!(target.hvps.get(), 0);
}
#[test]
fn evaluation_errors_and_changed_dimensions_preserve_fit() {
    let target = Quadratic::new();
    let f = fit(&target, Transform::identity(2).unwrap());
    let before = f.result().candidate.clone();
    target.mode.set(7);
    assert!(matches!(
        observed_information(&target, &f, options()),
        Err(ObservedInformationError::Evaluation(
            alea_core::target::EvaluationError::Model(Fault)
        ))
    ));
    assert_eq!(target.hvps.get(), 0);
    target.mode.set(9);
    assert!(matches!(
        observed_information(&target, &f, options()),
        Err(ObservedInformationError::InvalidFit)
    ));
    assert_eq!(f.result().candidate, before);
}
#[test]
fn typed_partial_missing_nonfinite_asymmetric_and_panicking_hvps_publish_nothing() {
    let target = Quadratic::new();
    let f = fit(&target, Transform::identity(2).unwrap());
    let before = f.result().candidate.clone();
    for mode in 1..=6 {
        target.mode.set(mode);
        target.hvps.set(0);
        if mode == 6 {
            assert!(
                catch_unwind(AssertUnwindSafe(|| observed_information(
                    &target,
                    &f,
                    options()
                )))
                .is_err()
            );
        } else {
            let error = observed_information(&target, &f, options()).unwrap_err();
            match mode {
                1 => assert!(matches!(
                    error,
                    ObservedInformationError::Hvp {
                        column: 1,
                        source: HvpError::Evaluation(alea_core::target::EvaluationError::Model(
                            Fault
                        ))
                    }
                )),
                2 | 3 => assert!(matches!(
                    error,
                    ObservedInformationError::Hvp {
                        source: HvpError::NonFiniteOutput { .. },
                        ..
                    }
                )),
                _ => assert!(matches!(error, ObservedInformationError::Matrix(_))),
            }
        }
        assert_eq!(f.result().candidate, before);
        target.mode.set(0);
        target.hvps.set(0);
        assert!(observed_information(&target, &f, options()).is_ok());
    }
    target.mode.set(8);
    target.hvps.set(0);
    assert!(matches!(
        observed_information(&target, &f, options()),
        Err(ObservedInformationError::ObjectiveMismatch)
    ));
    assert_eq!(target.hvps.get(), 0);
}
