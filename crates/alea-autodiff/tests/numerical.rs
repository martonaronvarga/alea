//! Stable adapter-contract tests; compiler-generated AD is tested separately.
use alea::{
    ensemble::chain_rng,
    output::ChainId,
    sampling::{
        Nuts, NutsOptions,
        hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
        nuts::NutsTermination,
    },
};
use alea_autodiff::autodiff::{AutodiffError, EnzymeHigherOrderTarget, EnzymeModel};
use alea_core::{
    capability::{PotentialHessianDerivative, PotentialHessianVector},
    model::{ConstrainedModel, TransformedTarget},
    target::{EvaluationError, LogDensityGradient, PointState, evaluate},
    transform::{ParameterLayout, Transform},
};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};

fn check_target<T: LogDensityGradient>(target: &T)
where
    T::Error: std::error::Error + 'static,
{
    let point = PointState::new(target, OwnedBuffer::new(1)).unwrap();
    let before = (
        point.position().to_vec(),
        point.gradient().to_vec(),
        point.log_density(),
    );
    let initial = PhaseState::new(point, OwnedBuffer::from_fn(1, |_| 1.)).unwrap();
    let metric = IdentityMetric::new(1);
    let mut workspace = PhaseWorkspace::new(initial.point());
    let error = workspace
        .start_from_phase(&initial, &metric)
        .unwrap()
        .step(SignedStep::new(1.).unwrap())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        PhaseError::Divergence(Divergence::TargetNumerical)
    ));
    // Rejection must preserve the live cache exactly, not merely approximately.
    assert_eq!(initial.point().position(), before.0);
    assert_eq!(initial.point().gradient(), before.1);
    assert_eq!(initial.point().log_density().to_bits(), before.2.to_bits());
    assert!(
        workspace
            .start_from_phase(&initial, &metric)
            .unwrap()
            .step(SignedStep::new(0.01).unwrap())
            .is_ok()
    );
    let mut chain = Nuts::new(
        target,
        OwnedBuffer::new(1),
        metric,
        NutsOptions::new(100., 4).unwrap(),
    )
    .unwrap();
    let transition = chain.step(&mut chain_rng([17; 32], ChainId(0))).unwrap();
    assert_eq!(
        transition.termination,
        NutsTermination::Divergence(Divergence::TargetNumerical)
    );
    assert_eq!(chain.point().position(), &[0.]);
    assert!(PointState::new(target, OwnedBuffer::from_fn(1, |_| 1.)).is_err());
}

#[test]
fn only_nonfinite_adapter_errors_are_recoverable_in_checked_integration() {
    for bad_gradient in [false, true] {
        let callback = |q: &[f64], g: &mut [f64], seed: f64| {
            // Deliberately accumulates, as Enzyme does. Calls after failure must reset.
            if q[0].abs() > 0.1 {
                g[0] += if bad_gradient { f64::NAN } else { 0. };
                return if bad_gradient { 0. } else { f64::INFINITY };
            }
            g[0] += -q[0] * seed;
            -0.5 * q[0] * q[0]
        };
        let model = EnzymeModel::new(1, callback);
        assert!(model.is_numerical_error(&AutodiffError::NonFinite));
        assert!(!model.is_numerical_error(&AutodiffError::Dimension));
        let mut g = [42.];
        assert!(matches!(
            model.logp_grad(&[], &mut g),
            Err(AutodiffError::Dimension)
        ));
        assert_eq!(g, [42.]);
        assert!(matches!(
            model.logp_grad(&[f64::NAN], &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert_eq!(g, [42.]);
        let target = TransformedTarget::new(
            model,
            ParameterLayout::new([Transform::identity(1).unwrap()]).unwrap(),
        )
        .unwrap();
        check_target(&target);
        let target = EnzymeHigherOrderTarget::new(
            1,
            callback,
            |_: &[f64], _: &[f64], out: &mut [f64]| out.fill(f64::NAN),
            |_: &[f64], h: &mut [f64], d: &mut [f64]| {
                h.fill(f64::NAN);
                d.fill(f64::NAN);
            },
        );
        assert!(target.is_numerical_error(&AutodiffError::NonFinite));
        assert!(!target.is_numerical_error(&AutodiffError::Dimension));
        assert!(matches!(
            evaluate(&target, &[1.], &mut g),
            Err(EvaluationError::Model(AutodiffError::NonFinite))
        ));
        assert!(matches!(
            target.logp_grad(&[], &mut g),
            Err(AutodiffError::Dimension)
        ));
        assert!(matches!(
            target.logp_grad(&[f64::NAN], &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hvp(&[0.], &[1.], &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hessian_and_derivatives(&[0.], &mut g, &mut [0.]),
            Err(AutodiffError::NonFinite)
        ));
        check_target(&target);
        target.logp_grad(&[0.01], &mut g).unwrap();
        assert_eq!(g, [-0.01]);
    }
}
