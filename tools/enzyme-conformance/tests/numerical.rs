//! Real generated native Gaussian AD, not callback-based derivative substitutes.
use alea::{
    autodiff::autodiff::AutodiffError,
    capability::PotentialHessianVector,
    math::{buffer::OwnedBuffer, metric::IdentityMetric},
    sampling::hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
    target::{EvaluationError, PointState, evaluate},
};

#[test]
fn generated_gaussian_overflow_diverges_only_during_integration() {
    let target = alea_enzyme_conformance::regression::target();
    let mut gradient = [0.; 2];
    assert!(matches!(
        evaluate(&target, &[0., 400.], &mut gradient),
        Err(EvaluationError::Model(AutodiffError::NonFinite))
    ));
    assert!(
        PointState::new(
            &target,
            OwnedBuffer::from_fn(2, |i| if i == 1 { 400. } else { 0. })
        )
        .is_err()
    );
    let point = PointState::new(&target, OwnedBuffer::new(2)).unwrap();
    let initial = PhaseState::new(
        point,
        OwnedBuffer::from_fn(2, |i| if i == 1 { 1000. } else { 0. }),
    )
    .unwrap();
    let metric = IdentityMetric::new(2);
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
    assert_eq!(initial.point().position(), &[0., 0.]);
    assert!(
        workspace
            .start_from_phase(&initial, &metric)
            .unwrap()
            .step(SignedStep::new(0.0001).unwrap())
            .is_ok()
    );
    // The sampling classification hook does not change curvature API errors.
    assert!(matches!(
        target.potential_hvp(&[0., 400.], &[0., 1.], &mut gradient),
        Err(AutodiffError::NonFinite)
    ));
}
