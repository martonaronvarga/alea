use alea_core::{
    model::{
        ModelError,
        assembly::{AssemblyError, ModelBuilder},
    },
    target::{EvaluationError, LogDensityGradient, PointState, evaluate},
    transform::TransformError,
};
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    Nuts, NutsOptions,
    hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep},
};
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn transform_failure_is_typed_in_evaluation_and_initialization_but_diverges_in_integration() {
    let mut b = ModelBuilder::new();
    let sigma = b.positive("sigma").unwrap();
    let logp = b.log(sigma.expr()).unwrap();
    b.prior("density", logp).unwrap();
    let target = b.compile().unwrap().into_target().unwrap();
    let mut gradient = [42.];
    let error = evaluate(&target, &[-1000.], &mut gradient).unwrap_err();
    assert!(matches!(
        error,
        EvaluationError::Model(ModelError::Transform(TransformError::PrecisionLoss {
            index: 0
        }))
    ));
    assert!(PointState::new(&target, OwnedBuffer::from_fn(1, |_| -1000.)).is_err());
    assert!(
        target.is_numerical_error(&ModelError::Transform(TransformError::PrecisionLoss {
            index: 0
        }))
    );
    for error in [
        ModelError::Dimension,
        ModelError::WorkspaceBusy,
        ModelError::Transform(TransformError::Domain { index: 0 }),
        ModelError::Transform(TransformError::NonFiniteInput { index: 0 }),
        ModelError::Evaluation(EvaluationError::Model(AssemblyError::Shape)),
        ModelError::Evaluation(EvaluationError::Model(AssemblyError::Factor(Box::new(
            std::io::Error::other("backend"),
        )))),
    ] {
        assert!(!target.is_numerical_error(&error), "{error:?}");
    }
    let initial = PhaseState::new(
        PointState::new(&target, OwnedBuffer::new(1)).unwrap(),
        OwnedBuffer::from_fn(1, |_| -1001.),
    )
    .unwrap();
    let mut workspace = PhaseWorkspace::new(initial.point());
    let metric = IdentityMetric::new(1);
    let phase = workspace.start_from_phase(&initial, &metric).unwrap();
    let error = phase.step(SignedStep::new(1.).unwrap()).err().unwrap();
    assert!(matches!(
        error,
        PhaseError::Divergence(Divergence::TargetNumerical)
    ));
    assert_eq!(initial.point().position(), &[0.]);
    assert_eq!(initial.point().gradient(), &[2.]);
    // Scratch is reusable after the failed proposal.
    assert!(
        workspace
            .start_from_phase(&initial, &metric)
            .unwrap()
            .step(SignedStep::new(0.001).unwrap())
            .is_ok()
    );
}

#[test]
fn native_numerical_expression_rejects_but_opaque_backend_failure_propagates() {
    use alea_core::distribution::DistributionError;
    for kind in 0..7 {
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap();
        let factor = b
            .custom(&[x.expr()], move |q, g| {
                if q[0] != 0. {
                    return Err(match kind {
                        0 => AssemblyError::Numerical,
                        1 => AssemblyError::Factor(Box::new(DistributionError::Numerical)),
                        2 => AssemblyError::Factor(Box::new(std::io::Error::other("backend"))),
                        3 => AssemblyError::Factor(Box::new(DistributionError::Parameter(
                            "injected",
                        ))),
                        4 => AssemblyError::Factor(Box::new(DistributionError::Shape)),
                        5 => AssemblyError::Factor(Box::new(DistributionError::Observation)),
                        _ => AssemblyError::Factor(Box::new(DistributionError::SamplingBudget)),
                    });
                }
                g[0] = 0.;
                Ok(0.)
            })
            .unwrap();
        b.prior("factor", factor).unwrap();
        let target = b.compile().unwrap().into_target().unwrap();
        let mut chain = Nuts::new(
            &target,
            OwnedBuffer::new(1),
            IdentityMetric::new(1),
            NutsOptions::new(0.2, 4).unwrap(),
        )
        .unwrap();
        let result = chain.step(&mut SmallRng::seed_from_u64(17));
        if kind < 2 {
            assert_eq!(
                result.unwrap().termination,
                alea_mcmc::nuts::NutsTermination::Divergence(Divergence::TargetNumerical)
            );
        } else {
            assert!(matches!(
                result,
                Err(alea_mcmc::nuts::NutsError::Phase {
                    source: PhaseError::Evaluation(EvaluationError::Model(ModelError::Evaluation(
                        EvaluationError::Model(AssemblyError::Factor(_))
                    ))),
                    ..
                })
            ));
        }
        assert_eq!(chain.point().position(), &[0.]);
    }
}
