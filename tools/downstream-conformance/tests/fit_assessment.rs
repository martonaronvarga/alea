use alea::{
    distributions::modeling::regression::{
        RegressionData, bernoulli_logit_regression,
        diagnostics::{
            ConeSearchOptions, MleExistence, RegressionAssessmentError, RegressionEvidence,
        },
    },
    model::{assembly::ModelBuilder, prediction::NamedOutput},
    optimize::{
        LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination,
        maximize,
    },
};

#[test]
fn finite_attainment_nonidentification_and_optimizer_budget_remain_separate() {
    let data = RegressionData::new(2, vec![1., 1., 1., 1.], vec![false, true]).unwrap();
    let evidence = data
        .assess_bernoulli(&["intercept", "duplicate"], ConeSearchOptions::default())
        .unwrap();
    assert!(matches!(
        evidence.evidence(),
        RegressionEvidence::BernoulliLogit(MleExistence::FiniteMaximum)
    ));
    assert!(evidence.design_rank().is_none());
    let mut builder = ModelBuilder::new();
    let a = builder.real("intercept").unwrap();
    let b = builder.real("duplicate").unwrap();
    let f = bernoulli_logit_regression(&mut builder, &[a.expr(), b.expr()], data).unwrap();
    builder.likelihood("responses", f).unwrap();
    let model = builder.compile().unwrap();
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap();
    let objective = model.select_likelihood().unwrap().into_objective().unwrap();
    let result = maximize(
        &objective,
        &[2., 2.],
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "coefficients".into(),
            model: "duplicate columns".into(),
            data: "two observations".into(),
            source: "public-facade test".into(),
        },
        LbfgsOptions::new(10, 1).unwrap(),
    )
    .unwrap();
    let original = result.candidate.clone();
    let fit = NamedFit::new(names, result)
        .unwrap()
        .with_assessment(evidence);
    assert_eq!(
        fit.result().termination,
        OptimizationTermination::EvaluationLimit
    );
    assert_eq!(fit.result().candidate, original);
    assert_eq!(fit.estimates().unwrap(), [2., 2.]);
    assert_eq!(
        fit.assessment().coefficient_names(),
        ["intercept", "duplicate"]
    );
    let (result, evidence) = fit.into_parts();
    assert_eq!(result.evaluations, 1);
    assert!(evidence.search_options().is_some());
}

#[test]
fn names_budgets_and_exact_witnesses_are_not_silently_reinterpreted() {
    let data = RegressionData::new(2, vec![1., -1., 1., 1.], vec![false, true]).unwrap();
    for names in [&["a"][..], &["a", "a"], &["a", ""]] {
        assert!(matches!(
            data.assess_bernoulli(names, ConeSearchOptions::default()),
            Err(RegressionAssessmentError::CoefficientNames)
        ));
    }
    let a = data
        .assess_bernoulli(&["intercept", "slope"], ConeSearchOptions::default())
        .unwrap();
    let RegressionEvidence::BernoulliLogit(MleExistence::NoFiniteMaximum(w)) = a.evidence() else {
        panic!("missing witness")
    };
    assert_eq!(w.improving_rows(), 2);
    let a = data
        .assess_bernoulli(
            &["intercept", "slope"],
            ConeSearchOptions {
                max_rows: 0,
                ..ConeSearchOptions::default()
            },
        )
        .unwrap();
    assert!(matches!(
        a.evidence(),
        RegressionEvidence::BernoulliLogit(MleExistence::Unresolved)
    ));
    let data = RegressionData::new(1, vec![1., 1.], vec![0_u64, 0])
        .unwrap()
        .with_offsets(vec![0., 2.])
        .unwrap();
    let a = data
        .assess_poisson(&["rate"], ConeSearchOptions::default())
        .unwrap();
    assert!(matches!(
        a.evidence(),
        RegressionEvidence::PoissonLog(MleExistence::NoFiniteMaximum(_))
    ));
}

#[cfg(feature = "faer")]
#[test]
fn rank_cutoff_and_gaussian_residual_evidence_are_explicit() {
    let data = RegressionData::new(2, vec![1., 1., 1., 1.], vec![false, true]).unwrap();
    let a = data
        .assess_bernoulli_with_rank(&["a", "b"], ConeSearchOptions::default(), 1e-12)
        .unwrap();
    assert_eq!(a.design_rank().unwrap().numerical_rank, 1);
    assert!(matches!(
        a.evidence(),
        RegressionEvidence::BernoulliLogit(MleExistence::FiniteMaximum)
    ));
    let data = RegressionData::new(2, vec![1., 1., 1., 1. + 1e-10], vec![1_u64, 2]).unwrap();
    let a = data
        .assess_poisson_with_rank(&["a", "b"], ConeSearchOptions::default(), 1e-8)
        .unwrap();
    let b = data
        .assess_poisson_with_rank(&["a", "b"], ConeSearchOptions::default(), 1e-14)
        .unwrap();
    assert_eq!(a.design_rank().unwrap().numerical_rank, 1);
    assert_eq!(b.design_rank().unwrap().numerical_rank, 2);
    assert!(
        data.assess_poisson_with_rank(&["a", "b"], ConeSearchOptions::default(), f64::NAN)
            .is_err()
    );
    let data = RegressionData::new(1, vec![1., 1., 1.], vec![2., 2., 2.]).unwrap();
    let a = data.assess_gaussian(&["mean"], 1e-12, 1e-12).unwrap();
    let RegressionEvidence::Gaussian(g) = a.evidence() else {
        panic!("wrong family")
    };
    assert!(g.scale_candidate.is_none());
    assert!(g.issues.contains(
        &alea::distributions::modeling::regression::diagnostics::GaussianIssue::VanishingResidual
    ));
    assert!(a.search_options().is_none());
}
