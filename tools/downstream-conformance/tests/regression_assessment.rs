use alea::distributions::modeling::regression::{
    RegressionData,
    diagnostics::{DirectionAssessment, RecessionKind},
};
#[test]
fn typed_likelihood_checks_find_problem_rays_without_rejecting_the_model() {
    let data = RegressionData::new(1, vec![-1., 0., 1.], vec![false, true, true]).unwrap();
    let Some(w) = data.bernoulli_coordinate_recession().unwrap() else {
        panic!("expected quasiseparation")
    };
    assert_eq!(w.kind(), RecessionKind::QuasiCompleteSeparation);
    assert_eq!(w.flat_rows(), 1);
    assert!(matches!(
        data.bernoulli_recession(&[0.]).unwrap(),
        DirectionAssessment::NotWitness { .. }
    ));
    let zeros = RegressionData::new(1, vec![1., 1.], vec![0_u64, 0]).unwrap();
    assert!(zeros.poisson_coordinate_recession().unwrap().is_some());
}
#[cfg(feature = "faer")]
#[test]
fn gaussian_assessment_is_available_from_the_facade_with_explicit_policies() {
    use alea::distributions::modeling::regression::diagnostics::GaussianIssue;
    let data = RegressionData::new(2, vec![1., -1., 1., 0., 1., 1.], vec![0., 2., 2.]).unwrap();
    let a = data.gaussian_assessment(1e-12, 1e-12).unwrap();
    assert!(a.issues.is_empty());
    assert!((a.least_squares.coefficients[0] - 4. / 3.).abs() < 1e-12);
    assert!((a.least_squares.coefficients[1] - 1.).abs() < 1e-12);
    assert!((a.scale_candidate.unwrap() - (2_f64 / 9.).sqrt()).abs() < 1e-12);
    let p = RegressionData::new(1, vec![1., 1.], vec![2., 2.])
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert!(p.issues.contains(&GaussianIssue::VanishingResidual));
    assert!(p.scale_candidate.is_none());
}
#[cfg(feature = "faer")]
#[test]
fn thin_svd_matches_independent_numpy_lstsq_when_external_checks_are_enabled() {
    if std::env::var("ALEA_ANALYSIS_EXTERNAL").as_deref() != Ok("1") {
        return;
    }
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let cases = [
        (3, 2, vec![1., 1., 2., 2., 3., 3.], vec![2., 4., 7.], 1e-12),
        (2, 3, vec![1., 0., 1., 0., 1., 1.], vec![1., 2.], 1e-12),
        (2, 2, vec![1., 0., 0., 1e-12], vec![1., 1e-12], 1e-10),
    ];
    for (rows, columns, x, y, tolerance) in cases {
        let fit = alea::math::least_squares::solve(rows, columns, &x, &y, tolerance).unwrap();
        let payload = format!(
            "{{\"rows\":{rows},\"columns\":{columns},\"x\":{x:?},\"y\":{y:?},\"tolerance\":{tolerance},\"coefficients\":{:?},\"singular\":{:?},\"scale\":{},\"rank\":{},\"residual\":{}}}",
            fit.coefficients,
            fit.scaled_singular_values,
            fit.design_scale,
            fit.numerical_rank,
            fit.residual_norm
        );
        let script = r#"import json,sys,numpy as np
v=json.load(sys.stdin)
x=np.array(v['x']).reshape(v['rows'],v['columns']); y=np.array(v['y'])
beta,_,rank,s=np.linalg.lstsq(x,y,rcond=v['tolerance'])
np.testing.assert_allclose(v['coefficients'],beta,rtol=1e-11,atol=1e-12)
np.testing.assert_allclose(np.array(v['singular'])*v['scale'],s,rtol=1e-11,atol=1e-12)
assert v['rank']==rank
np.testing.assert_allclose(v['residual'],np.linalg.norm(y-x@beta),rtol=1e-10,atol=1e-12)
"#;
        let mut child =
            Command::new(std::env::var("ALEA_PYTHON").unwrap_or_else(|_| "python3".into()))
                .args(["-c", script])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn small_numerical_scores_do_not_overrule_existence_witnesses() {
    use alea::{
        distributions::modeling::regression::{bernoulli_logit_regression, poisson_log_regression},
        model::assembly::ModelBuilder,
        optimize::{
            LbfgsOptions, ObjectiveDescription, ObjectiveKind, OptimizationTermination, maximize,
        },
    };
    for bernoulli in [false, true] {
        let mut b = ModelBuilder::new();
        let beta = b.real("coefficient").unwrap();
        let f = if bernoulli {
            let data = RegressionData::new(1, vec![-1., 1.], vec![false, true]).unwrap();
            assert!(data.bernoulli_coordinate_recession().unwrap().is_some());
            bernoulli_logit_regression(&mut b, &[beta.expr()], data).unwrap()
        } else {
            let data = RegressionData::new(1, vec![1., 1.], vec![0_u64, 0]).unwrap();
            assert!(data.poisson_coordinate_recession().unwrap().is_some());
            poisson_log_regression(&mut b, &[beta.expr()], data).unwrap()
        };
        b.likelihood("data", f).unwrap();
        let target = b
            .compile()
            .unwrap()
            .select_likelihood()
            .unwrap()
            .into_objective()
            .unwrap();
        let r = maximize(
            &target,
            &[if bernoulli { 1000. } else { -1000. }],
            ObjectiveDescription {
                kind: ObjectiveKind::Likelihood,
                measure: "unconstrained coefficients".into(),
                model: "boundary example".into(),
                data: "test".into(),
                source: "conformance".into(),
            },
            LbfgsOptions::new(10, 100).unwrap(),
        )
        .unwrap();
        assert_eq!(r.termination, OptimizationTermination::ScoreTolerance); // numerical evidence only
    }
}
