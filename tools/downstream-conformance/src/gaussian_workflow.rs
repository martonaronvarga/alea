//! Small fixed-effect Gaussian application consumer, not another inference API.
//! Numerical termination, independent least-squares intervals and point forecasts
//! remain separate. No distributions, gradients or optimizer are implemented here.
use alea::{
    analysis::{Provenance, write_fit},
    distributions::modeling::regression::{
        RegressionData,
        diagnostics::{GaussianAssessment, GaussianIntervalOptions, GaussianIssue},
        normal_regression,
    },
    math::numeric::CompensatedSum,
    model::{assembly::ModelBuilder, prediction::NamedOutput},
    optimize::{
        LbfgsOptions, MaximizeError, NamedFit, ObjectiveDescription, ObjectiveKind,
        OptimizationTermination, maximize_with_cancel,
    },
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, io::Write};

/// Planned cases, including all non-success controls; no conditional denominator.
pub const CASES: [&str; 6] = [
    "qualified",
    "budget",
    "cancelled",
    "initial_transform",
    "rank_deficient",
    "perfect_fit",
];
const COMPARISON_TOLERANCE: f64 = 1e-5;

fn error_chain(error: &(dyn Error + 'static)) -> Vec<String> {
    let mut result = vec![error.to_string()];
    let mut source = error.source();
    while let Some(error) = source {
        result.push(error.to_string());
        source = error.source();
    }
    result
}
fn assessment(a: &GaussianAssessment) -> Value {
    let issues: Vec<_> = a
        .issues
        .iter()
        .map(|issue| match issue {
            GaussianIssue::NumericalRankDeficiency => "numerical_rank_deficiency",
            GaussianIssue::VanishingResidual => "vanishing_residual",
            GaussianIssue::ScaleNotRepresentable => "scale_not_representable",
        })
        .collect();
    json!({"numerical_rank":a.least_squares.numerical_rank,
           "residual_df":a.least_squares.residual_degrees_of_freedom,
           "coefficients":a.least_squares.coefficients,
           "residual_norm":a.least_squares.residual_norm,
           "mle_sigma":a.scale_candidate,"issues":issues})
}

/// Writes one standalone alea-fit report with an explicitly versioned caller
/// assessment. Interval estimates belong to the separate OLS estimator even when
/// native optimization stops early. Native forecasts require stationarity,
/// admitted intervals and agreement with that reference under the stated policy.
/// # Errors
/// Rejects unknown cases/setup failures, and preserves serialization/I/O errors.
/// Expected numerical-fit and interval failures are retained inside the report,
/// not converted into successful fits. Partial I/O requires discarding the output.
pub fn write_case(case: &str, writer: &mut impl Write) -> Result<(), Box<dyn Error>> {
    if !CASES.contains(&case) {
        return Err("unknown Gaussian workflow case".into());
    }
    let x: Vec<_> = (-2..=3).map(f64::from).collect();
    let offsets = vec![0.25, -0.5, 0.75, -1., 1.25, -1.5];
    let response: Vec<_> = x
        .iter()
        .zip([1., -2., 1., 1., -2., 1.])
        .zip(&offsets)
        .map(|((&x, e), &o)| 1.5 - 0.75 * x + o + if case == "perfect_fit" { 0. } else { e })
        .collect();
    let design: Vec<_> = x
        .iter()
        .flat_map(|&x| [1., if case == "rank_deficient" { 1. } else { x }])
        .collect();
    let data =
        RegressionData::new(2, design.clone(), response.clone())?.with_offsets(offsets.clone())?;
    let policy = GaussianIntervalOptions {
        confidence_level: 0.95,
        relative_rank_tolerance: 1e-12,
        relative_residual_tolerance: 1e-12,
        quantile_relative_tolerance: 1e-8,
    };
    let policy_json = json!({"confidence_level":policy.confidence_level,"relative_rank_tolerance":policy.relative_rank_tolerance,
        "relative_residual_tolerance":policy.relative_residual_tolerance,"quantile_relative_tolerance":policy.quantile_relative_tolerance});
    let intervals = data.gaussian_coefficient_intervals(policy);
    let interval_json = match &intervals {
        Ok(result) => {
            json!({"status":"available","estimator":"separate_fixed_design_least_squares",
            "options":policy_json,"assessment":assessment(&result.assessment),
            "residual_variance":result.residual_variance,"covariance":result.covariance,
            "standard_errors":result.standard_errors,"critical_value":result.critical_value,"bounds":result.bounds})
        }
        Err(error) => {
            json!({"status":"unavailable","estimator":"separate_fixed_design_least_squares",
            "options":policy_json,"assessment":assessment(&data.gaussian_assessment(1e-12,1e-12)?),
            "error_chain":error_chain(error)})
        }
    };
    let mut builder = ModelBuilder::new();
    let a = builder.real("intercept")?;
    let b = builder.real("slope")?;
    let sigma = builder.positive("sigma")?;
    let term = normal_regression(&mut builder, &[a.expr(), b.expr()], &sigma, data)?;
    builder.likelihood("responses", term)?;
    let model = builder.compile()?;
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let objective = model.select_likelihood()?.into_objective()?;
    let description = ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "Lebesgue in intercept, slope and positive sigma".into(),
        model: "gaussian-fixed-effect-workflow:v1".into(),
        data: format!("gaussian-workflow:{case}:v1"),
        source: "public-native-gaussian-workflow:v1".into(),
    };
    let provenance = Provenance {
        model: description.model.clone(),
        data: description.data.clone(),
        source: description.source.clone(),
        method: "native L-BFGS likelihood; separate SVD/Student-t coefficient intervals".into(),
        settings: BTreeMap::from([("case".into(), case.into())]),
    };
    let initial = [
        0.1,
        -0.2,
        if case == "initial_transform" {
            -1000.
        } else {
            0.
        },
    ];
    let options = LbfgsOptions::new(200, if case == "budget" { 1 } else { 2000 })?
        .with_score_tolerance(1e-6)?;
    let (result, backend_error) =
        match maximize_with_cancel(&objective, &initial, description, options, || {
            case == "cancelled"
        }) {
            Ok(result) => (result, None),
            Err(MaximizeError::Evaluation(error)) => {
                (*error.result, Some(error_chain(&error.source)))
            }
            Err(error) => return Err(error.into()),
        };
    let fit = NamedFit::new(names, result)?;
    let status = if fit.estimates().is_none() {
        "no_candidate"
    } else if fit.result().termination != OptimizationTermination::ScoreTolerance {
        "not_stationary"
    } else if let Ok(reference) = &intervals {
        let native = fit.estimates().ok_or("missing checked native estimates")?;
        let mut expected = reference.assessment.least_squares.coefficients.clone();
        expected.push(
            reference
                .assessment
                .scale_candidate
                .ok_or("missing admitted ML scale")?,
        );
        if native
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() <= COMPARISON_TOLERANCE * (1. + b.abs()))
        {
            "consistent"
        } else {
            "mismatch"
        }
    } else {
        "uncertainty_unavailable"
    };
    let mut predictions = Vec::new();
    for (i, (x, offset)) in [(-3., 0.5), (0., -0.25), (4., 1.)].into_iter().enumerate() {
        let mean = if status == "consistent" {
            let beta = fit.estimates().ok_or("missing admitted native estimates")?;
            let mut sum = CompensatedSum::default();
            sum.add(offset);
            sum.add_product(1., beta[0]);
            sum.add_product(x, beta[1]);
            let mean = sum.total();
            if !mean.is_finite() {
                return Err("nonrepresentable plug-in conditional mean".into());
            }
            Some(mean)
        } else {
            None
        };
        predictions
            .push(json!({"id":format!("new-{i}"),"design":[1.,x],"offset":offset,"mean":mean}));
    }
    let evidence = json!({"format":"gaussian-workflow","schema_version":1,"case":case,
        "assumptions":"independent_homoskedastic_gaussian_fixed_design_known_offsets_unconstrained_no_priors_weights_penalties",
        "coefficient_names":["intercept","slope"],
        "training":{"rows":6,"columns":2,"design":design,"response":response,"offsets":offsets},
        "native_evaluation_error":backend_error,"intervals":interval_json,
        "comparison":{"status":status,"relative_tolerance":COMPARISON_TOLERANCE},
        "predictions":{"kind":"plug_in_conditional_mean","status":if status=="consistent" {"available"} else {"withheld"},"rows":predictions}});
    write_fit(
        writer,
        &fit.with_assessment(evidence),
        &provenance,
        |evidence| Ok(evidence.clone()),
    )?;
    Ok(())
}
