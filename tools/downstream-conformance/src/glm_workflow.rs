//! Native point-estimation application: existence, identification and numerical
//! termination stay separate. No likelihood, gradient or optimizer implementation.
use alea::{
    analysis::{Provenance, write_fit},
    distributions::{
        modeling::regression::{
            RegressionData, bernoulli_logit_regression,
            diagnostics::{
                ConeSearchOptions, MleExistence, RecessionKind, RegressionAssessment,
                RegressionEvidence,
            },
            poisson_log_regression,
        },
        scalar::Bernoulli,
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

/// Both public native GLM batch factors are exercised.
pub const FAMILIES: [&str; 2] = ["logistic", "poisson"];
/// Every planned case remains in the cohort, including non-success outcomes.
pub const CASES: [&str; 8] = [
    "qualified",
    "budget",
    "cancelled",
    "evaluation_failure",
    "rank_deficient",
    "boundary",
    "unresolved",
    "zero_no_intercept",
];
const NAMES: [&str; 2] = ["coefficient0", "coefficient1"];

fn chain(error: &(dyn Error + 'static)) -> Vec<String> {
    let mut chain = vec![error.to_string()];
    let mut source = error.source();
    while let Some(error) = source {
        chain.push(error.to_string());
        source = error.source();
    }
    chain
}
fn existence(e: &RegressionAssessment) -> &MleExistence {
    match e.evidence() {
        RegressionEvidence::BernoulliLogit(e) | RegressionEvidence::PoissonLog(e) => e,
        _ => unreachable!("only GLM evidence was requested"),
    }
}
fn encode(e: &RegressionAssessment) -> Value {
    let existence = match existence(e) {
        MleExistence::FiniteMaximum => json!({"status":"finite_maximum"}),
        MleExistence::Unresolved => json!({"status":"unresolved"}),
        MleExistence::NoFiniteMaximum(w) => {
            json!({"status":"no_finite_maximum","kind":match w.kind(){
            RecessionKind::CompleteSeparation=>"complete_separation",RecessionKind::QuasiCompleteSeparation=>"quasi_complete_separation",RecessionKind::PoissonBoundary=>"poisson_boundary"},
            "direction":w.direction().coefficients().iter().map(ToString::to_string).collect::<Vec<_>>(),"improving_rows":w.improving_rows(),"flat_rows":w.flat_rows()})
        }
    };
    let rank = e.design_rank().map(|r|json!({"rows":r.rows,"columns":r.columns,"numerical_rank":r.numerical_rank,"relative_tolerance":r.relative_tolerance}));
    let search = e.search_options().map(|o|json!({"max_columns":o.max_columns,"max_rows":o.max_rows,"max_coefficient_bits":o.max_coefficient_bits}));
    json!({"coefficient_names":e.coefficient_names(),"existence":existence,"rank":rank,"search":search})
}

/// Writes one standalone alea-fit1 record with glm-workflow1 caller evidence.
/// Forecasts require finite-attainment evidence, numerical full rank and native
/// score termination. No uncertainty or posterior axes are manufactured.
/// # Errors
/// Rejects unknown cases/setup failures and propagates original export/I/O errors.
/// Native evaluation errors and last valid candidates are retained in the report.
pub fn write_case(family: &str, case: &str, out: &mut impl Write) -> Result<(), Box<dyn Error>> {
    if !FAMILIES.contains(&family) || !CASES.contains(&case) {
        return Err("unknown GLM workflow family/case".into());
    }
    let mut design: Vec<_> = (0..8)
        .flat_map(|i| {
            [
                1.,
                if case == "rank_deficient" {
                    1.
                } else {
                    f64::from(i >= 4)
                },
            ]
        })
        .collect();
    let mut binary = vec![false, false, false, true, false, true, true, true];
    let mut counts = vec![0_u64, 1, 2, 1, 1, 2, 3, 2];
    let mut offsets: Vec<_> = (0..8)
        .map(|i| {
            if family == "logistic" {
                if i < 4 { 0.25 } else { -0.5 }
            } else if i < 4 {
                std::f64::consts::LN_2
            } else {
                -std::f64::consts::LN_2
            }
        })
        .collect();
    if case == "boundary" {
        binary = (0..8).map(|i| i >= 4).collect();
        counts.fill(0);
    }
    if case == "zero_no_intercept" {
        design = vec![1., 0., -1., 0., 0., 1., 0., -1.];
        binary = vec![false; 4];
        counts = vec![0; 4];
        offsets = vec![0.; 4];
    }
    let search = ConeSearchOptions {
        max_rows: if case == "unresolved" { 0 } else { 16384 },
        max_columns: 32,
        max_coefficient_bits: 8192,
    };
    let mut b = ModelBuilder::new();
    let beta = [b.real(NAMES[0])?.expr(), b.real(NAMES[1])?.expr()];
    let (term, assessment, response) = if family == "logistic" {
        let data = RegressionData::new(2, design.clone(), binary.clone())?
            .with_offsets(offsets.clone())?;
        let assessment = data.assess_bernoulli_with_rank(&NAMES, search, 1e-12)?;
        (
            bernoulli_logit_regression(&mut b, &beta, data)?,
            assessment,
            json!(binary),
        )
    } else {
        let data = RegressionData::new(2, design.clone(), counts.clone())?
            .with_offsets(offsets.clone())?;
        let assessment = data.assess_poisson_with_rank(&NAMES, search, 1e-12)?;
        (
            poisson_log_regression(&mut b, &beta, data)?,
            assessment,
            json!(counts),
        )
    };
    b.likelihood("responses", term)?;
    let model = b.compile()?;
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let objective = model.select_likelihood()?.into_objective()?;
    let description = ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "Lebesgue in two independent real coefficients".into(),
        model: format!("native-{family}-workflow:v1"),
        data: format!("glm-workflow:{family}:{case}:v1"),
        source: "public-native-glm-workflow:v1".into(),
    };
    let provenance = Provenance {
        model: description.model.clone(),
        data: description.data.clone(),
        source: description.source.clone(),
        method: "native batch L-BFGS; separate exact-cone and SVD assessment".into(),
        settings: BTreeMap::from([
            ("family".into(), family.into()),
            ("case".into(), case.into()),
        ]),
    };
    let initial = if case == "evaluation_failure" {
        [1e308, 1e308]
    } else if case == "zero_no_intercept" {
        [0., 0.]
    } else {
        [0.1, -0.2]
    };
    let options = LbfgsOptions::new(200, if case == "budget" { 1 } else { 2000 })?
        .with_score_tolerance(1e-6)?;
    let (result, error) =
        match maximize_with_cancel(&objective, &initial, description, options, || {
            case == "cancelled"
        }) {
            Ok(r) => (r, None),
            Err(MaximizeError::Evaluation(e)) => (*e.result, Some(chain(&e.source))),
            Err(e) => return Err(e.into()),
        };
    let fit = NamedFit::new(names, result)?;
    let encoded = encode(&assessment);
    let admission = match existence(&assessment) {
        MleExistence::NoFiniteMaximum(_) => "no_finite_maximum",
        MleExistence::Unresolved => "existence_unresolved",
        MleExistence::FiniteMaximum => {
            if assessment
                .design_rank()
                .is_none_or(|r| r.numerical_rank != 2)
            {
                "nonidentified"
            } else if fit.estimates().is_none() {
                "no_candidate"
            } else if fit.result().termination != OptimizationTermination::ScoreTolerance {
                "not_stationary"
            } else {
                "admitted_point_fit"
            }
        }
    };
    let mut predictions = Vec::new();
    for (i, (x, offset)) in [0., 1., 0.5]
        .into_iter()
        .zip(if family == "poisson" {
            [
                2. * std::f64::consts::LN_2,
                -2. * std::f64::consts::LN_2,
                0.,
            ]
        } else {
            [0.; 3]
        })
        .enumerate()
    {
        let mean = if admission == "admitted_point_fit" {
            let beta = fit.estimates().ok_or("missing admitted estimates")?;
            let mut sum = CompensatedSum::default();
            sum.add(offset);
            sum.add_product(1., beta[0]);
            sum.add_product(x, beta[1]);
            let eta = sum.total();
            let mean = if family == "logistic" {
                Bernoulli::logit_log_prob(eta, true)?.exp()
            } else {
                eta.exp()
            };
            if !mean.is_finite() {
                return Err("nonrepresentable GLM conditional mean".into());
            }
            Some(mean)
        } else {
            None
        };
        predictions
            .push(json!({"id":format!("new-{i}"),"design":[1.,x],"offset":offset,"mean":mean}));
    }
    let evidence = json!({"format":"glm-workflow","schema_version":1,"family":family,"case":case,
        "assumptions":"independent_fixed_design_known_offsets_unconstrained_no_priors_weights_penalties",
        "training":{"rows":offsets.len(),"columns":2,"design":design,"response":response,"offsets":offsets},
        "assessment":encoded,"admission":admission,"native_evaluation_error":error,
        "uncertainty":{"status":"not_computed","reason":"no_explicit_likelihood_hvp_supplied"},
        "predictions":{"kind":"plug_in_conditional_mean","quantity":if family=="logistic" {"event_probability"} else {"expected_count"},"status":if admission=="admitted_point_fit" {"available"} else {"withheld"},"rows":predictions}});
    write_fit(out, &fit.with_assessment(evidence), &provenance, |a| {
        Ok(a.clone())
    })?;
    Ok(())
}
