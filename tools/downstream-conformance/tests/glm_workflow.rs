#![cfg(feature = "faer")]
use alea::{
    distributions::modeling::regression::{
        RegressionData, bernoulli_logit_regression, poisson_log_regression,
    },
    model::assembly::ModelBuilder,
    target::LogDensityGradient,
};
use alea_downstream_conformance::glm_workflow::{CASES, FAMILIES, write_case};
use serde_json::Value;
use std::{
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn native_sixteen_case_cohort_preserves_numerical_existence_and_prediction_evidence() {
    let mut bytes = Vec::new();
    for family in FAMILIES {
        for case in CASES {
            write_case(family, case, &mut bytes).unwrap();
        }
    }
    assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 16);
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let mut child=Command::new("python3").env("PYTHONPATH",script).arg("-c").arg(r#"
import copy,json,sys
from glm_workflow import read,validate
text=sys.stdin.read()
records=read(text)
lines=text.splitlines()
for malformed in ('\n'.join(lines[:-1]),text+lines[0]+'\n','\n'.join([lines[0]]*16),text+'\n',lines[0][:-1],lines[0].replace('"schema_version":1','"schema_version":1,"schema_version":1')):
    try:read(malformed)
    except ValueError:pass
    else:raise AssertionError('accepted incomplete/duplicate/blank/truncated report')
admitted=[(r['assessment']['family'],r['assessment']['case']) for r in records if r['assessment']['admission']=='admitted_point_fit']
assert admitted==[(f,c) for f in ('logistic','poisson') for c in ('qualified','zero_no_intercept')]
for r in records:
    a=r['assessment']
    if a['case']=='boundary':
        # These fixed cases produce small scores without a finite maximizer.
        assert r['numerical']['termination']=='score_tolerance'
    if a['case']=='evaluation_failure':
        assert len(a['native_evaluation_error'])>=2
    for mutation in range(8):
        bad=copy.deepcopy(r);b=bad['assessment']
        if mutation==0:b['admission']='nonexistent_admission'
        if mutation==1:b['uncertainty']['covariance']=[1,0,0,1]
        if mutation==2:b['predictions']['kind']='posterior'
        if mutation==3:b['assessment']['rank']['relative_tolerance']=True
        if mutation==4:b['training']['offsets'][0]+=1
        if mutation==5:b['native_evaluation_error']=None if a['native_evaluation_error'] else ['fabricated error']
        if mutation==6:b['predictions']['rows'][0]['id']='new-1'
        if mutation==7:
            if a['case']=='boundary':b['assessment']['existence']['direction']=['0','0']
            elif a['predictions']['status']=='available':b['predictions']['rows'][0]['mean']=999
            else:b['predictions']['rows'][0]['mean']=0
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError(f"accepted mutation {a['family']}/{a['case']}/{mutation}")
print(json.dumps({'planned':16,'completed':len(records),'admitted':admitted}))
"#).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{}", String::from_utf8_lossy(&output.stdout));
    let mut again = Vec::new();
    write_case("logistic", "qualified", &mut again).unwrap();
    assert_eq!(&bytes[..again.len()], again.as_slice(), "same build replay");
    let mut empty = Vec::new();
    assert!(write_case("bad", "qualified", &mut empty).is_err());
    assert!(write_case("logistic", "bad", &mut empty).is_err());
    assert!(empty.is_empty());
}

#[test]
fn model_level_values_and_scores_match_independent_reference_before_any_sampler() {
    for family in FAMILIES {
        for case in [
            "qualified",
            "rank_deficient",
            "boundary",
            "zero_no_intercept",
        ] {
            let mut bytes = Vec::new();
            write_case(family, case, &mut bytes).unwrap();
            let report: Value = serde_json::from_slice(&bytes).unwrap();
            let t = &report["assessment"]["training"];
            let design: Vec<_> = t["design"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let offsets: Vec<_> = t["offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let y: Vec<_> = t["response"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    if family == "logistic" {
                        f64::from(v.as_bool().unwrap())
                    } else {
                        v.as_f64().unwrap()
                    }
                })
                .collect();
            let mut b = ModelBuilder::new();
            let beta = [
                b.real("coefficient0").unwrap().expr(),
                b.real("coefficient1").unwrap().expr(),
            ];
            let term = if family == "logistic" {
                bernoulli_logit_regression(
                    &mut b,
                    &beta,
                    RegressionData::new(2, design.clone(), y.iter().map(|&v| v > 0.).collect())
                        .unwrap()
                        .with_offsets(offsets.clone())
                        .unwrap(),
                )
                .unwrap()
            } else {
                let counts = t["response"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect();
                poisson_log_regression(
                    &mut b,
                    &beta,
                    RegressionData::new(2, design.clone(), counts)
                        .unwrap()
                        .with_offsets(offsets.clone())
                        .unwrap(),
                )
                .unwrap()
            };
            b.likelihood("responses", term).unwrap();
            let objective = b
                .compile()
                .unwrap()
                .select_likelihood()
                .unwrap()
                .into_objective()
                .unwrap();
            let reference = |q: [f64; 2]| -> f64 {
                y.iter()
                    .enumerate()
                    .map(|(i, &y)| {
                        let eta = offsets[i] + design[2 * i] * q[0] + design[2 * i + 1] * q[1];
                        if family == "logistic" {
                            let z = if y > 0. { eta } else { -eta };
                            if z >= 0. {
                                -(-z).exp().ln_1p()
                            } else {
                                z - z.exp().ln_1p()
                            }
                        } else {
                            let k = t["response"][i].as_u64().unwrap();
                            let log_factorial = (1..=k).map(|v| (v as f64).ln()).sum::<f64>();
                            y * eta - eta.exp() - log_factorial
                        }
                    })
                    .sum()
            };
            for q in [[0.1, -0.2], [-4., 3.], [4., -3.]] {
                let mut score = [0.; 2];
                let value = objective.logp_grad(&q, &mut score).unwrap();
                let truth = reference(q);
                assert!((value - truth).abs() < 1e-11 * (1. + truth.abs()));
                for i in 0..2 {
                    let (mut p, mut m) = (q, q);
                    p[i] += 1e-5;
                    m[i] -= 1e-5;
                    let fd = (reference(p) - reference(m)) / 2e-5;
                    assert!(
                        (score[i] - fd).abs() < 1e-7 * (1. + fd.abs()),
                        "{family}/{case}/{i}: {} vs {fd}",
                        score[i]
                    );
                }
            }
        }
    }
}

#[test]
fn original_export_io_error_is_not_replaced_by_a_success_report() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::from_raw_os_error(28))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let e = write_case("poisson", "qualified", &mut Broken).unwrap_err();
    let Some(alea::analysis::AnalysisError::Io(source)) =
        e.downcast_ref::<alea::analysis::AnalysisError>()
    else {
        panic!("original I/O error lost")
    };
    assert_eq!(source.raw_os_error(), Some(28));
}
