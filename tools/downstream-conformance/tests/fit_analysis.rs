//! Native public-model MLE outcomes -> JSON -> independent strict Python consumer.
use alea::{
    analysis::{Provenance, write_fit},
    distributions::modeling::regression::{RegressionData, normal_regression},
    model::{assembly::ModelBuilder, prediction::NamedOutput},
    optimize::{
        LbfgsOptions, MaximizeError, NamedFit, ObjectiveDescription, ObjectiveKind,
        OptimizationTermination, maximize_with_cancel,
    },
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn successful_exhausted_cancelled_and_failed_native_fits_round_trip_without_posterior_axes() {
    for case in 0..4 {
        let mut b = ModelBuilder::new();
        let mean = b.real("mean").unwrap();
        let sigma = b.positive("σ").unwrap();
        let data = RegressionData::new(1, vec![1.; 3], vec![0., 1., 2.]).unwrap();
        let term = normal_regression(&mut b, &[mean.expr()], &sigma, data).unwrap();
        b.likelihood("observations", term).unwrap();
        let model = b.compile().unwrap();
        let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap();
        let objective = model.select_likelihood().unwrap().into_objective().unwrap();
        let description = ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "Lebesgue in mean and sigma".into(),
            model: "normal-location-scale:v1".into(),
            data: "observations:0,1,2".into(),
            source: "fit-analysis-conformance:v1".into(),
        };
        let provenance = Provenance {
            model: description.model.clone(),
            data: description.data.clone(),
            source: description.source.clone(),
            method: "L-BFGS".into(),
            settings: BTreeMap::from([("case".into(), case.to_string())]),
        };
        let initial = if case == 3 { [0., -1000.] } else { [0., 0.] };
        let options = LbfgsOptions::new(300, if case == 1 { 1 } else { 3000 })
            .unwrap()
            .with_score_tolerance(1e-6)
            .unwrap();
        let (result, error) =
            match maximize_with_cancel(&objective, &initial, description, options, || case == 2) {
                Ok(r) => (r, None),
                Err(MaximizeError::Evaluation(error)) => {
                    (*error.result, Some(error.source.to_string()))
                }
                Err(e) => panic!("unexpected admission error: {e}"),
            };
        let expected = [
            OptimizationTermination::ScoreTolerance,
            OptimizationTermination::EvaluationLimit,
            OptimizationTermination::Cancelled,
            OptimizationTermination::EvaluationFailed,
        ][case];
        assert_eq!(result.termination, expected);
        let fit = NamedFit::new(names, result).unwrap().with_assessment(error);
        if case == 0 {
            assert!((fit.estimate("mean").unwrap() - 1.).abs() < 1e-5);
            assert!((fit.estimate("σ").unwrap() - (2_f64 / 3.).sqrt()).abs() < 1e-5);
        }
        let mut bytes = Vec::new();
        write_fit(&mut bytes, &fit, &provenance, |error| {
            Ok(json!({"backend_error":error,"uncertainty":"not computed"}))
        })
        .unwrap();
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
        let mut child = Command::new("python3")
            .env("PYTHONPATH", python)
            .arg("-c")
            .arg(
                r#"
import json, math, sys
from fit_analysis import validate
from analysis import unique_object, reject_constant
r=validate(json.load(sys.stdin,object_pairs_hook=unique_object,parse_constant=reject_constant))
case=int(r['provenance']['settings']['case'])
assert r['kind']=='point_estimate' and 'chains' not in r and 'draws' not in r
assert r['objective']['measure']=='Lebesgue in mean and sigma'
assert [p['name'] for p in r['parameters']]==['mean','σ']
assert r['assessment']['uncertainty']=='not computed'
if case==0:
    assert r['numerical']['termination']=='score_tolerance'
    mean,sigma=r['candidate']['constrained_estimates']
    assert abs(mean-1)<1e-5 and abs(sigma-math.sqrt(2/3))<1e-5
    expected=-1.5*(math.log(2*math.pi)+1+math.log(2/3))
    assert abs(r['candidate']['value']-expected)<1e-9
elif case==1:
    assert r['numerical']['termination']=='evaluation_limit'
    assert r['candidate']['free_position']==[0.,0.]
elif case==2:
    assert r['candidate'] is None and r['numerical']['evaluations']==0
else:
    assert r['candidate'] is None and r['numerical']['termination']=='evaluation_failed'
    assert r['assessment']['backend_error']
"#,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "case {case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
