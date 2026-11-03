#![cfg(feature = "analysis")]
use alea_core::{
    model::{assembly::Coordinate, prediction::NamedOutput},
    transform::{ParameterLayout, Transform},
};
use alea_distributions::gaussian::StandardNormalTarget;
use alea_runtime::{
    analysis::{AnalysisError, Provenance, write_fit},
    optimize::{
        LbfgsOptions, LineSearchFailure, NamedFit, ObjectiveDescription, ObjectiveKind,
        OptimizationCandidate, OptimizationResult, OptimizationTermination, maximize_with_cancel,
    },
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{self, Write},
    panic::{AssertUnwindSafe, catch_unwind},
};

fn provenance() -> Provenance {
    Provenance {
        model: "model:fit".into(),
        data: "data:fit".into(),
        source: "source:fit".into(),
        method: "L-BFGS".into(),
        settings: BTreeMap::from([("seed".into(), "18446744073709551615".into())]),
    }
}
fn result() -> OptimizationResult {
    let p = provenance();
    OptimizationResult {
        dimension: 2,
        options: LbfgsOptions::new(10, 100)
            .unwrap()
            .with_scales(vec![2., 0.5])
            .unwrap(),
        objective: ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "simplex Hausdorff".into(),
            model: p.model,
            data: p.data,
            source: p.source,
        },
        candidate: Some(OptimizationCandidate {
            position: vec![-0., 0.25],
            value: f64::MAX,
            gradient: vec![-0., f64::from_bits(1)],
        }),
        termination: OptimizationTermination::ScoreTolerance,
        line_search_failure: None,
        iterations: 0,
        evaluations: 1,
        domain_trials: 0,
        skipped_updates: 0,
    }
}
fn fit(r: OptimizationResult) -> NamedFit {
    let coordinates = (0..3)
        .map(|i| Coordinate {
            name: format!("weights,\"λ\"\n[{i}]"),
            dimension: Some("category".into()),
            coordinate: Some(i.to_string()),
            block: Some("weights".into()),
        })
        .collect::<Vec<_>>();
    NamedFit::new(
        NamedOutput::new(
            coordinates.into(),
            ParameterLayout::new([Transform::simplex(3).unwrap()]).unwrap(),
            (),
        )
        .unwrap(),
        r,
    )
    .unwrap()
}
fn record(r: OptimizationResult) -> Value {
    let f = fit(r).with_assessment(BTreeMap::from([("existence", "not assessed")]));
    let mut bytes = Vec::new();
    write_fit(&mut bytes, &f, &provenance(), |e| Ok(json!(e))).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
#[test]
fn physical_estimates_free_scores_provenance_and_assessment_round_trip_without_draw_axes() {
    let f = fit(result());
    let r = record(result());
    assert_eq!(r["format"], "alea-fit");
    assert_eq!(r["schema_version"], 1);
    assert_eq!(r["kind"], "point_estimate");
    assert_eq!(r["objective"]["kind"], "likelihood");
    assert_eq!(r["objective"]["measure"], "simplex Hausdorff");
    assert_eq!(
        r["parameters"][0]["name"],
        f.parameters().coordinates()[0].name
    );
    assert_eq!(r["unconstrained_dimension"], 2);
    assert_eq!(
        r["candidate"]["constrained_estimates"],
        json!(f.estimates())
    );
    assert_eq!(
        r["candidate"]["free_position"][0]
            .as_f64()
            .unwrap()
            .to_bits(),
        (-0_f64).to_bits()
    );
    assert_eq!(
        r["candidate"]["free_gradient"][1]
            .as_f64()
            .unwrap()
            .to_bits(),
        1
    );
    assert_eq!(
        r["candidate"]["value"].as_f64().unwrap().to_bits(),
        f64::MAX.to_bits()
    );
    assert_eq!(r["assessment"]["existence"], "not assessed");
    assert_eq!(r["provenance"]["settings"]["seed"], "18446744073709551615");
    for absent in [
        "chains",
        "draws",
        "standard_errors",
        "covariance",
        "converged",
    ] {
        assert!(r.get(absent).is_none());
    }
    assert_eq!(r["numerical"]["options"]["scales"], json!([2., 0.5]));
}
#[test]
fn objective_kinds_do_not_change_measure_and_wide_counters_remain_integers() {
    for (kind, label) in [
        (ObjectiveKind::Likelihood, "likelihood"),
        (ObjectiveKind::Posterior, "posterior"),
        (ObjectiveKind::Other, "other"),
    ] {
        let mut r = result();
        r.objective.kind = kind;
        let record = record(r);
        assert_eq!(record["objective"]["kind"], label);
        assert_eq!(record["objective"]["measure"], "simplex Hausdorff");
    }
    #[cfg(target_pointer_width = "64")]
    {
        let mut r = result();
        r.options = LbfgsOptions::new(10, usize::MAX).unwrap();
        r.evaluations = (1_usize << 53) + 1;
        r.domain_trials = usize::MAX;
        let record = record(r);
        assert_eq!(
            record["numerical"]["evaluations"].as_u64(),
            Some((1_u64 << 53) + 1)
        );
        assert_eq!(
            record["numerical"]["domain_trials"].as_u64(),
            Some(u64::MAX)
        );
    }
}
#[test]
fn every_termination_and_both_search_failures_remain_distinct() {
    for (term, label) in [
        (OptimizationTermination::ScoreTolerance, "score_tolerance"),
        (OptimizationTermination::IterationLimit, "iteration_limit"),
        (OptimizationTermination::EvaluationLimit, "evaluation_limit"),
        (OptimizationTermination::Cancelled, "cancelled"),
        (
            OptimizationTermination::LineSearchFailed,
            "line_search_failed",
        ),
        (
            OptimizationTermination::NumericalDirection,
            "numerical_direction",
        ),
        (
            OptimizationTermination::EvaluationFailed,
            "evaluation_failed",
        ),
    ] {
        let mut r = result();
        r.termination = term;
        if term == OptimizationTermination::IterationLimit {
            r.iterations = 10;
            r.evaluations = 11;
        }
        if term == OptimizationTermination::EvaluationLimit {
            r.evaluations = 100;
        }
        if term == OptimizationTermination::LineSearchFailed {
            r.line_search_failure = Some(LineSearchFailure::BacktrackLimit);
        }
        assert_eq!(record(r)["numerical"]["termination"], label);
    }
    for (reason, label) in [
        (LineSearchFailure::BacktrackLimit, "backtrack_limit"),
        (
            LineSearchFailure::NoRepresentableStep,
            "no_representable_step",
        ),
    ] {
        let mut r = result();
        r.termination = OptimizationTermination::LineSearchFailed;
        r.line_search_failure = Some(reason);
        assert_eq!(record(r)["numerical"]["line_search_failure"], label);
    }
    let mut r = result();
    r.termination = OptimizationTermination::EvaluationFailed;
    r.candidate = None;
    assert!(record(r)["candidate"].is_null());
}
#[test]
fn actual_preinitialization_cancellation_has_no_candidate_or_invented_estimate() {
    let r = maximize_with_cancel(
        &StandardNormalTarget::new(2),
        &[0., 0.],
        result().objective,
        LbfgsOptions::new(3, 10).unwrap(),
        || true,
    )
    .unwrap();
    let r = record(r);
    assert!(r["candidate"].is_null());
    assert_eq!(r["numerical"]["evaluations"], 0);
    assert_eq!(r["numerical"]["termination"], "cancelled");
}
#[test]
fn malformed_stopping_policy_counts_and_identity_write_nothing() {
    for i in 0..12 {
        let mut r = result();
        match i {
            0 => r.line_search_failure = Some(LineSearchFailure::BacktrackLimit),
            1 => r.iterations = 11,
            2 => r.evaluations = 101,
            3 => r.skipped_updates = 1,
            4 => r.evaluations = 0,
            5 => r.candidate.as_mut().unwrap().gradient[0] = 1.,
            6 => r.termination = OptimizationTermination::IterationLimit,
            7 => r.termination = OptimizationTermination::EvaluationLimit,
            8 => r.options = r.options.with_scales(vec![1.]).unwrap(),
            9 => r.objective.measure.clear(),
            10 => {
                r.candidate = None;
                r.termination = OptimizationTermination::Cancelled;
            }
            _ => {
                r.candidate = None;
                r.termination = OptimizationTermination::EvaluationFailed;
                r.evaluations = 2;
            }
        }
        let mut bytes = Vec::new();
        assert!(
            write_fit(&mut bytes, &fit(r), &provenance(), |_| panic!(
                "assessment must not run"
            ))
            .is_err(),
            "case {i}"
        );
        assert!(bytes.is_empty());
    }
    for i in 0..5 {
        let mut p = provenance();
        match i {
            0 => p.model.push('x'),
            1 => p.data.push('x'),
            2 => p.source.push('x'),
            3 => p.method.clear(),
            _ => {
                p.settings.insert("".into(), "bad".into());
            }
        }
        let mut bytes = Vec::new();
        assert!(write_fit(&mut bytes, &fit(result()), &p, |_| Ok(Value::Null)).is_err());
        assert!(bytes.is_empty());
    }
}
#[test]
fn assessment_errors_and_panics_leave_stream_untouched() {
    let mut bytes = Vec::new();
    let f = fit(result());
    assert!(matches!(
        write_fit(&mut bytes, &f, &provenance(), |_| Err(
            AnalysisError::Invalid("assessment failed")
        )),
        Err(AnalysisError::Invalid("assessment failed"))
    ));
    assert!(bytes.is_empty());
    assert!(
        catch_unwind(AssertUnwindSafe(|| write_fit(
            &mut bytes,
            &f,
            &provenance(),
            |_| panic!("assessment panic")
        )))
        .is_err()
    );
    assert!(bytes.is_empty());
}
struct Fault {
    bytes: Vec<u8>,
    flush: bool,
}
impl Write for Fault {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.flush {
            self.bytes.extend(bytes);
            Ok(bytes.len())
        } else if self.bytes.is_empty() {
            self.bytes.extend(&bytes[..3]);
            Ok(3)
        } else {
            Err(io::Error::other("write fault"))
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("flush fault"))
    }
}
#[test]
fn original_partial_write_and_flush_errors_propagate() {
    for flush in [false, true] {
        let mut writer = Fault {
            bytes: vec![],
            flush,
        };
        let error = write_fit(&mut writer, &fit(result()), &provenance(), |_| {
            Ok(Value::Null)
        })
        .unwrap_err();
        assert!(
            matches!(error,AnalysisError::Io(ref e) if e.to_string()==if flush {"flush fault"} else {"write fault"})
        );
        assert!(!writer.bytes.is_empty());
    }
}
