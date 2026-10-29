#![cfg(feature = "analysis")]
use alea_core::{
    model::{
        assembly::{ModelBuilder, PointwiseLikelihood, PredictiveUnit},
        prediction::NamedOutput,
    },
    transform::{ParameterLayout, Transform},
};
use alea_runtime::{
    analysis::{AnalysisError, AnalysisSpec, AnalysisWriter, ChainStatus, Provenance},
    output::{ChainId, Draw, Phase, SCHEMA_VERSION, TransitionStats},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
fn provenance() -> Provenance {
    Provenance {
        model: "model:example".into(),
        data: "data:example".into(),
        source: "revision/toolchain/features".into(),
        method: "NUTS".into(),
        settings: BTreeMap::from([("seed".into(), "7".into())]),
    }
}
fn spec() -> AnalysisSpec {
    let mut b = ModelBuilder::new();
    let x = b.positive("scale,\"λ\"\r\nnext").unwrap().expr();
    b.likelihood("joint\nunit", x).unwrap();
    let p = PointwiseLikelihood::new(
        b.compile().unwrap(),
        vec![PredictiveUnit {
            name: "joint\nunit".into(),
            factors: vec!["joint\nunit".into()],
            observations: vec!["row1".into(), "row2".into()],
            conditioned_on: vec!["scale".into()],
            integrated_out: vec![],
        }],
    )
    .unwrap();
    let mut spec =
        AnalysisSpec::from_likelihood(p, provenance(), vec![ChainId(9), ChainId(u64::MAX)])
            .unwrap();
    spec.save_unconstrained = true;
    spec.predictions = vec!["predict,\"λ\"".into()];
    spec
}
fn draw<'a>(
    chain: u64,
    phase: Phase,
    index: usize,
    q: &'a [f64],
    stats: &'a TransitionStats,
) -> Draw<'a, ()> {
    Draw {
        schema_version: SCHEMA_VERSION,
        chain: ChainId(chain),
        phase,
        index,
        position: q,
        transition: &(),
        stats: *stats,
    }
}
fn finish(writer: AnalysisWriter<Vec<u8>>) -> (Vec<u8>, Value) {
    let mut json = Vec::new();
    let csv = writer
        .finish(
            &mut json,
            &[
                (ChainId(9), ChainStatus::Completed),
                (
                    ChainId(u64::MAX),
                    ChainStatus::Failed("never initialized".into()),
                ),
            ],
        )
        .unwrap();
    (csv, serde_json::from_slice(&json).unwrap())
}
#[test]
fn named_rows_preserve_physical_values_absence_failures_and_planned_empty_chains() {
    let mut spec = spec();
    spec.units
        .insert("parameter:scale,\"λ\"\r\nnext".into(), "seconds".into());
    let mut w = AnalysisWriter::new(Vec::new(), spec).unwrap();
    let stats = TransitionStats {
        acceptance_probability: 0.75,
        energy: Some(-0.),
        divergent: Some(false),
        ..TransitionStats::default()
    };
    w.write_draw(&draw(9, Phase::Warmup, 40, &[2_f64.ln()], &stats), &[-0.])
        .unwrap();
    w.write_draw(
        &draw(9, Phase::Sampling, 80, &[2_f64.ln()], &stats),
        &[f64::from_bits(1)],
    )
    .unwrap();
    w.write_failure(
        ChainId(9),
        Phase::Sampling,
        81,
        "numerical failure, \"detail\"\r\nsecond line",
    )
    .unwrap();
    let (csv, m) = finish(w);
    assert_eq!(m["kind"], "sampling");
    assert_eq!(m["chains"][0]["sampling"]["rows"], 2);
    assert_eq!(m["chains"][0]["sampling"]["failures"], 1);
    assert_eq!(m["chains"][1]["sampling"]["first"], Value::Null);
    assert_eq!(m["chains"][1]["id"], u64::MAX);
    assert_eq!(m["predictive_units"][0]["observations"][1], "row2");
    let text = String::from_utf8(csv).unwrap();
    assert!(text.contains("\"parameter:scale,\"\"λ\"\"\r\nnext\""));
    assert!(text.contains("\"failure\",\"numerical failure, \"\"detail\"\"\r\nsecond line\""));
    assert!(text.contains("\"-0\""));
    assert!(text.contains("\"2\""));
}
#[test]
fn invalid_payloads_are_counted_as_failures_while_invalid_keys_are_rejected() {
    let mut w = AnalysisWriter::new(Vec::new(), spec()).unwrap();
    let stats = TransitionStats::default();
    for (index, q) in [&[f64::NAN][..], &[-1000.][..], &[][..]]
        .into_iter()
        .enumerate()
    {
        assert!(
            w.write_draw(&draw(9, Phase::Sampling, index + 4, q, &stats), &[0.])
                .is_err()
        );
    }
    let invalid = TransitionStats {
        acceptance_probability: f64::NAN,
        ..stats
    };
    assert!(
        w.write_draw(&draw(9, Phase::Sampling, 7, &[0.], &invalid), &[0.])
            .is_err()
    );
    assert!(
        w.write_draw(&draw(10, Phase::Sampling, 8, &[0.], &stats), &[0.])
            .is_err()
    );
    w.write_failure(ChainId(9), Phase::Sampling, 8, "failed evaluation")
        .unwrap();
    assert!(
        w.write_draw(&draw(9, Phase::Sampling, 8, &[0.], &stats), &[0.])
            .is_err()
    );
    assert!(
        w.write_draw(&draw(9, Phase::Sampling, 10, &[0.], &stats), &[0.])
            .is_err()
    );
    assert!(
        w.write_draw(&draw(9, Phase::Warmup, 0, &[0.], &stats), &[0.])
            .is_err()
    );
    w.write_draw(&draw(9, Phase::Sampling, 9, &[0.], &stats), &[0.])
        .unwrap();
    let (_, m) = finish(w);
    assert_eq!(m["chains"][0]["sampling"]["first"], 4);
    assert_eq!(m["chains"][0]["sampling"]["last"], 9);
    assert_eq!(m["chains"][0]["sampling"]["failures"], 5);
    assert_eq!(m["chains"][0]["sampling"]["rows"], 6);
}
#[test]
fn metadata_and_final_outcomes_are_validated_without_inventing_results() {
    for case in 0..4 {
        let mut s = spec();
        match case {
            0 => s.chains.push(ChainId(9)),
            1 => s.provenance.data.clear(),
            2 => s.predictions.push(s.predictions[0].clone()),
            _ => {
                s.units.insert("unknown".into(), "unit".into());
            }
        }
        let mut bytes = Vec::new();
        assert!(AnalysisWriter::new(&mut bytes, s).is_err());
        assert!(bytes.is_empty());
    }
    let mut s = spec();
    let mut b = ModelBuilder::new();
    let _ = b.real("different").unwrap();
    let f = b.constant(0.).unwrap();
    b.factor(f).unwrap();
    let m = b.compile().unwrap();
    s.parameters = NamedOutput::new(m.coordinates(), m.layout().clone(), ()).unwrap();
    assert!(AnalysisWriter::new(Vec::new(), s).is_err());
    for statuses in [
        vec![],
        vec![
            (ChainId(9), ChainStatus::Completed),
            (ChainId(9), ChainStatus::Completed),
        ],
        vec![
            (ChainId(9), ChainStatus::Completed),
            (ChainId(u64::MAX), ChainStatus::Failed(String::new())),
        ],
    ] {
        assert!(
            AnalysisWriter::new(Vec::new(), spec())
                .unwrap()
                .finish(Vec::new(), &statuses)
                .is_err()
        );
    }
}
struct Switch {
    failed: Arc<AtomicBool>,
}
impl Write for Switch {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failed.load(Ordering::Relaxed) {
            Err(io::Error::other("injected I/O"))
        } else {
            Ok(bytes.len())
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.failed.load(Ordering::Relaxed) {
            Err(io::Error::other("injected flush"))
        } else {
            Ok(())
        }
    }
}
#[test]
fn failed_writes_poison_the_stream_and_prevent_a_manifest() {
    let failed = Arc::new(AtomicBool::new(false));
    let mut w = AnalysisWriter::new(
        Switch {
            failed: Arc::clone(&failed),
        },
        spec(),
    )
    .unwrap();
    failed.store(true, Ordering::Relaxed);
    // A long quoted failure forces the bounded I/O buffer to flush.
    assert!(matches!(
        w.write_failure(ChainId(9), Phase::Sampling, 0, &"x".repeat(20_000)),
        Err(AnalysisError::Io(_))
    ));
    failed.store(false, Ordering::Relaxed);
    assert!(matches!(
        w.write_failure(ChainId(9), Phase::Sampling, 0, "retry"),
        Err(AnalysisError::Poisoned)
    ));
    let mut manifest = Vec::new();
    assert!(matches!(
        w.finish(&mut manifest, &[]),
        Err(AnalysisError::Poisoned)
    ));
    assert!(manifest.is_empty());
    let w = AnalysisWriter::new(
        Switch {
            failed: Arc::clone(&failed),
        },
        spec(),
    )
    .unwrap();
    failed.store(true, Ordering::Relaxed);
    let mut manifest = Vec::new();
    assert!(
        w.finish(
            &mut manifest,
            &[
                (ChainId(9), ChainStatus::Completed),
                (ChainId(u64::MAX), ChainStatus::Completed)
            ]
        )
        .is_err()
    );
    assert!(manifest.is_empty());
}
#[test]
fn free_coordinates_use_full_round_trip_decimal_and_no_unneeded_jacobian() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    b.factor(x).unwrap();
    let m = b.compile().unwrap();
    let params = NamedOutput::new(
        m.coordinates(),
        ParameterLayout::new(vec![Transform::identity(1).unwrap()]).unwrap(),
        (),
    )
    .unwrap();
    let mut s = AnalysisSpec::new(params, provenance(), vec![ChainId(9)]);
    s.save_unconstrained = true;
    let mut w = AnalysisWriter::new(Vec::new(), s).unwrap();
    let stats = TransitionStats::default();
    for (i, x) in [
        -0.,
        f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        1.0000000000000002,
    ]
    .iter()
    .enumerate()
    {
        w.write_draw(&draw(9, Phase::Sampling, i, &[*x], &stats), &[])
            .unwrap();
    }
    let mut json = Vec::new();
    let csv = w
        .finish(&mut json, &[(ChainId(9), ChainStatus::Completed)])
        .unwrap();
    let text = String::from_utf8(csv).unwrap();
    for (line, x) in text.lines().skip(1).zip([
        -0.,
        f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        1.0000000000000002,
    ]) {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(
            fields[5]
                .trim_matches('"')
                .parse::<f64>()
                .unwrap()
                .to_bits(),
            x.to_bits()
        );
    }
}

#[test]
fn pointwise_errors_are_recorded_but_callback_panics_poison_the_stream() {
    for panic in [false, true] {
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let f = b
            .custom(&[x], move |_, _| {
                if panic {
                    panic!("injected model panic");
                }
                Err(alea_core::model::assembly::AssemblyError::Numerical)
            })
            .unwrap();
        b.likelihood("data", f).unwrap();
        let p = PointwiseLikelihood::new(
            b.compile().unwrap(),
            vec![PredictiveUnit {
                name: "row".into(),
                factors: vec!["data".into()],
                observations: vec!["row".into()],
                conditioned_on: vec!["x".into()],
                integrated_out: vec![],
            }],
        )
        .unwrap();
        let s = AnalysisSpec::from_likelihood(p, provenance(), vec![ChainId(9)]).unwrap();
        let mut w = AnalysisWriter::new(Vec::new(), s).unwrap();
        let stats = TransitionStats::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            w.write_draw(&draw(9, Phase::Sampling, 0, &[0.], &stats), &[])
        }));
        let mut json = Vec::new();
        if panic {
            assert!(result.is_err());
            assert!(matches!(
                w.finish(
                    &mut json,
                    &[(ChainId(9), ChainStatus::Failed("panic".into()))]
                ),
                Err(AnalysisError::Poisoned)
            ));
            assert!(json.is_empty());
        } else {
            assert!(matches!(result.unwrap(), Err(AnalysisError::Pointwise(_))));
            w.finish(
                &mut json,
                &[(ChainId(9), ChainStatus::Failed("reporting".into()))],
            )
            .unwrap();
            let m: Value = serde_json::from_slice(&json).unwrap();
            assert_eq!(m["chains"][0]["sampling"]["rows"], 1);
            assert_eq!(m["chains"][0]["sampling"]["failures"], 1);
        }
    }
}

fn data_spec() -> AnalysisSpec {
    use alea_runtime::analysis::{
        ArrayValues, DataArray, DataAxis, ObservedData, PosteriorPredictive,
    };
    let mut s = spec();
    let axes = vec![DataAxis {
        name: "trial".into(),
        coordinates: vec!["row1".into(), "row2".into()],
    }];
    s.observed_data = vec![
        ObservedData {
            array: DataArray {
                name: "rt".into(),
                axes: axes.clone(),
                values: ArrayValues::Real(vec![0.4, 0.5]),
                unit: Some("seconds".into()),
            },
            observation_dimension: "trial".into(),
        },
        ObservedData {
            array: DataArray {
                name: "choice".into(),
                axes: axes.clone(),
                values: ArrayValues::Boolean(vec![true, false]),
                unit: None,
            },
            observation_dimension: "trial".into(),
        },
    ];
    s.constant_data = vec![
        DataArray {
            name: "signed".into(),
            axes: axes.clone(),
            values: ArrayValues::Integer(vec![i64::MIN, i64::MAX]),
            unit: None,
        },
        DataArray {
            name: "unsigned".into(),
            axes,
            values: ArrayValues::Unsigned(vec![0, u64::MAX]),
            unit: None,
        },
        DataArray {
            name: "label".into(),
            axes: vec![],
            values: ArrayValues::Text(vec!["λ\r\nlabel".into()]),
            unit: None,
        },
    ];
    s.predictions = vec!["rt1".into(), "rt2".into(), "unbound expectation".into()];
    s.posterior_predictive = vec![PosteriorPredictive {
        variable: "rt".into(),
        columns: vec!["rt1".into(), "rt2".into()],
    }];
    s
}
#[test]
fn typed_data_and_predictive_bindings_preserve_axes_integer_limits_and_semantics() {
    let mut w = AnalysisWriter::new(Vec::new(), data_spec()).unwrap();
    let stats = TransitionStats::default();
    w.write_draw(&draw(9, Phase::Sampling, 0, &[0.], &stats), &[0.4, 0.5, 2.])
        .unwrap();
    let (_, m) = finish(w);
    assert_eq!(m["schema_version"], 2);
    assert_eq!(
        m["observed_data"][0]["array"]["axes"][0]["coordinates"][1],
        "row2"
    );
    assert_eq!(m["observed_data"][1]["array"]["values"][0], true);
    assert_eq!(m["constant_data"][0]["values"][0], i64::MIN);
    assert_eq!(m["constant_data"][1]["values"][1], u64::MAX);
    assert_eq!(m["constant_data"][2]["values"][0], "λ\r\nlabel");
    assert_eq!(m["posterior_predictive"][0]["columns"][1], "rt2");
    assert_eq!(m["predictions"][2], "unbound expectation");
}
#[test]
fn invalid_bound_shapes_coordinates_data_ids_and_predictive_mappings_write_nothing() {
    use alea_runtime::analysis::ArrayValues;
    for case in 0..12 {
        let mut s = data_spec();
        match case {
            0 => s.observed_data[0].array.values = ArrayValues::Real(vec![0.4]),
            1 => s.observed_data[0].array.values = ArrayValues::Real(vec![f64::NAN, 0.4]),
            2 => s.observed_data[0].array.axes[0].name = "chain".into(),
            3 => s.observed_data[0].array.axes[0].coordinates[1] = "row1".into(),
            4 => s.observed_data[0].array.axes[0].coordinates[1] = "unknown".into(),
            5 => s.observed_data[0].observation_dimension = "missing".into(),
            6 => s.posterior_predictive[0].columns.pop().map(|_| ()).unwrap(),
            7 => s.posterior_predictive[0].columns[1] = "rt1".into(),
            8 => s.posterior_predictive[0].variable = "unknown".into(),
            9 => {
                s.units.insert("prediction:rt1".into(), "metres".into());
            }
            10 => s.constant_data[0].name = "trial".into(),
            _ => s.observed_data[0].array.values = ArrayValues::Text(vec!["a".into(), "b".into()]),
        }
        let mut bytes = Vec::new();
        assert!(AnalysisWriter::new(&mut bytes, s).is_err(), "case {case}");
        assert!(bytes.is_empty());
    }
}

#[test]
fn replicated_values_obey_the_observed_datatype_without_losing_failed_indices() {
    use alea_runtime::analysis::ArrayValues;
    for (values, invalid) in [
        (ArrayValues::Boolean(vec![false, true]), 0.5),
        (ArrayValues::Integer(vec![0, 1]), 9_223_372_036_854_775_808.),
        (ArrayValues::Unsigned(vec![0, 1]), -1.),
    ] {
        let mut s = data_spec();
        s.observed_data[0].array.values = values;
        let mut w = AnalysisWriter::new(Vec::new(), s).unwrap();
        let stats = TransitionStats::default();
        assert!(
            w.write_draw(
                &draw(9, Phase::Sampling, 0, &[0.], &stats),
                &[invalid, 1., 0.]
            )
            .is_err()
        );
        w.write_draw(&draw(9, Phase::Sampling, 1, &[0.], &stats), &[0., 1., 0.])
            .unwrap();
        let (_, m) = finish(w);
        assert_eq!(m["chains"][0]["sampling"]["failures"], 1);
        assert_eq!(m["chains"][0]["sampling"]["rows"], 2);
    }
}
