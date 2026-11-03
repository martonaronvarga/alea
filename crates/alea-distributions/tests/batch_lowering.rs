use alea_core::{
    model::{
        ConstrainedModel,
        assembly::{AssemblyError as E, ModelBuilder},
    },
    target::LogDensityGradient,
};
use alea_distributions::modeling::{normal, regression::*};

#[test]
fn batch_rows_form_explicit_predictive_partitions_without_scalarizing_the_graph() {
    use alea_core::model::assembly::{BatchUnit, PointwiseLikelihood, PredictiveUnit};
    for family in 0..3 {
        let batch = model(family).0.batch_likelihood().unwrap();
        let q = if family == 0 {
            vec![0.2, -0.4, 0.3]
        } else {
            vec![0.2, -0.4]
        };
        let mut rows = [0.; 3];
        let total = batch.evaluate(&q, &mut rows).unwrap();
        let count = batch.model().expression_count();
        let units: Vec<_> = [vec!["row0", "row2"], vec!["row1"]]
            .into_iter()
            .enumerate()
            .map(|(i, ids)| PredictiveUnit {
                name: format!("group{i}"),
                factors: vec!["data".into()],
                observations: ids.into_iter().map(str::to_owned).collect(),
                conditioned_on: vec!["coefficients".into()],
                integrated_out: vec![],
            })
            .collect();
        let assignments = vec![
            vec![
                BatchUnit {
                    factor: "data".into(),
                    row: 2,
                },
                BatchUnit {
                    factor: "data".into(),
                    row: 0,
                },
            ],
            vec![BatchUnit {
                factor: "data".into(),
                row: 1,
            }],
        ];
        let pointwise =
            PointwiseLikelihood::from_batch(batch, units.clone(), assignments.clone()).unwrap();
        let mut out = [0.; 2];
        assert_eq!(
            pointwise.evaluate(&q, &mut out).unwrap().to_bits(),
            total.to_bits()
        );
        close(out[0], rows[0] + rows[2]);
        close(out[1], rows[1]);
        assert_eq!(pointwise.model().expression_count(), count);
        assert_eq!(
            allocation_counter::measure(|| {
                pointwise.evaluate(&q, &mut out).unwrap();
            })
            .count_total,
            0
        );
        out.fill(99.);
        let mut invalid = q.clone();
        invalid[0] = f64::NAN;
        assert!(pointwise.evaluate(&invalid, &mut out).is_err());
        assert_eq!(out, [99.; 2]);
        pointwise.evaluate(&q, &mut out).unwrap();
        // Every actual row and observation ID must be assigned exactly once.
        for mutation in 0..6 {
            let mut units = units.clone();
            let mut assignments = assignments.clone();
            match mutation {
                0 => {
                    assignments[0].pop();
                }
                1 => assignments[1][0].row = 0,
                2 => assignments[1][0].row = 100,
                3 => units[1].observations = units[0].observations.clone(),
                4 => units[1].integrated_out = units[1].conditioned_on.clone(),
                _ => units[1].factors = vec!["prior".into()],
            }
            assert!(
                PointwiseLikelihood::from_batch(
                    model(family).0.batch_likelihood().unwrap(),
                    units,
                    assignments
                )
                .is_err()
            );
        }
        assert!(pointwise.into_model().select_posterior().is_ok());
    }
}

fn model(
    family: usize,
) -> (
    alea_core::model::assembly::CompiledModel,
    alea_core::model::assembly::Expr,
) {
    let mut b = ModelBuilder::new();
    let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
    let design = vec![1., -1., 1., 0.5, 1., 2.];
    let offsets = vec![0.25, -0.5, 0.75];
    let f = match family {
        0 => {
            let sigma = b.positive("sigma").unwrap();
            normal_regression(
                &mut b,
                &beta,
                &sigma,
                RegressionData::new(2, design, vec![0.5, -1., 1.5])
                    .unwrap()
                    .with_offsets(offsets)
                    .unwrap(),
            )
            .unwrap()
        }
        1 => bernoulli_logit_regression(
            &mut b,
            &beta,
            RegressionData::new(2, design, vec![false, true, true])
                .unwrap()
                .with_offsets(offsets)
                .unwrap(),
        )
        .unwrap(),
        _ => poisson_log_regression(
            &mut b,
            &beta,
            RegressionData::new(2, design, vec![0, 2, 3])
                .unwrap()
                .with_offsets(offsets)
                .unwrap(),
        )
        .unwrap(),
    };
    let prior = normal(&mut b, beta[0], 0., 2.).unwrap();
    b.prior("prior", prior).unwrap();
    b.likelihood("data", f).unwrap();
    (b.compile().unwrap(), f)
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-12 * (1. + b.abs()), "{a} != {b}");
}
#[test]
fn all_native_batches_lower_rows_without_priors_or_jacobians_and_allocate_nothing() {
    for family in 0..3 {
        let (model, _) = model(family);
        let q: Vec<f64> = if family == 0 {
            vec![0.2, -0.4, 0.3]
        } else {
            vec![0.2, -0.4]
        };
        let mut constrained = q.clone();
        if family == 0 {
            constrained[2] = q[2].exp();
        }
        let mut gradient = vec![0.; q.len()];
        let expected = model.select_likelihood().unwrap();
        let summed = expected.logp_grad(&constrained, &mut gradient).unwrap();
        assert!(expected.source_capabilities().supports_rust_primal());
        let source = expected.rust_primal().unwrap();
        assert!(source.contains("for i in 0..3"));
        let pointwise = expected.batch_likelihood().unwrap();
        assert_eq!(pointwise.units().len(), 3);
        for (i, unit) in pointwise.units().iter().enumerate() {
            assert_eq!(unit.factor, "data");
            assert_eq!(unit.row, i);
        }
        let mut out = [0.; 3];
        close(pointwise.evaluate(&q, &mut out).unwrap(), summed);
        close(out.iter().sum(), summed);
        let eta = [0.2 + 0.4 + 0.25, 0.2 - 0.2 - 0.5, 0.2 - 0.8 + 0.75];
        for i in 0..3 {
            let reference = match family {
                0 => {
                    -0.5 * (([0.5, -1., 1.5][i] - eta[i]) / q[2].exp()).powi(2)
                        - q[2]
                        - 0.5 * std::f64::consts::TAU.ln()
                }
                1 => {
                    let softplus = eta[i].max(0.) + (-eta[i].abs()).exp().ln_1p();
                    if [false, true, true][i] {
                        eta[i] - softplus
                    } else {
                        -softplus
                    }
                }
                _ => [0., 2., 3.][i] * eta[i] - eta[i].exp() - [0., 2_f64.ln(), 6_f64.ln()][i],
            };
            close(out[i], reference);
        }
        let allocations = allocation_counter::measure(|| {
            for _ in 0..16 {
                pointwise.evaluate(&q, &mut out).unwrap();
            }
        });
        assert_eq!(allocations.count_total, 0);
        if family == 0 {
            out.fill(19.);
            assert!(pointwise.evaluate(&[0.2, -0.4, -1000.], &mut out).is_err());
            assert_eq!(out, [19.; 3]);
            close(pointwise.evaluate(&q, &mut out).unwrap(), summed);
        }
    }
}
#[test]
fn all_three_rebindings_refresh_pointwise_rows_source_and_scores_without_changing_handles() {
    for family in 0..3 {
        let (model, root) = model(family);
        let coordinates = model.coordinates();
        let source = model.rust_primal().unwrap();
        let rebound = match family {
            0 => model.rebind::<NormalRegression>(root, |p| {
                p.with_data(
                    RegressionData::new(2, p.data().design().to_vec(), vec![1., 2., 3.])
                        .unwrap()
                        .with_offsets(p.data().offsets().to_vec())
                        .unwrap(),
                )
            }),
            1 => model.rebind::<BernoulliLogitRegression>(root, |p| {
                p.with_data(
                    RegressionData::new(2, p.data().design().to_vec(), vec![true, false, false])
                        .unwrap()
                        .with_offsets(p.data().offsets().to_vec())
                        .unwrap(),
                )
            }),
            _ => model.rebind::<PoissonLogRegression>(root, |p| {
                p.with_data(
                    RegressionData::new(2, p.data().design().to_vec(), vec![1, 0, 4])
                        .unwrap()
                        .with_offsets(p.data().offsets().to_vec())
                        .unwrap(),
                )
            }),
        }
        .unwrap();
        assert_eq!(*coordinates, *rebound.coordinates());
        assert_ne!(source, rebound.rust_primal().unwrap());
        assert!(rebound.graph().any(|n| n.expression == root));
        let pointwise = rebound.batch_likelihood().unwrap();
        let q = if family == 0 {
            vec![0.2, -0.4, 0.3]
        } else {
            vec![0.2, -0.4]
        };
        let mut out = [0.; 3];
        let total = pointwise.evaluate(&q, &mut out).unwrap();
        let target = pointwise.into_model().into_target().unwrap();
        let mut g = vec![0.; q.len()];
        close(
            target.logp_grad(&q, &mut g).unwrap(),
            total + if family == 0 { q[2] } else { 0. },
        );
        for i in 0..q.len() {
            let mut a = q.clone();
            let mut z = q.clone();
            a[i] += 1e-5;
            z[i] -= 1e-5;
            let fd = (target.logp_grad(&a, &mut vec![0.; q.len()]).unwrap()
                - target.logp_grad(&z, &mut vec![0.; q.len()]).unwrap())
                / 2e-5;
            assert!((fd - g[i]).abs() < 1e-7 * (1. + fd.abs()));
        }
    }
}
#[test]
fn failed_rebind_retains_original_model_and_original_typed_source() {
    let (model, root) = model(0);
    let source = model.rust_primal().unwrap();
    let error = model
        .rebind::<NormalRegression>(root, |p| {
            p.with_data(
                RegressionData::new(2, p.data().design().to_vec(), vec![f64::NAN; 3]).unwrap(),
            )
        })
        .err()
        .unwrap();
    assert!(matches!(error.source, E::Factor(_)));
    assert_eq!(error.model.rust_primal().unwrap(), source);
    let error = error
        .model
        .rebind::<NormalRegression>(root, |p| {
            p.with_data(RegressionData::new(2, vec![1.; 4], vec![0.; 2]).unwrap())
        })
        .err()
        .unwrap();
    assert!(matches!(error.source, E::Shape));
    assert_eq!(error.model.rust_primal().unwrap(), source);
}
#[test]
fn opaque_or_algebraic_joint_factors_never_acquire_an_invented_row_partition() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let f = b
        .custom(&[x], |x, g| {
            g[0] = 1.;
            Ok(x[0])
        })
        .unwrap();
    b.likelihood("opaque", f).unwrap();
    assert!(matches!(
        b.compile().unwrap().batch_likelihood(),
        Err(E::MissingPointwise { .. })
    ));
}
