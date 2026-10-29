use alea_core::{
    model::{
        ComposedObjective,
        assembly::{CompiledModel, ModelBuilder},
    },
    target::LogDensityGradient,
};
use alea_distributions::{
    modeling::{bernoulli_logit, normal_observation, observation, regression::*},
    scalar::Poisson,
};

fn data_design() -> Vec<f64> {
    vec![1., -2., 1., 0.5, 1., 3.]
}
fn model(family: usize, chunk: usize, scalar: bool) -> CompiledModel {
    let mut b = ModelBuilder::new();
    let beta = [
        b.real("intercept").unwrap().expr(),
        b.real("slope").unwrap().expr(),
    ];
    let sigma = b.positive("sigma").unwrap();
    let design = data_design();
    let offsets = [0.2, -0.1, 0.4];
    for start in (0..3).step_by(chunk) {
        let end = (start + chunk).min(3);
        if !scalar {
            let factor = match family {
                0 => normal_regression(
                    &mut b,
                    &beta,
                    &sigma,
                    RegressionData::new(
                        2,
                        design[2 * start..2 * end].to_vec(),
                        [0.3, -1., 2.][start..end].to_vec(),
                    )
                    .unwrap()
                    .with_offsets(offsets[start..end].to_vec())
                    .unwrap(),
                ),
                1 => bernoulli_logit_regression(
                    &mut b,
                    &beta,
                    RegressionData::new(
                        2,
                        design[2 * start..2 * end].to_vec(),
                        [false, true, true][start..end].to_vec(),
                    )
                    .unwrap()
                    .with_offsets(offsets[start..end].to_vec())
                    .unwrap(),
                ),
                _ => poisson_log_regression(
                    &mut b,
                    &beta,
                    RegressionData::new(
                        2,
                        design[2 * start..2 * end].to_vec(),
                        [0, 2, 4][start..end].to_vec(),
                    )
                    .unwrap()
                    .with_offsets(offsets[start..end].to_vec())
                    .unwrap(),
                ),
            }
            .unwrap();
            b.likelihood(&format!("batch{start}"), factor).unwrap();
        } else {
            for i in start..end {
                let x = b.constant(design[2 * i + 1]).unwrap();
                let offset = b.constant(offsets[i]).unwrap();
                let effect = b.multiply(beta[1], x).unwrap();
                let eta = b.add(beta[0], effect).unwrap();
                let eta = b.add(eta, offset).unwrap();
                let f = match family {
                    0 => normal_observation(&mut b, [0.3, -1., 2.][i], eta, &sigma).unwrap(),
                    1 => bernoulli_logit(&mut b, [false, true, true][i], eta).unwrap(),
                    _ => {
                        let rate = b.exp(eta).unwrap();
                        observation::<Poisson, _, _>(&mut b, [0_u64, 2, 4][i], &[rate], |x| {
                            Poisson::new(x[0])
                        })
                        .unwrap()
                    }
                };
                b.likelihood(&format!("row{i}"), f).unwrap();
            }
        }
    }
    b.compile().unwrap()
}
fn evaluate(m: &ComposedObjective<CompiledModel>, q: &[f64]) -> (f64, Vec<f64>) {
    let mut g = vec![0.; q.len()];
    (m.logp_grad(q, &mut g).unwrap(), g)
}
#[test]
fn all_families_match_scalar_assembly_chunks_and_independent_finite_differences() {
    for family in 0..3 {
        let batch = model(family, 3, false)
            .select_likelihood()
            .unwrap()
            .into_objective()
            .unwrap();
        let chunked = model(family, 2, false)
            .select_likelihood()
            .unwrap()
            .into_objective()
            .unwrap();
        let scalar = model(family, 1, true)
            .select_likelihood()
            .unwrap()
            .into_objective()
            .unwrap();
        for q in [[0.2, -0.3, 0.1], [-3., 0.8, -1.], [4., -0.2, 2.]] {
            let (v, g) = evaluate(&batch, &q);
            for other in [&chunked, &scalar] {
                let (w, h) = evaluate(other, &q);
                assert!((v - w).abs() < 2e-12 * (1. + v.abs()));
                for (g, h) in g.iter().zip(h) {
                    assert!((g - h).abs() < 2e-12 * (1. + g.abs()));
                }
            }
            for j in 0..q.len() {
                let mut plus = q;
                let mut minus = q;
                plus[j] += 1e-5;
                minus[j] -= 1e-5;
                let fd = (evaluate(&batch, &plus).0 - evaluate(&batch, &minus).0) / 2e-5;
                assert!(
                    (g[j] - fd).abs() < 2e-7 * (1. + g[j].abs()),
                    "family={family},coordinate={j}"
                );
            }
        }
    }
}
#[test]
fn tail_scores_underflow_and_predictor_cancellation_remain_visible() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap();
    let f = bernoulli_logit_regression(
        &mut b,
        &[x.expr()],
        RegressionData::new(1, vec![1.], vec![true]).unwrap(),
    )
    .unwrap();
    b.likelihood("y", f).unwrap();
    let m = b.compile().unwrap().into_objective().unwrap();
    let (_, g) = evaluate(&m, &[40.]);
    assert!(g[0] > 0.);
    assert!((g[0] / (-40_f64).exp() - 1.).abs() < 1e-14);
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap();
    let f = poisson_log_regression(
        &mut b,
        &[x.expr()],
        RegressionData::new(1, vec![1.], vec![1]).unwrap(),
    )
    .unwrap();
    b.likelihood("y", f).unwrap();
    let m = b.compile().unwrap().into_objective().unwrap();
    let (v, g) = evaluate(&m, &[-1000.]);
    assert_eq!(v, -1000.);
    assert_eq!(g, [1.]);
    let mut sentinel = [17.];
    assert!(m.logp_grad(&[1000.], &mut sentinel).is_err());
    assert_eq!(sentinel, [17.]);
    let mut b = ModelBuilder::new();
    let beta = [
        b.real("a").unwrap().expr(),
        b.real("b").unwrap().expr(),
        b.real("c").unwrap().expr(),
    ];
    let f = bernoulli_logit_regression(
        &mut b,
        &beta,
        RegressionData::new(3, vec![1e16, 1., -1e16], vec![true]).unwrap(),
    )
    .unwrap();
    b.likelihood("y", f).unwrap();
    let m = b.compile().unwrap().into_objective().unwrap();
    let (v, _) = evaluate(&m, &[1., 1., 1.]);
    assert!((v + (-1_f64).exp().ln_1p()).abs() < 1e-15);
}
#[test]
fn data_admission_rejects_empty_shapes_nonfinite_values_and_count_precision() {
    assert!(matches!(
        RegressionData::<f64>::new(1, vec![], vec![]),
        Err(RegressionDataError::Empty)
    ));
    assert!(matches!(
        RegressionData::new(usize::MAX, vec![], vec![1., 2.]),
        Err(RegressionDataError::Shape)
    ));
    assert!(RegressionData::new(1, vec![f64::NAN], vec![1.]).is_err());
    assert!(
        RegressionData::new(1, vec![1.], vec![1.])
            .unwrap()
            .with_offsets(vec![f64::NEG_INFINITY])
            .is_err()
    );
    assert!(
        RegressionData::new(1, vec![1.], vec![1.])
            .unwrap()
            .with_offsets(vec![])
            .is_err()
    );
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap();
    let s = b.positive("s").unwrap();
    assert!(
        normal_regression(
            &mut b,
            &[x.expr()],
            &s,
            RegressionData::new(1, vec![1.], vec![f64::NAN]).unwrap()
        )
        .is_err()
    );
    assert!(
        poisson_log_regression(
            &mut b,
            &[x.expr()],
            RegressionData::new(1, vec![1.], vec![(1 << 53) + 1]).unwrap()
        )
        .is_err()
    );
    assert!(
        bernoulli_logit_regression(
            &mut b,
            &[],
            RegressionData::new(1, vec![1.], vec![true]).unwrap()
        )
        .is_err()
    );
    let mut foreign = ModelBuilder::new();
    let z = foreign.real("z").unwrap();
    assert!(
        bernoulli_logit_regression(
            &mut b,
            &[z.expr()],
            RegressionData::new(1, vec![1.], vec![true]).unwrap()
        )
        .is_err()
    );
}
#[test]
fn graph_size_is_independent_of_rows_and_bad_estimators_still_have_likelihoods() {
    let mut b = ModelBuilder::new();
    let x = b.real("intercept").unwrap();
    let f = poisson_log_regression(
        &mut b,
        &[x.expr()],
        RegressionData::new(1, vec![1.; 10000], vec![0; 10000]).unwrap(),
    )
    .unwrap();
    b.likelihood("all zero counts", f).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.expression_count(), 2);
    let m = model.into_objective().unwrap();
    let (a, _) = evaluate(&m, &[-10.]);
    let (c, _) = evaluate(&m, &[-20.]);
    assert!(c > a); // supremum at rate zero, no finite log-rate MLE
    let mut b = ModelBuilder::new();
    let x = b.real("slope").unwrap();
    let f = bernoulli_logit_regression(
        &mut b,
        &[x.expr()],
        RegressionData::new(1, vec![-1., 1.], vec![false, true]).unwrap(),
    )
    .unwrap();
    b.likelihood("separated", f).unwrap();
    let m = b.compile().unwrap().into_objective().unwrap();
    assert!(evaluate(&m, &[20.]).0 > evaluate(&m, &[10.]).0);
}

#[test]
fn cancelling_predictor_scores_match_independent_decimal_reference() {
    // Decimal60 on the stored binary64 design/response/parameters, not a finite difference.
    let q = [-50151514.17263073, 50151515.175358005, -0.6022896768086836];
    let expected = [
        -4.7807209030239335e-6,
        -4.780720928196566e-6,
        -4.2292765882427065e-7,
    ];
    for batch in [false, true] {
        let mut b = ModelBuilder::new();
        let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
        let sigma = b.positive("sigma").unwrap();
        let mut design = Vec::new();
        let mut response = Vec::new();
        for i in 0..330 {
            let x = (i % 11) as f64 - 5.;
            let predictor = 1. + 1e-8 * x;
            let y = 1. + 0.5 * x + ((i * 17 % 19) as f64 - 9.) / 10.;
            design.extend([1., predictor]);
            response.push(y);
            if !batch {
                let mu = b.linear_combination(&beta, &[1., predictor], 0.).unwrap();
                let f = normal_observation(&mut b, y, mu, &sigma).unwrap();
                b.factor(f).unwrap();
            }
        }
        if batch {
            let f = normal_regression(
                &mut b,
                &beta,
                &sigma,
                RegressionData::new(2, design, response).unwrap(),
            )
            .unwrap();
            b.factor(f).unwrap();
        }
        let target = b.compile().unwrap().into_objective().unwrap();
        let mut gradient = [f64::NAN; 3];
        target.logp_grad(&q, &mut gradient).unwrap();
        for (actual, expected) in gradient.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    }
}

#[test]
fn standardization_preserves_offsets_null_columns_and_original_coefficient_order() {
    let eps = 2_f64.powi(-26);
    let data = RegressionData::new(3, vec![1. - eps, 1., 3., 1. + eps, 1., 3.], vec![1., 2.])
        .unwrap()
        .with_offsets(vec![0.25, -0.25])
        .unwrap();
    let standardized = data.standardize(1).unwrap();
    let map = standardized.coefficient_map();
    assert_eq!(map.intercept_column(), 1);
    assert_eq!(map.centers(), [1., 0., 3.]);
    assert_eq!(map.scales(), [eps, 1., 1.]);
    assert_eq!(standardized.data().design(), [-1., 1., 0., 1., 1., 0.]);
    assert_eq!(standardized.data().offsets(), data.offsets());
    assert!(std::ptr::eq(
        standardized.data().observations().as_ptr(),
        data.observations().as_ptr()
    ));
    let original = map.to_original(&[0.5, 1., 0.]).unwrap();
    assert_eq!(map.to_standardized(&original).unwrap(), [0.5, 1., 0.]);
    for i in 0..2 {
        let mut sum = alea_math::numeric::CompensatedSum::default();
        for (&x, &b) in data.design()[3 * i..3 * i + 3].iter().zip(&original) {
            sum.add_product(x, b);
        }
        sum.add(data.offsets()[i]);
        assert_eq!(sum.total(), [0.75, 1.25][i]);
    }
    assert!(matches!(
        data.standardize(0),
        Err(RegressionDataError::Intercept { column: 0 })
    ));
    assert!(data.standardize(99).is_err());
    assert!(map.to_original(&[0.]).is_err());
    assert!(map.to_standardized(&[0., f64::NAN, 0.]).is_err());
    let extremes = RegressionData::new(2, vec![1., -f64::MAX, 1., f64::MAX], vec![0., 1.]).unwrap();
    assert!(extremes.standardize(0).is_ok());
}

#[test]
fn constant_nonbinary_columns_are_exactly_removed_from_centered_design() {
    let data =
        RegressionData::new(2, (0..7).flat_map(|_| [1., 0.1]).collect(), vec![0.; 7]).unwrap();
    let standardized = data.standardize(0).unwrap();
    assert_eq!(standardized.coefficient_map().centers(), [0., 0.1]);
    assert_eq!(standardized.coefficient_map().scales(), [1., 1.]);
    for row in standardized.data().design().chunks_exact(2) {
        assert_eq!(row, [1., 0.]);
    }
    // A model without an intercept must request a different, explicit map.
    assert!(
        RegressionData::new(0, vec![], vec![0.])
            .unwrap()
            .standardize(0)
            .is_err()
    );
}
