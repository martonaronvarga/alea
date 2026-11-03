use alea_core::{
    model::assembly::{CompiledModel, ModelBuilder},
    target::LogDensityGradient,
};
use alea_distributions::modeling::regression::{RegressionData, poisson_log_regression};

fn model(n: usize, p: usize, chunk: usize) -> alea_core::model::ComposedObjective<CompiledModel> {
    let mut b = ModelBuilder::new();
    let beta: Vec<_> = (0..p)
        .map(|j| b.real(&format!("b{j}")).unwrap().expr())
        .collect();
    // Compiled models require a parameter block; retain zero-column factors
    // using an explicitly unused coordinate rather than changing that contract.
    let _unused = b.real("unused").unwrap();
    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        let matrix = (start * p..end * p)
            .map(|i| if p == 1 { 1. } else { (i as f64 / 13.).sin() })
            .collect();
        let data = RegressionData::new(p, matrix, (start..end).map(|i| (i % 7) as u64).collect())
            .unwrap()
            .with_offsets((start..end).map(|i| (i % 3) as f64 / 11.).collect())
            .unwrap();
        let factor = poisson_log_regression(&mut b, &beta, data).unwrap();
        b.likelihood(&format!("y{start}"), factor).unwrap();
    }
    b.compile().unwrap().into_objective().unwrap()
}

#[test]
fn poisson_tiles_match_single_rows_finite_differences_and_recover_without_allocation() {
    for n in [1, 4, 5, 8, 17, 128, 129] {
        for p in [0, 1, 3, 8, 9] {
            let batch = model(n, p, n);
            let rows = model(n, p, 1);
            let mut g = vec![0.; p + 1];
            let mut h = vec![0.; p + 1];
            for coordinate in [0.2, -30., 40.] {
                let q = vec![coordinate; p + 1];
                let value = batch.logp_grad(&q, &mut g).unwrap();
                let expected = rows.logp_grad(&q, &mut h).unwrap();
                assert!((value - expected).abs() < 2e-12 * (1. + expected.abs()));
                for (x, y) in g.iter().zip(&h) {
                    assert!((x - y).abs() < 2e-12 * (1. + y.abs()));
                }
                if coordinate == 0.2 {
                    for j in 0..p {
                        let mut plus = q.clone();
                        let mut minus = q.clone();
                        plus[j] += 1e-5;
                        minus[j] -= 1e-5;
                        let fd = (batch.logp_grad(&plus, &mut h).unwrap()
                            - batch.logp_grad(&minus, &mut h).unwrap())
                            / 2e-5;
                        assert!((g[j] - fd).abs() < 2e-7 * (1. + g[j].abs()));
                    }
                    let allocation = allocation_counter::measure(|| {
                        for _ in 0..8 {
                            std::hint::black_box(batch.logp_grad(&q, &mut h).unwrap());
                        }
                    });
                    assert_eq!(allocation.count_total, 0);
                    if p > 0 {
                        h.fill(17.);
                        assert!(batch.logp_grad(&vec![f64::MAX; p + 1], &mut h).is_err());
                        assert!(h.iter().all(|x| *x == 17.));
                        assert_eq!(
                            batch.logp_grad(&q, &mut h).unwrap().to_bits(),
                            value.to_bits()
                        );
                        assert_eq!(g, h);
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_poisson_rebinding_refreshes_all_statistics_and_keeps_row_reporting() {
    use alea_core::model::assembly::Primitive;
    use alea_distributions::modeling::regression::{PoissonLogRegression, RegressionExecution};
    let mut builder = ModelBuilder::new();
    let beta = builder.real("beta").unwrap().expr();
    let original = RegressionData::new(1, vec![1.; 129], vec![2; 129]).unwrap();
    let descriptor = PoissonLogRegression::new(&[beta], original)
        .unwrap()
        .with_execution(RegressionExecution::Scalar);
    let mut original_gradient = [0.];
    let original_value = descriptor
        .evaluate(&[0.2], &mut original_gradient, &mut descriptor.workspace())
        .unwrap();
    let mut original_rows = vec![0.; 129];
    (descriptor.batch_rule().unwrap().evaluate)(
        &descriptor,
        &[0.2],
        &mut original_rows,
        &mut descriptor.batch_workspace(),
    )
    .unwrap();
    assert!((original_value - original_rows.iter().sum::<f64>()).abs() < 1e-10);
    let new_counts = descriptor
        .with_data(descriptor.data().with_observations(vec![3; 129]).unwrap())
        .unwrap();
    let mut new_gradient = [0.];
    let new_value = new_counts
        .evaluate(&[0.2], &mut new_gradient, &mut new_counts.workspace())
        .unwrap();
    assert!((new_value - original_value - 129. * (0.2 - 3_f64.ln())).abs() < 1e-10);
    assert!((new_gradient[0] - original_gradient[0] - 129.).abs() < 1e-10);
    let changed = RegressionData::new(
        1,
        (0..129)
            .map(|i| if i % 2 == 0 { 1. } else { 0.5 })
            .collect(),
        (0..129).map(|i| (i % 5) as u64).collect(),
    )
    .unwrap()
    .with_offsets((0..129).map(|i| (i % 3) as f64 / 11.).collect())
    .unwrap();
    let rebound = descriptor.with_data(changed.clone()).unwrap();
    let fresh = PoissonLogRegression::new(&[beta], changed)
        .unwrap()
        .with_execution(RegressionExecution::Scalar);
    let mut w = rebound.workspace();
    let mut v = fresh.workspace();
    let mut g = [17.];
    let mut h = [0.];
    for q in [0.2, -1000., 1000., 0.2] {
        let actual = rebound.evaluate(&[q], &mut g, &mut w);
        let expected = fresh.evaluate(&[q], &mut h, &mut v);
        match (actual, expected) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.to_bits(), b.to_bits());
                assert_eq!(g, h);
            }
            (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
            other => panic!("{other:?}"),
        }
    }
    let value = rebound.evaluate(&[0.2], &mut g, &mut w).unwrap();
    let rule = rebound.batch_rule().unwrap();
    assert_eq!(rule.units, 129);
    let mut out = vec![0.; 129];
    (rule.evaluate)(&rebound, &[0.2], &mut out, &mut rebound.batch_workspace()).unwrap();
    assert!((value - out.iter().sum::<f64>()).abs() < 1e-10);
    let allocation = allocation_counter::measure(|| {
        rebound.evaluate(&[0.2], &mut g, &mut w).unwrap();
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn full_tiles_preserve_underflowed_rates_and_count_limits() {
    let mut b = ModelBuilder::new();
    let beta = b.real("b").unwrap();
    let f = poisson_log_regression(
        &mut b,
        &[beta.expr()],
        RegressionData::new(1, vec![1.; 4], vec![0, 1, 2, 1 << 53]).unwrap(),
    )
    .unwrap();
    b.likelihood("y", f).unwrap();
    let t = b.compile().unwrap().into_objective().unwrap();
    let mut g = [0.];
    assert!(t.logp_grad(&[-1000.], &mut g).unwrap().is_finite());
    assert_eq!(g[0], ((1_u64 << 53) + 3) as f64);
    g[0] = 17.;
    assert!(t.logp_grad(&[1000.], &mut g).is_err());
    assert_eq!(g, [17.]);
}

#[test]
fn common_poisson_full_model_matches_native_laws_finite_differences_and_error_recovery() {
    use alea_core::distribution::Distribution;
    use alea_distributions::scalar::PoissonLog;
    for count in [2, 1_000_000_000_000, 1 << 53] {
        let mut b = ModelBuilder::new();
        let beta = b.real("beta").unwrap();
        let data = RegressionData::new(1, vec![1.; 129], vec![count; 129]).unwrap();
        let f = poisson_log_regression(&mut b, &[beta.expr()], data).unwrap();
        b.likelihood("y", f).unwrap();
        let target = b.compile().unwrap().into_target().unwrap();
        let mut g = [0.];
        let mut scratch = [0.];
        for q in [0.2, -1000., (count as f64).ln()] {
            let value = target.logp_grad(&[q], &mut g).unwrap();
            let row = PoissonLog::new(q).unwrap().log_prob(&count).unwrap();
            assert!((value - 129. * row).abs() < 3e-12 * (1. + value.abs()));
            if count == 2 && q == 0.2 {
                let fd = (target.logp_grad(&[q + 1e-5], &mut scratch).unwrap()
                    - target.logp_grad(&[q - 1e-5], &mut scratch).unwrap())
                    / 2e-5;
                assert!((fd - g[0]).abs() < 1e-7 * (1. + g[0].abs()));
            }
        }
        let value = target.logp_grad(&[0.2], &mut g).unwrap();
        let expected = g;
        g = [17.];
        assert!(target.logp_grad(&[1000.], &mut g).is_err());
        assert_eq!(g, [17.]);
        assert_eq!(
            target.logp_grad(&[0.2], &mut g).unwrap().to_bits(),
            value.to_bits()
        );
        assert_eq!(g, expected);
        assert_eq!(
            allocation_counter::measure(|| {
                target.logp_grad(&[0.2], &mut g).unwrap();
            })
            .count_total,
            0
        );
    }
}
