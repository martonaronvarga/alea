use alea_core::{
    model::assembly::{AssemblyError as E, ModelBuilder, Primitive},
    target::LogDensityGradient,
};
use alea_distributions::modeling::regression::*;

fn data<Y>(rows: usize, columns: usize, response: impl Fn(usize) -> Y) -> RegressionData<Y> {
    RegressionData::new(
        columns,
        (0..rows)
            .flat_map(|i| (0..columns).map(move |j| ((i + 3 * j) % 13) as f64 / 16. - 0.375))
            .collect(),
        (0..rows).map(response).collect(),
    )
    .unwrap()
    .with_offsets((0..rows).map(|i| (i % 5) as f64 / 8. - 0.25).collect())
    .unwrap()
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * (1. + b.abs()),
        "{a} != {b} (tolerance {tolerance})"
    );
}
fn check<P: Primitive>(auto: P, scalar: P, q: &[f64]) {
    assert_eq!(auto.signature(), scalar.signature());
    let mut aw = auto.workspace();
    let mut sw = scalar.workspace();
    let mut ag = vec![0.; q.len()];
    let mut sg = vec![0.; q.len()];
    let av = auto.evaluate(q, &mut ag, &mut aw).unwrap();
    let sv = scalar.evaluate(q, &mut sg, &mut sw).unwrap();
    close(av, sv, 3e-12);
    for (&a, &s) in ag.iter().zip(&sg) {
        close(a, s, 3e-12);
    }
    // Each backend must differentiate its own evaluated objective consistently.
    for (primitive, work, gradient) in [(&auto, &mut aw, &ag), (&scalar, &mut sw, &sg)] {
        for j in 0..q.len() {
            let mut plus = q.to_vec();
            let mut minus = q.to_vec();
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let fd = (primitive
                .evaluate(&plus, &mut vec![0.; q.len()], work)
                .unwrap()
                - primitive
                    .evaluate(&minus, &mut vec![0.; q.len()], work)
                    .unwrap())
                / 2e-5;
            close(fd, gradient[j], 2e-6);
        }
    }
    let allocations = allocation_counter::measure(|| {
        for _ in 0..4 {
            std::hint::black_box(auto.evaluate(q, &mut ag, &mut aw).unwrap());
            std::hint::black_box(scalar.evaluate(q, &mut sg, &mut sw).unwrap());
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(
        auto.evaluate(q, &mut ag, &mut aw).unwrap().to_bits(),
        av.to_bits()
    );
    assert_eq!(
        scalar.evaluate(q, &mut sg, &mut sw).unwrap().to_bits(),
        sv.to_bits()
    );
}

#[test]
fn all_families_agree_across_dispatch_shapes_tails_and_zero_columns() {
    for rows in [1, 3, 4, 5, 7, 8, 9, 33, 1_000, 10_000] {
        for columns in [0, 1, 3, 4, 5, 9] {
            let mut b = ModelBuilder::new();
            let beta: Vec<_> = (0..columns)
                .map(|j| b.real(&format!("b{j}")).unwrap().expr())
                .collect();
            let sigma = b.positive("sigma").unwrap();
            let q: Vec<_> = (0..columns).map(|j| j as f64 / 32. - 0.125).collect();
            let nd = data(rows, columns, |i| (i % 7) as f64 / 4. - 0.5);
            let auto = NormalRegression::new(&beta, &sigma, nd.clone()).unwrap();
            let scalar = NormalRegression::new(&beta, &sigma, nd)
                .unwrap()
                .with_execution(RegressionExecution::Scalar);
            let mut nq = q.clone();
            nq.push(1.25);
            check(auto, scalar, &nq);
            let ld = data(rows, columns, |i| i % 3 != 0);
            check(
                BernoulliLogitRegression::new(&beta, ld.clone()).unwrap(),
                BernoulliLogitRegression::new(&beta, ld)
                    .unwrap()
                    .with_execution(RegressionExecution::Scalar),
                &q,
            );
            let pd = data(rows, columns, |i| (i % 4) as u64);
            check(
                PoissonLogRegression::new(&beta, pd.clone()).unwrap(),
                PoissonLogRegression::new(&beta, pd)
                    .unwrap()
                    .with_execution(RegressionExecution::Scalar),
                &q,
            );
        }
    }
}

#[test]
fn plans_are_inspectable_through_graph_descriptors_and_rebinding_preserves_policy() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        let mut b = ModelBuilder::new();
        let beta = [b.real("beta").unwrap().expr()];
        let sigma = b.positive("sigma").unwrap();
        let d = RegressionData::new(1, vec![1.; 5], vec![0.; 5]).unwrap();
        let p = NormalRegression::new(&beta, &sigma, d)
            .unwrap()
            .with_execution(execution);
        let plan = p.execution_plan();
        assert_eq!(plan.execution(), execution);
        assert_eq!(
            (
                plan.rows(),
                plan.columns(),
                plan.tile_rows(),
                plan.workers()
            ),
            (5, 1, 4, 1)
        );
        assert!(plan.is_broadcast());
        assert_eq!(plan.packed_design_bytes(), 0);
        if execution == RegressionExecution::Scalar {
            assert_eq!(plan.lanes(), 1);
        } else {
            assert!(plan.lanes() >= 1);
        }
        let root = b.primitive(p).unwrap();
        b.likelihood("y", root).unwrap();
        let m = b.compile().unwrap();
        let source = m.rust_primal().unwrap();
        // Same shape, different packing decision: old unit-column optimization
        // cannot survive an offset replacement.
        let m = m
            .rebind::<NormalRegression>(root, |p| {
                p.with_data(p.data().clone().with_offsets(vec![0.5; 5]).unwrap())
            })
            .unwrap();
        let descriptor = m
            .graph()
            .find(|n| n.expression == root)
            .unwrap()
            .primitive
            .unwrap()
            .descriptor::<NormalRegression>()
            .unwrap();
        let plan = descriptor.execution_plan();
        assert_eq!(plan.execution(), execution);
        assert!(!plan.is_broadcast());
        assert_eq!(plan.packed_design_bytes(), 2 * 4 * size_of::<f64>());
        assert_ne!(source, m.rust_primal().unwrap());
        let failure = m
            .rebind::<NormalRegression>(root, |p| {
                p.with_data(RegressionData::new(1, vec![1.; 4], vec![0.; 4]).unwrap())
            })
            .err()
            .unwrap();
        assert!(matches!(failure.source, E::Shape));
        let descriptor = failure
            .model
            .graph()
            .find(|n| n.expression == root)
            .unwrap()
            .primitive
            .unwrap()
            .descriptor::<NormalRegression>()
            .unwrap();
        assert_eq!(descriptor.execution_plan().execution(), execution);
        let target = failure.model.into_target().unwrap();
        let mut gradient = [0.; 2];
        assert!(
            target
                .logp_grad(&[0.2, 0.1], &mut gradient)
                .unwrap()
                .is_finite()
        );
        let old = gradient.map(f64::to_bits);
        assert!(target.logp_grad(&[0.2, -1000.], &mut gradient).is_err());
        assert_eq!(gradient.map(f64::to_bits), old);
        assert!(
            target
                .logp_grad(&[0.2, 0.1], &mut gradient)
                .unwrap()
                .is_finite()
        );
    }
}

#[test]
fn link_rebindings_rebuild_packing_without_resetting_scalar_dispatch() {
    let mut b = ModelBuilder::new();
    let beta = [b.real("beta").unwrap().expr()];
    let logit = BernoulliLogitRegression::new(
        &beta,
        RegressionData::new(1, vec![1.; 5], vec![true; 5]).unwrap(),
    )
    .unwrap()
    .with_execution(RegressionExecution::Scalar);
    let poisson = PoissonLogRegression::new(
        &beta,
        RegressionData::new(1, vec![1.; 5], vec![2; 5]).unwrap(),
    )
    .unwrap()
    .with_execution(RegressionExecution::Scalar);
    let logit = logit
        .with_data(logit.data().clone().with_offsets(vec![0.5; 5]).unwrap())
        .unwrap();
    let poisson = poisson
        .with_data(poisson.data().clone().with_offsets(vec![0.5; 5]).unwrap())
        .unwrap();
    for plan in [logit.execution_plan(), poisson.execution_plan()] {
        assert_eq!(plan.execution(), RegressionExecution::Scalar);
        assert_eq!(plan.lanes(), 1);
        assert!(!plan.is_broadcast());
        assert_eq!(plan.packed_design_bytes(), 64);
    }
    assert!(matches!(
        poisson.with_data(RegressionData::new(1, vec![1.; 5], vec![(1 << 53) + 1; 5]).unwrap()),
        Err(E::Factor(_))
    ));
}

#[test]
fn compressed_normal_rebinding_refreshes_response_offset_and_design_and_keeps_errors() {
    let mut b = ModelBuilder::new();
    let beta = [b.real("beta").unwrap().expr()];
    let sigma = b.positive("sigma").unwrap();
    let original = NormalRegression::new(
        &beta,
        &sigma,
        RegressionData::new(1, vec![1.; 1000], vec![0.; 1000]).unwrap(),
    )
    .unwrap();
    let replacement = original
        .with_data(
            RegressionData::new(1, vec![2.; 1000], vec![3.; 1000])
                .unwrap()
                .with_offsets(vec![0.5; 1000])
                .unwrap(),
        )
        .unwrap();
    for (model, expected_residual, expected_score) in
        [(&original, -0.25, -250.), (&replacement, 2., 4000.)]
    {
        let mut work = model.workspace();
        let mut g = [0.; 2];
        let value = model.evaluate(&[0.25, 1.], &mut g, &mut work).unwrap();
        close(
            value,
            -500. * (expected_residual * expected_residual + std::f64::consts::TAU.ln()),
            3e-12,
        );
        close(g[0], expected_score, 3e-12);
        let good = g;
        assert!(matches!(
            model.evaluate(&[f64::MAX, 1e-300], &mut g, &mut work),
            Err(E::Numerical)
        ));
        assert_eq!(g, good);
        assert_eq!(
            model
                .evaluate(&[0.25, 1.], &mut g, &mut work)
                .unwrap()
                .to_bits(),
            value.to_bits()
        );
        let mut rows = vec![0.; 1000];
        let rule = model.batch_rule().unwrap();
        (rule.evaluate)(model, &[0.25, 1.], &mut rows, &mut model.batch_workspace()).unwrap();
        close(value, rows.iter().sum(), 3e-12);
    }
}

#[test]
fn both_dispatch_modes_keep_tail_scores_domain_errors_and_recovery() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        let mut b = ModelBuilder::new();
        let beta = [b.real("beta").unwrap().expr()];
        let sigma = b.positive("sigma").unwrap();
        let normal = NormalRegression::new(
            &beta,
            &sigma,
            RegressionData::new(1, vec![1.; 5], vec![0.; 5]).unwrap(),
        )
        .unwrap()
        .with_execution(execution);
        let mut w = normal.workspace();
        let mut g = [0.; 2];
        let good = normal.evaluate(&[0.2, 1.], &mut g, &mut w).unwrap();
        let bits = g.map(f64::to_bits);
        assert!(matches!(
            normal.evaluate(&[0.2], &mut g, &mut w),
            Err(E::Shape)
        ));
        assert!(normal.evaluate(&[0.2, 0.], &mut g, &mut w).is_err());
        assert!(
            normal
                .evaluate(&[f64::MAX, 1e-300], &mut g, &mut w)
                .is_err()
        );
        assert_eq!(
            normal
                .evaluate(&[0.2, 1.], &mut g, &mut w)
                .unwrap()
                .to_bits(),
            good.to_bits()
        );
        assert_eq!(g.map(f64::to_bits), bits);

        let logit = BernoulliLogitRegression::new(
            &beta,
            RegressionData::new(1, vec![1.; 5], vec![true; 5]).unwrap(),
        )
        .unwrap()
        .with_execution(execution);
        let mut w = logit.workspace();
        let mut g = [0.];
        assert!(logit.evaluate(&[40.], &mut g, &mut w).unwrap() < 0.);
        assert!(g[0] > 0.); // Do not erase the representable positive tail score.
        assert!(logit.evaluate(&[f64::INFINITY], &mut g, &mut w).is_err());
        assert!(
            logit
                .evaluate(&[-800.], &mut g, &mut w)
                .unwrap()
                .is_finite()
        );

        let poisson = PoissonLogRegression::new(
            &beta,
            RegressionData::new(1, vec![1.; 5], vec![2; 5]).unwrap(),
        )
        .unwrap()
        .with_execution(execution);
        let mut w = poisson.workspace();
        assert!(matches!(
            poisson.evaluate(&[800.], &mut g, &mut w),
            Err(E::Numerical)
        ));
        assert!(
            poisson
                .evaluate(&[-800.], &mut g, &mut w)
                .unwrap()
                .is_finite()
        );
        assert_eq!(g, [10.]); // Underflowed rate still has a finite log-rate score.
    }
}
