use alea_core::{
    model::assembly::{AssemblyError as E, ModelBuilder, OperationDescriptor},
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

#[test]
fn standard_arithmetic_is_opt_in_and_cancellation_is_not_accuracy_equivalent() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        // Exact predictor is 1: all inputs and products are exactly representable.
        // Ordinary ordered addition loses the middle term, not a tail of a law.
        let data = RegressionData::new(3, [1e16, 1., -1e16].repeat(5), vec![true; 5]).unwrap();
        let reference = BernoulliLogitRegression::new(data.clone())
            .unwrap()
            .with_execution(execution);
        let standard = BernoulliLogitRegression::new(data)
            .unwrap()
            .with_arithmetic(RegressionArithmetic::Standard)
            .with_execution(execution);
        assert_eq!(
            reference.execution_plan().arithmetic(),
            RegressionArithmetic::Compensated
        );
        assert_eq!(
            standard.execution_plan().arithmetic(),
            RegressionArithmetic::Standard
        );
        let mut rg = [0.; 3];
        let mut sg = [0.; 3];
        let rv = reference
            .evaluate(&[1.; 3], &mut rg, &mut reference.workspace())
            .unwrap();
        let sv = standard
            .evaluate(&[1.; 3], &mut sg, &mut standard.workspace())
            .unwrap();
        close(rv, -5. * (-1_f64).exp().ln_1p(), 1e-14);
        close(sv, -5. * 2_f64.ln(), 1e-14);
        assert!((rv - sv).abs() > 1.);
        assert!((rg[1] - sg[1]).abs() > 1.);
        // Never silently export a differently accumulated AD primal/gradient.
        assert!(standard.rules().source.is_none());
        let rebound = standard
            .with_data(standard.data().clone().with_offsets(vec![0.25; 5]).unwrap())
            .unwrap();
        assert_eq!(
            rebound.execution_plan().arithmetic(),
            RegressionArithmetic::Standard
        );
        assert_eq!(rebound.execution_plan().execution(), execution);
    }
}

#[test]
fn standard_score_reduction_can_lose_a_residual_even_with_exact_predictors() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        // At beta=0 every predictor is exactly zero and every score multiplier
        // exactly 1/2. Exact gradient is 1e16 + 1 - 1e16 = 1, not zero.
        let d = RegressionData::new(1, vec![2e16, 2., -2e16], vec![true; 3]).unwrap();
        let reference = BernoulliLogitRegression::new(d.clone())
            .unwrap()
            .with_execution(execution);
        let standard = BernoulliLogitRegression::new(d)
            .unwrap()
            .with_execution(execution)
            .with_arithmetic(RegressionArithmetic::Standard);
        let mut rg = [0.];
        let mut sg = [0.];
        let rv = reference
            .evaluate(&[0.], &mut rg, &mut reference.workspace())
            .unwrap();
        let sv = standard
            .evaluate(&[0.], &mut sg, &mut standard.workspace())
            .unwrap();
        close(rv, sv, 1e-15);
        assert_eq!(rg[0].to_bits(), 1_f64.to_bits());
        assert_eq!(sg[0].to_bits(), 0_f64.to_bits());
    }
}

#[test]
fn standard_arithmetic_all_families_fd_shapes_allocations_and_rebinding() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        for rows in [1, 3, 4, 5, 33, 1000, 2500, 5000, 7500, 10000] {
            for columns in [0, 1, 3, 8] {
                let q: Vec<_> = (0..columns).map(|j| j as f64 / 32. - 0.125).collect();
                let nd = data(rows, columns, |i| (i % 7) as f64 / 4. - 0.5);
                let normal = NormalRegression::new(nd.clone())
                    .unwrap()
                    .with_execution(execution);
                let standard = NormalRegression::new(nd)
                    .unwrap()
                    .with_execution(execution)
                    .with_arithmetic(RegressionArithmetic::Standard);
                let standard = standard.with_data(standard.data().clone()).unwrap();
                assert_eq!(
                    standard.execution_plan().arithmetic(),
                    RegressionArithmetic::Standard
                );
                let mut nq = q.clone();
                nq.push(1.25);
                check(normal, standard, &nq);
                let ld = data(rows, columns, |i| i % 3 != 0);
                check(
                    BernoulliLogitRegression::new(ld.clone())
                        .unwrap()
                        .with_execution(execution),
                    BernoulliLogitRegression::new(ld)
                        .unwrap()
                        .with_execution(execution)
                        .with_arithmetic(RegressionArithmetic::Standard),
                    &q,
                );
                let pd = data(rows, columns, |i| (i % 4) as u64);
                let standard = PoissonLogRegression::new(pd.clone())
                    .unwrap()
                    .with_arithmetic(RegressionArithmetic::Standard)
                    .with_execution(execution);
                let standard = standard.with_data(standard.data().clone()).unwrap();
                assert_eq!(
                    standard.execution_plan().arithmetic(),
                    RegressionArithmetic::Standard
                );
                check(
                    PoissonLogRegression::new(pd)
                        .unwrap()
                        .with_execution(execution),
                    standard,
                    &q,
                );
            }
        }
    }
}

#[test]
fn standard_arithmetic_keeps_errors_tail_scores_and_transactional_recovery() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        let p = BernoulliLogitRegression::new(
            RegressionData::new(1, vec![1.; 5], vec![true; 5]).unwrap(),
        )
        .unwrap()
        .with_arithmetic(RegressionArithmetic::Standard)
        .with_execution(execution);
        let mut w = p.workspace();
        let mut g = [17.];
        let v = p.evaluate(&[40.], &mut g, &mut w).unwrap();
        assert!(g[0] > 0. && v < 0.);
        let saved = g;
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                p.evaluate(&[x], &mut g, &mut w),
                Err(E::Numerical)
            ));
            assert_eq!(g, saved);
        }
        assert!(matches!(p.evaluate(&[], &mut g, &mut w), Err(E::Shape)));
        assert_eq!(
            p.evaluate(&[40.], &mut g, &mut w).unwrap().to_bits(),
            v.to_bits()
        );
        let normal =
            NormalRegression::new(RegressionData::new(1, vec![1.; 5], vec![0.; 5]).unwrap())
                .unwrap()
                .with_arithmetic(RegressionArithmetic::Standard)
                .with_execution(execution);
        let mut nw = normal.workspace();
        let mut ng = [17.; 2];
        for scale in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(matches!(
                normal.evaluate(&[0., scale], &mut ng, &mut nw),
                Err(E::Numerical)
            ));
            assert_eq!(ng, [17.; 2]);
        }
        // A finite divided score must survive a nonrepresentable reciprocal.
        assert!(
            normal
                .evaluate(&[0., 1e200], &mut ng, &mut nw)
                .unwrap()
                .is_finite()
        );
        let poisson =
            PoissonLogRegression::new(RegressionData::new(1, vec![1.; 5], vec![1; 5]).unwrap())
                .unwrap()
                .with_arithmetic(RegressionArithmetic::Standard)
                .with_execution(execution);
        let mut pw = poisson.workspace();
        let mut pg = [17.];
        assert!(matches!(
            poisson.evaluate(&[800.], &mut pg, &mut pw),
            Err(E::Numerical)
        ));
        assert_eq!(pg, [17.]);
        assert!(
            poisson
                .evaluate(&[-800.], &mut pg, &mut pw)
                .unwrap()
                .is_finite()
        );
    }
}
fn check<P: OperationDescriptor>(auto: P, scalar: P, q: &[f64]) {
    let auto_rules = auto.rules();
    let scalar_rules = scalar.rules();
    assert_eq!(auto_rules.signature, scalar_rules.signature);
    let mut aw = (auto_rules.workspace)(&auto);
    let mut sw = (scalar_rules.workspace)(&scalar);
    let mut ag = vec![0.; q.len()];
    let mut sg = vec![0.; q.len()];
    let av = (auto_rules.evaluate)(&auto, q, &mut ag, &mut aw).unwrap();
    let sv = (scalar_rules.evaluate)(&scalar, q, &mut sg, &mut sw).unwrap();
    close(av, sv, 3e-12);
    for (&a, &s) in ag.iter().zip(&sg) {
        close(a, s, 3e-12);
    }
    // Each backend must differentiate its own evaluated objective consistently.
    for (primitive, rules, work, gradient) in [
        (&auto, &auto_rules, &mut aw, &ag),
        (&scalar, &scalar_rules, &mut sw, &sg),
    ] {
        for j in 0..q.len() {
            let mut plus = q.to_vec();
            let mut minus = q.to_vec();
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let fd = ((rules.evaluate)(primitive, &plus, &mut vec![0.; q.len()], work).unwrap()
                - (rules.evaluate)(primitive, &minus, &mut vec![0.; q.len()], work).unwrap())
                / 2e-5;
            close(fd, gradient[j], 2e-6);
        }
    }
    let allocations = allocation_counter::measure(|| {
        for _ in 0..4 {
            std::hint::black_box((auto_rules.evaluate)(&auto, q, &mut ag, &mut aw).unwrap());
            std::hint::black_box((scalar_rules.evaluate)(&scalar, q, &mut sg, &mut sw).unwrap());
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(
        (auto_rules.evaluate)(&auto, q, &mut ag, &mut aw)
            .unwrap()
            .to_bits(),
        av.to_bits()
    );
    assert_eq!(
        (scalar_rules.evaluate)(&scalar, q, &mut sg, &mut sw)
            .unwrap()
            .to_bits(),
        sv.to_bits()
    );
}

#[test]
fn all_families_agree_across_dispatch_shapes_tails_and_zero_columns() {
    for rows in [1, 3, 4, 5, 7, 8, 9, 33, 1_000, 10_000] {
        for columns in [0, 1, 3, 4, 5, 9] {
            let q: Vec<_> = (0..columns).map(|j| j as f64 / 32. - 0.125).collect();
            let nd = data(rows, columns, |i| (i % 7) as f64 / 4. - 0.5);
            let auto = NormalRegression::new(nd.clone()).unwrap();
            let scalar = NormalRegression::new(nd)
                .unwrap()
                .with_execution(RegressionExecution::Scalar);
            let mut nq = q.clone();
            nq.push(1.25);
            check(auto, scalar, &nq);
            let ld = data(rows, columns, |i| i % 3 != 0);
            check(
                BernoulliLogitRegression::new(ld.clone()).unwrap(),
                BernoulliLogitRegression::new(ld)
                    .unwrap()
                    .with_execution(RegressionExecution::Scalar),
                &q,
            );
            let pd = data(rows, columns, |i| (i % 4) as u64);
            check(
                PoissonLogRegression::new(pd.clone()).unwrap(),
                PoissonLogRegression::new(pd)
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
        let p = NormalRegression::new(d).unwrap().with_execution(execution);
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
        let root = b
            .primitive(&[&beta[..], &[sigma.expr()]].concat(), p)
            .unwrap();
        b.likelihood("y", root).unwrap();
        let m = b.compile().unwrap();
        let source = m.rust_primal().unwrap();
        // Same shape, different packing decision: old unit-column optimization
        // cannot survive an offset replacement.
        let m = m
            .binding()
            .rebind::<NormalRegression>(root, |p| {
                p.with_data(p.data().clone().with_offsets(vec![0.5; 5]).unwrap())
            })
            .unwrap()
            .compile()
            .unwrap()
            .into_native_model();
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
            .binding()
            .rebind::<NormalRegression>(root, |p| {
                p.with_data(RegressionData::new(1, vec![1.; 4], vec![0.; 4]).unwrap())
            })
            .err()
            .unwrap();
        assert!(matches!(failure, E::Shape));
        let descriptor = m
            .graph()
            .find(|n| n.expression == root)
            .unwrap()
            .primitive
            .unwrap()
            .descriptor::<NormalRegression>()
            .unwrap();
        assert_eq!(descriptor.execution_plan().execution(), execution);
        let target = m.into_target().unwrap();
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
    let logit =
        BernoulliLogitRegression::new(RegressionData::new(1, vec![1.; 5], vec![true; 5]).unwrap())
            .unwrap()
            .with_execution(RegressionExecution::Scalar);
    let poisson =
        PoissonLogRegression::new(RegressionData::new(1, vec![1.; 5], vec![2; 5]).unwrap())
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
    let original =
        NormalRegression::new(RegressionData::new(1, vec![1.; 1000], vec![0.; 1000]).unwrap())
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
        let rule = model.rules().batch.unwrap();
        (rule.evaluate)(model, &[0.25, 1.], &mut rows, &mut model.batch_workspace()).unwrap();
        close(value, rows.iter().sum(), 3e-12);
    }
}

#[test]
fn both_dispatch_modes_keep_tail_scores_domain_errors_and_recovery() {
    for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
        let normal =
            NormalRegression::new(RegressionData::new(1, vec![1.; 5], vec![0.; 5]).unwrap())
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

        let poisson =
            PoissonLogRegression::new(RegressionData::new(1, vec![1.; 5], vec![2; 5]).unwrap())
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
