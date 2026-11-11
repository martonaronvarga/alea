use alea_core::{
    distribution::{EventShape, ReferenceMeasure, Support},
    model::{
        ConstrainedModel,
        assembly::{
            ArgumentShape, AssemblyError, ModelBuilder, OperationDescriptor, OperationKind,
        },
    },
    target::LogDensityGradient,
};
use alea_distributions::modeling::{
    NormalFactor, normal,
    regression::{NormalRegression, RegressionData, normal_regression},
};

// R0.3a: one definition, recovered between views, rather than independently
// rebuilding targets that could accidentally have different constants/priors.
mod query_chart_reporting {
    use super::*;
    use alea_core::model::assembly::{NativeModel, PointwiseLikelihood, PredictiveUnit};
    use alea_distributions::modeling::bernoulli_logit;

    #[derive(Clone, Copy)]
    enum View {
        Prior,
        Likelihood,
        Posterior,
        General,
    }
    impl View {
        fn select(self, model: NativeModel) -> NativeModel {
            match self {
                Self::Prior => model.select_prior().unwrap(),
                Self::Likelihood => model.select_likelihood().unwrap(),
                Self::Posterior => model.select_posterior().unwrap(),
                Self::General => model.select_all(),
            }
        }
    }

    fn definition(potential: bool) -> NativeModel {
        let mut b = ModelBuilder::new();
        let beta = b.real("beta").unwrap();
        let amplitude = b.positive("amplitude").unwrap();
        let sigma = b.positive("sigma").unwrap();
        let square = b.multiply(amplitude, amplitude).unwrap();
        for (i, (x, y, event)) in [(-0.7, 0.2, true), (1.3, -1.1, false)]
            .into_iter()
            .enumerate()
        {
            let x = b.constant(x).unwrap();
            let effect = b.multiply(square, x).unwrap();
            let eta = b.add(beta, effect).unwrap();
            let n = normal(&mut b, y, eta, sigma).unwrap();
            let l = bernoulli_logit(&mut b, event, eta).unwrap();
            b.likelihood(&format!("n{i}"), n).unwrap();
            b.likelihood(&format!("l{i}"), l).unwrap();
            if i == 0 {
                // A second observation with the same value/predictor shares the
                // entire factor node, but its density must still be counted.
                b.likelihood("replica", n).unwrap();
            }
        }
        let prior = normal(&mut b, beta, 0., 2.).unwrap();
        b.prior("beta prior", prior).unwrap();
        for (name, value) in [
            ("amplitude prior", amplitude.expr()),
            ("scale prior", sigma.expr()),
        ] {
            let prior = normal(&mut b, value, 0., 1.).unwrap();
            let two = b.constant(2_f64.ln()).unwrap();
            let prior = b.add(prior, two).unwrap();
            b.prior(name, prior).unwrap();
        }
        if potential {
            let weight = b.constant(-0.3).unwrap();
            let penalty = b.multiply(weight, square).unwrap();
            b.potential("regularizer, not a prior", penalty).unwrap();
        }
        b.compile().unwrap()
    }

    // Independent full densities and analytic gradients in free coordinates.
    // Neither Alea's laws, chart helpers nor evaluator supply this oracle.
    fn reference(q: [f64; 3], view: View, density: bool, potential: bool) -> (f64, [f64; 3]) {
        let [beta, log_a, log_s] = q;
        let a2 = (2. * log_a).exp();
        let s2 = (2. * log_s).exp();
        let c = 0.5 * std::f64::consts::TAU.ln();
        let mut lp = 0.;
        let mut g = [0.; 3];
        if !matches!(view, View::Prior) {
            // likelihood or full sum
            for (i, (x, y, event)) in [(-0.7, 0.2, true), (1.3, -1.1, false)]
                .into_iter()
                .enumerate()
            {
                let eta = beta + a2 * x;
                let residual = y - eta;
                let copies = if i == 0 { 2. } else { 1. };
                lp += copies * (-c - log_s - residual * residual / (2. * s2));
                let sign = if event { 1. } else { -1. };
                // Direct log-sigmoid of the observed sign: no eta-softplus
                // cancellation, independent of the production link kernel.
                lp -= (1. + (-sign * eta).exp()).ln();
                let score = copies * residual / s2 + sign / (1. + (sign * eta).exp());
                g[0] += score;
                g[1] += score * 2. * a2 * x;
                g[2] += copies * (-1. + residual * residual / s2);
            }
        }
        if !matches!(view, View::Likelihood) {
            // prior or full sum
            lp += -beta * beta / 8. - 2_f64.ln() - c - 0.5 * a2 - c + 2_f64.ln() - 0.5 * s2 - c
                + 2_f64.ln();
            g[0] -= beta / 4.;
            g[1] -= a2;
            g[2] -= s2;
        }
        if potential {
            lp -= 0.3 * a2;
            g[1] -= 0.6 * a2;
        }
        if density {
            lp += log_a + log_s;
            g[1] += 1.;
            g[2] += 1.;
        }
        (lp, g)
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 2e-11 * (1. + b.abs()), "{a} != {b}");
    }

    #[test]
    fn one_definition_preserves_full_constants_and_exactly_one_chart_for_every_view() {
        let mut model = definition(false);
        let states = [[0.2, -0.4, 0.3], [-1.1, 0.5, -0.6], [2., -1., 0.8]];
        for view in [
            View::Prior,
            View::Likelihood,
            View::Posterior,
            View::General,
        ] {
            model = view.select(model);
            assert_eq!(model.factors().len(), 8);
            assert_eq!(model.coordinates().len(), 3);
            let objective = model.into_objective().unwrap();
            for q in states {
                let (lp, g) = reference(q, view, false, false);
                let mut actual = [99.; 3];
                close(objective.logp_grad(&q, &mut actual).unwrap(), lp);
                for (a, b) in actual.into_iter().zip(g) {
                    close(a, b);
                }
            }
            let target = objective.into_model().into_target().unwrap();
            for q in states {
                let (lp, g) = reference(q, view, true, false);
                let mut actual = [99.; 3];
                close(target.logp_grad(&q, &mut actual).unwrap(), lp);
                for k in 0..3 {
                    close(actual[k], g[k]);
                    for h in [1e-4, 1e-5, 1e-6] {
                        let mut plus = q;
                        let mut minus = q;
                        plus[k] += h;
                        minus[k] -= h;
                        let fd = (reference(plus, view, true, false).0
                            - reference(minus, view, true, false).0)
                            / (2. * h);
                        assert!((actual[k] - fd).abs() < 1e-6 * (1. + fd.abs()));
                    }
                }
            }
            model = target.into_model();
        }
        let unit = |name: &str, factors: &[&str], observations: &[&str]| PredictiveUnit {
            name: name.into(),
            factors: factors.iter().map(|s| (*s).into()).collect(),
            observations: observations.iter().map(|s| (*s).into()).collect(),
            conditioned_on: vec!["beta".into(), "amplitude".into(), "sigma".into()],
            integrated_out: vec![],
        };
        let pointwise = PointwiseLikelihood::new(
            model,
            vec![
                unit("group B", &["l1", "n1"], &["binary1", "continuous1"]),
                unit(
                    "group A",
                    &["replica", "l0", "n0"],
                    &["replicate0", "binary0", "continuous0"],
                ),
            ],
        )
        .unwrap();
        for q in states {
            let mut out = [99.; 2];
            close(
                pointwise.evaluate(&q, &mut out).unwrap(),
                reference(q, View::Likelihood, false, false).0,
            );
            let eta = q[0] + 1.3 * (2. * q[1]).exp();
            let group_b = -0.5 * std::f64::consts::TAU.ln()
                - q[2]
                - (-1.1 - eta).powi(2) / (2. * (2. * q[2]).exp())
                - (1. + eta.exp()).ln();
            close(out[0], group_b);
            close(
                out[1],
                reference(q, View::Likelihood, false, false).0 - group_b,
            );
        }
        assert_eq!(pointwise.units()[0].name, "group B");
        assert!(
            pointwise
                .units()
                .iter()
                .all(|u| u.integrated_out.is_empty())
        );
        let restored = pointwise
            .into_model()
            .select_posterior()
            .unwrap()
            .into_target()
            .unwrap();
        close(
            restored.logp_grad(&states[0], &mut [0.; 3]).unwrap(),
            reference(states[0], View::Posterior, true, false).0,
        );
    }

    #[test]
    fn custom_potential_stays_a_general_target_not_an_implied_posterior() {
        assert!(matches!(
            definition(true).select_posterior(),
            Err(AssemblyError::PosteriorPotential { index: 8 })
        ));
        let mut model = definition(true);
        let q = [0.2, -0.4, 0.3];
        for view in [View::Prior, View::Likelihood, View::General] {
            model = view.select(model);
            let target = model.into_target().unwrap();
            let (lp, g) = reference(q, view, true, matches!(view, View::General));
            let mut actual = [99.; 3];
            close(target.logp_grad(&q, &mut actual).unwrap(), lp);
            for (a, b) in actual.into_iter().zip(g) {
                close(a, b);
            }
            model = target.into_model();
        }
    }
}

#[test]
fn generic_observation_uses_the_fused_parameter_only_entry_once() {
    use alea_core::distribution::{Distribution, DistributionError, ParameterScore};
    use alea_distributions::{modeling::observation, scalar::BernoulliLogit};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Counted {
        law: BernoulliLogit,
        calls: Arc<AtomicUsize>,
    }
    impl Distribution for Counted {
        type Event = bool;
        type Error = DistributionError;
        fn event_shape(&self) -> EventShape {
            self.law.event_shape()
        }
        fn support(&self) -> Support {
            self.law.support()
        }
        fn reference_measure(&self) -> ReferenceMeasure {
            self.law.reference_measure()
        }
        fn log_prob(&self, x: &bool) -> Result<f64, Self::Error> {
            self.calls.fetch_add(100, Ordering::Relaxed);
            self.law.log_prob(x)
        }
    }
    impl ParameterScore for Counted {
        fn parameter_count(&self) -> usize {
            1
        }
        fn parameter_score(&self, x: &bool, g: &mut [f64]) -> Result<(), Self::Error> {
            self.calls.fetch_add(100, Ordering::Relaxed);
            self.law.parameter_score(x, g)
        }
        fn log_prob_parameter_score(&self, x: &bool, g: &mut [f64]) -> Result<f64, Self::Error> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.law.log_prob_parameter_score(x, g)
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let mut b = ModelBuilder::new();
    let eta = b.real("eta").unwrap().expr();
    let f = observation(&mut b, true, &[eta], move |x| {
        Ok(Counted {
            law: BernoulliLogit::new(x[0])?,
            calls: Arc::clone(&count),
        })
    })
    .unwrap();
    b.likelihood("response", f).unwrap();
    let model = b.compile().unwrap();
    let mut g = [0.];
    let lp = model.logp_grad(&[0.], &mut g).unwrap();
    assert_eq!(lp, -2_f64.ln());
    assert_eq!(g, [0.5]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn shared_normal_preparation_matches_scalar_callbacks_and_preserves_first_failure() {
    fn model(shared: bool) -> alea_core::model::assembly::NativeModel {
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let sigma = b.real("sigma").unwrap().expr();
        for i in 0..9 {
            let mu = b.constant(i as f64 / 4.).unwrap();
            let p = NormalFactor::new();
            let f = if shared {
                b.primitive(&[x, mu, sigma], p).unwrap()
            } else {
                let inputs = [x, mu, sigma];
                b.custom(&inputs, move |x, g| p.evaluate(x, g, &mut ()))
                    .unwrap()
            };
            b.likelihood(&format!("row{i}"), f).unwrap();
        }
        b.compile().unwrap()
    }
    let a = model(true);
    let b = model(false);
    for q in [
        [0., 1.],
        [3., 1e-150],
        [0., 0.],
        [1., -1.],
        [1e200, 1.],
        [0.2, 2.],
        [0.2, 2.],
    ] {
        let mut ga = [17.; 2];
        let mut gb = ga;
        let ra = a.logp_grad(&q, &mut ga);
        let rb = b.logp_grad(&q, &mut gb);
        assert_eq!(format!("{ra:?}"), format!("{rb:?}"));
        assert_eq!(ga.map(f64::to_bits), gb.map(f64::to_bits));
    }
    let allocation = allocation_counter::measure(|| {
        for sigma in [1., 2., 3., 1.] {
            a.logp_grad(&[0.2, sigma], &mut [0.; 2]).unwrap();
        }
    });
    assert_eq!(allocation.count_total, 0);
}

fn data(rows: usize) -> RegressionData<f64> {
    RegressionData::new(
        2,
        (0..rows).flat_map(|i| [1., (i as f64 - 2.) / 4.]).collect(),
        (0..rows).map(|i| (i as f64 - 1.) / 3.).collect(),
    )
    .unwrap()
    .with_offsets((0..rows).map(|i| i as f64 / 10.).collect())
    .unwrap()
}
fn regression(native: bool, rows: usize) -> alea_core::model::assembly::NativeModel {
    let mut b = ModelBuilder::new();
    let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
    let sigma = b.positive("sigma").unwrap();
    let d = data(rows);
    let f = if native {
        normal_regression(&mut b, &beta, &sigma, d).unwrap()
    } else {
        let primitive = NormalRegression::new(d).unwrap();
        let inputs = [&beta[..], &[sigma.expr()]].concat();
        let primitive = std::sync::Arc::new(primitive);
        let binding = std::sync::Arc::clone(&primitive);
        b.custom_with_scratch(
            &inputs,
            move || binding.workspace(),
            move |x, g, w| primitive.evaluate(x, g, w),
        )
        .unwrap()
    };
    b.likelihood("observations", f).unwrap();
    b.compile().unwrap()
}

#[test]
fn native_normal_is_recognizable_and_preserves_the_existing_primal_and_density_bits() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let location = b.constant(0.5).unwrap();
    let scale = b.constant(2.).unwrap();
    let f = normal(&mut b, x, location, scale).unwrap();
    b.prior("prior", f).unwrap();
    let model = b.compile().unwrap();
    let node = model.graph().find(|n| n.expression == f).unwrap();
    let metadata = node.primitive.unwrap();
    let primitive = metadata.descriptor::<NormalFactor>().unwrap();
    assert_eq!(node.inputs, [x, location, scale]);
    assert_eq!(node.inputs.len(), 3);
    assert!(metadata.descriptor::<NormalRegression>().is_none());
    assert_eq!(
        metadata
            .signature()
            .arguments
            .iter()
            .map(|a| a.name)
            .collect::<Vec<_>>(),
        ["event", "location", "scale"]
    );
    let probability = metadata.signature().probability.as_ref().unwrap();
    assert_eq!(probability.event, EventShape::Scalar);
    assert!(probability.batch.is_empty());
    assert_eq!(probability.support, Support::Real);
    assert_eq!(
        probability.measure,
        ReferenceMeasure::Lebesgue { dimension: 1 }
    );
    assert!(model.source_capabilities().supports_rust_primal());
    assert!(
        model
            .rust_primal()
            .unwrap()
            .contains(&primitive.rules().source.unwrap().body.render(primitive))
    );
    for value in [-10., 0., 2., 1e50] {
        let mut local_gradient = [0.; 3];
        let expected = primitive
            .evaluate(&[value, 0.5, 2.], &mut local_gradient, &mut ())
            .unwrap();
        let mut gradient = [0.];
        assert_eq!(
            model.logp_grad(&[value], &mut gradient).unwrap().to_bits(),
            expected.to_bits()
        );
        assert_eq!(gradient[0].to_bits(), local_gradient[0].to_bits());
    }
}

#[test]
fn regression_descriptor_preserves_shapes_data_offsets_and_one_fused_node() {
    let mut b = ModelBuilder::new();
    let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
    let sigma = b.positive("sigma").unwrap();
    let d = data(10_000);
    let original = d.clone();
    let f = normal_regression(&mut b, &beta, &sigma, d).unwrap();
    b.likelihood("data", f).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.expression_count(), 4);
    let node = model.graph().find(|n| n.expression == f).unwrap();
    assert_eq!(node.operation, OperationKind::Primitive);
    assert_eq!(node.inputs, [beta[0], beta[1], sigma.expr()]);
    assert_eq!(node.factors[0].name(), Some("data"));
    let metadata = node.primitive.unwrap();
    let descriptor = metadata.descriptor::<NormalRegression>().unwrap();
    assert_eq!(
        descriptor.data().design().as_ptr(),
        original.design().as_ptr()
    );
    assert_eq!(
        descriptor.data().observations().as_ptr(),
        original.observations().as_ptr()
    );
    assert_eq!(
        descriptor.data().offsets().as_ptr(),
        original.offsets().as_ptr()
    );
    assert_eq!(
        metadata.signature().arguments[0].shape,
        ArgumentShape::Vector(2)
    );
    assert_eq!(
        metadata.signature().arguments[1].shape,
        ArgumentShape::Scalar
    );
    let probability = metadata.signature().probability.as_ref().unwrap();
    assert_eq!(probability.event, EventShape::Scalar);
    assert_eq!(probability.batch, [10_000]);
    assert_eq!(probability.measure.dimension(), Some(1)); // Not a 10,000-dimensional event.
    assert!(model.source_capabilities().supports_rust_primal());
    assert!(model.rust_primal().unwrap().contains("for i in 0..10000"));
}

#[test]
fn registration_matches_opaque_lowering_bitwise_and_allocates_nothing_during_evaluation() {
    for rows in [1, 7, 1_000, 10_000] {
        let native = regression(true, rows).into_target().unwrap();
        let opaque = regression(false, rows).into_target().unwrap();
        for q in [[0.2, -0.4, 0.3], [-2., 1., -3.], [1e-30, 0., 10.]] {
            let mut a = [0.; 3];
            let mut b = [0.; 3];
            let expected = opaque.logp_grad(&q, &mut b).unwrap();
            assert_eq!(
                native.logp_grad(&q, &mut a).unwrap().to_bits(),
                expected.to_bits()
            );
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
            let allocations = allocation_counter::measure(|| {
                for _ in 0..8 {
                    std::hint::black_box(native.logp_grad(&q, &mut a).unwrap());
                }
            });
            assert_eq!(allocations.count_total, 0);
        }
    }
}

#[test]
fn full_posterior_matches_independent_density_gradient_and_single_sigma_jacobian() {
    let mut b = ModelBuilder::new();
    let a = b.real("a").unwrap();
    let slope = b.real("b").unwrap();
    let sigma = b.positive("sigma").unwrap();
    for (name, parameter) in [("a prior", a.expr()), ("b prior", slope.expr())] {
        let f = normal(&mut b, parameter, 0., 2.).unwrap();
        b.prior(name, f).unwrap();
    }
    let f = normal(&mut b, sigma, 0., 1.).unwrap();
    let log_two = b.constant(2_f64.ln()).unwrap();
    let half_normal = b.add(f, log_two).unwrap();
    b.prior("scale prior", half_normal).unwrap();
    let d = data(7);
    let f = normal_regression(&mut b, &[a.expr(), slope.expr()], &sigma, d.clone()).unwrap();
    b.likelihood("data", f).unwrap();
    let target = b
        .compile()
        .unwrap()
        .select_posterior()
        .unwrap()
        .into_target()
        .unwrap();
    let reference = |q: [f64; 3]| {
        let sigma = q[2].exp();
        let c = 0.5 * std::f64::consts::TAU.ln();
        let mut logp = -0.5 * (q[0] / 2.).powi(2)
            - 2_f64.ln()
            - c
            - 0.5 * (q[1] / 2.).powi(2)
            - 2_f64.ln()
            - c
            - 0.5 * sigma * sigma
            - c
            + 2_f64.ln()
            + q[2];
        for i in 0..d.rows() {
            let eta = q[0] + d.design()[2 * i + 1] * q[1] + d.offsets()[i];
            logp += -0.5 * ((d.observations()[i] - eta) / sigma).powi(2) - q[2] - c;
        }
        logp
    };
    for q in [[0.2, -0.4, 0.3], [-0.5, 0.1, -2.], [1., -1., 2.]] {
        let mut gradient = [0.; 3];
        let value = target.logp_grad(&q, &mut gradient).unwrap();
        assert!((value - reference(q)).abs() < 1e-10 * (1. + value.abs()));
        for i in 0..3 {
            let mut plus = q;
            let mut minus = q;
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let fd = (reference(plus) - reference(minus)) / 2e-5;
            assert!((fd - gradient[i]).abs() < 1e-7 * (1. + fd.abs()));
        }
    }
}

#[test]
fn selection_preserves_descriptors_but_capabilities_only_require_active_nodes() {
    let mut b = ModelBuilder::new();
    let a = b.real("a").unwrap();
    let sigma = b.positive("sigma").unwrap();
    let batch = normal_regression(
        &mut b,
        &[a.expr()],
        &sigma,
        RegressionData::new(1, vec![1.; 3], vec![0.; 3]).unwrap(),
    )
    .unwrap();
    b.prior("batch prior", batch).unwrap();
    let f = normal(&mut b, a, 0., 1.).unwrap();
    b.likelihood("scalar likelihood", f).unwrap();
    let model = b.compile().unwrap();
    assert!(model.source_capabilities().supports_rust_primal());
    let model = model.select_likelihood().unwrap();
    let inactive = model.graph().find(|n| n.expression == batch).unwrap();
    assert!(!inactive.active);
    assert!(
        inactive
            .primitive
            .unwrap()
            .descriptor::<NormalRegression>()
            .is_some()
    );
    assert_eq!(inactive.factors[0].name(), Some("batch prior"));
    assert!(model.source_capabilities().supports_rust_primal());
    assert!(model.rust_primal().is_ok());
    let restored = model.select_all();
    assert!(restored.source_capabilities().supports_rust_primal());
}

#[test]
fn descriptor_shape_errors_numerical_errors_and_recovery_are_explicit() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let sigma = b.positive("sigma").unwrap();
    let descriptor =
        NormalRegression::new(RegressionData::new(1, vec![1.; 3], vec![0.; 3]).unwrap()).unwrap();
    let mut workspace = descriptor.workspace();
    let mut g = [77.; 2];
    assert!(matches!(
        descriptor.evaluate(&[1.], &mut g, &mut workspace),
        Err(AssemblyError::Shape)
    ));
    assert_eq!(g, [77.; 2]);
    let f = b.primitive(&[x, sigma.expr()], descriptor).unwrap();
    b.likelihood("data", f).unwrap();
    let model = b.compile().unwrap();
    for bad in [0., -1., 1e-300, f64::INFINITY] {
        assert!(model.logp_grad(&[1., bad], &mut g).is_err());
        assert_eq!(g, [77.; 2]);
    }
    let target = model.into_target().unwrap();
    assert!(target.logp_grad(&[1., -1e3], &mut g).is_err());
    assert_eq!(g, [77.; 2]);
    assert!(target.logp_grad(&[1., 0.], &mut g).unwrap().is_finite());
    assert!(g.iter().all(|x| x.is_finite()));
}

#[test]
fn joint_batch_reporting_preserves_semantics_and_excludes_the_chart_jacobian() {
    use alea_core::model::assembly::{PointwiseLikelihood, PredictiveUnit};
    let model = regression(true, 7);
    let q: [f64; 3] = [0.2, -0.4, 0.3];
    let mut gradient = [0.; 3];
    let constrained = [q[0], q[1], q[2].exp()];
    let expected = model.logp_grad(&constrained, &mut gradient).unwrap();
    let pointwise = PointwiseLikelihood::new(
        model,
        vec![PredictiveUnit {
            name: "joint data".into(),
            factors: vec!["observations".into()],
            observations: (0..7).map(|i| format!("row{i}")).collect(),
            conditioned_on: vec!["a".into(), "b".into(), "sigma".into()],
            integrated_out: vec![],
        }],
    )
    .unwrap();
    let mut out = [0.];
    assert_eq!(
        pointwise.evaluate(&q, &mut out).unwrap().to_bits(),
        expected.to_bits()
    );
    assert_eq!(out[0].to_bits(), expected.to_bits());
    assert_eq!(pointwise.units().len(), 1);
    let model = pointwise.into_model();
    assert!(model.graph().any(|n| {
        n.primitive
            .is_some_and(|p| p.descriptor::<NormalRegression>().is_some())
    }));
    let posterior = model.select_posterior().unwrap().into_target().unwrap();
    assert!((posterior.logp_grad(&q, &mut gradient).unwrap() - expected - q[2]).abs() < 1e-12);
}

#[test]
fn repeated_derived_coefficients_preserve_graph_sharing_and_reverse_accumulation() {
    let mut b = ModelBuilder::new();
    let a = b.real("a").unwrap();
    let sigma = b.positive("sigma").unwrap();
    let square = b.multiply(a, a).unwrap();
    let f = normal_regression(&mut b, &[square, square], &sigma, data(7)).unwrap();
    b.likelihood("data", f).unwrap();
    let model = b.compile().unwrap();
    let node = model.graph().find(|n| n.expression == f).unwrap();
    assert_eq!(node.inputs, [square, square, sigma.expr()]);
    let target = model.into_target().unwrap();
    let q = [0.4, -0.2];
    let mut gradient = [0.; 2];
    target.logp_grad(&q, &mut gradient).unwrap();
    for i in 0..2 {
        let mut plus = q;
        let mut minus = q;
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        let difference = (target.logp_grad(&plus, &mut [0.; 2]).unwrap()
            - target.logp_grad(&minus, &mut [0.; 2]).unwrap())
            / 2e-5;
        assert!((difference - gradient[i]).abs() < 1e-7 * (1. + difference.abs()));
    }
}
