use alea_core::{
    distribution::{EventShape, ReferenceMeasure, Support},
    model::{
        ConstrainedModel,
        assembly::{ArgumentShape, AssemblyError, ModelBuilder, OperationKind, Primitive},
    },
    target::LogDensityGradient,
};
use alea_distributions::modeling::{
    NormalFactor, normal,
    regression::{NormalRegression, RegressionData, normal_regression},
};

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
fn regression(native: bool, rows: usize) -> alea_core::model::assembly::CompiledModel {
    let mut b = ModelBuilder::new();
    let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
    let sigma = b.positive("sigma").unwrap();
    let d = data(rows);
    let f = if native {
        normal_regression(&mut b, &beta, &sigma, d).unwrap()
    } else {
        let primitive = NormalRegression::new(&beta, &sigma, d).unwrap();
        let inputs = primitive.inputs().to_vec();
        let workspace = primitive.workspace();
        b.custom_with_scratch(&inputs, workspace, move |x, g, w| {
            primitive.evaluate(x, g, w)
        })
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
    assert_eq!(primitive.inputs(), node.inputs);
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
            .contains(primitive.rust_primal().unwrap())
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
    let descriptor = NormalRegression::new(
        &[x],
        &sigma,
        RegressionData::new(1, vec![1.; 3], vec![0.; 3]).unwrap(),
    )
    .unwrap();
    let mut workspace = descriptor.workspace();
    let mut g = [77.; 2];
    assert!(matches!(
        descriptor.evaluate(&[1.], &mut g, &mut workspace),
        Err(AssemblyError::Shape)
    ));
    assert_eq!(g, [77.; 2]);
    let f = b.primitive(descriptor).unwrap();
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
