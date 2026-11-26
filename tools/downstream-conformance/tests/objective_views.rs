//! External facade consumer: one model, explicit statistical views.
use alea::{
    distributions::modeling::{normal, normal_observation},
    model::assembly::{AssemblyError, ModelBuilder},
    target::LogDensityGradient,
};

#[test]
fn a_single_owned_model_supports_likelihood_and_posterior_evaluation() -> Result<(), AssemblyError>
{
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let sigma = b.positive("sigma")?;
    let zero = b.constant(0.)?;
    let one = b.constant(1.)?;
    let prior = normal(&mut b, mean.expr(), zero, one)?;
    b.prior("mean prior", prior)?;
    let prior = normal(&mut b, sigma.expr(), zero, one)?;
    b.prior("scale prior", prior)?;
    let normalizer = b.constant(std::f64::consts::LN_2)?;
    b.prior("half-normal normalization", normalizer)?;
    for (i, y) in [2., 3.].into_iter().enumerate() {
        let observation = normal_observation(&mut b, y, mean.expr(), &sigma)?;
        b.likelihood(&format!("observation {i}"), observation)?;
    }
    let objective = b.compile()?.select_likelihood()?.into_objective().unwrap();
    let q = [1., 2_f64.ln()];
    let mut g = [0.; 2];
    let likelihood = objective.logp_grad(&q, &mut g).unwrap();
    assert_eq!(g, [0.75, -0.75]);
    assert_eq!(objective.model().selected_factors().count(), 2);
    let posterior = objective
        .into_model()
        .select_posterior()?
        .into_objective()
        .unwrap();
    let posterior_value = posterior.logp_grad(&q, &mut g).unwrap();
    assert_eq!(g, [-0.25, -4.75]);
    let prior_value = -2.5 - (2. * std::f64::consts::PI).ln() + std::f64::consts::LN_2;
    assert!((posterior_value - likelihood - prior_value).abs() < 1e-13);
    let target = posterior.into_model().into_target().unwrap();
    assert_eq!(
        target.logp_grad(&q, &mut g).unwrap(),
        posterior_value + q[1]
    );
    assert_eq!(g, [-0.25, -3.75]);
    assert_eq!(target.model().coordinates()[1].name, "sigma");
    Ok(())
}

fn registered_laws_model() -> alea::model::assembly::NativeModel {
    use alea::{
        distributions::modeling::{
            bernoulli_logit_region_observations, normal_region_observations,
            poisson_log_region_observations,
        },
        model::tensor::{BinaryOp, Shape},
    };

    let mut b = ModelBuilder::new();
    let beta = b.real("beta").unwrap().expr();
    let scale = b.positive("scale").unwrap().expr();
    let interval = b.unit_interval("offset").unwrap().expr();
    let shared = b.multiply(beta, beta).unwrap();
    let predictor = b.add(shared, interval).unwrap();
    let shape = Shape::new(vec![3]).unwrap();
    let mut map = b.region();
    let x = map.input("predictor", Shape::scalar()).unwrap();
    let design = map.constant(shape.clone(), vec![0.5, -1., 2.]).unwrap();
    let eta = map.binary(BinaryOp::Multiply, design, x).unwrap();
    let eta = b
        .call_region(&[predictor], map.compile(eta).unwrap().bind(&[]).unwrap())
        .unwrap();

    for law in 0..3 {
        let mut r = b.region();
        let x = r.input("eta", shape.clone()).unwrap();
        let (rows, args) = match law {
            0 => {
                let s = r.input("scale", Shape::scalar()).unwrap();
                (
                    normal_region_observations(&mut r, vec![0.5, -0.6, 0.125], x, s).unwrap(),
                    vec![eta, scale],
                )
            }
            1 => (
                bernoulli_logit_region_observations(&mut r, vec![true, false, true], x).unwrap(),
                vec![eta],
            ),
            _ => (
                poisson_log_region_observations(&mut r, vec![0_u64, 1, 3], x).unwrap(),
                vec![eta],
            ),
        };
        // Keep the observation vector crossing the call boundary as well.
        let observations = b
            .call_region(&args, r.compile(rows).unwrap().bind(&[]).unwrap())
            .unwrap();
        let log_prob = b.sum_value(observations).unwrap();
        b.likelihood(&format!("law {law}"), log_prob).unwrap();
    }
    let opaque = b
        .custom(&[beta], |_, _| panic!("unselected prior"))
        .unwrap();
    b.prior("opaque", opaque).unwrap();
    b.compile().unwrap()
}

#[test]
fn selected_registered_laws_have_native_curvature_across_model_calls_and_charts() {
    use alea::{
        capability::{PotentialHessianVector, evaluate_hvp},
        model::assembly::{CurvatureRoute, DerivativeOrder, SourceMode, SourceRequest},
    };
    let model = registered_laws_model();
    assert!(matches!(
        model.objective_with_hvp(),
        Err(AssemblyError::MissingNativeHvp { source: None, .. })
    ));
    let model = model.select_likelihood().unwrap();
    assert!(matches!(
        model.source_capabilities().resolve(SourceRequest {
            mode: SourceMode::Objective,
            order: DerivativeOrder::Hvp,
            curvature: CurvatureRoute::Unavailable,
        }),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    let ordinary = model
        .executable()
        .clone()
        .into_native_model()
        .into_objective()
        .unwrap();
    let objective = model.objective_with_hvp().unwrap();
    let target = model.target_with_hvp().unwrap();
    let q = [0.35_f64, -0.25, -0.4];
    let v = [0.7, -0.5, 0.2];
    let s = 1. / (1. + (-q[2]).exp());
    let sp = s * (1. - s);
    let scale = q[1].exp();
    let p = q[0] * q[0] + s;
    let dp = 2. * q[0] * v[0] + sp * v[2];
    let (mut score, mut dscore, mut dlog_scale) = (0., 0., 0.);
    for (i, x) in [0.5, -1., 2.].into_iter().enumerate() {
        let eta = x * p;
        let deta = x * dp;
        let residual = [0.5, -0.6, 0.125][i] - eta;
        let rate = eta.exp();
        let logit_p = rate / (1. + rate);
        let gaussian_score = residual / scale.powi(2);
        score += x * (gaussian_score + [1., 0., 1.][i] - logit_p + [0., 1., 3.][i] - rate);
        dscore += x
            * (-deta * (1. / scale.powi(2) + logit_p * (1. - logit_p) + rate)
                - 2. * gaussian_score * v[1]);
        dlog_scale -= 2. * gaussian_score * deta + 2. * (residual / scale).powi(2) * v[1];
    }
    let expected = [
        -(2. * q[0] * dscore + 2. * v[0] * score),
        -dlog_scale,
        -(sp * dscore + sp * (1. - 2. * s) * v[2] * score),
    ];
    let (mut og, mut g, mut hvp, mut density) = ([0.; 3], [0.; 3], [0.; 3], [0.; 3]);
    assert_eq!(
        ordinary.logp_grad(&q, &mut og).unwrap().to_bits(),
        objective.logp_grad(&q, &mut g).unwrap().to_bits()
    );
    assert_eq!(og.map(f64::to_bits), g.map(f64::to_bits));
    evaluate_hvp(&objective, &q, &v, &mut hvp).unwrap();
    evaluate_hvp(&target, &q, &v, &mut density).unwrap();
    let (mut gp, mut gm) = ([0.; 3], [0.; 3]);
    let h = 1e-5;
    ordinary
        .logp_grad(
            &std::array::from_fn::<_, 3, _>(|i| q[i] + h * v[i]),
            &mut gp,
        )
        .unwrap();
    ordinary
        .logp_grad(
            &std::array::from_fn::<_, 3, _>(|i| q[i] - h * v[i]),
            &mut gm,
        )
        .unwrap();
    for i in 0..3 {
        let jacobian = if i == 2 { 2. * sp * v[i] } else { 0. };
        assert!((hvp[i] - expected[i]).abs() < 1e-12);
        assert!((hvp[i] + (gp[i] - gm[i]) / (2. * h)).abs() < 1e-8);
        assert!((density[i] - expected[i] - jacobian).abs() < 1e-12);
    }
    let mut unchanged = [19.; 3];
    assert!(
        objective
            .potential_hvp(&[q[0], -1000., q[2]], &[0.; 3], &mut unchanged)
            .is_err()
    );
    assert_eq!(unchanged, [19.; 3]);
    objective.potential_hvp(&q, &v, &mut unchanged).unwrap();
    assert_eq!(unchanged, hvp);
}

#[test]
fn first_order_regions_and_nonlinear_vector_charts_report_missing_native_routes() {
    use alea::{
        model::{
            assembly::Simplex,
            tensor::{BlockRules, BlockType, ElementType, Shape},
        },
        transform::Transform,
    };
    let mut builder = ModelBuilder::new();
    let q = builder.real("q").unwrap().expr();
    let operation = builder
        .register_block(
            &[q],
            (),
            BlockRules {
                inputs: vec![BlockType {
                    dtype: ElementType::Real,
                    shape: Shape::scalar(),
                }],
                output: Shape::scalar(),
                workspace: |_| (),
                heap_bytes: 0,
                preparation: None,
                forward: |_, input, out, _| {
                    out[0] = input.real(0)?[0];
                    Ok(())
                },
                vjp: Some(|_, _, _, seed, g, _| {
                    g[0] = seed[0];
                    Ok(())
                }),
            },
        )
        .unwrap();
    builder.factor(operation).unwrap();
    let first = builder.compile().unwrap();
    assert!(matches!(
        first.target_with_hvp(),
        Err(AssemblyError::MissingNativeHvp {
            source: Some(_),
            ..
        })
    ));
    assert_eq!(
        first
            .into_target()
            .unwrap()
            .logp_grad(&[2.], &mut [0.])
            .unwrap(),
        2.
    );

    let mut builder = ModelBuilder::new();
    let simplex = builder
        .vector::<Simplex>(
            "p",
            "i",
            &["a", "b"],
            "probabilities",
            Transform::simplex(2).unwrap(),
        )
        .unwrap();
    builder.factor(simplex.expressions()[0]).unwrap();
    let model = builder.compile().unwrap();
    assert!(matches!(
        model.objective_with_hvp(),
        Err(AssemblyError::NativeHvpChart { block: 0 })
    ));
    assert!(
        model
            .into_objective()
            .unwrap()
            .logp_grad(&[0.], &mut [0.])
            .is_ok()
    );
}

#[test]
fn native_curvature_uses_original_scalar_seeds_and_recovers_after_nonfinite_direction() {
    use alea::capability::PotentialHessianVector;

    let mut b = ModelBuilder::new();
    let q = b.real("q").unwrap().expr();
    let exp = b.exp(q).unwrap();
    let log = b.log(q).unwrap();
    let product = b.multiply(exp, log).unwrap();
    b.factor(product).unwrap();
    let model = b.compile().unwrap();
    let target = model.target_with_hvp().unwrap();
    let mut result = [19.];
    assert!(target.potential_hvp(&[0.], &[0.], &mut result).is_err());
    assert!(
        target
            .potential_hvp(&[1.], &[f64::INFINITY], &mut result)
            .is_err()
    );
    assert_eq!(result, [19.]);
    target.potential_hvp(&[1.], &[2.], &mut result).unwrap();
    assert!((result[0] + 2. * std::f64::consts::E).abs() < 1e-14);
}

#[test]
fn rewrapping_a_resolved_model_checks_the_new_chart_and_differentiates_bounds() {
    use alea::{
        capability::PotentialHessianVector,
        model::{ComposedObjective, ModelError},
        target::EvaluationError,
        transform::{ParameterLayout, Transform},
    };
    let mut b = ModelBuilder::new();
    let a = b.real("a").unwrap().expr();
    let c = b.real("c").unwrap().expr();
    let root = b.multiply(a, c).unwrap();
    b.factor(root).unwrap();
    let model = b.compile().unwrap();
    let resolved = model.objective_with_hvp().unwrap().into_model();
    let layout = ParameterLayout::new([Transform::simplex(2).unwrap()]).unwrap();
    let objective = ComposedObjective::new(resolved, layout).unwrap();
    let mut output = [19.];
    assert!(matches!(
        objective.potential_hvp(&[0.], &[1.], &mut output),
        Err(ModelError::Evaluation(EvaluationError::Model(
            AssemblyError::NativeHvpChart { block: 0 }
        )))
    ));
    assert_eq!(output, [19.]);
    let layout = ParameterLayout::new([
        Transform::upper(1, 2.).unwrap(),
        Transform::lower(1, 1.).unwrap(),
    ])
    .unwrap();
    let bounded = ComposedObjective::new(objective.into_model(), layout).unwrap();
    let mut output = [0.; 2];
    bounded
        .potential_hvp(&[0., 0.], &[3., -2.], &mut output)
        .unwrap();
    assert_eq!(output, [4., 5.]);
}
