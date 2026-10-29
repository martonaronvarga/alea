use alea_core::{
    model::{
        ConstrainedModel,
        assembly::{AssemblyError, ModelBuilder},
        prediction::{NamedOutput, PredictiveOrigin, posterior_predict, prior_predict},
    },
    target::{LogDensityGradient, TargetSpace},
    transform::Transform,
};
use alea_distributions::modeling;
use rand::{SeedableRng, rngs::SmallRng};
use std::convert::Infallible;
fn shared_model() -> alea_core::model::assembly::CompiledModel {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap();
    let scale = b.positive("scale").unwrap();
    b.annotate(&mu, "coefficient", "mean", "location").unwrap();
    b.annotate(&scale, "coefficient", "scale", "scale").unwrap();
    let zero = b.constant(0.0).unwrap();
    let one = b.constant(1.0).unwrap();
    let prior = modeling::normal(&mut b, mu.expr(), zero, one).unwrap();
    b.factor(prior).unwrap();
    let log_scale = b.log(scale.expr()).unwrap();
    let prior = modeling::normal(&mut b, log_scale, zero, one).unwrap();
    // Normal on log(scale) needs -log(scale) for a lognormal constrained density.
    let minus = b.constant(-1.0).unwrap();
    let negative_log = b.multiply(log_scale, minus).unwrap();
    let prior = b.add(prior, negative_log).unwrap();
    b.factor(prior).unwrap();
    b.plate(&[0.3, 0.7, -0.2], |b, _, y| {
        modeling::normal_observation(b, *y, mu.expr(), &scale)
    })
    .unwrap();
    b.compile().unwrap()
}
#[test]
fn shared_parameters_lower_once_with_independent_scores_and_metadata() {
    let model = shared_model();
    let names = model.coordinates();
    let target = model.into_target().unwrap();
    let q = [0.2, 0.3];
    let mut g = [0.0; 2];
    let value = target.logp_grad(&q, &mut g).unwrap();
    let normalizer = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut expected = -0.5 * q[0] * q[0] - normalizer - 0.5 * q[1] * q[1] - normalizer;
    for y in [0.3, 0.7, -0.2] {
        expected += -0.5 * ((y - q[0]) / q[1].exp()).powi(2) - q[1] - normalizer;
    }
    assert!((value - expected).abs() < 1e-12);
    for i in 0..2 {
        let mut a = q;
        let mut z = q;
        a[i] += 1e-5;
        z[i] -= 1e-5;
        let fd = (target.logp_grad(&a, &mut [0.0; 2]).unwrap()
            - target.logp_grad(&z, &mut [0.0; 2]).unwrap())
            / 2e-5;
        assert!((g[i] - fd).abs() < 2e-7);
    }
    let named = NamedOutput::new(names, target.layout().clone(), vec![q]).unwrap();
    assert_eq!(
        named.blocks(),
        vec![("location".into(), vec![0]), ("scale".into(), vec![1])]
    );
    let allocation = allocation_counter::measure(|| {
        for _ in 0..128 {
            std::hint::black_box(target.logp_grad(&q, &mut g).unwrap());
        }
    });
    assert_eq!(allocation.count_total, 0);
}
#[test]
fn foreign_handles_and_invalid_or_incomplete_factors_are_rejected() {
    let mut a = ModelBuilder::new();
    let foreign = a.real("x").unwrap();
    let mut b = ModelBuilder::new();
    let local = b.real("x").unwrap();
    assert!(matches!(
        b.add(local.expr(), foreign.expr()),
        Err(AssemblyError::ForeignHandle)
    ));
    assert!(b.real("x").is_err());
    let bad = b.custom(&[local.expr()], |_, _| Ok(1.0)).unwrap();
    b.factor(bad).unwrap();
    let model = b.compile().unwrap();
    let mut gradient = [123.0];
    assert!(model.logp_grad(&[0.0], &mut gradient).is_err());
    assert_eq!(gradient, [123.0]);
}
#[test]
fn vector_transform_uses_one_simplex_jacobian_and_preserves_coordinates() {
    let mut b = ModelBuilder::new();
    let p = b
        .vector::<alea_core::model::assembly::Simplex>(
            "p",
            "category",
            &["a", "b", "c"],
            "simplex",
            Transform::simplex(3).unwrap(),
        )
        .unwrap();
    let x = b.real("x").unwrap();
    assert_eq!(x.slot(), 3);
    let density = b
        .custom(p.expressions(), |q, g| {
            g.fill(0.0);
            assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            Ok(2.0f64.ln())
        })
        .unwrap();
    b.factor(density).unwrap();
    let model = b.compile().unwrap();
    let layout = model.layout().clone();
    let metadata = NamedOutput::new(model.coordinates(), layout.clone(), ()).unwrap();
    assert_eq!(metadata.blocks(), vec![("simplex".into(), vec![0, 1])]);
    let target = model.into_target().unwrap();
    let q = [0.2, -0.3, 0.7];
    let mut x = [0.0; 4];
    let jac = layout
        .constrain(&q, &mut x, &mut vec![0.0; layout.scratch_dimension()])
        .unwrap();
    let value = target.logp_grad(&q, &mut [0.0; 3]).unwrap();
    assert!((value - 2.0f64.ln() - jac).abs() < 1e-12);
}
#[test]
fn predictive_and_generated_quantities_are_separate_workflows() {
    let model = shared_model();
    let mut rng = SmallRng::seed_from_u64(14010);
    let draws = [[0.2, 0.3], [0.4, -0.1]];
    let prediction = posterior_predict(
        model.layout(),
        draws.iter().map(|q| q.as_slice()),
        &mut rng,
        |p, _| Ok::<_, Infallible>((p[0], p[1])),
    )
    .unwrap();
    assert_eq!(prediction.origin, PredictiveOrigin::Posterior);
    assert_eq!(prediction.generated[0], (0.2, 0.3f64.exp()));
    let prior = prior_predict(5, &mut rng, |_| Ok::<_, Infallible>(7.0)).unwrap();
    assert_eq!(prior.origin, PredictiveOrigin::Prior);
    assert_eq!(prior.generated, vec![7.0; 5]);
}

#[test]
fn empty_transform_blocks_have_no_geometry_slots() {
    let layout =
        alea_core::transform::ParameterLayout::new([Transform::identity(0).unwrap()]).unwrap();
    let output = NamedOutput::new(Vec::new().into(), layout, ()).unwrap();
    assert!(output.blocks().is_empty());
}

#[test]
fn correlation_and_dirichlet_priors_pull_back_through_their_declared_measure() {
    use alea_core::distribution::Distribution;
    use alea_distributions::multivariate::{Dirichlet, LkjCholesky};
    for correlation in [false, true] {
        let mut builder = ModelBuilder::new();
        if correlation {
            let p = builder
                .vector::<alea_core::model::assembly::CholeskyCorrelation>(
                    "L",
                    "matrix",
                    &["00", "01", "02", "10", "11", "12", "20", "21", "22"],
                    "correlation",
                    Transform::cholesky_correlation(3).unwrap(),
                )
                .unwrap();
            let law = LkjCholesky::new(3, 2.0).unwrap();
            let factor = builder
                .custom(p.expressions(), move |x, g| {
                    law.event_gradient(x, g).unwrap();
                    Ok(law.log_prob(x).unwrap())
                })
                .unwrap();
            builder.factor(factor).unwrap();
        } else {
            let p = builder
                .vector::<alea_core::model::assembly::Simplex>(
                    "p",
                    "category",
                    &["a", "b", "c"],
                    "simplex",
                    Transform::simplex(3).unwrap(),
                )
                .unwrap();
            let law = Dirichlet::new(vec![2.0, 3.0, 4.0]).unwrap();
            let factor = builder
                .custom(p.expressions(), move |x, g| {
                    law.event_gradient(x, g).unwrap();
                    Ok(law.log_prob(x).unwrap())
                })
                .unwrap();
            builder.factor(factor).unwrap();
        }
        let target = builder.compile().unwrap().into_target().unwrap();
        let q = vec![0.2; target.dimension()];
        let mut gradient = vec![0.0; q.len()];
        target.logp_grad(&q, &mut gradient).unwrap();
        for i in 0..q.len() {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let fd = (target.logp_grad(&plus, &mut vec![0.0; q.len()]).unwrap()
                - target.logp_grad(&minus, &mut vec![0.0; q.len()]).unwrap())
                / 2e-5;
            assert!((gradient[i] - fd).abs() < 2e-5 * (1.0 + fd.abs()));
        }
    }
}

#[path = "../../../tools/downstream-conformance/src/modeling.rs"]
mod downstream;
#[test]
fn all_three_downstream_models_evaluate_without_allocations() {
    use alea_distributions::wiener::{Boundary, WienerObservation};
    use downstream::Observation;
    let models = [
        downstream::regression(&[(-1.0, -0.2), (0.0, 0.4), (1.0, 1.3)]),
        downstream::hierarchy(&[Observation::Logistic {
            group: 0,
            event: true,
        }]),
        downstream::hierarchy(&[Observation::Ddm {
            group: 0,
            event: WienerObservation {
                rt: 0.6,
                boundary: Boundary::Upper,
            },
        }]),
    ];
    for model in models {
        let target = model.into_target().unwrap();
        let q = vec![0.2; target.dimension()];
        let mut gradient = vec![0.0; q.len()];
        target.logp_grad(&q, &mut gradient).unwrap();
        let allocation = allocation_counter::measure(|| {
            for _ in 0..128 {
                std::hint::black_box(target.logp_grad(&q, &mut gradient).unwrap());
            }
        });
        assert_eq!(allocation.count_total, 0);
    }
}

#[test]
fn structured_handles_reject_wrong_domains_before_mutating_builder() {
    use alea_core::model::assembly::{Real, Simplex, VectorParameter};
    let mut builder = ModelBuilder::new();
    let result = builder.vector::<Simplex>(
        "p",
        "category",
        &["a", "b"],
        "p",
        Transform::identity(2).unwrap(),
    );
    assert!(matches!(result, Err(AssemblyError::Shape)));
    let real: VectorParameter<Real> = builder
        .vector::<Real>(
            "p",
            "category",
            &["a", "b"],
            "p",
            Transform::identity(2).unwrap(),
        )
        .unwrap();
    assert_eq!(real.transform().unconstrained_dimension(), 2);
    let simplex: VectorParameter<Simplex> = builder
        .vector::<Simplex>(
            "q",
            "category",
            &["a", "b"],
            "q",
            Transform::simplex(2).unwrap(),
        )
        .unwrap();
    assert_eq!(simplex.transform().unconstrained_dimension(), 1);
    assert_eq!(simplex.transform().constrained_dimension(), 2);
    builder.factor(real.expressions()[0]).unwrap();
    assert_eq!(
        builder
            .compile()
            .unwrap()
            .layout()
            .unconstrained_dimension(),
        3
    );
}

#[test]
fn native_factor_sum_retains_small_terms_and_original_scores() {
    for filtered in [false, true] {
        let mut b = ModelBuilder::new();
        let q = b.real("q").unwrap();
        let big = b.constant(1e16).unwrap();
        let opposite = b.constant(-1e16).unwrap();
        b.likelihood("large positive", big).unwrap();
        b.likelihood("small", q.expr()).unwrap();
        b.likelihood("large negative", opposite).unwrap();
        let model = b.compile().unwrap();
        let model = if filtered {
            model.select_likelihood().unwrap()
        } else {
            model
        };
        let target = model.into_objective().unwrap();
        let mut score = [f64::NAN];
        assert_eq!(target.logp_grad(&[1.], &mut score).unwrap(), 1.);
        assert_eq!(score, [1.]);
        assert_eq!(target.logp_grad(&[-3.], &mut score).unwrap(), -3.);
        assert_eq!(score, [1.]);
    }
}

#[test]
fn fused_linear_predictor_retains_products_and_exposes_analytic_score() {
    let mut b = ModelBuilder::new();
    let q = b.real("q").unwrap();
    let a = 1. + 2_f64.powi(-27);
    let coefficient = 1. - 2_f64.powi(-27);
    let linear = b
        .linear_combination(&[q.expr()], &[coefficient], -1.)
        .unwrap();
    b.likelihood("linear", linear).unwrap();
    let m = b.compile().unwrap();
    assert!(matches!(
        m.rust_objective_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    let t = m.into_objective().unwrap();
    let mut g = [f64::NAN];
    assert_eq!(t.logp_grad(&[a], &mut g).unwrap(), -2_f64.powi(-54));
    assert_eq!(g, [coefficient]);
    let mut b = ModelBuilder::new();
    let q = b.real("q").unwrap();
    assert!(matches!(
        b.linear_combination(&[q.expr()], &[], 0.),
        Err(AssemblyError::Shape)
    ));
    assert!(matches!(
        b.linear_combination(&[q.expr()], &[f64::NAN], 0.),
        Err(AssemblyError::Numerical)
    ));
    let mut other = ModelBuilder::new();
    assert!(matches!(
        other.linear_combination(&[q.expr()], &[1.], 0.),
        Err(AssemblyError::ForeignHandle)
    ));
}

#[test]
fn owned_callback_scratch_recovers_from_unwind_and_is_model_local() {
    fn make() -> alea_core::model::assembly::CompiledModel {
        let mut builder = ModelBuilder::new();
        let x = builder.real("x").unwrap();
        let factor = builder
            .custom_with_scratch(&[x.expr()], vec![f64::NAN; 2], |x, g, work| {
                work.fill(0.);
                work[0] = x[0] * x[0];
                if x[0] == 9. {
                    panic!("test callback unwind");
                }
                if x[0] == 8. {
                    return Err(AssemblyError::Numerical);
                }
                g[0] = 2. * x[0];
                Ok(work[0])
            })
            .unwrap();
        builder.factor(factor).unwrap();
        builder.compile().unwrap()
    }
    let first = make();
    assert!(matches!(
        first.rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    let second = make();
    let mut g = [17.];
    assert_eq!(first.logp_grad(&[2.], &mut g).unwrap(), 4.);
    assert_eq!(g, [4.]);
    g.fill(17.);
    assert!(first.logp_grad(&[8.], &mut g).is_err());
    assert_eq!(g, [17.]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || first.logp_grad(&[9.], &mut g)
        ))
        .is_err()
    );
    assert_eq!(g, [17.]);
    assert_eq!(second.logp_grad(&[3.], &mut g).unwrap(), 9.);
    assert_eq!(g, [6.]);
    assert_eq!(first.logp_grad(&[2.], &mut g).unwrap(), 4.);
    assert_eq!(g, [4.]);
}
