//! Scalar composition is a different operation from transforming a density.
use alea_core::{
    model::{
        AnalyticModel, ComposedObjective, ConstrainedModel, ModelError, TransformedTarget,
        assembly::ModelBuilder,
    },
    target::{LogDensityGradient, TargetSpace},
    transform::{ParameterLayout, Transform, TransformError},
};
use alea_distributions::modeling::normal_observation;
use std::{cell::Cell, convert::Infallible};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * (1. + b.abs()), "{a} != {b}");
}
fn charts() -> Vec<Transform> {
    let mut result = Vec::new();
    for n in [1, 2, 3] {
        result.extend([
            Transform::identity(n).unwrap(),
            Transform::positive(n).unwrap(),
            Transform::lower(n, -2.).unwrap(),
            Transform::upper(n, 3.).unwrap(),
            Transform::interval(n, -2., 3.).unwrap(),
            Transform::ordered(n).unwrap(),
            Transform::positive_ordered(n).unwrap(),
            Transform::simplex(n).unwrap(),
            Transform::unit_vector(n + 1).unwrap(),
            Transform::cholesky_covariance(n).unwrap(),
            Transform::cholesky_correlation(n).unwrap(),
            Transform::covariance(n).unwrap(),
            Transform::correlation(n).unwrap(),
        ]);
    }
    result
}
fn scalar_value(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, x)| (i + 1) as f64 * (x * x / 2. + x))
        .sum()
}
#[test]
fn all_chart_compositions_match_independent_directional_chain_rules() {
    for chart in charts() {
        for scale in [-0.7, 0., 0.9] {
            let q: Vec<_> = (0..chart.unconstrained_dimension())
                .map(|i| scale + 0.05 * i as f64)
                .collect();
            let mut x = vec![0.; chart.constrained_dimension()];
            let mut scratch = vec![0.; chart.scratch_dimension()];
            let jac = chart.constrain(&q, &mut x, &mut scratch).unwrap();
            let expected_x = x.clone();
            chart.constrain_values(&q, &mut x, &mut scratch).unwrap();
            assert_eq!(x, expected_x);
            let gx: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(i, x)| (i + 1) as f64 * (x + 1.))
                .collect();
            let mut pure = vec![0.; q.len()];
            let mut density = pure.clone();
            chart
                .pullback_composition(&q, &gx, &mut pure, &mut scratch)
                .unwrap();
            chart.pullback(&q, &gx, &mut density, &mut scratch).unwrap();
            for k in 0..q.len() {
                let h = 1e-5;
                let mut plus = q.clone();
                let mut minus = q.clone();
                plus[k] += h;
                minus[k] -= h;
                let mut xp = x.clone();
                let mut xm = x.clone();
                let jp = chart.constrain(&plus, &mut xp, &mut scratch).unwrap();
                let jm = chart.constrain(&minus, &mut xm, &mut scratch).unwrap();
                close(
                    pure[k],
                    (scalar_value(&xp) - scalar_value(&xm)) / (2. * h),
                    2e-7,
                );
                close(density[k] - pure[k], (jp - jm) / (2. * h), 2e-7);
            }
            assert!(jac.is_finite());
        }
    }
}

fn regression() -> alea_core::model::assembly::CompiledModel {
    let mut b = ModelBuilder::new();
    let a = b.real("intercept").unwrap();
    let slope = b.real("slope").unwrap();
    let sigma = b.positive("sigma").unwrap();
    for (x, e) in [-2., -1., 0., 1., 2.]
        .into_iter()
        .zip([1., -2., 2., -2., 1.])
    {
        let predictor = b.constant(x).unwrap();
        let effect = b.multiply(predictor, slope.expr()).unwrap();
        let mean = b.add(a.expr(), effect).unwrap();
        let factor = normal_observation(&mut b, 1. + 2. * x + e, mean, &sigma).unwrap();
        b.factor(factor).unwrap();
    }
    b.compile().unwrap()
}
#[test]
fn gaussian_mle_sampling_mode_and_residual_variance_are_different() {
    let objective = regression().into_objective().unwrap();
    let target = regression().into_target().unwrap();
    assert_eq!(objective.dimension(), 3);
    assert_eq!(objective.model().dimension(), 3);
    assert_eq!(objective.layout().unconstrained_dimension(), 3);
    let n = 5.0_f64;
    let p = 2.;
    let rss = 14.;
    for variance in [rss / n, rss / (n - 1.), rss / (n - p)] {
        let q = [1., 2., 0.5 * variance.ln()];
        let mut g = [0.; 3];
        let mut sample_g = [0.; 3];
        let value = objective.logp_grad(&q, &mut g).unwrap();
        let sampled = target.logp_grad(&q, &mut sample_g).unwrap();
        close(
            value,
            -n * q[2] - rss / (2. * variance) - 0.5 * n * std::f64::consts::TAU.ln(),
            1e-13,
        );
        close(g[0], 0., 1e-13);
        close(g[1], 0., 1e-13);
        close(g[2], -n + rss / variance, 1e-13);
        close(sampled - value, q[2], 1e-13);
        close(sample_g[2], g[2] + 1., 1e-13);
        if variance == rss / n {
            close(g[2], 0., 1e-13);
            close(sample_g[2], 1., 1e-13);
        }
        if variance == rss / (n - 1.) {
            close(sample_g[2], 0., 1e-13);
        }
        if variance == rss / (n - p) {
            close(g[2], -p, 1e-13);
        }
    }
}

#[test]
fn declared_measure_map_does_not_move_to_the_chart_density_mode() {
    fn model() -> impl ConstrainedModel<Error = Infallible> {
        AnalyticModel::new(1, |x: &[f64], g: &mut [f64]| {
            g[0] = 2. / x[0] - 2.;
            Ok(2. * x[0].ln() - 2. * x[0])
        })
    }
    let layout = ParameterLayout::new([Transform::positive(1).unwrap()]).unwrap();
    let objective = ComposedObjective::new(model(), layout.clone()).unwrap();
    let density = TransformedTarget::new(model(), layout).unwrap();
    let mut g = [0.];
    objective.logp_grad(&[0.], &mut g).unwrap();
    close(g[0], 0., 1e-14);
    density.logp_grad(&[0.], &mut g).unwrap();
    close(g[0], 1., 1e-14);
    density.logp_grad(&[1.5f64.ln()], &mut g).unwrap();
    close(g[0], 0., 1e-14);
}

#[test]
fn mixed_layouts_and_failed_models_preserve_output_and_recover() {
    let mode = Cell::new(0);
    let callback = AnalyticModel::new(
        4,
        |x: &[f64], g: &mut [f64]| -> Result<f64, std::io::Error> {
            g.fill(1.);
            match mode.get() {
                1 => Err(std::io::ErrorKind::Other.into()),
                2 => {
                    g[2] = f64::NAN;
                    Ok(0.)
                }
                3 => panic!("backend panic"),
                _ => Ok(x.iter().sum()),
            }
        },
    );
    let layout = ParameterLayout::new([
        Transform::positive(1).unwrap(),
        Transform::simplex(3).unwrap(),
    ])
    .unwrap();
    let objective = ComposedObjective::new(callback, layout).unwrap();
    let q = [0.2, -0.4, 0.7];
    let mut g = [123.; 3];
    for invalid in [vec![0.], vec![f64::NAN, 0., 0.]] {
        assert!(objective.logp_grad(&invalid, &mut g).is_err());
        assert_eq!(g, [123.; 3]);
    }
    mode.set(1);
    assert!(matches!(
        objective.logp_grad(&q, &mut g),
        Err(ModelError::Evaluation(_))
    ));
    assert_eq!(g, [123.; 3]);
    mode.set(2);
    assert!(objective.logp_grad(&q, &mut g).is_err());
    assert_eq!(g, [123.; 3]);
    mode.set(3);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || objective.logp_grad(&q, &mut g)
        ))
        .is_err()
    );
    assert_eq!(g, [123.; 3]);
    mode.set(0);
    let value = objective.logp_grad(&q, &mut g).unwrap();
    close(value, 0.2f64.exp() + 1., 1e-14);
    close(g[0], 0.2f64.exp(), 1e-14);
    close(g[1], 0., 1e-14);
    close(g[2], 0., 1e-14);
}

#[test]
fn composition_shape_and_domain_failures_do_not_fabricate_gradients() {
    let chart = Transform::positive(1).unwrap();
    let mut x = [99.];
    assert!(matches!(
        chart.constrain_values(&[], &mut x, &mut []),
        Err(TransformError::Dimension { .. })
    ));
    assert_eq!(x, [99.]);
    assert!(chart.constrain_values(&[-1000.], &mut x, &mut []).is_err());
    let mut g = [99.];
    assert!(
        chart
            .pullback_composition(&[0.], &[], &mut g, &mut [])
            .is_err()
    );
    assert_eq!(g, [99.]);
}

#[test]
#[cfg(not(miri))]
fn objective_reuses_all_storage() {
    let objective = regression().into_objective().unwrap();
    let mut g = [0.; 3];
    let measured = allocation_counter::measure(|| {
        for _ in 0..64 {
            std::hint::black_box(objective.logp_grad(&[1., 2., 0.2], &mut g).unwrap());
        }
    });
    assert_eq!(measured.count_total, 0);
}
