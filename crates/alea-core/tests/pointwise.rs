//! Predictive partitions retain data alignment and evaluate the actual likelihood.
use alea_core::{
    model::assembly::{
        AssemblyError, CompiledModel, ModelBuilder, PointwiseError, PointwiseLikelihood,
        PredictiveUnit,
    },
    target::LogDensityGradient,
};
use alea_distributions::modeling::{normal, normal_observation};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn unit(name: &str, factors: &[&str], observations: &[&str]) -> PredictiveUnit {
    PredictiveUnit {
        name: name.into(),
        factors: factors.iter().map(|s| (*s).into()).collect(),
        observations: observations.iter().map(|s| (*s).into()).collect(),
        conditioned_on: vec!["mean".into(), "scale".into()],
        integrated_out: vec![],
    }
}
fn model() -> CompiledModel {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean").unwrap();
    let scale = b.positive("scale").unwrap();
    for (i, y) in [1., 2., -0.5].into_iter().enumerate() {
        let f = normal_observation(&mut b, y, mean.expr(), &scale).unwrap();
        b.likelihood(&format!("y{i}"), f).unwrap();
    }
    let prior = b.multiply(mean.expr(), scale.expr()).unwrap();
    b.prior("prior", prior).unwrap();
    b.compile().unwrap()
}
fn units() -> Vec<PredictiveUnit> {
    // Output order differs from declaration order; within a unit term order
    // is also deliberately reversed.
    vec![
        unit("joint", &["y2", "y0"], &["row2", "row0"]),
        unit("single", &["y1"], &["row1"]),
    ]
}
fn normal_log(y: f64, mean: f64, scale: f64) -> f64 {
    -0.5 * ((y - mean) / scale).powi(2) - scale.ln() - 0.5 * (2. * std::f64::consts::PI).ln()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-13 * (1. + b.abs()), "{a} != {b}");
}

#[test]
fn normalized_units_exclude_priors_and_jacobians_and_keep_output_order() {
    let pointwise = PointwiseLikelihood::new(model(), units()).unwrap();
    let mut output = [99.; 2];
    for (mean, scale) in [(0.3, 0.5_f64), (-2., 3.), (12., 0.1)] {
        let total = pointwise
            .evaluate(&[mean, scale.ln()], &mut output)
            .unwrap();
        let terms = [1., 2., -0.5].map(|y| normal_log(y, mean, scale));
        close(output[0], terms[0] + terms[2]);
        close(output[1], terms[1]);
        close(total, terms.iter().sum());
    }
    assert_eq!(pointwise.units()[0].observations, ["row2", "row0"]);
    assert_eq!(pointwise.model().factors().len(), 4);
    let posterior = pointwise
        .into_model()
        .select_posterior()
        .unwrap()
        .into_target()
        .unwrap();
    let mut g = [0.; 2];
    let q = [0.3, 0.5_f64.ln()];
    let expected: f64 = [1., 2., -0.5].map(|y| normal_log(y, 0.3, 0.5)).iter().sum();
    close(
        posterior.logp_grad(&q, &mut g).unwrap(),
        expected + 0.15 + q[1],
    );
}

#[test]
fn joint_gaussian_random_intercept_is_marginalized_by_the_model_not_metadata() {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean").unwrap();
    let sigma = b.positive("sigma").unwrap();
    let tau = b.positive("tau").unwrap();
    let sqrt_two = b.constant(2_f64.sqrt()).unwrap();
    let mean_sum = b.multiply(sqrt_two, mean.expr()).unwrap();
    let variance = b.multiply(sigma.expr(), sigma.expr()).unwrap();
    let tau2 = b.multiply(tau.expr(), tau.expr()).unwrap();
    let two = b.constant(2.).unwrap();
    let twice_tau2 = b.multiply(two, tau2).unwrap();
    let variance_sum = b.add(variance, twice_tau2).unwrap();
    let log_variance = b.log(variance_sum).unwrap();
    let half = b.constant(0.5).unwrap();
    let log_sd = b.multiply(half, log_variance).unwrap();
    let sd = b.exp(log_sd).unwrap();
    let sum = b.constant(3. / 2_f64.sqrt()).unwrap();
    let diff = b.constant(-1. / 2_f64.sqrt()).unwrap();
    let zero = b.constant(0.).unwrap();
    let plus = normal(&mut b, sum, mean_sum, sd).unwrap();
    let minus = normal(&mut b, diff, zero, sigma.expr()).unwrap();
    b.likelihood("sum", plus).unwrap();
    b.likelihood("difference", minus).unwrap();
    let spec = PredictiveUnit {
        name: "subject 1".into(),
        factors: vec!["sum".into(), "difference".into()],
        observations: vec!["subject 1:row 1".into(), "subject 1:row 2".into()],
        conditioned_on: vec!["mean".into(), "sigma".into(), "tau".into()],
        integrated_out: vec!["subject intercept".into()],
    };
    let pointwise = PointwiseLikelihood::new(b.compile().unwrap(), vec![spec]).unwrap();
    for (mean, sigma, tau) in [(0.3, 0.5_f64, 2_f64), (-1., 2., 0.1), (3., 0.2, 0.3)] {
        let mut out = [99.];
        let total = pointwise
            .evaluate(&[mean, sigma.ln(), tau.ln()], &mut out)
            .unwrap();
        let v = sigma * sigma;
        let t = tau * tau;
        let det = v * (v + 2. * t);
        let residuals = [1. - mean, 2. - mean];
        let quadratic = ((v + t) * (residuals[0].powi(2) + residuals[1].powi(2))
            - 2. * t * residuals[0] * residuals[1])
            / det;
        let expected = -(2. * std::f64::consts::PI).ln() - 0.5 * det.ln() - 0.5 * quadratic;
        close(total, expected);
        close(out[0], expected);
    }
}

#[test]
fn partitions_reject_missing_repeated_unknown_and_nonlikelihood_factors() {
    for bad in [
        vec![unit("a", &["y0"], &["0"])],
        vec![unit("a", &["y0", "y1", "y2", "y0"], &["0"])],
    ] {
        assert!(matches!(
            PointwiseLikelihood::new(model(), bad),
            Err(PointwiseError::Partition { .. })
        ));
    }
    for name in ["prior", "missing"] {
        assert!(matches!(
            PointwiseLikelihood::new(model(), vec![unit("a", &[name], &["0"])]),
            Err(PointwiseError::Factor { .. })
        ));
    }
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    b.factor(x).unwrap();
    assert!(matches!(
        PointwiseLikelihood::new(b.compile().unwrap(), vec![]),
        Err(PointwiseError::Model(
            AssemblyError::UnclassifiedFactor { .. }
        ))
    ));
}

#[test]
fn semantic_and_observation_alignment_errors_are_explicit() {
    for change in 0..7 {
        let mut specs = units();
        match change {
            0 => specs[1].name = specs[0].name.clone(),
            1 => specs[0].observations.clear(),
            2 => specs[0].observations[0] = String::new(),
            3 => specs[1].observations = specs[0].observations.clone(),
            4 => specs[0].conditioned_on.push("mean".into()),
            5 => specs[0].integrated_out.push("mean".into()),
            _ => specs[0].integrated_out.push(String::new()),
        }
        assert!(PointwiseLikelihood::new(model(), specs).is_err());
    }
}

#[test]
fn numerical_failures_and_unwind_preserve_output_and_allow_recovery() {
    let mode = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut b = ModelBuilder::new();
    let x = b.positive("x").unwrap().expr();
    let state = Arc::clone(&mode);
    let count = Arc::clone(&calls);
    let term = b
        .custom(&[x], move |x, g| {
            count.fetch_add(1, Ordering::Relaxed);
            g[0] = 1.;
            match state.load(Ordering::Relaxed) {
                1 => Err(AssemblyError::Numerical),
                2 => panic!("injected panic"),
                3 => {
                    g[0] = f64::NAN;
                    Ok(x[0])
                }
                4 => Ok(f64::NEG_INFINITY),
                _ => Ok(x[0]),
            }
        })
        .unwrap();
    b.likelihood("a", term).unwrap();
    b.likelihood("b", term).unwrap();
    let bad = b.custom(&[x], |_, _| panic!("excluded prior")).unwrap();
    b.prior("prior", bad).unwrap();
    let pointwise = PointwiseLikelihood::new(
        b.compile().unwrap(),
        vec![unit("a", &["a"], &["0"]), unit("b", &["b"], &["1"])],
    )
    .unwrap();
    let mut out = [99.; 2];
    for q in [vec![], vec![f64::NAN], vec![-1000.]] {
        assert!(pointwise.evaluate(&q, &mut out).is_err());
        assert_eq!(out, [99.; 2]);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(pointwise.evaluate(&[0.], &mut out[..1]).is_err());
    for value in [1, 2, 3, 4] {
        mode.store(value, Ordering::Relaxed);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pointwise.evaluate(&[0.], &mut out)
        }));
        if value == 2 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(out, [99.; 2]);
    }
    mode.store(0, Ordering::Relaxed);
    assert_eq!(pointwise.evaluate(&[0.], &mut out).unwrap(), 2.);
    assert_eq!(out, [1.; 2]);
    // Shared callbacks run once, not once per unit; repeated factors remain two terms.
    assert_eq!(calls.load(Ordering::Relaxed), 5);
}

#[test]
#[cfg(not(miri))]
fn evaluation_reuses_storage() {
    let p = PointwiseLikelihood::new(model(), units()).unwrap();
    let mut out = [0.; 2];
    let measured = allocation_counter::measure(|| {
        for _ in 0..64 {
            std::hint::black_box(p.evaluate(&[0.3, -0.4], &mut out).unwrap());
        }
    });
    assert_eq!(measured.count_total, 0);
}

#[test]
fn factor_membership_uses_declaration_order_even_with_cancellation() {
    let mut b = ModelBuilder::new();
    let _ = b.real("x").unwrap();
    for (name, value) in [("a", 1e16), ("b", -1e16), ("c", 1.)] {
        let term = b.constant(value).unwrap();
        b.likelihood(name, term).unwrap();
    }
    let p = PointwiseLikelihood::new(
        b.compile().unwrap(),
        vec![unit("all", &["c", "a", "b"], &["observation"])],
    )
    .unwrap();
    let mut out = [99.];
    assert_eq!(p.evaluate(&[0.], &mut out).unwrap(), 1.);
    assert_eq!(out, [1.]);
}
