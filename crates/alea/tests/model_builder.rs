//! External-facing ergonomics retain native graph/measure/error contracts.
#![cfg(feature = "distributions")]
// Exercise borrowed handles as well as owned Copy handles intentionally.
#![allow(clippy::needless_borrows_for_generic_args)]
use alea::{
    distributions::modeling::{ModelBuilderExt, normal, regression::RegressionData},
    model::{
        ConstrainedModel,
        assembly::{AssemblyError, FactorRole, ModelBuilder, NativeModel, ScalarInput},
    },
    target::LogDensityGradient,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-11 * (1. + b.abs()), "{a} != {b}");
}

#[test]
fn concise_macro_matches_explicit_native_graph_density_scores_and_metadata() {
    // An imported crate alias must not break expansion. Locals named like internal
    // macro concepts must not be captured. The caller need not return Result.
    use alea as inference;
    let observations = [0.8, 1.2, 1.7];
    let model = "caller local";
    let concise = inference::model! { |b|
        parameters { mean: real(), scale: positive(), }
        {
            b.normal_prior("mean prior", &mean, 0.0, 2.0)?;
            let prior = b.multiply(-1.0, &scale)?;
            b.prior("scale prior", prior)?;
            for (i, &y) in observations.iter().enumerate() {
                b.normal_likelihood(&format!("y[{i}]"), y, &mean, &scale)?;
            }
        }
    }
    .unwrap();
    assert_eq!(model, "caller local");
    let mut b = ModelBuilder::new();
    let mean = b.real("mean").unwrap();
    let scale = b.positive("scale").unwrap();
    let zero = b.constant(0.).unwrap();
    let two = b.constant(2.).unwrap();
    let prior = normal(&mut b, mean.expr(), zero, two).unwrap();
    b.prior("mean prior", prior).unwrap();
    let minus = b.constant(-1.).unwrap();
    let prior = b.multiply(minus, scale.expr()).unwrap();
    b.prior("scale prior", prior).unwrap();
    for (i, &y) in observations.iter().enumerate() {
        let event = b.constant(y).unwrap();
        let factor = normal(&mut b, event, mean.expr(), scale.expr()).unwrap();
        b.likelihood(&format!("y[{i}]"), factor).unwrap();
    }
    let explicit = b.compile().unwrap();
    assert_eq!(concise.coordinates(), explicit.coordinates());
    assert_eq!(concise.factors(), explicit.factors());
    assert_eq!(concise.expression_count(), explicit.expression_count());
    let a = concise.select_posterior().unwrap().into_target().unwrap();
    let b = explicit.select_posterior().unwrap().into_target().unwrap();
    for q in [[0.3, -0.5], [-1.2, 0.7], [0., 0.]] {
        let mut ag = [0.; 2];
        let mut bg = [0.; 2];
        // Same nodes/order => exact within-build bits, not a loose posterior test.
        assert_eq!(
            a.logp_grad(&q, &mut ag).unwrap().to_bits(),
            b.logp_grad(&q, &mut bg).unwrap().to_bits()
        );
        assert_eq!(ag.map(f64::to_bits), bg.map(f64::to_bits));
    }
}

#[test]
fn literal_arithmetic_and_input_errors_preserve_graph_and_handle_identity() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap();
    let mut other = ModelBuilder::new();
    let foreign = other.real("foreign").unwrap();
    let marker = b.constant(42.).unwrap();
    assert!(matches!(
        b.inputs([0.0.into(), ScalarInput::from(&foreign)]),
        Err(AssemblyError::ForeignHandle)
    ));
    assert!(matches!(
        b.inputs([0.0.into(), f64::NAN.into()]),
        Err(AssemblyError::Numerical)
    ));
    assert!(matches!(
        b.add(0.0, &foreign),
        Err(AssemblyError::ForeignHandle)
    ));
    assert!(matches!(
        normal(&mut b, 0.0, 1.0, &foreign),
        Err(AssemblyError::ForeignHandle)
    ));
    assert!(matches!(
        b.sum(&[x.expr(), foreign.expr()]),
        Err(AssemblyError::ForeignHandle)
    ));
    // Failed operations add no nodes; the final graph-size assertion below
    // catches even unused constants accidentally added before a later failure.
    let expression = b.expression(&x).unwrap();
    assert_eq!(expression, x.expr());
    let result = b.multiply(&x, 2.0).unwrap();
    let value = b.add(result, 1.0).unwrap();
    let value = b.log(value).unwrap();
    let value = b.exp(value).unwrap();
    b.likelihood("value", value).unwrap();
    let compiled = b.compile().unwrap();
    assert_eq!(compiled.expression_count(), 8); // x,marker,2,*,1,+,log,exp
    let mut g = [0.];
    close(compiled.logp_grad(&[3.], &mut g).unwrap(), 7.);
    close(g[0], 2.);
    let _ = marker;
}

#[test]
fn macro_errors_are_results_not_panics_and_imperative_form_remains_available() {
    let bad = alea::model! { |b| parameters { x: real() } {
        b.normal_prior("duplicate",&x,0.,1.)?;
        b.normal_prior("duplicate",&x,0.,1.)?;
    }};
    assert!(matches!(bad, Err(AssemblyError::Name)));
    assert!(matches!(alea::model!(|b| {}), Err(AssemblyError::Empty)));
    let bad = alea::model! { |b| parameters { x: real() } {
        b.normal_likelihood("data",f64::NAN,&x,1.)?;
    }};
    assert!(matches!(bad, Err(AssemblyError::Numerical)));
    let direct: Result<NativeModel, AssemblyError> = alea::model!(|b| {
        let x = b.real("custom name")?;
        b.normal_prior("prior", &x, 0., 1.)?;
    });
    assert_eq!(direct.unwrap().coordinates()[0].name, "custom name");
}

#[test]
fn vector_declarations_and_batch_helpers_keep_dimensions_roles_and_native_kernels() {
    let m = alea::model! { |b|
        parameters { beta: real_vector(&["intercept", "slope"]), sigma: positive() }
        {
            for (i,&coefficient) in beta.expressions().iter().enumerate() {
                b.normal_prior(&format!("prior {i}"),coefficient,0.,2.)?;
            }
            b.normal_prior("sigma",&sigma,0.,1.)?;
            let data = RegressionData::new(2,vec![1.,0.,1.,1.,1.,2.],vec![0.5,1.,2.])
                .map_err(|e|AssemblyError::Factor(Box::new(e)))?;
            b.normal_regression_likelihood("response",beta.expressions(),&sigma,data)?;
        }
    }
    .unwrap();
    assert_eq!(
        m.coordinates()
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["beta[intercept]", "beta[slope]", "sigma"]
    );
    assert_eq!(m.coordinates()[0].dimension.as_deref(), Some("beta"));
    assert_eq!(m.factors().last().unwrap().role(), FactorRole::Likelihood);
    let mut b = ModelBuilder::new();
    let vector = b.positive_vector("scales", &["a", "b"]).unwrap();
    assert_eq!(vector.expressions().len(), 2);
    assert!(matches!(
        b.unit_interval_vector("probabilities", &["a", "a"]),
        Err(AssemblyError::Name)
    ));
    let _probabilities = b
        .unit_interval_vector("probabilities", &["a", "b"])
        .unwrap();
}

#[test]
fn shared_normal_observations_match_scalar_terms_and_finite_differences() {
    fn make(batch: bool, n: usize) -> NativeModel {
        alea::model! { |b| parameters { mu: real(), sigma: positive() } {
            let y:Vec<_>=(0..n).map(|i|i as f64*0.01-0.3).collect();
            if batch { b.normal_observations("data",&y,&mu,&sigma)?; }
            else { for (i,&y) in y.iter().enumerate() { b.normal_likelihood(&format!("row {i}"),y,&mu,&sigma)?; } }
        }}.unwrap()
    }
    assert_eq!(
        make(true, 3).expression_count(),
        make(true, 1000).expression_count()
    );
    let batch = make(true, 37).into_target().unwrap();
    let scalar = make(false, 37).into_target().unwrap();
    for q in [[0.2, -0.5], [-0.4, 0.3]] {
        let mut g = [0.; 2];
        let mut reference = [0.; 2];
        close(
            batch.logp_grad(&q, &mut g).unwrap(),
            scalar.logp_grad(&q, &mut reference).unwrap(),
        );
        for i in 0..2 {
            close(g[i], reference[i]);
            let mut plus = q;
            let mut minus = q;
            let mut scratch = [0.; 2];
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let fd = (batch.logp_grad(&plus, &mut scratch).unwrap()
                - batch.logp_grad(&minus, &mut scratch).unwrap())
                / 2e-5;
            assert!((g[i] - fd).abs() < 1e-6 * (1. + g[i].abs()));
        }
    }
}

#[test]
fn named_glm_helpers_preserve_native_offsets_scores_and_factor_selection() {
    use alea::distributions::modeling::{bernoulli_logit, regression};
    for poisson in [false, true] {
        let build = |convenient: bool| -> Result<NativeModel, AssemblyError> {
            let mut b = ModelBuilder::new();
            let beta = b.real_vector("beta", &["a", "b"])?;
            for (i, &x) in beta.expressions().iter().enumerate() {
                b.normal_prior(&format!("prior{i}"), x, 0., 2.)?;
            }
            if poisson {
                let data = RegressionData::new(2, vec![1., -1., 1., 2.], vec![0_u64, 3])
                    .unwrap()
                    .with_offsets(vec![0.25, -0.5])
                    .unwrap();
                if convenient {
                    b.poisson_log_regression_likelihood("data", beta.expressions(), data)?;
                } else {
                    let f = regression::poisson_log_regression(&mut b, beta.expressions(), data)?;
                    b.likelihood("data", f)?;
                }
            } else {
                let data = RegressionData::new(2, vec![1., -1., 1., 2.], vec![false, true])
                    .unwrap()
                    .with_offsets(vec![0.25, -0.5])
                    .unwrap();
                if convenient {
                    b.bernoulli_logit_regression_likelihood("data", beta.expressions(), data)?;
                } else {
                    let f =
                        regression::bernoulli_logit_regression(&mut b, beta.expressions(), data)?;
                    b.likelihood("data", f)?;
                }
            }
            b.compile()?.select_likelihood()
        };
        let a = build(true).unwrap().into_target().unwrap();
        let b = build(false).unwrap().into_target().unwrap();
        let mut ag = [0.; 2];
        let mut bg = [0.; 2];
        assert_eq!(
            a.logp_grad(&[0.2, -0.3], &mut ag).unwrap().to_bits(),
            b.logp_grad(&[0.2, -0.3], &mut bg).unwrap().to_bits()
        );
        assert_eq!(ag.map(f64::to_bits), bg.map(f64::to_bits));
    }
    let mut b = ModelBuilder::new();
    let logit = b.real("logit").unwrap();
    b.bernoulli_logit_likelihood("yes", true, &logit).unwrap();
    let raw = bernoulli_logit(&mut b, false, &logit).unwrap();
    b.likelihood("no", raw).unwrap();
    assert!(matches!(
        b.normal_prior("yes", &logit, 0., 1.),
        Err(AssemblyError::Name)
    ));
    let model = b.compile().unwrap();
    assert_eq!(model.factors().len(), 2);
    assert!(
        model
            .factors()
            .iter()
            .all(|f| f.role() == FactorRole::Likelihood)
    );
}
