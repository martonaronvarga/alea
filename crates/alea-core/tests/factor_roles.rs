//! Statistical selection keeps the original graph and its declared measure.
use alea_core::{
    model::{
        ConstrainedModel,
        assembly::{AssemblyError, CompiledModel, FactorRole, ModelBuilder},
    },
    target::LogDensityGradient,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn named(potential: bool) -> CompiledModel {
    let mut b = ModelBuilder::new();
    let x = b.positive("theta").unwrap().expr();
    let square = b.multiply(x, x).unwrap();
    b.likelihood("joint data", square).unwrap();
    // Shared expression and repeated factor registration must preserve multiplicity.
    b.prior("prior", square).unwrap();
    if potential {
        b.potential("penalty", x).unwrap();
    }
    b.compile().unwrap()
}

#[test]
fn consuming_views_preserve_terms_coordinates_and_chart_semantics() {
    let objective = named(false)
        .select_likelihood()
        .unwrap()
        .into_objective()
        .unwrap();
    let q = [2_f64.ln()];
    let mut g = [0.];
    assert_eq!(objective.logp_grad(&q, &mut g).unwrap(), 4.);
    assert_eq!(g, [8.]);
    assert_eq!(objective.model().factors().len(), 2);
    assert_eq!(
        objective.model().selected_factors().next().unwrap().role(),
        FactorRole::Likelihood
    );
    let posterior = objective
        .into_model()
        .select_posterior()
        .unwrap()
        .into_objective()
        .unwrap();
    assert_eq!(posterior.logp_grad(&q, &mut g).unwrap(), 8.);
    assert_eq!(g, [16.]);
    let sampling = posterior.into_model().into_target().unwrap();
    assert_eq!(sampling.logp_grad(&q, &mut g).unwrap(), 8. + q[0]);
    assert_eq!(g, [17.]);
    let likelihood = sampling.into_model().select_likelihood().unwrap();
    assert_eq!(likelihood.selected_factors().count(), 1);
    let all = named(true).select_likelihood().unwrap().select_all();
    assert_eq!(all.logp_grad(&[2.], &mut g).unwrap(), 10.);
    assert_eq!(g, [9.]);
    assert_eq!(all.selected_factors().count(), 3);
}

#[test]
fn statistical_roles_are_never_guessed() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    b.factor(x).unwrap();
    assert!(matches!(
        b.compile().unwrap().select_likelihood(),
        Err(AssemblyError::UnclassifiedFactor { index: 0 })
    ));
    assert!(matches!(
        named(true).select_posterior(),
        Err(AssemblyError::PosteriorPotential { index: 2 })
    ));
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    b.prior("only prior", x).unwrap();
    assert!(matches!(
        b.compile().unwrap().select_likelihood(),
        Err(AssemblyError::Empty)
    ));
}

#[test]
fn invalid_declarations_do_not_add_terms_and_names_are_only_metadata() {
    let mut b = ModelBuilder::new();
    let x = b.real("same name").unwrap().expr();
    let mut foreign = ModelBuilder::new();
    let y = foreign.real("y").unwrap().expr();
    assert!(b.likelihood("", x).is_err());
    assert!(b.likelihood("foreign", y).is_err());
    b.likelihood("same name", x).unwrap();
    assert!(b.prior("same name", x).is_err());
    b.prior("λ\"; panic!()", x).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.factors().len(), 2);
    assert_eq!(model.factors()[1].name(), Some("λ\"; panic!()"));
    assert!(!model.rust_objective_primal().unwrap().contains("panic!"));
}

#[test]
fn excluded_callbacks_are_not_executed_or_required_for_source_generation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    b.likelihood("data", x).unwrap();
    let count = Arc::clone(&calls);
    let bad = b
        .custom(&[x], move |_, _| {
            count.fetch_add(1, Ordering::Relaxed);
            Err(AssemblyError::Numerical)
        })
        .unwrap();
    b.prior("opaque failing prior", bad).unwrap();
    let selected = b.compile().unwrap().select_likelihood().unwrap();
    let mut g = [99.];
    assert_eq!(selected.logp_grad(&[3.], &mut g).unwrap(), 3.);
    assert_eq!(g, [1.]);
    selected.rust_primal().unwrap();
    selected.rust_objective_primal().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let posterior = selected.select_posterior().unwrap();
    assert!(matches!(
        posterior.rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    g[0] = 99.;
    assert!(posterior.logp_grad(&[3.], &mut g).is_err());
    assert_eq!(g, [99.]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let recovered = posterior.select_likelihood().unwrap();
    assert_eq!(recovered.logp_grad(&[4.], &mut g).unwrap(), 4.);
    assert_eq!(g, [1.]);
}

#[test]
fn selected_callbacks_recover_after_unwind_without_publishing_partial_scores() {
    let mode = Arc::new(AtomicUsize::new(0));
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let state = Arc::clone(&mode);
    let term = b
        .custom(&[x], move |x, g| match state.load(Ordering::Relaxed) {
            1 => panic!("injected callback panic"),
            2 => {
                g[0] = f64::NAN;
                Ok(x[0])
            }
            _ => {
                g[0] = 2. * x[0];
                Ok(x[0] * x[0])
            }
        })
        .unwrap();
    b.likelihood("data", term).unwrap();
    let objective = b
        .compile()
        .unwrap()
        .select_likelihood()
        .unwrap()
        .into_objective()
        .unwrap();
    let mut g = [99.];
    mode.store(1, Ordering::Relaxed);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || objective.logp_grad(&[3.], &mut g)
        ))
        .is_err()
    );
    assert_eq!(g, [99.]);
    mode.store(2, Ordering::Relaxed);
    assert!(objective.logp_grad(&[3.], &mut g).is_err());
    assert_eq!(g, [99.]);
    mode.store(0, Ordering::Relaxed);
    assert_eq!(objective.logp_grad(&[3.], &mut g).unwrap(), 9.);
    assert_eq!(g, [6.]);
}

#[test]
#[cfg(not(miri))]
fn selected_evaluation_reuses_storage() {
    let objective = named(false)
        .select_likelihood()
        .unwrap()
        .into_objective()
        .unwrap();
    let mut g = [0.];
    let measured = allocation_counter::measure(|| {
        for _ in 0..64 {
            std::hint::black_box(objective.logp_grad(&[0.4], &mut g).unwrap());
        }
    });
    assert_eq!(measured.count_total, 0);
}

#[test]
fn likelihood_selection_retains_prior_only_coordinates_with_zero_score() {
    let mut b = ModelBuilder::new();
    let x = b.real("data parameter").unwrap().expr();
    let unused = b.positive("prior-only scale").unwrap().expr();
    b.likelihood("data", x).unwrap();
    b.prior("scale prior", unused).unwrap();
    let objective = b
        .compile()
        .unwrap()
        .select_likelihood()
        .unwrap()
        .into_objective()
        .unwrap();
    let mut g = [99.; 2];
    assert_eq!(objective.logp_grad(&[3., 0.4], &mut g).unwrap(), 3.);
    assert_eq!(g, [1., 0.]);
    assert_eq!(objective.model().coordinates().len(), 2);
    let posterior = objective
        .into_model()
        .select_posterior()
        .unwrap()
        .into_objective()
        .unwrap();
    assert_eq!(
        posterior.logp_grad(&[3., 0.4], &mut g).unwrap(),
        3. + 0.4_f64.exp()
    );
    assert_eq!(g, [1., 0.4_f64.exp()]);
}
