//! Downstream-owned examples using only public Alea APIs.
use alea_core::model::assembly::{CompiledModel, Expr, ModelBuilder};
use alea_distributions::{
    modeling,
    wiener::{WienerObservation, WienerOptions},
};

#[derive(Clone, Copy)]
pub enum Observation {
    Logistic {
        group: usize,
        event: bool,
    },
    Ddm {
        group: usize,
        event: WienerObservation,
    },
}
fn factor(b: &mut ModelBuilder, x: Expr) {
    b.factor(x).unwrap();
}
fn normal_prior(b: &mut ModelBuilder, x: Expr, mean: f64, scale: f64) {
    let m = b.constant(mean).unwrap();
    let s = b.constant(scale).unwrap();
    let p = modeling::normal(b, x, m, s).unwrap();
    factor(b, p);
}
/// Intercept/slope normal regression with fixed observation scale .7 and N(0,1) priors.
pub fn regression(data: &[(f64, f64)]) -> CompiledModel {
    let mut b = ModelBuilder::new();
    let a = b.real("intercept").unwrap();
    let slope = b.real("slope").unwrap();
    b.annotate(&a, "coefficient", "intercept", "regression")
        .unwrap();
    b.annotate(&slope, "coefficient", "slope", "regression")
        .unwrap();
    normal_prior(&mut b, a.expr(), 0.0, 1.0);
    normal_prior(&mut b, slope.expr(), 0.0, 1.0);
    let sigma = b.constant(0.7).unwrap();
    b.plate(data, |b, _, (x, y)| {
        let x = b.constant(*x)?;
        let y = b.constant(*y)?;
        let product = b.multiply(slope.expr(), x)?;
        let mean = b.add(a.expr(), product)?;
        modeling::normal(b, y, mean, sigma)
    })
    .unwrap();
    b.compile().unwrap()
}
/// Noncentered two-group hierarchy. Parameter order: `mu`, `tau`, `z[0]`, `z[1]`.
/// tau has LogNormal(-.7,.3), mu has Normal(0,.7), z have independent Normal(0,1).
pub fn hierarchy(data: &[Observation]) -> CompiledModel {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap();
    let tau = b.positive("tau").unwrap();
    let z = [b.real("z[0]").unwrap(), b.real("z[1]").unwrap()];
    b.annotate(&mu, "hyperparameter", "mean", "hyper").unwrap();
    b.annotate(&tau, "hyperparameter", "scale", "hyper")
        .unwrap();
    normal_prior(&mut b, mu.expr(), 0.0, 0.7);
    let logtau = b.log(tau.expr()).unwrap();
    normal_prior(&mut b, logtau, -0.7, 0.3);
    let minus = b.constant(-1.0).unwrap();
    let correction = b.multiply(logtau, minus).unwrap();
    factor(&mut b, correction);
    let mut effects = Vec::new();
    for (i, z) in z.iter().enumerate() {
        b.annotate(z, "group", if i == 0 { "a" } else { "b" }, "effects")
            .unwrap();
        normal_prior(&mut b, z.expr(), 0.0, 1.0);
        let scaled = b.multiply(tau.expr(), z.expr()).unwrap();
        effects.push(b.add(mu.expr(), scaled).unwrap());
    }
    let alpha = b.constant(1.2).unwrap();
    let ndt = b.constant(0.15).unwrap();
    let beta = b.constant(0.5).unwrap();
    b.plate(data, |b, _, row| match *row {
        Observation::Logistic { group, event } => {
            modeling::bernoulli_logit(b, event, effects[group])
        }
        Observation::Ddm { group, event } => modeling::wiener4(
            b,
            event,
            [alpha, ndt, beta, effects[group]],
            WienerOptions::default(),
        ),
    })
    .unwrap();
    b.compile().unwrap()
}
