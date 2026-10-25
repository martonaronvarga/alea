use alea_core::{
    model::assembly::{AssemblyError, CompiledModel, ModelBuilder},
    transform::Transform,
};
use alea_distributions::modeling::{bernoulli_logit, normal};

pub fn polynomial() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let x = b.real("x")?.expr();
    let y = b.real("y")?.expr();
    let xx = b.multiply(x, x)?;
    let yy = b.multiply(y, y)?;
    let sum = b.add(xx, yy)?;
    let c = b.constant(-0.5)?;
    let gaussian = b.multiply(c, sum)?;
    b.factor(gaussian)?;
    let product = b.multiply(xx, yy)?;
    let c = b.constant(-0.1)?;
    let cross = b.multiply(c, product)?;
    b.factor(cross)?;
    let fourth = b.multiply(xx, xx)?;
    let c = b.constant(-0.01)?;
    let fourth = b.multiply(c, fourth)?;
    b.factor(fourth)?;
    b.compile()
}

pub fn regression() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu")?.expr();
    let sigma = b.positive("sigma")?.expr();
    let zero = b.constant(0.0)?;
    let one = b.constant(1.0)?;
    let prior = normal(&mut b, mu, zero, one)?;
    b.factor(prior)?;
    // A proper half-normal scale prior; its support is supplied by the transform.
    let prior = normal(&mut b, sigma, zero, one)?;
    b.factor(prior)?;
    let half_normal_normalizer = b.constant(std::f64::consts::LN_2)?;
    b.factor(half_normal_normalizer)?;
    b.plate(&[-1.0, 0.5, 2.0], |b, _, &y| {
        let event = b.constant(y)?;
        normal(b, event, mu, sigma)
    })?;
    b.compile()
}

pub fn logistic() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let intercept = b.real("intercept")?.expr();
    let slope = b.real("slope")?.expr();
    let zero = b.constant(0.0)?;
    let one = b.constant(1.0)?;
    for parameter in [intercept, slope] {
        let prior = normal(&mut b, parameter, zero, one)?;
        b.factor(prior)?;
    }
    b.plate(
        &[(-1.0, false), (0.0, true), (0.5, false), (2.0, true)],
        |b, _, &(x, y)| {
            let x = b.constant(x)?;
            let linear = b.multiply(x, slope)?;
            let logit = b.add(intercept, linear)?;
            bernoulli_logit(b, y, logit)
        },
    )?;
    b.compile()
}

pub fn transformed(transform: Transform) -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let names: Vec<_> = (0..transform.constrained_dimension())
        .map(|i| i.to_string())
        .collect();
    let names: Vec<_> = names.iter().map(String::as_str).collect();
    let x = b.vector("x", "coordinates", &names, "x", transform)?;
    let half = b.constant(-0.5)?;
    for &x in x.expressions() {
        let square = b.multiply(x, x)?;
        let factor = b.multiply(half, square)?;
        b.factor(factor)?;
    }
    b.compile()
}

pub fn all() -> Result<Vec<(&'static str, CompiledModel)>, AssemblyError> {
    let mut models = vec![
        ("polynomial", polynomial()?),
        ("regression", regression()?),
        ("logistic", logistic()?),
    ];
    for (name, transform) in [
        ("interval", Transform::interval(1, -2.0, 3.0)?),
        ("lower", Transform::lower(1, -2.0)?),
        ("upper", Transform::upper(1, 3.0)?),
        ("ordered", Transform::ordered(2)?),
        ("positive_ordered", Transform::positive_ordered(2)?),
        ("simplex", Transform::simplex(3)?),
        ("unit_vector", Transform::unit_vector(3)?),
        ("cholesky_covariance", Transform::cholesky_covariance(2)?),
        ("cholesky_correlation", Transform::cholesky_correlation(2)?),
        ("covariance", Transform::covariance(2)?),
        ("correlation", Transform::correlation(2)?),
        ("covariance3", Transform::covariance(3)?),
        ("correlation3", Transform::correlation(3)?),
        ("gaussian16", Transform::identity(16)?),
    ] {
        models.push((name, transformed(transform)?));
    }
    Ok(models)
}
