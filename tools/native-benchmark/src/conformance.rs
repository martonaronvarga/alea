//! Native block models and the retained scalar source-reference definition.
use alea::distributions::{
    modeling::{continuous, normal},
    scalar::Cauchy,
};
use alea::model::assembly::{AssemblyError as E, ModelBuilder, NativeModel};

/// Retained scalar reference for source export and native cross-checks.
/// Existing PosteriorDB eight-schools models, in their original parameter order.
/// Covariance is a separate synthetic Cholesky model, not a PosteriorDB port.
pub fn build(name: &str, n: usize, k: usize, x: &[f64], y: &[f64]) -> Result<NativeModel, E> {
    if n == 0 || y.len() != n || n.checked_mul(k) != Some(x.len()) {
        return Err(E::Shape);
    }
    let mut b = ModelBuilder::new();
    if matches!(name, "schools" | "schools_nc") {
        if k != 1 || x.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(E::Shape);
        }
        let effects = (0..n)
            .map(|i| b.real(&format!("effect{i}")).map(|p| p.expr()))
            .collect::<Result<Vec<_>, _>>()?;
        let mu = b.real("mu")?;
        let tau = b.positive("tau")?;
        let zero = b.constant(0.)?;
        let one = b.constant(1.)?;
        let five = b.constant(5.)?;
        let prior = normal(&mut b, mu, zero, five)?;
        b.prior("mu", prior)?;
        // This existing opaque law has no source rule. Source AD must report
        // unsupported, never replace the prior or invent a benchmark derivative.
        let prior = continuous(&mut b, tau.expr(), &[zero, five], |p| {
            Cauchy::new(p[0], p[1])
        })?;
        b.prior("tau", prior)?;
        for (i, &effect) in effects.iter().enumerate() {
            let centered = name == "schools";
            let prior = normal(
                &mut b,
                effect,
                if centered { mu.expr() } else { zero },
                if centered { tau.expr() } else { one },
            )?;
            b.prior(&format!("effect{i}"), prior)?;
            let theta = if centered {
                effect
            } else {
                let scaled = b.multiply(effect, tau)?;
                b.add(mu, scaled)?
            };
            let obs = b.constant(y[i])?;
            let sd = b.constant(x[i])?;
            let likelihood = normal(&mut b, obs, theta, sd)?;
            b.likelihood(&format!("observation{i}"), likelihood)?;
        }
    } else if name == "covariance" {
        if k != 2 {
            return Err(E::Shape);
        }
        let mu0 = b.real("mu0")?;
        let mu1 = b.real("mu1")?;
        let s0 = b.positive("s0")?;
        let s1 = b.positive("s1")?;
        let loading = b.real("loading")?;
        for (name, p, sd) in [
            ("mu0", mu0.expr(), 2.),
            ("mu1", mu1.expr(), 2.),
            ("s0", s0.expr(), 1.),
            ("s1", s1.expr(), 1.),
            ("loading", loading.expr(), 1.),
        ] {
            let zero = b.constant(0.)?;
            let sd = b.constant(sd)?;
            let prior = normal(&mut b, p, zero, sd)?;
            b.prior(name, prior)?;
        }
        // L=[[s0,0],[loading,s1]]: conditional native normals are precisely
        // multi_normal_cholesky(mu,L). No benchmark density/score implementation.
        let log_s0 = b.log(s0)?;
        let negative_log_s0 = b.multiply(log_s0, -1.)?;
        let inverse_s0 = b.exp(negative_log_s0)?;
        let slope = b.multiply(loading, inverse_s0)?;
        let negative_mu0 = b.multiply(mu0, -1.)?;
        for i in 0..n {
            let a = b.constant(x[2 * i])?;
            let d = b.add(a, negative_mu0)?;
            let d = b.multiply(d, slope)?;
            let mean = b.add(mu1, d)?;
            let c = b.constant(x[2 * i + 1])?;
            let first = normal(&mut b, a, mu0, s0)?;
            let second = normal(&mut b, c, mean, s1)?;
            let pair = b.add(first, second)?;
            b.likelihood(&format!("pair{i}"), pair)?;
        }
    } else {
        return Err(E::Shape);
    }
    b.compile()
}

/// Dyadic fixture shared by the generated target and runtime data admission.
#[cfg(any(test, feature = "enzyme"))]
pub fn covariance_data() -> (Vec<f64>, Vec<f64>) {
    let x = (0..32)
        .flat_map(|i| {
            let a = (i % 7) as f64 / 4. - 0.75;
            [a, 0.5 * a + (i % 11) as f64 / 8. - 0.625]
        })
        .collect();
    (x, vec![0.; 32])
}

use alea::{
    distributions::{modeling::observations, scalar::Normal},
    model::{
        assembly::Value,
        tensor::{BinaryOp as B, Shape},
    },
};

fn binary(b: &mut ModelBuilder, op: B, a: Value, c: Value) -> Result<Value, E> {
    let mut r = b.region();
    let left = r.input("left", b.value_shape(a)?)?;
    let right = r.input("right", b.value_shape(c)?)?;
    let out = r.binary(op, left, right)?;
    b.call_region(&[a, c], r.compile(out)?.bind(&[])?)
}
fn normal_block(
    b: &mut ModelBuilder,
    events: Vec<f64>,
    mean: Value,
    scale: Value,
) -> Result<Value, E> {
    let rows =
        observations::<Normal, _, _>(b, events, &[mean, scale], |p| Normal::new(p[0], p[1]))?;
    b.sum_value(rows)
}

/// Ordinary native block compositions; parameter order matches the Stan models.
/// Source export is intentionally unsupported for these whole-value calls.
pub fn native(name: &str, n: usize, k: usize, x: &[f64], y: &[f64]) -> Result<NativeModel, E> {
    if n == 0 || y.len() != n || n.checked_mul(k) != Some(x.len()) {
        return Err(E::Shape);
    }
    let mut b = ModelBuilder::new();
    if matches!(name, "schools" | "schools_nc") {
        if k != 1 || x.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(E::Shape);
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        let labels: Vec<_> = (1..=n).map(|i| i.to_string()).collect();
        let labels: Vec<_> = labels.iter().map(String::as_str).collect();
        let effects = b.real_vector(if name == "schools" { "theta" } else { "z" }, &labels)?;
        let effects = b.vector_value(&effects, Shape::new(vec![n])?)?;
        let mu = b.real("mu")?;
        let tau = b.positive("tau")?;
        let zero = b.constant(0.)?;
        let one = b.constant(1.)?;
        let five = b.constant(5.)?;
        let prior = normal(&mut b, mu, zero, five)?;
        b.prior("mu", prior)?;
        // Preserve the actual opaque law and its source limitation.
        let prior = continuous(&mut b, tau.expr(), &[zero, five], |p| {
            Cauchy::new(p[0], p[1])
        })?;
        b.prior("tau", prior)?;
        let (prior_mean, prior_scale, theta) = if name == "schools" {
            let centered = binary(&mut b, B::Subtract, effects, mu.expr())?;
            (centered, tau.expr(), effects)
        } else {
            let scaled = binary(&mut b, B::Multiply, tau.expr(), effects)?;
            let theta = binary(&mut b, B::Add, mu.expr(), scaled)?;
            (effects, one, theta)
        };
        // Normal symmetry: N(effect | mean,sd) = N(0 | effect-mean,sd).
        // Fixed zeros let the existing parameter-only observed-law map own scores.
        let prior = normal_block(&mut b, vec![0.; n], prior_mean, prior_scale)?;
        b.prior("effects", prior)?;
        let sd = b.constant_block(Shape::new(vec![n])?, x.to_vec())?;
        let likelihood = normal_block(&mut b, y.to_vec(), theta, sd)?;
        b.likelihood("observations", likelihood)?;
    } else if name == "covariance" {
        if k != 2 {
            return Err(E::Shape);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        let mu0 = b.real("mu0")?;
        let mu1 = b.real("mu1")?;
        let s0 = b.positive("s0")?;
        let s1 = b.positive("s1")?;
        let loading = b.real("loading")?;
        for (name, p, sd) in [
            ("mu0", mu0.expr(), 2.),
            ("mu1", mu1.expr(), 2.),
            ("s0", s0.expr(), 1.),
            ("s1", s1.expr(), 1.),
            ("loading", loading.expr(), 1.),
        ] {
            let prior = normal(&mut b, p, 0., sd)?;
            b.prior(name, prior)?;
        }
        let log_s0 = b.log(s0)?;
        let negative = b.multiply(log_s0, -1.)?;
        let inverse = b.exp(negative)?;
        let slope = b.multiply(loading, inverse)?;
        let negative_mu0 = b.multiply(mu0, -1.)?;
        let a: Vec<_> = x.chunks_exact(2).map(|row| row[0]).collect();
        let c: Vec<_> = x.chunks_exact(2).map(|row| row[1]).collect();
        let data = b.constant_block(Shape::new(vec![n])?, a.clone())?;
        let delta = binary(&mut b, B::Add, data, negative_mu0)?;
        let scaled = binary(&mut b, B::Multiply, delta, slope)?;
        let mean = binary(&mut b, B::Add, mu1.expr(), scaled)?;
        let first = normal_block(&mut b, a, mu0.expr(), s0.expr())?;
        b.likelihood("first coordinate", first)?;
        let second = normal_block(&mut b, c, mean, s1.expr())?;
        b.likelihood("second conditional", second)?;
    } else {
        return Err(E::Shape);
    }
    b.compile()
}

/// Pinned Radon model, with already validated zero-based county indices.
/// No model registry, density formula, source port or reparameterization is added.
pub fn radon(j: usize, county: &[usize], floor: &[f64], y: &[f64]) -> Result<NativeModel, E> {
    radon_representation(j, county, floor, y, true)
}

/// Scalar/native control of exactly the same Radon posterior and coordinates.
#[allow(dead_code)] // Used by the shared Criterion fixture, not the sampling CLI.
pub fn radon_scalar(
    j: usize,
    county: &[usize],
    floor: &[f64],
    y: &[f64],
) -> Result<NativeModel, E> {
    radon_representation(j, county, floor, y, false)
}

fn radon_representation(
    j: usize,
    county: &[usize],
    floor: &[f64],
    y: &[f64],
    blocks: bool,
) -> Result<NativeModel, E> {
    let n = y.len();
    if j == 0 || n == 0 || county.len() != n || floor.len() != n || county.iter().any(|&i| i >= j) {
        return Err(E::Shape);
    }
    // Reject structural capacity overflow before allocating parameter labels.
    j.checked_add(4)
        .and_then(|slots| slots.checked_mul(size_of::<String>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(E::Shape)?;
    if floor.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(E::Numerical);
    }
    let mut b = ModelBuilder::new();
    let labels: Vec<_> = (1..=j).map(|i| i.to_string()).collect();
    let labels: Vec<_> = labels.iter().map(String::as_str).collect();
    let raw = b.real_vector("alpha_raw", &labels)?;
    let raw_value = if blocks {
        Some(b.vector_value(&raw, Shape::new(vec![j])?)?)
    } else {
        None
    };
    let beta = b.real("beta")?;
    let mu = b.real("mu_alpha")?;
    let tau = b.positive("sigma_alpha")?;
    let sigma = b.positive("sigma_y")?;
    let one = b.constant(1.)?;
    if let Some(values) = raw_value {
        let prior = normal_block(&mut b, vec![0.; j], values, one)?;
        b.prior("alpha_raw", prior)?;
    } else {
        for (i, &value) in raw.expressions().iter().enumerate() {
            let prior = normal(&mut b, value, 0., one)?;
            b.prior(&format!("alpha_raw[{i}]"), prior)?;
        }
    }
    for (name, p, sd) in [
        ("beta", beta.expr(), 10.),
        ("mu_alpha", mu.expr(), 10.),
        ("sigma_alpha", tau.expr(), 1.),
        ("sigma_y", sigma.expr(), 1.),
    ] {
        let prior = normal(&mut b, p, 0., sd)?;
        b.prior(name, prior)?;
    }
    if !blocks {
        for (i, ((&group, &floor), &event)) in county.iter().zip(floor).zip(y).enumerate() {
            let scaled = b.multiply(tau, raw.expressions()[group])?;
            let alpha = b.add(mu, scaled)?;
            let slope = b.multiply(beta, floor)?;
            let mean = b.add(alpha, slope)?;
            let likelihood = normal(&mut b, event, mean, sigma)?;
            b.likelihood(&format!("observation{i}"), likelihood)?;
        }
        return b.compile();
    }
    let raw = raw_value.expect("block representation has one shared vector view");
    let scaled = binary(&mut b, B::Multiply, tau.expr(), raw)?;
    let alpha = binary(&mut b, B::Add, mu.expr(), scaled)?;
    let mut r = b.region();
    let input = r.input("alpha", Shape::new(vec![j])?)?;
    let selected = r.gather(input, Shape::new(vec![n])?, county.to_vec())?;
    let binding = r.compile(selected)?.bind(&[])?;
    let selected = b.call_region(&[alpha], binding)?;
    let floors = b.constant_block(Shape::new(vec![n])?, floor.to_vec())?;
    let floor_effect = binary(&mut b, B::Multiply, floors, beta.expr())?;
    let mean = binary(&mut b, B::Add, selected, floor_effect)?;
    let likelihood = normal_block(&mut b, y.to_vec(), mean, sigma.expr())?;
    b.likelihood("observations", likelihood)?;
    b.compile()
}
