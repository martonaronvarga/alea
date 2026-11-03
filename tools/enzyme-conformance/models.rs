use alea::distributions::modeling::{bernoulli_logit, normal};
use alea::{
    model::assembly::{AssemblyError, CompiledModel, ModelBuilder},
    transform::Transform,
};

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
    // Same graph/order as the explicit construction, now exercising the public
    // macro in an actual downstream build-script -> Enzyme compilation path.
    alea::model! { |b| parameters { mu: real(), sigma: positive() } {
        let zero = b.constant(0.0)?;
        let one = b.constant(1.0)?;
        let prior = normal(&mut b, mu, zero, one)?;
        b.factor(prior)?;
        // A proper half-normal; its support comes from the declared chart.
        let prior = normal(&mut b, sigma, zero, one)?;
        b.factor(prior)?;
        let normalizer = b.constant(std::f64::consts::LN_2)?;
        b.factor(normalizer)?;
        b.plate(&[-1.0, 0.5, 2.0], |b, _, &y| {
            let event = b.constant(y)?;
            normal(b, event, mu, sigma)
        })?;
    }}
}

pub fn likelihood_location_scale() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu")?.expr();
    let sigma = b.positive("sigma")?.expr();
    for (i, y) in [-1., 0.5, 2.].into_iter().enumerate() {
        let event = b.constant(y)?;
        let term = normal(&mut b, event, mu, sigma)?;
        b.likelihood(&format!("observation {i}"), term)?;
    }
    b.compile()?.select_likelihood()
}

// Dyadic full-rank fixtures with a nonzero coefficient cross-covariance.
// The residual vector is orthogonal to both supplied design columns.
pub fn uncertainty_data(x_scale: f64, y_scale: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut design = Vec::new();
    let mut response = Vec::new();
    let mut offsets = Vec::new();
    for (i, (x, error)) in [
        (-2., 1.),
        (-1., -2.),
        (0., 1.),
        (1., 1.),
        (2., -2.),
        (3., 1.),
    ]
    .into_iter()
    .enumerate()
    {
        let offset = (i + 1) as f64 * 0.25 * if i % 2 == 0 { 1. } else { -1. };
        design.extend([1., x * x_scale]);
        offsets.push(offset * y_scale);
        response.push((1.5 - 0.75 * x + error + offset) * y_scale);
    }
    (design, response, offsets)
}

pub fn uncertainty_regression(x_scale: f64, y_scale: f64) -> Result<CompiledModel, AssemblyError> {
    let (design, response, offsets) = uncertainty_data(x_scale, y_scale);
    let mut b = ModelBuilder::new();
    let intercept = b.real("intercept")?.expr();
    let slope = b.real("slope")?.expr();
    let sigma = b.positive("sigma")?.expr();
    for i in 0..response.len() {
        let x = b.constant(design[2 * i + 1])?;
        let linear = b.multiply(x, slope)?;
        let linear = b.add(intercept, linear)?;
        let offset = b.constant(offsets[i])?;
        let mean = b.add(linear, offset)?;
        let y = b.constant(response[i])?;
        let term = normal(&mut b, y, mean, sigma)?;
        b.likelihood(&format!("observation {i}"), term)?;
    }
    b.compile()?.select_likelihood()
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
    let x = b.vector::<alea::model::assembly::DynamicDomain>(
        "x",
        "coordinates",
        &names,
        "x",
        transform,
    )?;
    let half = b.constant(-0.5)?;
    for &x in x.expressions() {
        let square = b.multiply(x, x)?;
        let factor = b.multiply(half, square)?;
        b.factor(factor)?;
    }
    b.compile()
}

fn statistical(posterior: bool) -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let theta = b.positive("theta")?.expr();
    let event = b.constant(2.)?;
    let one = b.constant(1.)?;
    let data = normal(&mut b, event, theta, one)?;
    b.likelihood("observed data", data)?;
    let prior = if posterior {
        normal(&mut b, theta, one, one)?
    } else {
        // An excluded opaque prior must not require a differentiable primal.
        b.custom(&[theta], |_, _| Err(AssemblyError::Numerical))?
    };
    b.prior("prior", prior)?;
    let model = b.compile()?;
    if posterior {
        model.select_posterior()
    } else {
        model.select_likelihood()
    }
}

pub fn native_batch(family: usize) -> Result<CompiledModel, AssemblyError> {
    use alea::distributions::modeling::regression::*;
    let mut b = ModelBuilder::new();
    let a = b.real("a")?;
    let slope = b.real("slope")?;
    let beta = [a.expr(), b.multiply(slope, slope)?];
    let design = vec![1., -1., 1., 0.5, 1., 2.];
    let offsets = vec![0.25, -0.5, 0.75];
    let term = match family {
        0 => {
            let sigma = b.positive("sigma")?;
            normal_regression(
                &mut b,
                &beta,
                &sigma,
                RegressionData::new(2, design, vec![0.5, -1., 1.5])
                    .unwrap()
                    .with_offsets(offsets)
                    .unwrap(),
            )?
        }
        1 => bernoulli_logit_regression(
            &mut b,
            &beta,
            RegressionData::new(2, design, vec![false, true, true])
                .unwrap()
                .with_offsets(offsets)
                .unwrap(),
        )?,
        _ => poisson_log_regression(
            &mut b,
            &beta,
            RegressionData::new(2, design, vec![0, 2, 3])
                .unwrap()
                .with_offsets(offsets)
                .unwrap(),
        )?,
    };
    b.likelihood("data", term)?;
    b.compile()
}
pub fn tensor_matrix() -> Result<CompiledModel, AssemblyError> {
    tensor_matrix_policy(false)
}
pub fn tensor_checkpoint_matrix() -> Result<CompiledModel, AssemblyError> {
    tensor_matrix_policy(true)
}
fn checkpoint_identity(
    t: &mut alea::model::tensor::TensorBuilder,
    value: alea::model::tensor::Value,
) -> Result<alea::model::tensor::Value, AssemblyError> {
    use alea::model::tensor::*;
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar())?;
    let _row = body.input("row", Shape::scalar())?;
    let body = body.compile(carry)?.bind(&[])?;
    let rows = t.constant(Shape::new(vec![0])?, vec![])?;
    t.checkpointed_scan(value, rows, &[], body)
}
fn tensor_matrix_policy(checkpoint: bool) -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::*;
    let mut t = TensorBuilder::new();
    let q = t.input("q", Shape::new(vec![2, 1])?)?;
    let matrix = t.data("matrix", Shape::new(vec![2, 2])?)?;
    let means = t.matmul(matrix, q)?;
    let squared = t.binary(BinaryOp::Multiply, means, means)?;
    let sum = t.sum(squared)?;
    let scale = t.constant(Shape::scalar(), vec![-0.5])?;
    let result = t.binary(BinaryOp::Multiply, sum, scale)?;
    let result = if checkpoint {
        checkpoint_identity(&mut t, result)?
    } else {
        result
    };
    let binding = t
        .compile(result)?
        .bind(&[("matrix", &[1., 0.5, -0.5, 1.5])])?;
    let mut b = ModelBuilder::new();
    let q = [b.real("a")?.expr(), b.real("b")?.expr()];
    let term = b.primitive(TensorFactor::new(binding, &q)?)?;
    b.potential("objective", term)?;
    b.compile()
}
pub fn tensor_control() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(3, false, false)
}
pub fn tensor_zero_scan() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(0, false, false)
}
pub fn tensor_checkpoint() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(3, true, false)
}
pub fn tensor_checkpoint_zero() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(0, true, false)
}
pub fn tensor_checkpoint_long() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(257, true, false)
}
pub fn tensor_retained_long() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(257, false, false)
}
pub fn tensor_checkpoint_mixed() -> Result<CompiledModel, AssemblyError> {
    tensor_control_steps(3, true, true)
}
pub fn tensor_checkpoint_activity() -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::*;
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar())?;
    let _row = body.input("row", Shape::scalar())?;
    let gain = body.input("gain", Shape::scalar())?;
    let root = body.unary(UnaryOp::Sqrt, carry)?;
    let next = body.binary(BinaryOp::Add, root, gain)?;
    let body = body.compile(next)?.bind(&[])?;
    let mut t = TensorBuilder::new();
    let gain = t.input("gain", Shape::scalar())?;
    let zero = t.constant(Shape::scalar(), vec![0.])?;
    let rows = t.constant(Shape::new(vec![3])?, vec![0.; 3])?;
    let result = t.checkpointed_scan(zero, rows, &[gain], body)?;
    let binding = t.compile(result)?.bind(&[])?;
    let mut b = ModelBuilder::new();
    let gain = b.real("gain")?.expr();
    let term = b.primitive(TensorFactor::new(binding, &[gain])?)?;
    b.potential("objective", term)?;
    b.compile()
}
pub fn tensor_checkpoint_immutable() -> Result<CompiledModel, AssemblyError> {
    tensor_checkpoint_immutable_policy(true)
}
pub fn tensor_checkpoint_constant() -> Result<CompiledModel, AssemblyError> {
    tensor_checkpoint_immutable_policy(false)
}
fn tensor_checkpoint_immutable_policy(variable: bool) -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::*;
    let mut t = TensorBuilder::new();
    let gain = t.input("gain", Shape::scalar())?;
    let alias = t.reshape(gain, Shape::new(vec![1])?)?;
    let data = t.data("data", Shape::new(vec![4])?)?;
    // sqrt(0) has a finite primal but a singular derivative. Immutable data
    // must never acquire that derivative during the source pullback.
    let roots = t.unary(UnaryOp::Sqrt, data)?;
    let sum = t.sum(roots)?;
    let scalar = if variable {
        let product = t.binary(BinaryOp::Multiply, alias, sum)?;
        t.sum(product)?
    } else {
        sum
    };
    let result = checkpoint_identity(&mut t, scalar)?;
    let binding = t.compile(result)?.bind(&[("data", &[0., 1., 4., 9.])])?;
    let mut b = ModelBuilder::new();
    let gain = b.real("gain")?.expr();
    let term = b.primitive(TensorFactor::new(binding, &[gain])?)?;
    b.potential("objective", term)?;
    b.compile()
}
pub fn tensor_checkpoint_vector() -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::*;
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::new(vec![2])?)?;
    let row = body.input("row", Shape::scalar())?;
    let gain = body.input("gain", Shape::scalar())?;
    let scaled = body.binary(BinaryOp::Multiply, carry, gain)?;
    let shifted = body.binary(BinaryOp::Add, scaled, row)?;
    let next = body.unary(UnaryOp::Tanh, shifted)?;
    let body = body.compile(next)?.bind(&[])?;
    let mut t = TensorBuilder::new();
    let initial = t.input("initial", Shape::new(vec![2])?)?;
    let gain = t.input("gain", Shape::scalar())?;
    let rows = t.constant(Shape::new(vec![65])?, vec![0.1; 65])?;
    let result = t.checkpointed_scan(initial, rows, &[gain], body)?;
    let result = t.sum(result)?;
    let binding = t.compile(result)?.bind(&[])?;
    let mut b = ModelBuilder::new();
    let q = [
        b.real("a")?.expr(),
        b.real("b")?.expr(),
        b.real("gain")?.expr(),
    ];
    let term = b.primitive(TensorFactor::new(binding, &q)?)?;
    b.potential("objective", term)?;
    b.compile()
}
pub fn tensor_checkpoint_routes() -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::*;
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar())?;
    let row = body.input("row", Shape::scalar())?;
    let bias = body.input("bias", Shape::scalar())?;
    let gain = body.input("gain", Shape::scalar())?;
    let other = body.input("other", Shape::scalar())?;
    let carry = body.binary(BinaryOp::Multiply, carry, gain)?;
    let row = body.binary(BinaryOp::Multiply, row, other)?;
    let next = body.binary(BinaryOp::Add, carry, row)?;
    let next = body.binary(BinaryOp::Add, next, bias)?;
    let body = body.compile(next)?.bind(&[])?;
    let mut t = TensorBuilder::new();
    let initial = t.input("initial", Shape::scalar())?;
    let rows = t.input("rows", Shape::new(vec![3])?)?;
    let gain = t.input("gain", Shape::scalar())?;
    let alias = t.reshape(gain, Shape::scalar())?;
    let bias = t.constant(Shape::scalar(), vec![0.25])?;
    let result = t.checkpointed_scan(initial, rows, &[bias, gain, alias], body)?;
    let binding = t.compile(result)?.bind(&[])?;
    let mut b = ModelBuilder::new();
    let q: Vec<_> = (0..5)
        .map(|i| b.real(&format!("q{i}")).map(|v| v.expr()))
        .collect::<Result<_, _>>()?;
    let term = b.primitive(TensorFactor::new(binding, &q)?)?;
    b.potential("objective", term)?;
    b.compile()
}
fn tensor_control_steps(
    steps: usize,
    checkpoint: bool,
    mixed: bool,
) -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::*;
    let mut body = TensorBuilder::new();
    let carry = body.input("carry", Shape::scalar())?;
    let row = body.input("row", Shape::scalar())?;
    let gain = body.input("gain", Shape::scalar())?;
    let scaled = body.binary(BinaryOp::Multiply, carry, gain)?;
    let next = body.binary(BinaryOp::Add, scaled, row)?;
    let body = body.compile(next)?.bind(&[])?;
    fn branch(negative: bool) -> Result<BoundProgram, AssemblyError> {
        let mut b = TensorBuilder::new();
        let x = b.input("x", Shape::scalar())?;
        let x = if negative {
            b.unary(UnaryOp::Neg, x)?
        } else {
            x
        };
        let y = b.unary(UnaryOp::Log, x)?;
        b.compile(y)?.bind(&[])
    }
    let mut t = TensorBuilder::new();
    let initial = t.input("initial", Shape::scalar())?;
    let gain = t.input("gain", Shape::scalar())?;
    let xs = t.data("xs", Shape::new(vec![steps])?)?;
    let carry = if checkpoint {
        t.checkpointed_scan(initial, xs, &[gain], body)?
    } else {
        t.scan(initial, xs, &[gain], body)?
    };
    let zero = t.constant(Shape::scalar(), vec![0.])?;
    let out = t.cond(
        carry,
        zero,
        Comparison::Greater,
        &[carry],
        branch(false)?,
        branch(true)?,
    )?;
    let rows: Vec<_> = (0..steps).map(|i| [-0.1, 0.25, 0.5][i % 3]).collect();
    let binding = t.compile(out)?.bind(&[("xs", &rows)])?;
    let mut b = ModelBuilder::new();
    let initial = if mixed {
        b.positive("initial")?.expr()
    } else {
        b.real("initial")?.expr()
    };
    let q = [initial, b.real("gain")?.expr()];
    let term = b.primitive(TensorFactor::new(binding, &q)?)?;
    let term = if mixed {
        let zero = b.constant(0.)?;
        let one = b.constant(1.)?;
        let prior = normal(&mut b, initial, zero, one)?;
        b.prior("prior", prior)?;
        b.multiply(term, term)?
    } else {
        term
    };
    b.potential("objective", term)?;
    b.compile()
}

pub fn tensor_empty() -> Result<CompiledModel, AssemblyError> {
    tensor_empty_policy(false)
}
pub fn tensor_checkpoint_empty() -> Result<CompiledModel, AssemblyError> {
    tensor_empty_policy(true)
}
fn tensor_empty_policy(checkpoint: bool) -> Result<CompiledModel, AssemblyError> {
    use alea::model::tensor::{Comparison, Shape, TensorBuilder, TensorFactor, UnaryOp};
    let branch = |log: bool| -> Result<_, AssemblyError> {
        let mut b = TensorBuilder::new();
        let x = b.input("x", Shape::scalar())?;
        let value = if log { b.unary(UnaryOp::Log, x)? } else { x };
        let result = b.gather(value, Shape::new(vec![0])?, vec![])?;
        b.compile(result)?.bind(&[])
    };
    let mut t = TensorBuilder::new();
    let x = t.input("x", Shape::scalar())?;
    let zero = t.constant(Shape::scalar(), vec![0.])?;
    let chosen = t.cond(
        x,
        zero,
        Comparison::Less,
        &[x],
        branch(true)?,
        branch(false)?,
    )?;
    let out = t.sum(chosen)?;
    let out = if checkpoint {
        checkpoint_identity(&mut t, out)?
    } else {
        out
    };
    let binding = t.compile(out)?.bind(&[])?;
    let mut b = ModelBuilder::new();
    let x = b.real("x")?.expr();
    let term = b.primitive(TensorFactor::new(binding, &[x])?)?;
    b.potential("objective", term)?;
    b.compile()
}
pub fn all() -> Result<Vec<(&'static str, CompiledModel)>, AssemblyError> {
    let mut models = vec![
        ("polynomial", polynomial()?),
        ("native_normal_batch", native_batch(0)?),
        ("native_logit_batch", native_batch(1)?),
        ("native_poisson_batch", native_batch(2)?),
        ("tensor_matrix", tensor_matrix()?),
        ("tensor_checkpoint_matrix", tensor_checkpoint_matrix()?),
        ("tensor_control", tensor_control()?),
        ("tensor_checkpoint", tensor_checkpoint()?),
        ("tensor_checkpoint_zero", tensor_checkpoint_zero()?),
        ("tensor_checkpoint_long", tensor_checkpoint_long()?),
        ("tensor_retained_long", tensor_retained_long()?),
        ("tensor_checkpoint_mixed", tensor_checkpoint_mixed()?),
        ("tensor_checkpoint_activity", tensor_checkpoint_activity()?),
        ("tensor_checkpoint_constant", tensor_checkpoint_constant()?),
        ("tensor_checkpoint_routes", tensor_checkpoint_routes()?),
        ("tensor_checkpoint_vector", tensor_checkpoint_vector()?),
        (
            "tensor_checkpoint_immutable",
            tensor_checkpoint_immutable()?,
        ),
        ("tensor_empty", tensor_empty()?),
        ("tensor_checkpoint_empty", tensor_checkpoint_empty()?),
        ("tensor_zero_scan", tensor_zero_scan()?),
        ("regression", regression()?),
        ("likelihood_location_scale", likelihood_location_scale()?),
        ("uncertainty_base", uncertainty_regression(1., 1.)?),
        ("uncertainty_large", uncertainty_regression(8., 4.)?),
        ("uncertainty_small", uncertainty_regression(0.125, 0.25)?),
        ("logistic", logistic()?),
        ("selected_likelihood", statistical(false)?),
        ("selected_posterior", statistical(true)?),
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
