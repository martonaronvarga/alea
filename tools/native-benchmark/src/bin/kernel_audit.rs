//! Fixed-work controls for production kernels; no surrogate probability laws.
use alea::distributions::{
    gaussian::StandardNormalTarget,
    mixture::FiniteMixture,
    modeling::{
        self,
        regression::{
            PoissonLogRegression, RegressionData, RegressionExecution, poisson_log_regression,
        },
    },
    scalar::{Normal, NormalInverseGaussian, VonMises},
};
use alea::{
    distribution::{Distribution, ParameterScore, ScalarEventScore},
    math::{
        buffer::OwnedBuffer,
        metric::{DiagonalMetric, EuclideanMetric},
        numeric::scaled_norm,
    },
    model::assembly::ModelBuilder,
    target::LogDensityGradient,
};
use std::{error::Error, hint::black_box, time::Instant};

fn normal_factors(
    b: &mut ModelBuilder,
    size: usize,
    vector: bool,
) -> Result<(), alea::model::assembly::AssemblyError> {
    let location = b.constant(0.2)?;
    let scale = b.constant(1.3)?;
    let mut events = Vec::with_capacity(if vector { size } else { 0 });
    for i in 0..size {
        let q = b.real(&format!("q{i}"))?.expr();
        if vector {
            events.push(q);
        } else {
            let factor = modeling::normal(b, q, location, scale)?;
            b.prior(&format!("p{i}"), factor)?;
        }
    }
    if vector {
        let factor = modeling::normal_batch(b, &events, &[location], &[scale])?;
        b.prior("batch", factor)?;
    }
    Ok(())
}

fn target_work<T: LogDensityGradient>(
    target: T,
    size: usize,
    calls: usize,
) -> Result<(f64, f64), Box<dyn Error>> {
    let mut q = vec![0.01; size];
    let mut g = vec![0.; size];
    target.logp_grad(&q, &mut g)?;
    let start = Instant::now();
    let mut checksum = 0.;
    for i in 0..calls {
        if let Some(x) = q.first_mut() {
            *x = (i % 10) as f64 * 0.01;
        }
        checksum += black_box(target.logp_grad(black_box(&q), black_box(&mut g))?);
        checksum += black_box(g.iter().copied().sum::<f64>());
    }
    Ok((start.elapsed().as_secs_f64(), checksum))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: kernel_audit CASE SIZE CALLS".into());
    }
    let case = &args[1];
    let size: usize = args[2].parse()?;
    let calls: usize = args[3].parse()?;
    if size == 0 || calls == 0 {
        return Err("positive sizes and calls required".into());
    }
    let construction = Instant::now();
    let construction_seconds;
    let (seconds, checksum) = match case.as_str() {
        "gemm" | "gemm_faer" | "gemm_blas" => {
            use alea::model::tensor::{MatrixKernel, Shape};
            let requested = std::env::var("ALEA_MATRIX_KERNEL").unwrap_or_else(|_| case.clone());
            let kernel = match requested.as_str() {
                "gemm" | "compensated" => MatrixKernel::Compensated,
                #[cfg(feature = "faer")]
                "gemm_faer" | "faer" => MatrixKernel::Faer,
                #[cfg(feature = "openblas")]
                "gemm_blas" | "blas" => MatrixKernel::Blas,
                _ => return Err("requested matrix provider was not compiled".into()),
            };
            let count = size.checked_mul(size).ok_or("matrix extent overflow")?;
            let mut b = ModelBuilder::new().region();
            let a = b.input("a", Shape::new(vec![size, size])?)?;
            let r = b.input("r", Shape::new(vec![size, size])?)?;
            let result = b.matmul_with_kernel(a, r, kernel)?;
            let program = b.compile(result)?.bind(&[])?;
            let mut w = program.workspace();
            let x: Vec<_> = (0..2 * count)
                .map(|i| ((i % 31) as f64 - 15.) / 32.)
                .collect();
            let seed = vec![0.5; count];
            let mut out = vec![0.; count];
            let mut g = vec![0.; 2 * count];
            program.value_and_vjp(&x, &seed, &mut out, &mut g, &mut w)?;
            // Full-coordinate admission against the existing compensated IR,
            // outside timing. These dyadic controls are not an accuracy theorem.
            let mut reference = ModelBuilder::new().region();
            let a = reference.input("a", Shape::new(vec![size, size])?)?;
            let r = reference.input("r", Shape::new(vec![size, size])?)?;
            let result = reference.matmul(a, r)?;
            let reference = reference.compile(result)?.bind(&[])?;
            let mut values = vec![0.; count];
            let mut scores = vec![0.; 2 * count];
            reference.value_and_vjp(
                &x,
                &seed,
                &mut values,
                &mut scores,
                &mut reference.workspace(),
            )?;
            if out
                .iter()
                .chain(&g)
                .zip(values.iter().chain(&scores))
                .any(|(a, b)| (a - b).abs() > 1e-10 * (1. + b.abs()))
            {
                return Err("matrix value or complete VJP differs from reference".into());
            }
            construction_seconds = construction.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut checksum = 0.;
            for _ in 0..calls {
                program.value_and_vjp(black_box(&x), &seed, &mut out, &mut g, &mut w)?;
                checksum += black_box(out.iter().sum::<f64>()) + black_box(g.iter().sum::<f64>());
            }
            (start.elapsed().as_secs_f64(), checksum)
        }
        "tensor_vector" | "tensor_reuse" => {
            use alea::model::tensor::{BinaryOp, Shape, UnaryOp};
            let mut b = ModelBuilder::new().region();
            let input = b.input("input", Shape::new(vec![size])?)?;
            let mut value = input;
            for _ in 0..32 {
                value = b.unary(UnaryOp::Tanh, value)?;
                value = if case == "tensor_reuse" {
                    b.reshape(value, Shape::new(vec![size])?)?
                } else {
                    b.binary(BinaryOp::Add, value, input)?
                };
            }
            let result = b.sum(value)?;
            let program = b.compile(result)?.bind(&[])?;
            let mut workspace = program.workspace();
            let mut x: Vec<_> = (0..size).map(|i| 0.3 + (i % 17) as f64 / 31.).collect();
            let mut out = [0.];
            let mut g = vec![0.; size];
            program.value_and_vjp(&x, &[0.7], &mut out, &mut g, &mut workspace)?;
            construction_seconds = construction.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut checksum = 0.;
            for i in 0..calls {
                x[0] = 0.2 + (i % 10) as f64 / 31.;
                program.value_and_vjp(
                    black_box(&x),
                    black_box(&[0.7]),
                    black_box(&mut out),
                    black_box(&mut g),
                    &mut workspace,
                )?;
                checksum += black_box(out[0]) + black_box(g.iter().sum::<f64>());
            }
            (start.elapsed().as_secs_f64(), checksum)
        }
        "poisson"
        | "poisson_scalar"
        | "poisson_common"
        | "normal_factors"
        | "mixture_factors"
        | "tensor"
        | "tensor_dag"
        | "tensor_scalar"
        | "tensor_matmul"
        | "tensor_scan"
        | "tensor_scan_checkpoint"
        | "tensor_cond" => {
            let mut b = ModelBuilder::new();
            let dimension;
            if case.starts_with("tensor") {
                use alea::model::tensor::{BinaryOp, Comparison, Shape};
                if case == "tensor_scalar" {
                    dimension = 1;
                    let slope = b.real("slope")?.expr();
                    let terms: Vec<_> = (0..size)
                        .map(|i| {
                            let p = b.multiply(slope, (i % 17) as f64 / 16.)?;
                            b.multiply(p, p)
                        })
                        .collect::<Result<_, _>>()?;
                    let factor = b.sum(&terms)?;
                    b.potential("scalar", factor)?;
                } else if case == "tensor_scan" || case == "tensor_scan_checkpoint" {
                    dimension = 2;
                    let inputs = [b.real("initial")?.expr(), b.real("gain")?.expr()];
                    let mut body = b.region();
                    let carry = body.input("carry", Shape::scalar())?;
                    let row = body.input("row", Shape::scalar())?;
                    let gain = body.input("gain", Shape::scalar())?;
                    let scaled = body.binary(BinaryOp::Multiply, carry, gain)?;
                    let next = body.binary(BinaryOp::Add, scaled, row)?;
                    let body = body.compile(next)?.bind(&[])?;
                    let mut tensor = b.region();
                    let initial = tensor.input("initial", Shape::scalar())?;
                    let gain = tensor.input("gain", Shape::scalar())?;
                    let rows = tensor.data("rows", Shape::new(vec![size])?)?;
                    let end = if case == "tensor_scan_checkpoint" {
                        tensor.checkpointed_scan(initial, rows, &[gain], body)?
                    } else {
                        tensor.scan(initial, rows, &[gain], body)?
                    };
                    let values: Vec<_> = (0..size).map(|i| (i % 17) as f64 / 16.).collect();
                    let factor =
                        b.apply_region(&inputs, tensor.compile(end)?.bind(&[("rows", &values)])?)?;
                    b.potential("scan", factor)?;
                } else if case == "tensor_matmul" {
                    dimension = 8;
                    let inputs: Vec<_> = (0..dimension)
                        .map(|j| b.real(&format!("q{j}")).map(|v| v.expr()))
                        .collect::<Result<_, _>>()?;
                    let mut tensor = b.region();
                    let coefficients =
                        tensor.input("coefficients", Shape::new(vec![dimension, 1])?)?;
                    let design = tensor.data("design", Shape::new(vec![size, dimension])?)?;
                    let product = tensor.matmul(design, coefficients)?;
                    let square = tensor.binary(BinaryOp::Multiply, product, product)?;
                    let sum = tensor.sum(square)?;
                    let values: Vec<_> = (0..size * dimension)
                        .map(|i| (i % 17) as f64 / 16.)
                        .collect();
                    let factor = b.apply_region(
                        &inputs,
                        tensor.compile(sum)?.bind(&[("design", &values)])?,
                    )?;
                    b.potential("matrix", factor)?;
                } else if case == "tensor_cond" {
                    dimension = 1;
                    let input = b.real("slope")?.expr();
                    let mut branch = b.region();
                    let slope = branch.input("slope", Shape::scalar())?;
                    let weights = branch.data("weights", Shape::new(vec![size])?)?;
                    let product = branch.binary(BinaryOp::Multiply, slope, weights)?;
                    let square = branch.binary(BinaryOp::Multiply, product, product)?;
                    let values: Vec<_> = (0..size).map(|i| (i % 17) as f64 / 16.).collect();
                    let branch = branch.compile(square)?.bind(&[("weights", &values)])?;
                    let mut tensor = b.region();
                    let slope = tensor.input("slope", Shape::scalar())?;
                    let threshold = tensor.constant(Shape::scalar(), vec![0.045])?;
                    let selected = tensor.cond(
                        slope,
                        threshold,
                        Comparison::Less,
                        &[slope],
                        branch.clone(),
                        branch,
                    )?;
                    let sum = tensor.sum(selected)?;
                    let factor = b.apply_region(&[input], tensor.compile(sum)?.bind(&[])?)?;
                    b.potential("branch", factor)?;
                } else {
                    dimension = 1;
                    let slope = b.real("slope")?.expr();
                    let mut tensor = b.region();
                    let input = tensor.input("input", Shape::scalar())?;
                    let weights = tensor.data("weights", Shape::new(vec![size])?)?;
                    let product = tensor.binary(BinaryOp::Multiply, input, weights)?;
                    let mut square = tensor.binary(BinaryOp::Multiply, product, product)?;
                    if case == "tensor_dag" {
                        // Shared operands exercise a DAG, not merely a unary chain.
                        // Both binaries evaluate exactly this same public-API graph.
                        for _ in 0..8 {
                            let shifted = tensor.binary(BinaryOp::Add, square, product)?;
                            square = tensor.binary(BinaryOp::Multiply, shifted, weights)?;
                        }
                    }
                    let sum = tensor.sum(square)?;
                    let values: Vec<_> = (0..size).map(|i| (i % 17) as f64 / 16.).collect();
                    let program = tensor.compile(sum)?.bind(&[("weights", &values)])?;
                    let factor = b.apply_region(&[slope], program)?;
                    b.potential("tensor", factor)?;
                }
            } else if case.starts_with("poisson") {
                dimension = 8;
                let beta: Vec<_> = (0..dimension)
                    .map(|j| b.real(&format!("b{j}")).map(|x| x.expr()))
                    .collect::<Result<_, _>>()?;
                let entries = size
                    .checked_mul(dimension)
                    .ok_or("design dimension overflow")?;
                let x = (0..entries)
                    .map(|i| {
                        if i % dimension == 0 {
                            1.
                        } else {
                            let entry = if case == "poisson_common" {
                                i % dimension
                            } else {
                                i
                            };
                            (entry as f64 / 13.).sin()
                        }
                    })
                    .collect();
                let data =
                    RegressionData::new(dimension, x, (0..size).map(|i| (i % 5) as u64).collect())?;
                let factor = if case == "poisson_scalar" {
                    b.primitive(
                        &beta,
                        PoissonLogRegression::new(data)?
                            .with_execution(RegressionExecution::Scalar),
                    )?
                } else {
                    poisson_log_regression(&mut b, &beta, data)?
                };
                b.likelihood("y", factor)?;
            } else if case == "mixture_factors" {
                dimension = size.checked_add(1).ok_or("mixture dimension overflow")?;
                let event = b.real("event")?.expr();
                let weight = b.constant(1. / size as f64)?;
                let scale = b.constant(1.3)?;
                let mut parameters = vec![weight; size];
                for i in 0..size {
                    parameters.push(b.real(&format!("mean{i}"))?.expr());
                    parameters.push(scale);
                }
                let factor = modeling::continuous(&mut b, event, &parameters, move |x| {
                    let components = x[size..]
                        .chunks_exact(2)
                        .map(|p| Normal::new(p[0], p[1]))
                        .collect::<Result<Vec<_>, _>>()?;
                    FiniteMixture::new(x[..size].to_vec(), components)
                })?;
                b.likelihood("mixture", factor)?;
            } else {
                dimension = size;
                let vector = std::env::var("ALEA_NORMAL_BATCH").is_ok_and(|v| v == "1");
                normal_factors(&mut b, size, vector)?;
            }
            let target = b.compile()?.into_target()?;
            construction_seconds = construction.elapsed().as_secs_f64();
            target_work(target, dimension, calls)?
        }
        "standard_normal" => {
            let target = StandardNormalTarget::new(size);
            construction_seconds = construction.elapsed().as_secs_f64();
            target_work(target, size, calls)?
        }
        "norm" | "momentum" | "velocity" | "log_det" => {
            let mut values: Vec<_> = (0..size).map(|i| (i as f64 / 17.).sin()).collect();
            let masses = OwnedBuffer::from_fn(size, |i| 0.3 + (i % 11) as f64 / 7.);
            let metric = DiagonalMetric::new(masses)?;
            let mut out = vec![0.; size];
            construction_seconds = construction.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut checksum = 0.;
            for i in 0..calls {
                values[0] = (i % 10) as f64 * 0.01;
                match case.as_str() {
                    "norm" => checksum += black_box(scaled_norm(black_box(&values))),
                    "log_det" => checksum += black_box(black_box(&metric).log_det()),
                    "momentum" => {
                        black_box(&metric)
                            .sample_momentum(black_box(&values), black_box(&mut out))?;
                        checksum += black_box(out[i % size]);
                    }
                    _ => {
                        black_box(&metric).velocity(black_box(&values), black_box(&mut out))?;
                        checksum += black_box(out[i % size]);
                    }
                }
            }
            (start.elapsed().as_secs_f64(), checksum)
        }
        "mixture" | "mixture_event_score" | "mixture_parameter_score" => {
            let components = (0..size)
                .map(|i| Normal::new(i as f64 / 3., 0.7 + (i % 3) as f64))
                .collect::<Result<Vec<_>, _>>()?;
            let law = FiniteMixture::new(vec![1. / size as f64; size], components)?;
            let mut scores = vec![0.; law.parameter_count()];
            construction_seconds = construction.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut checksum = 0.;
            for i in 0..calls {
                let x = black_box((i % 100) as f64 / 37.);
                checksum += match case.as_str() {
                    "mixture_event_score" => black_box(&law).event_score(x)?,
                    "mixture_parameter_score" => {
                        black_box(&law).parameter_score(&x, black_box(&mut scores))?;
                        black_box(scores.iter().copied().sum::<f64>())
                    }
                    _ => black_box(&law).log_prob(&x)?,
                };
            }
            (start.elapsed().as_secs_f64(), checksum)
        }
        "nig_density" | "von_mises_density" => {
            let nig = NormalInverseGaussian::new(2., 0.3, 0.2, 1.1)?;
            let vm = VonMises::new(0.2, 1.3)?;
            construction_seconds = construction.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut checksum = 0.;
            for i in 0..calls {
                let x = black_box((i % 100) as f64 / 37.);
                checksum += if case == "nig_density" {
                    black_box(&nig).log_prob(&x)?
                } else {
                    black_box(&vm).log_prob(&x)?
                };
            }
            (start.elapsed().as_secs_f64(), checksum)
        }
        _ => return Err("unknown kernel case".into()),
    };
    println!(
        "{}",
        serde_json::json!({"case":case,"size":size,"calls":calls,"construction_seconds":construction_seconds,"seconds":seconds,"checksum":checksum})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normal_representation_controls_preserve_exact_work_receipts_without_timing() {
        for n in [1, 3, 4, 5, 128, 1025] {
            let targets = [false, true].map(|vector| {
                let mut b = ModelBuilder::new();
                normal_factors(&mut b, n, vector).unwrap();
                b.compile().unwrap().into_target().unwrap()
            });
            let mut a = vec![0.; n];
            let mut b = a.clone();
            let mut q = vec![0.01; n];
            for i in 0..10 {
                q[0] = i as f64 * 0.01;
                let va = targets[0].logp_grad(&q, &mut a).unwrap();
                let vb = targets[1].logp_grad(&q, &mut b).unwrap();
                assert_eq!(va.to_bits(), vb.to_bits());
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
