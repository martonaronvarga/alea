use alea_core::{
    capability::{PotentialHessianVector, evaluate_hvp},
    model::{
        assembly::{AssemblyError as E, ModelBuilder},
        tensor::{
            BinaryOp as B, BlockRules, BlockType, Comparison, ElementType, MatrixRequest, Shape,
            UnaryOp as U,
        },
    },
    target::LogDensityGradient,
};

fn shape(axes: &[usize]) -> Shape {
    Shape::new(axes.to_vec()).unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn native_potential_hvp_matches_every_elementary_second_derivative() {
    let q = 0.37_f64;
    let v = -0.23;
    let y = q.tanh();
    for (op, second) in [
        (U::Neg, 0.),
        (U::Exp, q.exp()),
        (U::Log, -1. / q.powi(2)),
        (U::Log1p, -1. / (1. + q).powi(2)),
        (U::Expm1, q.exp()),
        (U::Tanh, -2. * y * (1. - y.powi(2))),
        (U::Sin, -q.sin()),
        (U::Cos, -q.cos()),
        (U::Sqrt, -0.25 / q.powf(1.5)),
    ] {
        let mut b = ModelBuilder::new().region();
        let input = b.input("x", Shape::scalar()).unwrap();
        let u = b.unary(op, input).unwrap();
        let p = b.compile(u).unwrap().bind(&[]).unwrap();
        let target = p.potential_hvp_target().unwrap();
        let mut out = [f64::NAN];
        evaluate_hvp(&target, &[q], &[v], &mut out).unwrap();
        near(out[0], -v * second, 2e-15);
        let (mut plus, mut minus) = ([0.], [0.]);
        target.logp_grad(&[q + 1e-5 * v], &mut plus).unwrap();
        target.logp_grad(&[q - 1e-5 * v], &mut minus).unwrap();
        near(out[0], (minus[0] - plus[0]) / 2e-5, 2e-9);
    }
}

#[test]
fn curvature_uses_maps_bilinear_matrix_rules_and_the_resolved_provider() {
    for kernel in [
        #[cfg(any(feature = "faer", feature = "openblas"))]
        MatrixRequest::standard(alea_core::model::tensor::MatrixProvider::Auto),
        MatrixRequest::default(),
        #[cfg(feature = "faer")]
        MatrixRequest::standard(alea_core::model::tensor::MatrixProvider::Faer),
        #[cfg(feature = "openblas")]
        MatrixRequest::standard(alea_core::model::tensor::MatrixProvider::Blas),
    ] {
        let mut b = ModelBuilder::new().region();
        let a = b.input("a", shape(&[2, 3])).unwrap();
        let r = b.input("r", shape(&[3, 2])).unwrap();
        let d = b.input("d", Shape::scalar()).unwrap();
        let product = b.matmul_with_request(a, r, kernel).unwrap();
        let indices = [0, 3, 1, 0];
        let map = b.gather(product, shape(&[2, 2]), indices.to_vec()).unwrap();
        let map = b.reshape(map, shape(&[4])).unwrap();
        let square = b.binary(B::Multiply, map, map).unwrap();
        let divided = b.binary(B::Divide, square, d).unwrap();
        let total = b.sum(divided).unwrap();
        let p = b.compile(total).unwrap().bind(&[]).unwrap();
        let target = p.potential_hvp_target().unwrap();
        let x = [
            0.3, -0.5, 0.7, 0.2, 0.9, -0.4, 0.1, 0.3, -0.5, 0.9, 0.2, 0.6, 1.3,
        ];
        let v = [
            0.4, 0.1, 0.2, -0.2, 0.05, 0.6, 0.9, -0.3, 0.2, 0.7, 0.15, 0.3, -0.35,
        ];
        let mut out = [0.; 13];
        evaluate_hvp(&target, &x, &v, &mut out).unwrap();

        // For each gathered scalar m=(A R)[i,j], explicitly form
        // H(m²/d) from its dense gradient and constant mixed partials.
        let d = x[12];
        let mut hessian = [[0.; 13]; 13];
        for index in indices {
            let (i, j) = (index / 2, index % 2);
            let mut gm = [0.; 13];
            let mut m = 0.;
            for k in 0..3 {
                let (a, r) = (i * 3 + k, 6 + k * 2 + j);
                m += x[a] * x[r];
                gm[a] = x[r];
                gm[r] = x[a];
            }
            for k in 0..3 {
                let (a, r) = (i * 3 + k, 6 + k * 2 + j);
                hessian[a][r] += 2. * m / d;
                hessian[r][a] += 2. * m / d;
            }
            for i in 0..12 {
                for j in 0..12 {
                    hessian[i][j] += 2. * gm[i] * gm[j] / d;
                }
                hessian[i][12] -= 2. * m * gm[i] / d.powi(2);
                hessian[12][i] -= 2. * m * gm[i] / d.powi(2);
            }
            hessian[12][12] += 2. * m.powi(2) / d.powi(3);
        }
        let (mut gp, mut gm) = ([0.; 13], [0.; 13]);
        let plus = std::array::from_fn::<_, 13, _>(|i| x[i] + 1e-5 * v[i]);
        let minus = std::array::from_fn::<_, 13, _>(|i| x[i] - 1e-5 * v[i]);
        target.logp_grad(&plus, &mut gp).unwrap();
        target.logp_grad(&minus, &mut gm).unwrap();
        for (i, row) in hessian.iter().enumerate() {
            near(
                out[i],
                -row.iter().zip(v).map(|(h, v)| h * v).sum::<f64>(),
                2e-14,
            );
            near(out[i], (gm[i] - gp[i]) / 2e-5, 2e-9);
        }
    }
}

#[test]
fn curvature_failures_are_transactional_and_admission_runs_even_at_zero_direction() {
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let sqrt = b.unary(U::Sqrt, x).unwrap();
    let p = b.compile(sqrt).unwrap().bind(&[]).unwrap();
    let target = p.potential_hvp_target().unwrap();
    let mut output = [19.];
    assert!(matches!(
        target.logp_grad(&[0.], &mut output),
        Err(E::Numerical)
    ));
    assert!(matches!(
        target.potential_hvp(&[0.], &[0.], &mut output),
        Err(E::Numerical)
    ));
    assert_eq!(output, [19.]);
    assert!(
        target
            .logp_grad(&[1e-300], &mut output)
            .unwrap()
            .is_finite()
    );
    let before = output;
    assert!(matches!(
        target.potential_hvp(&[1e-300], &[1.], &mut output),
        Err(E::Numerical)
    ));
    assert_eq!(output, before);
    assert!(matches!(
        target.potential_hvp(&[4.], &[], &mut output),
        Err(E::Shape)
    ));
    assert_eq!(output, before);
    target.potential_hvp(&[4.], &[2.], &mut output).unwrap();
    near(output[0], 1. / 16., 1e-15);

    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let log = b.unary(U::Log, x).unwrap();
    let p = b.compile(log).unwrap().bind(&[]).unwrap();
    let target = p.potential_hvp_target().unwrap();
    target
        .potential_hvp(&[1e-200], &[1e-200], &mut output)
        .unwrap();
    near(output[0], 1e200, 1e-15);
}

#[test]
fn registered_first_order_rules_do_not_supply_second_rules() {
    let mut body = ModelBuilder::new().region();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let _row = body.input("row", Shape::scalar()).unwrap();
    let out = body
        .register_block_with_jvp(
            &[carry],
            (),
            BlockRules {
                inputs: vec![BlockType {
                    dtype: ElementType::Real,
                    shape: Shape::scalar(),
                }],
                output: Shape::scalar(),
                workspace: |_| (),
                heap_bytes: 0,
                preparation: None,
                forward: |_, inputs, out, _| {
                    out[0] = inputs.real(0)?[0];
                    Ok(())
                },
                vjp: Some(|_, _, _, seed, scores, _| {
                    scores[0] = seed[0];
                    Ok(())
                }),
            },
            |_, _, tangent, _, out, _| {
                out[0] = tangent.real(0)?.map_or(0., |d| d[0]);
                Ok(())
            },
        )
        .unwrap();
    let body = body.compile(out).unwrap().bind(&[]).unwrap();
    assert!(body.jvp(&[true, true]).is_ok());
    let unsupported = body.potential_hvp_target().err().unwrap();
    assert!(
        unsupported
            .to_string()
            .contains("no native second derivative rule")
    );
    // The same opaque body is irrelevant to a zero-trip scan.
    let mut b = ModelBuilder::new().region();
    let init = b.input("init", Shape::scalar()).unwrap();
    let rows = b.constant(shape(&[0]), vec![]).unwrap();
    let scan = b.scan(init, rows, &[], body).unwrap();
    let p = b.compile(scan).unwrap().bind(&[]).unwrap();
    let mut out = [19.];
    evaluate_hvp(&p.potential_hvp_target().unwrap(), &[2.], &[3.], &mut out).unwrap();
    assert_eq!(out, [0.]);
}

#[test]
fn predicate_only_coefficients_do_not_participate_in_pathwise_curvature() {
    let branch = |op| {
        let mut b = ModelBuilder::new().region();
        let argument = b.input("argument", Shape::scalar()).unwrap();
        let out = b.binary(op, argument, argument).unwrap();
        b.compile(out).unwrap().bind(&[]).unwrap()
    };
    let mut b = ModelBuilder::new().region();
    let x = b.input("x", Shape::scalar()).unwrap();
    let y = b.input("y", Shape::scalar()).unwrap();
    let predicate = b.unary(U::Sqrt, x).unwrap();
    let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
    let out = b
        .cond(
            predicate,
            zero,
            Comparison::LessEqual,
            &[y],
            branch(B::Multiply),
            branch(B::Add),
        )
        .unwrap();
    let p = b.compile(out).unwrap().bind(&[]).unwrap();
    let target = p.potential_hvp_target().unwrap();
    let mut result = [19.; 2];
    target
        .potential_hvp(&[0., 2.], &[1., 3.], &mut result)
        .unwrap();
    assert_eq!(result, [0., -6.]);
    assert!(matches!(
        target.potential_hvp(&[-1., 2.], &[0., 0.], &mut result),
        Err(E::Numerical)
    ));
    assert_eq!(result, [0., -6.]);
}
