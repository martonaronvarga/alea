//! A tensor/control-flow deterministic mean inside native probability factors.
#![cfg(feature = "distributions")]
use alea::{
    distributions::modeling::normal,
    math::buffer::OwnedBuffer,
    model::{
        TransformedTarget,
        assembly::{CompiledModel, ModelBuilder},
        tensor::{BinaryOp, Comparison, Shape},
    },
    sampling::{
        NutsOptions,
        adapt::{NutsWarmup, WarmupOptions},
    },
    target::LogDensityGradient,
};
use rand::{SeedableRng, rngs::SmallRng};

fn target() -> TransformedTarget<CompiledModel> {
    let mut b = ModelBuilder::new();
    let q = b.real("q").unwrap();
    let mut tensor = b.region();
    let x = tensor.input("x", Shape::scalar()).unwrap();
    let weights = tensor
        .data("weights", Shape::new(vec![64]).unwrap())
        .unwrap();
    let product = tensor.binary(BinaryOp::Multiply, x, weights).unwrap();
    let average = tensor.sum(product).unwrap();
    let mut branch = b.region();
    let value = branch.input("value", Shape::scalar()).unwrap();
    let identity = branch.compile(value).unwrap().bind(&[]).unwrap();
    let zero = tensor.constant(Shape::scalar(), vec![0.]).unwrap();
    let selected = tensor
        .cond(
            x,
            zero,
            Comparison::Less,
            &[average],
            identity.clone(),
            identity,
        )
        .unwrap();
    let mut body = b.region();
    let carry = body.input("carry", Shape::scalar()).unwrap();
    let row = body.input("row", Shape::scalar()).unwrap();
    let add = body.binary(BinaryOp::Add, carry, row).unwrap();
    let body = body.compile(add).unwrap().bind(&[]).unwrap();
    let rows = tensor.data("rows", Shape::new(vec![3]).unwrap()).unwrap();
    let mean = tensor.scan(selected, rows, &[], body).unwrap();
    let program = tensor
        .compile(mean)
        .unwrap()
        .bind(&[("weights", &[1. / 64.; 64]), ("rows", &[0.; 3])])
        .unwrap();
    let mean = b.call_region(&[q.expr()], program).unwrap();
    let prior = normal(&mut b, q, 0., 1.).unwrap();
    b.prior("prior", prior).unwrap();
    let likelihood = normal(&mut b, 1., mean, 1.).unwrap();
    b.likelihood("observation", likelihood).unwrap();
    b.compile()
        .unwrap()
        .select_posterior()
        .unwrap()
        .into_target()
        .unwrap()
}

#[test]
fn tensor_maps_branches_and_scan_feed_native_density_scores_and_nuts() {
    let model = target();
    let mut gradient = [0.];
    for x in [-10., -0.3, 0., 0.5, 3.] {
        let value = model.logp_grad(&[x], &mut gradient).unwrap();
        let expected = -std::f64::consts::TAU.ln() - 0.5 * x * x - 0.5 * (1. - x) * (1. - x);
        assert!((value - expected).abs() < 1e-12 * (1. + expected.abs()));
        assert!((gradient[0] - (1. - 2. * x)).abs() < 1e-12);
    }
    // q|y=1 is N(1/2,1/2). A correctness workflow, not an ESS/sec benchmark.
    let mut chains = Vec::new();
    let mut squares = Vec::new();
    for seed in 720..724 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (mut sampler, report) = NutsWarmup::new(
            &model,
            OwnedBuffer::new(1),
            NutsOptions::new(0.1, 7).unwrap(),
            WarmupOptions::new(500),
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        assert_eq!(report.iterations, 500);
        let mut chain = Vec::new();
        let mut square = Vec::new();
        for _ in 0..1000 {
            let transition = sampler.step(&mut rng).unwrap();
            assert!(matches!(
                transition.termination,
                alea::sampling::nuts::NutsTermination::UTurn
            ));
            let q = sampler.point().position()[0];
            chain.push(q);
            square.push(q * q);
        }
        chains.push(chain);
        squares.push(square);
    }
    for (draws, expectation) in [(&chains, 0.5), (&squares, 0.75)] {
        let views: Vec<&[f64]> = draws.iter().map(Vec::as_slice).collect();
        let diagnostic = alea::diagnostics::modern::summarize(&views).unwrap();
        eprintln!("tensor workflow expectation={expectation}: {diagnostic:?}");
        assert!(diagnostic.rhat < 1.03, "{diagnostic:?}");
        assert!(diagnostic.bulk_ess > 100. && diagnostic.tail_ess > 100.);
        assert!(
            (diagnostic.mean - expectation).abs() < 6. * diagnostic.mcse_mean,
            "{diagnostic:?}"
        );
    }
}
