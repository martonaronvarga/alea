//! Descriptive fused-gradient versus analytic-tape microbenchmark, not sampler ESS.
#[path = "../models.rs"]
mod models;
use alea::{capability::PotentialHessianDerivative, target::LogDensityGradient};
use alea_enzyme_conformance as generated;
use std::{hint::black_box, time::Instant};

fn measure<T: LogDensityGradient>(target: &T, calls: usize) -> (f64, f64) {
    let mut q = [0.3, -0.2];
    let mut g = [0.0; 2];
    for _ in 0..64 {
        black_box(target.logp_grad(black_box(&q), &mut g).unwrap());
    }
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..calls {
        q[0] += 1e-7;
        checksum += black_box(target.logp_grad(black_box(&q), black_box(&mut g)).unwrap());
        checksum += black_box(g[0]);
    }
    (start.elapsed().as_secs_f64(), checksum)
}

fn compare<T: PotentialHessianDerivative>(name: &str, generated: T) {
    let analytic = models::all()
        .unwrap()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1
        .into_target()
        .unwrap();
    for round in 0..7 {
        let (a, b) = if round % 2 == 0 {
            (
                measure(&analytic, 1_000_000),
                measure(&generated, 1_000_000),
            )
        } else {
            let b = measure(&generated, 1_000_000);
            (measure(&analytic, 1_000_000), b)
        };
        assert!((a.1 - b.1).abs() < 1e-7 * (1.0 + a.1.abs()));
        println!("{name},{round},1000000,{:.9},{:.9},{:.9}", a.0, b.0, a.1);
    }
}

fn main() {
    println!("model,round,calls,analytic_tape_seconds,enzyme_seconds,checksum");
    compare("polynomial", generated::polynomial::target());
    compare("regression", generated::regression::target());
    compare("logistic", generated::logistic::target());
}
