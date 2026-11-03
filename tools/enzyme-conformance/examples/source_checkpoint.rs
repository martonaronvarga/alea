//! Fixed-work source checkpoint controls, not inference or statistical efficiency.
use alea::target::LogDensityGradient;
use alea_enzyme_conformance as generated;
use std::{hint::black_box, time::Instant};

fn measure(target: impl LogDensityGradient, q: &[f64], calls: usize) {
    let mut q = q.to_vec();
    let mut g = vec![0.; q.len()];
    for _ in 0..64 {
        black_box(target.logp_grad(black_box(&q), &mut g).unwrap());
    }
    let start = Instant::now();
    let mut checksum = vec![0.; q.len() + 1];
    for _ in 0..calls {
        q[0] += 1e-9;
        checksum[0] += black_box(target.logp_grad(black_box(&q), black_box(&mut g)).unwrap());
        for (sum, score) in checksum[1..].iter_mut().zip(&g) {
            *sum += black_box(*score);
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("calls={calls} seconds={seconds:.9} checksum={checksum:?}");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let case = args.next().expect("case");
    let calls: usize = args.next().expect("calls").parse().expect("integer calls");
    assert!(calls > 0 && args.next().is_none());
    match case.as_str() {
        "checkpoint" => measure(generated::tensor_checkpoint::target(), &[0.4, 0.53], calls),
        "long" => measure(
            generated::tensor_checkpoint_long::target(),
            &[0.4, 0.53],
            calls,
        ),
        "retained" => measure(
            generated::tensor_retained_long::target(),
            &[0.4, 0.53],
            calls,
        ),
        "mixed" => measure(
            generated::tensor_checkpoint_mixed::target(),
            &[0.4, 0.53],
            calls,
        ),
        "matrix" => measure(
            generated::tensor_checkpoint_matrix::target(),
            &[0.4, 0.53],
            calls,
        ),
        "routes" => measure(
            generated::tensor_checkpoint_routes::target(),
            &[0.2, 0.1, -0.4, 0.5, 0.7],
            calls,
        ),
        _ => panic!("unknown case"),
    }
}
