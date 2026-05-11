use kernels::dist::wiener::*;
use std::hint::black_box;

fn main() {
    // let params = Wiener7Params::with_params(1.5, 0.3, 0.55, 0.4, 0.05, 0.15, 0.1).unwrap();
    // let obs = WienerObservation {
    //     rt: 0.8,
    //     boundary: Boundary::Upper,
    // };
    // let n = 10000; // enough to profile

    // for _ in 0..n {
    //     let result = Wiener7.fused(black_box(&obs), black_box(&params), 1e-6);
    //     black_box(result);
    // }

    let testobs = WienerObservation {
        rt: 6.0,
        boundary: Boundary::Upper,
    };

    let testparams = Wiener7Params::with_params_unchecked(10.0, 0.01, 0.1, -3.0, 0.1, 0.0, 0.2);
    let fused = Wiener7.fused(&testobs, &testparams, 1e-12);

    println!("logp: {:?}\ngrad: {:?}", fused.log_prob, fused.grad);

    // let (nodes_25, weights25) = Wiener7::gauss_legendre_01(5);
    // let (nodes_15, weights15) = Wiener7::gauss_legendre_01(7);
    // println!("5 nodes, weights");
    // for (i, j) in nodes_25.iter().zip(weights25.iter()) {
    //     println!("{} {}", i, j);
    // }

    // println!("7 nodes, weights");
    // for (i, j) in nodes_15.iter().zip(weights15.iter()) {
    //     println!("{} {}", i, j);
    // }
}
