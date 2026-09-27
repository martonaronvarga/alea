use kernels::{
    density::FusedLogDensity,
    dist::{traits::Target, wiener::*},
};

fn observations(strategy: BatchStrategy, invalid: bool) -> WienerObservations {
    let mut data = Vec::new();
    for boundary in [Boundary::Upper, Boundary::Lower] {
        for rt in [0.201, 0.25, 0.5, 1.0, 4.0, 10.0] {
            data.push(WienerObservation { rt, boundary });
        }
    }
    if invalid {
        data.push(WienerObservation {
            rt: 0.1,
            boundary: Boundary::Upper,
        });
    }
    WienerObservations::from(data).with_batch_strategy(strategy)
}

#[test]
fn initialized_bucket_storage_matches_direct_wiener4_and_wiener5() {
    let p4 = Wiener4Params {
        alpha: 1.5,
        tau: 0.2,
        beta: 0.4,
        delta: 0.3,
    };
    let p5 = Wiener5Params::with_params_unchecked(1.5, 0.2, 0.4, 0.3, 0.6);
    let d4 = Target::new(Wiener4, observations(BatchStrategy::Direct, false));
    let b4 = Target::new(Wiener4, observations(BatchStrategy::BranchBuckets, false));
    let d5 = Target::new(Wiener5, observations(BatchStrategy::Direct, false));
    let b5 = Target::new(Wiener5, observations(BatchStrategy::BranchBuckets, false));
    let (mut dg4, mut bg4) = ([0.0; 4], [0.0; 4]);
    let (mut dg5, mut bg5) = ([0.0; 5], [0.0; 5]);
    let dl4 = d4.log_prob_and_grad(&p4, &mut dg4);
    let bl4 = b4.log_prob_and_grad(&p4, &mut bg4);
    let dl5 = d5.log_prob_and_grad(&p5, &mut dg5);
    let bl5 = b5.log_prob_and_grad(&p5, &mut bg5);
    for (direct, bucket) in [dl4, dl5]
        .into_iter()
        .zip([bl4, bl5])
        .chain(dg4.into_iter().zip(bg4))
        .chain(dg5.into_iter().zip(bg5))
    {
        assert!(
            (direct - bucket).abs() <= 1e-10 * direct.abs().max(1.0),
            "direct={direct}, bucket={bucket}"
        );
    }
}

#[test]
fn partial_bucket_initialization_rejects_invalid_observations() {
    let p4 = Wiener4Params {
        alpha: 1.5,
        tau: 0.2,
        beta: 0.4,
        delta: 0.3,
    };
    let p5 = Wiener5Params::with_params_unchecked(1.5, 0.2, 0.4, 0.3, 0.6);
    let b4 = Target::new(Wiener4, observations(BatchStrategy::BranchBuckets, true));
    let b5 = Target::new(Wiener5, observations(BatchStrategy::BranchBuckets, true));
    let mut g4 = [0.0; 4];
    let mut g5 = [0.0; 5];
    assert_eq!(b4.log_prob_and_grad(&p4, &mut g4), f64::NEG_INFINITY);
    assert_eq!(b5.log_prob_and_grad(&p5, &mut g5), f64::NEG_INFINITY);
    assert!(g4.iter().chain(&g5).all(|x| x.is_nan()));
}
