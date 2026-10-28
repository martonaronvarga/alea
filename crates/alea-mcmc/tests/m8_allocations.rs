#![cfg(not(miri))]

use std::hint::black_box;

use alea_distributions::StandardNormalTarget;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    MarkovChain,
    adapt::{CheesCriterion, CheesEnsembleAdapter, CheesObservation},
    approximate::{ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget},
    chees::{CheesHmc, CheesHmcOptions},
    ghmc::{DrGhmc, DrGhmcOptions, Ghmc, GhmcOptions},
    mams::{Mams, MamsOptions},
};
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn m8_retained_kernels_and_shared_update_allocate_nothing() {
    let target = StandardNormalTarget::new(3);
    let mut rng = SmallRng::seed_from_u64(8_008);
    let adapter = CheesEnsembleAdapter::new(
        3,
        0.1.try_into().unwrap(),
        0.8,
        0.1,
        2.0,
        CheesCriterion::Global,
    )
    .unwrap();
    let frozen = adapter.clone().finish().unwrap();
    let mut chees = CheesHmc::new(
        &target,
        OwnedBuffer::new(3),
        IdentityMetric::new(3),
        CheesHmcOptions::new(frozen),
    )
    .unwrap();
    let mut ghmc = Ghmc::new(
        &target,
        OwnedBuffer::new(3),
        IdentityMetric::new(3),
        GhmcOptions::new(0.1, 4, 0.8).unwrap(),
        &mut rng,
    )
    .unwrap();
    let mut dr = DrGhmc::new(
        &target,
        OwnedBuffer::new(3),
        IdentityMetric::new(3),
        DrGhmcOptions::new(0.2, 2.0, 0.8)
            .unwrap()
            .with_integration_steps(2)
            .unwrap(),
        &mut rng,
    )
    .unwrap();
    let mut mams = Mams::new(
        &target,
        OwnedBuffer::new(3),
        MamsOptions::new(0.05, 3).unwrap(),
    )
    .unwrap();
    let mut approximate = ApproximateMclmc::new(
        ApproximateTarget::acknowledge_bias(),
        &target,
        OwnedBuffer::new(3),
        ApproximateMclmcOptions::new(0.03, 2, Some(1.0)).unwrap(),
        &mut rng,
    )
    .unwrap();
    let mut adapter = adapter;
    let observations = [
        CheesObservation {
            initial: &[-1.0, 0.0, 0.5],
            proposal: &[0.2, 0.1, -0.3],
            endpoint_velocity: &[0.4, -0.2, 0.1],
            acceptance_probability: 0.8,
            trajectory_time: 0.7,
        },
        CheesObservation {
            initial: &[1.0, 0.2, -0.4],
            proposal: &[-0.1, -0.2, 0.6],
            endpoint_velocity: &[-0.3, 0.4, -0.2],
            acceptance_probability: 0.9,
            trajectory_time: 0.7,
        },
    ];

    let allocation = allocation_counter::measure(|| {
        for _ in 0..16 {
            let _ = black_box(chees.step(&mut rng).unwrap());
            let _ = black_box(ghmc.step(&mut rng).unwrap());
            let _ = black_box(dr.step(&mut rng).unwrap());
            let _ = black_box(mams.step(&mut rng).unwrap());
            let _ = black_box(approximate.step(&mut rng).unwrap());
            let _ = black_box(adapter.update(&observations).unwrap());
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}
