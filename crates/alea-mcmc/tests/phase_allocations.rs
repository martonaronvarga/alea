#![cfg(not(miri))]
use alea_core::target::{EvaluationWorkspace, PointState};
use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::IdentityMetric;
use alea_mcmc::hamiltonian::{PhaseState, PhaseWorkspace, SignedStep};
use std::hint::black_box;
#[path = "support/targets.rs"]
mod targets;

#[test]
fn phase_restart_snapshot_clone_and_update_allocate_nothing() {
    for dim in [1, 7, 33, 129] {
        let target = Gaussian::new(dim);
        let initial = PhaseState::new(
            PointState::new(&target, OwnedBuffer::new(dim)).unwrap(),
            OwnedBuffer::from_fn(dim, |_| 0.5),
        )
        .unwrap();
        let mut saved = initial.clone();
        let mut workspace = PhaseWorkspace::new(initial.point());
        let mut evaluation = EvaluationWorkspace::new(dim);
        let metric = IdentityMetric::new(dim);
        let step = SignedStep::new(0.1).unwrap();
        let count = allocation_counter::measure(|| {
            for _ in 0..16 {
                let phase = workspace
                    .start_from_phase(&initial, &metric)
                    .unwrap()
                    .step(step)
                    .unwrap();
                phase.save_into(&mut saved).unwrap();
                black_box(saved.point().position());
                saved.clone_from(&initial);
                saved
                    .try_update(
                        initial.point().position(),
                        initial.momentum(),
                        &mut evaluation,
                    )
                    .unwrap();
            }
        });
        assert_eq!(count.count_total, 0);
        assert_eq!(count.bytes_total, 0);
    }
}

fn model_steps<M: alea_math::metric::EuclideanMetric>(target: targets::Target, metric: M) {
    let initial = PhaseState::new(
        PointState::new(&target, OwnedBuffer::from_fn(2, |i| [0.3, -0.7][i])).unwrap(),
        OwnedBuffer::from_fn(2, |i| [0.2, 0.5][i]),
    )
    .unwrap();
    let mut workspace = PhaseWorkspace::new(initial.point());
    let step = SignedStep::new(0.001).unwrap();
    // No unmeasured warmup: includes the first inverse-metric/model invocation.
    let allocation = allocation_counter::measure(|| {
        let mut phase = workspace.start_from_phase(&initial, &metric).unwrap();
        for _ in 0..128 {
            phase = phase.step(step).unwrap();
        }
        black_box(phase.energy().unwrap());
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn non_gaussian_steps_allocate_nothing_for_all_metrics() {
    use alea_math::metric::{CholeskyFactor, DenseMetric, DiagonalMetric};
    for target in [
        targets::Target::Logistic,
        targets::Target::Funnel,
        targets::Target::Rotated,
    ] {
        model_steps(target, IdentityMetric::new(2));
        model_steps(
            target,
            DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| [4.0, 0.25][i])).unwrap(),
        );
        model_steps(
            target,
            DenseMetric::new(
                CholeskyFactor::new_lower(2, OwnedBuffer::from_fn(4, |i| [1.5, 0.4, 0.0, 0.8][i]))
                    .unwrap(),
            ),
        );
    }
}
