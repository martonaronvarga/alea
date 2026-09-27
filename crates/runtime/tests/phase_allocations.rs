#![cfg(not(miri))]
use kernels::{
    buffer::OwnedBuffer,
    dist::Gaussian,
    metric::IdentityMetric,
    target::{EvaluationWorkspace, FusedAdapter, PointState},
};
use runtime::hamiltonian::{PhaseState, PhaseWorkspace, SignedStep};
use std::hint::black_box;

#[test]
fn phase_restart_snapshot_clone_and_update_allocate_nothing() {
    for dim in [1, 7, 33, 129] {
        let target = FusedAdapter::new(&Gaussian, dim);
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
