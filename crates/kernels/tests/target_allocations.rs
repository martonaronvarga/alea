//! Native, thread-local allocation regression; the counter is a dev dependency
//! used only by this test binary, never the library's production allocator.
#![cfg(not(miri))]

use std::{cell::Cell, hint::black_box};

use kernels::{
    buffer::OwnedBuffer,
    density::FusedLogDensity,
    dist::Gaussian,
    target::{EvaluationWorkspace, LogDensityGradient, PointState},
};

#[derive(Debug, thiserror::Error)]
#[error("outside model support")]
struct OutsideSupport;

struct Target(Cell<bool>);

impl LogDensityGradient for Target {
    type Error = OutsideSupport;

    fn dimension(&self) -> usize {
        4
    }

    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        if self.0.get() {
            return Err(OutsideSupport);
        }
        Ok(Gaussian.log_prob_and_grad(q, gradient))
    }
}

#[test]
fn point_updates_allocate_nothing_after_construction() {
    // Confirm the counter is observing real allocations in this binary.
    let probe = allocation_counter::measure(|| {
        black_box(Box::new(42));
    });
    assert!(probe.count_total >= 1);

    let target = Target(Cell::new(false));
    let mut point = PointState::new(&target, OwnedBuffer::new(4)).unwrap();
    let mut workspace = EvaluationWorkspace::new(4);
    let success = allocation_counter::measure(|| {
        for _ in 0..100 {
            point
                .try_update(black_box(&[1.0, 2.0, 3.0, 4.0]), &mut workspace)
                .unwrap();
            black_box(point.gradient());
        }
    });
    assert_eq!(success.count_total, 0);
    assert_eq!(success.bytes_total, 0);

    target.0.set(true);
    let failure = allocation_counter::measure(|| {
        for _ in 0..100 {
            assert!(
                point
                    .try_update(black_box(&[0.0; 4]), &mut workspace)
                    .is_err()
            );
            assert!(
                point
                    .try_update(black_box(&[0.0; 3]), &mut workspace)
                    .is_err()
            );
            assert!(
                point
                    .try_update(black_box(&[f64::NAN; 4]), &mut workspace)
                    .is_err()
            );
            black_box(point.position());
        }
    });
    assert_eq!(failure.count_total, 0);
    assert_eq!(failure.bytes_total, 0);
    assert_eq!(point.position(), &[1.0, 2.0, 3.0, 4.0]);
}
