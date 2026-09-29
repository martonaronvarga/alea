use alea_core::{
    density::{DensityPoint, LogDensity, evaluate},
    target::EvaluationError,
};
use alea_math::buffer::OwnedBuffer;
use std::cell::Cell;

struct Target {
    dimension: Cell<usize>,
    calls: Cell<usize>,
    mode: Cell<u8>,
}
impl LogDensity for Target {
    type Error = std::io::Error;
    fn dimension(&self) -> usize {
        self.dimension.get()
    }
    fn logp(&self, q: &[f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        match self.mode.get() {
            1 => Err(std::io::ErrorKind::Other.into()),
            2 => panic!("injected failure"),
            3 => Ok(f64::NAN),
            _ => Ok(-0.5 * q.iter().map(|x| x * x).sum::<f64>()),
        }
    }
}

#[test]
fn derivative_free_cache_validates_before_evaluation_and_updates_transactionally() {
    let target = Target {
        dimension: Cell::new(2),
        calls: Cell::new(0),
        mode: Cell::new(0),
    };
    assert!(matches!(
        evaluate(&target, &[0.0]),
        Err(EvaluationError::DensityDimension { .. })
    ));
    assert!(evaluate(&target, &[0.0, f64::NAN]).is_err());
    assert_eq!(target.calls.get(), 0);
    let mut point = DensityPoint::new(&target, OwnedBuffer::from_fn(2, |_| 1.0)).unwrap();
    let address = point.position().as_ptr();
    assert_eq!(address as usize % 64, 0);
    for mode in 1..=3 {
        target.mode.set(mode);
        if mode == 2 {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || point.try_update(&[2.0, 3.0])
                ))
                .is_err()
            );
        } else {
            assert!(point.try_update(&[2.0, 3.0]).is_err());
        }
        assert_eq!(point.position(), &[1.0, 1.0]);
        assert_eq!(point.log_density(), -1.0);
    }
    target.mode.set(0);
    target.dimension.set(3);
    let calls = target.calls.get();
    assert!(matches!(
        point.try_update(&[0.0; 3]),
        Err(EvaluationError::TargetDimensionChanged { .. })
    ));
    assert_eq!(target.calls.get(), calls);
    target.dimension.set(2);
    point.try_update(&[2.0, 3.0]).unwrap();
    assert_eq!(point.position().as_ptr(), address);
    assert_eq!(point.log_density(), -6.5);
    let mut clone = point.clone();
    let clone_address = clone.position().as_ptr();
    clone.clone_from(&point);
    assert_eq!(clone.position().as_ptr(), clone_address);
}

#[test]
fn zero_dimension_and_trait_objects_are_supported() {
    let target = Target {
        dimension: Cell::new(0),
        calls: Cell::new(0),
        mode: Cell::new(0),
    };
    let erased: &dyn LogDensity<Error = std::io::Error> = &target;
    let mut point = DensityPoint::new(erased, OwnedBuffer::new(0)).unwrap();
    point.try_update(&[]).unwrap();
    assert_eq!(point.dimension(), 0);
    assert_eq!(point.log_density(), 0.0);
}
