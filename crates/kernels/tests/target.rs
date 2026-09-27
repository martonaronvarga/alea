use std::{cell::Cell, error::Error};

use kernels::{
    buffer::OwnedBuffer,
    density::{FusedLogDensity, GradLogDensity, LogDensity},
    dist::Gaussian,
    target::{
        EvaluationError, EvaluationWorkspace, FusedAdapter, LogDensityGradient, PointState,
        evaluate,
    },
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[derive(Debug, thiserror::Error)]
#[error("injected model-domain failure")]
struct ModelFailure;

#[derive(Debug, Clone, Copy)]
enum Mode {
    Good,
    Error,
    Panic,
    Partial,
    Log(f64),
    Gradient(f64),
}

struct Target {
    dimension: Cell<usize>,
    calls: Cell<usize>,
    mode: Cell<Mode>,
}

impl Target {
    fn new(dimension: usize) -> Self {
        Self {
            dimension: Cell::new(dimension),
            calls: Cell::new(0),
            mode: Cell::new(Mode::Good),
        }
    }
}

impl LogDensityGradient for Target {
    type Error = ModelFailure;

    fn dimension(&self) -> usize {
        self.dimension.get()
    }

    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        // Dirty some scratch even on failure to test that it is never committed.
        if let Some(first) = gradient.first_mut() {
            *first = 999.0;
        }
        match self.mode.get() {
            Mode::Error => return Err(ModelFailure),
            Mode::Panic => panic!("injected evaluation panic"),
            Mode::Partial => return Ok(-1.0),
            Mode::Good | Mode::Log(_) | Mode::Gradient(_) => {}
        }
        let value = Gaussian.log_prob_and_grad(q, gradient);
        match self.mode.get() {
            Mode::Log(value) => Ok(value),
            Mode::Gradient(value) => {
                gradient[1] = value;
                Ok(-1.0)
            }
            _ => Ok(value),
        }
    }
}

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}

#[test]
fn wrong_shapes_do_not_invoke_target_or_modify_output() {
    for n in [0, 1, 2, 7, 33] {
        let target = Target::new(n);
        for position_len in [0, 1, n, n + 1] {
            for gradient_len in [0, 1, n, n + 1] {
                if position_len == n && gradient_len == n {
                    continue;
                }
                let position = vec![0.0; position_len];
                let mut gradient = vec![123.0; gradient_len];
                match evaluate(&target, &position, &mut gradient) {
                    Err(EvaluationError::Dimension(error)) => {
                        assert_eq!(
                            (error.expected, error.position, error.gradient),
                            (n, position_len, gradient_len)
                        );
                    }
                    other => panic!("expected dimension error, got {other:?}"),
                }
                assert_eq!(target.calls.get(), 0);
                assert_eq!(gradient, vec![123.0; gradient_len]);
            }
        }
    }
}

#[test]
fn nonfinite_positions_do_not_invoke_target_or_modify_output() {
    let target = Target::new(3);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..3 {
            let mut position = [1.0; 3];
            position[index] = value;
            let mut gradient = [123.0; 3];
            assert!(matches!(evaluate(&target, &position, &mut gradient),
                Err(EvaluationError::NonFinitePosition { index: i }) if i == index));
            assert_eq!(gradient, [123.0; 3]);
        }
    }
    assert_eq!(target.calls.get(), 0);
}

#[test]
fn model_errors_preserve_concrete_source() {
    let target = Target::new(3);
    target.mode.set(Mode::Error);
    let error = evaluate(&target, &[1.0; 3], &mut [0.0; 3]).unwrap_err();
    assert!(matches!(error, EvaluationError::Model(ModelFailure)));
    assert!(
        error
            .source()
            .unwrap()
            .downcast_ref::<ModelFailure>()
            .is_some()
    );
    assert_eq!(target.calls.get(), 1);
}

#[test]
fn nonfinite_or_incomplete_outputs_are_rejected() {
    let target = Target::new(3);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        target.mode.set(Mode::Log(value));
        assert!(matches!(
            evaluate(&target, &[1.0; 3], &mut [0.0; 3]),
            Err(EvaluationError::NonFiniteLogDensity)
        ));
        target.mode.set(Mode::Gradient(value));
        assert!(matches!(
            evaluate(&target, &[1.0; 3], &mut [0.0; 3]),
            Err(EvaluationError::NonFiniteGradient { index: 1 })
        ));
    }
    let mut gradient = [0.0; 3];
    target.mode.set(Mode::Good);
    evaluate(&target, &[1.0; 3], &mut gradient).unwrap();
    target.mode.set(Mode::Partial);
    assert!(matches!(
        evaluate(&target, &[1.0; 3], &mut gradient),
        Err(EvaluationError::NonFiniteGradient { index: 1 })
    ));
    assert!(gradient[1].is_nan());
}

#[test]
fn construction_rejects_bad_shape_and_invalid_initial_cache() {
    let target = Target::new(3);
    assert!(matches!(
        PointState::new(&target, buffer(&[0.0])),
        Err(EvaluationError::Dimension(_))
    ));
    assert_eq!(target.calls.get(), 0);
    assert!(matches!(
        PointState::new(&target, buffer(&[0.0, f64::NAN, 0.0])),
        Err(EvaluationError::NonFinitePosition { index: 1 })
    ));
    assert_eq!(target.calls.get(), 0);
    target.mode.set(Mode::Error);
    assert!(matches!(
        PointState::new(&target, buffer(&[0.0; 3])),
        Err(EvaluationError::Model(_))
    ));
    target.mode.set(Mode::Partial);
    assert!(matches!(
        PointState::new(&target, buffer(&[0.0; 3])),
        Err(EvaluationError::NonFiniteGradient { index: 1 })
    ));
    target.mode.set(Mode::Log(f64::NEG_INFINITY));
    assert!(matches!(
        PointState::new(&target, buffer(&[0.0; 3])),
        Err(EvaluationError::NonFiniteLogDensity)
    ));
}

#[test]
fn point_update_commits_once_and_reuses_aligned_storage() {
    let target = Target::new(3);
    let position = buffer(&[1.0, 2.0, 3.0]);
    let original_position = position.as_ptr();
    let mut point = PointState::new(&target, position).unwrap();
    assert_eq!(target.calls.get(), 1);
    let mut workspace = EvaluationWorkspace::new(3);
    assert_eq!(workspace.dimension(), 3);
    let first_gradient = point.gradient().as_ptr();
    point.try_update(&[0.0; 3], &mut workspace).unwrap();
    let second_gradient = point.gradient().as_ptr();
    assert_ne!(first_gradient, second_gradient);
    assert_eq!(first_gradient as usize % 64, 0);
    assert_eq!(second_gradient as usize % 64, 0);
    for i in 0..100 {
        let q = [i as f64 / 10.0, -2.0, 0.5];
        point.try_update(&q, &mut workspace).unwrap();
        assert_eq!(point.position(), q);
        assert_eq!(point.gradient(), q.map(|x| -x));
        assert_eq!(point.log_density(), Gaussian.log_prob(&q));
        assert_eq!(point.position().as_ptr(), original_position);
        assert_eq!(
            point.gradient().as_ptr(),
            if i % 2 == 0 {
                first_gradient
            } else {
                second_gradient
            }
        );
    }
    assert_eq!(target.calls.get(), 102);
}

#[test]
fn failed_updates_preserve_every_cached_field_and_allow_recovery() {
    let target = Target::new(3);
    let mut point = PointState::new(&target, buffer(&[1.0, 2.0, 3.0])).unwrap();
    let mut workspace = EvaluationWorkspace::new(3);
    let gradient_ptr = point.gradient().as_ptr();
    for mode in [
        Mode::Error,
        Mode::Partial,
        Mode::Log(f64::NAN),
        Mode::Log(f64::NEG_INFINITY),
        Mode::Gradient(f64::INFINITY),
    ] {
        target.mode.set(mode);
        assert!(point.try_update(&[4.0, 5.0, 6.0], &mut workspace).is_err());
        assert_eq!(point.position(), &[1.0, 2.0, 3.0]);
        assert_eq!(point.gradient(), &[-1.0, -2.0, -3.0]);
        assert_eq!(point.log_density(), -7.0);
        assert_eq!(point.gradient().as_ptr(), gradient_ptr);
    }
    target.mode.set(Mode::Good);
    point.try_update(&[4.0, 5.0, 6.0], &mut workspace).unwrap();
    assert_eq!(point.position(), &[4.0, 5.0, 6.0]);
    assert_eq!(point.gradient(), &[-4.0, -5.0, -6.0]);
    assert_eq!(point.log_density(), -38.5);
}

#[test]
fn panic_during_evaluation_preserves_cache_and_workspace_can_be_reused() {
    let target = Target::new(3);
    let mut point = PointState::new(&target, buffer(&[1.0; 3])).unwrap();
    let mut workspace = EvaluationWorkspace::new(3);
    target.mode.set(Mode::Panic);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        point.try_update(&[2.0; 3], &mut workspace)
    }));
    assert!(result.is_err());
    assert_eq!(point.position(), &[1.0; 3]);
    assert_eq!(point.gradient(), &[-1.0; 3]);
    assert_eq!(point.log_density(), -1.5);
    target.mode.set(Mode::Good);
    point.try_update(&[2.0; 3], &mut workspace).unwrap();
    assert_eq!(point.log_density(), -6.0);
}

#[test]
fn update_validates_workspace_inputs_and_fixed_target_dimension() {
    let target = Target::new(3);
    let mut point = PointState::new(&target, buffer(&[1.0; 3])).unwrap();
    for n in [0, 2, 4] {
        assert!(matches!(
            point.try_update(&[2.0; 3], &mut EvaluationWorkspace::new(n)),
            Err(EvaluationError::Dimension(_))
        ));
        assert!(matches!(
            point.try_update(&vec![2.0; n], &mut EvaluationWorkspace::new(3)),
            Err(EvaluationError::Dimension(_))
        ));
    }
    let mut workspace = EvaluationWorkspace::new(3);
    assert!(matches!(
        point.try_update(&[1.0, f64::INFINITY, 2.0], &mut workspace),
        Err(EvaluationError::NonFinitePosition { index: 1 })
    ));
    target.dimension.set(4);
    assert!(matches!(
        point.try_update(&[2.0; 3], &mut workspace),
        Err(EvaluationError::TargetDimensionChanged {
            expected: 3,
            actual: 4
        })
    ));
    assert_eq!(point.position(), &[1.0; 3]);
    assert_eq!(point.gradient(), &[-1.0; 3]);
    assert_eq!(point.log_density(), -1.5);
    assert_eq!(target.calls.get(), 1);
}

struct FusedOnly(Cell<usize>);
impl LogDensity for FusedOnly {
    type Point = [f64];
    fn log_prob(&self, _: &[f64]) -> f64 {
        panic!("split density must not be called")
    }
}
impl GradLogDensity for FusedOnly {
    type Gradient = [f64];
    fn grad_log_prob(&self, _: &[f64], _: &mut [f64]) {
        panic!("split gradient must not be called")
    }
}
impl FusedLogDensity for FusedOnly {
    fn log_prob_and_grad(&self, q: &[f64], gradient: &mut [f64]) -> f64 {
        self.0.set(self.0.get() + 1);
        Gaussian.log_prob_and_grad(q, gradient)
    }
}

#[test]
fn legacy_adapter_uses_only_fused_evaluation_and_checks_direct_call_shapes() {
    let legacy = FusedOnly(Cell::new(0));
    let target = FusedAdapter::new(&legacy, 3);
    let mut gradient = [123.0; 3];
    assert!(target.logp_grad(&[1.0], &mut gradient).is_err());
    assert_eq!(gradient, [123.0; 3]);
    assert_eq!(legacy.0.get(), 0);
    assert_eq!(
        evaluate(&target, &[1.0, 2.0, 3.0], &mut gradient).unwrap(),
        -7.0
    );
    assert_eq!(gradient, [-1.0, -2.0, -3.0]);
    assert_eq!(legacy.0.get(), 1);
}

#[test]
fn randomized_dimensions_and_updates_match_gaussian_reference() {
    let mut rng = SmallRng::seed_from_u64(5423);
    for _ in 0..64 {
        let n = rng.random_range(0..66);
        let target = FusedAdapter::new(&Gaussian, n);
        let mut point = PointState::new(&target, OwnedBuffer::new(n)).unwrap();
        let mut workspace = EvaluationWorkspace::new(n);
        let q: Vec<_> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
        point.try_update(&q, &mut workspace).unwrap();
        assert_eq!(point.log_density(), Gaussian.log_prob(&q));
        assert_eq!(point.position(), q);
        assert!(point.gradient().iter().zip(&q).all(|(g, q)| *g == -*q));
        if n > 0 {
            let mut bad = q.clone();
            bad[rng.random_range(0..n)] = f64::NAN;
            assert!(point.try_update(&bad, &mut workspace).is_err());
            assert_eq!(point.position(), q);
        }
    }
}

#[test]
fn zero_dimensional_targets_and_dynamic_dispatch_are_supported() {
    let adapter = FusedAdapter::new(&Gaussian, 0);
    let target: &dyn LogDensityGradient<Error = kernels::target::DimensionError> = &adapter;
    let mut point = PointState::new(target, OwnedBuffer::new(0)).unwrap();
    point
        .try_update(&[], &mut EvaluationWorkspace::new(0))
        .unwrap();
    assert_eq!(point.dimension(), 0);
    assert!(point.gradient().is_empty());
    assert_eq!(point.log_density(), 0.0);
}

#[test]
fn cloning_preserves_complete_cache_and_clone_from_reuses_matching_buffers() {
    let target = FusedAdapter::new(&Gaussian, 2);
    let source = PointState::new(&target, buffer(&[1.0, 2.0])).unwrap();
    let mut copy = source.clone();
    assert!(std::ptr::eq(source.target(), copy.target()));
    assert_ne!(source.position().as_ptr(), copy.position().as_ptr());
    let q_ptr = copy.position().as_ptr();
    let g_ptr = copy.gradient().as_ptr();
    copy.clone_from(&source);
    assert_eq!(copy.position().as_ptr(), q_ptr);
    assert_eq!(copy.gradient().as_ptr(), g_ptr);
    assert_eq!(copy.gradient(), source.gradient());
    assert_eq!(copy.log_density(), source.log_density());
    let other_target = FusedAdapter::new(&Gaussian, 1);
    let other = PointState::new(&other_target, buffer(&[3.0])).unwrap();
    copy.clone_from(&other);
    assert!(std::ptr::eq(copy.target(), &other_target));
    assert_eq!(copy.position(), &[3.0]);
    assert_eq!(copy.gradient(), &[-3.0]);
    assert_eq!(copy.log_density(), -4.5);
}
