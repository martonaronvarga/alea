use alea_core::target::{EvaluationError, EvaluationWorkspace, LogDensityGradient, PointState};
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::IdentityMetric;
use alea_mcmc::hamiltonian::{Divergence, PhaseError, PhaseState, PhaseWorkspace, SignedStep};
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

#[derive(Debug, thiserror::Error)]
#[error("test model failure")]
struct ModelError;

#[derive(Debug)]
struct Target {
    dim: Cell<usize>,
    calls: Cell<usize>,
    mode: Cell<u8>,
    scale: f64,
}
impl Target {
    fn new(dim: usize, scale: f64) -> Self {
        Self {
            dim: Cell::new(dim),
            calls: Cell::new(0),
            mode: Cell::new(0),
            scale,
        }
    }
}
impl LogDensityGradient for Target {
    type Error = ModelError;
    fn dimension(&self) -> usize {
        self.dim.get()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, ModelError> {
        self.calls.set(self.calls.get() + 1);
        match self.mode.get() {
            1 => {
                g.fill(42.0);
                return Err(ModelError);
            }
            2 => {
                g.fill(42.0);
                panic!("model panic");
            }
            3 => {
                g.fill(f64::NAN);
                return Ok(0.0);
            }
            _ => (),
        }
        for (g, q) in g.iter_mut().zip(q) {
            *g = -self.scale * q;
        }
        Ok(-0.5 * self.scale * q.iter().map(|q| q * q).sum::<f64>())
    }
}
fn buffer(x: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(x.len(), |i| x[i])
}
fn state<'a>(target: &'a Target, q: &[f64], p: &[f64]) -> PhaseState<'a, Target> {
    PhaseState::new(PointState::new(target, buffer(q)).unwrap(), buffer(p)).unwrap()
}
fn equal(a: &PhaseState<'_, Target>, b: &PhaseState<'_, Target>) {
    assert_eq!(a.point().position(), b.point().position());
    assert_eq!(a.point().gradient(), b.point().gradient());
    assert_eq!(a.point().log_density(), b.point().log_density());
    assert_eq!(a.momentum(), b.momentum());
    assert!(std::ptr::eq(a.point().target(), b.point().target()));
}

#[test]
fn snapshot_updates_are_atomic_for_errors_nonfinite_and_panics() {
    let target = Target::new(2, 1.0);
    let mut live = state(&target, &[1.0, -2.0], &[0.4, -0.1]);
    let before = live.clone();
    let mut scratch = EvaluationWorkspace::new(2);
    for momentum in [&[0.0][..], &[f64::NAN, 0.0], &[0.0, f64::INFINITY]] {
        assert!(
            live.try_update(&[2.0, 3.0], momentum, &mut scratch)
                .is_err()
        );
        equal(&live, &before);
    }
    assert_eq!(target.calls.get(), 1);
    for q in [&[0.0][..], &[f64::NAN, 0.0], &[0.0, f64::INFINITY]] {
        assert!(live.try_update(q, &[0.0, 0.0], &mut scratch).is_err());
        equal(&live, &before);
    }
    for mode in [1, 3] {
        target.mode.set(mode);
        assert!(
            live.try_update(&[2.0, 3.0], &[0.0, 0.0], &mut scratch)
                .is_err()
        );
        equal(&live, &before);
    }
    target.mode.set(2);
    assert!(
        catch_unwind(AssertUnwindSafe(|| live.try_update(
            &[2.0, 3.0],
            &[0.0, 0.0],
            &mut scratch
        )))
        .is_err()
    );
    equal(&live, &before);
    target.mode.set(0);
    live.try_update(&[2.0, 3.0], &[0.0, 0.0], &mut scratch)
        .unwrap();
    assert_eq!(live.point().gradient(), &[-2.0, -3.0]);
    assert_eq!(live.point().log_density(), -6.5);
    assert_eq!(live.momentum(), &[0.0, 0.0]);
}

#[test]
fn saved_phases_copy_whole_target_binding_without_reevaluation() {
    let target = Target::new(2, 1.0);
    let other = Target::new(2, 3.0);
    let initial = state(&target, &[0.7, -0.2], &[-0.3, 0.6]);
    let mut saved = state(&other, &[2.0, 3.0], &[4.0, 5.0]);
    let mut workspace = PhaseWorkspace::new(initial.point());
    let metric = IdentityMetric::new(2);
    let mut phase = workspace.start_from_phase(&initial, &metric).unwrap();
    assert_eq!(target.calls.get(), 1);
    for _ in 0..10 {
        phase = phase.step(SignedStep::new(0.1).unwrap()).unwrap();
    }
    phase.save_into(&mut saved).unwrap();
    assert_eq!(target.calls.get(), 11);
    assert_eq!(other.calls.get(), 1);
    assert!(std::ptr::eq(saved.point().target(), &target));
    assert_eq!(saved.momentum(), phase.momentum());
    assert_eq!(saved.point().position(), phase.point().position());
    for _ in 0..10 {
        phase = phase.step(SignedStep::new(-0.1).unwrap()).unwrap();
    }
    phase.save_into(&mut saved).unwrap();
    for (q, original) in saved
        .point()
        .position()
        .iter()
        .zip(initial.point().position())
    {
        assert!((q - original).abs() < 1e-12);
    }
    for (p, original) in saved.momentum().iter().zip(initial.momentum()) {
        assert!((p - original).abs() < 1e-12);
    }
    assert_eq!(initial.point().position(), &[0.7, -0.2]);
    saved.clone_from(&initial);
    equal(&saved, &initial);
}

#[test]
fn failed_phase_can_only_restart_and_never_changes_snapshot() {
    let target = Target::new(2, 1.0);
    let initial = state(&target, &[0.7, -0.2], &[-0.3, 0.6]);
    let before = initial.clone();
    let mut workspace = PhaseWorkspace::new(initial.point());
    let metric = IdentityMetric::new(2);
    let step = SignedStep::new(0.1).unwrap();
    for mode in [1, 2, 3] {
        target.mode.set(mode);
        let phase = workspace.start_from_phase(&initial, &metric).unwrap();
        if mode == 2 {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let _ = phase.step(step);
                }))
                .is_err()
            );
        } else {
            assert!(phase.step(step).is_err());
        }
        equal(&initial, &before);
    }
    target.mode.set(0);
    let phase = workspace
        .start_from_phase(&initial, &metric)
        .unwrap()
        .step(step)
        .unwrap();
    assert_ne!(phase.point().position(), initial.point().position());
}

#[test]
fn shapes_and_changed_target_dimension_fail_without_mutating_destination() {
    let target = Target::new(2, 1.0);
    let other = Target::new(1, 1.0);
    let initial = state(&target, &[1.0, 2.0], &[0.0, 0.0]);
    let mut wrong = state(&other, &[3.0], &[4.0]);
    let before = wrong.clone();
    let mut workspace = PhaseWorkspace::new(initial.point());
    let metric = IdentityMetric::new(2);
    let phase = workspace.start_from_phase(&initial, &metric).unwrap();
    assert!(phase.save_into(&mut wrong).is_err());
    equal(&wrong, &before);
    let mut scratch = EvaluationWorkspace::new(2);
    let mut live = initial.clone();
    target.dim.set(3);
    assert!(matches!(
        live.try_update(&[1.0, 2.0], &[0.0, 0.0], &mut scratch),
        Err(PhaseError::Evaluation(
            EvaluationError::TargetDimensionChanged { .. }
        ))
    ));
    assert!(workspace.start_from_phase(&initial, &metric).is_err());
    equal(&live, &initial);
    target.dim.set(2);
    assert!(matches!(
        PhaseState::new(initial.point().clone(), buffer(&[0.0, f64::NAN])),
        Err(PhaseError::Divergence(Divergence::Momentum { index: 1 }))
    ));
}

#[test]
fn seeded_random_shapes_and_updates_preserve_coherent_caches() {
    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    let mut rng = SmallRng::seed_from_u64(981);
    for _ in 0..24 {
        let dim = rng.random_range(1..18);
        let target = Target::new(dim, 1.0);
        let q: Vec<f64> = (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect();
        let p: Vec<f64> = (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect();
        let mut live = state(&target, &q, &p);
        let before = live.clone();
        let mut eval = EvaluationWorkspace::new(dim);
        let mut bad = p.clone();
        bad[rng.random_range(0..dim)] = f64::NAN;
        assert!(live.try_update(&p, &bad, &mut eval).is_err());
        equal(&live, &before);
        assert!(live.try_update(&q[..dim - 1], &p, &mut eval).is_err());
        equal(&live, &before);
        live.try_update(&p, &q, &mut eval).unwrap();
        assert_eq!(
            live.point().gradient(),
            p.iter().map(|x| -x).collect::<Vec<_>>()
        );
        assert_eq!(live.momentum(), q);
    }
}
