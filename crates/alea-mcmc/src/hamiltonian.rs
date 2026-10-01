//! Euclidean phase evolution and persistent snapshots, independent of selection/MH.
//!
//! A `Phase` is a borrowed validity token. Advancing consumes it; only success
//! returns another token. Failure/unwinding may dirty workspace but cannot leave
//! a usable partial phase. Starting again resets from a coherent cached point.
#![forbid(unsafe_code)]

use alea_core::target::{
    DimensionError, EvaluationError, EvaluationWorkspace, LogDensityGradient, PointState,
};
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{EuclideanMetric, MetricError};
use rand::Rng;

/// Numerical reason a trajectory was rejected without changing the live point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Divergence {
    /// Integration produced a non-finite coordinate.
    Position { index: usize },
    /// The target returned a non-finite log density (including negative infinity).
    LogDensity,
    /// The target returned an incomplete or non-finite gradient.
    Gradient { index: usize },
    /// Momentum generation or an integration kick produced a non-finite value.
    Momentum { index: usize },
    /// Kinetic/Hamiltonian energy or its difference was not finite.
    Energy,
    /// The finite absolute endpoint energy error exceeded the configured limit.
    EnergyErrorLimit,
}

/// A phase could not be constructed or advanced. Failed steps consume the token.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PhaseError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Metric(MetricError),
    #[error(transparent)]
    Evaluation(EvaluationError<E>),
    #[error("numerical trajectory divergence: {0:?}")]
    Divergence(Divergence),
}

/// Finite nonzero signed integration step. Construct once outside the hot loop.
#[derive(Debug, Clone, Copy)]
pub struct SignedStep(f64);

impl SignedStep {
    /// Returns `None` for zero or a non-finite value.
    pub fn new(value: f64) -> Option<Self> {
        (value.is_finite() && value != 0.0).then_some(Self(value))
    }

    /// The validated signed step.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// A coherent owned position/gradient/log-density/momentum snapshot.
/// Fields cannot be mutated independently. Copies retain the target binding.
/// Reuse snapshots for trajectory endpoints; tree selection belongs to M5.
#[derive(Debug)]
pub struct PhaseState<'a, T: LogDensityGradient + ?Sized> {
    point: PointState<'a, T>,
    momentum: OwnedBuffer,
}

impl<'a, T: LogDensityGradient + ?Sized> PhaseState<'a, T> {
    /// Takes a cached point and momentum without reevaluating the target.
    /// # Errors
    /// Rejects mismatched dimensions, changed target dimensions or non-finite momentum.
    pub fn new(
        point: PointState<'a, T>,
        momentum: OwnedBuffer,
    ) -> Result<Self, PhaseError<T::Error>> {
        Self::validate(&point, &momentum)?;
        Ok(Self { point, momentum })
    }

    fn validate(point: &PointState<'a, T>, momentum: &[f64]) -> Result<(), PhaseError<T::Error>> {
        let expected = point.dimension();
        if momentum.len() != expected {
            return Err(PhaseError::Metric(MetricError::VectorLength {
                expected,
                source_len: momentum.len(),
                destination_len: expected,
            }));
        }
        let actual = point.target().dimension();
        if actual != expected {
            return Err(PhaseError::Evaluation(
                EvaluationError::TargetDimensionChanged { expected, actual },
            ));
        }
        if let Some(index) = momentum.iter().position(|p| !p.is_finite()) {
            return Err(PhaseError::Divergence(Divergence::Momentum { index }));
        }
        Ok(())
    }

    /// The complete cached model point.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.point
    }
    /// Read-only momentum, in the same coordinate system as the point.
    pub fn momentum(&self) -> &[f64] {
        &self.momentum
    }
    /// Number of position and momentum coordinates.
    pub fn dimension(&self) -> usize {
        self.point.dimension()
    }

    /// Replaces position, gradient, density and momentum together, with no allocation.
    /// An error or model panic leaves this entire snapshot unchanged; target-side
    /// effects are not rolled back. Evaluation scratch may be dirtied.
    /// # Errors
    /// Rejects invalid shapes, non-finite inputs/results and target evaluation errors.
    pub fn try_update(
        &mut self,
        position: &[f64],
        momentum: &[f64],
        scratch: &mut EvaluationWorkspace,
    ) -> Result<(), PhaseError<T::Error>> {
        Self::validate(&self.point, momentum)?;
        self.point
            .try_update(position, scratch)
            .map_err(PhaseError::Evaluation)?;
        self.momentum.copy_from_slice(momentum);
        Ok(())
    }
}

impl<T: LogDensityGradient + ?Sized> Clone for PhaseState<'_, T> {
    fn clone(&self) -> Self {
        Self {
            point: self.point.clone(),
            momentum: self.momentum.clone(),
        }
    }
    /// Same-sized copies reuse all allocations, including the complete target binding.
    fn clone_from(&mut self, source: &Self) {
        self.point.clone_from(&source.point);
        if self.momentum.len() == source.momentum.len() {
            self.momentum.copy_from_slice(&source.momentum);
        } else {
            self.momentum = source.momentum.clone();
        }
    }
}

/// Reusable proposal plus scratch. Dirty storage is inaccessible without a phase
/// token. Six aligned buffers here plus the live point's two preserve HMC's budget.
#[derive(Debug)]
pub struct PhaseWorkspace<'a, T: LogDensityGradient + ?Sized> {
    point: PointState<'a, T>,
    momentum: OwnedBuffer,
    velocity: OwnedBuffer,
    next_position: OwnedBuffer,
    evaluation: EvaluationWorkspace,
}

impl<'a, T: LogDensityGradient + ?Sized> PhaseWorkspace<'a, T> {
    /// Allocates chain-local scratch once. Copies the cache without model evaluation.
    pub fn new(point: &PointState<'a, T>) -> Self {
        let dim = point.dimension();
        Self {
            point: point.clone(),
            momentum: OwnedBuffer::new(dim),
            velocity: OwnedBuffer::new(dim),
            next_position: OwnedBuffer::new(dim),
            evaluation: EvaluationWorkspace::new(dim),
        }
    }

    fn prepare(&mut self, point: &PointState<'a, T>) -> Result<(), EvaluationError<T::Error>> {
        let expected = self.momentum.len();
        if point.dimension() != expected {
            return Err(DimensionError {
                expected,
                position: point.dimension(),
                gradient: point.gradient().len(),
            }
            .into());
        }
        let actual = point.target().dimension();
        if actual != expected {
            return Err(EvaluationError::TargetDimensionChanged { expected, actual });
        }
        self.point.clone_from(point);
        Ok(())
    }

    /// Starts from a complete point and newly sampled metric-correct momentum.
    /// Model evaluation is not repeated. Errors never mutate the supplied point.
    /// # Errors
    /// Rejects dimension changes, metric errors, and non-finite sampled momentum.
    pub fn start<'w, M: EuclideanMetric, R: Rng + ?Sized>(
        &'w mut self,
        point: &PointState<'a, T>,
        metric: &'w M,
        rng: &mut R,
    ) -> Result<Phase<'w, 'a, T, M>, PhaseError<T::Error>> {
        self.prepare(point).map_err(PhaseError::Evaluation)?;
        // Check before consuming RNG, including when this boundary is used by
        // a caller other than Hmc.
        if metric.dimension() != self.momentum.len() {
            return Err(PhaseError::Metric(MetricError::VectorLength {
                expected: metric.dimension(),
                source_len: self.velocity.len(),
                destination_len: self.momentum.len(),
            }));
        }
        for z in self.velocity.iter_mut() {
            *z = alea_math::random::standard_normal(rng);
        }
        metric
            .sample_momentum(&self.velocity, &mut self.momentum)
            .map_err(PhaseError::Metric)?;
        self.check_momentum()?;
        Ok(Phase {
            workspace: self,
            metric,
        })
    }

    fn check_momentum(&self) -> Result<(), PhaseError<T::Error>> {
        if let Some(index) = self.momentum.iter().position(|p| !p.is_finite()) {
            return Err(PhaseError::Divergence(Divergence::Momentum { index }));
        }
        Ok(())
    }

    /// Reuses evaluation scratch for the chain's explicit transactional reset.
    pub(crate) fn update_point(
        &mut self,
        point: &mut PointState<'a, T>,
        position: &[f64],
    ) -> Result<(), EvaluationError<T::Error>> {
        point.try_update(position, &mut self.evaluation)
    }

    /// Restarts from a complete snapshot without allocation or model evaluation.
    /// # Errors
    /// Rejects incompatible dimensions, a changed target dimension or invalid momentum.
    pub fn start_from_phase<'w, M: EuclideanMetric>(
        &'w mut self,
        state: &PhaseState<'a, T>,
        metric: &'w M,
    ) -> Result<Phase<'w, 'a, T, M>, PhaseError<T::Error>> {
        self.start_with_momentum(&state.point, metric, &state.momentum)
    }

    fn start_with_momentum<'w, M: EuclideanMetric>(
        &'w mut self,
        point: &PointState<'a, T>,
        metric: &'w M,
        momentum: &[f64],
    ) -> Result<Phase<'w, 'a, T, M>, PhaseError<T::Error>> {
        self.prepare(point).map_err(PhaseError::Evaluation)?;
        if momentum.len() != self.momentum.len() || metric.dimension() != self.momentum.len() {
            return Err(PhaseError::Metric(MetricError::VectorLength {
                expected: metric.dimension(),
                source_len: momentum.len(),
                destination_len: self.momentum.len(),
            }));
        }
        self.momentum.copy_from_slice(momentum);
        self.check_momentum()?;
        Ok(Phase {
            workspace: self,
            metric,
        })
    }
}

/// Only a complete phase has this token. It is deliberately neither Clone nor Copy.
/// Dropping a token discards the proposal; it never changes the starting snapshot.
/// A failed advance cannot leave a token that callers could save:
/// ```compile_fail
/// use alea_core::target::LogDensityGradient;
/// use alea_math::metric::EuclideanMetric;
/// use alea_mcmc::hamiltonian::{Phase, SignedStep};
/// fn advance<T: LogDensityGradient, M: EuclideanMetric>(phase: Phase<'_, '_, T, M>) {
///     let _result = phase.step(SignedStep::new(0.1).unwrap());
///     let _invalid_reuse = phase.point(); // phase was moved, even on error
/// }
/// ```
#[must_use = "save the complete phase or explicitly discard the proposal"]
pub struct Phase<'w, 'a, T: LogDensityGradient + ?Sized, M> {
    workspace: &'w mut PhaseWorkspace<'a, T>,
    metric: &'w M,
}

impl<T: LogDensityGradient + ?Sized, M: EuclideanMetric> Phase<'_, '_, T, M> {
    /// The same signed kick/drift/kick map is used in sampling and reference tests.
    /// # Errors
    /// EuclideanMetric/model failures or non-finite evolution consume the token, preventing
    /// partial phases from being saved. Restart the workspace after error or panic.
    pub fn step(self, step: SignedStep) -> Result<Self, PhaseError<T::Error>> {
        self.step_with(step, &crate::integrator::Leapfrog)
    }

    /// Advance one complete symmetric macrostep. Only its endpoint can be saved
    /// or selected by a trajectory. Every drift refreshes the fused cache once.
    /// # Errors
    /// Any failed internal stage consumes the phase token, just like leapfrog.
    pub fn step_with<I: crate::integrator::Integrator>(
        self,
        step: SignedStep,
        integrator: &I,
    ) -> Result<Self, PhaseError<T::Error>> {
        let s = &mut *self.workspace;
        let eps = step.0;
        for stage in 0..integrator.stages() {
            for (p, g) in s.momentum.iter_mut().zip(s.point.gradient()) {
                *p += integrator.kick(stage) * eps * g;
            }
            s.check_momentum()?;
            self.metric
                .velocity(&s.momentum, &mut s.velocity)
                .map_err(PhaseError::Metric)?;
            for ((next, q), v) in s
                .next_position
                .iter_mut()
                .zip(s.point.position())
                .zip(s.velocity.iter())
            {
                *next = q + (eps * integrator.drift(stage)) * v;
            }
            if let Err(error) = s.point.try_update(&s.next_position, &mut s.evaluation) {
                return Err(match error {
                    EvaluationError::NonFinitePosition { index } => {
                        PhaseError::Divergence(Divergence::Position { index })
                    }
                    EvaluationError::NonFiniteLogDensity => {
                        PhaseError::Divergence(Divergence::LogDensity)
                    }
                    EvaluationError::NonFiniteGradient { index } => {
                        PhaseError::Divergence(Divergence::Gradient { index })
                    }
                    other => PhaseError::Evaluation(other),
                });
            }
        }
        for (p, g) in s.momentum.iter_mut().zip(s.point.gradient()) {
            *p += integrator.kick(integrator.stages()) * eps * g;
        }
        s.check_momentum()?;
        Ok(self)
    }

    /// Computes Hamiltonian without caching kinetic energy across momentum changes.
    /// # Errors
    /// Returns metric errors or a non-finite energy divergence.
    pub fn energy(&mut self) -> Result<f64, PhaseError<T::Error>> {
        let s = &mut self.workspace;
        self.metric
            .velocity(&s.momentum, &mut s.velocity)
            .map_err(PhaseError::Metric)?;
        let kinetic = 0.5
            * s.momentum
                .iter()
                .zip(s.velocity.iter())
                .map(|(p, v)| p * v)
                .sum::<f64>();
        let energy = -s.point.log_density() + kinetic;
        if !energy.is_finite() {
            return Err(PhaseError::Divergence(Divergence::Energy));
        }
        Ok(energy)
    }
}

impl<'a, T: LogDensityGradient + ?Sized, M> Phase<'_, 'a, T, M> {
    /// Read-only cached endpoint.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.workspace.point
    }
    /// Read-only endpoint momentum.
    pub fn momentum(&self) -> &[f64] {
        &self.workspace.momentum
    }

    /// Saves the whole phase into a preallocated same-sized snapshot. The target
    /// binding is copied too, so a snapshot may be coherently rebound to a target
    /// of the same Rust type/lifetime. No reevaluation or allocation occurs.
    /// # Errors
    /// Rejects destination dimension mismatch before changing any destination field.
    pub fn save_into(
        &self,
        destination: &mut PhaseState<'a, T>,
    ) -> Result<(), PhaseError<T::Error>> {
        let expected = self.workspace.point.dimension();
        if destination.dimension() != expected {
            return Err(PhaseError::Metric(MetricError::VectorLength {
                expected,
                source_len: expected,
                destination_len: destination.dimension(),
            }));
        }
        destination.point.clone_from(&self.workspace.point);
        destination
            .momentum
            .copy_from_slice(&self.workspace.momentum);
        Ok(())
    }

    /// Selection policy lives in the caller; only a complete phase can commit.
    pub(crate) fn commit(self, live: &mut PointState<'a, T>) {
        std::mem::swap(live, &mut self.workspace.point);
    }
}

#[cfg(test)]
mod reference_tests;

#[cfg(test)]
mod tests {
    use super::*;

    use alea_distributions::Gaussian;
    use alea_math::metric::{CholeskyFactor, DenseMetric, DiagonalMetric, IdentityMetric};
    use rand::{SeedableRng, rngs::SmallRng};
    use std::{
        cell::Cell,
        panic::{AssertUnwindSafe, catch_unwind},
    };

    fn buffer(values: &[f64]) -> OwnedBuffer {
        OwnedBuffer::from_fn(values.len(), |i| values[i])
    }

    #[test]
    fn signed_step_rejects_zero_and_nonfinite_but_accepts_both_directions() {
        for value in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(SignedStep::new(value).is_none());
        }
        assert!(SignedStep::new(0.1).is_some());
        assert!(SignedStep::new(-0.1).is_some());
    }

    fn check_reversal<M: EuclideanMetric>(metric: M) {
        let target = Gaussian::new(2);
        let point = PointState::new(&target, buffer(&[0.7, -0.2])).unwrap();
        let mut scratch = PhaseWorkspace::new(&point);
        let mut phase = scratch
            .start_with_momentum(&point, &metric, &[-0.3, 0.6])
            .unwrap();
        for eps in [0.1, -0.1] {
            let step = SignedStep::new(eps).unwrap();
            for _ in 0..10 {
                phase = phase.step(step).unwrap();
            }
        }
        for (actual, expected) in phase.workspace.point.position().iter().zip([0.7, -0.2]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in phase.workspace.momentum.iter().zip([-0.3, 0.6]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(point.position(), &[0.7, -0.2]);
    }

    #[test]
    fn fallible_leapfrog_reverses_for_all_builtin_mass_types() {
        check_reversal(IdentityMetric::new(2));
        check_reversal(DiagonalMetric::new(buffer(&[4.0, 0.5])).unwrap());
        check_reversal(DenseMetric::new(
            CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 1.3])).unwrap(),
        ));
    }

    #[test]
    fn fallible_leapfrog_has_second_order_energy_error() {
        let target = Gaussian::new(1);
        let metric = IdentityMetric::new(1);
        let point = PointState::new(&target, buffer(&[0.7])).unwrap();
        let integrate = |eps: f64, count| {
            let mut scratch = PhaseWorkspace::new(&point);
            let mut phase = scratch
                .start_with_momentum(&point, &metric, &[-0.3])
                .unwrap();
            let initial = phase.energy().unwrap();
            for _ in 0..count {
                phase = phase.step(SignedStep::new(eps).unwrap()).unwrap();
            }
            (phase.energy().unwrap() - initial).abs()
        };
        assert!(integrate(0.05, 20) < integrate(0.1, 10) / 3.5);
    }

    #[test]
    fn start_rejects_shapes_and_nonfinite_momentum_before_exposing_phase() {
        let target = Gaussian::new(2);
        let other = Gaussian::new(1);
        let point = PointState::new(&target, buffer(&[0.7, -0.2])).unwrap();
        let wrong_point = PointState::new(&other, buffer(&[0.7])).unwrap();
        let mut scratch = PhaseWorkspace::new(&point);
        let metric = IdentityMetric::new(2);
        let wrong_metric = IdentityMetric::new(1);
        let mut rng = SmallRng::seed_from_u64(91);
        let before = rng.clone();
        assert!(matches!(
            scratch.start(&wrong_point, &metric, &mut rng),
            Err(PhaseError::Evaluation(EvaluationError::Dimension(_)))
        ));
        assert!(matches!(
            scratch.start(&point, &wrong_metric, &mut rng),
            Err(PhaseError::Metric(_))
        ));
        assert_eq!(rng, before);
        assert!(matches!(
            scratch.start_with_momentum(&point, &metric, &[0.0]),
            Err(PhaseError::Metric(_))
        ));
        assert!(matches!(
            scratch.start_with_momentum(&point, &metric, &[0.0, f64::NAN]),
            Err(PhaseError::Divergence(Divergence::Momentum { index: 1 }))
        ));
        assert!(
            scratch
                .start_with_momentum(&point, &metric, &[0.1, 0.2])
                .is_ok()
        );
    }

    #[derive(Debug, thiserror::Error)]
    #[error("injected evaluation failure")]
    struct Failure;

    struct Target {
        fault: Cell<u8>,
        calls: Cell<usize>,
    }
    impl LogDensityGradient for Target {
        type Error = Failure;
        fn dimension(&self) -> usize {
            1
        }
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
            self.calls.set(self.calls.get() + 1);
            match self.fault.get() {
                1 => {
                    g[0] = 123.0;
                    return Err(Failure);
                }
                2 => panic!("injected panic"),
                3 => {
                    g[0] = f64::MAX;
                    return Ok(0.0);
                }
                _ => {}
            }
            Ok(Gaussian::new(q.len()).logp_grad(q, g).unwrap())
        }
    }

    #[test]
    fn partial_step_error_and_unwind_require_restart_and_leave_live_cache_untouched() {
        let target = Target {
            fault: Cell::new(0),
            calls: Cell::new(0),
        };
        let metric = IdentityMetric::new(1);
        let mut live = PointState::new(&target, buffer(&[0.7])).unwrap();
        let mut scratch = PhaseWorkspace::new(&live);
        assert_eq!(target.calls.get(), 1); // workspace construction only copies cache
        for fault in [1, 2, 3] {
            let phase = scratch
                .start_with_momentum(&live, &metric, &[-0.3])
                .unwrap();
            let phase = phase.step(SignedStep::new(0.1).unwrap()).unwrap();
            target.fault.set(fault);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                phase.step(SignedStep::new(4.0).unwrap())
            }));
            match fault {
                1 => assert!(matches!(
                    outcome,
                    Ok(Err(PhaseError::Evaluation(EvaluationError::Model(Failure))))
                )),
                2 => assert!(outcome.is_err()),
                3 => assert!(matches!(
                    outcome,
                    Ok(Err(PhaseError::Divergence(Divergence::Momentum {
                        index: 0
                    })))
                )),
                _ => unreachable!(),
            }
            // Exact preservation, including after a finite point update followed
            // by overflow in the final kick (fault 3).
            assert_eq!(live.position(), &[0.7]);
            assert_eq!(live.gradient(), &[-0.7]);
            target.fault.set(0);
        }
        let calls = target.calls.get();
        let phase = scratch
            .start_with_momentum(&live, &metric, &[-0.3])
            .unwrap();
        let phase = phase.step(SignedStep::new(0.1).unwrap()).unwrap();
        assert_eq!(target.calls.get(), calls + 1);
        phase.commit(&mut live);
        assert!(live.position()[0] != 0.7);
        assert_eq!(live.gradient()[0], -live.position()[0]);
    }

    #[test]
    fn dropping_phase_is_rejection_and_next_start_copies_whole_live_point() {
        let target = Gaussian::new(1);
        let metric = IdentityMetric::new(1);
        let mut live = PointState::new(&target, buffer(&[0.7])).unwrap();
        let mut scratch = PhaseWorkspace::new(&live);
        {
            let phase = scratch
                .start_with_momentum(&live, &metric, &[-0.3])
                .unwrap();
            let _unselected = phase.step(SignedStep::new(0.1).unwrap()).unwrap();
        }
        assert_eq!(live.position(), &[0.7]);
        scratch.update_point(&mut live, &[2.0]).unwrap();
        let phase = scratch.start_with_momentum(&live, &metric, &[0.0]).unwrap();
        assert_eq!(phase.workspace.point.position(), &[2.0]);
        assert_eq!(phase.workspace.point.gradient(), &[-2.0]);
        assert_eq!(phase.workspace.point.log_density(), -2.0);
    }
}
