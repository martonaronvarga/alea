//! Identity-preconditioned isokinetic dynamics, exact MAMS correction, and an
//! explicitly approximate unadjusted MCLMC kernel.
//!
//! These kernels do not reuse canonical Hamiltonian energy. Their velocity has
//! unit norm and each analytic score kick contributes the isokinetic Jacobian /
//! kinetic correction derived for microcanonical dynamics.

use std::{error::Error, num::NonZeroUsize};

use alea_core::target::{EvaluationError, EvaluationWorkspace, LogDensityGradient, PointState};
use alea_math::buffer::OwnedBuffer;
use rand::{Rng, RngExt};
use thiserror::Error;

use crate::{
    MarkovChain,
    hmc::{HmcConfigError, HmcOptions, StepSize},
};

/// Validated exact/approximate isokinetic trajectory settings.
#[derive(Debug, Clone, Copy, Default)]
pub struct MamsOptions {
    hmc: HmcOptions,
    trajectory: Option<(f64, NonZeroUsize)>,
}

impl MamsOptions {
    /// Creates a fixed-length identity-preconditioned isokinetic trajectory.
    ///
    /// # Errors
    /// Rejects an invalid step, zero count, or invalid correction-error limit.
    pub fn new(step_size: f64, integration_steps: usize) -> Result<Self, HmcConfigError> {
        Ok(Self {
            hmc: HmcOptions::new(step_size, integration_steps)?,
            trajectory: None,
        })
    }

    /// During step adaptation, keep duration near this frozen pilot scale by
    /// recomputing `ceil(time / step)`, capped by `maximum_steps` (1..=1024).
    /// Retained transitions use the resulting frozen count and never adapt it.
    /// # Errors
    /// Rejects nonpositive/nonfinite duration or an invalid work cap.
    pub fn with_trajectory_time(
        mut self,
        time: f64,
        maximum_steps: usize,
    ) -> Result<Self, HmcConfigError> {
        if !time.is_finite() || time <= 0. || !(1..=1024).contains(&maximum_steps) {
            return Err(HmcConfigError::IntegrationCount);
        }
        self.trajectory = Some((
            time,
            NonZeroUsize::new(maximum_steps).expect("positive cap"),
        ));
        self.update_step(self.step_size());
        Ok(self)
    }

    fn update_step(&mut self, step: StepSize) {
        let count = self
            .trajectory
            .map_or(self.integration_steps().get(), |(time, cap)| {
                (time / step.value()).ceil().clamp(1., cap.get() as f64) as usize
            });
        self.hmc = HmcOptions::new(step.value(), count)
            .expect("validated step and bounded count")
            .with_max_energy_error(self.max_correction_error())
            .expect("validated cutoff");
    }

    /// Sets the absolute negative-log acceptance correction limit.
    pub fn with_max_correction_error(mut self, limit: f64) -> Result<Self, HmcConfigError> {
        self.hmc = self.hmc.with_max_energy_error(limit)?;
        Ok(self)
    }

    /// Positive integration step.
    pub fn step_size(self) -> StepSize {
        self.hmc.step_size()
    }

    /// Complete velocity-Verlet macrosteps.
    pub fn integration_steps(self) -> NonZeroUsize {
        self.hmc.integration_steps()
    }

    /// Absolute correction-error diagnostic threshold.
    pub fn max_correction_error(self) -> f64 {
        self.hmc.max_energy_error()
    }
}

/// Numerical rejection from noncanonical isokinetic integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MamsDivergence {
    /// Unit-direction or correction arithmetic became nonfinite.
    Dynamics,
    /// The finite correction error exceeded the configured diagnostic limit.
    CorrectionLimit,
}

/// Construction or checked target failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MamsError<E: Error + 'static> {
    /// Isokinetic dynamics requires at least two dimensions because of `d - 1`.
    #[error("isokinetic dynamics requires target dimension at least two")]
    Dimension,
    /// Direction sampling, refreshment, or unadjusted dynamics became nonfinite.
    #[error("isokinetic direction arithmetic became nonfinite")]
    Numerical,
    /// Initial or trajectory target evaluation failed.
    #[error("isokinetic target evaluation failed at step {integration_step}: {source}")]
    Evaluation {
        /// Zero during construction, otherwise the one-based macrostep.
        integration_step: usize,
        /// Checked target failure retaining the concrete backend source.
        #[source]
        source: EvaluationError<E>,
    },
}

impl<E: Error + 'static> MamsError<E> {
    pub(crate) fn is_numerical(&self) -> bool {
        matches!(
            self,
            Self::Numerical
                | Self::Evaluation {
                    source: EvaluationError::NonFinitePosition { .. }
                        | EvaluationError::NonFiniteLogDensity
                        | EvaluationError::NonFiniteGradient { .. },
                    ..
                }
        )
    }
}

/// One Metropolis-adjusted microcanonical transition.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct MamsTransition {
    /// Whether the proposal cache was committed.
    pub accepted: bool,
    /// `min(1, exp(logp_new - logp_old - correction))`.
    pub acceptance_probability: f64,
    /// Accumulated noncanonical kick correction.
    pub isokinetic_correction: Option<f64>,
    /// Negative log acceptance ratio, the MAMS analogue of energy error.
    pub correction_error: Option<f64>,
    /// Attempted complete isokinetic macrosteps.
    pub integration_steps: usize,
    /// Numerical rejection reason.
    pub divergence: Option<MamsDivergence>,
}

/// Exact-target, identity-preconditioned MAMS kernel.
#[derive(Debug)]
pub struct Mams<'a, T: LogDensityGradient + ?Sized> {
    live: PointState<'a, T>,
    proposal: PointState<'a, T>,
    evaluation: EvaluationWorkspace,
    next_position: OwnedBuffer,
    direction: OwnedBuffer,
    options: MamsOptions,
}

impl<'a, T: LogDensityGradient + ?Sized> Mams<'a, T> {
    /// Constructs an exact-target MAMS chain. A fresh unit direction is sampled
    /// for every transition.
    ///
    /// # Errors
    /// Requires dimension at least two and a valid initial target evaluation.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        options: MamsOptions,
    ) -> Result<Self, MamsError<T::Error>> {
        let dimension = target.dimension();
        if dimension < 2 {
            return Err(MamsError::Dimension);
        }
        let live = PointState::new(target, position).map_err(|source| MamsError::Evaluation {
            integration_step: 0,
            source,
        })?;
        Ok(Self {
            proposal: live.clone(),
            evaluation: EvaluationWorkspace::new(dimension),
            next_position: OwnedBuffer::new(dimension),
            direction: OwnedBuffer::new(dimension),
            live,
            options,
        })
    }

    /// Frozen corrected trajectory options.
    pub fn options(&self) -> MamsOptions {
        self.options
    }

    pub(crate) fn set_step_size(&mut self, step: StepSize) {
        self.options.update_step(step);
    }

    /// Current coherent exact-target cache.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.live
    }

    /// Attempts one independently directed, Metropolis-corrected trajectory.
    ///
    /// # Errors
    /// Target/backend failures leave the live point unchanged.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<MamsTransition, MamsError<T::Error>> {
        self.proposal.clone_from(&self.live);
        sample_unit_direction(&mut self.direction, rng).map_err(|_| MamsError::Numerical)?;
        let initial_log_density = self.live.log_density();
        let mut correction = 0.0;
        let mut transition = MamsTransition {
            accepted: false,
            acceptance_probability: 0.0,
            isokinetic_correction: None,
            correction_error: None,
            integration_steps: 0,
            divergence: None,
        };
        let half_step = 0.5 * self.options.step_size().value();
        for index in 1..=self.options.integration_steps().get() {
            let first =
                match isokinetic_kick(&mut self.direction, self.proposal.gradient(), half_step) {
                    Some(value) => value,
                    None => {
                        transition.divergence = Some(MamsDivergence::Dynamics);
                        return Ok(transition);
                    }
                };
            correction += first;
            for ((next, position), direction) in self
                .next_position
                .iter_mut()
                .zip(self.proposal.position())
                .zip(self.direction.iter())
            {
                *next = position + self.options.step_size().value() * direction;
            }
            if let Err(source) = self
                .proposal
                .try_update(&self.next_position, &mut self.evaluation)
            {
                let failure = MamsError::Evaluation {
                    integration_step: index,
                    source,
                };
                if failure.is_numerical() {
                    transition.integration_steps = index;
                    transition.divergence = Some(MamsDivergence::Dynamics);
                    return Ok(transition);
                }
                return Err(failure);
            }
            let second =
                match isokinetic_kick(&mut self.direction, self.proposal.gradient(), half_step) {
                    Some(value) => value,
                    None => {
                        transition.divergence = Some(MamsDivergence::Dynamics);
                        transition.integration_steps = index;
                        return Ok(transition);
                    }
                };
            correction += second;
            transition.integration_steps = index;
            if !correction.is_finite() {
                transition.divergence = Some(MamsDivergence::Dynamics);
                return Ok(transition);
            }
        }
        // The deterministic proposal involution includes velocity reversal. It
        // has no effect on the position-only state returned by this fresh-direction kernel.
        for direction in self.direction.iter_mut() {
            *direction = -*direction;
        }
        let log_ratio = self.proposal.log_density() - initial_log_density - correction;
        if !log_ratio.is_finite() {
            transition.divergence = Some(MamsDivergence::Dynamics);
            return Ok(transition);
        }
        let correction_error = -log_ratio;
        transition.isokinetic_correction = Some(correction);
        transition.correction_error = Some(correction_error);
        if correction_error.abs() > self.options.max_correction_error() {
            transition.divergence = Some(MamsDivergence::CorrectionLimit);
            return Ok(transition);
        }
        transition.acceptance_probability = log_ratio.min(0.0).exp();
        transition.accepted = rng.random::<f64>() < transition.acceptance_probability;
        if transition.accepted {
            self.live.clone_from(&self.proposal);
        }
        Ok(transition)
    }
}

impl<T: LogDensityGradient + ?Sized> MarkovChain for Mams<'_, T> {
    type Error = MamsError<T::Error>;
    type Transition = MamsTransition;

    fn position(&self) -> &[f64] {
        self.live.position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        Mams::step(self, rng)
    }
}

/// Explicit acknowledgement that an unadjusted kernel targets only an
/// approximation at finite step size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApproximateTarget(());

impl ApproximateTarget {
    /// Opts into finite-step discretization bias.
    pub const fn acknowledge_bias() -> Self {
        Self(())
    }
}

/// Settings for explicitly approximate MCLMC.
#[derive(Debug, Clone, Copy)]
pub struct ApproximateMclmcOptions {
    mams: MamsOptions,
    refresh_length: Option<f64>,
}

impl ApproximateMclmcOptions {
    /// Frozen integration step.
    pub fn step_size(self) -> StepSize {
        self.mams.step_size()
    }
    /// Frozen angular refresh length; None disables refreshment.
    pub fn refresh_length(self) -> Option<f64> {
        self.refresh_length
    }
    /// Complete macrosteps per retained transition.
    pub fn integration_steps(self) -> NonZeroUsize {
        self.mams.integration_steps()
    }

    /// Creates unadjusted isokinetic integration. `refresh_length=None` disables
    /// angular noise; otherwise it must be finite and positive.
    pub fn new(
        step_size: f64,
        integration_steps: usize,
        refresh_length: Option<f64>,
    ) -> Result<Self, HmcConfigError> {
        if refresh_length.is_some_and(|length| !length.is_finite() || length <= 0.0) {
            return Err(HmcConfigError::StepSize);
        }
        Ok(Self {
            mams: MamsOptions::new(step_size, integration_steps)?,
            refresh_length,
        })
    }
}

/// Unadjusted MCLMC transition; acceptance metadata is intentionally absent.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ApproximateMclmcTransition {
    /// Complete integration macrosteps.
    pub integration_steps: usize,
    /// Sum of corrections deliberately not used for MH adjustment.
    pub discarded_isokinetic_correction: f64,
    /// Negative log density change plus the deterministic isokinetic Jacobian.
    /// This is diagnostic only and never supplies an acceptance decision.
    pub correction_error: f64,
    /// Static finite-step guarantee marker.
    pub guarantee: ApproximateTarget,
}

/// Persistent-direction unadjusted MCLMC, constructible only with an explicit
/// [`ApproximateTarget`] acknowledgement.
#[derive(Debug)]
pub struct ApproximateMclmc<'a, T: LogDensityGradient + ?Sized> {
    live: PointState<'a, T>,
    proposal: PointState<'a, T>,
    evaluation: EvaluationWorkspace,
    next_position: OwnedBuffer,
    direction: OwnedBuffer,
    proposal_direction: OwnedBuffer,
    noise: OwnedBuffer,
    options: ApproximateMclmcOptions,
    attempted_steps: usize,
}

impl<'a, T: LogDensityGradient + ?Sized> ApproximateMclmc<'a, T> {
    /// Constructs an explicitly biased finite-step chain.
    ///
    /// # Errors
    /// Requires dimension at least two and a valid initial target evaluation.
    pub fn new<R: Rng + ?Sized>(
        _guarantee: ApproximateTarget,
        target: &'a T,
        position: OwnedBuffer,
        options: ApproximateMclmcOptions,
        rng: &mut R,
    ) -> Result<Self, MamsError<T::Error>> {
        let dimension = target.dimension();
        if dimension < 2 {
            return Err(MamsError::Dimension);
        }
        let live = PointState::new(target, position).map_err(|source| MamsError::Evaluation {
            integration_step: 0,
            source,
        })?;
        let mut direction = OwnedBuffer::new(dimension);
        sample_unit_direction(&mut direction, rng).map_err(|_| MamsError::Numerical)?;
        Ok(Self {
            proposal: live.clone(),
            evaluation: EvaluationWorkspace::new(dimension),
            next_position: OwnedBuffer::new(dimension),
            proposal_direction: direction.clone(),
            noise: OwnedBuffer::new(dimension),
            direction,
            live,
            options,
            attempted_steps: 0,
        })
    }

    /// Frozen finite-step options, including the explicit refresh length.
    pub fn options(&self) -> ApproximateMclmcOptions {
        self.options
    }

    pub(crate) fn set_step_size(&mut self, step: StepSize) {
        self.options.mams.hmc =
            HmcOptions::new(step.value(), self.options.mams.integration_steps().get())
                .expect("validated options")
                .with_max_energy_error(self.options.mams.max_correction_error())
                .expect("validated cutoff");
    }

    pub(crate) fn attempted_steps(&self) -> usize {
        self.attempted_steps
    }

    /// Current coherent approximate-target cache.
    pub fn point(&self) -> &PointState<'a, T> {
        &self.live
    }

    /// Advances without MH correction. Backend failure publishes neither point
    /// nor persistent direction, while RNG consumption is not rolled back.
    pub fn step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<ApproximateMclmcTransition, MamsError<T::Error>> {
        self.attempted_steps = 0;
        self.proposal.clone_from(&self.live);
        self.proposal_direction.copy_from_slice(&self.direction);
        let initial_log_density = self.live.log_density();
        let mut discarded_correction = 0.0;
        let half_step = 0.5 * self.options.mams.step_size().value();
        for index in 1..=self.options.mams.integration_steps().get() {
            self.attempted_steps = index;
            if let Some(length) = self.options.refresh_length {
                angular_refresh(
                    &mut self.proposal_direction,
                    &mut self.noise,
                    half_step,
                    length,
                    rng,
                )
                .map_err(|_| MamsError::Numerical)?;
            }
            discarded_correction += isokinetic_kick(
                &mut self.proposal_direction,
                self.proposal.gradient(),
                half_step,
            )
            .ok_or(MamsError::Numerical)?;
            for ((next, position), direction) in self
                .next_position
                .iter_mut()
                .zip(self.proposal.position())
                .zip(self.proposal_direction.iter())
            {
                *next = position + self.options.mams.step_size().value() * direction;
            }
            self.proposal
                .try_update(&self.next_position, &mut self.evaluation)
                .map_err(|source| MamsError::Evaluation {
                    integration_step: index,
                    source,
                })?;
            discarded_correction += isokinetic_kick(
                &mut self.proposal_direction,
                self.proposal.gradient(),
                half_step,
            )
            .ok_or(MamsError::Numerical)?;
            if let Some(length) = self.options.refresh_length {
                angular_refresh(
                    &mut self.proposal_direction,
                    &mut self.noise,
                    half_step,
                    length,
                    rng,
                )
                .map_err(|_| MamsError::Numerical)?;
            }
        }
        let correction_error =
            initial_log_density - self.proposal.log_density() + discarded_correction;
        if !discarded_correction.is_finite() || !correction_error.is_finite() {
            return Err(MamsError::Numerical);
        }
        self.live.clone_from(&self.proposal);
        self.direction.copy_from_slice(&self.proposal_direction);
        Ok(ApproximateMclmcTransition {
            integration_steps: self.options.mams.integration_steps().get(),
            discarded_isokinetic_correction: discarded_correction,
            correction_error,
            guarantee: ApproximateTarget::acknowledge_bias(),
        })
    }
}

impl<T: LogDensityGradient + ?Sized> MarkovChain for ApproximateMclmc<'_, T> {
    type Error = MamsError<T::Error>;
    type Transition = ApproximateMclmcTransition;

    fn position(&self) -> &[f64] {
        self.live.position()
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        ApproximateMclmc::step(self, rng)
    }
}

fn sample_unit_direction<R: Rng + ?Sized>(direction: &mut [f64], rng: &mut R) -> Result<(), ()> {
    for value in direction.iter_mut() {
        *value = alea_math::random::standard_normal(rng);
    }
    normalize(direction)
}

fn angular_refresh<R: Rng + ?Sized>(
    direction: &mut [f64],
    noise: &mut [f64],
    half_step: f64,
    refresh_length: f64,
    rng: &mut R,
) -> Result<(), ()> {
    let variance = (2.0 * half_step / refresh_length).exp_m1() / direction.len() as f64;
    if !variance.is_finite() || variance < 0.0 {
        return Err(());
    }
    let scale = variance.sqrt();
    for value in noise.iter_mut() {
        *value = scale * alea_math::random::standard_normal(rng);
    }
    for (direction, noise) in direction.iter_mut().zip(noise.iter()) {
        *direction += noise;
    }
    normalize(direction)
}

/// Exact analytic score kick and log-Jacobian/kinetic correction. Returns None
/// on any nonfinite arithmetic. `score` is grad log pi, not grad potential.
fn isokinetic_kick(direction: &mut [f64], score: &[f64], step: f64) -> Option<f64> {
    debug_assert_eq!(direction.len(), score.len());
    let gradient_norm = scaled_norm(score);
    if gradient_norm == 0.0 {
        return Some(0.0);
    }
    if !gradient_norm.is_finite() {
        return None;
    }
    let mut projection = 0.0;
    for (&velocity, &gradient) in direction.iter().zip(score) {
        projection += velocity * (gradient / gradient_norm);
    }
    let delta = step * gradient_norm / (direction.len() as f64 - 1.0);
    if !delta.is_finite() || delta < 0.0 || !projection.is_finite() {
        return None;
    }
    let zeta = (-delta).exp();
    let one_minus = -(-delta).exp_m1();
    let along = one_minus * (1.0 + zeta + projection * one_minus);
    for (velocity, &gradient) in direction.iter_mut().zip(score) {
        *velocity = (gradient / gradient_norm) * along + 2.0 * zeta * *velocity;
    }
    normalize(direction).ok()?;
    let one_minus_zeta_squared = -(-2.0 * delta).exp_m1();
    // a / 2 = 1 + (projection - 1) (1 - exp(-2 delta)) / 2.
    // `ln_1p` avoids losing the small kick correction through subtraction.
    let relative = 0.5 * (projection - 1.0) * one_minus_zeta_squared;
    if !relative.is_finite() || relative <= -1.0 {
        return None;
    }
    let correction = (delta + relative.ln_1p()) * (direction.len() as f64 - 1.0);
    correction.is_finite().then_some(correction)
}

fn scaled_norm(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    scale
        * values
            .iter()
            .map(|value| (value / scale) * (value / scale))
            .sum::<f64>()
            .sqrt()
}

fn normalize(values: &mut [f64]) -> Result<(), ()> {
    let norm = scaled_norm(values);
    if !norm.is_finite() || norm == 0.0 {
        return Err(());
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alea_distributions::Gaussian;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn analytic_kick_preserves_unit_speed_and_reverse_correction() {
        let mut direction = [0.6, -0.8];
        let score = [0.7, -1.2];
        let forward = isokinetic_kick(&mut direction, &score, 0.15).unwrap();
        assert!((scaled_norm(&direction) - 1.0).abs() < 2e-15);
        for value in &mut direction {
            *value = -*value;
        }
        let reverse = isokinetic_kick(&mut direction, &score, 0.15).unwrap();
        for value in &mut direction {
            *value = -*value;
        }
        assert!((direction[0] - 0.6).abs() < 2e-14);
        assert!((direction[1] + 0.8).abs() < 2e-14);
        assert!((forward + reverse).abs() < 2e-14);
    }

    #[test]
    fn mams_uses_noncanonical_correction_and_keeps_gaussian_cache_coherent() {
        let target = Gaussian::new(3);
        let mut chain = Mams::new(
            &target,
            OwnedBuffer::new(3),
            MamsOptions::new(0.05, 4).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(91);
        for _ in 0..100 {
            let transition = chain.step(&mut rng).unwrap();
            assert!(transition.isokinetic_correction.is_some());
            assert!((0.0..=1.0).contains(&transition.acceptance_probability));
            let expected = -0.5 * chain.position().iter().map(|x| x * x).sum::<f64>();
            assert!((chain.point().log_density() - expected).abs() < 1e-13);
        }
    }

    #[test]
    fn unadjusted_kernel_requires_and_reports_approximate_marker() {
        let target = Gaussian::new(3);
        let mut rng = SmallRng::seed_from_u64(17);
        let mut chain = ApproximateMclmc::new(
            ApproximateTarget::acknowledge_bias(),
            &target,
            OwnedBuffer::new(3),
            ApproximateMclmcOptions::new(0.02, 2, Some(1.0)).unwrap(),
            &mut rng,
        )
        .unwrap();
        let transition = chain.step(&mut rng).unwrap();
        assert_eq!(transition.guarantee, ApproximateTarget::acknowledge_bias());
        assert!(transition.discarded_isokinetic_correction.is_finite());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "statistical workload runs natively; Miri covers bounded map contracts"
    )]
    fn mams_recovers_standard_gaussian_first_two_moments() {
        let target = Gaussian::new(3);
        let mut chain = Mams::new(
            &target,
            OwnedBuffer::from_fn(3, |i| [1.0, -1.0, 0.5][i]),
            MamsOptions::new(0.12, 8).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(1_937);
        let mut sum = [0.0; 3];
        let mut squares = [0.0; 3];
        for iteration in 0..22_000 {
            let _ = chain.step(&mut rng).unwrap();
            if iteration >= 2_000 {
                for ((sum, square), &value) in
                    sum.iter_mut().zip(squares.iter_mut()).zip(chain.position())
                {
                    *sum += value;
                    *square += value * value;
                }
            }
        }
        for (sum, square) in sum.into_iter().zip(squares) {
            let mean = sum / 20_000.0;
            let variance = square / 20_000.0 - mean * mean;
            assert!(mean.abs() < 0.06, "mean={mean}");
            assert!((variance - 1.0).abs() < 0.1, "variance={variance}");
        }
    }
}
