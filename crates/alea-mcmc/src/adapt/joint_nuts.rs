//! Native NUTS joint geometry/integrator adaptation with a final frozen fast stage.
use super::{
    AdaptationError, CovarianceError, DualAveraging, JointWarmupOptions, NutsWarmup,
    NutsWarmupError, WarmupMetric, WarmupStage, WindowSchedule,
};
use crate::{
    Nuts, NutsOptions,
    atlas::{CurvatureProbeError, probe_metric_curvature},
    config::StepSize,
    integrator::{AdvancedIntegratorError, CurvatureBound, TwoStage, select_saia_two_stage},
    nuts::{NutsError, NutsTermination},
};
use alea_core::capability::PotentialHessianVector;
use alea_math::buffer::OwnedBuffer;
use rand::Rng;
use std::error::Error;

/// Failures preserve concrete target errors; no partly adapted chain escapes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JointNutsWarmupError<E: Error + 'static> {
    /// Initial chain, covariance, search or controller failure.
    #[error(transparent)]
    Warmup(#[from] NutsWarmupError<E>),
    /// A native NUTS transition failed.
    #[error(transparent)]
    Nuts(#[from] NutsError<E>),
    /// Covariance fit failed.
    #[error(transparent)]
    Covariance(#[from] CovarianceError),
    /// Step controller failed.
    #[error(transparent)]
    Adaptation(#[from] AdaptationError),
    /// Warmup curvature estimation failed.
    #[error(transparent)]
    Curvature(#[from] CurvatureProbeError<E>),
    /// The inflated oscillator bound is unrepresentable.
    #[error(transparent)]
    SymmetricSplitting(#[from] AdvancedIntegratorError),
    /// Work counters overflowed.
    #[error("joint NUTS warmup counter overflow")]
    CounterOverflow,
}
/// Frozen coefficient and the actual warmup costs; an observed curvature policy
/// is not a certified bound on a nonlinear target's global spectrum.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct JointNutsWarmupReport {
    /// Completed native NUTS transitions.
    pub iterations: usize,
    /// Numerical tree terminations.
    pub divergences: usize,
    /// Installed metric fits.
    pub metric_updates: usize,
    /// Initial and slow-window coefficient selections.
    pub coefficient_updates: usize,
    /// Completed transitions after the last coefficient selection.
    pub final_fast_iterations: usize,
    /// Search proposals.
    pub search_probes: usize,
    /// Trajectory and search macrosteps.
    pub integration_attempts: usize,
    /// Actual calls to the curvature HVP capability.
    pub hvp_calls: usize,
    /// Warmup steps capped by the explicit bounds or oscillator policy.
    pub bounded_step_updates: usize,
    /// Frozen step size.
    pub step_size: StepSize,
    /// Frozen two-stage coefficient.
    pub two_stage_coefficient: f64,
    /// Last observed curvature after explicit inflation.
    pub curvature_upper_bound: f64,
    /// Last selection interval, capped at3.8.
    pub selection_dimensionless_limit: f64,
}
/// A completely frozen native NUTS chain and adaptation report.
pub type JointNutsWarmupResult<'a, T> = Result<
    (Nuts<'a, T, WarmupMetric, TwoStage>, JointNutsWarmupReport),
    JointNutsWarmupError<<T as alea_core::target::TargetSpace>::Error>,
>;

/// Owning warmup; coefficients change only initially and at slow-window ends.
/// Final fast adaptation therefore tunes the actual retained integrator.
#[derive(Debug)]
#[must_use = "joint warmup must finish before retained sampling"]
pub struct JointNutsWarmup<'a, T: PotentialHessianVector + ?Sized> {
    inner: NutsWarmup<'a, T, TwoStage>,
    options: JointWarmupOptions,
}
impl<'a, T: PotentialHessianVector + ?Sized> JointNutsWarmup<'a, T> {
    /// Construct with identity initial geometry and OMF two-stage dynamics.
    /// # Errors
    /// Returns typed target, options or storage errors.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        nuts: NutsOptions,
        options: JointWarmupOptions,
    ) -> Result<Self, JointNutsWarmupError<T::Error>> {
        Ok(Self {
            inner: NutsWarmup::new_with_integrator(
                target,
                position,
                nuts,
                options.base,
                TwoStage::OMF,
            )?,
            options,
        })
    }
    /// Consume all adaptation. Zero iterations consume no RNG or curvature work.
    /// # Errors
    /// Returns typed kernel, curvature, covariance, controller or counter errors.
    pub fn run<R: Rng + ?Sized>(mut self, rng: &mut R) -> JointNutsWarmupResult<'a, T> {
        let mut report = JointNutsWarmupReport {
            iterations: 0,
            divergences: 0,
            metric_updates: 0,
            coefficient_updates: 0,
            final_fast_iterations: 0,
            search_probes: 0,
            integration_attempts: 0,
            hvp_calls: 0,
            bounded_step_updates: 0,
            step_size: self.inner.chain.options().step_size(),
            two_stage_coefficient: TwoStage::OMF.lambda(),
            curvature_upper_bound: 0.,
            selection_dimensionless_limit: 0.,
        };
        if self.options.base.iterations == 0 {
            return Ok((self.inner.chain, report));
        }
        // Measure curvature before search so even the initial search has a valid
        // finite oscillator interval. Search/selection remain warmup-only policies.
        self.epoch(rng, &mut report)?;
        let mut adapter = DualAveraging::new(
            self.inner.chain.options().step_size(),
            self.options.base.target,
        );
        for stage in WindowSchedule::new(self.options.base.iterations) {
            let transition = self.inner.chain.step(rng)?;
            report.iterations += 1;
            report.divergences += usize::from(matches!(
                transition.termination,
                NutsTermination::Divergence(_)
            ));
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(JointNutsWarmupError::CounterOverflow)?;
            if stage == WarmupStage::FinalFast {
                report.final_fast_iterations += 1;
            }
            let step = adapter.update(transition.acceptance_probability)?;
            if let WarmupStage::Slow { window_end } = stage {
                let covariance = self
                    .inner
                    .covariance
                    .as_mut()
                    .expect("slow stage has covariance");
                covariance.update(self.inner.chain.point().position())?;
                if window_end {
                    let metric = covariance.metric()?;
                    covariance.reset();
                    self.inner.chain.replace_metric(metric)?;
                    self.inner.chain.set_step_size(adapter.finish());
                    self.epoch(rng, &mut report)?;
                    adapter = DualAveraging::new(
                        self.inner.chain.options().step_size(),
                        self.options.base.target,
                    );
                    report.metric_updates += 1;
                    continue;
                }
            }
            let step = self.bound(step, &mut report);
            self.inner.chain.set_step_size(step);
        }
        let step = self.bound(adapter.finish(), &mut report);
        self.inner.chain.set_step_size(step);
        report.step_size = step;
        Ok((self.inner.chain, report))
    }
    fn bound(&self, step: StepSize, report: &mut JointNutsWarmupReport) -> StepSize {
        let maximum = (3.8 / report.curvature_upper_bound.sqrt()).min(1e6);
        let value = step.value().clamp(f64::MIN_POSITIVE, maximum);
        report.bounded_step_updates += usize::from(value != step.value());
        value.try_into().expect("bounded step")
    }
    fn probe<R: Rng + ?Sized>(
        &mut self,
        step: f64,
        rng: &mut R,
        report: &mut JointNutsWarmupReport,
    ) -> Result<f64, JointNutsWarmupError<T::Error>> {
        let acceptance = self
            .inner
            .chain
            .probe(step.try_into().expect("bounded step"), rng)?;
        report.search_probes = report
            .search_probes
            .checked_add(1)
            .ok_or(JointNutsWarmupError::CounterOverflow)?;
        report.integration_attempts = report
            .integration_attempts
            .checked_add(1)
            .ok_or(JointNutsWarmupError::CounterOverflow)?;
        Ok(acceptance)
    }
    fn epoch<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        report: &mut JointNutsWarmupReport,
    ) -> Result<(), JointNutsWarmupError<T::Error>> {
        let probe = probe_metric_curvature(
            self.inner.chain.point(),
            self.inner.chain.metric(),
            self.options.curvature_probe,
            rng,
        )?;
        report.hvp_calls = report
            .hvp_calls
            .checked_add(probe.hvp_count)
            .ok_or(JointNutsWarmupError::CounterOverflow)?;
        let bound =
            CurvatureBound::new(probe.curvature_estimate, self.options.curvature_inflation)?;
        report.curvature_upper_bound = bound.upper();
        let step = self.bound(self.inner.chain.options().step_size(), report);
        self.inner.chain.set_step_size(step);
        if let Some(mut search) = self.options.base.search {
            search.max = (3.8 / bound.upper().sqrt())
                .min(search.max.value())
                .try_into()
                .map_err(|_| AdvancedIntegratorError::Curvature)?;
            // A curvature cap can be below the default numerical search floor.
            if search.min.value() >= search.max.value() {
                search.min = (search.max.value() * 1e-6)
                    .try_into()
                    .map_err(|_| AdvancedIntegratorError::Curvature)?;
            }
            let mut step = self
                .inner
                .chain
                .options()
                .step_size()
                .value()
                .clamp(search.min.value(), search.max.value());
            let acceptable = self.probe(step, rng, report)? >= search.target.value();
            let mut selected = None;
            for _ in 1..search.max_probes {
                let candidate = (if acceptable { step * 2. } else { step * 0.5 })
                    .clamp(search.min.value(), search.max.value());
                if candidate == step {
                    if acceptable {
                        selected = Some(step);
                        report.bounded_step_updates += 1;
                    }
                    break;
                }
                let candidate_ok = self.probe(candidate, rng, report)? >= search.target.value();
                if candidate_ok != acceptable {
                    selected = Some(if acceptable { step } else { candidate });
                    break;
                }
                step = candidate;
            }
            let step = selected.ok_or(JointNutsWarmupError::Warmup(
                NutsWarmupError::SearchExhausted,
            ))?;
            self.inner
                .chain
                .set_step_size(step.try_into().expect("bounded search step"));
        }
        let limit = bound.dimensionless_step(self.inner.chain.options().step_size().value())?;
        let selection = select_saia_two_stage(limit)?;
        *self.inner.chain.integrator_mut() = selection.integrator;
        report.coefficient_updates += 1;
        report.two_stage_coefficient = selection.integrator.lambda();
        report.selection_dimensionless_limit = limit;
        Ok(())
    }
}
