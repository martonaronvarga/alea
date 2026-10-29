//! Epoch-bounded joint metric, step-size and two-stage-coefficient warmup.

use std::error::Error;

use alea_core::capability::PotentialHessianVector;
use alea_math::buffer::OwnedBuffer;
use rand::Rng;

use super::{
    AdaptationError, CovarianceError, DualAveraging, HmcWarmup, WarmupError, WarmupMetric,
    WarmupOptions, WarmupStage, WindowSchedule, find_reasonable_step_size,
};
use crate::{
    Hmc, HmcOptions,
    atlas::{CurvatureProbeError, CurvatureProbeOptions, probe_metric_curvature},
    config::StepSize,
    hmc::HmcError,
    integrator::{AdvancedIntegratorError, CurvatureBound, TwoStage, select_saia_two_stage},
};

/// Joint warmup policy. Coefficients change only at epoch boundaries and all
/// adaptation state is consumed before the returned chain can sample.
#[derive(Debug, Clone, Copy)]
pub struct JointWarmupOptions {
    pub(super) base: WarmupOptions,
    pub(super) curvature_probe: CurvatureProbeOptions,
    pub(super) curvature_inflation: f64,
}

impl JointWarmupOptions {
    pub fn new(
        base: WarmupOptions,
        curvature_probe: CurvatureProbeOptions,
        curvature_inflation: f64,
    ) -> Result<Self, AdvancedIntegratorError> {
        // Validate independently of an observed estimate.
        if !curvature_inflation.is_finite() || curvature_inflation <= 1.0 {
            return Err(AdvancedIntegratorError::Curvature);
        }
        Ok(Self {
            base,
            curvature_probe,
            curvature_inflation,
        })
    }
}

/// Warmup-only derivative and freeze evidence.
#[derive(Debug, Clone, Copy)]
pub struct JointWarmupReport {
    pub iterations: usize,
    pub accepted: usize,
    pub divergences: usize,
    pub metric_updates: usize,
    pub coefficient_updates: usize,
    pub search_probes: usize,
    pub integration_attempts: usize,
    pub hvp_calls: usize,
    pub step_size: StepSize,
    pub two_stage_coefficient: f64,
    pub curvature_upper_bound: f64,
    pub dimensionless_limit: f64,
}

/// Joint warmup errors preserve the original target/backend type.
#[derive(Debug, thiserror::Error)]
pub enum JointWarmupError<E: Error + 'static> {
    #[error(transparent)]
    Hmc(#[from] HmcError<E>),
    #[error(transparent)]
    Covariance(#[from] CovarianceError),
    #[error(transparent)]
    Adaptation(#[from] AdaptationError),
    #[error(transparent)]
    Curvature(#[from] CurvatureProbeError<E>),
    #[error(transparent)]
    SymmetricSplitting(#[from] AdvancedIntegratorError),
    #[error("step-size search did not bracket acceptance within its limits")]
    SearchExhausted,
    #[error("joint warmup counter overflow")]
    CounterOverflow,
}

impl<E: Error + 'static> From<WarmupError<E>> for JointWarmupError<E> {
    fn from(error: WarmupError<E>) -> Self {
        match error {
            WarmupError::Hmc(error) => Self::Hmc(error),
            WarmupError::Covariance(error) => Self::Covariance(error),
            WarmupError::Adaptation(error) => Self::Adaptation(error),
            WarmupError::SearchExhausted => Self::SearchExhausted,
            WarmupError::CounterOverflow => Self::CounterOverflow,
        }
    }
}

pub type JointWarmupResult<'a, T> = Result<
    (Hmc<'a, T, WarmupMetric, TwoStage>, JointWarmupReport),
    JointWarmupError<<T as alea_core::target::TargetSpace>::Error>,
>;

/// Owning joint controller; it deliberately does not implement `MarkovChain`.
#[derive(Debug)]
#[must_use = "joint warmup must finish before retained sampling"]
pub struct JointHmcWarmup<'a, T: PotentialHessianVector + ?Sized> {
    inner: HmcWarmup<'a, T, TwoStage>,
    options: JointWarmupOptions,
}

impl<'a, T: PotentialHessianVector + ?Sized> JointHmcWarmup<'a, T> {
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        hmc: HmcOptions,
        options: JointWarmupOptions,
    ) -> Result<Self, JointWarmupError<T::Error>> {
        let inner =
            HmcWarmup::new_with_integrator(target, position, hmc, options.base, TwoStage::OMF)?;
        Ok(Self { inner, options })
    }

    /// Consume all estimators/controllers and publish one fully frozen kernel.
    pub fn run<R: Rng + ?Sized>(self, rng: &mut R) -> JointWarmupResult<'a, T> {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!(target: "alea::adaptation", "joint_hmc_warmup", guarantee = "method_specific").entered();
        #[cfg(feature = "tracing")]
        tracing::debug!(target: "alea::adaptation", schema_version = 1_u64, event = "started");
        let outcome = self.run_inner(rng);
        #[cfg(feature = "tracing")]
        match &outcome {
            Ok(_) => {
                tracing::debug!(target: "alea::adaptation", schema_version = 1_u64, event = "finished")
            }
            Err(_) => {
                tracing::error!(target: "alea::adaptation", schema_version = 1_u64, event = "failed")
            }
        }
        outcome
    }
    fn run_inner<R: Rng + ?Sized>(mut self, rng: &mut R) -> JointWarmupResult<'a, T> {
        let mut report = JointWarmupReport {
            iterations: self.options.base.iterations,
            accepted: 0,
            divergences: 0,
            metric_updates: 0,
            coefficient_updates: 0,
            search_probes: 0,
            integration_attempts: 0,
            hvp_calls: 0,
            step_size: self.inner.chain.options().step_size(),
            two_stage_coefficient: TwoStage::OMF.lambda(),
            curvature_upper_bound: 0.0,
            dimensionless_limit: 0.0,
        };
        if self.options.base.iterations == 0 {
            return Ok((self.inner.chain, report));
        }
        self.search(rng, &mut report)?;
        self.select_coefficient(rng, &mut report)?;
        let mut adapter = DualAveraging::new(
            self.inner.chain.options().step_size(),
            self.options.base.target,
        );
        for stage in WindowSchedule::new(self.options.base.iterations) {
            let transition = self.inner.chain.step(rng)?;
            report.accepted += usize::from(transition.accepted);
            report.divergences += usize::from(transition.divergence.is_some());
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(JointWarmupError::CounterOverflow)?;
            let step = adapter.update(transition.acceptance_probability)?;
            if let WarmupStage::Slow { window_end } = stage {
                let covariance = self
                    .inner
                    .covariance
                    .as_mut()
                    .expect("slow schedule has covariance storage");
                covariance.update(self.inner.chain.point().position())?;
                if window_end {
                    let metric = covariance.metric()?;
                    covariance.reset();
                    self.inner.chain.replace_metric(metric)?;
                    self.inner.chain.set_step_size(adapter.finish());
                    self.search(rng, &mut report)?;
                    self.select_coefficient(rng, &mut report)?;
                    adapter = DualAveraging::new(
                        self.inner.chain.options().step_size(),
                        self.options.base.target,
                    );
                    report.metric_updates += 1;
                    continue;
                }
            }
            self.inner.chain.set_step_size(step);
        }
        self.inner.chain.set_step_size(adapter.finish());
        // The final selection is based on the actual frozen step and geometry.
        self.select_coefficient(rng, &mut report)?;
        report.step_size = self.inner.chain.options().step_size();
        report.two_stage_coefficient = self.inner.chain.integrator_mut().lambda();
        Ok((self.inner.chain, report))
    }

    fn search<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        report: &mut JointWarmupReport,
    ) -> Result<(), JointWarmupError<T::Error>> {
        if let Some(options) = self.options.base.search {
            let result = find_reasonable_step_size(&mut self.inner.chain, options, rng)?;
            report.search_probes = report
                .search_probes
                .checked_add(result.probes)
                .ok_or(JointWarmupError::CounterOverflow)?;
            report.integration_attempts = report
                .integration_attempts
                .checked_add(result.probes)
                .ok_or(JointWarmupError::CounterOverflow)?;
            self.inner.chain.set_step_size(result.step_size);
        }
        Ok(())
    }

    fn select_coefficient<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        report: &mut JointWarmupReport,
    ) -> Result<(), JointWarmupError<T::Error>> {
        let probe = probe_metric_curvature(
            self.inner.chain.point(),
            self.inner.chain.metric(),
            self.options.curvature_probe,
            rng,
        )?;
        report.hvp_calls = report
            .hvp_calls
            .checked_add(probe.hvp_count)
            .ok_or(JointWarmupError::CounterOverflow)?;
        let bound =
            CurvatureBound::new(probe.curvature_estimate, self.options.curvature_inflation)?;
        let limit = bound.dimensionless_step(self.inner.chain.options().step_size().value())?;
        let selection = select_saia_two_stage(limit)?;
        *self.inner.chain.integrator_mut() = selection.integrator;
        report.coefficient_updates = report
            .coefficient_updates
            .checked_add(1)
            .ok_or(JointWarmupError::CounterOverflow)?;
        report.curvature_upper_bound = bound.upper();
        report.dimensionless_limit = limit;
        report.two_stage_coefficient = selection.integrator.lambda();
        Ok(())
    }
}
