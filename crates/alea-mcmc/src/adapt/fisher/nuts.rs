//! Fisher estimators driven by native NUTS transitions, with a finite freeze.
use super::*;
use crate::adapt::{
    NutsWarmupError, NutsWarmupObservation, NutsWarmupReport, WarmupStage,
    find_reasonable_nuts_step_size,
};
use crate::{
    Nuts, NutsOptions,
    nuts::{NutsError, NutsTermination},
};

/// Original fitting and transition errors remain typed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FisherNutsWarmupError<E: Error + 'static> {
    /// Initial Fisher configuration or target failure.
    #[error(transparent)]
    Initialization(#[from] FisherWarmupError<E>),
    /// Native NUTS transition, step-search or controller failure.
    #[error(transparent)]
    Warmup(#[from] NutsWarmupError<E>),
    /// Paired moment or low-rank fit failure.
    #[error(transparent)]
    Fisher(#[from] FisherError),
}
impl<E: Error + 'static> From<NutsError<E>> for FisherNutsWarmupError<E> {
    fn from(error: NutsError<E>) -> Self {
        Self::Warmup(NutsWarmupError::Nuts(error))
    }
}

/// Frozen experimental metric and native NUTS kernel, without adaptation state.
pub type FisherNutsWarmupResult<'a, T, I = Leapfrog> = Result<
    (Nuts<'a, T, LowRankDiagonalMetric, I>, NutsWarmupReport),
    FisherNutsWarmupError<<T as alea_core::target::TargetSpace>::Error>,
>;

/// Owning Fisher warmup using tree-mean acceptance and paired live NUTS states.
/// Estimator mathematics and the 30%/85% switch/freeze schedule match the existing
/// Fisher HMC workflow; this is not whole-policy equivalence with nuts-rs.
#[derive(Debug)]
#[must_use = "finish adaptation before retaining samples"]
pub struct FisherNutsWarmup<'a, T: LogDensityGradient + ?Sized, I = Leapfrog> {
    inner: FisherHmcWarmup<'a, T, I>,
    depth: crate::config::TreeDepth,
}
impl<'a, T: LogDensityGradient + ?Sized> FisherNutsWarmup<'a, T> {
    /// Construct native Fisher/NUTS warmup with leapfrog.
    /// # Errors
    /// Propagates configuration, storage and initial target errors.
    pub fn new(
        target: &'a T,
        position: OwnedBuffer,
        nuts: NutsOptions,
        options: FisherOptions,
    ) -> Result<Self, FisherNutsWarmupError<T::Error>> {
        Self::new_with_integrator(target, position, nuts, options, Leapfrog)
    }
}
impl<'a, T: LogDensityGradient + ?Sized, I: SymmetricSplitting> FisherNutsWarmup<'a, T, I> {
    /// Construct with a fixed sealed symmetric integrator.
    /// # Errors
    /// Propagates the same initial-state errors as [`Self::new`].
    pub fn new_with_integrator(
        target: &'a T,
        position: OwnedBuffer,
        nuts: NutsOptions,
        options: FisherOptions,
        integrator: I,
    ) -> Result<Self, FisherNutsWarmupError<T::Error>> {
        let hmc = HmcOptions::new(nuts.step_size().value(), 1)
            .expect("validated step")
            .with_max_energy_error(nuts.max_energy_error())
            .expect("validated cutoff");
        Ok(Self {
            inner: FisherHmcWarmup::new_with_integrator(
                target, position, hmc, options, integrator,
            )?,
            depth: nuts.max_depth(),
        })
    }
    /// Consume the controller and return only a frozen native NUTS chain.
    /// # Errors
    /// Propagates typed backend, bounded search, fitting and controller failures.
    pub fn run<R: Rng + ?Sized>(self, rng: &mut R) -> FisherNutsWarmupResult<'a, T, I> {
        self.run_observed(rng, |_| {})
    }
    /// Observe coherent post-transition states before geometry updates. Fisher
    /// stages are slow until the 85% freeze, then final-fast; they are not the
    /// covariance window schedule. Observation borrows cannot outlive the callback.
    /// # Errors
    /// Same failures as [`Self::run`]; an observer panic publishes no chain.
    pub fn run_observed<R: Rng + ?Sized>(
        self,
        rng: &mut R,
        mut observe: impl FnMut(NutsWarmupObservation<'_>),
    ) -> FisherNutsWarmupResult<'a, T, I> {
        let FisherHmcWarmup {
            chain,
            mut adapter,
            initial_scales,
            mut scales,
            options,
            mut weighted,
        } = self.inner;
        let mut chain = Nuts::from_hmc(chain, self.depth)?;
        let initial = chain.point().log_density();
        let mut report = NutsWarmupReport {
            iterations: 0,
            moved: 0,
            divergences: 0,
            depth_saturations: 0,
            metric_updates: 0,
            search_probes: 0,
            integration_attempts: 0,
            bounded_step_updates: 0,
            step_size: chain.options().step_size(),
            initial_log_density: initial,
            final_log_density: initial,
            minimum_log_density: initial,
            maximum_log_density: initial,
        };
        let n = options.iterations;
        if n == 0 {
            return Ok((chain, report));
        }
        let search = find_reasonable_nuts_step_size(&mut chain, SearchOptions::default(), rng)?;
        chain.set_step_size(search.step_size);
        report.search_probes = search.probes;
        report.integration_attempts = search.probes;
        let mut controller =
            StepSizeController::new(search.step_size, options.target, options.step_adaptation);
        let switch = n / 10 * 3 + (n % 10) * 3 / 10;
        let freeze = n / 100 * 85 + (n % 100) * 85 / 100;
        for i in 0..n {
            if i == switch {
                adapter.reset(options.period)?;
            }
            if i == freeze {
                controller = StepSizeController::new(
                    chain.options().step_size(),
                    options.target,
                    options.step_adaptation,
                );
            }
            let transition = chain.step(rng)?;
            let step_size = transition.step_size;
            report.iterations += 1;
            report.moved += usize::from(transition.accepted);
            report.divergences += usize::from(matches!(
                transition.termination,
                NutsTermination::Divergence(_)
            ));
            report.depth_saturations +=
                usize::from(transition.termination == NutsTermination::MaxDepth);
            report.integration_attempts = report
                .integration_attempts
                .checked_add(transition.integration_steps)
                .ok_or(NutsWarmupError::CounterOverflow)?;
            report.final_log_density = chain.point().log_density();
            report.minimum_log_density = report.minimum_log_density.min(report.final_log_density);
            report.maximum_log_density = report.maximum_log_density.max(report.final_log_density);
            observe(NutsWarmupObservation {
                completed: i + 1,
                position: chain.point().position(),
                log_density: report.final_log_density,
                step_size,
                transition,
                stage: if i < freeze {
                    WarmupStage::Slow {
                        window_end: (i + 1 == freeze)
                            || adapter.seen.is_multiple_of(adapter.period),
                    }
                } else {
                    WarmupStage::FinalFast
                },
            });
            if i < freeze {
                if let Some(weighted) = &mut weighted {
                    weighted.observe(chain.point().position(), chain.point().gradient())?;
                    weighted.scales_into(&initial_scales, 1e-5, &mut scales)?;
                } else {
                    adapter.observe(chain.point().position(), chain.point().gradient())?;
                    adapter.scales_into(&initial_scales, 1e-5, &mut scales)?;
                }
                chain
                    .metric_mut()
                    .set_scales(&scales)
                    .map_err(FisherError::from)?;
                #[cfg(feature = "faer")]
                if options.max_rank > 0
                    && (adapter.seen.is_multiple_of(adapter.period) || i + 1 == freeze)
                    && adapter.retained >= 2
                {
                    chain.replace_metric(adapter.fit_low_rank(
                        &scales,
                        1e-5,
                        2.0,
                        options.max_rank,
                    )?)?;
                }
                report.metric_updates += 1;
            }
            // A late failed edge contributes zero to the tree average; it does
            // not erase valid earlier edges or become an HMC-style zero statistic.
            let step = controller
                .update(StepObservation::Acceptance(
                    transition.acceptance_probability,
                ))
                .map_err(NutsWarmupError::from)?;
            let value = step.value().clamp(1e-12, 1e6);
            report.bounded_step_updates += usize::from(value != step.value());
            chain.set_step_size(value.try_into().expect("bounded step"));
        }
        let step = controller.finish();
        let value = step.value().clamp(1e-12, 1e6);
        report.bounded_step_updates += usize::from(value != step.value());
        chain.set_step_size(value.try_into().expect("bounded step"));
        report.step_size = chain.options().step_size();
        Ok((chain, report))
    }
}
