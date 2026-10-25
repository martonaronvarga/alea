//! Conventional paired diagonal geometry; all learning ends before retention.
use super::*;
use alea_math::metric::InverseDiagonalMetric;

#[derive(Debug)]
pub(super) struct Estimate {
    q: OnlineCovariance,
    score: OnlineCovariance,
    background_q: OnlineCovariance,
    background_score: OnlineCovariance,
    inverse: OwnedBuffer,
    metric: Option<InverseDiagonalMetric>,
}

impl Estimate {
    pub(super) fn new(d: usize) -> Result<Self, CovarianceError> {
        if d.checked_mul(32)
            .is_none_or(|n| n > isize::MAX as usize / size_of::<f64>())
        {
            return Err(CovarianceError::Dimension);
        }
        let kind = super::super::MetricKind::Diagonal;
        Ok(Self {
            q: OnlineCovariance::new(d, kind)?,
            score: OnlineCovariance::new(d, kind)?,
            background_q: OnlineCovariance::new(d, kind)?,
            background_score: OnlineCovariance::new(d, kind)?,
            inverse: OwnedBuffer::from_fn(d, |_| 1.0),
            metric: Some(InverseDiagonalMetric::new(OwnedBuffer::from_fn(d, |_| {
                1.0
            }))?),
        })
    }
    fn observe(&mut self, q: &[f64], score: &[f64]) -> Result<(), CovarianceError> {
        // A failed controller is consumed; no partly learned metric escapes.
        self.q.update(q)?;
        self.score.update(score)?;
        self.background_q.update(q)?;
        self.background_score.update(score)
    }
    fn switch(&mut self) {
        std::mem::swap(&mut self.q, &mut self.background_q);
        std::mem::swap(&mut self.score, &mut self.background_score);
        self.background_q.reset();
        self.background_score.reset();
    }
    fn fit(&mut self) -> bool {
        if self.q.count() < 3 {
            return false;
        }
        for ((out, q), score) in self
            .inverse
            .iter_mut()
            .zip(self.q.scatter())
            .zip(self.score.scatter())
        {
            if *q > 0.0 && *score > 0.0 {
                // Counts cancel. Separate square roots avoid overflowing q/score.
                let value = q.sqrt() / score.sqrt();
                if value.is_finite() && value > 0.0 {
                    *out = value.clamp(1e-20, 1e20);
                }
            }
        }
        true
    }
}

impl<'a, T: LogDensityGradient + ?Sized, I: Integrator> NutsWarmup<'a, T, I> {
    pub(super) fn drive_gradient<R: Rng + ?Sized>(
        mut self,
        mut estimate: Estimate,
        rng: &mut R,
        mut proceed: impl FnMut(usize) -> bool,
        mut observe: impl FnMut(NutsWarmupObservation<'_>),
        mut report: NutsWarmupReport,
    ) -> ControlledNutsWarmupResult<'a, T, I> {
        for (out, score) in estimate
            .inverse
            .iter_mut()
            .zip(self.chain.point().gradient())
        {
            // A small or zero score is not evidence for enormous initial variance.
            *out = 1.0 / score.abs().clamp(1.0, 1e20);
        }
        estimate.observe(self.chain.point().position(), self.chain.point().gradient())?;
        let mut metric = estimate
            .metric
            .take()
            .expect("preallocated gradient metric");
        metric
            .set_inverse_masses(&estimate.inverse)
            .map_err(CovarianceError::from)?;
        self.chain
            .replace_metric(WarmupMetric::InverseDiagonal(metric))?;
        self.search(rng, &mut report)?;
        let mut adapter = DualAveraging::new(self.chain.options().step_size(), self.options.target);
        let n = self.options.iterations;
        let early_end = n / 10 * 3 + n % 10 * 3 / 10;
        let freeze = n / 100 * 85 + n % 100 * 85 / 100;
        let mut window = 80usize;
        let mut initial_fit = true;
        let bounds = self.options.search.unwrap_or_default();
        let cap = std::f64::consts::PI
            .min(bounds.max.value())
            .max(bounds.min.value())
            .try_into()
            .expect("bounded positive step cap");
        for i in 0..n {
            if !proceed(report.iterations) {
                return Ok(None);
            }
            // Geometry changed throughout the preceding phase. Tune afresh on
            // the frozen metric so its retained step does not average obsolete
            // geometry, following the reference covariance controller's final
            // fixed-geometry restart.
            if self.gradient_final_restart && i == freeze && i > 0 {
                self.search(rng, &mut report)?;
                adapter = DualAveraging::new(self.chain.options().step_size(), self.options.target);
            }
            let early = i < early_end;
            if i == early_end {
                window = window.max(estimate.background_q.count());
            }
            let period = if early { 10 } else { window };
            let next_window = if early {
                10
            } else {
                window
                    .saturating_add(window.div_ceil(2))
                    .max(window.saturating_add(1))
            };
            let late = i >= freeze || next_window.saturating_add(i) > freeze;
            // The final stage targets the ordinary tree acceptance on a fixed
            // metric, as reference covariance warmup does. Symmetric tuning is
            // confined to the late phase with changing geometry.
            let symmetric_late = late && (i < freeze || !self.gradient_final_restart);
            let (transition, symmetric) = if symmetric_late {
                self.chain.step_with_symmetric_acceptance(rng)?
            } else {
                (self.chain.step(rng)?, 0.0)
            };
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
            report.final_log_density = self.chain.point().log_density();
            report.minimum_log_density = report.minimum_log_density.min(report.final_log_density);
            report.maximum_log_density = report.maximum_log_density.max(report.final_log_density);
            let switch =
                i < freeze && estimate.background_q.count().saturating_add(1) >= period && !late;
            observe(NutsWarmupObservation {
                completed: report.iterations,
                position: self.chain.point().position(),
                log_density: report.final_log_density,
                step_size,
                transition,
                stage: if i < freeze {
                    WarmupStage::Slow {
                        window_end: switch || i + 1 == freeze,
                    }
                } else {
                    WarmupStage::FinalFast
                },
            });
            if !proceed(report.iterations) {
                return Ok(None);
            }
            if i < freeze {
                estimate.observe(self.chain.point().position(), self.chain.point().gradient())?;
                if switch {
                    estimate.switch();
                    if !early {
                        window = next_window;
                    }
                }
                if estimate.fit() {
                    let WarmupMetric::InverseDiagonal(metric) = self.chain.metric_mut() else {
                        unreachable!("gradient profile owns diagonal metric")
                    };
                    metric
                        .set_inverse_masses(&estimate.inverse)
                        .map_err(CovarianceError::from)?;
                    report.metric_updates += 1;
                    if initial_fit {
                        initial_fit = false;
                        self.search(rng, &mut report)?;
                        adapter = DualAveraging::new(
                            self.chain.options().step_size(),
                            self.options.target,
                        );
                        continue;
                    }
                }
            }
            let step = adapter.update_capped(
                if symmetric_late {
                    symmetric
                } else {
                    transition.acceptance_probability
                },
                cap,
            )?;
            let step = self.bound(step, &mut report);
            self.chain.set_step_size(step);
        }
        let step = self.bound(adapter.finish(), &mut report);
        self.chain.set_step_size(step);
        report.step_size = step;
        Ok(Some((self.chain, report)))
    }
}

#[cfg(test)]
mod tests {
    use super::Estimate;

    #[test]
    fn paired_batch_estimate_counts_repeats_and_switches_both_moments_together() {
        let mut estimate = Estimate::new(2).unwrap();
        for q in [[2., -4.], [2., -4.], [4., 0.], [4., 0.], [6., 4.]] {
            estimate.observe(&q, &[-q[0] / 4., -q[1] / 16.]).unwrap();
        }
        assert_eq!(estimate.q.count(), 5);
        assert!(estimate.fit());
        assert_eq!(&*estimate.inverse, &[4., 16.]);
        estimate.switch();
        for q in [[100., 5.], [102., 9.], [104., 13.]] {
            estimate.observe(&q, &[-q[0] / 9., -q[1] / 25.]).unwrap();
        }
        assert_eq!(estimate.q.count(), 8);
        assert_eq!(estimate.background_q.count(), 3);
        estimate.switch();
        assert_eq!(estimate.q.count(), 3);
        assert_eq!(estimate.score.count(), 3);
        assert_eq!(estimate.background_q.count(), 0);
        assert_eq!(estimate.background_score.count(), 0);
        assert!(estimate.fit());
        for (&actual, expected) in estimate.inverse.iter().zip([9., 25.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
