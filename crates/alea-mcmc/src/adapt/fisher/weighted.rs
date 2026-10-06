//! Normalized paired weighted central moments; no unweighted history approximation.
use super::{FisherError, log_add_positive};
use crate::adapt::DiminishingSchedule;
use alea_math::buffer::OwnedBuffer;

/// Diagonal paired moments under identical normalized weights, warmup-only.
/// The first sample has weight one. Subsequent alpha values come from the
/// independent diminishing schedule. Updates allocate nothing and commit atomically.
#[derive(Debug)]
pub struct WeightedFisherMoments {
    dim: usize,
    count: usize,
    squared_weights: f64,
    schedule: DiminishingSchedule,
    values: OwnedBuffer,
    next: OwnedBuffer,
}
impl WeightedFisherMoments {
    /// # Errors
    /// Rejects zero dimension and storage arithmetic overflow.
    pub fn new(dim: usize, schedule: DiminishingSchedule) -> Result<Self, FisherError> {
        let length = dim
            .checked_mul(4)
            .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
            .ok_or(FisherError::Configuration)?;
        if dim == 0 {
            return Err(FisherError::Configuration);
        }
        Ok(Self {
            dim,
            count: 0,
            squared_weights: 0.0,
            schedule,
            values: OwnedBuffer::new(length),
            next: OwnedBuffer::new(length),
        })
    }
    /// Number of successfully incorporated observations.
    pub fn count(&self) -> usize {
        self.count
    }
    /// Sum of squared normalized observation weights (zero before observations).
    pub fn squared_weights(&self) -> f64 {
        self.squared_weights
    }
    /// Position and score means followed by their normalized central scatters.
    pub fn moments(&self) -> &[f64] {
        &self.values
    }
    /// # Errors
    /// Shape/nonfinite input, count overflow or unrepresentable moments preserve
    /// all committed values, weights and count.
    pub fn observe(&mut self, position: &[f64], score: &[f64]) -> Result<(), FisherError> {
        if position.len() != self.dim || score.len() != self.dim {
            return Err(FisherError::Configuration);
        }
        if position.iter().chain(score).any(|x| !x.is_finite()) {
            return Err(FisherError::Numerical);
        }
        let count = self
            .count
            .checked_add(1)
            .ok_or(FisherError::CountOverflow)?;
        let alpha = if self.count == 0 {
            1.0
        } else {
            self.schedule.weight(self.count)
        };
        let keep = 1.0 - alpha;
        for (axis, signal) in [position, score].into_iter().enumerate() {
            for (i, &x) in signal.iter().enumerate() {
                let mean = axis * self.dim + i;
                let scatter = mean + 2 * self.dim;
                if self.count == 0 {
                    self.next[mean] = x;
                    self.next[scatter] = 0.0;
                } else {
                    let delta = x - self.values[mean];
                    self.next[mean] = self.values[mean] + alpha * delta;
                    self.next[scatter] =
                        keep * self.values[scatter] + (keep * delta) * (alpha * delta);
                }
            }
        }
        let squared_weights = keep * keep * self.squared_weights + alpha * alpha;
        if self.next.iter().any(|x| !x.is_finite()) || !squared_weights.is_finite() {
            return Err(FisherError::Numerical);
        }
        std::mem::swap(&mut self.values, &mut self.next);
        self.count = count;
        self.squared_weights = squared_weights;
        Ok(())
    }
    /// Coordinate scales with ridge on NORMALIZED weighted scatters in fallback
    /// coordinates. No 1/(1-sum(w^2)) correction is applied to either scatter or
    /// ridge. This is deliberately different from raw-window scatter regularization.
    /// # Errors
    /// Invalid shapes/ridge/fallback return an error; output may be partly written.
    pub fn scales_into(
        &self,
        fallback: &[f64],
        ridge: f64,
        out: &mut [f64],
    ) -> Result<(), FisherError> {
        if fallback.len() != self.dim || out.len() != self.dim || !ridge.is_finite() || ridge <= 0.0
        {
            return Err(FisherError::Configuration);
        }
        for i in 0..self.dim {
            let scale = fallback[i];
            if !scale.is_finite() || !(1e-10..=1e10).contains(&scale) {
                return Err(FisherError::Configuration);
            }
            let log_scale = scale.ln();
            let c = log_add_positive(
                self.values[2 * self.dim + i].ln() - 2.0 * log_scale,
                ridge.ln(),
            );
            let f = log_add_positive(
                self.values[3 * self.dim + i].ln() + 2.0 * log_scale,
                ridge.ln(),
            );
            out[i] = (log_scale + 0.25 * (c - f))
                .clamp(1e-10_f64.ln(), 1e10_f64.ln())
                .exp()
                .clamp(1e-10, 1e10);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_extreme_observation_and_count_overflow_are_transactional() {
        let mut m =
            WeightedFisherMoments::new(1, DiminishingSchedule::new(1.0, 0.75).unwrap()).unwrap();
        m.observe(&[f64::MAX], &[-f64::MAX]).unwrap();
        assert_eq!(m.moments(), [f64::MAX, -f64::MAX, 0.0, 0.0]);
        assert!(m.observe(&[-f64::MAX], &[f64::MAX]).is_err());
        assert_eq!(m.count(), 1);
        assert_eq!(m.squared_weights(), 1.0);
        m.count = usize::MAX;
        assert!(matches!(
            m.observe(&[0.0], &[0.0]),
            Err(FisherError::CountOverflow)
        ));
        assert_eq!(m.moments(), [f64::MAX, -f64::MAX, 0.0, 0.0]);
    }
}
