use super::*;

/// Which absorbing boundary terminated evidence accumulation.
///
/// Convention: `Upper` ≡ correct response (drift favours upper boundary
/// when `v > 0`); `Lower` ≡ error response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Process hit the upper boundary at `a`.
    Upper,
    /// Process hit the lower boundary at `0`.
    Lower,
}

/// Wiener type data for scalar interfaces
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WienerObservation {
    pub rt: f64,
    pub boundary: Boundary,
}

/// Batch evaluation strategy for `WienerObservations` targets.
///
/// `Direct` is the conservative default. `BranchBuckets` groups observations by
/// series branch and term count; it can help some workloads, but it allocates a
/// scratch buffer per evaluation unless a [`WienerBatchWorkspace`](crate::wiener::WienerBatchWorkspace)
/// or cached [`WienerBatchModel`](crate::wiener::WienerBatchModel) is used.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BatchStrategy {
    #[default]
    Direct,
    BranchBuckets,
}

/// Structure-of-arrays storage for Wiener reaction-time observations.
///
/// Upper and lower boundary hits are stored separately so batch likelihoods can
/// avoid a branch in the hot loop and use contiguous reaction-time data. Each
/// allocation has 64-byte base alignment; arbitrary subslices may be unaligned.
#[derive(Debug)]
pub struct WienerObservations {
    upper_rt: OwnedBuffer<f64>,
    lower_rt: OwnedBuffer<f64>,
    batch_strategy: BatchStrategy,
}

impl WienerObservations {
    #[must_use]
    pub fn new(upper_rt: OwnedBuffer<f64>, lower_rt: OwnedBuffer<f64>) -> Self {
        Self {
            upper_rt,
            lower_rt,
            batch_strategy: BatchStrategy::default(),
        }
    }

    /// Selects the batch strategy used by `Target<Wiener*, WienerObservations>`.
    #[must_use]
    pub fn with_batch_strategy(mut self, batch_strategy: BatchStrategy) -> Self {
        self.batch_strategy = batch_strategy;
        self
    }

    #[must_use]
    pub fn batch_strategy(&self) -> BatchStrategy {
        self.batch_strategy
    }

    #[must_use]
    pub fn upper_rt(&self) -> &[f64] {
        self.upper_rt.as_slice()
    }

    #[must_use]
    pub fn lower_rt(&self) -> &[f64] {
        self.lower_rt.as_slice()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.upper_rt.len() + self.lower_rt.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FromIterator<WienerObservation> for WienerObservations {
    fn from_iter<T: IntoIterator<Item = WienerObservation>>(iter: T) -> Self {
        let observations: Vec<WienerObservation> = iter.into_iter().collect();
        let upper_len = observations
            .iter()
            .filter(|obs| matches!(obs.boundary, Boundary::Upper))
            .count();
        let lower_len = observations.len() - upper_len;

        let mut upper_rt = OwnedBuffer::new(upper_len);
        let mut lower_rt = OwnedBuffer::new(lower_len);
        let mut upper_idx = 0;
        let mut lower_idx = 0;

        for obs in observations {
            match obs.boundary {
                Boundary::Upper => {
                    upper_rt.as_mut_slice()[upper_idx] = obs.rt;
                    upper_idx += 1;
                }
                Boundary::Lower => {
                    lower_rt.as_mut_slice()[lower_idx] = obs.rt;
                    lower_idx += 1;
                }
            }
        }

        Self::new(upper_rt, lower_rt)
    }
}

impl From<Vec<WienerObservation>> for WienerObservations {
    fn from(observations: Vec<WienerObservation>) -> Self {
        observations.into_iter().collect()
    }
}
