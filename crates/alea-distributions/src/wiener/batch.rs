use super::*;
use alea_math::buffer::OwnedBuffer;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BatchEntry {
    pub core: Wiener5Core,
    pub bucket: u8,
    pub k: usize,
}

/// Reusable initialized scratch for Wiener4/5 branch-bucket evaluations.
///
/// One contiguous aligned allocation stores all cores. A counting partition gives
/// seven contiguous buckets, avoiding seven full scratch scans. Construction or
/// growth may allocate; repeated evaluations within capacity do not allocate.
#[derive(Debug)]
pub struct WienerBatchWorkspace {
    pub(super) entries: OwnedBuffer<BatchEntry>,
    pub(super) offsets: [usize; 8],
}

impl Default for WienerBatchWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}

impl WienerBatchWorkspace {
    /// Allocate initialized scratch for at least `observations` entries.
    #[must_use]
    pub fn new(observations: usize) -> Self {
        Self {
            entries: OwnedBuffer::new(observations),
            offsets: [0; 8],
        }
    }

    /// Number of observations that fit without growing storage.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    pub(super) fn prepare(
        &mut self,
        entries: impl ExactSizeIterator<Item = Result<BatchEntry>>,
    ) -> bool {
        self.entries.resize(entries.len());
        self.offsets.fill(0);
        for (slot, entry) in self.entries.iter_mut().zip(entries) {
            let Ok(entry) = entry else {
                return false;
            };
            self.offsets[usize::from(entry.bucket) + 1] += 1;
            *slot = entry;
        }
        for i in 1..8 {
            self.offsets[i] += self.offsets[i - 1];
        }
        let mut next = self.offsets;
        for bucket in 0..7 {
            while next[bucket] < self.offsets[bucket + 1] {
                let i = next[bucket];
                let destination = usize::from(self.entries[i].bucket);
                if destination == bucket {
                    next[bucket] += 1;
                } else {
                    self.entries.swap(i, next[destination]);
                    next[destination] += 1;
                }
            }
        }
        true
    }

    pub(super) fn bucket(&self, bucket: u8) -> &[BatchEntry] {
        let b = usize::from(bucket);
        &self.entries[self.offsets[b]..self.offsets[b + 1]]
    }
}
