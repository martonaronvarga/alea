//! Numerically stable online weights shared by dynamic trajectory kernels.

/// A max-shifted positive sum represented by a log maximum and scaled mass.
///
/// Keeping the multiplicity separately is important when a large absolute log
/// weight makes `log(exp(x) + exp(x))` round back to `x`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Weight {
    pub(crate) max: f64,
    pub(crate) sum: f64,
}

impl Weight {
    pub(crate) fn leaf(log: f64) -> Self {
        Self { max: log, sum: 1.0 }
    }

    /// Merge `other` into `self` and return the probability of selecting it.
    ///
    /// Ordinary multinomial selection divides by the merged mass. Biased
    /// progressive selection divides by the old mass and caps at one.
    pub(crate) fn merge_select(self, other: Self, biased: bool) -> (Self, f64) {
        let max = self.max.max(other.max);
        let gap = other.max - self.max;
        let (old, new) = if gap >= 0.0 {
            (self.sum * (-gap).exp(), other.sum)
        } else {
            (self.sum, other.sum * gap.exp())
        };
        let weight = Self {
            max,
            sum: old + new,
        };
        let probability = if gap.abs() <= 500.0 {
            if biased {
                (new / old).min(1.0)
            } else {
                new / weight.sum
            }
        } else {
            // Avoid underflow before a large scaled subtree mass can rescue a
            // representable ratio.
            let ratio = gap + (other.sum.ln() - self.sum.ln());
            if biased {
                ratio.min(0.0).exp()
            } else if ratio >= 0.0 {
                1.0 / (1.0 + (-ratio).exp())
            } else {
                let r = ratio.exp();
                r / (1.0 + r)
            }
        };
        (weight, probability)
    }

    #[cfg(test)]
    pub(crate) fn merge(self, other: Self) -> Self {
        self.merge_select(other, false).0
    }

    #[cfg(test)]
    pub(crate) fn probability(self, other: Self, biased: bool) -> f64 {
        other.merge_select(self, biased).1
    }
}
