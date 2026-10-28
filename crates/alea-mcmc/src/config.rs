//! Validated sampler inputs shared by present and future trajectory policies.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// A scale must be finite and strictly positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("scale must be finite and positive")]
pub struct InvalidScale;
macro_rules! scale {
    ($name:ident, $doc:literal) => {
        #[doc=$doc]
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use]
        pub struct $name(f64);
        impl TryFrom<f64> for $name {
            type Error = InvalidScale;
            fn try_from(value: f64) -> Result<Self, Self::Error> {
                if value.is_finite() && value > 0. {
                    Ok(Self(value))
                } else {
                    Err(InvalidScale)
                }
            }
        }
        impl $name {
            /// The validated positive scalar.
            pub fn value(self) -> f64 {
                self.0
            }
        }
    };
}
scale!(StepSize, "Finite positive integration step size.");
scale!(
    ProposalScale,
    "Finite positive random-walk proposal scale, not an integration step."
);
// Explicit retagging permits shared scalar adaptation without conflating APIs.
impl From<StepSize> for ProposalScale {
    fn from(step: StepSize) -> Self {
        Self(step.value())
    }
}
impl From<ProposalScale> for StepSize {
    fn from(scale: ProposalScale) -> Self {
        Self(scale.value())
    }
}

/// Configuration/shape errors detected before allocation, evaluation, or RNG use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SamplerConfigError {
    #[error(transparent)]
    /// Invalid integration or proposal step size.
    StepSize(#[from] InvalidScale),
    #[error("acceptance target must be finite and strictly between zero and one")]
    /// Acceptance target must be finite and strictly between zero and one.
    AcceptanceTarget,
    #[error("tree depth must be nonzero and smaller than usize::BITS")]
    /// Tree depth must be nonzero and smaller than usize::BITS.
    TreeDepth,
    #[error("sampler dimension must be nonzero")]
    /// Sampler dimension must be nonzero.
    EmptyDimension,
    #[error("sampler dimension exceeds addressable aligned storage")]
    /// Sampler dimension exceeds addressable aligned storage.
    DimensionOverflow,
    #[error("draw buffer size exceeds addressable storage")]
    /// Draw buffer size exceeds addressable storage.
    DrawCountOverflow,
    #[error("sampler dimension is {expected}, but state dimension is {actual}")]
    /// Sampler dimension is `expected`, but state dimension is `actual`.
    StateDimension {
        /// Required dimension or storage length.
        expected: usize,
        /// Observed dimension or storage length.
        actual: usize,
    },
}

/// A finite target acceptance probability strictly inside `(0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct AcceptanceTarget(f64);

impl TryFrom<f64> for AcceptanceTarget {
    type Error = SamplerConfigError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() || value <= 0.0 || value >= 1.0 {
            return Err(SamplerConfigError::AcceptanceTarget);
        }
        Ok(Self(value))
    }
}

impl AcceptanceTarget {
    /// The validated probability.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

/// A nonzero binary trajectory depth with representable node/step counts.
/// This validates arithmetic, not a practical memory or execution-time budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct TreeDepth(usize);

impl TryFrom<usize> for TreeDepth {
    type Error = SamplerConfigError;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value == 0 || value >= usize::BITS as usize {
            return Err(SamplerConfigError::TreeDepth);
        }
        Ok(Self(value))
    }
}

impl TreeDepth {
    /// The validated depth.
    #[must_use]
    pub fn value(self) -> usize {
        self.0
    }
    /// Maximum leapfrog count for doubling depths `0..depth` (not including origin).
    #[must_use]
    pub fn max_integration_steps(self) -> usize {
        (1_usize << self.0) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_rejects_endpoints_and_nonfinite_values() {
        for value in [
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(
                AcceptanceTarget::try_from(value),
                Err(SamplerConfigError::AcceptanceTarget)
            );
        }
        for value in [f64::MIN_POSITIVE, 0.234, 0.8, 1.0 - f64::EPSILON] {
            assert_eq!(AcceptanceTarget::try_from(value).unwrap().value(), value);
        }
    }

    #[test]
    fn every_representable_depth_has_overflow_free_counts() {
        for depth in 1..usize::BITS as usize {
            let value = TreeDepth::try_from(depth).unwrap();
            assert_eq!(value.value(), depth);
            assert_eq!(value.max_integration_steps(), 2_usize.pow(depth as u32) - 1);
        }
        for depth in [0, usize::BITS as usize, usize::MAX] {
            assert_eq!(
                TreeDepth::try_from(depth),
                Err(SamplerConfigError::TreeDepth)
            );
        }
    }
}
