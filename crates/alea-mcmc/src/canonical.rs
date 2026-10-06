//! Fixed canonical step control and endpoint correction, independent of trajectory selection.
//! Canonical phase dynamics remain in `hamiltonian`; this module neither retries
//! failed paths nor generalizes Metropolis to non-volume-preserving dynamics.
use crate::{
    hamiltonian::{Divergence, SignedStep},
    hmc::StepSize,
};

/// Geometry-independent constant step control, shared by probes and trajectories.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FixedStepControl(StepSize);
impl FixedStepControl {
    pub(crate) fn new(step: StepSize) -> Self {
        Self(step)
    }
    pub(crate) fn forward(self) -> SignedStep {
        SignedStep::new(self.0.value()).expect("validated positive step")
    }
}

/// Canonical Metropolis correction after the absolute energy-error diagnostic.
/// The limit is validated by HmcOptions; energies are checked independently here.
/// No RNG is consumed here; the trajectory owner retains its established policy.
pub(crate) fn metropolis(initial: f64, proposed: f64, limit: f64) -> Result<f64, Divergence> {
    let error = proposed - initial;
    if !initial.is_finite() || !proposed.is_finite() || !error.is_finite() {
        return Err(Divergence::Energy);
    }
    if error.abs() > limit {
        return Err(Divergence::EnergyErrorLimit);
    }
    Ok((-error).min(0.0).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correction_preserves_sign_cutoff_and_finite_energy_policy() {
        assert_eq!(metropolis(1.0, 0.0, 1.0).unwrap(), 1.0);
        assert!((metropolis(0.0, 1.0, 1.0).unwrap() - (-1.0_f64).exp()).abs() < 1e-14);
        assert_eq!(metropolis(0.0, 0.0, 1.0).unwrap(), 1.0);
        assert_eq!(
            metropolis(0.0, 1.01, 1.0),
            Err(Divergence::EnergyErrorLimit)
        );
        assert_eq!(
            metropolis(0.0, -1.01, 1.0),
            Err(Divergence::EnergyErrorLimit)
        );
        for (a, b) in [(f64::MAX, -f64::MAX), (f64::NAN, 0.0), (0.0, f64::INFINITY)] {
            assert_eq!(metropolis(a, b, 1.0), Err(Divergence::Energy));
        }
    }
}
