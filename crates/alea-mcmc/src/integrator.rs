//! Validated, statically dispatched symmetric kick/drift macrosteps.
mod private {
    pub trait Sealed {}
}
/// A complete palindromic, consistent separable-Hamiltonian map. Internal stages
/// are never trajectory candidates. Geometry and coefficients stay fixed during
/// a trajectory and after warmup. This is not WALNUTS's local-step controller.
pub trait Integrator: private::Sealed {
    /// Number of drifts/new fused evaluations on a successful macrostep.
    fn stages(&self) -> usize;
    fn kick(&self, index: usize) -> f64;
    fn drift(&self, index: usize) -> f64;
}
#[derive(Debug, Clone, Copy, Default)]
pub struct Leapfrog;
impl private::Sealed for Leapfrog {}
impl Integrator for Leapfrog {
    fn stages(&self) -> usize {
        1
    }
    fn kick(&self, _: usize) -> f64 {
        0.5
    }
    fn drift(&self, _: usize) -> f64 {
        1.0
    }
}
/// `B(lambda h) A(h/2) B((1-2lambda)h) A(h/2) B(lambda h)`.
#[derive(Debug, Clone, Copy)]
pub struct TwoStage {
    lambda: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("two-stage coefficient must be finite and between zero and one half")]
pub struct InvalidCoefficient;
impl TwoStage {
    pub const OMF: Self = Self {
        lambda: 0.193_183_327_503_783_6,
    };
    /// BCSS two-stage coefficient, optimized over a finite oscillator interval.
    pub const BCSS: Self = Self { lambda: 0.211_781 };
    /// # Errors
    /// Rejects non-finite values or coefficients outside [0, 1/2].
    pub fn new(lambda: f64) -> Result<Self, InvalidCoefficient> {
        if !lambda.is_finite() || !(0.0..=0.5).contains(&lambda) {
            return Err(InvalidCoefficient);
        }
        Ok(Self { lambda })
    }
    pub fn lambda(self) -> f64 {
        self.lambda
    }
}
impl private::Sealed for TwoStage {}
impl Integrator for TwoStage {
    fn stages(&self) -> usize {
        2
    }
    fn kick(&self, index: usize) -> f64 {
        if index == 1 {
            1.0 - 2.0 * self.lambda
        } else {
            self.lambda
        }
    }
    fn drift(&self, _: usize) -> f64 {
        0.5
    }
}
