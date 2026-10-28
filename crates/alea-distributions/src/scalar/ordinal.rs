//! Ordered response models, with zero-based categories and increasing cutpoints.
use super::jet::Jet;
use super::{checked, finite, standard_normal, uniform};
use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, Sample, Support,
};
use alea_math::numeric::softplus;
use rand::Rng;
#[derive(Debug, Clone, PartialEq)]
struct Ordered {
    location: f64,
    cuts: Vec<f64>,
    probit: bool,
}
impl Ordered {
    fn new(location: f64, cuts: Vec<f64>, probit: bool) -> Result<Self, E> {
        finite(location, "location")?;
        if cuts.is_empty() {
            return Err(E::Shape);
        }
        for c in &cuts {
            finite(*c, "cutpoint")?;
        }
        if cuts.windows(2).any(|x| x[0] >= x[1]) {
            return Err(E::Parameter("ordered cutpoints"));
        }
        Ok(Self {
            location,
            cuts,
            probit,
        })
    }
    fn log_cdf(&self, z: f64) -> f64 {
        if self.probit {
            Jet::<0>::c(z).log_phi().v
        } else {
            -softplus(-z)
        }
    }
    fn log_pdf(&self, z: f64) -> f64 {
        if self.probit {
            -0.5 * z * z - 0.5 * super::LN_TAU
        } else {
            -softplus(-z) - softplus(z)
        }
    }
    fn logp(&self, k: usize) -> Result<f64, E> {
        let n = self.cuts.len();
        if k > n {
            return Ok(f64::NEG_INFINITY);
        }
        let value = if k == 0 {
            self.log_cdf(self.cuts[0] - self.location)
        } else if k == n {
            self.log_cdf(self.location - self.cuts[n - 1])
        } else {
            let lo = self.cuts[k - 1] - self.location;
            let hi = self.cuts[k] - self.location;
            if self.probit {
                let (a, b) = if lo >= 0.0 {
                    (self.log_cdf(-lo), self.log_cdf(-hi))
                } else {
                    (self.log_cdf(hi), self.log_cdf(lo))
                };
                a + (-(b - a).exp_m1()).ln()
            } else {
                -softplus(-hi) - softplus(lo)
                    + (-(-(self.cuts[k] - self.cuts[k - 1])).exp_m1()).ln()
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(E::Numerical)
        }
    }
    fn scores(&self, k: usize, out: &mut [f64]) -> Result<(), E> {
        if out.len() != self.cuts.len() + 1 {
            return Err(E::Shape);
        }
        if k > self.cuts.len() {
            return Err(E::Observation);
        }
        let lp = self.logp(k)?;
        out.fill(0.0);
        let lo = if k == 0 {
            0.0
        } else {
            (self.log_pdf(self.cuts[k - 1] - self.location) - lp).exp()
        };
        let hi = if k == self.cuts.len() {
            0.0
        } else {
            (self.log_pdf(self.cuts[k] - self.location) - lp).exp()
        };
        out[0] = lo - hi;
        if k > 0 {
            out[k] = -lo;
        }
        if k < self.cuts.len() {
            out[k + 1] = hi;
        }
        if out.iter().all(|s| s.is_finite()) {
            Ok(())
        } else {
            Err(E::Numerical)
        }
    }
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<usize, E> {
        let z = if self.probit {
            standard_normal(rng)
        } else {
            let u = uniform(rng);
            u.ln() - (-u).ln_1p()
        };
        let latent = self.location + z;
        if !latent.is_finite() {
            return Err(E::Numerical);
        }
        Ok(self.cuts.partition_point(|c| *c < latent))
    }
}
macro_rules! ordered {
    ($name:ident,$probit:literal,$doc:literal) => {
        #[doc=$doc]
        #[derive(Debug, Clone, PartialEq)]
        #[must_use]
        pub struct $name(Ordered);
        impl $name {
            /// # Errors
            /// Requires finite location and a nonempty, finite, strictly increasing cutpoint vector.
            pub fn new(location: f64, cutpoints: Vec<f64>) -> Result<Self, E> {
                Ok(Self(Ordered::new(location, cutpoints, $probit)?))
            }
        }
        impl Distribution for $name {
            type Error = alea_core::distribution::DistributionError;
            type Event = usize;
            fn event_shape(&self) -> EventShape {
                EventShape::Scalar
            }
            fn support(&self) -> Support {
                Support::Finite {
                    outcomes: self.0.cuts.len() + 1,
                }
            }
            fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
                alea_core::distribution::ReferenceMeasure::Counting
            }
            fn log_prob(&self, k: &usize) -> Result<f64, E> {
                checked(self.0.logp(*k)?)
            }
        }
        impl ParameterScore for $name {
            fn parameter_count(&self) -> usize {
                self.0.cuts.len() + 1
            }
            fn parameter_score(&self, k: &usize, out: &mut [f64]) -> Result<(), E> {
                self.0.scores(*k, out)
            }
        }
        impl Sample for $name {
            type Draw = usize;
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<usize, E> {
                self.0.draw(rng)
            }
        }
    };
}
ordered!(
    OrderedLogistic,
    false,
    "Ordered logistic with zero-based categories. Score order: location, cutpoints. Common shifts of location and cutpoints are unidentified."
);
ordered!(
    OrderedProbit,
    true,
    "Ordered probit with zero-based categories. Score order: location, cutpoints. Residual normal standard deviation is one."
);
