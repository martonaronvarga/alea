use crate::wiener::{Wiener4, Wiener4Params, WienerObservation};
use alea_core::state_space::ObservationModel;

#[derive(Debug, Clone, Copy)]
pub struct LatentState<Context, Latent> {
    pub context: Context,
    pub latent: Latent,
}

impl<Context, Latent> LatentState<Context, Latent> {
    #[inline]
    pub fn new(context: Context, latent: Latent) -> Self {
        Self { context, latent }
    }

    #[inline]
    pub fn into_parts(self) -> (Context, Latent) {
        (self.context, self.latent)
    }
}

pub trait LatentStateView {
    type Context;
    type Latent;

    fn context(&self) -> &Self::Context;
    fn latent(&self) -> &Self::Latent;
}

impl<Context, Latent> LatentStateView for LatentState<Context, Latent> {
    type Context = Context;
    type Latent = Latent;

    #[inline]
    fn context(&self) -> &Self::Context {
        &self.context
    }

    #[inline]
    fn latent(&self) -> &Self::Latent {
        &self.latent
    }
}

pub trait WienerParameterMap<Context, Latent> {
    fn write_params(
        &self,
        context: &Context,
        latent: &Latent,
        t: usize,
        out: &mut Wiener4Params,
    ) -> crate::wiener::Result<()>;
}

#[derive(Debug, Clone, Copy)]
pub struct WienerObservationModel<M> {
    pub map: M,
    pub density: Wiener4,
}

impl<M> WienerObservationModel<M> {
    #[inline]
    pub fn new(map: M, density: Wiener4) -> Self {
        Self { map, density }
    }
}

impl<Context, Latent, M> ObservationModel<LatentState<Context, Latent>, WienerObservation>
    for WienerObservationModel<M>
where
    M: WienerParameterMap<Context, Latent>,
{
    type Error = crate::wiener::ProbError;
    #[inline]
    fn log_likelihood(
        &self,
        state: &LatentState<Context, Latent>,
        obs: &WienerObservation,
        t: usize,
    ) -> Result<f64, Self::Error> {
        let mut params = Wiener4Params {
            alpha: 1.0,
            tau: 0.0,
            beta: 0.5,
            delta: 0.0,
        };
        self.map
            .write_params(&state.context, &state.latent, t, &mut params)?;
        super::wiener::Wiener4Law::new(params, super::wiener::WienerOptions::new(1e-12))
            .and_then(|law| alea_core::distribution::Distribution::log_prob(&law, obs))
    }
}

use alea_core::state_space::TransitionModel;

/// Single-time latent state used by transition and observation models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentFrame {
    pub c: f64,
    pub m: f64,
}

/// Borrowed structure-of-arrays view of latent coordinates.
#[derive(Debug, Clone, Copy)]
pub struct LatentFrameSoA<'a> {
    pub c: &'a [f64],
    pub m: &'a [f64],
}

impl LatentFrameSoA<'_> {
    pub fn len(&self) -> usize {
        self.c.len()
    }

    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    pub fn validate(&self) -> bool {
        self.c.len() == self.m.len()
    }
}

/// Canonical affine map from latent factors to unconstrained Wiener parameters.
///
/// The affine form is on the transformed scale:
///   η_alpha = b0 + b1 c_t + b2 m_t
///   η_tau   = b0 + b1 c_t + b2 m_t
///   η_beta  = b0 + b1 c_t + b2 m_t
///   δ       = b0 + b1 c_t + b2 m_t
///
/// Then:
///   alpha = exp(η_alpha), tau = exp(η_tau), beta = sigmoid(η_beta)
///   delta = η_delta
///
/// This keeps the semantics layer separate from the distribution layer.
#[derive(Debug, Clone, Copy)]
pub struct AffineLatentWienerMap {
    pub alpha: [f64; 3],
    pub tau: [f64; 3],
    pub beta: [f64; 3],
    pub delta: [f64; 3],
}

impl AffineLatentWienerMap {
    #[inline]
    pub fn new(alpha: [f64; 3], tau: [f64; 3], beta: [f64; 3], delta: [f64; 3]) -> Self {
        Self {
            alpha,
            tau,
            beta,
            delta,
        }
    }

    #[inline]
    fn dot(coeffs: &[f64; 3], c: f64, m: f64) -> f64 {
        coeffs[0] + coeffs[1] * c + coeffs[2] * m
    }

    /// Map one latent frame to one Wiener parameter tuple.
    #[inline]
    pub fn map_frame(&self, frame: LatentFrame) -> Wiener4Params {
        Wiener4Params {
            alpha: Self::dot(&self.alpha, frame.c, frame.m).exp(),
            tau: Self::dot(&self.tau, frame.c, frame.m).exp(),
            beta: alea_math::numeric::sigmoid(Self::dot(&self.beta, frame.c, frame.m)),
            delta: Self::dot(&self.delta, frame.c, frame.m),
        }
    }

    /// Fill a batch of parameters from SoA latent factors.
    #[inline]
    pub fn map_into(
        &self,
        latent: LatentFrameSoA<'_>,
        out_alpha: &mut [f64],
        out_tau: &mut [f64],
        out_beta: &mut [f64],
        out_delta: &mut [f64],
    ) -> bool {
        if !latent.validate() {
            return false;
        }
        let n = latent.len();
        if out_alpha.len() < n || out_tau.len() < n || out_beta.len() < n || out_delta.len() < n {
            return false;
        }

        for i in 0..n {
            let c = latent.c[i];
            let m = latent.m[i];
            out_alpha[i] = Self::dot(&self.alpha, c, m).exp();
            out_tau[i] = Self::dot(&self.tau, c, m).exp();
            out_beta[i] = alea_math::numeric::sigmoid(Self::dot(&self.beta, c, m));
            out_delta[i] = Self::dot(&self.delta, c, m);
        }
        true
    }
}

/// A latent dynamics model that evolves `c_t` and `m_t`.
///
/// This is the structural layer that plugs into `TransitionModel`, while the
/// DDM/Wiener layer remains separate.
#[derive(Debug, Clone, Copy)]
pub struct LatentRandomWalk {
    sigma_c: f64,
    sigma_m: f64,
}

impl LatentRandomWalk {
    #[inline]
    pub fn new(sigma_c: f64, sigma_m: f64) -> Option<Self> {
        if !(sigma_c.is_finite() && sigma_c >= 0.0 && sigma_m.is_finite() && sigma_m >= 0.0) {
            return None;
        }
        Some(Self { sigma_c, sigma_m })
    }

    #[inline]
    fn gaussian_log_density(x: f64, mean: f64, sigma: f64) -> f64 {
        if sigma == 0.0 {
            return if x == mean { 0.0 } else { f64::NEG_INFINITY };
        }
        let z = (x - mean) / sigma;
        -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

impl TransitionModel<LatentFrame> for LatentRandomWalk {
    type Error = alea_core::distribution::DistributionError;
    #[inline]
    fn log_transition(
        &self,
        prev: &LatentFrame,
        next: &LatentFrame,
        _t: usize,
    ) -> Result<f64, Self::Error> {
        if [prev.c, prev.m, next.c, next.m]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err(Self::Error::Observation);
        }
        let value = Self::gaussian_log_density(next.c, prev.c, self.sigma_c)
            + Self::gaussian_log_density(next.m, prev.m, self.sigma_m);
        if value.is_nan() {
            Err(Self::Error::Numerical)
        } else {
            Ok(value)
        }
    }

    #[inline]
    fn sample_next<R: rand::Rng + ?Sized>(
        &self,
        prev: &LatentFrame,
        _t: usize,
        rng: &mut R,
    ) -> Result<LatentFrame, Self::Error> {
        if !prev.c.is_finite() || !prev.m.is_finite() {
            return Err(Self::Error::Observation);
        }
        let dc = if self.sigma_c > 0.0 {
            alea_math::random::standard_normal(rng) * self.sigma_c
        } else {
            0.0
        };
        let dm = if self.sigma_m > 0.0 {
            alea_math::random::standard_normal(rng) * self.sigma_m
        } else {
            0.0
        };
        let draw = LatentFrame {
            c: prev.c + dc,
            m: prev.m + dm,
        };
        if draw.c.is_finite() && draw.m.is_finite() {
            Ok(draw)
        } else {
            Err(Self::Error::Numerical)
        }
    }
}

/// Observation model that turns a latent frame into Wiener parameters and then
/// evaluates the exact first-passage density.
#[derive(Debug, Clone, Copy)]
pub struct LatentWienerObservation {
    pub projection: AffineLatentWienerMap,
    pub distribution: Wiener4,
}

impl LatentWienerObservation {
    #[inline]
    pub fn new(projection: AffineLatentWienerMap, distribution: Wiener4) -> Self {
        Self {
            projection,
            distribution,
        }
    }
}

impl ObservationModel<LatentFrame, WienerObservation> for LatentWienerObservation {
    type Error = crate::wiener::ProbError;
    #[inline]
    fn log_likelihood(
        &self,
        state: &LatentFrame,
        obs: &WienerObservation,
        _t: usize,
    ) -> Result<f64, Self::Error> {
        let params = self.projection.map_frame(*state);
        super::wiener::Wiener4Law::new(params, super::wiener::WienerOptions::new(1e-12))
            .and_then(|law| alea_core::distribution::Distribution::log_prob(&law, obs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    #[cfg_attr(miri, ignore = "long statistical regression; native CI covers it")]
    fn random_walk_increments_have_gaussian_moments() {
        let walk = LatentRandomWalk::new(2.0, 0.5).unwrap();
        let origin = LatentFrame { c: 3.0, m: -2.0 };
        let mut rng = SmallRng::seed_from_u64(723);
        let mut sums = [0.0; 2];
        let mut squares = [0.0; 2];
        let mut fourths = [0.0; 2];
        const N: usize = 50_000;
        for _ in 0..N {
            let draw = walk.sample_next(&origin, 0, &mut rng).unwrap();
            for (i, z) in [(draw.c - origin.c) / 2.0, (draw.m - origin.m) / 0.5]
                .into_iter()
                .enumerate()
            {
                sums[i] += z;
                squares[i] += z * z;
                fourths[i] += z.powi(4);
            }
        }
        for i in 0..2 {
            assert!((sums[i] / N as f64).abs() < 0.025);
            assert!((squares[i] / N as f64 - 1.0).abs() < 0.04);
            assert!((fourths[i] / N as f64 - 3.0).abs() < 0.2);
        }
    }

    #[test]
    fn zero_scale_is_a_point_mass_and_invalid_scales_are_rejected() {
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(LatentRandomWalk::new(invalid, 1.0).is_none());
            assert!(LatentRandomWalk::new(1.0, invalid).is_none());
        }
        let walk = LatentRandomWalk::new(0.0, 0.0).unwrap();
        let origin = LatentFrame { c: 0.0, m: 1.0 };
        let mut rng = SmallRng::seed_from_u64(7);
        assert_eq!(walk.sample_next(&origin, 0, &mut rng).unwrap(), origin);
        assert_eq!(walk.log_transition(&origin, &origin, 0).unwrap(), 0.0);
        let nearby = LatentFrame {
            c: f64::EPSILON / 2.0,
            ..origin
        };
        assert_eq!(
            walk.log_transition(&origin, &nearby, 0).unwrap(),
            f64::NEG_INFINITY
        );
    }
}
