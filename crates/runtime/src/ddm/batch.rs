use kernels::dist::traits::ParameterMap;
use kernels::{
    density::{GradLogDensity, LogDensity},
    dist::wiener::{Wiener4, Wiener4Params},
};

/// Concrete parameter map for an affine latent-state model.
///
/// The unconstrained parameter vector is interpreted as:
///   [a0, a1, a2, t0, t1, t2, b0, b1, b2, d0, d1, d2]
///
/// For each time point i with features (c_i, m_i):
///   eta_alpha = a0 + a1 c_i + a2 m_i
///   eta_tau   = t0 + t1 c_i + t2 m_i
///   eta_beta  = b0 + b1 c_i + b2 m_i
///   delta     = d0 + d1 c_i + d2 m_i
///
/// Then:
///   alpha = exp(eta_alpha), tau = exp(eta_tau), beta = sigmoid(eta_beta)
///   delta = delta
#[derive(Debug, Clone, Copy)]
pub struct AffineLatentParameterMap<'a> {
    pub c: &'a [f64],
    pub m: &'a [f64],
}

impl<'a> AffineLatentParameterMap<'a> {
    #[inline]
    pub fn new(c: &'a [f64], m: &'a [f64]) -> Self {
        Self { c, m }
    }

    #[inline]
    fn dot(coeffs: &[f64; 3], c: f64, m: f64) -> f64 {
        coeffs[0] + coeffs[1] * c + coeffs[2] * m
    }

    #[inline]
    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    #[inline]
    fn read_coeffs(theta: &[f64], offset: usize) -> Option<[f64; 3]> {
        if theta.len() < offset + 3 {
            return None;
        }
        Some([theta[offset], theta[offset + 1], theta[offset + 2]])
    }

    #[inline]
    fn n(&self) -> usize {
        self.c.len().min(self.m.len())
    }

    #[inline]
    fn coeff_count(&self) -> usize {
        12
    }
}

#[cfg(feature = "std-autodiff")]
mod autodiff {
    use super::*;

    use std::autodiff::*;

    pub struct AffineLatentAutodiff;

    impl AffineLatentAutodiff {
        #[autodiff_reverse(affine_latent_log_prob_rev, Duplicated, Const, Const, Const, Active)]
        pub fn log_prob(
            theta: &[f64],
            distribution: &WienerFpt,
            observations: WienerObservationSoA<'_>,
            map: &AffineLatentParameterMap<'_>,
        ) -> f64 {
            let mut workspace = DdmWorkspace::with_len(observations.len());
            map.log_prob_with_workspace(theta, distribution, observations, &mut workspace)
        }
    }
}

// #[cfg(not(feature = "std-autodiff"))]
// panic!("enable the `std-autodiff` feature to obtain gradients");
