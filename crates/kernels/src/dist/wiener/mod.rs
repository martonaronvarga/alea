use crate::density::{FusedLogDensity, GradLogDensity};
use crate::dist::traits::{Family, Parameter, Target};
use crate::error::{ProbError, Result};
use core::option::Option;
use ffi::{hcubature_into, Bounds, ErrorNorm, Options};
use std::f64::consts::PI;

#[allow(clippy::unreadable_literal)]
const LN_PI: f64 = 1.1447298858494001741434273513530587;

mod quadrature;
mod series;
use quadrature::{
    GL_15_NODES, GL_15_WTS, GL_1_NODES, GL_1_WTS, GL_25_NODES, GL_25_WTS, GL_5_NODES,
    GL_5_WTS, GL_7_NODES, GL_7_WTS,
};

mod types;
pub use types::{
    BatchStrategy, Boundary, Quadrature, Wiener4, Wiener4Eval, Wiener4Grad, Wiener4Params, Wiener5, Wiener5Eval,
    SeriesBranch, WienerBranchCounts, Wiener5Grad, Wiener5Params, Wiener7, Wiener7Eval, Wiener7Grad, Wiener7Params,
    Wiener7ParamsBuilder, WienerObservation, WienerObservations, WienerOptions,
};
use types::{FAIL_EVAL_5, NAN_GRAD_4, NAN_GRAD_5, NAN_GRAD_7, Wiener4Core, Wiener5Core, Wiener7Core};


#[inline]
pub(super) fn likely(value: bool) -> bool {
    #[cfg(feature = "branch-hints")]
    {
        std::hint::likely(value)
    }
    #[cfg(not(feature = "branch-hints"))]
    {
        value
    }
}

#[inline]
pub(super) fn unlikely(value: bool) -> bool {
    #[cfg(feature = "branch-hints")]
    {
        std::hint::unlikely(value)
    }
    #[cfg(not(feature = "branch-hints"))]
    {
        value
    }
}

mod wiener4;
mod wiener5;
mod wiener7;

impl Parameter for Wiener5Params {
    type Constrained = Self;
    type Unconstrained = [f64; 5];

    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.base.alpha.ln(),
            c.base.tau.ln(),
            crate::numeric::logit(c.base.beta),
            c.base.delta, // Drift is already unconstrained (R)
            c.s_delta.ln(),
        ]
    }

    fn from_unconstrained(u: &Self::Unconstrained) -> Self::Constrained {
        Wiener5Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            crate::numeric::sigmoid(u[2]),
            u[3],
            u[4].exp(),
        )
    }

    // Add this to your Parameter trait!
    fn log_abs_det_jacobian(u: &Self::Unconstrained) -> f64 {
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta = crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2]);
        let log_jac_var_delta = u[4];

        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_var_delta
    }
}

impl Parameter for Wiener7Params {
    type Constrained = Self;
    type Unconstrained = [f64; 7];

    fn to_unconstrained(c: &Self::Constrained) -> Self::Unconstrained {
        [
            c.base.base.alpha.ln(),
            c.base.base.tau.ln(),
            crate::numeric::logit(c.base.base.beta),
            c.base.base.delta, // already unconstrained
            c.base.s_delta.ln(),
            // sw ∈ (0,1) → logit sw
            crate::numeric::logit(c.s_beta),
            c.s_tau.ln(),
        ]
    }

    fn from_unconstrained(u: &Self::Unconstrained) -> Self::Constrained {
        Wiener7Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            crate::numeric::sigmoid(u[2]),
            u[3],
            crate::numeric::sigmoid(u[5]),
            u[6].exp(),
            u[4].exp(),
        )
    }

    fn log_abs_det_jacobian(u: &Self::Unconstrained) -> f64 {
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta = crate::numeric::log_sigmoid(u[2]) + crate::numeric::log1m_sigmoid(u[2]);
        let log_jac_s_delta = u[4];
        let log_jac_sw = crate::numeric::log_sigmoid(u[5]) + crate::numeric::log1m_sigmoid(u[5]);
        let log_jac_st0 = u[6];
        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_s_delta + log_jac_sw + log_jac_st0
    }
}

#[cfg(test)]
mod tests;
