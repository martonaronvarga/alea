use crate::error::{ProbError, Result};
use alea_ffi::{Bounds, ErrorNorm, Options, hcubature_into};
use core::option::Option;
use std::f64::consts::PI;

#[allow(clippy::unreadable_literal)]
const LN_PI: f64 = 1.1447298858494001741434273513530587;

mod quadrature;
mod series;
use quadrature::{
    GL_1_NODES, GL_1_WTS, GL_5_NODES, GL_5_WTS, GL_7_NODES, GL_7_WTS, GL_15_NODES, GL_15_WTS,
    GL_25_NODES, GL_25_WTS,
};

mod types;
pub use types::{
    BatchStrategy, Boundary, Quadrature, SeriesBranch, Wiener4, Wiener4Eval, Wiener4Grad,
    Wiener4Params, Wiener5, Wiener5Eval, Wiener5Grad, Wiener5Params, Wiener7, Wiener7Eval,
    Wiener7Grad, Wiener7Params, Wiener7ParamsBuilder, WienerBranchCounts, WienerObservation,
    WienerObservations, WienerOptions,
};
use types::{
    FAIL_EVAL_5, NAN_GRAD_4, NAN_GRAD_5, NAN_GRAD_7, Wiener4Core, Wiener5Core, Wiener7Core,
};

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

impl Wiener5Params {
    pub fn to_unconstrained(c: &Self) -> [f64; 5] {
        [
            c.base.alpha.ln(),
            c.base.tau.ln(),
            alea_math::numeric::logit(c.base.beta),
            c.base.delta, // Drift is already unconstrained (R)
            c.s_delta.ln(),
        ]
    }

    pub fn from_unconstrained(u: &[f64; 5]) -> Self {
        Wiener5Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            alea_math::numeric::sigmoid(u[2]),
            u[3],
            u[4].exp(),
        )
    }

    pub fn log_abs_det_jacobian(u: &[f64; 5]) -> f64 {
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta =
            alea_math::numeric::log_sigmoid(u[2]) + alea_math::numeric::log1m_sigmoid(u[2]);
        let log_jac_var_delta = u[4];

        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_var_delta
    }
}

impl Wiener7Params {
    pub fn to_unconstrained(c: &Self) -> [f64; 7] {
        [
            c.base.base.alpha.ln(),
            c.base.base.tau.ln(),
            alea_math::numeric::logit(c.base.base.beta),
            c.base.base.delta, // already unconstrained
            c.base.s_delta.ln(),
            // sw ∈ (0,1) → logit sw
            alea_math::numeric::logit(c.s_beta),
            c.s_tau.ln(),
        ]
    }

    pub fn from_unconstrained(u: &[f64; 7]) -> Self {
        Wiener7Params::with_params_unchecked(
            u[0].exp(),
            u[1].exp(),
            alea_math::numeric::sigmoid(u[2]),
            u[3],
            alea_math::numeric::sigmoid(u[5]),
            u[6].exp(),
            u[4].exp(),
        )
    }

    pub fn log_abs_det_jacobian(u: &[f64; 7]) -> f64 {
        let log_jac_alpha = u[0];
        let log_jac_tau = u[1];
        let log_jac_beta =
            alea_math::numeric::log_sigmoid(u[2]) + alea_math::numeric::log1m_sigmoid(u[2]);
        let log_jac_s_delta = u[4];
        let log_jac_sw =
            alea_math::numeric::log_sigmoid(u[5]) + alea_math::numeric::log1m_sigmoid(u[5]);
        let log_jac_st0 = u[6];
        log_jac_alpha + log_jac_tau + log_jac_beta + log_jac_s_delta + log_jac_sw + log_jac_st0
    }
}

mod primitive;
pub use primitive::{WienerPrimitive, WienerPrimitiveError};

mod model;
pub use model::{WienerModel, WienerModelError};

#[cfg(test)]
mod tests;
