#![allow(dead_code)]
use alea_core::model::{AnalyticModel, PrimitiveModel, TransformedTarget};
use alea_core::transform::{ParameterLayout, Transform};
use alea_distributions::wiener::WienerPrimitive;
use alea_distributions::wiener::{Boundary, WienerObservation};
use std::convert::Infallible;

#[cfg(feature = "std-autodiff")]
use std::autodiff::autodiff_reverse;

pub const NAMES: [&str; 5] = ["gaussian", "correlated", "logistic", "banana", "funnel"];

#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_gaussian, Duplicated, Active)
)]
pub fn gaussian(q: &[f64]) -> f64 {
    -0.5 * (q[0] * q[0] + q[1] * q[1])
}
#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_correlated, Duplicated, Active)
)]
pub fn correlated(q: &[f64]) -> f64 {
    -0.5 / 0.36 * (q[0] * q[0] - 1.6 * q[0] * q[1] + q[1] * q[1])
}
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}
#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_logistic, Duplicated, Active)
)]
pub fn logistic(q: &[f64]) -> f64 {
    let mut lp = gaussian(q);
    for (x, y) in [(-1.5, 0.0), (-0.2, 1.0), (0.7, 0.0), (2.0, 1.0)] {
        let eta = q[0] + q[1] * x;
        lp += y * eta - softplus(eta);
    }
    lp
}
#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_banana, Duplicated, Active)
)]
pub fn banana(q: &[f64]) -> f64 {
    let z = q[1] - 0.3 * (q[0] * q[0] - 1.0);
    -0.5 * (q[0] * q[0] + z * z)
}
#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_funnel, Duplicated, Active)
)]
pub fn funnel(q: &[f64]) -> f64 {
    -q[0] * q[0] / 18.0 - 0.5 * q[0] - 0.5 * q[1] * q[1] * (-q[0]).exp()
}

pub fn analytic(kind: usize, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
    let lp = match kind {
        0 => {
            g[0] = -q[0];
            g[1] = -q[1];
            gaussian(q)
        }
        1 => {
            g[0] = -(q[0] - 0.8 * q[1]) / 0.36;
            g[1] = -(q[1] - 0.8 * q[0]) / 0.36;
            correlated(q)
        }
        2 => {
            g[0] = -q[0];
            g[1] = -q[1];
            for (x, y) in [(-1.5, 0.0), (-0.2, 1.0), (0.7, 0.0), (2.0, 1.0)] {
                let eta = q[0] + q[1] * x;
                let p = if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let t = eta.exp();
                    t / (1.0 + t)
                };
                g[0] += y - p;
                g[1] += x * (y - p);
            }
            logistic(q)
        }
        3 => {
            let z = q[1] - 0.3 * (q[0] * q[0] - 1.0);
            g[0] = -q[0] + 0.6 * q[0] * z;
            g[1] = -z;
            banana(q)
        }
        4 => {
            g[0] = -q[0] / 9.0 - 0.5 + 0.5 * q[1] * q[1] * (-q[0]).exp();
            g[1] = -q[1] * (-q[0]).exp();
            funnel(q)
        }
        _ => panic!("unknown test model"),
    };
    Ok(lp)
}

pub fn analytic_target(kind: usize) -> TransformedTarget<impl alea_core::model::ConstrainedModel> {
    TransformedTarget::new(
        AnalyticModel::new(2, move |q: &[f64], g: &mut [f64]| analytic(kind, q, g)),
        ParameterLayout::new([Transform::identity(2).unwrap()]).unwrap(),
    )
    .unwrap()
}

pub fn wiener_layout() -> ParameterLayout {
    ParameterLayout::new([
        Transform::positive(1).unwrap(),
        Transform::interval(1, 0.0, 0.4).unwrap(),
        Transform::interval(1, 0.0, 1.0).unwrap(),
        Transform::identity(1).unwrap(),
    ])
    .unwrap()
}
pub fn wiener_primitive() -> PrimitiveModel<WienerPrimitive, 4> {
    PrimitiveModel(
        WienerPrimitive::new(
            WienerObservation {
                rt: 0.8,
                boundary: Boundary::Upper,
            },
            1e-12,
        )
        .unwrap(),
    )
}
