//! Analytic test targets, independently differentiated by JAX in the generator.
#![allow(dead_code)]
use alea_core::target::LogDensityGradient;
use std::convert::Infallible;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Correlated,
    Banana,
    Logistic,
    Funnel,
    Rotated,
}
impl Target {
    pub fn parse(name: &str) -> Self {
        match name {
            "correlated" => Self::Correlated,
            "banana" => Self::Banana,
            "logistic" => Self::Logistic,
            "funnel" => Self::Funnel,
            "rotated" => Self::Rotated,
            _ => panic!("unknown reference target: {name}"),
        }
    }
    pub fn observables(self, q: &[f64]) -> [f64; 5] {
        let [x, y] = match self {
            Self::Correlated => [q[0], (q[1] - 0.6 * q[0]) / 0.8],
            Self::Banana => [q[0], q[1] - 0.4 * (q[0] * q[0] - 1.0)],
            Self::Logistic => [q[0], q[1]],
            Self::Funnel => [q[0], q[1] * (-0.5 * q[0]).exp()],
            Self::Rotated => [(0.8 * q[0] + 0.6 * q[1]) / 0.01, -0.6 * q[0] + 0.8 * q[1]],
        };
        [x, y, x * x, y * y, x * y]
    }
}
impl LogDensityGradient for Target {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        let lp = match self {
            Self::Correlated => {
                let z = (q[1] - 0.6 * q[0]) / 0.8;
                g[0] = -q[0] + 0.75 * z;
                g[1] = -z / 0.8;
                -0.5 * (q[0] * q[0] + z * z)
            }
            Self::Banana => {
                let z = q[1] - 0.4 * (q[0] * q[0] - 1.0);
                g[0] = -q[0] + 0.8 * q[0] * z;
                g[1] = -z;
                -0.5 * (q[0] * q[0] + z * z)
            }
            Self::Logistic => {
                let mut lp = -0.5 * (q[0] * q[0] + q[1] * q[1]);
                g[0] = -q[0];
                g[1] = -q[1];
                for (x, y) in [(-1.5, 0.0), (-0.2, 1.0), (0.7, 0.0), (2.0, 1.0)] {
                    let eta = q[0] + q[1] * x;
                    let t = (-eta.abs()).exp();
                    let p = if eta >= 0.0 {
                        1.0 / (1.0 + t)
                    } else {
                        t / (1.0 + t)
                    };
                    lp += y * eta - eta.max(0.0) - t.ln_1p();
                    g[0] += y - p;
                    g[1] += x * (y - p);
                }
                lp
            }
            Self::Funnel => {
                let precision = (-q[0]).exp();
                g[0] = -q[0] - 0.5 + 0.5 * q[1] * q[1] * precision;
                g[1] = -q[1] * precision;
                -0.5 * q[0] * q[0] - 0.5 * q[0] - 0.5 * q[1] * q[1] * precision
            }
            Self::Rotated => {
                let x = (0.8 * q[0] + 0.6 * q[1]) / 0.01;
                let y = -0.6 * q[0] + 0.8 * q[1];
                g[0] = -80.0 * x + 0.6 * y;
                g[1] = -60.0 * x - 0.8 * y;
                -0.5 * (x * x + y * y)
            }
        };
        Ok(lp)
    }
}
