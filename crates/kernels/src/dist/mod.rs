pub mod traits;
pub mod wiener;

use crate::density::{GradLogDensity, LogDensity};

#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;

impl LogDensity for Gaussian {
    type Point = [f64];
    fn log_prob(&self, x: &[f64]) -> f64 {
        gauss::gaussian_log_prob(x)
    }
}

impl GradLogDensity for Gaussian {
    type Gradient = [f64];
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        gauss::gaussian_grad(x, grad)
    }
}

#[cxx::bridge]
mod gauss {
    unsafe extern "C++" {
        include!("../../../ffi/cpp/include/gaussian.hpp");
        fn gaussian_log_prob(x: &[f64]) -> f64;
        fn gaussian_grad(x: &[f64], grad: &mut [f64]);
    }
}
