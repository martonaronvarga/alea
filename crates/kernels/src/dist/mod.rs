pub mod traits;
pub mod wiener;

use crate::density::{FusedLogDensity, GradLogDensity, LogDensity};

#[derive(Debug, Clone, Copy, Default)]
/// Standard multivariate Gaussian target, omitting the normalization constant.
pub struct Gaussian;

impl LogDensity for Gaussian {
    type Point = [f64];
    fn log_prob(&self, x: &[f64]) -> f64 {
        -0.5 * x.iter().map(|v| v * v).sum::<f64>()
    }
}

impl GradLogDensity for Gaussian {
    type Gradient = [f64];
    fn grad_log_prob(&self, x: &[f64], grad: &mut [f64]) {
        assert_eq!(x.len(), grad.len(), "gradient dimension mismatch");
        for (g, x) in grad.iter_mut().zip(x) {
            *g = -*x;
        }
    }
}

impl FusedLogDensity for Gaussian {
    fn log_prob_and_grad(&self, x: &[f64], grad: &mut [f64]) -> f64 {
        assert_eq!(x.len(), grad.len(), "gradient dimension mismatch");
        let mut square_sum = 0.0;
        for (g, x) in grad.iter_mut().zip(x) {
            *g = -*x;
            square_sum += x * x;
        }
        -0.5 * square_sum
    }
}
