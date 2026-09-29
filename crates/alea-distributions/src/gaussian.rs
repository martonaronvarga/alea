//! Dimension-bound standard multivariate Gaussian, omitting its normalizing constant.
use alea_core::target::{DimensionError, LogDensityGradient};

#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    dimension: usize,
}

impl Gaussian {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl alea_core::density::LogDensity for Gaussian {
    type Error = DimensionError;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp(&self, q: &[f64]) -> Result<f64, Self::Error> {
        if q.len() != self.dimension {
            return Err(DimensionError {
                expected: self.dimension,
                position: q.len(),
                gradient: self.dimension,
            });
        }
        Ok(-0.5 * q.iter().map(|x| x * x).sum::<f64>())
    }
}

impl LogDensityGradient for Gaussian {
    type Error = DimensionError;

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        if q.len() != self.dimension || g.len() != self.dimension {
            return Err(DimensionError {
                expected: self.dimension,
                position: q.len(),
                gradient: g.len(),
            });
        }
        let mut squared = 0.0;
        for (&q, g) in q.iter().zip(g) {
            *g = -q;
            squared += q * q;
        }
        Ok(-0.5 * squared)
    }
}
