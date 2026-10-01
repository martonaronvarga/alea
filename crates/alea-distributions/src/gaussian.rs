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

impl alea_core::capability::HessianVector for Gaussian {
    fn potential_hvp(
        &self,
        q: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        if q.len() != self.dimension
            || vector.len() != self.dimension
            || output.len() != self.dimension
        {
            return Err(DimensionError {
                expected: self.dimension,
                position: q.len(),
                gradient: output.len(),
            });
        }
        output.copy_from_slice(vector);
        Ok(())
    }
}

impl alea_core::capability::BatchLogDensityGradient for Gaussian {
    fn batch_logp_grad(
        &self,
        shape: alea_core::capability::BatchShape,
        positions: &[f64],
        gradients: &mut [f64],
        results: &mut [alea_core::capability::BatchLane<Self::Error>],
    ) {
        for (lane, result) in results.iter_mut().enumerate() {
            let mut squared = 0.0;
            for coordinate in 0..shape.dimension() {
                let index = shape
                    .index(lane, coordinate)
                    .expect("batch wrapper validates shape");
                let value = positions[index];
                gradients[index] = -value;
                squared += value * value;
            }
            *result = alea_core::capability::BatchLane::Complete(Ok(-0.5 * squared));
        }
    }
}
