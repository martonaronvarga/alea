use alea::target::{DimensionError, LogDensityGradient};

pub(crate) struct StandardNormalTarget {
    dimension: usize,
}

impl StandardNormalTarget {
    pub(crate) const fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl alea::target::TargetSpace for StandardNormalTarget {
    type Error = DimensionError;
    fn dimension(&self) -> usize {
        self.dimension
    }
}
impl LogDensityGradient for StandardNormalTarget {
    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        if position.len() != self.dimension || gradient.len() != self.dimension {
            return Err(DimensionError {
                expected: self.dimension,
                position: position.len(),
                gradient: gradient.len(),
            });
        }
        let mut squared = 0.0;
        for (&value, output) in position.iter().zip(gradient) {
            *output = -value;
            squared += value * value;
        }
        Ok(-0.5 * squared)
    }
}
