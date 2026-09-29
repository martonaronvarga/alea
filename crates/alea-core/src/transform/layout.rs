use super::{Transform, TransformError, finite, log_jacobian, shape, size};
use std::ops::Range;

/// Immutable block offsets, computed by the layout rather than supplied by callers.
#[derive(Clone, Debug)]
pub struct ParameterBlock {
    transform: Transform,
    unconstrained: Range<usize>,
    constrained: Range<usize>,
}
impl ParameterBlock {
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    pub fn unconstrained_range(&self) -> Range<usize> {
        self.unconstrained.clone()
    }
    pub fn constrained_range(&self) -> Range<usize> {
        self.constrained.clone()
    }
}

/// Flat, validated parameter layout. Construction allocates; evaluation does not.
#[derive(Clone, Debug)]
pub struct ParameterLayout {
    blocks: Vec<ParameterBlock>,
    input: usize,
    output: usize,
    scratch: usize,
}
impl ParameterLayout {
    pub fn new(transforms: impl IntoIterator<Item = Transform>) -> Result<Self, TransformError> {
        let mut layout = Self {
            blocks: Vec::new(),
            input: 0,
            output: 0,
            scratch: 0,
        };
        for transform in transforms {
            let input = size(
                layout
                    .input
                    .checked_add(transform.input)
                    .ok_or(TransformError::InvalidDimension)?,
            )?;
            let output = size(
                layout
                    .output
                    .checked_add(transform.output)
                    .ok_or(TransformError::InvalidDimension)?,
            )?;
            layout.blocks.push(ParameterBlock {
                transform,
                unconstrained: layout.input..input,
                constrained: layout.output..output,
            });
            layout.input = input;
            layout.output = output;
            layout.scratch = layout.scratch.max(transform.scratch);
        }
        Ok(layout)
    }
    pub fn blocks(&self) -> &[ParameterBlock] {
        &self.blocks
    }
    pub fn unconstrained_dimension(&self) -> usize {
        self.input
    }
    pub fn constrained_dimension(&self) -> usize {
        self.output
    }
    pub fn scratch_dimension(&self) -> usize {
        self.scratch
    }

    fn check(&self, q: &[f64], x: &[f64], scratch: &[f64]) -> Result<(), TransformError> {
        shape(self.input, q.len())?;
        shape(self.output, x.len())?;
        if scratch.len() < self.scratch {
            return Err(TransformError::Dimension {
                expected: self.scratch,
                actual: scratch.len(),
            });
        }
        Ok(())
    }
    /// Errors carry block-local coordinate indices; block ranges remain available.
    pub fn constrain(
        &self,
        q: &[f64],
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<f64, TransformError> {
        self.check(q, x, scratch)?;
        finite(q)?;
        let mut jac = 0.0;
        for block in &self.blocks {
            jac += block.transform.constrain(
                &q[block.unconstrained.clone()],
                &mut x[block.constrained.clone()],
                scratch,
            )?;
        }
        log_jacobian(jac)
    }
    pub fn unconstrain(
        &self,
        x: &[f64],
        q: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        self.check(q, x, scratch)?;
        finite(x)?;
        for block in &self.blocks {
            block.transform.unconstrain(
                &x[block.constrained.clone()],
                &mut q[block.unconstrained.clone()],
                scratch,
            )?;
        }
        Ok(())
    }
    /// Compose model gradients with all transforms, including their Jacobians.
    pub fn pullback(
        &self,
        q: &[f64],
        gx: &[f64],
        gq: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        self.check(q, gx, scratch)?;
        shape(self.input, gq.len())?;
        finite(q)?;
        finite(gx)?;
        for block in &self.blocks {
            block.transform.pullback(
                &q[block.unconstrained.clone()],
                &gx[block.constrained.clone()],
                &mut gq[block.unconstrained.clone()],
                scratch,
            )?;
        }
        Ok(())
    }
}
