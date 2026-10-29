use super::{Transform, TransformError, finite, log_jacobian, shape, size};
use std::ops::Range;

/// Immutable block offsets, computed by the layout rather than supplied by callers.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct ParameterBlock {
    transform: Transform,
    unconstrained: Range<usize>,
    constrained: Range<usize>,
}
impl ParameterBlock {
    /// Transform associated with this parameter block.
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    #[must_use]
    /// Half-open range in the unconstrained coordinate vector.
    pub fn unconstrained_range(&self) -> Range<usize> {
        self.unconstrained.clone()
    }
    #[must_use]
    /// Half-open range in the constrained coordinate vector.
    pub fn constrained_range(&self) -> Range<usize> {
        self.constrained.clone()
    }
}

/// Flat, validated parameter layout. Construction allocates; evaluation does not.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "a parameter layout has no effect unless it is retained or applied"]
pub struct ParameterLayout {
    blocks: Vec<ParameterBlock>,
    input: usize,
    output: usize,
    scratch: usize,
}
impl ParameterLayout {
    /// Builds a checked flat layout in iteration order.
    ///
    /// # Errors
    /// Returns [`TransformError::InvalidDimension`] if aggregate dimensions
    /// overflow addressable storage.
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
    /// Parameter blocks in declaration order.
    pub fn blocks(&self) -> &[ParameterBlock] {
        &self.blocks
    }
    #[must_use]
    /// Number of input unconstrained coordinates.
    pub fn unconstrained_dimension(&self) -> usize {
        self.input
    }
    #[must_use]
    /// Number of output constrained coordinates.
    pub fn constrained_dimension(&self) -> usize {
        self.output
    }
    #[must_use]
    /// Number of scratch elements required for one layout evaluation.
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
    ///
    /// # Errors
    /// Returns [`TransformError`] for shape, non-finite input, domain, precision,
    /// output, or log-Jacobian failures in any block.
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
    /// Maps constrained parameters back to the layout's unconstrained coordinates.
    ///
    /// # Errors
    /// Returns [`TransformError`] for shape, non-finite input, domain, precision,
    /// or output failures in any block.
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
    ///
    /// # Errors
    /// Returns [`TransformError`] for shape, non-finite input, or numerical
    /// pullback failures in any block.
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

impl ParameterLayout {
    /// Constrains parameter values without a density-change determinant.
    /// Errors carry block-local indices, as with `constrain`.
    /// # Errors
    /// Rejects shape, input, domain, precision and output failures.
    pub fn constrain_values(
        &self,
        q: &[f64],
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        self.check(q, x, scratch)?;
        finite(q)?;
        for block in &self.blocks {
            block.transform.constrain_values(
                &q[block.unconstrained.clone()],
                &mut x[block.constrained.clone()],
                scratch,
            )?;
        }
        Ok(())
    }
    /// Pulls back a scalar objective gradient with no log-Jacobian contribution.
    /// # Errors
    /// Rejects shape, nonfinite input and pullback arithmetic failures.
    pub fn pullback_composition(
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
            block.transform.pullback_composition(
                &q[block.unconstrained.clone()],
                &gx[block.constrained.clone()],
                &mut gq[block.unconstrained.clone()],
                scratch,
            )?;
        }
        Ok(())
    }
}
