//! Typed statistical map/reduce lowering. No scalar observation tape is built.
use super::*;

/// Retained affine observation map `eta = X beta + offsets`. Data rows remain
/// conditional observations, not inferred marginal leave-out units. Construction
/// preserves mathematical structure until an existing native law is selected.
pub struct LinearPredictor<Y> {
    coefficients: Box<[Value]>,
    data: RegressionData<Y>,
}
impl<Y> RegressionData<Y> {
    /// Describes an affine observation map without creating per-row expressions.
    /// # Errors
    /// Rejects coefficient-count mismatch. Ownership is checked during lowering.
    pub fn linear_map(self, coefficients: &[Value]) -> Result<LinearPredictor<Y>, E> {
        check_columns(&self, coefficients)?;
        Ok(LinearPredictor {
            coefficients: coefficients.into(),
            data: self,
        })
    }
}
/// Typed conditional log-probability map awaiting a sum reduction. Lowering
/// selects the law's existing fused native implementation and CPU plan; it does
/// not transform arbitrary closures or add priors, roles or Jacobians.
pub struct ObservationMap<P: OperationDescriptor> {
    inputs: Box<[Value]>,
    primitive: P,
}
impl<P: OperationDescriptor> ObservationMap<P> {
    /// Retained semantic descriptor, including data, shape and selected CPU plan.
    pub fn primitive(&self) -> &P {
        &self.primitive
    }
    /// Lowers the statistical map/sum as one native graph node, not N callbacks.
    /// Explicitly attach the result as a named likelihood using the builder.
    /// # Errors
    /// Preserves primitive ownership/signature/registration errors.
    pub fn sum_log_prob(self, builder: &mut ModelBuilder) -> Result<Value, E> {
        builder.primitive(&self.inputs, self.primitive)
    }
}
impl LinearPredictor<bool> {
    /// Applies the existing stable Bernoulli-logit law to each conditional row.
    /// # Errors
    /// Preserves native descriptor validation errors.
    pub fn bernoulli_logit(self) -> Result<ObservationMap<BernoulliLogitRegression>, E> {
        Ok(ObservationMap {
            inputs: self.coefficients,
            primitive: BernoulliLogitRegression::new(self.data)?,
        })
    }
}
impl LinearPredictor<u64> {
    /// Applies Poisson-log; fixed offsets are log exposures, not exposures.
    /// # Errors
    /// Preserves native count precision and descriptor validation errors.
    pub fn poisson_log(self) -> Result<ObservationMap<PoissonLogRegression>, E> {
        Ok(ObservationMap {
            inputs: self.coefficients,
            primitive: PoissonLogRegression::new(self.data)?,
        })
    }
}
impl LinearPredictor<f64> {
    /// Applies the existing Normal law with a shared positive residual scale.
    /// # Errors
    /// Preserves native response and descriptor validation errors.
    pub fn normal(
        self,
        scale: &Parameter<Positive>,
    ) -> Result<ObservationMap<NormalRegression>, E> {
        let mut inputs = self.coefficients.into_vec();
        inputs.push(scale.expr());
        Ok(ObservationMap {
            inputs: inputs.into(),
            primitive: NormalRegression::new(self.data)?,
        })
    }
}
