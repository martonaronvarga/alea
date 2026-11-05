//! Named native factors without coupling core's builder to distributions.
use super::{
    bernoulli_logit, normal,
    regression::{self, RegressionData},
};
use alea_core::model::assembly::{
    AssemblyError, Expr, ModelBuilder, Parameter, Positive, ScalarInput,
};

/// Distribution conveniences for [`ModelBuilder`], also imported by Alea's prelude.
///
/// Methods attach exactly one named prior or likelihood using existing native
/// factors. They infer neither priors nor statistical independence. On failure no
/// named factor is attached, but unused construction expressions may remain;
/// discard an unfinished builder if construction fails. For custom distributions
/// use the free factor constructors and explicit `prior`/`likelihood` methods.
pub trait ModelBuilderExt {
    /// Attaches a normalized Normal prior in physical parameter coordinates.
    /// A positive event's truncation normalizer is **not** inferred.
    /// # Errors
    /// Rejects foreign inputs, nonfinite literals and invalid/duplicate names.
    /// Runtime Normal parameter and numerical errors remain typed.
    fn normal_prior(
        &mut self,
        name: &str,
        event: impl Into<ScalarInput>,
        location: impl Into<ScalarInput>,
        scale: impl Into<ScalarInput>,
    ) -> Result<(), AssemblyError>;

    /// Attaches one normalized Normal likelihood term (not a batch).
    /// # Errors
    /// Same input/name/runtime checks as [`Self::normal_prior`].
    fn normal_likelihood(
        &mut self,
        name: &str,
        event: impl Into<ScalarInput>,
        location: impl Into<ScalarInput>,
        scale: impl Into<ScalarInput>,
    ) -> Result<(), AssemblyError>;

    /// Attaches one stable Bernoulli-logit likelihood for a typed boolean event.
    /// # Errors
    /// Rejects foreign/nonfinite inputs or invalid/duplicate names. Runtime
    /// distribution errors are preserved.
    fn bernoulli_logit_likelihood(
        &mut self,
        name: &str,
        event: bool,
        logit: impl Into<ScalarInput>,
    ) -> Result<(), AssemblyError>;

    /// A fused shared-location/scale Normal likelihood for a nonempty data slice.
    /// Uses the existing native regression kernel with an explicit unit column;
    /// does not expand one graph node per row. The whole data block is ONE factor,
    /// not an automatically declared pointwise/leave-out unit. The retained native
    /// descriptor supplies explicit conditional rows and counted source lowering;
    /// actual compiler derivatives are a separate qualification.
    /// # Errors
    /// Rejects empty/nonfinite observations, foreign handles and invalid names.
    /// Runtime scale/numerical errors retain their native sources.
    fn normal_observations(
        &mut self,
        name: &str,
        observations: &[f64],
        location: impl Into<ScalarInput>,
        scale: &Parameter<Positive>,
    ) -> Result<(), AssemblyError>;

    /// A Normal observation map with vector locations and/or scales. Each input
    /// slice has length one or the observation count. Lowers directly to the
    /// native broadcast/SIMD primitive, not individual scalar graph callbacks.
    /// Conditional rows remain available through explicit batch reporting.
    /// # Errors
    /// Rejects incompatible/empty/nonfinite data, foreign handles and bad names;
    /// retains all scalar score admission even for fixed observations.
    fn normal_vector_likelihood(
        &mut self,
        name: &str,
        observations: &[f64],
        locations: &[Expr],
        scales: &[Expr],
    ) -> Result<(), AssemblyError>;

    /// Attaches the existing fused Normal regression likelihood; no priors,
    /// intercept, standardization or pointwise units are inferred.
    /// # Errors
    /// Preserves native shape/handle/response/scale errors and name checks.
    fn normal_regression_likelihood(
        &mut self,
        name: &str,
        beta: &[Expr],
        scale: &Parameter<Positive>,
        data: RegressionData<f64>,
    ) -> Result<(), AssemblyError>;

    /// Attaches the native fused Bernoulli-logit regression likelihood.
    /// # Errors
    /// Preserves native shape/handle/numerical errors and name checks.
    fn bernoulli_logit_regression_likelihood(
        &mut self,
        name: &str,
        beta: &[Expr],
        data: RegressionData<bool>,
    ) -> Result<(), AssemblyError>;

    /// Attaches native fused Poisson-log regression. Data offsets remain log
    /// exposures; no intercept, exposure conversion or prior is added.
    /// # Errors
    /// Preserves native shape/count-precision/handle/numerical errors and names.
    fn poisson_log_regression_likelihood(
        &mut self,
        name: &str,
        beta: &[Expr],
        data: RegressionData<u64>,
    ) -> Result<(), AssemblyError>;
}

impl ModelBuilderExt for ModelBuilder {
    fn normal_prior(
        &mut self,
        name: &str,
        event: impl Into<ScalarInput>,
        location: impl Into<ScalarInput>,
        scale: impl Into<ScalarInput>,
    ) -> Result<(), AssemblyError> {
        let factor = normal(self, event, location, scale)?;
        self.prior(name, factor)
    }
    fn normal_likelihood(
        &mut self,
        name: &str,
        event: impl Into<ScalarInput>,
        location: impl Into<ScalarInput>,
        scale: impl Into<ScalarInput>,
    ) -> Result<(), AssemblyError> {
        let factor = normal(self, event, location, scale)?;
        self.likelihood(name, factor)
    }
    fn bernoulli_logit_likelihood(
        &mut self,
        name: &str,
        event: bool,
        logit: impl Into<ScalarInput>,
    ) -> Result<(), AssemblyError> {
        let factor = bernoulli_logit(self, event, logit)?;
        self.likelihood(name, factor)
    }
    fn normal_observations(
        &mut self,
        name: &str,
        observations: &[f64],
        location: impl Into<ScalarInput>,
        scale: &Parameter<Positive>,
    ) -> Result<(), AssemblyError> {
        let data = RegressionData::new(1, vec![1.; observations.len()], observations.to_vec())
            .map_err(super::error)?;
        let location = self.expression(location)?;
        self.normal_regression_likelihood(name, &[location], scale, data)
    }
    fn normal_vector_likelihood(
        &mut self,
        name: &str,
        observations: &[f64],
        locations: &[Expr],
        scales: &[Expr],
    ) -> Result<(), AssemblyError> {
        let primitive = super::NormalBatch::observed(observations.to_vec(), locations, scales)?;
        let factor = self.primitive(primitive)?;
        self.likelihood(name, factor)
    }
    fn normal_regression_likelihood(
        &mut self,
        name: &str,
        beta: &[Expr],
        scale: &Parameter<Positive>,
        data: RegressionData<f64>,
    ) -> Result<(), AssemblyError> {
        let factor = regression::normal_regression(self, beta, scale, data)?;
        self.likelihood(name, factor)
    }
    fn bernoulli_logit_regression_likelihood(
        &mut self,
        name: &str,
        beta: &[Expr],
        data: RegressionData<bool>,
    ) -> Result<(), AssemblyError> {
        let factor = regression::bernoulli_logit_regression(self, beta, data)?;
        self.likelihood(name, factor)
    }
    fn poisson_log_regression_likelihood(
        &mut self,
        name: &str,
        beta: &[Expr],
        data: RegressionData<u64>,
    ) -> Result<(), AssemblyError> {
        let factor = regression::poisson_log_regression(self, beta, data)?;
        self.likelihood(name, factor)
    }
}
