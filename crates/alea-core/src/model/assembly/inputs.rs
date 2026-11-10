//! Cold ergonomic inputs; the compiled operation tape is unchanged.
use super::{
    AssemblyError, ModelBuilder, Op, Parameter, Positive, Real, UnitInterval, Value,
    VectorParameter,
};
use crate::transform::Transform;

/// A scalar model expression or a literal constant, never a new parameter.
/// Parameter conversions retain their original builder identity and chart.
/// Only `f64` constants are accepted: discrete events keep their own types.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ScalarInput {
    /// An existing expression, checked for builder ownership before use.
    Expression(Value),
    /// A finite constant, inserted as an ordinary constant node on use.
    Constant(f64),
}
impl From<f64> for ScalarInput {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}
impl From<Value> for ScalarInput {
    fn from(value: Value) -> Self {
        Self::Expression(value)
    }
}
impl From<&Value> for ScalarInput {
    fn from(value: &Value) -> Self {
        Self::Expression(*value)
    }
}
impl<S> From<Parameter<S>> for ScalarInput {
    fn from(value: Parameter<S>) -> Self {
        Self::Expression(value.expr())
    }
}
impl<S> From<&Parameter<S>> for ScalarInput {
    fn from(value: &Parameter<S>) -> Self {
        Self::Expression(value.expr())
    }
}
impl ModelBuilder {
    /// Resolves a literal, expression or parameter handle into this graph.
    /// Existing expressions are reused; literals add a constant node.
    /// # Errors
    /// Rejects foreign handles or nonfinite literals before graph mutation.
    pub fn expression(&mut self, value: impl Into<ScalarInput>) -> Result<Value, AssemblyError> {
        let [value] = self.inputs([value.into()])?;
        Ok(value)
    }

    /// Resolves a fixed-size argument list after validating **all** arguments.
    /// Useful for distribution adapters: a late foreign handle/nonfinite literal
    /// cannot leave constant nodes from earlier arguments behind. Reusing an
    /// expression never duplicates it. This is construction work only.
    /// # Errors
    /// Rejects foreign handles/nonfinite literals without adding any nodes.
    pub fn inputs<const N: usize>(
        &mut self,
        values: [ScalarInput; N],
    ) -> Result<[Value; N], AssemblyError> {
        for value in &values {
            match value {
                ScalarInput::Expression(expr) => {
                    self.index(*expr)?;
                }
                ScalarInput::Constant(value) if !value.is_finite() => {
                    return Err(AssemblyError::Numerical);
                }
                ScalarInput::Constant(_) => {}
            }
        }
        Ok(values.map(|value| match value {
            ScalarInput::Expression(expr) => expr,
            ScalarInput::Constant(value) => self.push(Op::Constant(value)),
        }))
    }

    /// Named real vector with its scoped name as dimension and geometry-block label.
    /// Use [`Self::vector`] for explicit metadata or other charts.
    /// # Errors
    /// Rejects empty/duplicate labels or parameter names before mutation.
    pub fn real_vector(
        &mut self,
        name: &str,
        labels: &[&str],
    ) -> Result<VectorParameter<Real>, AssemblyError> {
        let metadata = self.qualified_name(name);
        self.vector(
            name,
            &metadata,
            labels,
            &metadata,
            Transform::identity(labels.len())?,
        )
    }
    /// Named positive vector; applies one exponential chart to the whole block.
    /// Dimension and geometry-block labels default to the scoped name.
    /// # Errors
    /// Rejects empty/duplicate labels or parameter names before mutation.
    pub fn positive_vector(
        &mut self,
        name: &str,
        labels: &[&str],
    ) -> Result<VectorParameter<Positive>, AssemblyError> {
        let metadata = self.qualified_name(name);
        self.vector(
            name,
            &metadata,
            labels,
            &metadata,
            Transform::positive(labels.len())?,
        )
    }
    /// Named unit-interval vector with its scoped name as dimension/geometry block.
    /// # Errors
    /// Rejects empty/duplicate labels or parameter names before mutation.
    pub fn unit_interval_vector(
        &mut self,
        name: &str,
        labels: &[&str],
    ) -> Result<VectorParameter<UnitInterval>, AssemblyError> {
        let metadata = self.qualified_name(name);
        self.vector(
            name,
            &metadata,
            labels,
            &metadata,
            Transform::interval(labels.len(), 0., 1.)?,
        )
    }
}
