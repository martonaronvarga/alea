//! Lexically scoped naming for ordinary reusable Rust model functions.
use super::{AssemblyError, ModelBuilder};
use std::ops::{Deref, DerefMut};

/// A borrowed builder namespace. Dropping it restores the enclosing namespace,
/// including during unwinding. Expressions may escape; graph construction is
/// never rolled back. Explicit dimension/coordinate/block annotations are unchanged.
#[must_use = "dropping the scope restores the enclosing namespace"]
pub struct ModelScope<'a> {
    builder: &'a mut ModelBuilder,
    previous_len: usize,
}
impl ModelBuilder {
    /// Builds a reusable component in a temporary namespace and returns its
    /// handles/result. Restores names after success, error or unwind, without
    /// rolling back graph declarations. Use [`Self::scope`] for a longer-lived guard.
    /// # Errors
    /// Rejects an invalid scope name or propagates the component's original error.
    pub fn with_scope<T>(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut Self) -> Result<T, AssemblyError>,
    ) -> Result<T, AssemblyError> {
        let mut scope = self.scope(name)?;
        build(&mut scope)
    }
    /// Enters a naming scope for parameters and factors, e.g. `group::mean`.
    /// Nested scopes compose. Use ordinary functions taking `&mut ModelBuilder`
    /// to build reusable submodels; returned handles remain owned by this builder.
    ///
    /// Scope components must be nonempty and cannot contain `::`. This does not
    /// reserve names: redeclaration is still checked by the existing builder.
    /// Failed construction may leave unused nodes; discard a failed builder.
    /// # Errors
    /// Rejects an empty component or the namespace separator before mutation.
    pub fn scope(&mut self, name: &str) -> Result<ModelScope<'_>, AssemblyError> {
        if name.is_empty() || name.contains("::") {
            return Err(AssemblyError::Name);
        }
        let previous_len = self.namespace.len();
        self.namespace.push_str(name);
        self.namespace.push_str("::");
        Ok(ModelScope {
            builder: self,
            previous_len,
        })
    }
    pub(super) fn qualified_name(&self, name: &str) -> String {
        format!("{}{name}", self.namespace)
    }
}
impl Deref for ModelScope<'_> {
    type Target = ModelBuilder;
    fn deref(&self) -> &Self::Target {
        self.builder
    }
}
impl DerefMut for ModelScope<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.builder
    }
}
impl Drop for ModelScope<'_> {
    fn drop(&mut self) {
        self.builder.namespace.truncate(self.previous_len);
    }
}
