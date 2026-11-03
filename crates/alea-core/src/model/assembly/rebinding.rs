//! Consuming, transactional descriptor rebinding. Never mutate a borrowed target.
use super::{AssemblyError, CompiledModel, Expr, Op, Primitive, PrimitiveMetadata};
use std::{fmt, sync::Arc};

/// Failed rebinding retains the unchanged model and original typed error.
pub struct RebindError {
    /// Original model, without partial replacement.
    pub model: Box<CompiledModel>,
    /// Original descriptor-construction or validation failure.
    pub source: AssemblyError,
}
impl fmt::Debug for RebindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RebindError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for RebindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model rebinding failed: {}", self.source)
    }
}
impl std::error::Error for RebindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
impl CompiledModel {
    /// Replaces immutable data without rebuilding topology. Consumes the model:
    /// no existing target/phase cache may survive a changed density. Inputs and
    /// the full signature must remain identical. External generated code must be
    /// rebuilt from the new source; already compiled functions are not mutated.
    /// # Errors
    /// Returns the original model on a foreign/wrong-type handle, construction
    /// error or signature/layout change. Panics propagate and consume the model.
    pub fn rebind<P: Primitive>(
        mut self,
        expression: Expr,
        update: impl FnOnce(&P) -> Result<P, AssemblyError>,
    ) -> Result<Self, RebindError> {
        let prepare = || {
            if expression.owner != self.owner {
                return Err(AssemblyError::ForeignHandle);
            }
            let index = self
                .primitives
                .binary_search_by_key(&expression.node, |(i, _)| *i)
                .map_err(|_| AssemblyError::ForeignHandle)?;
            let old = &self.primitives[index].1;
            let old_descriptor = old.descriptor::<P>().ok_or(AssemblyError::Shape)?;
            let new = update(old_descriptor)?;
            let signature = new.signature();
            if new.inputs() != old_descriptor.inputs() || signature != *old.signature() {
                return Err(AssemblyError::Shape);
            }
            let static_source = new.rust_primal();
            let owned_source = new.source_rule();
            let source_workspace = new.source_workspace();
            if source_workspace > isize::MAX as usize / size_of::<f64>() {
                return Err(AssemblyError::Shape);
            }
            let batch = new.batch_rule();
            if let Some(rule) = &batch {
                rule.validate()?;
            }
            let old_units = {
                let tape = self
                    .workspace
                    .try_borrow()
                    .map_err(|_| AssemblyError::WorkspaceBusy)?;
                tape.batch_rules
                    .binary_search_by_key(&expression.node, |(node, _, _)| *node)
                    .ok()
                    .map(|i| tape.batch_rules[i].1)
            };
            if batch.as_ref().map(|rule| rule.units) != old_units {
                return Err(AssemblyError::Shape);
            }
            let mut work = new.workspace();
            let batch = batch.map(|rule| (rule, new.batch_workspace()));
            let descriptor = Arc::new(new);
            let executable = Arc::clone(&descriptor);
            let callback: Box<super::Callback> =
                Box::new(move |x, g| executable.evaluate(x, g, &mut work));
            let batch = batch.map(|(rule, mut workspace)| {
                let descriptor = Arc::clone(&descriptor);
                let callback: Box<super::BatchCallback> =
                    Box::new(move |x, out| (rule.evaluate)(&descriptor, x, out, &mut workspace));
                (expression.node, rule.units, callback)
            });
            let owned_source = owned_source.map(|source| {
                let descriptor = Arc::clone(&descriptor);
                let source: Box<super::SourceCallback> = Box::new(move || source(&descriptor));
                source
            });
            Ok((
                index,
                PrimitiveMetadata {
                    signature,
                    descriptor,
                    source_workspace,
                },
                callback,
                batch,
                static_source,
                owned_source,
            ))
        };
        let (index, descriptor, callback, batch, static_source, owned_source) = match prepare() {
            Ok(prepared) => prepared,
            Err(source) => {
                return Err(RebindError {
                    model: Box::new(self),
                    source,
                });
            }
        };
        let Op::Factor { callback: slot, .. } = self.nodes[expression.node] else {
            unreachable!("registered primitive")
        };
        let workspace = self.workspace.get_mut();
        workspace.callbacks[slot] = callback;
        workspace
            .batch_rules
            .retain(|(node, _, _)| *node != expression.node);
        if let Some(batch) = batch {
            workspace.batch_rules.push(batch);
            workspace.batch_rules.sort_by_key(|(node, _, _)| *node);
        }
        self.primitives[index].1 = descriptor;
        self.primals.retain(|(node, _)| *node != expression.node);
        self.generated_primals
            .retain(|(node, _)| *node != expression.node);
        if let Some(source) = static_source {
            self.primals.push((expression.node, source));
            self.primals.sort_by_key(|(node, _)| *node);
        }
        if let Some(source) = owned_source {
            self.generated_primals.push((expression.node, source));
            self.generated_primals.sort_by_key(|(node, _)| *node);
        }
        Ok(self)
    }
}
