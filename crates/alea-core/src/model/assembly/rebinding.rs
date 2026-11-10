//! Consuming, transactional descriptor rebinding. Never mutate a borrowed target.
use super::primitives::{Registration, ReportingConstraint};
use super::{AssemblyError, CompiledModel, Op, OperationDescriptor, Value};
use std::fmt;

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
    /// Replaces named data in a registered region with a same-shape snapshot.
    /// # Errors
    /// Returns the original model on a wrong handle or invalid replacement.
    pub fn rebind_region(
        mut self,
        value: Value,
        name: &str,
        values: &[f64],
    ) -> Result<Self, RebindError> {
        if value.owner == self.owner
            && let Some(Op::Region(call)) = self.nodes.get_mut(value.node)
        {
            return match call.binding.rebind(name, values) {
                Ok(binding) => {
                    call.binding = binding;
                    Ok(self)
                }
                Err(source) => Err(RebindError {
                    model: Box::new(self),
                    source,
                }),
            };
        }
        self.rebind(value, |region: &crate::model::tensor::TensorFactor| {
            region.with_data(name, values)
        })
    }

    /// Replaces immutable data without rebuilding topology. Consumes the model:
    /// no existing target/phase cache may survive a changed density. The graph's
    /// ordered inputs stay fixed and the full signature must remain identical.
    /// External generated code must be
    /// rebuilt from the new source; already compiled functions are not mutated.
    /// # Errors
    /// Returns the original model on a foreign/wrong-type handle, construction
    /// error or signature/layout change. Panics propagate and consume the model.
    pub fn rebind<P: OperationDescriptor>(
        mut self,
        expression: Value,
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
            let rules = new.rules();
            if rules.signature != *old.signature() {
                return Err(AssemblyError::Shape);
            }
            let Op::Factor { inputs, .. } = &self.nodes[expression.node] else {
                unreachable!("registered primitive")
            };
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
            let Registration {
                metadata,
                callback,
                batch,
                primal,
                source,
            } = Registration::prepare(
                new,
                rules,
                inputs.len(),
                ReportingConstraint::SameUnits(old_units),
            )?;
            Ok((
                index,
                metadata,
                callback,
                batch.map(|(units, callback)| (expression.node, units, callback)),
                primal,
                source,
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
        workspace.preparation =
            super::prepare::Plan::new(&self.nodes, &self.primitives, workspace.callbacks.len());
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
