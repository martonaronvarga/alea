//! Immutable binding replacement. Failure leaves every old snapshot usable.
use super::primitives::{Registration, ReportingConstraint};
use super::{AssemblyError, DataBinding, Op, OperationDescriptor, Value, next_value_owner};

impl DataBinding {
    /// Creates a same-schema snapshot with replacement region data. Previously
    /// compiled executables and their contexts remain bound to the old snapshot.
    /// # Errors
    /// Rejects a foreign handle, missing data slot, invalid shape or native data.
    pub fn rebind_region(
        &self,
        value: Value,
        name: &str,
        values: &[f64],
    ) -> Result<Self, AssemblyError> {
        if value.owner != self.owner {
            return Err(AssemblyError::ForeignHandle);
        }
        if let Some(Op::Region(call)) = self.nodes.get(value.node) {
            let binding = self.regions[call.slot].rebind(name, values)?;
            let mut next = self.clone();
            next.regions[call.slot] = binding;
            next.identity = next_value_owner();
            return Ok(next);
        }
        self.rebind(value, |region: &crate::model::tensor::TensorFactor| {
            region.with_data(name, values)
        })
    }

    /// Replaces a typed descriptor through the same schema/rule validator used
    /// at registration. Immutable captures feed both native execution and source
    /// specialization; parameter-dependent scratch is never copied or rebound.
    /// A new identity is issued even for identical replacement values.
    /// # Errors
    /// Preserves typed construction errors and rejects foreign handles, descriptor
    /// types, argument signatures or reporting layouts. The old binding survives.
    pub fn rebind<P: OperationDescriptor>(
        &self,
        expression: Value,
        update: impl FnOnce(&P) -> Result<P, AssemblyError>,
    ) -> Result<Self, AssemblyError> {
        if expression.owner != self.owner {
            return Err(AssemblyError::ForeignHandle);
        }
        let index = self
            .primitives
            .binary_search_by_key(&expression.node, |(node, _)| *node)
            .map_err(|_| AssemblyError::ForeignHandle)?;
        let old = &self.primitives[index].1;
        let descriptor = old.descriptor::<P>().ok_or(AssemblyError::Shape)?;
        let replacement = update(descriptor)?;
        let rules = replacement.rules();
        if rules.signature != *old.signature() {
            return Err(AssemblyError::Shape);
        }
        let Op::Factor {
            inputs,
            callback: slot,
            ..
        } = &self.nodes[expression.node]
        else {
            unreachable!("registered primitive")
        };
        let units = self
            .batch_rules
            .binary_search_by_key(&expression.node, |(node, _, _)| *node)
            .ok()
            .map(|i| self.batch_rules[i].1);
        let Registration {
            metadata,
            callback,
            batch,
            primal,
            source,
        } = Registration::prepare(
            replacement,
            rules,
            inputs.len(),
            ReportingConstraint::SameUnits(units),
        )?;
        let mut next = self.clone();
        next.callbacks[*slot] = callback;
        next.primitives[index].1 = metadata;
        next.batch_rules
            .retain(|(node, _, _)| *node != expression.node);
        if let Some((units, make)) = batch {
            next.batch_rules.push((expression.node, units, make));
        }
        next.batch_rules.sort_by_key(|(node, _, _)| *node);
        next.primals.retain(|(node, _)| *node != expression.node);
        next.generated_primals
            .retain(|(node, _)| *node != expression.node);
        if let Some(primal) = primal {
            next.primals.push((expression.node, primal));
        }
        if let Some(source) = source {
            next.generated_primals.push((expression.node, source));
        }
        next.primals.sort_by_key(|(node, _)| *node);
        next.generated_primals.sort_by_key(|(node, _)| *node);
        next.identity = next_value_owner();
        Ok(next)
    }
}
