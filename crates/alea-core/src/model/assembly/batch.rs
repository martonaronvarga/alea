//! Explicit primitive rules lower joint native batches to conditional row values.
use super::{AssemblyError, CompiledModel, Op};
use std::cell::RefCell;

/// Local batch coordinate, not a globally unique observation ID or a marginal
/// leave-out specification. Callers must align study IDs across fitted models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchUnit {
    /// Named likelihood factor in original declaration order.
    pub factor: String,
    /// Original zero-based row within that factor.
    pub row: usize,
}
struct Workspace {
    parameters: Vec<f64>,
    chart: Vec<f64>,
    values: Vec<f64>,
}
/// Automatic conditional batch likelihood reporting, separate from the hot
/// gradient loop. Only declared primitive rules may split factors. Arbitrary
/// joint/opaque factors remain unsupported, rather than invented independent rows.
pub struct BatchLikelihood {
    model: CompiledModel,
    units: Vec<BatchUnit>,
    roots: Vec<(usize, usize, usize)>,
    workspace: RefCell<Workspace>,
}
impl CompiledModel {
    /// Selects the likelihood and lowers author-supplied batch rules in factor/
    /// row order. Excludes priors, arbitrary potentials and chart Jacobians.
    /// # Errors
    /// Rejects unclassified/empty selections, missing rules and size overflow.
    pub fn batch_likelihood(self) -> Result<BatchLikelihood, AssemblyError> {
        let model = self.select_likelihood()?;
        let mut roots = Vec::new();
        let mut units = Vec::new();
        {
            let tape = model.workspace.borrow();
            for (root, metadata) in model.factors.iter().zip(&model.factor_metadata) {
                if metadata.role() != super::FactorRole::Likelihood {
                    continue;
                }
                let index = tape
                    .batch_rules
                    .binary_search_by_key(root, |(node, _, _)| *node)
                    .map_err(|_| AssemblyError::MissingPointwise { node: *root })?;
                let count = tape.batch_rules[index].1;
                let start = units.len();
                start
                    .checked_add(count)
                    .filter(|&n| n <= isize::MAX as usize / size_of::<BatchUnit>())
                    .ok_or(AssemblyError::Shape)?;
                for row in 0..count {
                    units.push(BatchUnit {
                        factor: metadata.name().expect("named likelihood").into(),
                        row,
                    });
                }
                roots.push((*root, index, start));
            }
        }
        let workspace = RefCell::new(Workspace {
            parameters: vec![0.; model.layout.constrained_dimension()],
            chart: vec![0.; model.layout.scratch_dimension()],
            values: vec![0.; units.len()],
        });
        Ok(BatchLikelihood {
            model,
            units,
            roots,
            workspace,
        })
    }
}
impl BatchLikelihood {
    /// Ordered local row coordinates; conditional, not integrated likelihoods.
    pub fn units(&self) -> &[BatchUnit] {
        &self.units
    }
    /// Original graph and retained semantic descriptors.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }
    /// Recovers the model for another selection or consuming data rebinding.
    pub fn into_model(self) -> CompiledModel {
        self.model
    }
    /// Evaluates in free coordinates without density Jacobians. Rows are staged
    /// and finite-checked; errors or panics publish nothing. A forward graph pass
    /// admits required native values/scores, followed by explicit row lowering.
    /// # Errors
    /// Preserves chart/primitive failures, shape and reentrant-workspace errors.
    pub fn evaluate(&self, q: &[f64], output: &mut [f64]) -> Result<f64, AssemblyError> {
        if q.len() != self.model.layout.unconstrained_dimension()
            || output.len() != self.units.len()
        {
            return Err(AssemblyError::Shape);
        }
        if q.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        let mut work = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| AssemblyError::WorkspaceBusy)?;
        let Workspace {
            parameters,
            chart,
            values,
        } = &mut *work;
        self.model.layout.constrain_values(q, parameters, chart)?;
        let mut tape = self
            .model
            .workspace
            .try_borrow_mut()
            .map_err(|_| AssemblyError::WorkspaceBusy)?;
        self.model.forward::<true>(parameters, &mut tape)?;
        for &(root, rule, start) in &self.roots {
            let Op::Factor {
                inputs: indices, ..
            } = &self.model.nodes[root]
            else {
                unreachable!("batch rule root")
            };
            let super::PlanWorkspace {
                inputs,
                values: tape_values,
                batch_rules,
                ..
            } = &mut *tape;
            for (out, &index) in inputs.iter_mut().zip(indices.iter()) {
                *out = tape_values[index];
            }
            let (_, count, callback) = &mut batch_rules[rule];
            let out = &mut values[start..start + *count];
            out.fill(f64::NAN);
            callback(&inputs[..indices.len()], out)?;
        }
        let mut sum = alea_math::numeric::CompensatedSum::default();
        for &value in values.iter() {
            sum.add(value);
        }
        let total = sum.total();
        if !total.is_finite() || values.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical);
        }
        output.copy_from_slice(values);
        Ok(total)
    }
}
