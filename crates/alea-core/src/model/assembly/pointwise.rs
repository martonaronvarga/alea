//! Explicit predictive partitions over the declared likelihood factors.
use super::{AssemblyError, CompiledModel, FactorRole};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

/// A declared predictive unit, which may contain a joint block of observations.
///
/// This is a specification validated by [`PointwiseLikelihood::new`], not proof
/// of conditional independence, normalization or analytic marginalization.
/// Observation identifiers must refer to the same data across compared models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictiveUnit {
    /// Unique output coordinate, independent of the factor names.
    pub name: String,
    /// Named likelihood factors whose sum defines this unit's log likelihood.
    pub factors: Vec<String>,
    /// Nonempty, disjoint observed-data identifiers represented by this unit.
    pub observations: Vec<String>,
    /// Variables held conditional in this likelihood, such as subject effects.
    /// These are semantic labels, not necessarily graph parameter names.
    pub conditioned_on: Vec<String>,
    /// Variables already integrated out by the supplied likelihood calculation.
    /// Declaring them here performs no integration.
    pub integrated_out: Vec<String>,
}

/// Invalid predictive partitions or numerical evaluation failures.
#[derive(Debug, thiserror::Error)]
pub enum PointwiseError {
    /// A unit declaration is empty, duplicated or internally inconsistent.
    #[error("invalid predictive unit {unit}: {reason}")]
    Unit {
        /// Unit index in the requested output order.
        unit: usize,
        /// Description of the invalid declaration.
        reason: &'static str,
    },
    /// A factor is unknown or is not explicitly declared as a likelihood.
    #[error("unit {unit} names an unknown or non-likelihood factor {factor:?}")]
    Factor {
        /// Unit index.
        unit: usize,
        /// Invalid factor name.
        factor: String,
    },
    /// Every likelihood factor must occur in exactly one predictive unit.
    #[error("likelihood factor {factor:?} is missing or assigned more than once")]
    Partition {
        /// Missing or repeated factor name.
        factor: String,
    },
    /// Observations cannot overlap across predictive units.
    #[error("observation {observation:?} occurs more than once")]
    Observation {
        /// Repeated observation identifier.
        observation: String,
    },
    /// Original model, transform, shape, workspace or callback failure.
    #[error(transparent)]
    Model(#[from] AssemblyError),
}

struct Workspace {
    parameters: Vec<f64>,
    scratch: Vec<f64>,
    units: Vec<f64>,
}

/// Allocation-free pointwise likelihood reporting in free parameter coordinates.
///
/// Owns one model and local scratch. Excludes priors, potentials and transform
/// Jacobians. Each declared likelihood factor must be assigned exactly once;
/// joint factors are never split into invented observation-level values. The
/// original graph remains recoverable for posterior evaluation or sampling.
#[must_use]
pub struct PointwiseLikelihood {
    model: CompiledModel,
    units: Vec<PredictiveUnit>,
    nodes: Vec<Vec<usize>>,
    workspace: RefCell<Workspace>,
}

impl PointwiseLikelihood {
    /// Validates a complete, nonoverlapping partition of named likelihood terms.
    ///
    /// Unit order defines output order; terms within a unit are summed in graph
    /// declaration order. All original parameter coordinates remain present.
    /// # Errors
    /// Rejects unclassified factors, absent likelihoods, invalid names, incomplete
    /// or overlapping factor/observation assignments and ambiguous conditioning.
    pub fn new(model: CompiledModel, units: Vec<PredictiveUnit>) -> Result<Self, PointwiseError> {
        let model = model.select_likelihood()?;
        let factors: HashMap<_, _> = model
            .factor_metadata
            .iter()
            .enumerate()
            .filter(|(_, f)| f.role() == FactorRole::Likelihood)
            .map(|(i, f)| (f.name().expect("named likelihood"), i))
            .collect();
        let mut assigned = HashSet::new();
        let mut observations = HashSet::new();
        let mut names = HashSet::new();
        let mut nodes = Vec::with_capacity(units.len());
        for (unit, spec) in units.iter().enumerate() {
            let invalid = |reason| PointwiseError::Unit { unit, reason };
            if spec.name.is_empty() || !names.insert(spec.name.as_str()) {
                return Err(invalid("empty or duplicate unit name"));
            }
            if spec.factors.is_empty() || spec.observations.is_empty() {
                return Err(invalid("factors and observations must both be nonempty"));
            }
            let mut variables = HashSet::new();
            for name in spec.conditioned_on.iter().chain(&spec.integrated_out) {
                if name.is_empty() || !variables.insert(name.as_str()) {
                    return Err(invalid(
                        "empty, duplicate or contradictory conditioning variable",
                    ));
                }
            }
            for observation in &spec.observations {
                if observation.is_empty() {
                    return Err(invalid("empty observation identifier"));
                }
                if !observations.insert(observation.as_str()) {
                    return Err(PointwiseError::Observation {
                        observation: observation.clone(),
                    });
                }
            }
            let mut indices = Vec::with_capacity(spec.factors.len());
            for name in &spec.factors {
                let Some(&index) = factors.get(name.as_str()) else {
                    return Err(PointwiseError::Factor {
                        unit,
                        factor: name.clone(),
                    });
                };
                if !assigned.insert(index) {
                    return Err(PointwiseError::Partition {
                        factor: name.clone(),
                    });
                }
                indices.push(index);
            }
            indices.sort_unstable();
            nodes.push(indices.into_iter().map(|i| model.factors[i]).collect());
        }
        // Declaration order makes the first missing-factor error deterministic.
        for (index, factor) in model.factor_metadata.iter().enumerate() {
            if factor.role() == FactorRole::Likelihood && !assigned.contains(&index) {
                return Err(PointwiseError::Partition {
                    factor: factor.name().expect("named likelihood").into(),
                });
            }
        }
        let workspace = RefCell::new(Workspace {
            parameters: vec![0.; model.layout.constrained_dimension()],
            scratch: vec![0.; model.layout.scratch_dimension()],
            units: vec![0.; units.len()],
        });
        Ok(Self {
            model,
            units,
            nodes,
            workspace,
        })
    }

    /// Validated unit metadata in output order.
    pub fn units(&self) -> &[PredictiveUnit] {
        &self.units
    }

    /// The original graph, with its likelihood selection active.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    /// Recovers the owned graph; excluded factors remain available for selection.
    pub fn into_model(self) -> CompiledModel {
        self.model
    }

    /// Evaluates each unit and returns the complete likelihood in declaration order.
    ///
    /// `q` contains free coordinates; `output` has one slot per unit. Unit sums
    /// may differ from the returned total by floating-point regrouping. The tape
    /// runs once without reverse propagation; opaque callbacks still supply
    /// their existing value/VJP contract. No evaluation allocates unless a callback
    /// does. Errors and unwinding leave caller output untouched.
    /// # Errors
    /// Rejects shape mismatches, nonfinite inputs/values (including support zeros
    /// represented by negative infinity), invalid charts and callback failures.
    /// Does not substitute a numerical failure with a zero probability.
    pub fn evaluate(&self, q: &[f64], output: &mut [f64]) -> Result<f64, PointwiseError> {
        if q.len() != self.model.layout.unconstrained_dimension()
            || output.len() != self.units.len()
        {
            return Err(AssemblyError::Shape.into());
        }
        if q.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical.into());
        }
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| AssemblyError::WorkspaceBusy)?;
        let Workspace {
            parameters,
            scratch,
            units,
        } = &mut *workspace;
        self.model
            .layout
            .constrain_values(q, parameters, scratch)
            .map_err(AssemblyError::from)?;
        let mut tape = self
            .model
            .workspace
            .try_borrow_mut()
            .map_err(|_| AssemblyError::WorkspaceBusy)?;
        self.model.forward::<true>(parameters, &mut tape)?;
        for (sum, indices) in units.iter_mut().zip(&self.nodes) {
            *sum = indices.iter().fold(0., |sum, &i| sum + tape.values[i]);
        }
        let total: f64 = self
            .model
            .retained_factors()
            .iter()
            .fold(0., |sum, &i| sum + tape.values[i]);
        if !total.is_finite() || units.iter().any(|x| !x.is_finite()) {
            return Err(AssemblyError::Numerical.into());
        }
        output.copy_from_slice(units);
        Ok(total)
    }
}
