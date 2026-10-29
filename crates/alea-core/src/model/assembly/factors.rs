//! Statistical factor declarations and explicit, consuming graph selections.
use super::{AssemblyError, CompiledModel, Expr, ModelBuilder, Op};

/// Declared statistical meaning. This is a caller contract, not a proof that a
/// custom callback is normalized or that a posterior is proper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorRole {
    /// Generic `factor` calls have no inferred statistical interpretation.
    Unclassified,
    /// Observational likelihood term, including its stated normalizing terms.
    Likelihood,
    /// Prior density relative to the model's declared parameter measure.
    Prior,
    /// Arbitrary penalty/log potential, without an automatic prior interpretation.
    Potential,
}
/// Cold, immutable factor metadata in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorMetadata {
    name: Option<String>,
    role: FactorRole,
}
impl FactorMetadata {
    pub(super) fn unclassified() -> Self {
        Self {
            name: None,
            role: FactorRole::Unclassified,
        }
    }
    /// Declared label; generic factors have no label.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// Declared statistical interpretation.
    pub fn role(&self) -> FactorRole {
        self.role
    }
}
impl ModelBuilder {
    /// Adds a named likelihood term. A term can represent an observation or a
    /// joint data block; the name alone does not declare a leave-out unit.
    /// # Errors
    /// Rejects foreign handles and empty or duplicate factor names before mutation.
    pub fn likelihood(&mut self, name: &str, factor: Expr) -> Result<(), AssemblyError> {
        self.named_factor(name, factor, FactorRole::Likelihood)
    }
    /// Adds a named prior density in the physical parameter measure.
    /// # Errors
    /// Rejects foreign handles and empty or duplicate names before mutation.
    pub fn prior(&mut self, name: &str, factor: Expr) -> Result<(), AssemblyError> {
        self.named_factor(name, factor, FactorRole::Prior)
    }
    /// Adds a named arbitrary penalty/potential, without claiming it is a prior.
    /// # Errors
    /// Rejects foreign handles and empty or duplicate names before mutation.
    pub fn potential(&mut self, name: &str, factor: Expr) -> Result<(), AssemblyError> {
        self.named_factor(name, factor, FactorRole::Potential)
    }
    fn named_factor(
        &mut self,
        name: &str,
        factor: Expr,
        role: FactorRole,
    ) -> Result<(), AssemblyError> {
        let node = self.index(factor)?;
        if name.is_empty() || self.factor_metadata.iter().any(|f| f.name() == Some(name)) {
            return Err(AssemblyError::Name);
        }
        self.factors.push(node);
        self.factor_metadata.push(FactorMetadata {
            name: Some(name.into()),
            role,
        });
        Ok(())
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) enum Selection {
    All,
    Likelihood,
    Posterior,
}
impl Selection {
    fn includes(self, role: FactorRole) -> bool {
        match self {
            Self::All | Self::Posterior => true,
            Self::Likelihood => role == FactorRole::Likelihood,
        }
    }
}
impl CompiledModel {
    /// Metadata for all declared factors, including terms excluded by a selection.
    /// Names do not become generated Rust identifiers.
    pub fn factors(&self) -> &[FactorMetadata] {
        &self.factor_metadata
    }

    /// Metadata for the active sum, preserving declared term order.
    pub fn selected_factors(&self) -> impl Iterator<Item = &FactorMetadata> {
        self.factor_metadata
            .iter()
            .filter(|factor| self.selection.includes(factor.role))
    }
    pub(super) fn retained_factors(&self) -> &[usize] {
        self.active_factors.as_deref().unwrap_or(&self.factors)
    }
    /// Restores the complete generic sum, including unclassified factors and
    /// potentials. This declares no likelihood or posterior interpretation.
    pub fn select_all(mut self) -> Self {
        self.active_nodes = None;
        self.active_factors = None;
        self.selection = Selection::All;
        self
    }

    /// Keeps only explicitly declared likelihood factors, consuming the model
    /// without copying its data or callbacks. Prior and potential factors are
    /// excluded deliberately. Unclassified factors are errors, never guessed.
    /// Declarations and graph storage are retained, so selecting another view
    /// later does not rebuild the graph or lose the excluded terms.
    /// Parameter coordinates/layout are retained, including any parameter that
    /// becomes unidentified after selection. Use `into_objective` to compose it.
    /// # Errors
    /// Rejects unclassified terms or a selection with no likelihood factors.
    pub fn select_likelihood(self) -> Result<Self, AssemblyError> {
        self.select(false)
    }

    /// Keeps the declared likelihood and prior factors. Arbitrary potentials and
    /// unclassified factors are errors rather than implied prior densities.
    /// Use `into_target` for sampling, or `into_objective` for a declared-measure
    /// MAP objective. Neither selection proves posterior propriety.
    /// # Errors
    /// Rejects unclassified factors, arbitrary potentials or an empty sum.
    pub fn select_posterior(self) -> Result<Self, AssemblyError> {
        self.select(true)
    }

    fn select(mut self, posterior: bool) -> Result<Self, AssemblyError> {
        for (index, factor) in self.factor_metadata.iter().enumerate() {
            if factor.role == FactorRole::Unclassified {
                return Err(AssemblyError::UnclassifiedFactor { index });
            }
            if posterior && factor.role == FactorRole::Potential {
                return Err(AssemblyError::PosteriorPotential { index });
            }
        }
        let selection = if posterior {
            Selection::Posterior
        } else {
            Selection::Likelihood
        };
        let factors: Vec<_> = self
            .factors
            .iter()
            .zip(&self.factor_metadata)
            .filter_map(|(node, factor)| selection.includes(factor.role).then_some(*node))
            .collect();
        if factors.is_empty() {
            return Err(AssemblyError::Empty);
        }
        let mut active = vec![false; self.nodes.len()];
        for &node in &factors {
            active[node] = true;
        }
        for (i, op) in self.nodes.iter().enumerate().rev() {
            if !active[i] {
                continue;
            }
            match op {
                Op::Add(a, b) | Op::Multiply(a, b) => {
                    active[*a] = true;
                    active[*b] = true;
                }
                Op::Exp(a) | Op::Log(a) => active[*a] = true,
                Op::Factor { inputs, .. } => {
                    for input in inputs {
                        active[*input] = true;
                    }
                }
                Op::Parameter(_) | Op::Constant(_) => {}
            }
        }
        self.active_factors = Some(factors);
        self.selection = selection;
        self.active_nodes = Some(active);
        Ok(self)
    }
}
