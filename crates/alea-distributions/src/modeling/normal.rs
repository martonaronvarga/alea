//! A recognizable normal-density primitive, using the existing Normal kernel.
use super::{E, error};
use crate::scalar::Normal;
use alea_core::{
    distribution::{EventShape, ReferenceMeasure, Support},
    model::assembly::{
        ArgumentShape, BatchRule, Expr, Primitive, PrimitiveArgument, PrimitiveSignature,
        ProbabilitySemantics,
    },
};
pub(super) const NORMAL_PRIMAL: &str = "if x[2] <= 0.0 { return f64::NAN; }\nlet z = (x[0] - x[1]) / x[2];\n-0.5 * z * z - x[2].ln() - 0.5 * ::std::f64::consts::TAU.ln()";

/// Immutable scalar Normal log-density description. Arguments are event,
/// location and standard deviation; constants remain ordinary graph nodes.
/// This describes a density operation, not a declaration of a generative site.
#[derive(Debug, Clone)]
pub struct NormalFactor {
    inputs: [Expr; 3],
}
impl NormalFactor {
    /// Describes `Normal(location, scale)` evaluated at `event`. Handles are
    /// owner-checked when registered with [`alea_core::model::assembly::ModelBuilder::primitive`].
    /// Parameter values and numerical failures retain the existing Normal checks.
    pub fn new(event: Expr, location: Expr, scale: Expr) -> Self {
        Self {
            inputs: [event, location, scale],
        }
    }
}
impl Primitive for NormalFactor {
    type Workspace = ();
    fn inputs(&self) -> &[Expr] {
        &self.inputs
    }
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "normal",
            arguments: ["event", "location", "scale"]
                .map(|name| PrimitiveArgument {
                    name,
                    shape: ArgumentShape::Scalar,
                })
                .into(),
            probability: Some(ProbabilitySemantics {
                event: EventShape::Scalar,
                batch: vec![],
                support: Support::Real,
                measure: ReferenceMeasure::Lebesgue { dimension: 1 },
            }),
        }
    }
    fn workspace(&self) {}
    fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
        if x.len() != 3 || g.len() != 3 {
            return Err(E::Shape);
        }
        let law = Normal::new(x[1], x[2]).map_err(error)?;
        let (value, scores) = law.log_prob_event_parameter_scores(x[0]).map_err(error)?;
        g.copy_from_slice(&scores);
        Ok(value)
    }
    fn rust_primal(&self) -> Option<&'static str> {
        Some(NORMAL_PRIMAL)
    }
    fn batch_rule(&self) -> Option<BatchRule<Self>> {
        Some(BatchRule {
            units: 1,
            evaluate: |this, x, out, workspace| {
                if out.len() != 1 {
                    return Err(E::Shape);
                }
                out[0] = this.evaluate(x, &mut [0.; 3], workspace)?;
                Ok(())
            },
        })
    }
}
