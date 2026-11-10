//! A recognizable normal-density primitive, using the existing Normal kernel.
mod batch;
use super::{E, error};
use crate::scalar::Normal;
use alea_core::model::assembly::{NativeRules, OperationDescriptor, SourceBody, SourceRule};
use alea_core::{
    distribution::{EventShape, ReferenceMeasure, Support},
    model::assembly::{
        ArgumentShape, BatchRule, PrimitiveArgument, PrimitiveSignature, ProbabilitySemantics,
        ScalarPreparation,
    },
};
pub use batch::{NormalBatch, NormalBatchWorkspace};
pub(super) const NORMAL_PRIMAL: &str = "if x[2] <= 0.0 { return f64::NAN; }\nlet z = (x[0] - x[1]) / x[2];\n-0.5 * z * z - x[2].ln() - 0.5 * ::std::f64::consts::TAU.ln()";

/// Immutable scalar Normal log-density description. Arguments are event,
/// location and standard deviation; constants remain ordinary graph nodes.
/// This describes a density operation, not a declaration of a generative site.
#[derive(Debug, Clone)]
pub struct NormalFactor;
impl NormalFactor {
    /// Describes the Normal density independently of graph inputs. Supply event,
    /// location and scale, in that order, at graph registration.
    pub fn new() -> Self {
        Self
    }
}
impl Default for NormalFactor {
    fn default() -> Self {
        Self::new()
    }
}
impl NormalFactor {
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
    /// Native kernel workspace.
    pub fn workspace(&self) {}
    fn scalar_preparation(&self) -> Option<ScalarPreparation<Self>> {
        Some(ScalarPreparation {
            argument: 2,
            prepare: f64::ln,
            evaluate: |_, x, g, _, log_scale| {
                if x.len() != 3 || g.len() != 3 {
                    return Err(E::Shape);
                }
                let law = Normal::new(x[1], x[2]).map_err(error)?;
                let (value, scores) = law
                    .log_prob_event_parameter_scores_prepared(x[0], log_scale)
                    .map_err(error)?;
                g.copy_from_slice(&scores);
                Ok(value)
            },
        })
    }
    /// Native kernel evaluate.
    pub fn evaluate(&self, x: &[f64], g: &mut [f64], _: &mut ()) -> Result<f64, E> {
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
    fn source_gradient_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|_| {
            "let law=match ::alea::distributions::scalar::Normal::new(x[1],x[2]) {Ok(v)=>v,Err(_)=>return f64::NAN};\nlet(value,event)=match ::alea::distribution::ParameterScore::log_prob_event_parameter_score(&law,x[0],&mut g[1..]) {Ok(v)=>v,Err(_)=>return f64::NAN};g[0]=event;value".into()
        })
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
impl OperationDescriptor for NormalFactor {
    type Workspace = ();
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        rules.batch = self.batch_rule();
        rules.preparation = self.scalar_preparation();
        let body = self.rust_primal().map(SourceBody::Static);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: self.source_gradient_rule(),
            workspace: 0,
        });
        rules
    }
}
