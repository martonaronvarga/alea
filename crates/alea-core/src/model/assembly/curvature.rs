//! Native differentiated assembly transpose using the resolved tensor calls.
use super::{AssemblyError as E, EvaluationContext, Executable, NativeModel, Op, PlanWorkspace};
use crate::model::{
    ComposedObjective, ConstrainedModel, TransformedTarget, tensor::RegionCurvature,
};

struct Region<'a> {
    program: RegionCurvature<'a>,
    direction: Vec<f64>,
    tangent: Vec<f64>,
    adjoint: Vec<f64>,
}

struct Work<'a> {
    context: EvaluationContext,
    regions: Vec<Option<Region<'a>>>,
    tangents: Vec<f64>,
    adjoints: Vec<f64>,
    gradient: Vec<f64>,
}

/// A resolved native curvature model borrowed from an immutable executable.
/// Each adapter owns independent first-order and curvature scratch.
pub struct NativeCurvature<'a> {
    executable: &'a Executable,
    work: std::cell::RefCell<Work<'a>>,
    pub(in crate::model) chart: std::cell::RefCell<crate::model::HvpWorkspace>,
}

impl Executable {
    /// Resolves the selected model as a native potential-HVP objective in free
    /// coordinates. Use a likelihood selection for observed information.
    /// # Errors
    /// Rejects active unsupported rules and charts. No density Jacobian is added.
    pub fn objective_with_hvp(&self) -> Result<ComposedObjective<NativeCurvature<'_>>, E> {
        ComposedObjective::new(NativeCurvature::new(self)?, self.layout.clone())
            .map_err(|_| E::Shape)
    }

    /// Resolves the selected model as a native potential-HVP sampling target,
    /// including the declared chart density Jacobians.
    /// # Errors
    /// Rejects active unsupported rules and charts.
    pub fn target_with_hvp(&self) -> Result<TransformedTarget<NativeCurvature<'_>>, E> {
        TransformedTarget::new(NativeCurvature::new(self)?, self.layout.clone())
            .map_err(|_| E::Shape)
    }
}

impl<'a> NativeCurvature<'a> {
    pub(in crate::model) fn check_charts(
        layout: &crate::transform::ParameterLayout,
    ) -> Result<(), E> {
        for (block, parameter) in layout.blocks().iter().enumerate() {
            if !parameter.transform().supports_elementwise_curvature() {
                return Err(E::NativeHvpChart { block });
            }
        }
        Ok(())
    }

    fn new(executable: &'a Executable) -> Result<Self, E> {
        Self::check_charts(&executable.layout)?;
        let mut regions = Vec::with_capacity(executable.regions.len());
        let active = |i| executable.active_nodes.as_ref().is_none_or(|a| a[i]);
        // Reject unsupported active scalar leaves before invoking any factories.
        for (node, op) in executable.nodes.iter().enumerate() {
            if active(node) && executable.dynamic[node] && matches!(op, Op::Factor { .. }) {
                return Err(E::MissingNativeHvp { node, source: None });
            }
        }
        let context = executable.context_with_region_views(false);
        for (node, op) in executable.nodes.iter().enumerate() {
            let Op::Region(call) = op else { continue };
            let region = if active(node) && executable.dynamic[node] {
                let binding = &executable.regions[call.slot];
                let activity = call
                    .inputs
                    .iter()
                    .map(|&i| executable.dynamic[i])
                    .collect::<Vec<_>>();
                let tensor = context.work.blocks[call.slot]
                    .as_ref()
                    .expect("active call")
                    .tensor
                    .as_ref()
                    .expect("curvature tensor frame");
                let program =
                    RegionCurvature::new(binding, &activity, tensor).map_err(|source| {
                        E::MissingNativeHvp {
                            node,
                            source: Some(Box::new(source)),
                        }
                    })?;
                let elements = binding.output_shape().elements();
                Some(Region {
                    program,
                    direction: vec![0.; binding.dimension()],
                    tangent: vec![0.; elements],
                    adjoint: vec![0.; elements],
                })
            } else {
                None
            };
            regions.push(region);
        }
        let nodes = executable.nodes.len();
        let d = executable.layout.constrained_dimension();
        Ok(Self {
            executable,
            work: std::cell::RefCell::new(Work {
                context,
                regions,
                tangents: vec![0.; nodes],
                adjoints: vec![0.; nodes],
                gradient: vec![0.; d],
            }),
            chart: std::cell::RefCell::new(crate::model::HvpWorkspace::new(d)),
        })
    }

    // The chart adapter calls this immediately after ordinary value/score
    // admission in this same private context.
    pub(in crate::model) fn differentiate_score(
        &self,
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), E> {
        let e = self.executable;
        let mut work = self.work.try_borrow_mut().map_err(|_| E::WorkspaceBusy)?;
        let Work {
            context,
            regions,
            tangents,
            adjoints,
            gradient,
        } = &mut *work;
        let PlanWorkspace {
            values,
            adjoints: seeds,
            blocks,
            preparation,
            ..
        } = &mut context.work;
        let active = e.active_nodes.as_deref();
        tangents.fill(0.);
        adjoints.fill(0.);
        gradient.fill(0.);
        for (i, op) in e.nodes.iter().enumerate() {
            if active.is_some_and(|a| !a[i]) || !e.dynamic[i] {
                continue;
            }
            tangents[i] = match op {
                Op::Parameter(slot) => v[*slot],
                Op::Add(a, b) => tangents[*a] + tangents[*b],
                Op::Multiply(a, b) => tangents[*a] * values[*b] + values[*a] * tangents[*b],
                Op::Exp(a) => tangents[*a] * values[i],
                Op::Log(a) => tangents[*a] / values[*a],
                Op::Region(call) => {
                    let (previous, current) = regions.split_at_mut(call.slot);
                    let region = current[0].as_mut().expect("resolved active call");
                    let (previous_blocks, current_block) = blocks.split_at_mut(call.slot);
                    let block = current_block[0].as_mut().expect("admitted active call");
                    let mut offset = 0;
                    for &input in &call.inputs {
                        if let Op::Region(arg) = &e.nodes[input]
                            && !arg.binding.output_shape().axes().is_empty()
                        {
                            let end = offset + arg.binding.output_shape().elements();
                            let output = &mut region.direction[offset..end];
                            if let Some(arg) = &previous[arg.slot] {
                                output.copy_from_slice(&arg.tangent);
                            } else {
                                output.fill(0.);
                            }
                            offset = end;
                        } else {
                            region.direction[offset] = tangents[input];
                            offset += 1;
                        }
                    }
                    let prepared = call.preparation.as_ref().and_then(|rule| {
                        preparation
                            .as_mut()?
                            .blocks
                            .value(call.slot, values[call.inputs[rule.argument]])
                    });
                    region.program.forward(
                        &block.binding,
                        block.tensor.as_mut().expect("curvature tensor frame"),
                        &|input| block.operands[input].value(values, previous_blocks),
                        &region.direction,
                        prepared,
                        &mut region.tangent,
                    )?;
                    region.adjoint.fill(0.);
                    if call.binding.output_shape().axes().is_empty() {
                        region.tangent[0]
                    } else {
                        0.
                    }
                }
                Op::Constant(_) | Op::Factor { .. } => unreachable!("checked active expression"),
            };
            if !tangents[i].is_finite() {
                return Err(E::Numerical);
            }
        }

        e.reverse.directional(
            (values, seeds),
            tangents,
            adjoints,
            gradient,
            active,
            |node, adjoints| {
                let Op::Region(call) = &e.nodes[node] else {
                    unreachable!()
                };
                let (previous, current) = regions.split_at_mut(call.slot);
                let region = current[0].as_mut().expect("resolved active call");
                let block = blocks[call.slot].as_mut().expect("admitted active call");
                let (seed, dseed) = if call.binding.output_shape().axes().is_empty() {
                    (
                        std::slice::from_ref(&seeds[node]),
                        std::slice::from_ref(&adjoints[node]),
                    )
                } else {
                    (block.adjoint.as_slice(), region.adjoint.as_slice())
                };
                let scores = region.program.reverse(
                    &block.binding,
                    block.tensor.as_mut().expect("curvature tensor frame"),
                    seed,
                    dseed,
                )?;
                let mut offset = 0;
                for &input in &call.inputs {
                    let width = match &e.nodes[input] {
                        Op::Region(arg) if !arg.binding.output_shape().axes().is_empty() => {
                            arg.binding.output_shape().elements()
                        }
                        _ => 1,
                    };
                    if e.dynamic[input] {
                        let destination = match &e.nodes[input] {
                            Op::Region(arg) if !arg.binding.output_shape().axes().is_empty() => {
                                previous[arg.slot]
                                    .as_mut()
                                    .expect("resolved input call")
                                    .adjoint
                                    .as_mut_slice()
                            }
                            _ => std::slice::from_mut(&mut adjoints[input]),
                        };
                        for (dest, score) in
                            destination.iter_mut().zip(&scores[offset..offset + width])
                        {
                            *dest += score;
                        }
                    }
                    offset += width;
                }
                Ok(())
            },
        )?;
        if gradient.iter().any(|g| !g.is_finite()) {
            return Err(E::Numerical);
        }
        out.copy_from_slice(gradient);
        Ok(())
    }
}

impl ConstrainedModel for NativeCurvature<'_> {
    type Error = E;

    fn dimension(&self) -> usize {
        self.executable.layout.constrained_dimension()
    }

    fn is_numerical_error(&self, error: &E) -> bool {
        NativeModel::numerical(error)
    }

    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, E> {
        let mut work = self.work.try_borrow_mut().map_err(|_| E::WorkspaceBusy)?;
        self.executable.evaluate(&mut work.context, q, g)
    }
}
