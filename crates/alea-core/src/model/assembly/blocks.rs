//! Whole-value calls lowered to immutable tensor regions and private execution frames.
use super::prepare;
use super::{AssemblyError as E, ModelBuilder, Op, Value, VectorDomain, VectorParameter};
use crate::model::tensor::{
    BlockRules, BlockType, BoundProgram, ElementType, RegionBuilder, Shape, TensorWorkspace,
};

pub(super) struct Call {
    pub inputs: Box<[usize]>,
    pub binding: BoundProgram,
    pub slot: usize,
    pub preparation: Option<prepare::BlockRule>,
    input_view: bool,
}

pub(super) struct Work {
    pub(super) binding: BoundProgram,
    pub(super) adjoint: Vec<f64>,
    pub(super) operands: Box<[Operand]>,
    pub(super) tensor: Option<TensorWorkspace>,
    projection: Option<Operand>,
}

#[derive(Clone, Copy)]
pub(super) enum Operand {
    Scalar(usize),
    Block(usize),
}

impl Operand {
    pub(super) fn value<'frame>(
        self,
        scalars: &'frame [f64],
        blocks: &'frame [Option<Work>],
    ) -> &'frame [f64] {
        match self {
            Self::Scalar(node) => std::slice::from_ref(&scalars[node]),
            Self::Block(slot) => {
                let work = blocks[slot].as_ref().expect("active argument frame");
                if let Some(projection) = work.projection {
                    projection.value(scalars, blocks)
                } else {
                    work.binding
                        .region_result(work.tensor.as_ref().expect("placed tensor frame"))
                }
            }
        }
    }
}

impl Call {
    pub fn workspace(
        &self,
        nodes: &[Op],
        dynamic: &[bool],
        binding: &BoundProgram,
        previous: &[Option<Work>],
        inline_views: bool,
    ) -> Work {
        let operands: Box<[_]> = self
            .inputs
            .iter()
            .map(|&input| match &nodes[input] {
                Op::Region(call) if !call.binding.output_shape().axes().is_empty() => {
                    Operand::Block(call.slot)
                }
                _ => Operand::Scalar(input),
            })
            .collect();
        let projection =
            (inline_views && self.preparation.is_none() && self.input_view).then(|| match operands
                [0]
            {
                Operand::Block(slot) => previous[slot]
                    .as_ref()
                    .expect("active argument frame")
                    .projection
                    .unwrap_or(operands[0]),
                scalar => scalar,
            });
        Work {
            binding: binding.clone(),
            adjoint: if binding.output_shape().axes().is_empty() {
                Vec::new()
            } else {
                vec![0.; binding.output_shape().elements()]
            },
            operands,
            tensor: projection.is_none().then(|| {
                binding.workspace_with_activity(
                    &self
                        .inputs
                        .iter()
                        .map(|&input| dynamic[input])
                        .collect::<Vec<_>>(),
                )
            }),
            projection,
        }
    }

    pub fn forward(
        &self,
        values: &[f64],
        blocks: &mut [Option<Work>],
        preparation: Option<&mut prepare::Plan>,
    ) -> Result<Option<f64>, E> {
        let (previous, current) = blocks.split_at_mut(self.slot);
        let work = current[0].as_mut().expect("active region frame");
        let prepared = self.preparation.as_ref().and_then(|rule| {
            preparation?
                .blocks
                .value(self.slot, values[self.inputs[rule.argument]])
        });
        let scalar_result = self.binding.output_shape().axes().is_empty();
        let value = if let Some(tensor) = &mut work.tensor {
            work.binding.forward_region(
                &|input| work.operands[input].value(values, previous),
                tensor,
                prepared,
            )?;
            scalar_result.then(|| work.binding.region_result(tensor)[0])
        } else {
            // The projection reads an already admitted parent value. Keep
            // its reverse destination distinct to preserve consumer order.
            scalar_result.then(|| work.projection.expect("input view").value(values, previous)[0])
        };
        work.adjoint.fill(0.);
        Ok(value)
    }

    pub fn reverse(
        &self,
        adjoints: &mut [f64],
        blocks: &mut [Option<Work>],
        node: usize,
    ) -> Result<(), E> {
        let (previous, current) = blocks.split_at_mut(self.slot);
        let work = current[0].as_mut().expect("active region frame");
        let (adjoints, seed_slot) = adjoints.split_at_mut(node);
        let seed = if self.binding.output_shape().axes().is_empty() {
            &seed_slot[..1]
        } else {
            &work.adjoint
        };
        let scores = if let Some(tensor) = &mut work.tensor {
            work.binding.reverse_region(seed, tensor)?
        } else {
            if seed.iter().any(|score| !score.is_finite()) {
                return Err(E::Numerical);
            }
            seed
        };
        let mut offset = 0;
        for operand in &work.operands {
            let destination = match *operand {
                Operand::Block(slot) => previous[slot]
                    .as_mut()
                    .expect("active argument frame")
                    .adjoint
                    .as_mut_slice(),
                Operand::Scalar(input) => std::slice::from_mut(&mut adjoints[input]),
            };
            let end = offset + destination.len();
            for (adjoint, &score) in destination.iter_mut().zip(&scores[offset..end]) {
                *adjoint += score;
            }
            offset = end;
        }
        Ok(())
    }
}

impl ModelBuilder {
    /// Returns the shape of an existing model value.
    /// # Errors
    /// Rejects a value from another model or a lexical region.
    pub fn value_shape(&self, value: Value) -> Result<Shape, E> {
        Ok(match &self.nodes[self.value_index(value)?] {
            Op::Region(call) => call.binding.output_shape().clone(),
            _ => Shape::scalar(),
        })
    }

    /// Calls an immutable region with one whole model value per declared input.
    /// The result can feed multiple calls. Native evaluation executes its
    /// forward rule once and accumulates all consumers before its pullback.
    /// # Errors
    /// Rejects foreign values, incorrect arity or mismatched input shapes.
    pub fn call_region(&mut self, inputs: &[Value], binding: BoundProgram) -> Result<Value, E> {
        if inputs.len() != binding.input_shapes().len() {
            return Err(E::Shape);
        }
        let indices = inputs
            .iter()
            .zip(binding.input_shapes())
            .map(|(&value, expected)| {
                if &self.value_shape(value)? != expected {
                    return Err(E::Shape);
                }
                self.value_index(value)
            })
            .collect::<Result<Box<[_]>, _>>()?;
        let slot = self.block_count;
        self.block_count += 1;
        Ok(self.push(Op::Region(Box::new(Call {
            inputs: indices,
            input_view: binding.is_input_view(),
            binding,
            slot,
            preparation: None,
        }))))
    }

    /// Declares a fixed real block in the model.
    /// # Errors
    /// Rejects a shape mismatch or non-finite values.
    pub fn constant_block(&mut self, shape: Shape, values: Vec<f64>) -> Result<Value, E> {
        let mut region = self.region();
        let result = region.constant(shape, values)?;
        let binding = region.compile(result)?.bind(&[])?;
        self.call_region(&[], binding)
    }

    /// Reduces a model value to a scalar using the tensor sum rule.
    /// # Errors
    /// Rejects a foreign value or overflowing region storage.
    pub fn sum_value(&mut self, value: Value) -> Result<Value, E> {
        let shape = self.value_shape(value)?;
        if shape.axes().is_empty() {
            return Ok(value);
        }
        self.region_operation(&[value], |region, inputs| region.sum(inputs[0]))
    }

    /// Reshapes a model value without changing its element order.
    /// # Errors
    /// Rejects a foreign value or a different element count.
    pub fn reshape(&mut self, value: Value, shape: Shape) -> Result<Value, E> {
        self.region_operation(&[value], |region, inputs| region.reshape(inputs[0], shape))
    }

    /// Computes a matrix product of two whole model values.
    /// # Errors
    /// Rejects foreign values or incompatible rank-two shapes.
    pub fn matmul(&mut self, left: Value, right: Value) -> Result<Value, E> {
        self.region_operation(&[left, right], |region, inputs| {
            region.matmul(inputs[0], inputs[1])
        })
    }

    /// Registers typed native rules on ordered whole model values.
    /// Use a lexical region for operations that also read discrete or immutable
    /// data leaves. This convenience binds immediately: an opted-in descriptor-
    /// only preparation recipe may fail during construction, before registration.
    /// # Errors
    /// Rejects foreign handles, non-real formal inputs or shape/layout mismatch.
    pub fn register_block<D, W>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
    {
        self.region_operation(inputs, |region, arguments| {
            region.register_block(arguments, descriptor, rules)
        })
    }

    /// Registers native rules with pure, typed preparation of one scalar argument.
    ///
    /// Calls using the same recipe and model value share its result during one
    /// forward evaluation. Pass a function item or captureless closure as the
    /// recipe. Its type identifies the recipe across code generation units.
    /// It must be deterministic and side-effect free.
    /// Its payload is available only during forward via
    /// [crate::model::tensor::BlockInputs::evaluation_prepared]. The forward rule
    /// still owns domain validation and must save any residuals its VJP needs.
    /// # Errors
    /// Rejects an invalid or nonscalar argument and recipes with runtime state,
    /// including function pointers, in addition to ordinary registration errors.
    pub fn register_block_with_preparation<D, W, P, F>(
        &mut self,
        inputs: &[Value],
        descriptor: D,
        rules: BlockRules<D, W>,
        argument: usize,
        prepare: F,
    ) -> Result<Value, E>
    where
        D: Send + Sync + 'static,
        W: Send + 'static,
        P: Copy + Send + Sync + 'static,
        F: Fn(f64) -> P + Copy + Send + Sync + 'static,
    {
        if size_of::<F>() != 0
            || self.value_shape(*inputs.get(argument).ok_or(E::Shape)?)? != Shape::scalar()
        {
            return Err(E::Shape);
        }
        let value = self.register_block(inputs, descriptor, rules)?;
        let Op::Region(call) = self.nodes.last_mut().expect("registered block") else {
            unreachable!("registered region")
        };
        call.preparation = Some(prepare::BlockRule::new(argument, prepare));
        Ok(value)
    }

    /// Views an existing vector parameter as one shaped model value.
    /// # Errors
    /// Rejects a parameter from another model or a shape of the wrong size.
    pub fn vector_value<S: VectorDomain>(
        &mut self,
        parameter: &VectorParameter<S>,
        shape: Shape,
    ) -> Result<Value, E> {
        let expressions = parameter.expressions();
        if expressions.len() != shape.elements() {
            return Err(E::Shape);
        }
        self.register_block(
            expressions,
            (),
            BlockRules {
                preparation: None,
                inputs: vec![
                    BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::scalar()
                    };
                    expressions.len()
                ],
                output: shape,
                workspace: |_| (),
                heap_bytes: 0,
                forward: |_, inputs, out, _| {
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = inputs.real(i)?[0];
                    }
                    Ok(())
                },
                vjp: Some(|_, _, _, seed, scores, _| {
                    scores.copy_from_slice(seed);
                    Ok(())
                }),
            },
        )
    }

    fn region_operation(
        &mut self,
        values: &[Value],
        build: impl FnOnce(&mut RegionBuilder, &[Value]) -> Result<Value, E>,
    ) -> Result<Value, E> {
        let mut region = self.region();
        let inputs = values
            .iter()
            .enumerate()
            .map(|(i, &value)| region.input(&i.to_string(), self.value_shape(value)?))
            .collect::<Result<Vec<_>, _>>()?;
        let result = build(&mut region, &inputs)?;
        let binding = region.compile(result)?.bind(&[])?;
        self.call_region(values, binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tensor::{BinaryOp, UnaryOp};

    #[test]
    fn lexical_reshape_chain_borrows_one_result_and_keeps_repeated_scores_ordered() {
        let mut model = ModelBuilder::new();
        let vector = Shape::new(vec![2]).unwrap();
        let row = Shape::new(vec![1, 2]).unwrap();
        let parameter = model.real_vector("beta", &["a", "b"]).unwrap();
        let value = model.vector_value(&parameter, vector.clone()).unwrap();
        let first = model.reshape(value, row).unwrap();
        let second = model.reshape(first, vector.clone()).unwrap();
        let mut region = model.region();
        let a = region.input("a", vector.clone()).unwrap();
        let b = region.input("b", vector).unwrap();
        let product = region.binary(BinaryOp::Multiply, a, b).unwrap();
        let logarithms = region.unary(UnaryOp::Log, product).unwrap();
        let sum = region.sum(logarithms).unwrap();
        let program = region.compile(sum).unwrap().bind(&[]).unwrap();
        let term = model.call_region(&[second, second], program).unwrap();
        model.factor(term).unwrap();

        let executable = model.finish().unwrap().bind().compile().unwrap();
        let (mut viewed, mut other) = (executable.context(), executable.context());
        let mut reference = executable.context_with_region_views(false);
        let (mut g, mut reference_g) = ([9.; 2], [10.; 2]);
        assert!(matches!(
            executable.evaluate(&mut viewed, &[0., 2.], &mut g),
            Err(E::Numerical)
        ));
        assert_eq!(g, [9.; 2]);
        let expected = executable
            .evaluate(&mut reference, &[2., 4.], &mut reference_g)
            .unwrap();
        let actual = executable.evaluate(&mut viewed, &[2., 4.], &mut g).unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(g.map(f64::to_bits), reference_g.map(f64::to_bits));
        assert_eq!(g, [1., 0.5]);
        for slot in [1, 2] {
            let work = viewed.work.blocks[slot].as_ref().unwrap();
            assert!(work.tensor.is_none());
            let borrowed = Operand::Block(slot).value(&viewed.work.values, &viewed.work.blocks);
            let origin = Operand::Block(0).value(&viewed.work.values, &viewed.work.blocks);
            assert_eq!(borrowed.as_ptr(), origin.as_ptr());
        }
        executable.evaluate(&mut other, &[4., 8.], &mut g).unwrap();
        assert_eq!(g, [0.5, 0.25]);
        assert_eq!(
            Operand::Block(2).value(&viewed.work.values, &viewed.work.blocks),
            &[2., 4.],
        );
    }

    #[test]
    fn lexical_scalar_view_rejects_overflowed_seed_before_publication() {
        let mut model = ModelBuilder::new();
        let x = model.real("x").unwrap().expr();
        let y = model.real("y").unwrap().expr();
        let vector = model.reshape(x, Shape::new(vec![1]).unwrap()).unwrap();
        let scalar = model.reshape(vector, Shape::scalar()).unwrap();
        let scale = model.multiply(y, f64::MAX).unwrap();
        let term = model.multiply(scalar, scale).unwrap();
        model.factor(term).unwrap();
        model.factor(term).unwrap();
        let executable = model.finish().unwrap().bind().compile().unwrap();
        let mut viewed = executable.context();
        let mut reference = executable.context_with_region_views(false);
        let mut gradient = [19., 23.];
        for context in [&mut viewed, &mut reference] {
            assert!(matches!(
                executable.evaluate(context, &[0., 1.], &mut gradient),
                Err(E::Numerical)
            ));
            assert_eq!(gradient, [19., 23.]);
        }
        for context in [&mut viewed, &mut reference] {
            assert_eq!(
                executable
                    .evaluate(context, &[0., 0.25], &mut gradient)
                    .unwrap(),
                0.
            );
            assert_eq!(gradient, [f64::MAX * 0.5, 0.]);
        }
    }
}
