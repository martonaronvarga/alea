//! Whole-value calls lowered to immutable tensor regions and private execution frames.
use super::{AssemblyError as E, ModelBuilder, Op, Value, VectorDomain, VectorParameter};
use crate::model::tensor::{
    BlockRules, BlockType, BoundProgram, ElementType, RegionBuilder, Shape, TensorWorkspace,
};

pub(super) struct Call {
    pub inputs: Box<[usize]>,
    pub binding: BoundProgram,
    pub slot: usize,
}

pub(super) struct Work {
    binding: BoundProgram,
    value: Vec<f64>,
    adjoint: Vec<f64>,
    arguments: Vec<f64>,
    scores: Vec<f64>,
    tensor: TensorWorkspace,
}

impl Call {
    pub fn workspace(&self, dynamic: &[bool], binding: &BoundProgram) -> Work {
        let elements = binding.output_shape().elements();
        let dimension = binding.dimension();
        Work {
            binding: binding.clone(),
            value: vec![0.; elements],
            adjoint: vec![0.; elements],
            arguments: vec![0.; dimension],
            scores: vec![0.; dimension],
            tensor: binding.workspace_with_activity(
                &self
                    .inputs
                    .iter()
                    .map(|&input| dynamic[input])
                    .collect::<Vec<_>>(),
            ),
        }
    }

    pub fn forward(
        &self,
        nodes: &[Op],
        values: &[f64],
        blocks: &mut [Option<Work>],
    ) -> Result<Option<f64>, E> {
        let (previous, current) = blocks.split_at_mut(self.slot);
        let work = current[0].as_mut().expect("active region frame");
        let mut offset = 0;
        for &input in &self.inputs {
            let value = match &nodes[input] {
                Op::Region(call) if !call.binding.output_shape().axes().is_empty() => previous
                    [call.slot]
                    .as_ref()
                    .expect("active argument frame")
                    .value
                    .as_slice(),
                _ => std::slice::from_ref(&values[input]),
            };
            let end = offset + value.len();
            work.arguments[offset..end].copy_from_slice(value);
            offset = end;
        }
        work.binding
            .evaluate(&work.arguments, &mut work.value, &mut work.tensor)?;
        work.adjoint.fill(0.);
        Ok(self
            .binding
            .output_shape()
            .axes()
            .is_empty()
            .then(|| work.value[0]))
    }

    pub fn reverse(
        &self,
        nodes: &[Op],
        adjoints: &mut [f64],
        blocks: &mut [Option<Work>],
        node: usize,
    ) -> Result<(), E> {
        let (previous, current) = blocks.split_at_mut(self.slot);
        let work = current[0].as_mut().expect("active region frame");
        let seed = if self.binding.output_shape().axes().is_empty() {
            std::slice::from_ref(&adjoints[node])
        } else {
            &work.adjoint
        };
        work.binding
            .reverse_point(seed, &mut work.scores, &mut work.tensor)?;
        let mut offset = 0;
        for &input in &self.inputs {
            let destination = match &nodes[input] {
                Op::Region(call) if !call.binding.output_shape().axes().is_empty() => previous
                    [call.slot]
                    .as_mut()
                    .expect("active argument frame")
                    .adjoint
                    .as_mut_slice(),
                _ => std::slice::from_mut(&mut adjoints[input]),
            };
            let end = offset + destination.len();
            for (adjoint, &score) in destination.iter_mut().zip(&work.scores[offset..end]) {
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
            binding,
            slot,
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
    /// Use a lexical region for operations that also read discrete data.
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
