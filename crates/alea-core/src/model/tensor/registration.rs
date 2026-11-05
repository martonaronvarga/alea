//! R0.2a/b private, test-only registration/lifecycle proof on the existing SSA.
//! No second graph/executor, probability law, source rule or public API is added.
//! R0.4 decides whether/how these boundaries become production interfaces.
use super::*;
mod lifecycle;
mod observations;

// Use the existing non-recycled tensor identity space, never allocation addresses.
fn identity() -> u64 {
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("tensor identity space exhausted")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BindingId(u64);

// Binding-dependent preparations are immutable and separate from Work's
// parameter-dependent residual/scratch. Clones retain the same snapshot identity.
pub(super) struct BindingState {
    id: BindingId,
    prepared: Box<[Option<Arc<[f64]>>]>,
}
pub(super) fn bind_snapshot(plan: Arc<Plan>, data: Vec<Arc<[f64]>>) -> Result<BoundProgram, E> {
    let mut prepared = Vec::with_capacity(plan.nodes.len());
    for (i, node) in plan.nodes.iter().enumerate() {
        let rule = match &node.op {
            Op::Prototype(op) if plan.active[i] => match &**op {
                Operation::Native(call) => call.kernel.binding.as_ref(),
                Operation::Data(_) => None,
            },
            _ => None,
        };
        let value = if let Some(rule) = rule {
            let mut output = vec![f64::NAN; rule.elements];
            let Op::Prototype(op) = &node.op else {
                unreachable!()
            };
            (rule.prepare)(
                BindingInputs {
                    call: op.inputs(),
                    rule,
                    nodes: &plan.nodes,
                    data: &data,
                },
                &mut output,
            )?;
            if output.iter().any(|v| !v.is_finite()) {
                return Err(E::Numerical);
            }
            Some(Arc::from(output))
        } else {
            None
        };
        prepared.push(value);
    }
    Ok(BoundProgram {
        plan,
        data,
        snapshot: Arc::new(BindingState {
            id: BindingId(identity()),
            prepared: prepared.into_boxed_slice(),
        }),
    })
}
impl BoundProgram {
    pub(super) fn preparation(&self, node: usize) -> &[f64] {
        self.snapshot.prepared[node].as_deref().unwrap_or(&[])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DType {
    Real,
    Bool,
    Count,
    Index,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Type {
    dtype: DType,
    shape: Shape,
}
impl Type {
    fn of(node: &Node) -> Self {
        let dtype = match &node.op {
            Op::Prototype(op) => match &**op {
                Operation::Data(Data::Bool(_)) => DType::Bool,
                Operation::Data(Data::Count(_)) => DType::Count,
                Operation::Data(Data::Index(_)) => DType::Index,
                Operation::Native(_) => DType::Real,
            },
            _ => DType::Real,
        };
        Self {
            dtype,
            shape: node.shape.clone(),
        }
    }
}
pub(super) enum Data {
    Bool(Box<[bool]>),
    Count(Box<[u64]>),
    Index(Box<[usize]>),
}
impl Data {
    fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Count(v) => v.len(),
            Self::Index(v) => v.len(),
        }
    }
}

// Descriptors contain types/configuration, never graph handles. Registration
// resolves their edges and prepares the native rule only AFTER structural checks.
struct Signature {
    inputs: Vec<Type>,
    output: Type,
    scratch: usize,
}
type Forward = dyn Fn(Inputs<'_>, &mut [f64], &mut [f64]) -> Result<(), E> + Send + Sync;
type Vjp =
    dyn Fn(Inputs<'_>, &[f64], &[f64], &mut [f64], &mut [f64]) -> Result<(), E> + Send + Sync;
struct Kernel {
    forward: Box<Forward>,
    // An optional native VJP implies neither JVP, source nor higher derivatives.
    vjp: Option<Box<Vjp>>,
    binding: Option<BindingRule>,
}
type Prepare = dyn Fn(BindingInputs<'_>, &mut [f64]) -> Result<(), E> + Send + Sync;
struct BindingRule {
    // Logical argument positions, independent of graph handles. Only direct
    // immutable data/constants may be read during this bounded binding proof.
    arguments: Box<[usize]>,
    elements: usize,
    prepare: Box<Prepare>,
}
enum BindingValue<'a> {
    Real(&'a [f64]),
    Bool(&'a [bool]),
    Count(&'a [u64]),
    Index(&'a [usize]),
}
struct BindingInputs<'a> {
    call: &'a [Value],
    rule: &'a BindingRule,
    nodes: &'a [Node],
    data: &'a [Arc<[f64]>],
}
impl BindingInputs<'_> {
    fn get(&self, i: usize) -> Result<BindingValue<'_>, E> {
        let argument = *self.rule.arguments.get(i).ok_or(E::Shape)?;
        let node = &self.nodes[self.call[argument].node];
        match &node.op {
            Op::Data(i) => Ok(BindingValue::Real(&self.data[*i])),
            Op::Constant(v) => Ok(BindingValue::Real(v)),
            Op::Prototype(op) => match &**op {
                Operation::Data(Data::Bool(v)) => Ok(BindingValue::Bool(v)),
                Operation::Data(Data::Count(v)) => Ok(BindingValue::Count(v)),
                Operation::Data(Data::Index(v)) => Ok(BindingValue::Index(v)),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
}
pub(super) struct Call {
    inputs: Box<[Value]>,
    contributions: Box<[Range<usize>]>,
    scores: usize,
    scratch: usize,
    output: usize,
    kernel: Kernel,
}
pub(super) enum Operation {
    Data(Data),
    Native(Call),
}
impl Operation {
    pub(super) fn inputs(&self) -> &[Value] {
        match self {
            Self::Data(_) => &[],
            Self::Native(call) => &call.inputs,
        }
    }
}

// Typed immutable views. Counts/indices are NEVER rounded through f64, and no
// cotangent slots exist for them. A bad callback access remains a typed error.
struct Inputs<'a> {
    arguments: &'a [Value],
    nodes: &'a [Node],
    values: &'a [f64],
    prepared: &'a [f64],
}
impl Inputs<'_> {
    fn node(&self, i: usize) -> Result<&Node, E> {
        let v = self.arguments.get(i).ok_or(E::Shape)?;
        self.nodes.get(v.node).ok_or(E::Shape)
    }
    fn real(&self, i: usize) -> Result<&[f64], E> {
        let n = self.node(i)?;
        if n.discrete() {
            return Err(E::Shape);
        }
        self.values.get(n.range()).ok_or(E::Shape)
    }
    fn booleans(&self, i: usize) -> Result<&[bool], E> {
        match &self.node(i)?.op {
            Op::Prototype(op) => match &**op {
                Operation::Data(Data::Bool(v)) => Ok(v),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
    fn counts(&self, i: usize) -> Result<&[u64], E> {
        match &self.node(i)?.op {
            Op::Prototype(op) => match &**op {
                Operation::Data(Data::Count(v)) => Ok(v),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
    fn indices(&self, i: usize) -> Result<&[usize], E> {
        match &self.node(i)?.op {
            Op::Prototype(op) => match &**op {
                Operation::Data(Data::Index(v)) => Ok(v),
                _ => Err(E::Shape),
            },
            _ => Err(E::Shape),
        }
    }
}
fn extent(a: usize, b: usize) -> Result<usize, E> {
    a.checked_add(b)
        .filter(|&v| v <= isize::MAX as usize / size_of::<f64>())
        .ok_or(E::Shape)
}
impl TensorBuilder {
    fn typed_data(&mut self, shape: Shape, data: Data) -> Result<Value, E> {
        if data.len() != shape.elements {
            return Err(E::Shape);
        }
        self.push(shape, Op::Prototype(Box::new(Operation::Data(data))))
    }
    fn indices(&mut self, shape: Shape, values: Vec<usize>, bound: usize) -> Result<Value, E> {
        if values.iter().any(|&i| i >= bound) {
            return Err(E::Shape);
        }
        self.typed_data(shape, Data::Index(values.into_boxed_slice()))
    }
    fn register(
        &mut self,
        signature: Signature,
        inputs: &[Value],
        prepare: impl FnOnce() -> Result<Kernel, E>,
    ) -> Result<Value, E> {
        if inputs.len() != signature.inputs.len() || signature.output.dtype != DType::Real {
            return Err(E::Shape);
        }
        // Check ALL handles before types or executing user preparation.
        for &input in inputs {
            self.node(input)?;
        }
        let mut scores = 0;
        let mut contributions = Vec::with_capacity(inputs.len());
        for (&input, expected) in inputs.iter().zip(&signature.inputs) {
            let actual = Type::of(self.node(input)?);
            if &actual != expected {
                return Err(E::Shape);
            }
            let start = scores;
            if actual.dtype == DType::Real {
                scores = extent(scores, actual.shape.elements)?;
            }
            contributions.push(start..scores);
        }
        extent(self.slots, signature.output.shape.elements)?;
        extent(
            extent(scores, signature.scratch)?,
            signature.output.shape.elements,
        )?;
        let kernel = prepare()?;
        if let Some(rule) = &kernel.binding {
            extent(0, rule.elements)?;
            for &i in &rule.arguments {
                let input = *inputs.get(i).ok_or(E::Shape)?;
                let node = self.node(input)?;
                if !matches!(&node.op, Op::Data(_) | Op::Constant(_)) && !node.discrete() {
                    return Err(E::Shape);
                }
            }
        }
        self.push(
            signature.output.shape.clone(),
            Op::Prototype(Box::new(Operation::Native(Call {
                inputs: inputs.into(),
                contributions: contributions.into_boxed_slice(),
                scores,
                scratch: signature.scratch,
                output: signature.output.shape.elements,
                kernel,
            }))),
        )
    }
}

pub(super) struct Work {
    scratch: Vec<f64>,
    contributions: Vec<f64>,
    seed: Vec<f64>,
}
impl Call {
    pub(super) fn workspace(&self) -> Work {
        Work {
            scratch: vec![0.; self.scratch],
            contributions: vec![0.; self.scores],
            seed: vec![0.; self.output],
        }
    }
    pub(super) fn workspace_elements(&self) -> Result<usize, E> {
        extent(extent(self.scores, self.scratch)?, self.output)
    }
    pub(super) fn forward(
        &self,
        nodes: &[Node],
        node: &Node,
        values: &mut [f64],
        w: &mut Work,
        prepared: &[f64],
    ) -> Result<(), E> {
        let (before, after) = values.split_at_mut(node.start);
        let output = &mut after[..self.output];
        output.fill(f64::NAN);
        (self.kernel.forward)(
            Inputs {
                arguments: &self.inputs,
                nodes,
                values: before,
                prepared,
            },
            output,
            &mut w.scratch,
        )
    }
    pub(super) fn reverse(
        &self,
        nodes: &[Node],
        values: &[f64],
        seed: &[f64],
        primal: &[f64],
        w: &mut Work,
        prepared: &[f64],
    ) -> Result<(), E> {
        let vjp = self
            .kernel
            .vjp
            .as_ref()
            .ok_or_else(|| E::Factor(Box::new(MissingVjp)))?;
        w.seed.copy_from_slice(seed);
        w.contributions.fill(f64::NAN);
        vjp(
            Inputs {
                arguments: &self.inputs,
                nodes,
                values,
                prepared,
            },
            primal,
            &w.seed,
            &mut w.contributions,
            &mut w.scratch,
        )?;
        if w.contributions.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(())
    }
    // Local contributions are complete/admitted BEFORE graph accumulation. Each
    // occurrence has a separate range, including repeated argument handles.
    pub(super) fn accumulate(&self, w: &Work, mut add: impl FnMut(Value, &[f64])) {
        for (&input, range) in self.inputs.iter().zip(&self.contributions) {
            if !range.is_empty() {
                add(input, &w.contributions[range.clone()]);
            }
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("native block operation has no VJP rule")]
struct MissingVjp;

pub(super) fn missing_source(plan: &Plan) -> Option<usize> {
    plan.nodes.iter().enumerate().find_map(|(i, node)| {
        let missing = match &node.op {
            Op::Prototype(_) => true,
            Op::Cond(c) => {
                missing_source(&c.yes.plan).is_some() || missing_source(&c.no.plan).is_some()
            }
            Op::Scan(s) => s.steps != 0 && missing_source(&s.body.plan).is_some(),
            _ => false,
        };
        (plan.active[i] && missing).then_some(i)
    })
}
// No replay/purity contract is supplied by this native-only experiment. Do not
// infer it from mathematical shape, a VJP or the absence of a source rule.
pub(super) fn blocks_replay(plan: &Plan) -> bool {
    plan.nodes.iter().enumerate().any(|(i, node)| {
        plan.active[i]
            && match &node.op {
                Op::Prototype(op) => matches!(&**op, Operation::Native(_)),
                Op::Cond(c) => blocks_replay(&c.yes.plan) || blocks_replay(&c.no.plan),
                Op::Scan(s) => s.steps != 0 && blocks_replay(&s.body.plan),
                _ => false,
            }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn shape(n: usize) -> Shape {
        Shape::new(vec![n]).unwrap()
    }
    fn ty(dtype: DType, shape: Shape) -> Type {
        Type { dtype, shape }
    }
    fn unary_signature(n: usize) -> Signature {
        Signature {
            inputs: vec![ty(DType::Real, shape(n))],
            output: ty(DType::Real, shape(n)),
            scratch: 1,
        }
    }
    fn identity() -> Kernel {
        Kernel {
            binding: None,
            forward: Box::new(|x, out, _| {
                out.copy_from_slice(x.real(0)?);
                Ok(())
            }),
            vjp: Some(Box::new(|_, _, seed, g, _| {
                g.copy_from_slice(seed);
                Ok(())
            })),
        }
    }
    fn selected(i: usize) -> bool {
        i % 3 != 1 && i.is_multiple_of(2) && i.is_multiple_of(5)
    }
    fn model(n: usize) -> (TensorProgram, Arc<AtomicUsize>) {
        let mut b = TensorBuilder::new();
        let x = b.input("x", shape(n)).unwrap();
        let neg = b.unary(UnaryOp::Neg, x).unwrap();
        let squared = b.binary(BinaryOp::Multiply, neg, neg).unwrap();
        let mask = b
            .typed_data(shape(n), Data::Bool((0..n).map(|i| i % 3 != 1).collect()))
            .unwrap();
        let counts = b
            .typed_data(
                shape(n),
                Data::Count(
                    (0..n)
                        .map(|i| {
                            if i.is_multiple_of(2) {
                                u64::MAX
                            } else {
                                u64::MAX - 1
                            }
                        })
                        .collect(),
                ),
            )
            .unwrap();
        let indices = b
            .indices(
                shape(n),
                (0..n)
                    .map(|i| {
                        if i.is_multiple_of(5) {
                            usize::MAX - 1
                        } else {
                            usize::MAX - 2
                        }
                    })
                    .collect(),
                usize::MAX,
            )
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let signature = Signature {
            inputs: [
                DType::Real,
                DType::Real,
                DType::Bool,
                DType::Count,
                DType::Index,
            ]
            .into_iter()
            .map(|dtype| ty(dtype, shape(n)))
            .collect(),
            output: ty(DType::Real, shape(n)),
            scratch: 1,
        };
        let result = b
            .register(
                signature,
                &[squared, squared, mask, counts, indices],
                || {
                    Ok(Kernel {
                        binding: None,
                        forward: Box::new(move |x, out, scratch| {
                            count.fetch_add(1, Ordering::Relaxed);
                            scratch[0] = 1.; // Reinitialize, regardless of previous execution.
                            let (a, b, m, c, ix) = (
                                x.real(0)?,
                                x.real(1)?,
                                x.booleans(2)?,
                                x.counts(3)?,
                                x.indices(4)?,
                            );
                            for (i, out) in out.iter_mut().enumerate() {
                                *out = if m[i] && c[i] == u64::MAX && ix[i] == usize::MAX - 1 {
                                    a[i] * b[i]
                                } else {
                                    a[i] + b[i]
                                };
                            }
                            Ok(())
                        }),
                        vjp: Some(Box::new(move |x, primal, seed, g, scratch| {
                            let (a, b, m, c, ix) = (
                                x.real(0)?,
                                x.real(1)?,
                                x.booleans(2)?,
                                x.counts(3)?,
                                x.indices(4)?,
                            );
                            if primal.len() != n || g.len() != 2 * n {
                                return Err(E::Shape);
                            }
                            let (ga, gb) = g.split_at_mut(n);
                            for i in 0..n {
                                let multiply = m[i] && c[i] == u64::MAX && ix[i] == usize::MAX - 1;
                                ga[i] = seed[i] * if multiply { b[i] } else { scratch[0] };
                                gb[i] = seed[i] * if multiply { a[i] } else { scratch[0] };
                            }
                            Ok(())
                        })),
                    })
                },
            )
            .unwrap();
        let total = b.sum(result).unwrap();
        (b.compile(total).unwrap(), calls)
    }

    #[test]
    fn typed_block_and_opaque_vjp_compose_on_existing_ssa_without_scalarizing() {
        let mut node_count = None;
        for n in [0, 1, 3, 17, 257] {
            let (program, calls) = model(n);
            assert_eq!(
                *node_count.get_or_insert(program.node_count()),
                program.node_count()
            );
            assert_eq!(
                program
                    .nodes()
                    .filter(|node| node.operation == TensorOperation::Registered)
                    .count(),
                1
            );
            let native = program
                .plan
                .nodes
                .iter()
                .find_map(|node| match &node.op {
                    Op::Prototype(op) => match &**op {
                        Operation::Native(call) => Some(call),
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap();
            assert_eq!(native.inputs[0], native.inputs[1]);
            assert_eq!(native.scores, 2 * n); // Discrete arguments have no cotangents.
            assert_eq!(
                program
                    .plan
                    .nodes
                    .iter()
                    .filter(|node| node.discrete())
                    .count(),
                3
            );
            assert!(
                program
                    .plan
                    .nodes
                    .iter()
                    .filter(|node| node.discrete())
                    .all(|node| node.range().is_empty())
            );
            let p = program.bind(&[]).unwrap();
            let mut w = p.workspace();
            let x: Vec<_> = (0..n).map(|i| 0.1 + (i % 7) as f64 * 0.07).collect();
            let expected: f64 = x
                .iter()
                .enumerate()
                .map(|(i, x)| if selected(i) { x.powi(4) } else { 2. * x * x })
                .sum();
            let mut out = [91.];
            let mut g = vec![92.; n];
            for seed in [0., 1., -0.7] {
                let before = calls.load(Ordering::Relaxed);
                p.value_and_vjp(&x, &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!(calls.load(Ordering::Relaxed), before + 1); // Not two forward scans.
                assert!((out[0] - expected).abs() < 2e-10 * (1. + expected.abs()));
                for (i, (&x, &g)) in x.iter().zip(&g).enumerate() {
                    let reference = seed * if selected(i) { 4. * x.powi(3) } else { 4. * x };
                    assert!(
                        (g - reference).abs() < 2e-12 * (1. + reference.abs()),
                        "{i}: {g} vs {reference}"
                    );
                }
            }
            let wrong = vec![0.; n + 1];
            let before = calls.load(Ordering::Relaxed);
            assert!(matches!(
                p.value_and_vjp(&wrong, &[1.], &mut out, &mut g, &mut w),
                Err(E::Shape)
            ));
            if n != 0 {
                let mut invalid = x.clone();
                invalid[0] = f64::NAN;
                assert!(matches!(
                    p.value_and_vjp(&invalid, &[1.], &mut out, &mut g, &mut w),
                    Err(E::Numerical)
                ));
            }
            assert!(matches!(
                p.value_and_vjp(&x, &[f64::NAN], &mut out, &mut g, &mut w),
                Err(E::Numerical)
            ));
            assert_eq!(calls.load(Ordering::Relaxed), before);
            let allocation = allocation_counter::measure(|| {
                for _ in 0..8 {
                    p.value_and_vjp(&x, &[1.], &mut out, &mut g, &mut w)
                        .unwrap();
                }
            });
            assert_eq!(allocation.count_total, 0);
            let bytes = p.workspace_bytes();
            assert!(p.workspace_with_limit(bytes - 1).is_err());
            let measured = allocation_counter::measure(|| {
                let _work = p.workspace();
            });
            assert!(measured.bytes_max as usize <= bytes);
            if n == 3 {
                use crate::model::{ConstrainedModel, assembly::ModelBuilder};
                let mut builder = ModelBuilder::new();
                let params: Vec<_> = (0..n)
                    .map(|i| builder.real(&format!("x{i}")).unwrap().expr())
                    .collect();
                let factor = TensorFactor::new(p.clone(), &params).unwrap();
                assert!(factor.source_rule().is_none());
                let root = builder.primitive(factor).unwrap();
                builder.potential("deterministic", root).unwrap();
                let model = builder.compile().unwrap();
                let mut expected_gradient = vec![0.; n];
                p.value_and_vjp(&x, &[1.], &mut out, &mut expected_gradient, &mut w)
                    .unwrap();
                let value = model.logp_grad(&x, &mut g).unwrap();
                assert_eq!(value.to_bits(), out[0].to_bits());
                assert_eq!(g, expected_gradient);
            }
            // Independent central differences at several scales, not shared VJP math.
            for i in [0, n / 2, n.saturating_sub(1)] {
                if n == 0 {
                    break;
                }
                for h in [1e-4, 1e-5, 1e-6] {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[i] += h;
                    minus[i] -= h;
                    let mut a = [0.];
                    let mut b = [0.];
                    p.evaluate(&plus, &mut a, &mut w).unwrap();
                    p.evaluate(&minus, &mut b, &mut w).unwrap();
                    assert!(
                        ((a[0] - b[0]) / (2. * h) - g[i]).abs() < 3e-7,
                        "n{n}, coordinate{i}, h{h}"
                    );
                }
            }
        }
    }

    #[test]
    fn registration_rejects_foreign_dtype_shape_arity_and_overflow_before_preparation_or_mutation()
    {
        let mut b = TensorBuilder::new();
        let x = b.input("x", shape(3)).unwrap();
        let other = b.input("other", shape(2)).unwrap();
        let same_elements = b.input("matrix", Shape::new(vec![1, 3]).unwrap()).unwrap();
        let scalar = b.input("scalar", Shape::scalar()).unwrap();
        let mask = b
            .typed_data(shape(3), Data::Bool(vec![true; 3].into()))
            .unwrap();
        let foreign = TensorBuilder::new().input("foreign", shape(3)).unwrap();
        let calls = AtomicUsize::new(0);
        for (args, foreign_error) in [
            (vec![foreign], true),
            (
                vec![Value {
                    owner: b.owner,
                    node: usize::MAX,
                }],
                true,
            ),
            (vec![mask], false),
            (vec![other], false),
            (vec![same_elements], false),
            (vec![], false),
            (vec![x, x], false),
        ] {
            let before = (b.nodes.len(), b.slots);
            let error = b
                .register(unary_signature(3), &args, || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(identity())
                })
                .unwrap_err();
            if foreign_error {
                assert!(matches!(error, E::ForeignHandle));
            } else {
                assert!(matches!(error, E::Shape));
            }
            assert_eq!((b.nodes.len(), b.slots), before);
        }
        assert!(matches!(
            b.register(unary_signature(1), &[scalar], || panic!(
                "rank-zero is not rank-one"
            )),
            Err(E::Shape)
        ));
        let mut bad = unary_signature(3);
        bad.scratch = usize::MAX;
        assert!(matches!(
            b.register(bad, &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        let mut bad = unary_signature(3);
        bad.output.dtype = DType::Bool;
        assert!(matches!(
            b.register(bad, &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        // The same handle in two REAL positions is legal; a repeated handle
        // cannot pretend to satisfy an INDEX argument with identical shape.
        let bad = Signature {
            inputs: vec![ty(DType::Real, shape(3)), ty(DType::Index, shape(3))],
            output: ty(DType::Real, shape(3)),
            scratch: 0,
        };
        assert!(matches!(
            b.register(bad, &[x, x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let old = b.slots;
        b.slots = isize::MAX as usize / size_of::<f64>();
        assert!(matches!(
            b.register(unary_signature(3), &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            }),
            Err(E::Shape)
        ));
        b.slots = old;
        let before = (b.nodes.len(), b.slots);
        assert!(matches!(
            b.register(unary_signature(3), &[x], || Err(E::Name)),
            Err(E::Name)
        ));
        assert_eq!((b.nodes.len(), b.slots), before);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let y = b
            .register(unary_signature(3), &[x], || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(identity())
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut out = [0.; 3];
        let mut inputs = vec![0.; p.dimension()];
        inputs[..3].copy_from_slice(&[1., 2., 3.]);
        p.evaluate(&inputs, &mut out, &mut p.workspace()).unwrap();
        assert_eq!(out, [1., 2., 3.]);
        let mut large = TensorBuilder::new();
        let n = isize::MAX as usize / size_of::<f64>() / 2 + 1;
        let huge = large.input("huge", shape(n)).unwrap(); // Metadata only, no dense allocation.
        let sig = Signature {
            inputs: vec![ty(DType::Real, shape(n)); 2],
            output: ty(DType::Real, Shape::scalar()),
            scratch: 0,
        };
        assert!(matches!(
            large.register(sig, &[huge, huge], || panic!(
                "overflow must precede preparation"
            )),
            Err(E::Shape)
        ));
        assert!(Shape::new(vec![usize::MAX, 2]).is_err());
    }

    #[test]
    fn discrete_data_cannot_leak_into_real_builtins_or_get_rounded() {
        let mut b = TensorBuilder::new();
        let scalar = Shape::scalar();
        let count = b
            .typed_data(scalar.clone(), Data::Count(vec![u64::MAX].into()))
            .unwrap();
        let mask = b
            .typed_data(scalar.clone(), Data::Bool(vec![true].into()))
            .unwrap();
        let index = b
            .indices(scalar.clone(), vec![usize::MAX - 1], usize::MAX)
            .unwrap();
        let x = b.input("x", scalar.clone()).unwrap();
        let before = (b.nodes.len(), b.slots);
        assert!(matches!(b.unary(UnaryOp::Neg, count), Err(E::Shape)));
        assert!(matches!(b.binary(BinaryOp::Add, mask, x), Err(E::Shape)));
        assert!(matches!(b.sum(index), Err(E::Shape)));
        assert!(matches!(b.reshape(count, shape(1)), Err(E::Shape)));
        assert!(matches!(
            b.gather(count, scalar.clone(), vec![0]),
            Err(E::Shape)
        ));
        assert!(matches!(
            b.typed_data(shape(2), Data::Count(vec![1].into())),
            Err(E::Shape)
        ));
        assert!(matches!(
            b.indices(scalar.clone(), vec![3], 3),
            Err(E::Shape)
        ));
        assert_eq!((b.nodes.len(), b.slots), before);
        assert_eq!(Type::of(b.node(count).unwrap()).shape, scalar);
        let sig = Signature {
            inputs: vec![
                ty(DType::Count, scalar.clone()),
                ty(DType::Bool, scalar.clone()),
                ty(DType::Index, scalar.clone()),
            ],
            output: ty(DType::Real, scalar),
            scratch: 0,
        };
        let y = b
            .register(sig, &[count, mask, index], || {
                Ok(Kernel {
                    binding: None,
                    forward: Box::new(|x, out, _| {
                        assert_eq!(x.counts(0)?, [u64::MAX]);
                        assert_eq!(x.indices(2)?, [usize::MAX - 1]);
                        assert!(matches!(x.real(0), Err(E::Shape)));
                        assert!(matches!(x.counts(2), Err(E::Shape)));
                        assert!(matches!(x.real(9), Err(E::Shape)));
                        out[0] = if x.booleans(1)?[0] { 1. } else { 0. };
                        Ok(())
                    }),
                    vjp: None,
                })
            })
            .unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut out = [0.];
        // Unused real input remains in the fixed input schema, not a discrete tangent.
        let mut g = [17.];
        p.value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut p.workspace())
            .unwrap();
        assert_eq!(out, [1.]);
        assert_eq!(g, [0.]);
        let mut b = TensorBuilder::new();
        let v = b
            .typed_data(Shape::scalar(), Data::Bool(vec![true].into()))
            .unwrap();
        assert!(matches!(b.compile(v), Err(E::Shape)));
    }

    #[test]
    fn opaque_failures_preserve_caller_outputs_and_recover_even_with_zero_seeds() {
        let mode = Arc::new(AtomicUsize::new(0));
        let f = Arc::clone(&mode);
        let r = Arc::clone(&mode);
        let mut b = TensorBuilder::new();
        let x = b.input("x", shape(1)).unwrap();
        let y = b
            .register(unary_signature(1), &[x], || {
                Ok(Kernel {
                    binding: None,
                    forward: Box::new(move |x, out, scratch| {
                        scratch[0] = f64::NAN;
                        match f.load(Ordering::Relaxed) {
                            1 => return Err(E::Shape),
                            2 => return Ok(()),
                            6 => panic!("opaque callback"),
                            _ => {}
                        }
                        scratch[0] = x.real(0)?[0];
                        out[0] = scratch[0] * scratch[0];
                        Ok(())
                    }),
                    vjp: Some(Box::new(move |_, _, seed, g, scratch| {
                        match r.load(Ordering::Relaxed) {
                            3 => return Err(E::Name),
                            4 => return Ok(()),
                            5 => {
                                g[0] = f64::INFINITY;
                                return Ok(());
                            }
                            _ => {}
                        }
                        g[0] = 2. * scratch[0] * seed[0];
                        Ok(())
                    })),
                })
            })
            .unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        for failure in 1..=6 {
            for seed in [0., 1.] {
                mode.store(failure, Ordering::Relaxed);
                let mut out = [19.];
                let mut g = [23.];
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    p.value_and_vjp(&[0.5], &[seed], &mut out, &mut g, &mut w)
                }));
                match failure {
                    1 => assert!(matches!(result, Ok(Err(E::Shape)))),
                    3 => assert!(matches!(result, Ok(Err(E::Name)))),
                    6 => assert!(result.is_err()),
                    _ => assert!(matches!(result, Ok(Err(E::Numerical)))),
                }
                assert_eq!(out, [19.]);
                assert_eq!(g, [23.]);
                mode.store(0, Ordering::Relaxed);
                p.value_and_vjp(&[0.75], &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!(out, [0.5625]);
                assert_eq!(g, [1.5 * seed]);
            }
        }
    }

    #[test]
    fn missing_vjp_is_structural_not_a_fake_zero_gradient_or_source_capability() {
        let mut b = TensorBuilder::new();
        let x = b.input("x", shape(1)).unwrap();
        let y = b
            .register(unary_signature(1), &[x], || {
                let mut k = identity();
                k.vjp = None;
                Ok(k)
            })
            .unwrap();
        let total = b.sum(y).unwrap();
        let p = b.compile(total).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        let mut out = [17.];
        let mut g = [19.];
        p.evaluate(&[2.], &mut out, &mut w).unwrap();
        assert_eq!(out, [2.]);
        let error = p
            .value_and_vjp(&[2.], &[0.], &mut out, &mut g, &mut w)
            .unwrap_err();
        assert!(matches!(error,E::Factor(e) if e.downcast_ref::<MissingVjp>().is_some()));
        assert_eq!(out, [2.]);
        assert_eq!(g, [19.]);
        assert!(matches!(
            p.pullback_source("unsupported"),
            Err(E::MissingPrimal { .. })
        ));
        assert!(p.rust_function().contains("compile_error!"));
        let mut model = super::super::super::assembly::ModelBuilder::new();
        let x = model.real("x").unwrap().expr();
        let factor = TensorFactor::new(p, &[x]).unwrap();
        assert!(factor.source_rule().is_none());
        let root = model.primitive(factor).unwrap();
        model.potential("deterministic", root).unwrap();
        assert!(matches!(
            model.compile().unwrap().rust_primal_with_workspace(),
            Err(E::MissingPrimal { .. })
        ));
    }

    #[test]
    fn lazy_native_regions_skip_unchosen_opaque_failures_and_recover_from_selected_errors_and_unwind()
     {
        let mode = Arc::new(AtomicUsize::new(1));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut branch = TensorBuilder::new();
        let x = branch.input("x", Shape::scalar()).unwrap();
        let flag = Arc::clone(&mode);
        let count = Arc::clone(&calls);
        let out = branch
            .register(
                Signature {
                    inputs: vec![ty(DType::Real, Shape::scalar())],
                    output: ty(DType::Real, Shape::scalar()),
                    scratch: 0,
                },
                &[x],
                || {
                    let mut k = identity();
                    k.forward = Box::new(move |x, out, _| {
                        count.fetch_add(1, Ordering::Relaxed);
                        match flag.load(Ordering::Relaxed) {
                            1 => return Err(E::Name),
                            2 => panic!("selected opaque region"),
                            _ => {}
                        }
                        out.copy_from_slice(x.real(0)?);
                        Ok(())
                    });
                    Ok(k)
                },
            )
            .unwrap();
        let yes = branch.compile(out).unwrap().bind(&[]).unwrap();
        let mut branch = TensorBuilder::new();
        let x = branch.input("x", Shape::scalar()).unwrap();
        let no = branch.compile(x).unwrap().bind(&[]).unwrap();
        let mut b = TensorBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let zero = b.constant(Shape::scalar(), vec![0.]).unwrap();
        let out = b.cond(x, zero, Comparison::Greater, &[x], yes, no).unwrap();
        let p = b.compile(out).unwrap().bind(&[]).unwrap();
        let mut w = p.workspace();
        for failure in [1, 2] {
            for seed in [0., 1.] {
                mode.store(failure, Ordering::Relaxed);
                let (mut out, mut g) = ([91.], [92.]);
                let before = calls.load(Ordering::Relaxed);
                p.value_and_vjp(&[-2.], &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!((out, g), ([-2.], [seed]));
                assert_eq!(calls.load(Ordering::Relaxed), before);
                let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    p.value_and_vjp(&[2.], &[seed], &mut out, &mut g, &mut w)
                }));
                if failure == 1 {
                    assert!(matches!(failed, Ok(Err(E::Name))));
                } else {
                    assert!(failed.is_err());
                }
                assert_eq!((out, g), ([-2.], [seed]));
                mode.store(0, Ordering::Relaxed);
                p.value_and_vjp(&[3.], &[seed], &mut out, &mut g, &mut w)
                    .unwrap();
                assert_eq!((out, g), ([3.], [seed]));
            }
        }
        assert!(matches!(
            p.pullback_source("native_only"),
            Err(E::MissingPrimal { .. })
        ));
    }

    #[test]
    fn unqualified_opaque_callbacks_cannot_be_silently_checkpoint_replayed() {
        let mut body = TensorBuilder::new();
        let carry = body.input("carry", Shape::scalar()).unwrap();
        body.input("row", Shape::scalar()).unwrap();
        let signature = Signature {
            inputs: vec![ty(DType::Real, Shape::scalar())],
            output: ty(DType::Real, Shape::scalar()),
            scratch: 0,
        };
        let output = body
            .register(signature, &[carry], || Ok(identity()))
            .unwrap();
        let body = body.compile(output).unwrap().bind(&[]).unwrap();
        let mut b = TensorBuilder::new();
        let initial = b.input("initial", Shape::scalar()).unwrap();
        let xs = b.constant(shape(128), vec![0.; 128]).unwrap();
        let before = (b.nodes.len(), b.slots);
        assert!(matches!(
            b.checkpointed_scan(initial, xs, &[], body.clone()),
            Err(E::Shape)
        ));
        assert_eq!((b.nodes.len(), b.slots), before);
        // Rejection of replay is not rejection of ordinary native composition.
        let output = b.scan(initial, xs, &[], body).unwrap();
        let p = b.compile(output).unwrap().bind(&[]).unwrap();
        let mut out = [0.];
        let mut g = [0.];
        p.value_and_vjp(&[0.75], &[2.], &mut out, &mut g, &mut p.workspace())
            .unwrap();
        assert_eq!(out, [0.75]);
        assert_eq!(g, [2.]);
        let factor = TensorFactor::new(
            p.clone(),
            &[crate::model::assembly::ModelBuilder::new()
                .real("x")
                .unwrap()
                .expr()],
        )
        .unwrap();
        assert!(factor.source_rule().is_none());
        assert!(matches!(
            p.pullback_source("nested_opaque"),
            Err(E::MissingPrimal { .. })
        ));
    }

    #[test]
    fn repeated_argument_contributions_keep_declared_order_before_graph_accumulation() {
        let mut b = TensorBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let signature = Signature {
            inputs: vec![ty(DType::Real, Shape::scalar()); 3],
            output: ty(DType::Real, Shape::scalar()),
            scratch: 0,
        };
        let y = b
            .register(signature, &[x, x, x], || {
                Ok(Kernel {
                    binding: None,
                    forward: Box::new(|args, out, _| {
                        out[0] =
                            (1e16 * args.real(0)?[0] + args.real(1)?[0]) - 1e16 * args.real(2)?[0];
                        Ok(())
                    }),
                    vjp: Some(Box::new(|_, _, seed, g, _| {
                        for (g, score) in g.iter_mut().zip([1e16, 1., -1e16]) {
                            *g = score * seed[0];
                        }
                        Ok(())
                    })),
                })
            })
            .unwrap();
        let p = b.compile(y).unwrap().bind(&[]).unwrap();
        let mut out = [0.];
        let mut g = [0.];
        p.value_and_vjp(&[1.], &[1.], &mut out, &mut g, &mut p.workspace())
            .unwrap();
        let mut ordered = 0.0_f64;
        for contribution in [1e16, 1., -1e16] {
            ordered += contribution;
        }
        // This is an ORDER oracle, not a claim of exact real-arithmetic scores.
        assert_eq!(g[0].to_bits(), ordered.to_bits());
    }
}
