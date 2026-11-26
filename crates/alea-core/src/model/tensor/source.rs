//! Loop-preserving scalar Rust lowering for actual ahead-of-time compiler AD.
use super::analysis::{FrameInstance, ResidualGuard};
use super::*;
use lowering::Operation;
use std::fmt::Write;

impl BoundProgram {
    /// Emits `tensor_region(x: &[f64; D], valid: &mut bool) -> [f64; N]`.
    /// Set `valid=true` before calling; failures set it false, including for empty
    /// results. Data are frozen from THIS binding. Rebinding requires regenerating
    /// code. This standalone primal wrapper owns local scratch; for compiler AD
    /// use the workspace-explicit compiled-model exporter. Arrays are fixed-size;
    /// this is not a dynamic-shape/JIT/GPU backend.
    /// # Errors
    /// Returns `MissingPrimal` for reachable native-only operations, including
    /// matrix providers without a qualified source lowering.
    pub fn rust_function(&self) -> Result<String, E> {
        if let Some(node) = registration::missing_source(&self.plan) {
            return Err(E::MissingPrimal { node });
        }
        let dimension = self.dimension();
        let len = self.output_shape().elements;
        let slots = self.plan.source_slots;
        Ok(format!(
            "{}\nfn tensor_region(x: &[f64; {dimension}], valid: &mut bool) -> [f64; {len}] {{ let mut scratch=[0.0_f64; {slots}]; let (out, work)=scratch.split_at_mut({len}); tensor_region_work(x, out, work, valid); ::std::array::from_fn(|j| out[j]) }}",
            self.function_source("tensor_region_work")
        ))
    }
    pub(super) fn scalar_source(&self) -> String {
        format!(
            "{}\nlet mut valid=true; let (out, work)=scratch.split_at_mut(1); tensor_region_work(x, out, work, &mut valid); if valid {{ out[0] }} else {{ f64::NAN }}",
            self.function_source("tensor_region_work")
        )
    }
    fn guarded_function_source(&self, name: &str, guard: ResidualGuard) -> String {
        // Generated child functions are lexical residual scopes. Their binding
        // must be the child selected by the original canonical call, just as in
        // native frames. Iteration zero denotes the static body scope here;
        // emitted step loops supply the dynamic iteration inside that scope.
        assert_eq!(
            guard.child, self.plan.owner,
            "compiled child residual scope"
        );
        self.function_source(name)
    }
    pub(super) fn function_source(&self, name: &str) -> String {
        let p = &self.plan;
        // Public emission, registration and explicit pullbacks all check the
        // same recursive capability guard before entering this renderer.
        debug_assert!(registration::missing_source(p).is_none());
        let failure = "*valid=false; return;";
        let mut source = format!(
            "#[inline(always)]\nfn {name}(x: &[f64], out: &mut [f64], scratch: &mut [f64], valid: &mut bool) {{\nlet _ = (x, &*valid, &*out); let rest=scratch;\n"
        );
        writeln!(
            source,
            "if x.iter().any(|v| !v.is_finite()) {{ {failure} }}"
        )
        .unwrap();
        let schedule = fusion::schedule(p, false);
        let mut groups = schedule
            .iter()
            .filter_map(|step| match step {
                fusion::Step::Fused(group) => Some(group),
                _ => None,
            })
            .peekable();
        let mut fused_until = 0;
        for call in p.facts.lowered.calls() {
            let i = call.origin().node;
            if i < fused_until {
                continue;
            }
            if groups.peek().is_some_and(|group| group.start() == i) {
                let group = groups.next().expect("peeked fusion group");
                fused_source(&mut source, p, group, failure);
                fused_until = group.end();
                continue;
            }
            let len = call.elements();
            let operation = call.operation(&p.nodes);
            let borrowed = match operation {
                Operation::Input(index) => {
                    let start = p.inputs[index].start;
                    let end = start + len;
                    Some(format!("&x[{start}..{end}]"))
                }
                Operation::Data(index) => Some(format!("&{:?}", self.data[index])),
                Operation::Constant(values) => Some(format!("&{values:?}")),
                Operation::Alias(a) => Some(format!("n{}", a.node)),
                _ => None,
            };
            if let Some(value) = borrowed {
                writeln!(source,"let n{i}: &[f64] = {value}; if n{i}.iter().any(|v| !v.is_finite()) {{ {failure} }}").unwrap();
                continue;
            }
            writeln!(source, "let (n{i}, rest)=rest.split_at_mut({len});").unwrap();
            if let Some(binding::Fixed::Value(values)) = self.fixed(i) {
                // Keep the existing scratch/residual layout: a prepared read
                // replaces arithmetic, not its identity or publication boundary.
                // Scalar FP stores retain type information for the pinned AD
                // compiler; an anonymous constant memcpy is not type-qualified.
                for (j, value) in values.iter().enumerate() {
                    writeln!(source, "n{i}[{j}] = {value:?}_f64;").unwrap();
                }
                continue;
            }
            // Empty pure blocks do no element work. Control-flow regions still
            // execute: an empty result cannot erase a selected domain failure.
            if len == 0
                && !matches!(
                    operation,
                    Operation::Cond(..) | Operation::Scan(..) | Operation::Registered(..)
                )
            {
                writeln!(source, "let _ = &n{i};").unwrap();
                continue;
            }
            match operation {
                Operation::Registered(block, reads) => {
                    let name = format!("registered{i}");
                    source.push_str(&block.source_definition(&name));
                    let args = reads
                        .iter()
                        .map(|read| format!("n{},", read.value.node))
                        .collect::<String>();
                    writeln!(source,"n{i}.fill(f64::NAN); if !{name}(&[{args}],n{i},None,&mut[]) {{ {failure} }}").unwrap();
                }
                Operation::NativeOnly => {
                    unreachable!("native-only operation rejected before emission")
                }
                Operation::Input(_)
                | Operation::Data(_)
                | Operation::Constant(_)
                | Operation::Alias(_) => {
                    unreachable!("borrowed nodes emitted above")
                }
                Operation::Unary(..) | Operation::Binary(..) => {
                    let value = element_source(operation);
                    if len == 1 {
                        writeln!(source, "{{ let j=0_usize; n{i}[j] = {value}; }}").unwrap();
                    } else {
                        writeln!(source, "for j in 0..{len} {{ n{i}[j] = {value}; }}").unwrap();
                    }
                }
                Operation::Sum(a) => {
                    writeln!(source,"{{ let mut sum=::alea::math::numeric::CompensatedSum::default(); for &v in n{}.iter() {{ sum.add(v); }} n{i}[0]=sum.total(); }}",a.value.node).unwrap();
                }
                Operation::Gather(a, indices) => {
                    writeln!(source,"{{ let indices: [usize; {len}] = {indices:?}; for j in 0..{len} {{n{i}[j]=n{}[indices[j]];}} }}",a.node).unwrap();
                }
                Operation::Matrix(a, b, calls) => {
                    let kernel = calls.kernel;
                    let (m, k, n) = calls.primal.dimensions;
                    if kernel.is_accelerated() {
                        let (ta, tb) = calls.primal.transpose;
                        let add = calls.primal.add;
                        writeln!(source, "if !::alea::math::gemm::try_gemm(::alea::math::gemm::ResolvedMatrix::{},n{},n{},n{i},({m},{k},{n}),({ta},{tb}),{add}) {{{failure}}}", kernel.source_name(),a.node,b.node).unwrap();
                        writeln!(
                            source,
                            "if n{i}.iter().any(|v|!v.is_finite()) {{{failure}}}"
                        )
                        .unwrap();
                        continue;
                    }
                    if k == 0 {
                        writeln!(source, "n{i}.fill(0.0);").unwrap();
                        continue;
                    }
                    let row_k = scaled("row", k);
                    let j_n = scaled("j", n);
                    let row_n = scaled("row", n);
                    writeln!(source,"for row in 0..{m} {{for column in 0..{n} {{ let mut sum=::alea::math::numeric::CompensatedSum::default(); for j in 0..{k} {{sum.add_product(n{}[{row_k}+j],n{}[{j_n}+column]);}} n{i}[{row_n}+column]=sum.total(); }} }}",a.node,b.node).unwrap();
                }
                Operation::Cond(c, [lhs, rhs], args) => {
                    let site = call.origin();
                    let yes = p
                        .facts
                        .guard(site, FrameInstance::Branch(0))
                        .expect("compiled yes scope");
                    let no = p
                        .facts
                        .guard(site, FrameInstance::Branch(1))
                        .expect("compiled no scope");
                    source.push_str(
                        &c.yes
                            .guarded_function_source(&format!("yes{}", yes.site.node), yes),
                    );
                    source.push_str(
                        &c.no
                            .guarded_function_source(&format!("no{}", no.site.node), no),
                    );
                    let (arguments, child) = operation.child_scratch();
                    writeln!(
                        source,
                        "let (args{i}, rest)=rest.split_at_mut({arguments}); let (work{i}, rest)=rest.split_at_mut({child});",
                    )
                    .unwrap();
                    pack(&mut source, &format!("args{i}"), 0, args);
                    let comparison = match c.comparison {
                        Comparison::Less => "<",
                        Comparison::LessEqual => "<=",
                        Comparison::Greater => ">",
                        Comparison::GreaterEqual => ">=",
                    };
                    writeln!(source,"if n{}[0] {comparison} n{}[0] {{yes{i}(args{i}, n{i}, work{i}, valid)}} else {{no{i}(args{i}, n{i}, work{i}, valid)}}; if !*valid {{ {failure} }}",lhs.node,rhs.node).unwrap();
                }
                Operation::Scan(s, [initial, xs], captures) => {
                    if s.steps == 0 {
                        copy(
                            &mut source,
                            &format!("n{i}"),
                            &format!("n{}", initial.node),
                            len,
                        );
                        continue;
                    }
                    let site = call.origin();
                    let guard = p
                        .facts
                        .guard(site, FrameInstance::Iteration(0))
                        .expect("compiled scan scope");
                    source.push_str(
                        &s.body
                            .guarded_function_source(&format!("body{}", guard.site.node), guard),
                    );
                    let carry = len;
                    let row = s.row;
                    let initial = initial.node;
                    let xs = xs.node;
                    let steps = s.steps;
                    let (arguments, child) = operation.child_scratch();
                    writeln!(
                        source,
                        "let (args{i}, rest)=rest.split_at_mut({arguments}); let (work{i}, rest)=rest.split_at_mut({child});",
                    )
                    .unwrap();
                    copy(&mut source, &format!("n{i}"), &format!("n{initial}"), carry);
                    pack(&mut source, &format!("args{i}"), carry + row, captures);
                    let start = scaled("step", row);
                    let end = scaled("(step+1)", row);
                    writeln!(source, "for step in 0..{steps} {{ let _ = step;").unwrap();
                    copy(
                        &mut source,
                        &format!("args{i}[..{carry}]"),
                        &format!("n{i}"),
                        carry,
                    );
                    copy(
                        &mut source,
                        &format!("args{i}[{carry}..{}]", carry + row),
                        &format!("n{xs}[{start}..{end}]"),
                        row,
                    );
                    writeln!(
                        source,
                        "body{i}(args{i}, n{i}, work{i}, valid); if !*valid {{ {failure} }} }}"
                    )
                    .unwrap();
                }
            }
            writeln!(
                source,
                "if n{i}.iter().any(|v| !v.is_finite()) {{ {failure} }}"
            )
            .unwrap();
        }
        writeln!(source, "let _ = rest;").unwrap();
        copy(
            &mut source,
            "out",
            &format!("n{}", p.output.node),
            self.output_shape().elements,
        );
        source.push_str("}\n");
        source
    }
}
fn element_source(operation: Operation<'_>) -> String {
    match operation {
        Operation::Unary(op, a) => {
            let a = format!("n{}[j]", a.node);
            match op {
                UnaryOp::Neg => format!("-{a}"),
                UnaryOp::Exp => format!("{a}.exp()"),
                UnaryOp::Log => format!("{a}.ln()"),
                UnaryOp::Log1p => format!("{a}.ln_1p()"),
                UnaryOp::Expm1 => format!("{a}.exp_m1()"),
                UnaryOp::Tanh => format!("{a}.tanh()"),
                UnaryOp::Sin => format!("{a}.sin()"),
                UnaryOp::Cos => format!("{a}.cos()"),
                UnaryOp::Sqrt => format!("{a}.sqrt()"),
            }
        }
        Operation::Binary(kind, (a, left), (b, right)) => {
            let op = match kind {
                BinaryOp::Add => "+",
                BinaryOp::Subtract => "-",
                BinaryOp::Multiply => "*",
                BinaryOp::Divide => "/",
            };
            format!(
                "n{}[{}] {op} n{}[{}]",
                a.node,
                left.index_source(),
                b.node,
                right.index_source()
            )
        }
        _ => unreachable!("pointwise expression"),
    }
}
fn fused_source(source: &mut String, p: &Plan, group: &fusion::Group, failure: &str) {
    for i in group.maps().chain(group.reduction) {
        writeln!(
            source,
            "let (n{i}, rest)=rest.split_at_mut({});",
            p.facts
                .lowered
                .call(i)
                .expect("checked fused call")
                .elements()
        )
        .unwrap();
    }
    let elements = group.domain.end;
    source.push_str("{\n");
    if group.reduction.is_some() {
        source.push_str("let mut total=::alea::math::numeric::CompensatedSum::default();\n");
    }
    writeln!(source, "for j in 0..{elements} {{").unwrap();
    for i in group.maps() {
        writeln!(
            source,
            "n{i}[j]={}; if !n{i}[j].is_finite() {{ {failure} }}",
            element_source(
                p.facts
                    .lowered
                    .call(i)
                    .expect("checked fused call")
                    .operation(&p.nodes)
            )
        )
        .unwrap();
    }
    if let Some(i) = group.reduction {
        let Operation::Sum(a) = p
            .facts
            .lowered
            .call(i)
            .expect("checked fused reduction")
            .operation(&p.nodes)
        else {
            unreachable!("compiled reduction")
        };
        writeln!(source, "total.add(n{}[j]);", a.value.node).unwrap();
    }
    source.push_str("}\n");
    if let Some(i) = group.reduction {
        writeln!(
            source,
            "n{i}[0]=total.total(); if !n{i}[0].is_finite() {{ {failure} }}"
        )
        .unwrap();
    }
    source.push_str("}\n");
}
fn scaled(index: &str, stride: usize) -> String {
    match stride {
        0 => "0".into(),
        1 => index.into(),
        _ => format!("{index}*{stride}"),
    }
}
fn pack(source: &mut String, name: &str, mut start: usize, args: &[lowering::Read]) {
    for arg in args {
        let end = start + arg.elements();
        copy(
            source,
            &format!("{name}[{start}..{end}]"),
            &format!("n{}", arg.value.node),
            end - start,
        );
        start = end;
    }
}
fn copy(source: &mut String, dst: &str, src: &str, len: usize) {
    if len == 1 {
        writeln!(source, "{dst}[0]={src}[0];").unwrap();
    } else if len != 0 {
        writeln!(
            source,
            "for (a,b) in {dst}.iter_mut().zip({src}.iter()) {{ *a = *b; }}"
        )
        .unwrap();
    }
}
