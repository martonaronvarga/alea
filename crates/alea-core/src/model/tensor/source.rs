//! Loop-preserving scalar Rust lowering for actual ahead-of-time compiler AD.
use super::*;
use std::fmt::Write;

pub(super) fn workspace_elements(p: &Plan) -> Result<usize, E> {
    let mut slots = p.nodes[p.output.node].shape.elements;
    for (i, node) in p.nodes.iter().enumerate() {
        if !p.active[i] {
            continue;
        }
        let extra = match &node.op {
            Op::Cond(c) => c
                .yes
                .dimension()
                .checked_add(c.yes.plan.source_slots.max(c.no.plan.source_slots)),
            Op::Scan(s) if s.steps != 0 => s.body.dimension().checked_add(s.body.plan.source_slots),
            _ => Some(0),
        }
        .ok_or(E::Shape)?;
        let own = if matches!(
            &node.op,
            Op::Input(_) | Op::Data(_) | Op::Constant(_) | Op::Alias(_)
        ) {
            0
        } else {
            node.shape.elements
        };
        slots = slots
            .checked_add(own)
            .and_then(|n| n.checked_add(extra))
            .filter(|n| *n <= isize::MAX as usize / size_of::<f64>())
            .ok_or(E::Shape)?;
    }
    Ok(slots)
}

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
                fusion::Step::Nodes(_) => None,
            })
            .peekable();
        let mut fused_until = 0;
        for (i, node) in p.nodes.iter().enumerate() {
            if !p.active[i] || i < fused_until {
                continue;
            }
            if groups.peek().is_some_and(|group| group.maps.start == i) {
                let group = groups.next().expect("peeked fusion group");
                fused_source(&mut source, p, group, failure);
                fused_until = group.end();
                continue;
            }
            let len = node.shape.elements;
            let borrowed = match &node.op {
                Op::Input(index) => {
                    let start = p.inputs[*index].start;
                    let end = start + len;
                    Some(format!("&x[{start}..{end}]"))
                }
                Op::Data(index) => Some(format!("&{:?}", self.data[*index])),
                Op::Constant(values) => Some(format!("&{values:?}")),
                Op::Alias(a) => Some(format!("n{}", a.node)),
                _ => None,
            };
            if let Some(value) = borrowed {
                writeln!(source,"let n{i}: &[f64] = {value}; if n{i}.iter().any(|v| !v.is_finite()) {{ {failure} }}").unwrap();
                continue;
            }
            writeln!(source, "let (n{i}, rest)=rest.split_at_mut({len});").unwrap();
            // Empty pure blocks do no element work. Control-flow regions still
            // execute: an empty result cannot erase a selected domain failure.
            if len == 0 && !matches!(&node.op, Op::Cond(_) | Op::Scan(_)) {
                writeln!(source, "let _ = &n{i};").unwrap();
                continue;
            }
            match &node.op {
                Op::Registered(_) => unreachable!("native-only operation rejected before emission"),
                Op::Input(_) | Op::Data(_) | Op::Constant(_) | Op::Alias(_) => {
                    unreachable!("borrowed nodes emitted above")
                }
                Op::Unary(..) | Op::Binary(_) => {
                    let value = element_source(node);
                    if len == 1 {
                        writeln!(source, "{{ let j=0_usize; n{i}[j] = {value}; }}").unwrap();
                    } else {
                        writeln!(source, "for j in 0..{len} {{ n{i}[j] = {value}; }}").unwrap();
                    }
                }
                Op::Sum(a) => {
                    writeln!(source,"{{ let mut sum=::alea::math::numeric::CompensatedSum::default(); for &v in n{}.iter() {{ sum.add(v); }} n{i}[0]=sum.total(); }}",a.node).unwrap();
                }
                Op::Gather(a, indices) => {
                    writeln!(source,"{{ let indices: [usize; {len}] = {indices:?}; for j in 0..{len} {{n{i}[j]=n{}[indices[j]];}} }}",a.node).unwrap();
                }
                Op::Matmul {
                    a,
                    b,
                    m,
                    k,
                    n,
                    kernel,
                } => {
                    if *kernel != MatrixKernel::Compensated {
                        writeln!(source, "if !::alea::math::gemm::try_gemm(::alea::math::gemm::MatrixKernel::{kernel:?},n{},n{},n{i},({m},{k},{n}),(false,false),false) {{{failure}}}", a.node,b.node).unwrap();
                        writeln!(
                            source,
                            "if n{i}.iter().any(|v|!v.is_finite()) {{{failure}}}"
                        )
                        .unwrap();
                        continue;
                    }
                    if *k == 0 {
                        writeln!(source, "n{i}.fill(0.0);").unwrap();
                        continue;
                    }
                    let row_k = scaled("row", *k);
                    let j_n = scaled("j", *n);
                    let row_n = scaled("row", *n);
                    writeln!(source,"for row in 0..{m} {{for column in 0..{n} {{ let mut sum=::alea::math::numeric::CompensatedSum::default(); for j in 0..{k} {{sum.add_product(n{}[{row_k}+j],n{}[{j_n}+column]);}} n{i}[{row_n}+column]=sum.total(); }} }}",a.node,b.node).unwrap();
                }
                Op::Cond(c) => {
                    source.push_str(&c.yes.function_source(&format!("yes{i}")));
                    source.push_str(&c.no.function_source(&format!("no{i}")));
                    writeln!(
                        source,
                        "let (args{i}, rest)=rest.split_at_mut({}); let (work{i}, rest)=rest.split_at_mut({});",
                        c.yes.dimension(), c.yes.plan.source_slots.max(c.no.plan.source_slots)
                    )
                    .unwrap();
                    pack(&mut source, &format!("args{i}"), 0, &c.args, p);
                    let comparison = match c.comparison {
                        Comparison::Less => "<",
                        Comparison::LessEqual => "<=",
                        Comparison::Greater => ">",
                        Comparison::GreaterEqual => ">=",
                    };
                    writeln!(source,"if n{}[0] {comparison} n{}[0] {{yes{i}(args{i}, n{i}, work{i}, valid)}} else {{no{i}(args{i}, n{i}, work{i}, valid)}}; if !*valid {{ {failure} }}",c.lhs.node,c.rhs.node).unwrap();
                }
                Op::Scan(s) => {
                    if s.steps == 0 {
                        copy(
                            &mut source,
                            &format!("n{i}"),
                            &format!("n{}", s.initial.node),
                            len,
                        );
                        continue;
                    }
                    source.push_str(&s.body.function_source(&format!("body{i}")));
                    let carry = node.shape.elements;
                    let row = s.row;
                    let initial = s.initial.node;
                    let xs = s.xs.node;
                    let steps = s.steps;
                    writeln!(
                        source,
                        "let (args{i}, rest)=rest.split_at_mut({}); let (work{i}, rest)=rest.split_at_mut({});",
                        s.body.dimension(), s.body.plan.source_slots
                    )
                    .unwrap();
                    copy(&mut source, &format!("n{i}"), &format!("n{initial}"), carry);
                    pack(
                        &mut source,
                        &format!("args{i}"),
                        carry + row,
                        &s.captures,
                        p,
                    );
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
fn element_source(node: &Node) -> String {
    match &node.op {
        Op::Unary(op, a) => {
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
        Op::Binary(b) => {
            let op = match b.kind {
                BinaryOp::Add => "+",
                BinaryOp::Subtract => "-",
                BinaryOp::Multiply => "*",
                BinaryOp::Divide => "/",
            };
            format!(
                "n{}[{}] {op} n{}[{}]",
                b.a.node,
                b.left.index_source(),
                b.b.node,
                b.right.index_source()
            )
        }
        _ => unreachable!("pointwise expression"),
    }
}
fn fused_source(source: &mut String, p: &Plan, group: &fusion::Group, failure: &str) {
    for i in group.maps.clone().chain(group.reduction) {
        writeln!(
            source,
            "let (n{i}, rest)=rest.split_at_mut({});",
            p.nodes[i].shape.elements
        )
        .unwrap();
    }
    let elements = group.elements;
    source.push_str("{\n");
    if group.reduction.is_some() {
        source.push_str("let mut total=::alea::math::numeric::CompensatedSum::default();\n");
    }
    writeln!(source, "for j in 0..{elements} {{").unwrap();
    for i in group.maps.clone() {
        writeln!(
            source,
            "n{i}[j]={}; if !n{i}[j].is_finite() {{ {failure} }}",
            element_source(&p.nodes[i])
        )
        .unwrap();
    }
    if let Some(i) = group.reduction {
        let Op::Sum(a) = p.nodes[i].op else {
            unreachable!("compiled reduction")
        };
        writeln!(source, "total.add(n{}[j]);", a.node).unwrap();
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
fn pack(source: &mut String, name: &str, mut start: usize, args: &[Value], p: &Plan) {
    for arg in args {
        let end = start + p.nodes[arg.node].shape.elements;
        copy(
            source,
            &format!("{name}[{start}..{end}]"),
            &format!("n{}", arg.node),
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
