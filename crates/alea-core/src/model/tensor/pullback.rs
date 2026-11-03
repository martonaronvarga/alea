//! Explicit first-order source reverse with segmented scan rematerialization.
//! Forward AD over this pullback is NOT qualified on the pinned compiler.
use super::*;
use std::fmt::Write;

pub(crate) struct PullbackSource {
    pub source: String,
    pub workspace: usize,
}
fn add(a: usize, b: usize) -> Result<usize, E> {
    a.checked_add(b)
        .filter(|&v| v <= isize::MAX as usize / size_of::<f64>())
        .ok_or(E::Shape)
}
fn mul(a: usize, b: usize) -> Result<usize, E> {
    a.checked_mul(b)
        .filter(|&v| v <= isize::MAX as usize / size_of::<f64>())
        .ok_or(E::Shape)
}
impl BoundProgram {
    pub(crate) fn has_source_checkpoints(&self) -> bool {
        self.plan.nodes.iter().enumerate().any(|(i, n)| {
            self.plan.active[i]
                && match &n.op {
                    Op::Scan(s) => s.checkpointed || s.body.has_source_checkpoints(),
                    Op::Cond(c) => c.yes.has_source_checkpoints() || c.no.has_source_checkpoints(),
                    _ => false,
                }
        })
    }
    fn activity(&self, inputs: &[bool]) -> Vec<bool> {
        let p = &self.plan;
        let mut active = vec![false; p.nodes.len()];
        for (i, n) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                continue;
            }
            active[i] = match &n.op {
                Op::Input(j) => inputs[*j],
                Op::Cond(c) => {
                    let args: Vec<_> = c.args.iter().map(|v| active[v.node]).collect();
                    c.yes.activity(&args)[c.yes.plan.output.node]
                        || c.no.activity(&args)[c.no.plan.output.node]
                }
                Op::Scan(s) if s.steps != 0 => {
                    let args: Vec<_> = [s.initial, s.xs]
                        .iter()
                        .chain(s.captures.iter())
                        .map(|v| active[v.node])
                        .collect();
                    s.body.activity(&args)[s.body.plan.output.node]
                }
                Op::Scan(s) => active[s.initial.node],
                _ => n.inputs().iter().any(|v| active[v.node]),
            };
        }
        active
    }
    pub(crate) fn pullback_source(&self, name: &str) -> Result<PullbackSource, E> {
        self.pullback_with_activity(name, &vec![true; self.plan.inputs.len()])
    }
    fn pullback_with_activity(&self, name: &str, inputs: &[bool]) -> Result<PullbackSource, E> {
        let p = &self.plan;
        let active = self.activity(inputs);
        let mut offsets = vec![0; p.nodes.len()];
        let mut cursor = self.output_shape().elements;
        for (i, n) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                continue;
            }
            if !matches!(
                n.op,
                Op::Input(_) | Op::Data(_) | Op::Constant(_) | Op::Alias(_)
            ) {
                offsets[i] = cursor;
                cursor = add(cursor, n.shape.elements)?;
            }
            cursor = match &n.op {
                Op::Cond(c) => add(
                    cursor,
                    add(
                        c.yes.dimension(),
                        c.yes.plan.source_slots.max(c.no.plan.source_slots),
                    )?,
                )?,
                Op::Scan(s) if s.steps != 0 => {
                    add(cursor, add(s.body.dimension(), s.body.plan.source_slots)?)?
                }
                _ => cursor,
            };
        }
        debug_assert_eq!(cursor, p.source_slots);
        let mut source = self.function_source(&format!("{name}_primal"));
        let mut reverse = String::new();
        let mut extra = 0;
        let value = |v: Value, j: &str| value_source(p, &offsets, v, j);
        for (i, n) in p.nodes.iter().enumerate().rev() {
            if !p.active[i] || !active[i] {
                continue;
            }
            let base = n.start;
            let len = n.shape.elements;
            match &n.op {
                Op::Input(j) => {
                    let input = &p.inputs[*j];
                    writeln!(
                        reverse,
                        "for j in 0..{len} {{ g[{}+j]=a[{base}+j]; }}",
                        input.start
                    )
                    .unwrap();
                }
                Op::Data(_) | Op::Constant(_) | Op::Alias(_) => {}
                Op::Unary(op, v) => {
                    let x = value(*v, "j");
                    let y = value(
                        Value {
                            owner: p.owner,
                            node: i,
                        },
                        "j",
                    );
                    let derivative = match op {
                        UnaryOp::Neg => "-1.0_f64".into(),
                        UnaryOp::Exp => y.clone(),
                        UnaryOp::Log => format!("1.0/({x})"),
                        UnaryOp::Log1p => format!("1.0/(1.0+({x}))"),
                        UnaryOp::Expm1 => format!("({x}).exp()"),
                        UnaryOp::Tanh => format!("1.0-({y})*({y})"),
                        UnaryOp::Sin => format!("({x}).cos()"),
                        UnaryOp::Cos => format!("-({x}).sin()"),
                        UnaryOp::Sqrt => format!("0.5/({y})"),
                    };
                    writeln!(reverse,"for j in 0..{len} {{ let s=a[{base}+j]; let d={derivative}; if s!=0.0 || d.is_finite() {{ a[{}+j]+=s*d; }} }}",p.nodes[v.node].start).unwrap();
                }
                Op::Binary(b) => {
                    let l = map_source(&b.left);
                    let r = map_source(&b.right);
                    let x = value(b.a, &l);
                    let y = value(b.b, &r);
                    let (dl, dr) = match b.kind {
                        BinaryOp::Add => ("1.0_f64".into(), "1.0_f64".into()),
                        BinaryOp::Subtract => ("1.0_f64".into(), "-1.0_f64".into()),
                        BinaryOp::Multiply => (y.clone(), x.clone()),
                        BinaryOp::Divide => (format!("1.0/({y})"), format!("-(({x})/({y}))/({y})")),
                    };
                    writeln!(reverse, "for j in 0..{len} {{ let s=a[{base}+j];").unwrap();
                    if active[b.a.node] {
                        writeln!(
                            reverse,
                            "{{let d={dl};if s!=0.0 || d.is_finite() {{a[{}+{l}]+=s*d;}}}}",
                            p.nodes[b.a.node].start
                        )
                        .unwrap();
                    }
                    if active[b.b.node] {
                        writeln!(
                            reverse,
                            "{{let d={dr};if s!=0.0 || d.is_finite() {{a[{}+{r}]+=s*d;}}}}",
                            p.nodes[b.b.node].start
                        )
                        .unwrap();
                    }
                    reverse.push_str("}\n");
                }
                Op::Sum(v) => writeln!(
                    reverse,
                    "for j in 0..{} {{ a[{}+j]+=a[{base}]; }}",
                    p.nodes[v.node].shape.elements, p.nodes[v.node].start
                )
                .unwrap(),
                Op::Gather(v, indices) => writeln!(
                    reverse,
                    "let indices:[usize;{len}]={indices:?};for (j,k) in indices.into_iter().enumerate() {{a[{}+k]+=a[{base}+j];}}",
                    p.nodes[v.node].start
                )
                .unwrap(),
                Op::Matmul {
                    a: l,
                    b: r,
                    m,
                    k,
                    n,
                } => {
                    if len == 0 {
                        continue;
                    }
                    writeln!(reverse,"for row in 0..{m} {{for col in 0..{n} {{let s=a[{base}+row*{n}+col]; for j in 0..{k} {{").unwrap();
                    if active[l.node] {
                        writeln!(
                            reverse,
                            "a[{}+row*{k}+j]+=s*({});",
                            p.nodes[l.node].start,
                            value(*r, &format!("j*{n}+col"))
                        )
                        .unwrap();
                    }
                    if active[r.node] {
                        writeln!(
                            reverse,
                            "a[{}+j*{n}+col]+=s*({});",
                            p.nodes[r.node].start,
                            value(*l, &format!("row*{k}+j"))
                        )
                        .unwrap();
                    }
                    reverse.push_str("}}}\n");
                }
                Op::Cond(c) => {
                    let args: Vec<_> = c.args.iter().map(|v| active[v.node]).collect();
                    let yes = c
                        .yes
                        .pullback_with_activity(&format!("{name}_yes{i}"), &args)?;
                    let no =
                        c.no.pullback_with_activity(&format!("{name}_no{i}"), &args)?;
                    extra = extra.max(add(
                        mul(c.yes.dimension(), 2)?,
                        add(mul(len, 2)?, yes.workspace.max(no.workspace))?,
                    )?);
                    source.push_str(&yes.source);
                    source.push_str(&no.source);
                    child_buffers(&mut reverse, c.yes.dimension(), len);
                    pack_values(&mut reverse, "args", 0, &c.args, p, &offsets);
                    writeln!(
                        reverse,
                        "seeds.copy_from_slice(&a[{base}..{}]);",
                        base + len
                    )
                    .unwrap();
                    let cmp = match c.comparison {
                        Comparison::Less => "<",
                        Comparison::LessEqual => "<=",
                        Comparison::Greater => ">",
                        Comparison::GreaterEqual => ">=",
                    };
                    writeln!(reverse,"if {} {cmp} {} {{ {name}_yes{i}(args,seeds,result,scores,child,valid); }} else {{ {name}_no{i}(args,seeds,result,scores,child,valid); }} if !*valid {{ return; }}",value(c.lhs,"0"),value(c.rhs,"0")).unwrap();
                    accumulate(&mut reverse, &c.args, p, 0, "scores");
                    reverse.push_str("}\n");
                }
                Op::Scan(s) if s.steps == 0 => writeln!(
                    reverse,
                    "for j in 0..{len} {{a[{}+j]+=a[{base}+j];}}",
                    p.nodes[s.initial.node].start
                )
                .unwrap(),
                Op::Scan(s) => {
                    let first: Vec<_> = [s.initial, s.xs]
                        .iter()
                        .chain(s.captures.iter())
                        .map(|v| active[v.node])
                        .collect();
                    let mut steady = first.clone();
                    steady[0] = s.body.activity(&first)[s.body.plan.output.node];
                    let one = format!("{name}_first{i}");
                    let later = if first == steady {
                        one.clone()
                    } else {
                        format!("{name}_steady{i}")
                    };
                    let body = s.body.pullback_with_activity(&one, &first)?;
                    let mut work = body.workspace;
                    source.push_str(&body.source);
                    if first != steady {
                        let other = s.body.pullback_with_activity(&later, &steady)?;
                        work = work.max(other.workspace);
                        source.push_str(&other.source);
                    }
                    let interval = s.steps.isqrt().max(1);
                    let checkpoints = mul(s.steps.div_ceil(interval), len)?;
                    let states = mul(interval, len)?;
                    extra = extra.max(add(
                        add(mul(s.body.dimension(), 2)?, mul(len, 2)?)?,
                        add(work, add(checkpoints, states)?)?,
                    )?);
                    child_buffers(&mut reverse, s.body.dimension(), len);
                    writeln!(reverse,"let (checkpoints,child)=child.split_at_mut({checkpoints}); let (states,child)=child.split_at_mut({states});").unwrap();
                    pack_values(&mut reverse, "args", 0, &[s.initial], p, &offsets);
                    pack_values(&mut reverse, "args", len + s.row, &s.captures, p, &offsets);
                    writeln!(reverse,"for step in 0..{} {{if step%{interval}==0 {{let at=step/{interval}*{len}; checkpoints[at..at+{len}].copy_from_slice(&args[..{len}]);}}",s.steps).unwrap();
                    pack_row(&mut reverse, s, p, &offsets, len);
                    writeln!(reverse,"{one}_primal(args,result,child,valid); if !*valid {{return;}} args[..{len}].copy_from_slice(result); }}").unwrap();
                    writeln!(reverse,"seeds.copy_from_slice(&a[{base}..{}]);let mut end={}_usize;while end!=0 {{let start=(end-1)/{interval}*{interval};let at=start/{interval}*{len};args[..{len}].copy_from_slice(&checkpoints[at..at+{len}]);for step in start..end {{ let at=(step-start)*{len}; states[at..at+{len}].copy_from_slice(&args[..{len}]);",base+len,s.steps).unwrap();
                    pack_row(&mut reverse, s, p, &offsets, len);
                    writeln!(reverse,"{one}_primal(args,result,child,valid);if !*valid{{return;}}args[..{len}].copy_from_slice(result);}} for step in (start..end).rev() {{let at=(step-start)*{len};args[..{len}].copy_from_slice(&states[at..at+{len}]);").unwrap();
                    pack_row(&mut reverse, s, p, &offsets, len);
                    writeln!(reverse,"if step==0 {{{one}(args,seeds,result,scores,child,valid);}} else {{{later}(args,seeds,result,scores,child,valid);}}if !*valid{{return;}}seeds.copy_from_slice(&scores[..{len}]);").unwrap();
                    writeln!(
                        reverse,
                        "for j in 0..{} {{a[{}+step*{}+j]+=scores[{len}+j];}}",
                        s.row, p.nodes[s.xs.node].start, s.row
                    )
                    .unwrap();
                    accumulate(&mut reverse, &s.captures, p, len + s.row, "scores");
                    writeln!(
                        reverse,
                        "}}end=start;}}for j in 0..{len} {{a[{}+j]+=seeds[j];}} }}",
                        p.nodes[s.initial.node].start
                    )
                    .unwrap();
                }
            }
        }
        let workspace = add(add(p.source_slots, p.slots)?, extra)?;
        let output = self.output_shape().elements;
        writeln!(source,"#[allow(clippy::too_many_arguments,clippy::needless_range_loop)]\nfn {name}(x:&[f64],seed:&[f64],out:&mut[f64],g:&mut[f64],scratch:&mut[f64],valid:&mut bool) {{let (primal,rest)=scratch.split_at_mut({});let (a,extra)=rest.split_at_mut({});a.fill(0.0);g.fill(0.0);{{let (result,work)=primal.split_at_mut({output});{name}_primal(x,result,work,valid);}}if !*valid || seed.iter().any(|s|!s.is_finite()) {{*valid=false;return;}}",p.source_slots,p.slots).unwrap();
        for (i, n) in p.nodes.iter().enumerate() {
            if !p.active[i] {
                continue;
            }
            match &n.op {
                Op::Data(j) => writeln!(
                    source,
                    "let data{i}:&[f64]=&{:?};let _=data{i};",
                    self.data[*j]
                )
                .unwrap(),
                Op::Constant(data) => {
                    writeln!(source, "let data{i}:&[f64]=&{data:?};let _=data{i};").unwrap()
                }
                _ => {}
            }
        }
        writeln!(source,"a[{}..{}].copy_from_slice(seed);{reverse}\nif g.iter().chain(a.iter()).any(|v|!v.is_finite()) {{*valid=false;return;}}out.copy_from_slice(&primal[..{output}]);let _=extra;}}",p.nodes[p.output.node].start,p.nodes[p.output.node].range().end).unwrap();
        Ok(PullbackSource { source, workspace })
    }
}

fn value_source(p: &Plan, offsets: &[usize], mut v: Value, j: &str) -> String {
    while let Op::Alias(a) = p.nodes[v.node].op {
        v = a;
    }
    match p.nodes[v.node].op {
        Op::Input(i) => format!("x[{}+({j})]", p.inputs[i].start),
        Op::Data(_) | Op::Constant(_) => format!("data{}[{j}]", v.node),
        _ => format!("primal[{}+({j})]", offsets[v.node]),
    }
}
fn map_source(m: &Map) -> String {
    match m {
        Map::Identity => "j".into(),
        Map::Scalar => "0".into(),
        Map::Gather(v) => format!("({{let map:[usize;{}]={v:?};map[j]}})", v.len()),
    }
}
fn child_buffers(s: &mut String, dimension: usize, output: usize) {
    writeln!(s,"{{let(args,child)=extra.split_at_mut({dimension});let(scores,child)=child.split_at_mut({dimension});let(seeds,child)=child.split_at_mut({output});let(result,child)=child.split_at_mut({output});").unwrap();
}
fn pack_values(
    s: &mut String,
    to: &str,
    mut offset: usize,
    args: &[Value],
    p: &Plan,
    slots: &[usize],
) {
    for &v in args {
        let n = p.nodes[v.node].shape.elements;
        writeln!(
            s,
            "for j in 0..{n} {{{to}[{offset}+j]={};}}",
            value_source(p, slots, v, "j")
        )
        .unwrap();
        offset += n;
    }
}
fn pack_row(s: &mut String, scan: &Scan, p: &Plan, slots: &[usize], carry: usize) {
    writeln!(
        s,
        "for j in 0..{} {{args[{carry}+j]={};}}",
        scan.row,
        value_source(p, slots, scan.xs, &format!("step*{}+j", scan.row))
    )
    .unwrap();
}
fn accumulate(s: &mut String, args: &[Value], p: &Plan, mut offset: usize, from: &str) {
    for v in args {
        let n = &p.nodes[v.node];
        writeln!(
            s,
            "for j in 0..{} {{a[{}+j]+={from}[{offset}+j];}}",
            n.shape.elements, n.start
        )
        .unwrap();
        offset += n.shape.elements;
    }
}
