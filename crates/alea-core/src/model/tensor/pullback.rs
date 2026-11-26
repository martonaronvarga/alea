//! Explicit first-order source reverse with segmented scan rematerialization.
//! Forward AD over this pullback is NOT qualified on the pinned compiler.
use super::*;
use lowering::{Operation, Read};
use std::fmt::Write;
use storage::SourceResidualLayout;

pub(crate) struct PullbackSource {
    pub source: String,
    pub workspace: usize,
    residuals: usize,
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

// Balance boundary carries against segment inputs AND materialized body primals.
// Widen before multiplication: T*C may overflow usize even when the selected
// checkpoint layout is addressable. This is a storage heuristic, not Revolve.
fn replay_interval(steps: usize, carry: usize, primal: usize) -> usize {
    debug_assert!(steps > 0);
    let per_step = (carry as u128 + primal as u128).max(1);
    let square = steps as u128 * carry as u128 / per_step;
    usize::try_from(square.isqrt()).map_or(steps, |k| k.clamp(1, steps))
}
impl BoundProgram {
    pub(crate) fn requires_explicit_source(&self) -> bool {
        self.plan.facts.explicit_source
    }
    pub(crate) fn has_source_checkpoints(&self) -> bool {
        self.plan.facts.source_checkpoints
    }
    pub(super) fn activity(&self, inputs: &[bool]) -> Vec<bool> {
        self.plan.facts.activity(inputs)
    }
    pub(crate) fn pullback_source(&self, name: &str) -> Result<PullbackSource, E> {
        self.pullback_with_activity(name, &vec![true; self.plan.inputs.len()])
    }
    fn pullback_with_activity(&self, name: &str, inputs: &[bool]) -> Result<PullbackSource, E> {
        let p = &self.plan;
        if let Some(node) = registration::missing_source(p) {
            return Err(E::MissingPrimal { node });
        }
        let active = self.activity(inputs);
        let (adjoints, adjoint_slots) = storage::adjoint_layout(p, &active)?;
        let layout = storage::source_residuals(p, &active)?;
        let residuals = layout.slots;
        let mut source = self.function_source(&format!("{name}_primal"));
        let mut reverse = String::new();
        let mut extra = 0;
        let value = |v: Value, j: &str| value_source(p, &layout, v, j);
        for call in p.facts.lowered.calls().rev() {
            let i = call.origin().node;
            if !active[i] {
                continue;
            }
            let base = adjoints[i];
            let len = call.elements();
            match call.operation(&p.nodes) {
                Operation::NativeOnly => return Err(E::MissingPrimal { node: i }),
                Operation::Registered(block, reads) => {
                    let helper = format!("{name}_registered{i}");
                    source.push_str(&block.source_definition(&helper));
                    let slots = block.source_score_slots();
                    extra = extra.max(slots);
                    let args = reads
                        .iter()
                        .map(|read| format!("{},", value_slice_source(p, &layout, read.value)))
                        .collect::<String>();
                    writeln!(reverse,"{{let scores=&mut extra[..{slots}];scores.fill(f64::NAN);if !{helper}(&[{args}],&mut[],Some(&a[{base}..{}]),scores) || scores.iter().any(|v|!v.is_finite()) {{*valid=false;return;}}",base+len).unwrap();
                    accumulate(&mut reverse, reads, &active, &adjoints, 0, "scores");
                    reverse.push_str("}\n");
                }
                Operation::Input(j) => {
                    let input = &p.inputs[j];
                    writeln!(
                        reverse,
                        "for j in 0..{len} {{ g[{}+j]=a[{base}+j]; }}",
                        input.start
                    )
                    .unwrap();
                }
                Operation::Data(_) | Operation::Constant(_) | Operation::Alias(_) => {}
                Operation::Unary(op, v) => {
                    let x = value(v, "j");
                    let y = value(call.origin(), "j");
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
                    writeln!(reverse,"for j in 0..{len} {{ let s=a[{base}+j]; let d={derivative}; if s!=0.0 || d.is_finite() {{ a[{}+j]+=s*d; }} }}",adjoints[v.node]).unwrap();
                }
                Operation::Binary(kind, (a, left), (b, right)) => {
                    let l = left.index_source();
                    let r = right.index_source();
                    let x = value(a, &l);
                    let y = value(b, &r);
                    let (dl, dr) = match kind {
                        BinaryOp::Add => ("1.0_f64".into(), "1.0_f64".into()),
                        BinaryOp::Subtract => ("1.0_f64".into(), "-1.0_f64".into()),
                        BinaryOp::Multiply => (y.clone(), x.clone()),
                        BinaryOp::Divide => (format!("1.0/({y})"), format!("-(({x})/({y}))/({y})")),
                    };
                    writeln!(reverse, "for j in 0..{len} {{ let s=a[{base}+j];").unwrap();
                    if active[a.node] {
                        writeln!(
                            reverse,
                            "{{let d={dl};if s!=0.0 || d.is_finite() {{a[{}+{l}]+=s*d;}}}}",
                            adjoints[a.node]
                        )
                        .unwrap();
                    }
                    if active[b.node] {
                        writeln!(
                            reverse,
                            "{{let d={dr};if s!=0.0 || d.is_finite() {{a[{}+{r}]+=s*d;}}}}",
                            adjoints[b.node]
                        )
                        .unwrap();
                    }
                    reverse.push_str("}\n");
                }
                Operation::Sum(read) => writeln!(
                    reverse,
                    "for j in 0..{} {{ a[{}+j]+=a[{base}]; }}",
                    read.elements(), adjoints[read.value.node]
                )
                .unwrap(),
                Operation::Gather(v, indices) => writeln!(
                    reverse,
                    "let indices:[usize;{len}]={indices:?};for (j,k) in indices.into_iter().enumerate() {{a[{}+k]+=a[{base}+j];}}",
                    adjoints[v.node]
                )
                .unwrap(),
                Operation::Matrix(l, r, calls) => {
                    let kernel = calls.kernel;
                    let (m, k, n) = calls.primal.dimensions;
                    if len == 0 || k == 0 {
                        continue;
                    }
                    let accelerated = kernel.is_accelerated()
                        && (!active[l.node] || !active[r.node] || adjoints[l.node] != adjoints[r.node]);
                    if accelerated {
                        extra = extra.max(len);
                        let left = value_slice_source(p, &layout, l);
                        let right = value_slice_source(p, &layout, r);
                        writeln!(reverse, "if a[{base}..{}].iter().all(|s|*s!=0.0) {{let seeds=&mut extra[..{len}];seeds.copy_from_slice(&a[{base}..{}]);",base+len,base+len).unwrap();
                        for (enabled, first, second, destination, count, call) in [
                            (active[l.node], "seeds", right.as_str(), adjoints[l.node], m*k, calls.left_adjoint),
                            (active[r.node], left.as_str(), "seeds", adjoints[r.node], k*n, calls.right_adjoint),
                        ] {
                            if enabled {
                                let lowering::MatrixCall {dimensions, transpose, add} = call;
                                writeln!(reverse,"if !::alea::math::gemm::try_gemm(::alea::math::gemm::ResolvedMatrix::{},{first},{second},&mut a[{destination}..{}],{dimensions:?},{transpose:?},{add}) {{*valid=false;return;}}",kernel.source_name(),destination+count).unwrap();
                            }
                        }
                        reverse.push_str("} else {\n");
                    }
                    let skip_zero=if kernel.is_accelerated() {"if s==0.0 {continue;}"}else{""};
                    writeln!(reverse,"for row in 0..{m} {{for col in 0..{n} {{let s=a[{base}+row*{n}+col]; {skip_zero} for j in 0..{k} {{").unwrap();
                    if active[l.node] {
                        writeln!(
                            reverse,
                            "a[{}+row*{k}+j]+=s*({});",
                            adjoints[l.node],
                            value(r, &format!("j*{n}+col"))
                        )
                        .unwrap();
                    }
                    if active[r.node] {
                        writeln!(
                            reverse,
                            "a[{}+j*{n}+col]+=s*({});",
                            adjoints[r.node],
                            value(l, &format!("row*{k}+j"))
                        )
                        .unwrap();
                    }
                    reverse.push_str("}}}\n");
                    if accelerated { reverse.push_str("}\n"); }
                }
                Operation::Cond(c, [lhs, rhs], arguments) => {
                    let args: Vec<_> = arguments.iter().map(|read| active[read.value.node]).collect();
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
                    pack_values(&mut reverse, "args", 0, arguments, p, &layout);
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
                    writeln!(reverse,"if {} {cmp} {} {{ {name}_yes{i}(args,seeds,result,scores,child,valid); }} else {{ {name}_no{i}(args,seeds,result,scores,child,valid); }} if !*valid {{ return; }}",value(lhs,"0"),value(rhs,"0")).unwrap();
                    accumulate(&mut reverse, arguments, &active, &adjoints, 0, "scores");
                    reverse.push_str("}\n");
                }
                Operation::Scan(s, [initial, _], _) if s.steps == 0 => writeln!(
                    reverse,
                    "for j in 0..{len} {{a[{}+j]+=a[{base}+j];}}",
                    adjoints[initial.node]
                )
                .unwrap(),
                Operation::Scan(s, [initial, xs], captures) => {
                    let first: Vec<_> = [initial, xs]
                        .into_iter()
                        .chain(captures.iter().map(|read| read.value))
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
                    let primal = s.body.plan.source_slots;
                    let mut saved=body.residuals;
                    let mut work = body.workspace - primal;
                    source.push_str(&body.source);
                    if first != steady {
                        let other = s.body.pullback_with_activity(&later, &steady)?;
                        work = work.max(other.workspace - primal);
                        saved=saved.max(other.residuals);
                        source.push_str(&other.source);
                    }
                    let interval = replay_interval(s.steps, len, saved);
                    let checkpoints = mul(s.steps.div_ceil(interval), len)?;
                    let states = mul(interval, len)?;
                    let primals = mul(interval, saved)?;
                    extra = extra.max(add(
                        add(mul(s.body.dimension(), 2)?, mul(len, 2)?)?,
                        add(work.max(primal), add(checkpoints, add(states, primals)?)?)?,
                    )?);
                    child_buffers(&mut reverse, s.body.dimension(), len);
                    writeln!(reverse,"let (checkpoints,child)=child.split_at_mut({checkpoints}); let (states,child)=child.split_at_mut({states});let (primals,child)=child.split_at_mut({primals});").unwrap();
                    writeln!(reverse, "for j in 0..{len} {{args[0+j]={};}}", value(initial, "j")).unwrap();
                    pack_values(&mut reverse, "args", len + s.row, captures, p, &layout);
                    writeln!(reverse,"for step in 0..{} {{if step%{interval}==0 {{let at=step/{interval}*{len}; checkpoints[at..at+{len}].copy_from_slice(&args[..{len}]);}}",s.steps).unwrap();
                    pack_row(&mut reverse, xs, s.row, p, &layout, len);
                    writeln!(reverse,"{one}_primal(args,result,child,valid); if !*valid {{return;}} args[..{len}].copy_from_slice(result); }}").unwrap();
                    writeln!(reverse,"seeds.copy_from_slice(&a[{base}..{}]);let mut end={}_usize;while end!=0 {{let start=(end-1)/{interval}*{interval};let at=start/{interval}*{len};args[..{len}].copy_from_slice(&checkpoints[at..at+{len}]);for step in start..end {{ let at=(step-start)*{len}; states[at..at+{len}].copy_from_slice(&args[..{len}]);",base+len,s.steps).unwrap();
                    pack_row(&mut reverse, xs, s.row, p, &layout, len);
                    writeln!(reverse,"let at=(step-start)*{saved};let frame=&mut child[..{primal}];{{let(result,work)=frame.split_at_mut({len});{one}_primal(args,result,work,valid);if !*valid{{return;}}args[..{len}].copy_from_slice(result);}}let saved=&mut primals[at..at+{saved}];if step==0 {{{one}_pack(frame,saved);}}else{{{later}_pack(frame,saved);}}}} for step in (start..end).rev() {{let at=(step-start)*{len};args[..{len}].copy_from_slice(&states[at..at+{len}]);").unwrap();
                    pack_row(&mut reverse, xs, s.row, p, &layout, len);
                    writeln!(reverse,"let at=(step-start)*{saved};let frame=&primals[at..at+{saved}];if step==0 {{{one}_prepared(args,seeds,result,scores,frame,child,valid);}} else {{{later}_prepared(args,seeds,result,scores,frame,child,valid);}}if !*valid{{return;}}seeds.copy_from_slice(&scores[..{len}]);").unwrap();
                    if active[xs.node] {
                        writeln!(
                            reverse,
                            "for j in 0..{} {{a[{}+step*{}+j]+=scores[{len}+j];}}",
                            s.row, adjoints[xs.node], s.row
                        )
                        .unwrap();
                    }
                    accumulate(&mut reverse, captures, &active, &adjoints, len + s.row, "scores");
                    reverse.push_str("}end=start;}");
                    if active[initial.node] {
                        writeln!(
                            reverse,
                            "for j in 0..{len} {{a[{}+j]+=seeds[j];}}",
                            adjoints[initial.node]
                        )
                        .unwrap();
                    }
                    reverse.push_str("}\n");
                }
            }
        }
        let workspace = add(add(p.source_slots, adjoint_slots)?, extra)?;
        let output = self.output_shape().elements;
        let mut compact = String::new();
        let mut in_place = String::new();
        for &(from, to, len) in &layout.copies {
            // Full forward slots start after the same output prefix and are
            // allocated in node order. Filtering preserves to<=from, so these
            // ascending in-place moves cannot clobber later source ranges.
            debug_assert!(to <= from);
            writeln!(
                compact,
                "saved[{to}..{}].copy_from_slice(&primal[{from}..{}]);",
                to + len,
                from + len
            )
            .unwrap();
            if to != from {
                writeln!(in_place, "primal.copy_within({from}..{},{to});", from + len).unwrap();
            }
        }
        writeln!(source,"fn {name}_pack(primal:&[f64],saved:&mut[f64]) {{saved[..{output}].copy_from_slice(&primal[..{output}]);{compact}}}").unwrap();
        writeln!(source,"#[allow(clippy::too_many_arguments,clippy::needless_range_loop)]\nfn {name}(x:&[f64],seed:&[f64],out:&mut[f64],g:&mut[f64],scratch:&mut[f64],valid:&mut bool) {{g.fill(0.0);let (primal,rest)=scratch.split_at_mut({});{{let (result,work)=primal.split_at_mut({output});{name}_primal(x,result,work,valid);}}if !*valid {{return;}}{in_place}{name}_prepared(x,seed,out,g,&primal[..{residuals}],rest,valid);}}\n#[allow(clippy::too_many_arguments,clippy::needless_range_loop)]\nfn {name}_prepared(x:&[f64],seed:&[f64],out:&mut[f64],g:&mut[f64],primal:&[f64],scratch:&mut[f64],valid:&mut bool) {{let _=x;let (a,extra)=scratch.split_at_mut({});a.fill(0.0);g.fill(0.0);if !*valid || seed.iter().any(|s|!s.is_finite()) {{*valid=false;return;}}",p.source_slots,adjoint_slots).unwrap();
        for call in p.facts.lowered.calls() {
            let i = call.origin().node;
            match call.operation(&p.nodes) {
                Operation::Data(j) => writeln!(
                    source,
                    "let data{i}:&[f64]=&{:?};let _=data{i};",
                    self.data[j]
                )
                .unwrap(),
                Operation::Constant(data) => {
                    writeln!(source, "let data{i}:&[f64]=&{data:?};let _=data{i};").unwrap()
                }
                _ => {}
            }
        }
        if active[p.output.node] {
            let start = adjoints[p.output.node];
            writeln!(
                source,
                "a[{start}..{}].copy_from_slice(seed);",
                start + output
            )
            .unwrap();
        }
        // Separate checks preserve admission/order without a Chain iterator's
        // mixed pointer/integer state. The pinned compiler fails type analysis
        // when forward-differentiating that state in this rematerialized VJP.
        writeln!(source,"{reverse}\nif g.iter().any(|v|!v.is_finite()) || a.iter().any(|v|!v.is_finite()) {{*valid=false;return;}}out.copy_from_slice(&primal[..{output}]);let _=extra;}}").unwrap();
        Ok(PullbackSource {
            source,
            workspace,
            residuals,
        })
    }
}

fn value_slice_source(p: &Plan, layout: &SourceResidualLayout, v: Value) -> String {
    let owner = layout.owners[v.node];
    let call = p.facts.lowered.call(owner).expect("checked residual owner");
    let len = call.elements();
    match call.operation(&p.nodes) {
        Operation::Input(i) => format!("&x[{}..{}]", p.inputs[i].start, p.inputs[i].start + len),
        Operation::Data(_) | Operation::Constant(_) => format!("data{owner}"),
        _ => format!(
            "&primal[{}..{}]",
            layout.offsets[owner],
            layout.offsets[owner] + len
        ),
    }
}

fn value_source(p: &Plan, layout: &SourceResidualLayout, v: Value, j: &str) -> String {
    let owner = layout.owners[v.node];
    let call = p.facts.lowered.call(owner).expect("checked residual owner");
    match call.operation(&p.nodes) {
        Operation::Input(i) => format!("x[{}+({j})]", p.inputs[i].start),
        Operation::Data(_) | Operation::Constant(_) => format!("data{owner}[{j}]"),
        _ => format!("primal[{}+({j})]", layout.offsets[owner]),
    }
}

fn child_buffers(s: &mut String, dimension: usize, output: usize) {
    writeln!(s,"{{let(args,child)=extra.split_at_mut({dimension});let(scores,child)=child.split_at_mut({dimension});let(seeds,child)=child.split_at_mut({output});let(result,child)=child.split_at_mut({output});").unwrap();
}
fn pack_values(
    s: &mut String,
    to: &str,
    mut offset: usize,
    args: &[Read],
    p: &Plan,
    layout: &SourceResidualLayout,
) {
    for read in args {
        let n = read.elements();
        writeln!(
            s,
            "for j in 0..{n} {{{to}[{offset}+j]={};}}",
            value_source(p, layout, read.value, "j")
        )
        .unwrap();
        offset += n;
    }
}
fn pack_row(
    s: &mut String,
    xs: Value,
    row: usize,
    p: &Plan,
    layout: &SourceResidualLayout,
    carry: usize,
) {
    writeln!(
        s,
        "for j in 0..{} {{args[{carry}+j]={};}}",
        row,
        value_source(p, layout, xs, &format!("step*{row}+j"))
    )
    .unwrap();
}
fn accumulate(
    s: &mut String,
    args: &[Read],
    active: &[bool],
    adjoints: &[usize],
    mut offset: usize,
    from: &str,
) {
    for read in args {
        let v = read.value;
        if active[v.node] {
            writeln!(
                s,
                "for j in 0..{} {{a[{}+j]+={from}[{offset}+j];}}",
                read.elements(),
                adjoints[v.node]
            )
            .unwrap();
        }
        offset += read.elements();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::single_element_loop)] // Optional providers extend this matrix.
    fn matrix_provider_pullbacks_do_not_substitute_compensated_source() {
        for kernel in [
            MatrixRequest::default(),
            #[cfg(feature = "faer")]
            MatrixRequest::standard(MatrixProvider::Faer),
            #[cfg(feature = "openblas")]
            MatrixRequest::standard(MatrixProvider::Blas),
        ] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::new(vec![1, 1]).unwrap()).unwrap();
            let product = b.matmul_with_request(x, x, kernel).unwrap();
            let p = b.compile(product).unwrap().bind(&[]).unwrap();
            let source = p.pullback_source("matrix").unwrap().source;
            assert_eq!(
                source.contains("try_gemm"),
                kernel != MatrixRequest::default()
            );
        }
    }

    #[cfg(feature = "openblas")]
    #[test]
    fn lp64_source_overflow_rejects_before_emission_or_workspace_allocation() {
        let mut b = RegionBuilder::new();
        let a = b
            .input("a", Shape::new(vec![i32::MAX as usize + 1, 1]).unwrap())
            .unwrap();
        let r = b.input("b", Shape::new(vec![1, 1]).unwrap()).unwrap();
        let product = b
            .matmul_with_request(a, r, MatrixRequest::standard(MatrixProvider::Blas))
            .unwrap();
        let p = b.compile(product).unwrap().bind(&[]).unwrap();
        assert!(matches!(p.rust_function(), Err(E::MissingPrimal { .. })));
        assert!(matches!(
            p.pullback_source("overflow"),
            Err(E::MissingPrimal { .. })
        ));
        assert!(p.check_workspace_limit(0).is_err());
    }

    #[test]
    fn source_residuals_retain_only_values_read_by_the_pullback() {
        let n = 17;
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![n]).unwrap()).unwrap();
        let mut out = x;
        for _ in 0..32 {
            out = b.binary(BinaryOp::Add, out, x).unwrap();
        }
        let binding = b.compile(out).unwrap().bind(&[]).unwrap();
        let source = binding.pullback_source("additive").unwrap();
        // Additions read seeds, not primals. The returned output alone remains.
        assert_eq!(source.residuals, n);
        assert!(binding.plan.source_slots > 32 * n);

        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![n]).unwrap()).unwrap();
        let exponential = b.unary(UnaryOp::Exp, x).unwrap();
        let alias = b
            .reshape(exponential, Shape::new(vec![n]).unwrap())
            .unwrap();
        let sine = b.unary(UnaryOp::Sin, alias).unwrap();
        let out = b.binary(BinaryOp::Add, sine, x).unwrap();
        let binding = b.compile(out).unwrap().bind(&[]).unwrap();
        let source = binding.pullback_source("nonlinear").unwrap();
        // Exp needs its result; Sin needs that same result through the alias.
        // Neither the Sin result nor an additional alias copy is retained.
        assert_eq!(source.residuals, 2 * n);
    }

    #[test]
    fn replay_segments_account_for_body_primals_without_integer_overflow() {
        assert_eq!(replay_interval(257, 1, 3), 8);
        assert_eq!(replay_interval(257, 1, 1024), 1);
        assert_eq!(replay_interval(257, 0, 0), 1);
        assert_eq!(replay_interval(1, 1, 1), 1);
        for steps in [1, 3, 17, 257, usize::MAX] {
            for carry in [0, 1, 1024, usize::MAX] {
                for primal in [0, 1, 1024, usize::MAX] {
                    assert!((1..=steps).contains(&replay_interval(steps, carry, primal)));
                }
            }
        }
    }

    #[test]
    fn source_adjoints_exclude_immutable_ranges_and_share_aliases() {
        for len in [0, 1, 257, 10_000] {
            let mut b = RegionBuilder::new();
            let x = b.input("x", Shape::scalar()).unwrap();
            let alias = b.reshape(x, Shape::new(vec![1]).unwrap()).unwrap();
            let data = b.data("data", Shape::new(vec![len]).unwrap()).unwrap();
            let sum = b.sum(data).unwrap();
            let out = b.binary(BinaryOp::Multiply, alias, sum).unwrap();
            let program = b.compile(out).unwrap();
            let values = vec![1.; len];
            let binding = program.bind(&[("data", &values)]).unwrap();
            let active = binding.activity(&[true]);
            let (offsets, slots) = storage::adjoint_layout(&binding.plan, &active).unwrap();
            assert_eq!(slots, 2); // one input plus one product; alias shares input
            assert_eq!(offsets[x.node], offsets[alias.node]);
            assert!(!active[data.node]);
            let source = binding.pullback_source("test").unwrap();
            // No control-flow scratch: primal storage plus exactly two scores.
            assert_eq!(source.workspace, binding.plan.source_slots + 2);
            println!(
                "immutable elements={len}: dense scratch={} doubles; compact scratch={} doubles",
                binding.plan.source_slots + binding.plan.slots,
                source.workspace
            );

            let inactive = binding
                .pullback_with_activity("inactive", &[false])
                .unwrap();
            assert_eq!(inactive.workspace, binding.plan.source_slots);
            assert!(!inactive.source.contains("copy_from_slice(seed)"));
        }
    }
}
