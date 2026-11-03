//! Explicit graph reverse composition around rematerialized tensor primitives.
use super::*;
use crate::model::tensor::TensorFactor;

fn extent(a: usize, b: usize) -> Result<usize, AssemblyError> {
    a.checked_add(b)
        .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
        .ok_or(AssemblyError::Shape)
}

impl CompiledModel {
    /// Emits an explicit source pullback when an active tensor uses checkpointing.
    ///
    /// Compiler integration hook: returns helper functions and a scratch element
    /// count, or `None` for the ordinary compiler reverse path. `density` selects
    /// whether to include chart Jacobians. The source contains `__WORKSPACE__`,
    /// which the consumer replaces with its final fixed arena extent. Tensor
    /// scans use segmented replay; opaque source factors still use compiler AD
    /// and their compiler-owned tapes are not bounded by this count.
    /// # Errors
    /// Rejects missing source rules and overflowing scratch extents.
    pub fn rust_checkpoint_gradient(
        &self,
        density: bool,
    ) -> Result<Option<(String, usize)>, AssemblyError> {
        let tensor = |i| {
            self.primitives
                .binary_search_by_key(&i, |(node, _)| *node)
                .ok()
                .and_then(|j| self.primitives[j].1.descriptor::<TensorFactor>())
        };
        let retained = |i: usize| self.active_nodes.as_ref().is_none_or(|a| a[i]);
        if !(0..self.nodes.len())
            .any(|i| retained(i) && tensor(i).is_some_and(|t| t.binding().has_source_checkpoints()))
        {
            return Ok(None);
        }
        let c = self.layout.constrained_dimension();
        let mut source = String::new();
        let mut chart = String::from("#[allow(unused_mut)] let mut jacobian=0.0_f64;\n");
        for b in self.layout.blocks() {
            chart.push_str(&b.transform().rust_primal(
                b.unconstrained_range().start,
                b.constrained_range().start,
                density,
            ));
        }
        writeln!(
            source,
            "fn __coordinates(q:&[f64;DIMENSION],x:&mut[f64])->f64 {{{chart}"
        )
        .unwrap();
        for j in 0..c {
            writeln!(source, "x[{j}]=x{j};").unwrap();
        }
        source.push_str("jacobian}\n");
        let mut forward = String::new();
        let mut reverse = String::new();
        let mut temporary = 0;
        for (i, op) in self.nodes.iter().enumerate() {
            if !retained(i) {
                continue;
            }
            let expression = match op {
                Op::Parameter(j) => format!("x[{j}]"),
                Op::Constant(v) => format!("{v:?}_f64"),
                Op::Add(a, b) => format!("n[{a}]+n[{b}]"),
                Op::Multiply(a, b) => format!("n[{a}]*n[{b}]"),
                Op::Exp(a) => format!("n[{a}].exp()"),
                Op::Log(a) => format!("n[{a}].ln()"),
                Op::Factor { inputs, .. } => {
                    let primal = self
                        .primal_source(i)
                        .ok_or(AssemblyError::MissingPrimal { node: i })?;
                    let size = self
                        .primitives
                        .binary_search_by_key(&i, |(j, _)| *j)
                        .ok()
                        .map_or(0, |j| self.primitives[j].1.source_workspace);
                    let d = inputs.len();
                    let pullback = tensor(i)
                        .map(|t| t.binding().pullback_source(&format!("__tensor{i}")))
                        .transpose()
                        .map_err(|_| AssemblyError::Shape)?;
                    let work = pullback.as_ref().map_or(size, |v| v.workspace.max(size));
                    temporary = temporary.max(extent(extent(d, d)?, extent(work, 1)?)?);
                    if let Some(pullback) = &pullback {
                        source.push_str(&pullback.source);
                    } else {
                        writeln!(source,"#[::std::autodiff::autodiff_reverse(__factor_reverse{i},Duplicated,Duplicated,Active)]").unwrap();
                    }
                    writeln!(source,"fn __factor{i}(x:&[f64;{d}],scratch:&mut[f64;{size}])->f64 {{let _=&scratch;{primal}}}").unwrap();
                    let pack = inputs
                        .iter()
                        .enumerate()
                        .map(|(j, v)| format!("args[{j}]=n[{v}];"))
                        .collect::<String>();
                    let eval = format!(
                        "{{let(args,rest)=temporary.split_at_mut({d});{pack}__factor{i}((&*args).try_into().expect(\"arguments\"),(&mut rest[..{size}]).try_into().expect(\"workspace\"))}}"
                    );
                    let mut r = format!(
                        "{{let(args,rest)=temporary.split_at_mut({d});let(scores,work)=rest.split_at_mut({d});{pack}scores.fill(0.0);"
                    );
                    if pullback.is_some() {
                        write!(r,"let(result,work)=work.split_at_mut(1);let mut valid=true;__tensor{i}(args,&[1.0],result,scores,work,&mut valid);if !valid {{g.fill(f64::NAN);return f64::NAN;}}").unwrap();
                    } else {
                        write!(r,"let(work,shadow)=work.split_at_mut({size});shadow[..{size}].fill(0.0);__factor_reverse{i}((&*args).try_into().expect(\"arguments\"),scores.try_into().expect(\"scores\"),work.try_into().expect(\"workspace\"),(&mut shadow[..{size}]).try_into().expect(\"shadow\"),1.0);").unwrap();
                        temporary = temporary.max(extent(extent(d, d)?, extent(size, size)?)?);
                    }
                    r.push_str("if scores.iter().any(|v|!v.is_finite()) {g.fill(f64::NAN);return f64::NAN;}");
                    for (j, v) in inputs.iter().enumerate() {
                        write!(r, "a[{v}]+=a[{i}]*scores[{j}];").unwrap();
                    }
                    r.push_str("}\n");
                    reverse.insert_str(0, &r);
                    writeln!(
                        forward,
                        "n[{i}]={eval};if !n[{i}].is_finite() {{g.fill(f64::NAN);return f64::NAN;}}"
                    )
                    .unwrap();
                    continue;
                }
            };
            writeln!(
                forward,
                "n[{i}]={expression};if !n[{i}].is_finite() {{g.fill(f64::NAN);return f64::NAN;}}"
            )
            .unwrap();
            let r = match op {
                Op::Parameter(j) => format!("gc[{j}]+=a[{i}];"),
                Op::Constant(_) => String::new(),
                Op::Add(a, b) => format!("a[{a}]+=a[{i}];a[{b}]+=a[{i}];"),
                Op::Multiply(a, b) => {
                    format!("a[{a}]+=a[{i}]*n[{b}];a[{b}]+=a[{i}]*n[{a}];")
                }
                Op::Exp(a) => format!("a[{a}]+=a[{i}]*n[{i}];"),
                Op::Log(a) => format!("a[{a}]+=a[{i}]/n[{a}];"),
                Op::Factor { .. } => unreachable!(),
            };
            reverse.insert_str(0, &r);
        }
        let nodes = self.nodes.len();
        temporary = temporary.max(self.layout.scratch_dimension());
        let workspace = extent(extent(nodes, nodes)?, extent(extent(c, c)?, temporary)?)?;
        writeln!(source,"pub(super) fn __gradient(q:&[f64;DIMENSION],g:&mut[f64;DIMENSION],work:&mut[f64;__WORKSPACE__],shadow:&mut[f64;__WORKSPACE__],seed:f64)->f64 {{work.fill(0.0);shadow.fill(0.0);g.fill(0.0);let(n,rest)=work.split_at_mut({nodes});let(a,rest)=rest.split_at_mut({nodes});let(x,rest)=rest.split_at_mut({c});let(gc,temporary)=rest.split_at_mut({c});let jacobian=__coordinates(q,x);if !jacobian.is_finite() {{g.fill(f64::NAN);return f64::NAN;}}{forward}\nlet mut total=0.0_f64;").unwrap();
        for i in self.retained_factors() {
            writeln!(source, "total+=n[{i}];a[{i}]+=1.0;").unwrap();
        }
        // Match the existing source primal's rounding order: factor sum first,
        // then exactly one layout Jacobian, not a Jacobian-seeded reduction.
        source.push_str("total+=jacobian;\n");
        writeln!(source,"{reverse}\nif gc.iter().any(|v|!v.is_finite()) || !total.is_finite() {{g.fill(f64::NAN);return f64::NAN;}}").unwrap();
        for b in self.layout.blocks() {
            let qr = b.unconstrained_range();
            let cr = b.constrained_range();
            let method = if density {
                "pullback"
            } else {
                "pullback_composition"
            };
            writeln!(source,"if {}.{method}(&q[{}..{}],&gc[{}..{}],&mut g[{}..{}],temporary).is_err() {{g.fill(f64::NAN);return f64::NAN;}}",b.transform().rust_constructor(),qr.start,qr.end,cr.start,cr.end,qr.start,qr.end).unwrap();
        }
        source.push_str("for v in &mut *g {*v*=seed;}if g.iter().any(|v|!v.is_finite()) {g.fill(f64::NAN);return f64::NAN;}total}\n");
        Ok(Some((source, workspace)))
    }
}
