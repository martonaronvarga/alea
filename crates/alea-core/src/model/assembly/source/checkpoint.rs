//! Explicit graph reverse composition around rematerialized tensor primitives.
use super::*;

fn extent(a: usize, b: usize) -> Result<usize, AssemblyError> {
    a.checked_add(b)
        .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
        .ok_or(AssemblyError::Shape)
}

impl Executable {
    /// Emits a first-order composition of trusted primitive score rules and
    /// tensor replay. The named receipt reports actual tensor checkpointing, not
    /// merely the presence of an analytic primitive. Higher AD uses the primal.
    /// # Errors
    /// Preserves missing-primal and checked workspace overflow errors.
    pub fn source_first_order(
        &self,
        mode: SourceMode,
    ) -> Result<Option<FirstOrderSource>, AssemblyError> {
        self.rust_composed_gradient(mode == SourceMode::Density, false)
    }

    /// Emits checked tensor score validation for a compiler-derived gradient.
    ///
    /// Compiler derivative conventions can hide a singular local tensor score
    /// (for example, the pinned sqrt rule supplies zero at zero). This reuses
    /// the existing explicit tensor pullback and scalar composition for admission
    /// only. Its returned score must not replace the requested compiler gradient.
    /// Source-only opaque leaves still require their compiler derivative; this
    /// does not validate arbitrary handwritten callbacks against their source.
    /// Returns `None` without an active tensor. Call outside differentiation and
    /// separately from value-only evaluation. Skip it when the actual first-order
    /// route already performs explicit tensor pullbacks.
    /// # Errors
    /// Preserves missing-primal and checked workspace overflow errors.
    pub fn source_score_admission(
        &self,
        mode: SourceMode,
    ) -> Result<Option<SourceProgram>, AssemblyError> {
        Ok(self
            .rust_composed_gradient(mode == SourceMode::Density, true)?
            .map(|first| first.program))
    }
    fn rust_composed_gradient(
        &self,
        density: bool,
        tensor_validation: bool,
    ) -> Result<Option<FirstOrderSource>, AssemblyError> {
        let tensor = |i| {
            self.primitives
                .binary_search_by_key(&i, |(node, _)| *node)
                .ok()
                .and_then(|j| self.primitives[j].1.region_binding())
        };
        let retained = |i: usize| self.active_nodes.as_ref().is_none_or(|a| a[i]);
        let checkpointed = (0..self.nodes.len())
            .any(|i| retained(i) && tensor(i).is_some_and(|t| t.has_source_checkpoints()));
        let native = |i| {
            self.primitives
                .binary_search_by_key(&i, |(node, _)| *node)
                .ok()
                .and_then(|j| self.primitives[j].1.source_gradient.as_ref())
        };
        if tensor_validation && !(0..self.nodes.len()).any(|i| retained(i) && tensor(i).is_some()) {
            return Ok(None);
        }
        if !tensor_validation
            && !checkpointed
            && !(0..self.nodes.len()).any(|i| retained(i) && native(i).is_some())
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
        let mut state_fields = String::new();
        let mut state_initializers = String::new();
        let mut forward = String::new();
        let mut reverse = String::new();
        let mut temporary = 0;
        let mut local_count = 0;
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
                Op::Region(_) => return Err(AssemblyError::MissingPrimal { node: i }),
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
                    if let Some(rule) = native(i) {
                        let first = local_count;
                        local_count = extent(local_count, d)?;
                        temporary = temporary.max(extent(d, size)?);
                        let rule = rule();
                        let (state_parameter, state_argument) = if let Some(state) = rule.state {
                            writeln!(state_fields, "s{i}: {},", state.rust_type).unwrap();
                            writeln!(state_initializers, "s{i}: {{ {} }},", state.initializer)
                                .unwrap();
                            (
                                format!(",state:&mut {}", state.rust_type),
                                format!(",&mut context.s{i}"),
                            )
                        } else {
                            (String::new(), String::new())
                        };
                        writeln!(source,"fn __native{i}(x:&[f64],g:&mut[f64],scratch:&mut[f64]{state_parameter})->f64 {{let _=&scratch;{}}}",rule.body).unwrap();
                        let pack = inputs
                            .iter()
                            .enumerate()
                            .map(|(j, node)| format!("args[{j}]=n[{node}];"))
                            .collect::<String>();
                        writeln!(forward,"n[{i}]={{let(args,work)=temporary.split_at_mut({d});{pack}let scores=&mut locals[{first}..{local_count}];scores.fill(f64::NAN);let value=__native{i}(args,scores,&mut work[..{size}]{state_argument});if !value.is_finite() || scores.iter().any(|s|!s.is_finite()) {{g.fill(f64::NAN);return f64::NAN;}}value}};").unwrap();
                        let mut r = String::new();
                        for (j, node) in inputs.iter().enumerate() {
                            writeln!(r, "a[{node}]+=a[{i}]*locals[{}];", first + j).unwrap();
                        }
                        reverse.insert_str(0, &r);
                        continue;
                    }
                    let explicit =
                        tensor(i).is_some_and(|t| tensor_validation || t.has_source_checkpoints());
                    let pullback = tensor(i)
                        .map(|t| t.pullback_source(&format!("__tensor{i}")))
                        .transpose()
                        .map_err(|_| AssemblyError::Shape)?;
                    let work = pullback.as_ref().map_or(size, |v| v.workspace.max(size));
                    temporary = temporary.max(extent(extent(d, d)?, extent(work, 1)?)?);
                    if let Some(pullback) = &pullback {
                        source.push_str(&pullback.source);
                    }
                    if !explicit {
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
                        write!(r,"{{let(result,work)=work.split_at_mut(1);let mut valid=true;__tensor{i}(args,&[1.0],result,scores,work,&mut valid);if !valid || scores.iter().any(|v|!v.is_finite()) {{g.fill(f64::NAN);return f64::NAN;}}}}").unwrap();
                    }
                    if !explicit {
                        // A retained tensor still needs checked local-score admission.
                        // Its explicit scores are discarded before compiler reverse.
                        r.push_str("scores.fill(0.0);");
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
                "n[{i}] = {expression};if !n[{i}].is_finite() {{g.fill(f64::NAN);return f64::NAN;}}"
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
                Op::Factor { .. } | Op::Region(_) => unreachable!(),
            };
            reverse.insert_str(0, &r);
        }
        let nodes = self.nodes.len();
        temporary = temporary.max(self.layout.scratch_dimension());
        let workspace = extent(
            extent(nodes, nodes)?,
            extent(extent(c, c)?, extent(local_count, temporary)?)?,
        )?;
        let stateful = !state_fields.is_empty();
        writeln!(source,"pub(super) struct __SourceContext {{{state_fields}}}\npub(super) fn __first_context()->__SourceContext {{__SourceContext {{{state_initializers}}}}}").unwrap();
        writeln!(source,"pub(super) fn __gradient(q:&[f64;DIMENSION],g:&mut[f64;DIMENSION],work:&mut[f64;{workspace}],shadow:&mut[f64;{workspace}],seed:f64,context:&mut __SourceContext)->f64 {{let _=&context;work.fill(0.0);shadow.fill(0.0);g.fill(0.0);let(n,rest)=work.split_at_mut({nodes});let(a,rest)=rest.split_at_mut({nodes});let(x,rest)=rest.split_at_mut({c});let(gc,rest)=rest.split_at_mut({c});let(locals,temporary)=rest.split_at_mut({local_count});let jacobian=__coordinates(q,x);if !jacobian.is_finite() {{g.fill(f64::NAN);return f64::NAN;}}{forward}\nlet mut total=0.0_f64;").unwrap();
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
        source.push_str("let _=&locals;for v in &mut *g {*v*=seed;}if g.iter().any(|v|!v.is_finite()) {g.fill(f64::NAN);return f64::NAN;}total}\n");
        Ok(Some(FirstOrderSource {
            program: SourceProgram {
                source,
                workspace: FloatWorkspace {
                    elements: workspace,
                },
            },
            checkpointed,
            stateful,
        }))
    }
}
