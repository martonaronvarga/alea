//! Explicit graph reverse composition around rematerialized tensor primitives.
use super::*;

fn extent(a: usize, b: usize) -> Result<usize, AssemblyError> {
    a.checked_add(b)
        .filter(|&n| n <= isize::MAX as usize / size_of::<f64>())
        .ok_or(AssemblyError::Shape)
}

enum FactorRoute<'a> {
    Compiler,
    Tensor {
        program: &'a crate::model::tensor::BoundProgram,
        explicit: bool,
    },
    Trusted(&'a super::super::SourceGradientCallback),
}

impl Executable {
    /// Resolves and emits the derivative portion of a source request.
    ///
    /// Trusted primitive scores and explicit tensor pullbacks are selected
    /// per retained factor. Compiler tensor factors still admit checked
    /// local scores. A higher-order request requires the separately
    /// qualified compiler-primal route.
    /// # Errors
    /// Rejects missing primals, unsupported orders, a zero-dimensional
    /// derivative target, and overflowing generated workspace.
    pub fn source_derivatives(
        &self,
        request: SourceRequest,
    ) -> Result<DerivativeSource, AssemblyError> {
        let request = self.source_capabilities().resolve(request)?;
        if request.order == DerivativeOrder::Value {
            return Ok(DerivativeSource::Value);
        }
        if self.layout.unconstrained_dimension() == 0 {
            return Err(AssemblyError::Empty);
        }

        let mut composed = false;
        let mut has_tensor = false;
        let mut checkpointed = false;
        let mut routes = Vec::new();
        for (i, op) in self.source_nodes() {
            if !matches!(op, Op::Factor { .. }) {
                continue;
            }
            let primitive = self.source_primitive(i);
            let route = if let Some(rule) = primitive.and_then(|p| p.source_gradient.as_deref()) {
                composed = true;
                FactorRoute::Trusted(rule)
            } else if let Some(program) = primitive.and_then(|p| p.region_binding()) {
                has_tensor = true;
                let replays = program.has_source_checkpoints();
                let explicit = replays || program.requires_explicit_source();
                checkpointed |= replays;
                composed |= explicit;
                FactorRoute::Tensor { program, explicit }
            } else {
                FactorRoute::Compiler
            };
            routes.push(route);
        }

        let density = request.mode == SourceMode::Density;
        if composed {
            return Ok(DerivativeSource::Composed(self.rust_composed_gradient(
                density,
                false,
                checkpointed,
                &routes,
            )?));
        }
        let admission = if has_tensor {
            Some(
                self.rust_composed_gradient(density, true, false, &routes)?
                    .program,
            )
        } else {
            None
        };
        Ok(DerivativeSource::Compiler { admission })
    }

    fn rust_composed_gradient(
        &self,
        density: bool,
        tensor_validation: bool,
        checkpointed: bool,
        routes: &[FactorRoute<'_>],
    ) -> Result<FirstOrderSource, AssemblyError> {
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
        let mut routes = routes.iter();
        for (i, op) in self.source_nodes() {
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
                    let primitive = self.source_primitive(i);
                    let size = primitive.map_or(0, |p| p.source_workspace);
                    let d = inputs.len();
                    let route = routes.next().expect("retained factor route");
                    if let FactorRoute::Trusted(rule) = route {
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
                    let (pullback, explicit) = match route {
                        FactorRoute::Tensor { program, explicit } => (
                            Some(
                                program
                                    .pullback_source(&format!("__tensor{i}"))
                                    .map_err(|_| AssemblyError::Shape)?,
                            ),
                            tensor_validation || *explicit,
                        ),
                        FactorRoute::Compiler => (None, false),
                        FactorRoute::Trusted(_) => unreachable!("trusted route emitted above"),
                    };
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
        Ok(FirstOrderSource {
            program: SourceProgram {
                source,
                workspace: FloatWorkspace {
                    elements: workspace,
                },
            },
            checkpointed,
            stateful,
        })
    }
}
