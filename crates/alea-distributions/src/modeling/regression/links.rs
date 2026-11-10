//! Native link-regression descriptors, reusing the original fused kernels/laws.
use super::*;
use crate::scalar::{BernoulliLogit, PoissonLog};
use alea_core::model::assembly::{NativeRules, OperationDescriptor, SourceBody, SourceRule};

/// Execution-local scratch for native link-regression primitives.
pub struct LinkRegressionWorkspace {
    scores: Scores,
}
/// Fixed-design Bernoulli-logit batch, with original bool observations/offsets.
pub struct BernoulliLogitRegression {
    data: RegressionData<bool>,
    pub(super) design: kernels::Design,
}
/// Fixed-design Poisson-log batch, with exact integer counts and log exposures.
/// Eligible batches with identical design rows and offsets use count sufficient
/// statistics; numerical-limit states retain the original fused row evaluation.
/// Row reporting and generated source always retain the original observations.
pub struct PoissonLogRegression {
    data: RegressionData<u64>,
    pub(super) design: kernels::Design,
    normalizers: Vec<f64>,
    statistics: Option<super::poisson_statistics::PoissonStatistics>,
}
impl BernoulliLogitRegression {
    /// Prepares the existing native batch kernel without adding priors/intercepts.
    /// # Errors
    /// Rejects unrepresentable packed storage. Graph registration separately
    /// checks coefficient count and ownership.
    pub fn new(data: RegressionData<bool>) -> Result<Self, E> {
        let design = kernels::Design::new(&data)?;
        Ok(Self { data, design })
    }
    /// Original typed observations, design and offsets.
    pub fn data(&self) -> &RegressionData<bool> {
        &self.data
    }
    /// Native CPU plan, not the separately generated-source execution backend.
    pub fn execution_plan(&self) -> &RegressionPlan {
        &self.design.plan
    }
    /// Selects dispatch without repacking data or changing the numerical policy.
    #[must_use]
    pub fn with_execution(mut self, execution: RegressionExecution) -> Self {
        self.design.set_execution(execution);
        self
    }
    /// Selects opt-in conventional f64 or default compensated accumulation.
    /// Standard arithmetic retains validation but can lose accuracy under
    /// cancellation. Source AD export is unavailable for the Standard policy.
    #[must_use]
    pub fn with_arithmetic(mut self, arithmetic: RegressionArithmetic) -> Self {
        self.design.plan.arithmetic = arithmetic;
        self
    }
    /// Returns an immutable same-shape replacement descriptor.
    /// # Errors
    /// Rejects a changed row/column count or preparation failure.
    pub fn with_data(&self, data: RegressionData<bool>) -> Result<Self, E> {
        if data.rows() != self.data.rows() || data.columns() != self.data.columns() {
            return Err(E::Shape);
        }
        Ok(Self::new(data)?
            .with_execution(self.execution_plan().execution())
            .with_arithmetic(self.execution_plan().arithmetic()))
    }
}
impl PoissonLogRegression {
    /// Prepares row kernels, log-factorial constants and eligible sufficient statistics.
    /// # Errors
    /// Rejects counts above 2^53 or packed-storage overflow. Graph registration
    /// separately checks coefficient count and ownership.
    pub fn new(data: RegressionData<u64>) -> Result<Self, E> {
        let mut normalizers = Vec::with_capacity(data.rows());
        for (index, &y) in data.observations.iter().enumerate() {
            if y > 1 << 53 {
                return Err(error(RegressionDataError::CountPrecision { index }));
            }
            normalizers.push(statrs::function::gamma::ln_gamma(y as f64 + 1.));
        }
        let design = kernels::Design::new(&data)?;
        let statistics = super::poisson_statistics::PoissonStatistics::new(&data, &normalizers);
        Ok(Self {
            data,
            design,
            normalizers,
            statistics,
        })
    }
    /// Original exact counts, design and log-predictor offsets.
    pub fn data(&self) -> &RegressionData<u64> {
        &self.data
    }
    /// Native CPU plan, not the separately generated-source execution backend.
    pub fn execution_plan(&self) -> &RegressionPlan {
        &self.design.plan
    }
    /// Selects dispatch without repacking data or changing the numerical policy.
    #[must_use]
    pub fn with_execution(mut self, execution: RegressionExecution) -> Self {
        self.design.set_execution(execution);
        self
    }
    /// Selects row-kernel accumulation. Eligible common-predictor sufficient
    /// statistics remain unchanged. Standard arithmetic is opt-in and does not
    /// support source AD export; all validation remains enabled.
    #[must_use]
    pub fn with_arithmetic(mut self, arithmetic: RegressionArithmetic) -> Self {
        self.design.plan.arithmetic = arithmetic;
        self
    }
    /// Returns an immutable same-shape replacement descriptor.
    /// # Errors
    /// Rejects changed shape, count precision or preparation failures.
    pub fn with_data(&self, data: RegressionData<u64>) -> Result<Self, E> {
        if data.rows() != self.data.rows() || data.columns() != self.data.columns() {
            return Err(E::Shape);
        }
        Ok(Self::new(data)?
            .with_execution(self.execution_plan().execution())
            .with_arithmetic(self.execution_plan().arithmetic()))
    }
}
fn signature(
    name: &'static str,
    rows: usize,
    columns: usize,
    support: Support,
) -> PrimitiveSignature {
    PrimitiveSignature {
        name,
        arguments: vec![PrimitiveArgument {
            name: "coefficients",
            shape: ArgumentShape::Vector(columns),
        }],
        probability: Some(ProbabilitySemantics {
            event: EventShape::Scalar,
            batch: vec![rows],
            support,
            measure: ReferenceMeasure::Counting,
        }),
    }
}
impl BernoulliLogitRegression {
    /// Native kernel batch workspace.
    pub fn batch_workspace(&self) -> LinkRegressionWorkspace {
        LinkRegressionWorkspace {
            scores: Scores::new(0),
        }
    }
    fn signature(&self) -> PrimitiveSignature {
        signature(
            "bernoulli_logit_regression",
            self.data.rows(),
            self.data.columns(),
            Support::Finite { outcomes: 2 },
        )
    }
    /// Native kernel workspace.
    pub fn workspace(&self) -> LinkRegressionWorkspace {
        LinkRegressionWorkspace {
            scores: Scores::new(self.data.columns),
        }
    }
    /// Native kernel evaluate.
    pub fn evaluate(
        &self,
        x: &[f64],
        g: &mut [f64],
        w: &mut LinkRegressionWorkspace,
    ) -> Result<f64, E> {
        if x.len() != self.data.columns || g.len() != x.len() || w.scores.sums.len() != x.len() {
            return Err(E::Shape);
        }
        self.design.plan.dispatch(LogitEval {
            data: &self.data,
            design: &self.design,
            x,
            g,
            scores: &mut w.scores,
        })
    }
    fn batch_rule(&self) -> Option<BatchRule<Self>> {
        Some(BatchRule {
            units: self.data.rows(),
            evaluate: |this, x, out, _| {
                if x.len() != this.data.columns || out.len() != this.data.rows() {
                    return Err(E::Shape);
                }
                for (i, out) in out.iter_mut().enumerate() {
                    let law =
                        BernoulliLogit::new(this.data.linear_predictor(i, x)).map_err(error)?;
                    *out = law.log_prob(&this.data.observations[i]).map_err(error)?;
                }
                Ok(())
            },
        })
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        if self.execution_plan().arithmetic() != RegressionArithmetic::Compensated {
            return None;
        }
        Some(|this| source(&this.data, "bool", "BernoulliLogit"))
    }
    fn source_gradient_rule(&self) -> Option<fn(&Self) -> String> {
        if self.execution_plan().arithmetic() != RegressionArithmetic::Compensated {
            return None;
        }
        Some(|this| {
            native_score_source(
                &this.data,
                "BernoulliLogitRegression",
                this.execution_plan().execution(),
            )
        })
    }
}
impl OperationDescriptor for BernoulliLogitRegression {
    type Workspace = LinkRegressionWorkspace;
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        rules.batch_workspace = Self::batch_workspace;
        rules.batch = self.batch_rule();
        let body = self.source_rule().map(SourceBody::Generated);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: self.source_gradient_rule(),
            workspace: 0,
        });
        rules
    }
}
impl PoissonLogRegression {
    /// Native kernel batch workspace.
    pub fn batch_workspace(&self) -> LinkRegressionWorkspace {
        LinkRegressionWorkspace {
            scores: Scores::new(0),
        }
    }
    fn signature(&self) -> PrimitiveSignature {
        signature(
            "poisson_log_regression",
            self.data.rows(),
            self.data.columns(),
            Support::Count,
        )
    }
    /// Native kernel workspace.
    pub fn workspace(&self) -> LinkRegressionWorkspace {
        LinkRegressionWorkspace {
            scores: Scores::new(self.data.columns),
        }
    }
    /// Native kernel evaluate.
    pub fn evaluate(
        &self,
        x: &[f64],
        g: &mut [f64],
        w: &mut LinkRegressionWorkspace,
    ) -> Result<f64, E> {
        if x.len() != self.data.columns || g.len() != x.len() || w.scores.sums.len() != x.len() {
            return Err(E::Shape);
        }
        if let Some(statistics) = &self.statistics {
            let candidate = self
                .design
                .plan
                .dispatch(super::poisson_statistics::Evaluate {
                    evaluation: PoissonEval {
                        data: &self.data,
                        design: &self.design,
                        normalizers: &self.normalizers,
                        x,
                        g,
                        scores: &mut w.scores,
                    },
                    statistics,
                });
            if let Ok(value) = candidate {
                return Ok(value);
            }
            // A declined candidate is not an accepted proposal. Retry the
            // authoritative rows; their original error is returned unchanged.
        }
        self.design.plan.dispatch(PoissonEval {
            data: &self.data,
            design: &self.design,
            normalizers: &self.normalizers,
            x,
            g,
            scores: &mut w.scores,
        })
    }
    fn batch_rule(&self) -> Option<BatchRule<Self>> {
        Some(BatchRule {
            units: self.data.rows(),
            evaluate: |this, x, out, _| {
                if x.len() != this.data.columns || out.len() != this.data.rows() {
                    return Err(E::Shape);
                }
                for (i, out) in out.iter_mut().enumerate() {
                    let law = PoissonLog::new(this.data.linear_predictor(i, x)).map_err(error)?;
                    *out = law.log_prob(&this.data.observations[i]).map_err(error)?;
                }
                Ok(())
            },
        })
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        if self.execution_plan().arithmetic() != RegressionArithmetic::Compensated {
            return None;
        }
        Some(|this| source(&this.data, "u64", "PoissonLog"))
    }
    fn source_gradient_rule(&self) -> Option<fn(&Self) -> String> {
        if self.execution_plan().arithmetic() != RegressionArithmetic::Compensated {
            return None;
        }
        Some(|this| {
            native_score_source(
                &this.data,
                "PoissonLogRegression",
                this.execution_plan().execution(),
            )
        })
    }
}
impl OperationDescriptor for PoissonLogRegression {
    type Workspace = LinkRegressionWorkspace;
    fn rules(&self) -> NativeRules<Self> {
        let mut rules = NativeRules::new(self.signature(), Self::workspace, Self::evaluate);
        rules.batch_workspace = Self::batch_workspace;
        rules.batch = self.batch_rule();
        let body = self.source_rule().map(SourceBody::Generated);
        rules.source = body.map(|body| SourceRule {
            body,
            gradient: self.source_gradient_rule(),
            workspace: 0,
        });
        rules
    }
}
fn source<Y: std::fmt::Debug>(data: &RegressionData<Y>, event: &str, law: &str) -> String {
    let n = data.rows();
    let p = data.columns();
    // Calls the ACTUAL existing law, not a second probability implementation.
    format!(
        "let design: [f64; {}] = {:?}; let observations: [{event}; {n}] = {:?}; let offsets: [f64; {n}] = {:?};\n\
        let mut total=::alea::math::numeric::CompensatedSum::default();\n\
        for i in 0..{n} {{ let mut eta=::alea::math::numeric::CompensatedSum::default(); eta.add(offsets[i]);\n\
          for j in 0..{p} {{ eta.add_product(design[i*{p}+j],x[j]); }}\n\
          let law=match ::alea::distributions::scalar::{law}::new(eta.total()) {{Ok(law)=>law,Err(_)=>return f64::NAN}};\n\
          let value=match ::alea::distribution::Distribution::log_prob(&law,&observations[i]) {{Ok(value)=>value,Err(_)=>return f64::NAN}};\n\
          if !value.is_finite() {{return f64::NAN;}} total.add(value);\n\
        }} total.total()",
        data.design().len(),
        data.design(),
        data.observations(),
        data.offsets()
    )
}

#[cfg(test)]
mod execution_tests {
    use super::*;

    fn compare_result<P: OperationDescriptor>(
        p: &P,
        x: &[f64],
        fresh: impl FnOnce(&mut [f64]) -> Result<f64, E>,
    ) {
        let mut a = vec![19.; x.len()];
        let mut b = a.clone();
        let rules = p.rules();
        match (
            (rules.evaluate)(p, x, &mut a, &mut (rules.workspace)(p)),
            fresh(&mut b),
        ) {
            (Ok(a_value), Ok(b_value)) => {
                assert_eq!(a_value.to_bits(), b_value.to_bits());
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
            (a, b) => panic!("tail-only dispatch changed admission: {a:?} versus {b:?}"),
        }
    }

    #[test]
    fn tail_only_selection_preserves_fresh_dispatch_bits_errors_and_scale_limits() {
        for rows in 1..kernels::TILE {
            for columns in 0..kernels::TILE {
                let normal = NormalRegression::new(
                    RegressionData::new(columns, vec![1.; rows * columns], vec![0.5; rows])
                        .unwrap(),
                )
                .unwrap();
                let logit = BernoulliLogitRegression::new(
                    RegressionData::new(columns, vec![1.; rows * columns], vec![true; rows])
                        .unwrap(),
                )
                .unwrap();
                let poisson = PoissonLogRegression::new(
                    RegressionData::new(columns, vec![1.; rows * columns], vec![2; rows]).unwrap(),
                )
                .unwrap();
                if normal.execution_plan().is_scalar_for_shape() {
                    assert_eq!(normal.execution_plan().lanes(), 1);
                }
                for value in [0., -0., 0.25, -800., 800., 1e150, f64::MAX] {
                    let x = vec![value; columns];
                    for scale in [
                        1.25,
                        1e-200,
                        f64::MIN_POSITIVE,
                        1e200,
                        f64::MAX,
                        0.,
                        -1.,
                        f64::INFINITY,
                        f64::NAN,
                    ] {
                        let mut q = x.clone();
                        q.push(scale);
                        compare_result(&normal, &q, |g| {
                            pulp::Arch::new().dispatch(NormalEval {
                                data: &normal.data,
                                design: &normal.design,
                                broadcast: normal.design.broadcast,
                                x: &q,
                                g,
                                scores: &mut Scores::new(columns),
                            })
                        });
                    }
                    compare_result(&logit, &x, |g| {
                        pulp::Arch::new().dispatch(LogitEval {
                            data: &logit.data,
                            design: &logit.design,
                            x: &x,
                            g,
                            scores: &mut Scores::new(columns),
                        })
                    });
                    compare_result(&poisson, &x, |g| {
                        pulp::Arch::new().dispatch(PoissonEval {
                            data: &poisson.data,
                            design: &poisson.design,
                            normalizers: &poisson.normalizers,
                            x: &x,
                            g,
                            scores: &mut Scores::new(columns),
                        })
                    });
                }
            }
        }
    }

    fn compare<P: OperationDescriptor>(
        primitive: &P,
        x: &[f64],
        fresh: impl FnOnce(&mut [f64]) -> Result<f64, E>,
    ) {
        let mut a = vec![0.; x.len()];
        let mut b = a.clone();
        let rules = primitive.rules();
        let cached =
            (rules.evaluate)(primitive, x, &mut a, &mut (rules.workspace)(primitive)).unwrap();
        let original = fresh(&mut b).unwrap();
        assert_eq!(cached.to_bits(), original.to_bits());
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cached_default_preserves_original_fresh_dispatch_bits_for_all_families() {
        for rows in [1, 4, 7, 1_000] {
            let design: Vec<_> = (0..rows).flat_map(|i| [1., (i % 7) as f64 / 8.]).collect();
            let mut normal = NormalRegression::new(
                RegressionData::new(2, design.clone(), vec![0.5; rows]).unwrap(),
            )
            .unwrap();
            // This test isolates CPU dispatch caching, not Gaussian compression.
            // Compression has independent row-oracle and conditioning tests.
            normal.compressed = None;
            compare(&normal, &[0.2, -0.3, 1.25], |g| {
                pulp::Arch::new().dispatch(NormalEval {
                    data: &normal.data,
                    design: &normal.design,
                    broadcast: normal.design.broadcast,
                    x: &[0.2, -0.3, 1.25],
                    g,
                    scores: &mut Scores::new(2),
                })
            });
            let logit = BernoulliLogitRegression::new(
                RegressionData::new(2, design.clone(), vec![true; rows]).unwrap(),
            )
            .unwrap();
            compare(&logit, &[0.2, -0.3], |g| {
                pulp::Arch::new().dispatch(LogitEval {
                    data: &logit.data,
                    design: &logit.design,
                    x: &[0.2, -0.3],
                    g,
                    scores: &mut Scores::new(2),
                })
            });
            let poisson =
                PoissonLogRegression::new(RegressionData::new(2, design, vec![2; rows]).unwrap())
                    .unwrap();
            compare(&poisson, &[0.2, -0.3], |g| {
                pulp::Arch::new().dispatch(PoissonEval {
                    data: &poisson.data,
                    design: &poisson.design,
                    normalizers: &poisson.normalizers,
                    x: &[0.2, -0.3],
                    g,
                    scores: &mut Scores::new(2),
                })
            });
        }
    }
}
