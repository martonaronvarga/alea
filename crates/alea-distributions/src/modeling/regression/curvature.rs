//! Explicit fixed-design link likelihood curvature, not a generic graph Hessian.
use super::*;
use alea_core::{
    capability::PotentialHessianVector,
    target::{LogDensityGradient, TargetSpace},
};
use std::cell::RefCell;

enum Family {
    Bernoulli(BernoulliLogitRegression),
    Poisson(PoissonLogRegression),
}
struct Work {
    evaluation: LinkRegressionWorkspace,
    gradient: Vec<f64>,
    product: Vec<f64>,
    scores: Scores,
}
/// A native fixed-design likelihood on unconstrained real coefficients, with
/// explicit potential HVPs. Values/scores delegate the SAME fused regression
/// primitive as ModelBuilder; no generated-source replay mismatch or finite
/// differences. No priors, coefficient transforms or uncertainty assumptions are
/// inferred. This does not confer HVPs on arbitrary composed model expressions.
pub struct GlmLikelihood {
    family: Family,
    work: RefCell<Work>,
    dimension: usize,
}
impl GlmLikelihood {
    /// Builds a Bernoulli-logit likelihood on real coefficients in column order.
    /// # Errors
    /// Preserves native data and packed-storage validation errors.
    pub fn bernoulli_logit(data: RegressionData<bool>) -> Result<Self, E> {
        let dimension = data.columns();
        let primitive = BernoulliLogitRegression::new(data)?;
        let workspace = primitive.workspace();
        Ok(Self::new(
            Family::Bernoulli(primitive),
            workspace,
            dimension,
        ))
    }
    /// Builds a Poisson-log likelihood; offsets remain log exposures.
    /// # Errors
    /// Preserves native data/count precision and packed-storage validation errors.
    pub fn poisson_log(data: RegressionData<u64>) -> Result<Self, E> {
        let dimension = data.columns();
        let primitive = PoissonLogRegression::new(data)?;
        let workspace = primitive.workspace();
        Ok(Self::new(Family::Poisson(primitive), workspace, dimension))
    }
    fn new(family: Family, evaluation: LinkRegressionWorkspace, dimension: usize) -> Self {
        Self {
            family,
            dimension,
            work: RefCell::new(Work {
                evaluation,
                gradient: vec![0.; dimension],
                product: vec![0.; dimension],
                scores: Scores::new(dimension),
            }),
        }
    }
    fn evaluate(
        &self,
        x: &[f64],
        g: &mut [f64],
        w: &mut LinkRegressionWorkspace,
    ) -> Result<f64, E> {
        match &self.family {
            Family::Bernoulli(p) => p.evaluate(x, g, w),
            Family::Poisson(p) => p.evaluate(x, g, w),
        }
    }
    fn check(&self, x: &[f64], out: &[f64]) -> Result<(), E> {
        if x.len() != self.dimension || out.len() != self.dimension {
            return Err(E::Shape);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Ok(())
    }
}
impl TargetSpace for GlmLikelihood {
    type Error = E;
    fn dimension(&self) -> usize {
        self.dimension
    }
}
impl LogDensityGradient for GlmLikelihood {
    fn is_numerical_error(&self, error: &E) -> bool {
        matches!(error, E::Numerical)
    }
    fn logp_grad(&self, x: &[f64], out: &mut [f64]) -> Result<f64, E> {
        self.check(x, out)?;
        let mut work = self.work.try_borrow_mut().map_err(|_| E::WorkspaceBusy)?;
        let Work {
            evaluation,
            gradient,
            ..
        } = &mut *work;
        let value = self.evaluate(x, gradient, evaluation)?;
        if !value.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        out.copy_from_slice(gradient);
        Ok(value)
    }
}
impl PotentialHessianVector for GlmLikelihood {
    fn potential_hvp(&self, x: &[f64], vector: &[f64], out: &mut [f64]) -> Result<(), E> {
        self.check(x, out)?;
        self.check(vector, out)?;
        let mut work = self.work.try_borrow_mut().map_err(|_| E::WorkspaceBusy)?;
        let Work {
            evaluation,
            gradient,
            product,
            scores,
        } = &mut *work;
        // Retain native primal/score admission even if the HVP happens to be zero.
        let value = self.evaluate(x, gradient, evaluation)?;
        if !value.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        match &self.family {
            Family::Bernoulli(p) => p.execution_plan().dispatch(Curvature {
                data: p.data(),
                design: &p.design,
                x,
                vector,
                scores,
                weight: |eta: f64| {
                    let e = (-eta.abs()).exp();
                    (e / (1. + e)) / (1. + e)
                },
            }),
            Family::Poisson(p) => p.execution_plan().dispatch(Curvature {
                data: p.data(),
                design: &p.design,
                x,
                vector,
                scores,
                weight: f64::exp,
            }),
        }?;
        scores.finish(CompensatedSum::default(), product)?;
        out.copy_from_slice(product);
        Ok(())
    }
}
struct Curvature<'a, Y, F> {
    data: &'a RegressionData<Y>,
    design: &'a kernels::Design,
    x: &'a [f64],
    vector: &'a [f64],
    scores: &'a mut Scores,
    weight: F,
}
impl<Y, F: Fn(f64) -> f64> pulp::WithSimd for Curvature<'_, Y, F> {
    type Output = Result<f64, E>;
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            data,
            design,
            x,
            vector,
            scores,
            weight,
        } = self;
        scores.reset();
        let full = data.rows() / kernels::TILE;
        for tile in 0..full {
            let start = tile * kernels::TILE;
            let eta =
                design.predictors(simd, tile, x, &data.offsets()[start..start + kernels::TILE]);
            let direction = design.predictors(simd, tile, vector, &[0.; kernels::TILE]);
            for r in 0..kernels::TILE {
                add_row(simd, data, start + r, eta[r], direction[r], &weight, scores)?;
            }
        }
        for i in full * kernels::TILE..data.rows() {
            let row = &data.design()[i * data.columns()..(i + 1) * data.columns()];
            let mut direction = CompensatedSum::default();
            for (&a, &b) in row.iter().zip(vector) {
                direction.add_product(a, b);
            }
            add_row(
                simd,
                data,
                i,
                data.linear_predictor(i, x),
                direction.total(),
                &weight,
                scores,
            )?;
        }
        Ok(0.)
    }
}
fn add_row<S: pulp::Simd, Y>(
    simd: S,
    data: &RegressionData<Y>,
    i: usize,
    eta: f64,
    direction: f64,
    weight: &impl Fn(f64) -> f64,
    scores: &mut Scores,
) -> Result<(), E> {
    if !eta.is_finite() || !direction.is_finite() {
        return Err(E::Numerical);
    }
    let residual = weight(eta) * direction;
    if !residual.is_finite() {
        return Err(E::Numerical);
    }
    let row = &data.design()[i * data.columns()..(i + 1) * data.columns()];
    scores.add_scaled(simd, row, residual);
    Ok(())
}
