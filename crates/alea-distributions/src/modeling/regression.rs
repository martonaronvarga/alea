//! Fused fixed-design regression likelihoods. One node per batch, not per row.
//!
//! Design rows include an intercept column only if supplied. Offsets are on the
//! linear-predictor scale: for Poisson exposure e>0 supply ln(e), not e. Captured
//! data are shared across cloned batches. Native descriptors retain explicit
//! conditional row and lazy Rust-source rules; compiler derivative support is a
//! separate capability. Each likelihood stays one joint node. CPU plans are
//! inspected through the typed descriptor, without changing the modeling calls.
#![deny(missing_docs)]
use super::{E, error};
use crate::scalar::{Bernoulli, Normal};
use alea_core::model::assembly::{Expr, ModelBuilder, Parameter, Positive};
use alea_core::{
    distribution::{Distribution, EventShape, ReferenceMeasure, Support},
    model::assembly::{
        ArgumentShape, BatchRule, Primitive, PrimitiveArgument, PrimitiveSignature,
        ProbabilitySemantics,
    },
};
use alea_math::numeric::{CompensatedSum, first_nonfinite};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
mod curvature;
mod execution;
mod kernels;
mod links;
mod map;
mod normal_compression;
mod poisson_statistics;
pub use curvature::GlmLikelihood;
pub use execution::{RegressionExecution, RegressionPlan};
pub use links::{BernoulliLogitRegression, LinkRegressionWorkspace, PoissonLogRegression};
pub use map::{LinearPredictor, ObservationMap};

/// Invalid fixed regression data or diagnostic input; no rows are silently dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RegressionDataError {
    /// At least one observation is required.
    #[error("empty regression observations")]
    Empty,
    /// Design storage or offsets do not match response dimensions.
    #[error("regression data dimensions do not agree")]
    Shape,
    /// A predictor, offset, real response or diagnostic direction is not finite.
    #[error("nonfinite regression {field} at index {index}")]
    NonFinite {
        /// Predictor, offset, observation or diagnostic direction.
        field: &'static str,
        /// Flat index in that field.
        index: usize,
    },
    /// Standardization needs an explicit column containing exactly one in every row.
    #[error("standardization requires column {column} to be a unit intercept")]
    Intercept {
        /// Requested intercept column in the original design.
        column: usize,
    },
    /// Count conversion would lose integer precision.
    #[error("Poisson count exceeds exact binary64 integers at row {index}")]
    CountPrecision {
        /// Observation index.
        index: usize,
    },
}
/// Owned row-major design and typed observations, shared cheaply when cloned.
/// Shape/predictor checks occur at construction; family-specific response checks
/// occur when attaching a factor. Rank and estimator existence are not inferred.
#[derive(Debug, Clone)]
pub struct RegressionData<Y> {
    columns: usize,
    design: Arc<[f64]>,
    observations: Arc<[Y]>,
    offsets: Arc<[f64]>,
    // Only predictor tiles are shared. Per-binding offsets/dispatch and mutable
    // score buffers must never enter this cache.
    packed: Arc<OnceLock<Arc<[[f64; kernels::TILE]]>>>,
}
impl<Y> RegressionData<Y> {
    /// Constructs an explicitly sized design with zero offsets. Zero columns
    /// permit an offset-only likelihood; empty observations are rejected.
    /// # Errors
    /// Rejects empty responses, shape overflow/mismatch and nonfinite predictors.
    pub fn new(
        columns: usize,
        design: Vec<f64>,
        observations: Vec<Y>,
    ) -> Result<Self, RegressionDataError> {
        if observations.is_empty() {
            return Err(RegressionDataError::Empty);
        }
        if columns.checked_mul(observations.len()) != Some(design.len()) {
            return Err(RegressionDataError::Shape);
        }
        finite(&design, "predictor")?;
        let offsets = vec![0.; observations.len()].into();
        Ok(Self {
            columns,
            design: design.into(),
            observations: observations.into(),
            offsets,
            packed: Arc::new(OnceLock::new()),
        })
    }
    /// Replaces responses while sharing the validated design, offsets and any
    /// prepared predictor tiles. Family-specific response checks occur when the
    /// new data are attached to a native factor.
    /// # Errors
    /// Rejects changed observation count; the old binding is unchanged.
    pub fn with_observations<Z>(
        &self,
        observations: Vec<Z>,
    ) -> Result<RegressionData<Z>, RegressionDataError> {
        if observations.len() != self.rows() {
            return Err(RegressionDataError::Shape);
        }
        Ok(RegressionData {
            columns: self.columns,
            design: Arc::clone(&self.design),
            observations: observations.into(),
            offsets: Arc::clone(&self.offsets),
            packed: Arc::clone(&self.packed),
        })
    }
    /// Replaces zero/default offsets with explicit linear-predictor offsets.
    /// # Errors
    /// Rejects a mismatched length or nonfinite offsets, including log(0) exposures.
    pub fn with_offsets(mut self, offsets: Vec<f64>) -> Result<Self, RegressionDataError> {
        if offsets.len() != self.observations.len() {
            return Err(RegressionDataError::Shape);
        }
        finite(&offsets, "offset")?;
        self.offsets = offsets.into();
        Ok(self)
    }
    /// Number of observations.
    pub fn rows(&self) -> usize {
        self.observations.len()
    }
    /// Number of explicitly supplied coefficient columns.
    pub fn columns(&self) -> usize {
        self.columns
    }
    /// Flat row-major predictors.
    pub fn design(&self) -> &[f64] {
        &self.design
    }
    /// Responses in row order.
    pub fn observations(&self) -> &[Y] {
        &self.observations
    }
    /// Additive linear-predictor offsets.
    pub fn offsets(&self) -> &[f64] {
        &self.offsets
    }
    #[inline]
    fn linear_predictor(&self, row: usize, beta: &[f64]) -> f64 {
        let mut sum = CompensatedSum::default();
        sum.add(self.offsets[row]);
        for (x, b) in self.design[row * self.columns..(row + 1) * self.columns]
            .iter()
            .zip(beta)
        {
            sum.add_product(*x, *b);
        }
        sum.total()
    }
}
fn finite(values: &[f64], field: &'static str) -> Result<(), RegressionDataError> {
    if let Some(index) = first_nonfinite(values) {
        return Err(RegressionDataError::NonFinite { field, index });
    }
    Ok(())
}
fn check_columns<Y>(data: &RegressionData<Y>, beta: &[Expr]) -> Result<(), E> {
    if data.columns != beta.len() {
        Err(E::Shape)
    } else {
        Ok(())
    }
}
/// Normal observations with a shared positive residual standard deviation.
/// The local score is with respect to supplied coefficient expressions and sigma;
/// the model's parameter chart supplies its own pullback. No priors are added.
/// # Errors
/// Rejects nonfinite responses, wrong coefficient counts and foreign handles.
/// Evaluation rejects nonpositive scales or nonrepresentable density/scores.
pub fn normal_regression(
    builder: &mut ModelBuilder,
    beta: &[Expr],
    sigma: &Parameter<Positive>,
    data: RegressionData<f64>,
) -> Result<Expr, E> {
    builder.primitive(NormalRegression::new(beta, sigma, data)?)
}
/// Immutable native normal-regression description. Retains row-major data and
/// offsets. Eligible tall designs use square-root sufficient statistics; other
/// designs and numerical-limit states use the fused SIMD row kernel. Conditional
/// row reporting and generated source retain the original observations.
/// Coefficients may be arbitrary deterministic graph expressions. No intercept,
/// prior, standardization, independent leave-out units or generation is inferred.
pub struct NormalRegression {
    inputs: Box<[Expr]>,
    data: RegressionData<f64>,
    design: kernels::Design,
    compressed: Option<normal_compression::CompressedNormal>,
}
/// Reusable execution-local normal-regression derivative scratch. Immutable data
/// and prepared predictors remain in [`NormalRegression`], not in this workspace.
pub struct NormalRegressionWorkspace {
    scores: Scores,
}
impl NormalRegression {
    /// Creates an immutable replacement binding with the same rows/columns and
    /// handles. The old descriptor/data remain unchanged on failure.
    /// # Errors
    /// Rejects changed shape, invalid responses or overflowing packed storage.
    pub fn with_data(&self, data: RegressionData<f64>) -> Result<Self, E> {
        if data.rows() != self.data.rows() || data.columns() != self.data.columns() {
            return Err(E::Shape);
        }
        finite(data.observations(), "observation").map_err(error)?;
        let mut design = kernels::Design::new(&data)?;
        design.set_execution(self.execution_plan().execution());
        let compressed = normal_compression::CompressedNormal::new(&data);
        Ok(Self {
            inputs: self.inputs.clone(),
            data,
            design,
            compressed,
        })
    }
    /// Describes a fixed-design normal likelihood with a shared positive scale.
    /// # Errors
    /// Rejects coefficient/data shape mismatch, nonfinite responses or overflowing
    /// prepared design storage. Handle ownership is checked at graph registration.
    pub fn new(
        beta: &[Expr],
        sigma: &Parameter<Positive>,
        data: RegressionData<f64>,
    ) -> Result<Self, E> {
        check_columns(&data, beta)?;
        finite(data.observations(), "observation").map_err(error)?;
        let mut inputs = beta.to_vec();
        inputs.push(sigma.expr());
        let design = kernels::Design::new(&data)?;
        let compressed = normal_compression::CompressedNormal::new(&data);
        Ok(Self {
            inputs: inputs.into_boxed_slice(),
            data,
            design,
            compressed,
        })
    }
    /// Original observations, row-major design and offsets; no dataset copy.
    pub fn data(&self) -> &RegressionData<f64> {
        &self.data
    }
    /// Native CPU plan, including cached dispatch and prepared-design storage.
    pub fn execution_plan(&self) -> &RegressionPlan {
        &self.design.plan
    }
    /// Selects native dispatch before graph registration. Existing modeling
    /// helpers use automatic dispatch. This selects the row-kernel CPU policy;
    /// Gaussian compression and its numerical fallback are independent of it.
    #[must_use]
    pub fn with_execution(mut self, execution: RegressionExecution) -> Self {
        self.design.set_execution(execution);
        self
    }
}
impl Primitive for NormalRegression {
    type Workspace = NormalRegressionWorkspace;
    fn batch_workspace(&self) -> Self::Workspace {
        NormalRegressionWorkspace {
            scores: Scores::new(0),
        }
    }
    fn inputs(&self) -> &[Expr] {
        &self.inputs
    }
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "normal_regression",
            arguments: vec![
                PrimitiveArgument {
                    name: "coefficients",
                    shape: ArgumentShape::Vector(self.data.columns),
                },
                PrimitiveArgument {
                    name: "scale",
                    shape: ArgumentShape::Scalar,
                },
            ],
            probability: Some(ProbabilitySemantics {
                event: EventShape::Scalar,
                batch: vec![self.data.rows()],
                support: Support::Real,
                measure: ReferenceMeasure::Lebesgue { dimension: 1 },
            }),
        }
    }
    fn workspace(&self) -> Self::Workspace {
        NormalRegressionWorkspace {
            scores: Scores::new(self.data.columns),
        }
    }
    fn batch_rule(&self) -> Option<BatchRule<Self>> {
        Some(BatchRule {
            units: self.data.rows(),
            evaluate: |this, x, out, _| {
                if x.len() != this.inputs.len() || out.len() != this.data.rows() {
                    return Err(E::Shape);
                }
                let p = this.data.columns();
                for (row, out) in out.iter_mut().enumerate() {
                    let eta = this.data.linear_predictor(row, &x[..p]);
                    let law = Normal::new(eta, x[p]).map_err(error)?;
                    *out = law.log_prob(&this.data.observations[row]).map_err(error)?;
                }
                Ok(())
            },
        })
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        Some(Self::emit_source)
    }
    fn evaluate(
        &self,
        x: &[f64],
        g: &mut [f64],
        workspace: &mut Self::Workspace,
    ) -> Result<f64, E> {
        if x.len() != self.inputs.len()
            || g.len() != self.inputs.len()
            || workspace.scores.sums.len() != self.data.columns
        {
            return Err(E::Shape);
        }
        if let Some(compressed) = &self.compressed
            && let Some(value) = compressed.evaluate(x, g, &mut workspace.scores.sums)
        {
            return Ok(value);
        }
        self.design.plan.dispatch(NormalEval {
            data: &self.data,
            design: &self.design,
            broadcast: self.design.broadcast,
            x,
            g,
            scores: &mut workspace.scores,
        })
    }
}
impl NormalRegression {
    fn emit_source(&self) -> String {
        let n = self.data.rows();
        let p = self.data.columns();
        // Counted loops reuse the scalar Normal primal, not another density.
        format!(
            "fn normal(x: &[f64; 3]) -> f64 {{ {} }}\n\
            let design: [f64; {}] = {:?}; let observations: [f64; {n}] = {:?}; let offsets: [f64; {n}] = {:?};\n\
            let mut total = ::alea::math::numeric::CompensatedSum::default();\n\
            for i in 0..{n} {{\n\
              let mut eta = ::alea::math::numeric::CompensatedSum::default(); eta.add(offsets[i]);\n\
              for j in 0..{p} {{ eta.add_product(design[i*{p}+j], x[j]); }}\n\
              let value = normal(&[observations[i], eta.total(), x[{p}]]);\n\
              if !value.is_finite() {{ return f64::NAN; }} total.add(value);\n\
            }} total.total()",
            super::normal::NORMAL_PRIMAL,
            self.data.design().len(),
            self.data.design(),
            self.data.observations(),
            self.data.offsets()
        )
    }
}
/// Bernoulli observations with a logit link, preserving representable tail scores.
/// # Errors
/// Rejects wrong coefficient counts/foreign handles and nonrepresentable evaluation.
pub fn bernoulli_logit_regression(
    builder: &mut ModelBuilder,
    beta: &[Expr],
    data: RegressionData<bool>,
) -> Result<Expr, E> {
    builder.primitive(BernoulliLogitRegression::new(beta, data)?)
}
/// Poisson observations with a log-rate link. Offsets are log exposures when used.
/// Evaluates `y*eta-exp(eta)-ln(y!)` directly, retaining finite log probabilities
/// when exp(eta) underflows. Counts above 2^53 are explicitly rejected.
/// # Errors
/// Rejects count precision loss, coefficient mismatches and foreign handles.
/// Overflowing rates, predictors, sums or scores are numerical errors.
pub fn poisson_log_regression(
    builder: &mut ModelBuilder,
    beta: &[Expr],
    data: RegressionData<u64>,
) -> Result<Expr, E> {
    builder.primitive(PoissonLogRegression::new(beta, data)?)
}

// Separate arrays let coefficient-score SIMD preserve each coordinate's row
// order and Neumaier residual. Predictor and likelihood tiling live separately
// in kernels; scalar distribution APIs are not changed by these private kernels.
struct Scores {
    sums: Vec<f64>,
    corrections: Vec<f64>,
}
impl Scores {
    fn new(columns: usize) -> Self {
        Self {
            sums: vec![0.; columns],
            corrections: vec![0.; columns],
        }
    }
    fn reset(&mut self) {
        self.sums.fill(0.);
        self.corrections.fill(0.);
    }
    #[inline(always)]
    fn add_scaled<S: pulp::Simd>(&mut self, simd: S, row: &[f64], residual: f64) {
        let width = self.sums.len();
        let row = &row[..width];
        let (sums, st) = S::as_mut_simd_f64s(&mut self.sums);
        let (corrections, ct) = S::as_mut_simd_f64s(&mut self.corrections[..width]);
        let (row, rt) = S::as_simd_f64s(row);
        let scale = simd.splat_f64s(residual);
        for ((sum, correction), x) in sums.iter_mut().zip(corrections).zip(row) {
            let value = simd.mul_f64s(*x, scale);
            let next = simd.add_f64s(*sum, value);
            let large = simd.greater_than_or_equal_f64s(simd.abs_f64s(*sum), simd.abs_f64s(value));
            let error = simd.select_f64s(
                large,
                simd.add_f64s(simd.sub_f64s(*sum, next), value),
                simd.add_f64s(simd.sub_f64s(value, next), *sum),
            );
            *correction = simd.add_f64s(*correction, error);
            *sum = next;
        }
        for ((sum, correction), x) in st.iter_mut().zip(ct).zip(rt) {
            let value = x * residual;
            let next = *sum + value;
            *correction += if sum.abs() >= value.abs() {
                (*sum - next) + value
            } else {
                (value - next) + *sum
            };
            *sum = next;
        }
    }
    // Keep heads/corrections in registers across four observations. Reversing
    // these loops does NOT reassociate a coordinate's row reduction: each still
    // receives rows 0,1,2,3 in order, with the same Neumaier operations.
    #[inline(always)]
    fn add_tile<S: pulp::Simd>(
        &mut self,
        simd: S,
        rows: [&[f64]; kernels::TILE],
        residuals: &[f64; kernels::TILE],
    ) {
        // State the already validated common extent in the slice types. This
        // lets LLVM remove the independent zip-length minima in the hot loop.
        let width = self.sums.len();
        let rows = rows.map(|row| &row[..width]);
        let (sums, st) = S::as_mut_simd_f64s(&mut self.sums);
        let (corrections, ct) = S::as_mut_simd_f64s(&mut self.corrections[..width]);
        let [(a, at), (b, bt), (c, ctail), (d, dt)] = rows.map(S::as_simd_f64s);
        let scales = residuals.map(|r| simd.splat_f64s(r));
        for (((((sum, correction), a), b), c), d) in
            sums.iter_mut().zip(corrections).zip(a).zip(b).zip(c).zip(d)
        {
            let mut head = *sum;
            let mut low = *correction;
            for (x, scale) in [*a, *b, *c, *d].into_iter().zip(scales) {
                let value = simd.mul_f64s(x, scale);
                let next = simd.add_f64s(head, value);
                let large =
                    simd.greater_than_or_equal_f64s(simd.abs_f64s(head), simd.abs_f64s(value));
                let error = simd.select_f64s(
                    large,
                    simd.add_f64s(simd.sub_f64s(head, next), value),
                    simd.add_f64s(simd.sub_f64s(value, next), head),
                );
                low = simd.add_f64s(low, error);
                head = next;
            }
            *sum = head;
            *correction = low;
        }
        for (((((sum, correction), a), b), c), d) in
            st.iter_mut().zip(ct).zip(at).zip(bt).zip(ctail).zip(dt)
        {
            for (x, scale) in [*a, *b, *c, *d].into_iter().zip(residuals) {
                let value = x * scale;
                let next = *sum + value;
                *correction += if sum.abs() >= value.abs() {
                    (*sum - next) + value
                } else {
                    (value - next) + *sum
                };
                *sum = next;
            }
        }
    }
    fn finish(&self, total: CompensatedSum, g: &mut [f64]) -> Result<f64, E> {
        let value = total.total();
        if !value.is_finite()
            || self
                .sums
                .iter()
                .zip(&self.corrections)
                .any(|(s, c)| !(s + c).is_finite())
        {
            return Err(E::Numerical);
        }
        for ((out, sum), correction) in g.iter_mut().zip(&self.sums).zip(&self.corrections) {
            *out = sum + correction;
        }
        Ok(value)
    }
}
struct NormalEval<'a> {
    data: &'a RegressionData<f64>,
    design: &'a kernels::Design,
    broadcast: bool,
    x: &'a [f64],
    g: &'a mut [f64],
    scores: &'a mut Scores,
}
impl pulp::WithSimd for NormalEval<'_> {
    type Output = Result<f64, E>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let sigma = self.x[self.data.columns];
        if !sigma.is_finite() || sigma <= 0. {
            return Err(E::Numerical);
        }
        let inverse = sigma.recip();
        // A reciprocal is not interchangeable with division near representability
        // limits. Keep the original division path when either squared scale is
        // nonnormal (including under/overflow); never clip or reject that input.
        let ordinary = (sigma * sigma).is_normal() && (inverse * inverse).is_normal();
        match (ordinary, self.broadcast) {
            (true, true) => self.evaluate::<S, true, true>(simd, inverse),
            (true, false) => self.evaluate::<S, true, false>(simd, inverse),
            (false, true) => self.evaluate::<S, false, true>(simd, inverse),
            (false, false) => self.evaluate::<S, false, false>(simd, inverse),
        }
    }
}
impl NormalEval<'_> {
    #[inline(always)]
    fn evaluate<S: pulp::Simd, const RECIPROCAL: bool, const BROADCAST: bool>(
        self,
        simd: S,
        inverse: f64,
    ) -> Result<f64, E> {
        if RECIPROCAL && size_of::<S::f64s>() <= size_of::<[f64; kernels::TILE]>() {
            return self.batch::<S, BROADCAST>(simd, inverse);
        }
        let Self {
            data, x, g, scores, ..
        } = self;
        let p = data.columns;
        let sigma = x[p];
        let normalizer = -sigma.ln() - 0.5 * std::f64::consts::TAU.ln();
        let mut total = CompensatedSum::default();
        let mut sigma_score = CompensatedSum::default();
        scores.reset();
        for i in 0..data.rows() {
            let eta = if BROADCAST {
                x[0] + 0.
            } else {
                data.linear_predictor(i, &x[..p])
            };
            let residual = data.observations[i] - eta;
            let z = if RECIPROCAL {
                residual * inverse
            } else {
                residual / sigma
            };
            let value = -0.5 * z * z;
            total.add(if RECIPROCAL {
                value
            } else {
                value + normalizer
            });
            let mean_score = if RECIPROCAL { z * inverse } else { z / sigma };
            if BROADCAST {
                scores.add_scaled(simd, &[1.], mean_score);
            } else {
                scores.add_scaled(simd, &data.design[i * p..(i + 1) * p], mean_score);
            }
            sigma_score.add(if RECIPROCAL {
                (z * z - 1.) * inverse
            } else {
                (z * z - 1.) / sigma
            });
        }
        if RECIPROCAL {
            // Accumulate the quadratic scaled by -1/2, rather than sum(z^2),
            // so summing squares cannot overflow an otherwise finite density.
            total.add_product(normalizer, data.rows() as f64);
        }
        if !sigma_score.total().is_finite() {
            return Err(E::Numerical);
        }
        let value = scores.finish(total, &mut g[..p])?;
        g[p] = sigma_score.total();
        Ok(value)
    }
}
struct LogitEval<'a> {
    data: &'a RegressionData<bool>,
    design: &'a kernels::Design,
    x: &'a [f64],
    g: &'a mut [f64],
    scores: &'a mut Scores,
}
struct PoissonEval<'a> {
    data: &'a RegressionData<u64>,
    design: &'a kernels::Design,
    normalizers: &'a [f64],
    x: &'a [f64],
    g: &'a mut [f64],
    scores: &'a mut Scores,
}
impl pulp::WithSimd for PoissonEval<'_> {
    type Output = Result<f64, E>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        self.batch(simd)
    }
}
impl pulp::WithSimd for LogitEval<'_> {
    type Output = Result<f64, E>;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        self.batch(simd)
    }
}

#[cfg(test)]
mod score_tests {
    use super::*;

    struct AddTile<'a> {
        scores: &'a mut Scores,
        rows: [&'a [f64]; kernels::TILE],
        residuals: &'a [f64; kernels::TILE],
    }
    impl pulp::WithSimd for AddTile<'_> {
        type Output = ();
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            self.scores.add_tile(simd, self.rows, self.residuals);
        }
    }

    #[test]
    fn tiled_scores_preserve_ordered_heads_and_corrections_for_all_widths() {
        for p in 0..=65 {
            let mut tiled = Scores::new(p);
            let mut ordered = Scores::new(p);
            for block in 0..8 {
                let rows: [Vec<f64>; kernels::TILE] = std::array::from_fn(|r| {
                    (0..p)
                        .map(|j| match (j + r + block) % 6 {
                            0 => 1e16,
                            1 => 1.,
                            2 => -1e16,
                            3 => f64::MIN_POSITIVE,
                            4 => -0.,
                            _ => -0.125,
                        })
                        .collect()
                });
                let residuals = [1., -0.5, 0.25, 2.];
                let rows = rows.each_ref().map(Vec::as_slice);
                pulp::Arch::new().dispatch(AddTile {
                    scores: &mut tiled,
                    rows,
                    residuals: &residuals,
                });
                for (row, scale) in rows.into_iter().zip(residuals) {
                    pulp::Arch::new().dispatch(Add {
                        scores: &mut ordered,
                        row,
                        scale,
                    });
                }
                for (a, b) in tiled
                    .sums
                    .iter()
                    .chain(&tiled.corrections)
                    .zip(ordered.sums.iter().chain(&ordered.corrections))
                {
                    assert_eq!(a.to_bits(), b.to_bits(), "p={p},block={block}");
                }
            }
        }
    }

    struct Add<'a> {
        scores: &'a mut Scores,
        row: &'a [f64],
        scale: f64,
    }
    impl pulp::WithSimd for Add<'_> {
        type Output = ();
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            self.scores.add_scaled(simd, self.row, self.scale);
        }
    }

    #[test]
    fn vector_scores_preserve_scalar_bits_for_all_tails_and_cancellation() {
        for p in 0..=65 {
            let mut vector = Scores::new(p);
            let mut scalar = Scores::new(p);
            let mut reference = vec![CompensatedSum::default(); p];
            let mut row = vec![0.; p];
            for i in 0..100 {
                for (j, x) in row.iter_mut().enumerate() {
                    *x = match i % 5 {
                        0 => 1e16,
                        1 => (j + 1) as f64 / 7.,
                        2 => -1e16,
                        3 => f64::MIN_POSITIVE,
                        _ => -((j + 1) as f64) / 13.,
                    };
                }
                let scale = if i % 7 == 0 { -0.75 } else { 1. };
                pulp::Arch::new().dispatch(Add {
                    scores: &mut vector,
                    row: &row,
                    scale,
                });
                scalar.add_scaled(pulp::Scalar, &row, scale);
                for (j, expected) in reference.iter_mut().enumerate() {
                    expected.add(row[j] * scale);
                    // Exact replay, not an approximate mathematical identity.
                    assert_eq!(
                        (vector.sums[j] + vector.corrections[j]).to_bits(),
                        expected.total().to_bits()
                    );
                    assert_eq!(
                        (scalar.sums[j] + scalar.corrections[j]).to_bits(),
                        expected.total().to_bits()
                    );
                }
            }
            vector.reset();
            assert!(
                vector
                    .sums
                    .iter()
                    .chain(&vector.corrections)
                    .all(|v| *v == 0.)
            );
        }
    }

    #[test]
    fn vector_score_failure_preserves_outputs_and_scratch_can_be_reused() {
        for p in [0, 1, 4, 5, 8, 9, 16, 17] {
            let mut scores = Scores::new(p);
            let mut output = vec![17.; p];
            for value in [f64::MAX, f64::INFINITY, f64::NAN] {
                let row = vec![value; p];
                pulp::Arch::new().dispatch(Add {
                    scores: &mut scores,
                    row: &row,
                    scale: 2.,
                });
                if p > 0 {
                    assert!(
                        scores
                            .finish(CompensatedSum::default(), &mut output)
                            .is_err()
                    );
                    assert!(output.iter().all(|v| *v == 17.));
                }
                scores.reset();
            }
            // A bad density must also leave otherwise valid scores unpublished.
            let mut total = CompensatedSum::default();
            total.add(f64::INFINITY);
            assert!(scores.finish(total, &mut output).is_err());
            assert!(output.iter().all(|v| *v == 17.));
            let row = vec![0.25; p];
            pulp::Arch::new().dispatch(Add {
                scores: &mut scores,
                row: &row,
                scale: 2.,
            });
            assert_eq!(
                scores
                    .finish(CompensatedSum::default(), &mut output)
                    .unwrap(),
                0.
            );
            assert!(output.iter().all(|v| *v == 0.5));
        }
    }
}

/// Likelihood-specific existence diagnostics, separate from numerical convergence.
pub mod diagnostics;

mod standardization;
pub use standardization::*;
