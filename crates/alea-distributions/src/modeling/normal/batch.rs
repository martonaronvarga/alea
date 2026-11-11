//! One-axis Normal map/reduce, retaining row semantics and native scalar errors.
use super::*;
use crate::scalar::{normal_local_scores, normal_standardized_value};
use alea_core::model::assembly::{NativeRules, OperationDescriptor, SourceBody, SourceRule};
use alea_math::numeric::CompensatedSum;

/// Independent scalar Normal terms with explicit scalar/vector broadcasting.
/// This is a joint log-density reduction, not a multivariate Normal event.
/// The existing Normal arithmetic is shared by scalar and SIMD execution.
/// Row values and broadcast scores use ordered compensated reductions; bitwise
/// equality with a separately assembled scalar-factor graph is not promised.
pub struct NormalBatch {
    observations: Option<Box<[f64]>>,
    lengths: [usize; 3],
    rows: usize,
    arch: pulp::Arch,
}
/// Reusable native scores and scale preparation, owned by one execution.
pub struct NormalBatchWorkspace {
    scores: Vec<f64>,
    logs: Vec<(Option<u64>, f64)>,
}
impl NormalBatch {
    /// Describes event, location and scale argument lengths, independently of
    /// graph handles. Each must be nonempty and equal one or the largest length.
    /// # Errors
    /// Rejects incompatible shapes/overflow; ownership is checked on registration.
    pub fn new(lengths: [usize; 3]) -> Result<Self, E> {
        Self::build(lengths, None)
    }
    /// Captures fixed finite observations without creating constant graph nodes.
    /// Location and scale argument lengths are one or the observation count.
    /// All three scalar scores are still checked, including the fixed event score.
    /// # Errors
    /// Rejects empty/nonfinite observations, incompatible shapes or extent overflow.
    pub fn observed(observations: Vec<f64>, locations: usize, scales: usize) -> Result<Self, E> {
        if observations.is_empty() {
            return Err(E::Shape);
        }
        if observations.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        Self::build(
            [0, locations, scales],
            Some(observations.into_boxed_slice()),
        )
    }
    fn build(lengths: [usize; 3], observations: Option<Box<[f64]>>) -> Result<Self, E> {
        let rows = observations
            .as_ref()
            .map_or_else(|| *lengths.iter().max().unwrap_or(&0), |v| v.len());
        if rows == 0
            || lengths
                .iter()
                .enumerate()
                .any(|(i, &n)| !(i == 0 && observations.is_some()) && n != 1 && n != rows)
        {
            return Err(E::Shape);
        }
        let count = lengths
            .iter()
            .try_fold(0_usize, |sum, n| sum.checked_add(*n))
            .ok_or(E::Shape)?;
        if count > isize::MAX as usize / size_of::<(Option<u64>, f64)>() {
            return Err(E::Shape);
        }
        Ok(Self {
            observations,
            lengths,
            rows,
            arch: pulp::Arch::new(),
        })
    }
    /// Number of conditional scalar terms, before the joint reduction.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Immutable fixed responses, absent when events are model expressions.
    pub fn observations(&self) -> Option<&[f64]> {
        self.observations.as_deref()
    }
    /// Same-shape immutable response replacement; previously generated source
    /// must be rebuilt, and cached target states must not cross this boundary.
    /// # Errors
    /// Rejects expression-event batches, changed lengths and nonfinite data.
    pub fn with_observations(&self, observations: Vec<f64>) -> Result<Self, E> {
        if self.observations.is_none() || observations.len() != self.rows {
            return Err(E::Shape);
        }
        Self::observed(observations, self.lengths[1], self.lengths[2])
    }
    fn row(&self, x: &[f64], i: usize) -> [f64; 3] {
        let [e, m, s] = self.lengths;
        [
            self.observations
                .as_ref()
                .map_or_else(|| x[if e == 1 { 0 } else { i }], |y| y[i]),
            x[e + if m == 1 { 0 } else { i }],
            x[e + m + if s == 1 { 0 } else { i }],
        ]
    }
    fn accumulate(
        &self,
        i: usize,
        gradient: [f64; 3],
        work: &mut NormalBatchWorkspace,
        shared: &mut [CompensatedSum; 3],
    ) {
        let mut first = 0;
        for (j, &n) in self.lengths.iter().enumerate() {
            if n == 1 {
                shared[j].add(gradient[j]);
            } else if n != 0 {
                work.scores[first + i] = gradient[j];
            }
            first += n;
        }
    }
    fn scalar_row(
        &self,
        x: &[f64],
        i: usize,
        work: &mut NormalBatchWorkspace,
    ) -> Result<(f64, [f64; 3]), E> {
        let [y, mu, sigma] = self.row(x, i);
        let law = Normal::new(mu, sigma).map_err(error)?;
        law.log_prob_event_parameter_scores_prepared(
            y,
            work.logs[if self.lengths[2] == 1 { 0 } else { i }].1,
        )
        .map_err(error)
    }
    fn evaluate_simd<S: pulp::Simd>(
        &self,
        simd: S,
        x: &[f64],
        g: &mut [f64],
        work: &mut NormalBatchWorkspace,
    ) -> Result<f64, E> {
        // Vector argument slots are written exactly once; only broadcast
        // arguments require a reduction. Failure scratch is never published.
        let mut shared = [CompensatedSum::default(); 3];
        let first = self.lengths[0] + self.lengths[1];
        for (log, &scale) in work.logs.iter_mut().zip(&x[first..]) {
            if log.0 != Some(scale.to_bits()) {
                *log = (Some(scale.to_bits()), scale.ln());
            }
        }
        let mut total = CompensatedSum::default();
        for start in (0..self.rows).step_by(4) {
            let len = (self.rows - start).min(4);
            let mut events = [0.; 4];
            let mut locations = [0.; 4];
            let mut scales = [1.; 4];
            let mut logs = [0.; 4];
            for j in 0..len {
                [events[j], locations[j], scales[j]] = self.row(x, start + j);
                logs[j] = work.logs[if self.lengths[2] == 1 { 0 } else { start + j }].1;
            }
            // Decline the whole packet if any constructor/event is invalid, so
            // a later parameter error cannot hide an earlier numeric score error.
            let ordinary = (0..len).all(|j| {
                events[j].is_finite()
                    && locations[j].is_finite()
                    && scales[j].is_finite()
                    && scales[j] > 0.
            });
            let mut values = [0.; 4];
            let mut gradients = [[0.; 4]; 3];
            if ordinary {
                let (es, et) = S::as_simd_f64s(&events);
                let (ms, _) = S::as_simd_f64s(&locations);
                let (ss, _) = S::as_simd_f64s(&scales);
                let (ls, _) = S::as_simd_f64s(&logs);
                let (vs, _) = S::as_mut_simd_f64s(&mut values);
                let [ge, gm, gs] = &mut gradients;
                let (ge, _) = S::as_mut_simd_f64s(ge);
                let (gm, _) = S::as_mut_simd_f64s(gm);
                let (gs, _) = S::as_mut_simd_f64s(gs);
                for j in 0..es.len() {
                    let (z, value) = normal_standardized_value(simd, es[j], ms[j], ss[j], ls[j]);
                    vs[j] = value;
                    [ge[j], gm[j], gs[j]] = normal_local_scores(simd, z, ss[j]);
                }
                if !et.is_empty() {
                    // A future packet wider than four: scalar fallback.
                    for j in 0..len {
                        let (v, s) = self.scalar_row(x, start + j, work)?;
                        values[j] = v;
                        for k in 0..3 {
                            gradients[k][j] = s[k];
                        }
                    }
                }
            }
            for j in 0..len {
                let (value, scores) = if ordinary {
                    (values[j], std::array::from_fn(|k| gradients[k][j]))
                } else {
                    self.scalar_row(x, start + j, work)?
                };
                // Exact scalar admission, in row order. Even omitted event or
                // constant-argument scores cannot be silently discarded.
                if value.is_nan() || scores.iter().any(|v| !v.is_finite()) {
                    return Err(error(alea_core::distribution::DistributionError::Numerical));
                }
                total.add(value);
                self.accumulate(start + j, scores, work, &mut shared);
            }
        }
        let value = total.total();
        let mut first = 0;
        for (j, &n) in self.lengths.iter().enumerate() {
            if n == 1 {
                work.scores[first] = shared[j].total();
            }
            first += n;
        }
        if !value.is_finite() || work.scores.iter().any(|v| !v.is_finite()) {
            return Err(E::Numerical);
        }
        g.copy_from_slice(&work.scores);
        Ok(value)
    }
}
impl NormalBatch {
    fn signature(&self) -> PrimitiveSignature {
        PrimitiveSignature {
            name: "normal_batch",
            arguments: ["event", "location", "scale"]
                .into_iter()
                .zip(self.lengths)
                .filter(|(_, n)| *n != 0)
                .map(|(name, n)| PrimitiveArgument {
                    name,
                    shape: if n == 1 {
                        ArgumentShape::Scalar
                    } else {
                        ArgumentShape::Vector(n)
                    },
                })
                .collect(),
            probability: Some(ProbabilitySemantics {
                event: EventShape::Scalar,
                batch: vec![self.rows],
                support: Support::Real,
                measure: ReferenceMeasure::Lebesgue { dimension: 1 },
            }),
        }
    }
    /// Native kernel workspace.
    pub fn workspace(&self) -> NormalBatchWorkspace {
        NormalBatchWorkspace {
            scores: vec![0.; self.lengths.iter().sum::<usize>()],
            logs: vec![(None, 0.); self.lengths[2]],
        }
    }
    /// Native kernel batch workspace.
    pub fn batch_workspace(&self) -> NormalBatchWorkspace {
        NormalBatchWorkspace {
            scores: Vec::new(),
            logs: Vec::new(),
        }
    }
    /// Native kernel evaluate.
    pub fn evaluate(
        &self,
        x: &[f64],
        g: &mut [f64],
        work: &mut NormalBatchWorkspace,
    ) -> Result<f64, E> {
        if x.len() != self.lengths.iter().sum::<usize>()
            || g.len() != x.len()
            || work.scores.len() != x.len()
            || work.logs.len() != self.lengths[2]
        {
            return Err(E::Shape);
        }
        struct Evaluate<'a> {
            batch: &'a NormalBatch,
            x: &'a [f64],
            g: &'a mut [f64],
            work: &'a mut NormalBatchWorkspace,
        }
        impl pulp::WithSimd for Evaluate<'_> {
            type Output = Result<f64, E>;
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                self.batch.evaluate_simd(simd, self.x, self.g, self.work)
            }
        }
        self.arch.dispatch(Evaluate {
            batch: self,
            x,
            g,
            work,
        })
    }
    fn batch_rule(&self) -> Option<BatchRule<Self>> {
        Some(BatchRule {
            units: self.rows,
            evaluate: |this, x, out, _| {
                if x.len() != this.lengths.iter().sum::<usize>() || out.len() != this.rows {
                    return Err(E::Shape);
                }
                for (i, out) in out.iter_mut().enumerate() {
                    let [y, mu, sigma] = this.row(x, i);
                    *out = Normal::new(mu, sigma)
                        .map_err(error)?
                        .log_prob_event_parameter_scores(y)
                        .map_err(error)?
                        .0;
                }
                Ok(())
            },
        })
    }
    fn source_rule(&self) -> Option<fn(&Self) -> String> {
        Some(|this| {
            let [e, m, s] = this.lengths;
            let index = |first, n| {
                if n == 1 {
                    format!("x[{first}]")
                } else if first == 0 {
                    "x[i]".to_owned()
                } else {
                    format!("x[{first}+i]")
                }
            };
            let event = this
                .observations
                .as_ref()
                .map_or_else(|| index(0, e), |y| format!("{:?}[i]", y));
            format!(
                "fn normal(x:&[f64;3])->f64 {{{}}}\nlet mut total=::alea::math::numeric::CompensatedSum::default();for i in 0..{} {{let value=normal(&[{event},{},{}]);if !value.is_finite() {{return f64::NAN;}}total.add(value);}}total.total()",
                super::NORMAL_PRIMAL,
                this.rows,
                index(e, m),
                index(e + m, s)
            )
        })
    }
    fn source_gradient_rule(
        &self,
    ) -> Option<fn(&Self) -> alea_core::model::assembly::SourceGradient> {
        Some(|this| {
            let [e, m, _] = this.lengths;
            let constructor = this.observations.as_ref().map_or_else(
                || format!("Native::new([{e},{m},{}])", this.lengths[2]),
                |y| format!("Native::observed(vec!{:?},{m},{})", y, this.lengths[2]),
            );
            let setup = constructor;
            super::super::owned_primitive_source(
                "::alea::distributions::modeling::NormalBatch",
                &setup,
            )
        })
    }
}
impl OperationDescriptor for NormalBatch {
    type Workspace = NormalBatchWorkspace;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packets_preserve_scalar_bits_and_extreme_admission_including_signed_zero() {
        for rows in [1, 3, 4, 5, 129] {
            let batch = NormalBatch::new([rows, rows, rows]).unwrap();
            for (event, location, scale) in [
                (0., -0., 1.),
                (-0., 0., 1.),
                (0.3, -0.2, 1.7),
                (1e150, -1e150, 1e150),
                (1., 0., 1e-200),
                (0., 0., f64::from_bits(1)),
                (f64::MAX, -f64::MAX, 1.),
                (f64::NAN, 0., 1.),
            ] {
                let mut x = vec![event; rows];
                x.extend(vec![location; rows]);
                x.extend(vec![scale; rows]);
                let mut scalar = vec![17.; x.len()];
                let mut vector = scalar.clone();
                let expected =
                    batch.evaluate_simd(pulp::Scalar, &x, &mut scalar, &mut batch.workspace());
                let actual = batch.evaluate(&x, &mut vector, &mut batch.workspace());
                match (expected, actual) {
                    (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                    (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                    other => panic!("{other:?}"),
                }
                assert_eq!(
                    scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
