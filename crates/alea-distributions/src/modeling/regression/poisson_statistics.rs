//! Sufficient statistics for Poisson batches with one common linear predictor.
//! Inspired by Pichler/Jewson/Avalos-Pacheco (2025), §2.1/§3.4. Distinct design
//! rows still need parameter-dependent rates; their fused row kernel is retained.
use super::{CompensatedSum, E, PoissonEval, RegressionData};

pub(super) struct PoissonStatistics {
    count: f64,
    normalizer: f64,
}
impl PoissonStatistics {
    pub(super) fn new(data: &RegressionData<u64>, normalizers: &[f64]) -> Option<Self> {
        // A bounded initial amortization envelope, not an optimal cost model.
        // Check eligibility before computing statistics for unrelated designs.
        if data.rows() < 128
            || data.columns > 64
            || !data
                .offsets
                .iter()
                .all(|v| v.to_bits() == data.offsets[0].to_bits())
            || !(0..data.rows()).all(|i| {
                data.design[i * data.columns..(i + 1) * data.columns]
                    .iter()
                    .zip(&data.design[..data.columns])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            })
        {
            return None;
        }
        let mut count = CompensatedSum::default();
        let mut normalizer = CompensatedSum::default();
        for (&y, &constant) in data.observations.iter().zip(normalizers) {
            count.add(y as f64);
            normalizer.add(constant);
        }
        let result = Self {
            count: count.total(),
            normalizer: normalizer.total(),
        };
        (result.count.is_finite() && result.normalizer.is_finite()).then_some(result)
    }
}

pub(super) struct Evaluate<'a> {
    pub(super) evaluation: PoissonEval<'a>,
    pub(super) statistics: &'a PoissonStatistics,
}
impl pulp::WithSimd for Evaluate<'_> {
    type Output = Result<f64, E>;
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let PoissonEval {
            data, x, g, scores, ..
        } = self.evaluation;
        let eta = data.linear_predictor(0, x);
        // Limit the optimization, not the model. Extreme states retry the
        // unchanged row evaluator, including underflow and typed failures.
        if !eta.is_finite() || eta.abs() > 32. {
            return Err(E::Numerical);
        }
        let rate = eta.exp();
        let n = data.rows() as f64;
        let count = self.statistics.count;
        let normalizer = self.statistics.normalizer;
        let mut density = CompensatedSum::default();
        density.add_product(count, eta);
        density.add_product(-n, rate);
        density.add(-normalizer);
        let mut residual = CompensatedSum::default();
        residual.add(count);
        residual.add_product(-n, rate);
        let residual = residual.total();
        scores.reset();
        scores.add_scaled(simd, &data.design[..data.columns], residual);
        // Conservative engineering screens, NOT certified rounding bounds.
        // Near-stationary scores / cancellation of large normalizers should use
        // the original per-observation residuals rather than clipped statistics.
        let uncertainty = 64. * f64::EPSILON * (count + n * rate);
        for (&a, &score) in data.design[..data.columns].iter().zip(&scores.sums) {
            let bound = a.abs() * uncertainty;
            if !bound.is_finite() || bound > 1e-10 * (1. + score.abs()) {
                return Err(E::Numerical);
            }
        }
        let magnitude = count * eta.abs() + n * rate + normalizer.abs();
        if !magnitude.is_finite()
            || 64. * f64::EPSILON * magnitude > 1e-10 * (1. + density.total().abs())
        {
            return Err(E::Numerical);
        }
        scores.finish(density, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modeling::regression::{RegressionExecution, Scores, kernels::Design};

    #[test]
    fn common_statistics_match_rows_and_decline_any_different_design_or_offset() {
        for n in [128, 129, 1000, 10000] {
            for p in [0, 1, 8] {
                let data = RegressionData::new(
                    p,
                    (0..n * p)
                        .map(|i| (i % p.max(1)) as f64 / 8. + 0.5)
                        .collect(),
                    (0..n).map(|i| (i % 5) as u64).collect(),
                )
                .unwrap()
                .with_offsets(vec![0.2; n])
                .unwrap();
                let constants: Vec<_> = data
                    .observations
                    .iter()
                    .map(|&y| statrs::function::gamma::ln_gamma(y as f64 + 1.))
                    .collect();
                let statistics = PoissonStatistics::new(&data, &constants).unwrap();
                let mut design = Design::new(&data).unwrap();
                for execution in [RegressionExecution::Auto, RegressionExecution::Scalar] {
                    design.set_execution(execution);
                    for coordinate in [-1., 0.03, 1.] {
                        let x = vec![coordinate; p];
                        let mut g = vec![17.; p];
                        let mut h = vec![0.; p];
                        let mut scratch = Scores::new(p);
                        let value = design
                            .plan
                            .dispatch(Evaluate {
                                evaluation: PoissonEval {
                                    data: &data,
                                    design: &design,
                                    normalizers: &constants,
                                    x: &x,
                                    g: &mut g,
                                    scores: &mut scratch,
                                },
                                statistics: &statistics,
                            })
                            .unwrap();
                        let reference = design
                            .plan
                            .dispatch(PoissonEval {
                                data: &data,
                                design: &design,
                                normalizers: &constants,
                                x: &x,
                                g: &mut h,
                                scores: &mut scratch,
                            })
                            .unwrap();
                        for (a, b) in g.iter().chain([&value]).zip(h.iter().chain([&reference])) {
                            assert!(
                                (a - b).abs() < 3e-12 * (1. + b.abs()),
                                "{n}/{p}: {a} != {b}"
                            );
                        }
                    }
                }
                let mut offsets = vec![0.2; n];
                offsets[n - 1] = f64::from_bits(0.2_f64.to_bits() + 1);
                assert!(
                    PoissonStatistics::new(
                        &data.clone().with_offsets(offsets).unwrap(),
                        &constants
                    )
                    .is_none()
                );
                if p > 0 {
                    let mut different = data.design.to_vec();
                    different[n * p - 1] += 0.01;
                    let changed =
                        RegressionData::new(p, different, data.observations.to_vec()).unwrap();
                    assert!(PoissonStatistics::new(&changed, &constants).is_none());
                }
            }
        }
    }

    #[test]
    fn cancellation_and_extreme_coordinates_decline_without_publishing_scores() {
        let data = RegressionData::new(1, vec![1.; 128], vec![1_000_000_000_000; 128]).unwrap();
        let constants: Vec<_> = data
            .observations
            .iter()
            .map(|&y| statrs::function::gamma::ln_gamma(y as f64 + 1.))
            .collect();
        let statistics = PoissonStatistics::new(&data, &constants).unwrap();
        let design = Design::new(&data).unwrap();
        for q in [1e12_f64.ln(), -1000., 1000., f64::MAX] {
            let mut g = [17.];
            assert!(
                design
                    .plan
                    .dispatch(Evaluate {
                        evaluation: PoissonEval {
                            data: &data,
                            design: &design,
                            normalizers: &constants,
                            x: &[q],
                            g: &mut g,
                            scores: &mut Scores::new(1)
                        },
                        statistics: &statistics
                    })
                    .is_err()
            );
            assert_eq!(g, [17.]);
        }
    }
}
