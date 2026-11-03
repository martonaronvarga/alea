//! Square-root sufficient statistics for a fixed Gaussian design.
//!
//! Streaming orthogonal rotations compress [X, y-offset] to an upper triangular
//! square root. Thus ||y-offset-X beta|| = ||R [-beta, 1]|| without subtracting
//! large quadratic forms. No inverse, rank decision or ridge penalty is needed.
//! The original rows remain authoritative at numerical limits and near a fit.
use super::RegressionData;
use alea_math::numeric::CompensatedSum;

pub(super) struct CompressedNormal {
    root: Box<[f64]>,
    columns: usize,
    rows: usize,
    // Input norm includes y and offsets separately: cancellation in y-offset
    // must not make the admission test optimistic.
    input_norm: f64,
}

impl CompressedNormal {
    pub(super) fn new(data: &RegressionData<f64>) -> Option<Self> {
        let p = data.columns;
        // Bound preparation/storage for wide problems. Small batches cannot
        // amortize O(N P^2) preparation; they retain the existing SIMD kernel.
        if p > 64 || data.rows() < 128 || data.rows() < 16 * (p + 1) {
            return None;
        }
        let width = p + 1;
        let mut root = vec![0_f64; width * width];
        let mut row = vec![0.; width];
        let mut input_norm = 0_f64;
        for i in 0..data.rows() {
            row[..p].copy_from_slice(&data.design[i * p..(i + 1) * p]);
            row[p] = data.observations[i] - data.offsets[i];
            for &v in &row[..p] {
                input_norm = input_norm.hypot(v);
            }
            input_norm = input_norm
                .hypot(data.observations[i])
                .hypot(data.offsets[i]);
            for j in 0..width {
                let diagonal = root[j * width + j];
                let incoming = row[j];
                // Exact zero check is structural, not an approximate rank test.
                if incoming == 0. {
                    continue;
                }
                let radius = diagonal.hypot(incoming);
                if !radius.is_normal() {
                    return None;
                }
                let c = diagonal / radius;
                let s = incoming / radius;
                root[j * width + j] = radius;
                for k in j + 1..width {
                    let a = root[j * width + k];
                    let b = row[k];
                    let mut top = CompensatedSum::default();
                    top.add_product(c, a);
                    top.add_product(s, b);
                    let mut bottom = CompensatedSum::default();
                    bottom.add_product(-s, a);
                    bottom.add_product(c, b);
                    root[j * width + k] = top.total();
                    row[k] = bottom.total();
                }
            }
        }
        if !input_norm.is_normal() || root.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Self {
            root: root.into(),
            columns: p,
            rows: data.rows(),
            input_norm,
        })
    }

    // None means retry the unchanged row kernel, not accept an invalid density.
    // Staged scores ensure fallback/error never exposes partial output.
    pub(super) fn evaluate(&self, x: &[f64], g: &mut [f64], scratch: &mut [f64]) -> Option<f64> {
        let p = self.columns;
        let sigma = x[p];
        let inverse = sigma.recip();
        if sigma <= 0. || !(sigma * sigma).is_normal() || !(inverse * inverse).is_normal() {
            return None;
        }
        let width = p + 1;
        let mut residuals = [0.; 65];
        let mut residual_norm = 0_f64;
        let mut beta_norm = 1_f64;
        for &b in &x[..p] {
            beta_norm = beta_norm.hypot(b);
        }
        for (i, residual) in residuals[..width].iter_mut().enumerate() {
            let mut sum = CompensatedSum::default();
            sum.add(self.root[i * width + p]);
            for (j, &b) in x[..p].iter().enumerate().skip(i) {
                sum.add_product(-self.root[i * width + j], b);
            }
            *residual = sum.total();
            residual_norm = residual_norm.hypot(*residual);
        }
        // Conservative conditioning screen, not a certified error bound: reject
        // near fits, large predictor cancellation, and under/overflow. The
        // multiplier covers accumulated rotation roundoff; no residual clipping.
        let perturbation = 64. * f64::EPSILON * self.rows as f64 * self.input_norm * beta_norm;
        if !residual_norm.is_normal()
            || !perturbation.is_finite()
            || perturbation > 1e-8 * residual_norm
        {
            return None;
        }
        let mut quadratic = CompensatedSum::default();
        for &r in &residuals[..width] {
            let z = r * inverse;
            quadratic.add_product(-0.5 * z, z);
        }
        let quadratic = quadratic.total();
        let sigma_score = (-2. * quadratic - self.rows as f64) * inverse;
        let mut density = CompensatedSum::default();
        density.add(quadratic);
        density.add_product(
            -sigma.ln() - 0.5 * std::f64::consts::TAU.ln(),
            self.rows as f64,
        );
        for (j, score) in scratch.iter_mut().enumerate() {
            let mut sum = CompensatedSum::default();
            for (i, &r) in residuals[..=j].iter().enumerate() {
                sum.add_product(self.root[i * width + j], (r * inverse) * inverse);
            }
            *score = sum.total();
        }
        let density = density.total();
        if !density.is_finite()
            || !sigma_score.is_finite()
            || scratch.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        g[..p].copy_from_slice(scratch);
        g[p] = sigma_score;
        Some(density)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modeling::regression::{NormalEval, Scores, kernels};

    fn check(data: RegressionData<f64>, beta: &[f64], sigma: f64, admitted: bool) {
        let compressed = CompressedNormal::new(&data).unwrap();
        let mut x = beta.to_vec();
        x.push(sigma);
        let mut actual = vec![17.; x.len()];
        let mut scratch = vec![0.; beta.len()];
        let candidate = compressed.evaluate(&x, &mut actual, &mut scratch);
        assert_eq!(candidate.is_some(), admitted);
        if let Some(value) = candidate {
            let design = kernels::Design::new(&data).unwrap();
            let mut scores = Scores::new(beta.len());
            let mut expected = vec![0.; x.len()];
            let reference = design
                .plan
                .dispatch(NormalEval {
                    data: &data,
                    design: &design,
                    broadcast: design.broadcast,
                    x: &x,
                    g: &mut expected,
                    scores: &mut scores,
                })
                .unwrap();
            for (a, b) in actual
                .into_iter()
                .chain([value])
                .zip(expected.into_iter().chain([reference]))
            {
                assert!((a - b).abs() <= 3e-12 * (1. + b.abs()), "{a} != {b}");
            }
        } else {
            assert!(actual.iter().all(|v| *v == 17.));
        }
    }

    #[test]
    fn compressed_values_scores_offsets_and_rank_deficiency_match_rows() {
        for n in [128, 1_000, 2_500, 5_000, 7_500, 10_000] {
            for p in [0, 1, 3, 8] {
                if n < 16 * (p + 1) {
                    continue;
                }
                let design: Vec<_> = (0..n)
                    .flat_map(|i| (0..p).map(move |j| ((i + 3 * j) % 17) as f64 / 8. - 1.))
                    .collect();
                let y = (0..n).map(|i| ((i * 7) % 23) as f64 / 7. - 1.).collect();
                let data = RegressionData::new(p, design, y)
                    .unwrap()
                    .with_offsets((0..n).map(|i| (i % 3) as f64 / 4.).collect())
                    .unwrap();
                let beta: Vec<_> = (0..p).map(|j| j as f64 / 5. - 0.2).collect();
                for sigma in [0.01, 1., 10.] {
                    check(data.clone(), &beta, sigma, true);
                }
                // Identical columns require no inverse or approximate rank test.
                let data = RegressionData::new(p, vec![1.; n * p], vec![0.125; n]).unwrap();
                check(data, &beta, 0.75, true);
            }
        }
    }

    #[test]
    fn near_fits_and_extreme_coordinates_retry_without_publishing_scores() {
        let data = RegressionData::new(1, vec![1.; 1_000], vec![1.; 1_000]).unwrap();
        for (beta, sigma) in [
            (1., 1.),
            (1. + 1e-12, 1.),
            (0., 1e-200),
            (0., 1e200),
            (f64::MAX, 1.),
            (f64::NAN, 1.),
            (0., -1.),
            (0., 0.),
        ] {
            check(data.clone(), &[beta], sigma, false);
        }
    }
}
