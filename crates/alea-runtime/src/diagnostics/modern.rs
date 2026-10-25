//! Rank/folded split R-hat, Geyer bulk/tail ESS and mean MCSE.
//!
//! Equal-length chains only; odd chains omit the middle draw when splitting.
//! Definitions follow Vehtari et al. (2021); executable ArviZ fixtures pin details.
//! Constant/undefined diagnostics are errors, never evidence for stopping.
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use statrs::distribution::{ContinuousCDF, Normal};
use std::sync::Arc;

/// A diagnostic could not be defined for the supplied observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Chains must be rectangular, with at least two chains and four draws.
    #[error("diagnostics require two equal-length chains with at least four draws")]
    Shape,
    /// Missing/nonfinite observations are not silently discarded.
    #[error("diagnostic input contains nonfinite values")]
    Nonfinite,
    /// Zero variance or numerically unrepresentable results.
    #[error("diagnostic is undefined for degenerate or unrepresentable draws")]
    Undefined,
}

/// Diagnostics for one explicitly selected scalar observable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
pub struct Summary {
    /// Maximum of rank-normalized split and folded R-hat.
    pub rhat: f64,
    /// Rank-normalized split R-hat before folding.
    pub rank_rhat: f64,
    /// Rank-normalized absolute deviation from the pooled split median.
    pub folded_rhat: f64,
    /// Cross-chain ESS of rank-normalized split draws.
    pub bulk_ess: f64,
    /// Minimum indicator ESS at pooled 5% and 95% quantiles.
    pub tail_ess: f64,
    /// ESS of the untransformed split draws (used for mean MCSE).
    pub mean_ess: f64,
    /// Pooled sample standard deviation divided by sqrt(mean ESS).
    pub mcse_mean: f64,
    /// Mean over all supplied draws (including an odd middle draw).
    pub mean: f64,
}

/// Computes modern diagnostics without truncating ragged chains or dropping NaNs.
/// Memory is O(chains × draws), autocovariances use O(n log n) FFTs.
///
/// # Errors
/// Returns [`Error`] on invalid shape, nonfinite input or undefined variance.
pub fn summarize(chains: &[&[f64]]) -> Result<Summary, Error> {
    let n = validate(chains)?;
    let original: Vec<f64> = chains.iter().flat_map(|x| x.iter().copied()).collect();
    let scale = original.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    if scale == 0.0 {
        return Err(Error::Undefined);
    }
    let scaled: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|x| x / scale).collect())
        .collect();
    let split = split_chains(&scaled);
    let raw_split: Vec<Vec<f64>> = chains
        .iter()
        .flat_map(|c| [c[..n / 2].to_vec(), c[n - n / 2..].to_vec()])
        .collect();
    let z = ranks(&raw_split);
    let rank_rhat = rhat(&z)?;
    let mut pooled: Vec<f64> = raw_split.iter().flatten().copied().collect();
    let median = quantile(&mut pooled, 0.5);
    let mut folded: Vec<Vec<f64>> = raw_split
        .iter()
        .map(|c| c.iter().map(|x| (x - median).abs()).collect())
        .collect();
    // Preserve floating-point ties in ordinary raw deviations. Scale only when
    // subtraction itself overflows; unconditional scaling can break median ties.
    if folded.iter().flatten().any(|x| !x.is_finite()) {
        folded = raw_split
            .iter()
            .map(|c| {
                c.iter()
                    .map(|x| (x / scale - median / scale).abs())
                    .collect()
            })
            .collect();
    }
    let folded_rhat = rhat(&ranks(&folded))?;
    let mut fft = Autocovariance::new(n / 2);
    let bulk_ess = ess(&z, &mut fft)?;
    let mean_ess = ess(&split, &mut fft)?;
    let mut sorted = original;
    let mut tail_ess = f64::INFINITY;
    for p in [0.05, 0.95] {
        let q = quantile(&mut sorted, p);
        let indicator: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.iter().map(|&x| f64::from(x <= q)).collect())
            .collect();
        tail_ess = tail_ess.min(ess(&split_chains(&indicator), &mut fft)?);
    }
    let all: Vec<f64> = scaled.into_iter().flatten().collect();
    let mean = average(&all);
    let mcse_mean = (variance(&all, mean) / mean_ess).sqrt() * scale;
    let result = Summary {
        rhat: rank_rhat.max(folded_rhat),
        rank_rhat,
        folded_rhat,
        bulk_ess,
        tail_ess,
        mean_ess,
        mcse_mean,
        mean: mean * scale,
    };
    if !result.mcse_mean.is_finite() || !result.mean.is_finite() {
        return Err(Error::Undefined);
    }
    Ok(result)
}

fn validate(chains: &[&[f64]]) -> Result<usize, Error> {
    let n = chains.first().map_or(0, |c| c.len());
    if chains.len() < 2 || n < 4 || chains.iter().any(|c| c.len() != n) {
        return Err(Error::Shape);
    }
    if chains.iter().any(|c| c.iter().any(|x| !x.is_finite())) {
        return Err(Error::Nonfinite);
    }
    Ok(n)
}

fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    chains
        .iter()
        .flat_map(|c| {
            let half = c.len() / 2;
            [c[..half].to_vec(), c[c.len() - half..].to_vec()]
        })
        .collect()
}
fn average(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn variance(x: &[f64], mean: f64) -> f64 {
    x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}
fn quantile(values: &mut [f64], p: f64) -> f64 {
    let at = (values.len() - 1) as f64 * p;
    let i = at.floor() as usize;
    // Only adjacent order statistics are needed. Total ordering retains the
    // sorted implementation's signed-zero and equal-value conventions.
    let (_, lower, above) = values.select_nth_unstable_by(i, f64::total_cmp);
    let lower = *lower;
    let upper = above
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(lower);
    if lower == upper {
        return lower;
    }
    // Convex form avoids overflowing the difference of extreme endpoints.
    (1.0 - at.fract()) * lower + at.fract() * upper
}
fn ranks(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = chains[0].len();
    let mut sorted: Vec<(usize, f64)> = chains.iter().flatten().copied().enumerate().collect();
    // Equal-value groups receive one rank, so their internal order is irrelevant.
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let mut out = vec![vec![0.0; n]; chains.len()];
    let normal = Normal::new(0.0, 1.0).expect("standard normal parameters are valid");
    let mut start = 0;
    while start < sorted.len() {
        let mut end = start + 1;
        // Equality is intentional: exact ties, including signed zeros, share a rank.
        while end < sorted.len() && sorted[end].1 == sorted[start].1 {
            end += 1;
        }
        let rank = (start + 1) as f64 / 2.0 + end as f64 / 2.0;
        let z = normal.inverse_cdf((rank - 0.375) / (sorted.len() as f64 + 0.25));
        for &(i, _) in &sorted[start..end] {
            out[i / n][i % n] = z;
        }
        start = end;
    }
    out
}
fn rhat(chains: &[Vec<f64>]) -> Result<f64, Error> {
    let n = chains[0].len() as f64;
    let means: Vec<_> = chains.iter().map(|c| average(c)).collect();
    let w = chains
        .iter()
        .zip(&means)
        .map(|(c, &m)| variance(c, m))
        .sum::<f64>()
        / chains.len() as f64;
    if w <= 0.0 {
        return Err(Error::Undefined);
    }
    finite_positive(((n - 1.0) / n + variance(&means, average(&means)) / w).sqrt())
}

fn finite_positive(x: f64) -> Result<f64, Error> {
    if x.is_finite() && x > 0.0 {
        Ok(x)
    } else {
        Err(Error::Undefined)
    }
}

// One caller-owned FFT plan and scratch set serves all four equal-sized ESS
// calculations in a summary. No process-global cache or thread state is needed.
struct Autocovariance {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex<f64>>,
    buffer: Vec<Complex<f64>>,
    acov: Vec<f64>,
}

impl Autocovariance {
    fn new(n: usize) -> Self {
        let size = (2 * n).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        let scratch = vec![
            Complex::default();
            forward
                .get_inplace_scratch_len()
                .max(inverse.get_inplace_scratch_len())
        ];
        Self {
            forward,
            inverse,
            scratch,
            buffer: vec![Complex::default(); size],
            acov: vec![0.0; n],
        }
    }

    // Biased autocovariance (denominator n); zero padding avoids circular wrap.
    fn compute(&mut self, chains: &[Vec<f64>]) -> &[f64] {
        let n = self.acov.len();
        let size = self.buffer.len();
        self.acov.fill(0.0);
        for c in chains {
            self.buffer.fill(Complex::default());
            let mean = average(c);
            for (v, &x) in self.buffer.iter_mut().zip(c) {
                v.re = x - mean;
            }
            self.forward
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
            for v in &mut self.buffer {
                *v = Complex::new(v.norm_sqr(), 0.0);
            }
            self.inverse
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
            for (a, v) in self.acov.iter_mut().zip(&self.buffer) {
                *a += v.re / size as f64 / n as f64 / chains.len() as f64;
            }
        }
        &self.acov
    }
}

fn ess(chains: &[Vec<f64>], fft: &mut Autocovariance) -> Result<f64, Error> {
    let n = chains[0].len();
    let total = (chains.len() * n) as f64;
    // ArviZ's constant-indicator convention matters for discrete quantiles.
    // Constant observables themselves still fail R-hat before reaching here.
    if chains.iter().flatten().all(|x| *x == chains[0][0]) {
        return Ok(total);
    }
    let acov = fft.compute(chains);
    let means: Vec<_> = chains.iter().map(|c| average(c)).collect();
    let within = acov[0] * n as f64 / (n - 1) as f64;
    let var_plus = acov[0] + variance(&means, average(&means));
    if var_plus <= 0.0 {
        return Err(Error::Undefined);
    }
    let correlation = |lag: usize| 1.0 - (within - acov[lag]) / var_plus;
    let mut rho = vec![0.0; n];
    rho[0] = 1.0;
    rho[1] = correlation(1);
    let (mut even, mut odd) = (1.0, rho[1]);
    let mut t = 1;
    while t < n.saturating_sub(3) && even + odd > 0.0 {
        even = correlation(t + 1);
        odd = correlation(t + 2);
        if even + odd >= 0.0 {
            rho[t + 1] = even;
            rho[t + 2] = odd;
        }
        t += 2;
    }
    let end = t - 1;
    if even > 0.0 {
        rho[end] = even;
    }
    for i in (1..end.saturating_sub(1)).step_by(2) {
        let prev = rho[i - 1] + rho[i];
        if rho[i + 1] + rho[i + 2] > prev {
            rho[i + 1] = prev / 2.0;
            rho[i + 2] = prev / 2.0;
        }
    }
    let tau = (-1.0 + 2.0 * rho[..end].iter().sum::<f64>() + rho[end]).max(1.0 / total.log10());
    finite_positive(total / tau)
}

/// E-BFMI of a contiguous per-transition energy sequence: mean squared first
/// differences divided by sample variance. Missing energies must not be bridged.
/// # Errors
/// Requires at least two finite, nonconstant energies with representable variance.
pub fn ebfmi(energy: &[f64]) -> Result<f64, Error> {
    if energy.len() < 2 {
        return Err(Error::Shape);
    }
    if energy.iter().any(|x| !x.is_finite()) {
        return Err(Error::Nonfinite);
    }
    let scale = energy.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    if scale == 0.0 {
        return Err(Error::Undefined);
    }
    let e: Vec<_> = energy.iter().map(|x| x / scale).collect();
    let variance = variance(&e, average(&e));
    let differences =
        e.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>() / (e.len() - 1) as f64;
    finite_positive(differences / variance)
}

#[cfg(test)]
mod tests {
    use super::{Autocovariance, quantile};

    #[test]
    fn selected_quantiles_match_sorted_interpolation_bit_for_bit() {
        for n in 1..=128 {
            for magnitude in [f64::from_bits(1), 1e-200, 1.0, 1e200, f64::MAX / 16.0] {
                let values: Vec<_> = (0..n)
                    .map(|i| match i % 9 {
                        0 => -0.0,
                        1 => 0.0,
                        _ => (((i * 37) % 15) as f64 - 7.0) * magnitude,
                    })
                    .collect();
                let mut sorted = values.clone();
                sorted.sort_by(f64::total_cmp);
                for p in [0.0, 0.05, 0.5, 0.95, 1.0] {
                    let at = (n - 1) as f64 * p;
                    let i = at.floor() as usize;
                    let j = (i + 1).min(n - 1);
                    let expected = if sorted[i] == sorted[j] {
                        sorted[i]
                    } else {
                        (1.0 - at.fract()) * sorted[i] + at.fract() * sorted[j]
                    };
                    assert_eq!(
                        quantile(&mut values.clone(), p).to_bits(),
                        expected.to_bits(),
                        "length={n}, magnitude={magnitude}, p={p}"
                    );
                }
            }
        }
    }

    #[test]
    fn fft_scratch_reuse_clears_previous_observations() {
        for n in [4, 7, 16, 33, 128] {
            let a = vec![(0..n).map(|i| (i as f64 * 0.17).sin()).collect()];
            let b = vec![(0..n).map(|i| i as f64).collect()];
            let mut fft = Autocovariance::new(n);
            let expected = fft.compute(&a).to_vec();
            let _ = fft.compute(&b);
            assert_eq!(fft.compute(&a), expected);
        }
    }
}
