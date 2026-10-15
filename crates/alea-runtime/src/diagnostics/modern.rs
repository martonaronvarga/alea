//! Rank/folded split R-hat, Geyer bulk/tail ESS and mean MCSE.
//!
//! Equal-length chains only; odd chains omit the middle draw when splitting.
//! Definitions follow Vehtari et al. (2021); executable ArviZ fixtures pin details.
//! Constant/undefined diagnostics are errors, never evidence for stopping.
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::distribution::{ContinuousCDF, Normal};

/// A diagnostic could not be defined for the supplied observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
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
    pooled.sort_by(f64::total_cmp);
    let median = quantile(&pooled, 0.5);
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
    let bulk_ess = ess(&z)?;
    let mean_ess = ess(&split)?;
    let mut sorted = original;
    sorted.sort_by(f64::total_cmp);
    let mut tail_ess = f64::INFINITY;
    for p in [0.05, 0.95] {
        let q = quantile(&sorted, p);
        let indicator: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.iter().map(|&x| f64::from(x <= q)).collect())
            .collect();
        tail_ess = tail_ess.min(ess(&split_chains(&indicator))?);
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
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let at = (sorted.len() - 1) as f64 * p;
    let i = at.floor() as usize;
    let j = (i + 1).min(sorted.len() - 1);
    if sorted[i] == sorted[j] {
        return sorted[i];
    }
    // Convex form avoids overflowing the difference of extreme endpoints.
    (1.0 - at.fract()) * sorted[i] + at.fract() * sorted[j]
}
fn ranks(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = chains[0].len();
    let mut sorted: Vec<(usize, f64)> = chains.iter().flatten().copied().enumerate().collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
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

// Biased autocovariance (denominator n) with zero padding avoids circular wrap.
fn autocovariances(chains: &[Vec<f64>]) -> Vec<f64> {
    let n = chains[0].len();
    let size = (2 * n).next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(size);
    let inverse = planner.plan_fft_inverse(size);
    let mut scratch = vec![
        Complex::default();
        forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len())
    ];
    let mut buffer = vec![Complex::default(); size];
    let mut acov = vec![0.0; n];
    for c in chains {
        buffer.fill(Complex::default());
        let mean = average(c);
        for (v, &x) in buffer.iter_mut().zip(c) {
            v.re = x - mean;
        }
        forward.process_with_scratch(&mut buffer, &mut scratch);
        for v in &mut buffer {
            *v = Complex::new(v.norm_sqr(), 0.0);
        }
        inverse.process_with_scratch(&mut buffer, &mut scratch);
        for (a, v) in acov.iter_mut().zip(&buffer) {
            *a += v.re / size as f64 / n as f64 / chains.len() as f64;
        }
    }
    acov
}

fn ess(chains: &[Vec<f64>]) -> Result<f64, Error> {
    let n = chains[0].len();
    let total = (chains.len() * n) as f64;
    // ArviZ's constant-indicator convention matters for discrete quantiles.
    // Constant observables themselves still fail R-hat before reaching here.
    if chains.iter().flatten().all(|x| *x == chains[0][0]) {
        return Ok(total);
    }
    let acov = autocovariances(chains);
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
