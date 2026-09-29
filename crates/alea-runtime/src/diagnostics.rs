//! Classical diagnostics retained from the pre-migration runtime.
//! Neither estimator is rank-normalized; these are not M6's bulk/tail ESS or modern R-hat.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiagnosticError {
    #[error("diagnostic needs finite, nonconstant draws and enough observations")]
    InvalidDraws,
    #[error("split R-hat needs at least two equal-length chains of length four or more")]
    Shape,
}

#[inline]
pub fn acceptance_rate(accepted: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        accepted as f64 / total as f64
    }
}

pub fn autocorrelation_ess(values: &[f64]) -> Result<f64, DiagnosticError> {
    let n = values.len();
    if n < 4 || values.iter().any(|x| !x.is_finite()) {
        return Err(DiagnosticError::InvalidDraws);
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let mut centered = vec![0.0; n];
    for (i, v) in values.iter().enumerate() {
        centered[i] = *v - mean;
    }

    let var = centered.iter().map(|x| x * x).sum::<f64>() / (n - 1) as f64;
    if !var.is_finite() || var <= 0.0 {
        return Err(DiagnosticError::InvalidDraws);
    }

    let mut tau = 1.0;
    let max_lag = (n / 2).min(1000);
    for lag in 1..max_lag {
        let mut acov = 0.0;
        for i in 0..(n - lag) {
            acov += centered[i] * centered[i + lag];
        }
        acov /= (n - lag) as f64;
        let rho = acov / var;
        if rho <= 0.0 || !rho.is_finite() {
            break;
        }
        tau += 2.0 * rho;
    }

    Ok((n as f64 / tau).max(1.0))
}

pub fn classical_split_rhat(chains: &[&[f64]]) -> Result<f64, DiagnosticError> {
    let m = chains.len();
    if m < 2 {
        return Err(DiagnosticError::Shape);
    }
    let n = chains.iter().map(|c| c.len()).min().unwrap_or(0);
    if n < 4 {
        return Err(DiagnosticError::Shape);
    }

    if chains.iter().any(|c| c.len() != n) {
        return Err(DiagnosticError::Shape);
    }
    if chains.iter().any(|c| c.iter().any(|x| !x.is_finite())) {
        return Err(DiagnosticError::InvalidDraws);
    }
    let half = n / 2;
    let mut split = Vec::with_capacity(2 * m);
    for c in chains {
        split.push(&c[..half]);
        split.push(&c[half..(2 * half)]);
    }

    let m2 = split.len() as f64;
    let n2 = half as f64;

    let means: Vec<f64> = split.iter().map(|c| c.iter().sum::<f64>() / n2).collect();
    let mean_all = means.iter().sum::<f64>() / m2;

    let b = n2 * means.iter().map(|mu| (mu - mean_all).powi(2)).sum::<f64>() / (m2 - 1.0);

    let w = split
        .iter()
        .map(|c| {
            let mu = c.iter().sum::<f64>() / n2;
            c.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n2 - 1.0)
        })
        .sum::<f64>()
        / m2;

    if !w.is_finite() || !b.is_finite() || w <= 0.0 {
        return Err(DiagnosticError::InvalidDraws);
    }

    let var_hat = ((n2 - 1.0) / n2) * w + (b / n2);
    let result = (var_hat / w).sqrt();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(DiagnosticError::InvalidDraws)
    }
}
