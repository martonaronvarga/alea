//! Limited-memory inverse-BFGS actions, independent of statistical targets.
#![deny(missing_docs)]
use thiserror::Error;

/// An invalid secant history or nonrepresentable inverse-BFGS action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LbfgsError {
    /// Vector, output or secant dimensions do not agree.
    #[error("inconsistent inverse-BFGS dimensions")]
    Dimension,
    /// Inputs, curvature or arithmetic do not define a finite positive update.
    #[error("invalid inverse-BFGS curvature or arithmetic")]
    Numerical,
}

/// Applies the inverse-BFGS approximation to `vector` using the two-loop recursion.
/// Secants are ordered oldest first: `s = x_new-x_old`, `y = grad_new-grad_old`
/// for a **minimized** objective. With no history the initial matrix is identity;
/// otherwise its scale is `(s_last dot y_last)/(y_last dot y_last)`.
///
/// This is a numerical preconditioner, not a statistical covariance estimate.
/// Output may be partial after an arithmetic failure; shape errors do not write.
///
/// # Errors
/// Rejects inconsistent shapes, nonfinite inputs, nonpositive curvature or
/// nonrepresentable arithmetic. Callers should filter weak secant pairs before use.
pub fn lbfgs_inverse_product(
    vector: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    output: &mut [f64],
) -> Result<(), LbfgsError> {
    if output.len() != vector.len()
        || s_history.len() != y_history.len()
        || s_history
            .iter()
            .chain(y_history)
            .any(|v| v.len() != vector.len())
    {
        return Err(LbfgsError::Dimension);
    }
    if vector
        .iter()
        .chain(s_history.iter().flatten())
        .chain(y_history.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(LbfgsError::Numerical);
    }
    output.copy_from_slice(vector);
    let mut alphas = vec![0.0; s_history.len()];
    for index in (0..s_history.len()).rev() {
        let sy = dot(&s_history[index], &y_history[index]);
        if !sy.is_finite() || sy <= 0.0 {
            return Err(LbfgsError::Numerical);
        }
        alphas[index] = dot(&s_history[index], output) / sy;
        for (value, &y) in output.iter_mut().zip(&y_history[index]) {
            *value -= alphas[index] * y;
        }
    }
    let gamma = if let (Some(s), Some(y)) = (s_history.last(), y_history.last()) {
        dot(s, y) / dot(y, y)
    } else {
        1.0
    };
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(LbfgsError::Numerical);
    }
    for value in output.iter_mut() {
        *value *= gamma;
    }
    for index in 0..s_history.len() {
        let sy = dot(&s_history[index], &y_history[index]);
        let beta = dot(&y_history[index], output) / sy;
        for (value, &s) in output.iter_mut().zip(&s_history[index]) {
            *value += s * (alphas[index] - beta);
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(LbfgsError::Numerical);
    }
    Ok(())
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
