//! Dense experimental SoftAbs Hessian metric.

use std::error::Error;

use alea_core::capability::{PositionMetric, PotentialHessianDerivative};

/// Validated SoftAbs spectral controls.
#[derive(Debug, Clone, Copy)]
pub struct SoftAbsOptions {
    alpha: f64,
    jacobi_sweeps: usize,
    jacobi_tolerance: f64,
}

impl SoftAbsOptions {
    /// Constructs a finite positive SoftAbs regularization and eigensolver budget.
    ///
    /// # Errors
    /// Rejects non-positive/non-finite values or a zero sweep budget.
    pub fn new(
        alpha: f64,
        jacobi_sweeps: usize,
        jacobi_tolerance: f64,
    ) -> Result<Self, SoftAbsConfigError> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(SoftAbsConfigError::Alpha);
        }
        if jacobi_sweeps == 0 {
            return Err(SoftAbsConfigError::Sweeps);
        }
        if !jacobi_tolerance.is_finite() || jacobi_tolerance <= 0.0 {
            return Err(SoftAbsConfigError::Tolerance);
        }
        Ok(Self {
            alpha,
            jacobi_sweeps,
            jacobi_tolerance,
        })
    }

    pub fn alpha(self) -> f64 {
        self.alpha
    }
}

/// Invalid SoftAbs configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SoftAbsConfigError {
    #[error("SoftAbs alpha must be finite and positive")]
    Alpha,
    #[error("SoftAbs Jacobi sweep count must be nonzero")]
    Sweeps,
    #[error("SoftAbs Jacobi tolerance must be finite and positive")]
    Tolerance,
    #[error("SoftAbs workspace size overflow")]
    SizeOverflow,
}

/// SoftAbs backend or numerical failure.
#[derive(Debug, thiserror::Error)]
pub enum SoftAbsError<E: Error + 'static> {
    #[error("SoftAbs tensor backend failed: {0}")]
    Backend(#[source] E),
    #[error("SoftAbs Hessian tensors contain a non-finite value")]
    NonFiniteTensor,
    #[error("SoftAbs potential Hessian is asymmetric at ({row}, {column})")]
    AsymmetricHessian { row: usize, column: usize },
    #[error("SoftAbs third derivative is inconsistent at ({first}, {second}, {third})")]
    AsymmetricThirdDerivative {
        first: usize,
        second: usize,
        third: usize,
    },
    #[error("SoftAbs symmetric eigensolver did not converge; residual {residual:e}")]
    EigenNotConverged { residual: f64 },
    #[error("SoftAbs spectral map produced a non-finite value")]
    NonFiniteMap,
}

/// SoftAbs transform of a target's dense potential Hessian.
///
/// The target must supply the potential Hessian and all third derivatives
/// explicitly. This adapter uses only caller-owned scratch during evaluation.
#[derive(Debug)]
pub struct SoftAbsMetric<'a, T: PotentialHessianDerivative + ?Sized> {
    target: &'a T,
    options: SoftAbsOptions,
    workspace_len: usize,
}

impl<'a, T: PotentialHessianDerivative + ?Sized> SoftAbsMetric<'a, T> {
    /// Binds the Hessian capability and precomputes its exact scratch size.
    ///
    /// # Errors
    /// Returns a size-overflow error for unsupported dimensions.
    pub fn new(target: &'a T, options: SoftAbsOptions) -> Result<Self, SoftAbsConfigError> {
        let d = target.dimension();
        let addressable = |len: &usize| *len <= isize::MAX as usize / size_of::<f64>();
        let d2 = d
            .checked_mul(d)
            .filter(addressable)
            .ok_or(SoftAbsConfigError::SizeOverflow)?;
        let d3 = d2
            .checked_mul(d)
            .filter(addressable)
            .ok_or(SoftAbsConfigError::SizeOverflow)?;
        let workspace_len = d3
            .checked_add(d2.checked_mul(3).ok_or(SoftAbsConfigError::SizeOverflow)?)
            .and_then(|value| value.checked_add(d))
            .filter(addressable)
            .ok_or(SoftAbsConfigError::SizeOverflow)?;
        Ok(Self {
            target,
            options,
            workspace_len,
        })
    }
}

impl<T: PotentialHessianDerivative + ?Sized> PositionMetric for SoftAbsMetric<'_, T> {
    type Error = SoftAbsError<T::Error>;

    fn dimension(&self) -> usize {
        self.target.dimension()
    }

    fn workspace_len(&self) -> usize {
        self.workspace_len
    }

    fn metric_and_derivatives(
        &self,
        position: &[f64],
        metric: &mut [f64],
        derivatives: &mut [f64],
        workspace: &mut [f64],
    ) -> Result<(), Self::Error> {
        let d = self.dimension();
        let d2 = d * d;
        let d3 = d2 * d;
        let (hessian, rest) = workspace.split_at_mut(d2);
        let (hessian_derivatives, rest) = rest.split_at_mut(d3);
        let (eigenvectors, rest) = rest.split_at_mut(d2);
        let (temporary, eigenvalues) = rest.split_at_mut(d2);
        debug_assert_eq!(eigenvalues.len(), d);

        self.target
            .potential_hessian_and_derivatives(position, hessian, hessian_derivatives)
            .map_err(SoftAbsError::Backend)?;
        if hessian
            .iter()
            .chain(hessian_derivatives.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SoftAbsError::NonFiniteTensor);
        }
        canonicalize_symmetric(hessian, d)
            .map_err(|(row, column)| SoftAbsError::AsymmetricHessian { row, column })?;
        canonicalize_third_derivative(hessian_derivatives, d).map_err(
            |(first, second, third)| SoftAbsError::AsymmetricThirdDerivative {
                first,
                second,
                third,
            },
        )?;
        let residual = jacobi_eigen(
            hessian,
            eigenvectors,
            eigenvalues,
            d,
            self.options.jacobi_sweeps,
            self.options.jacobi_tolerance,
        );
        if !residual.is_finite() || residual > self.options.jacobi_tolerance {
            return Err(SoftAbsError::EigenNotConverged { residual });
        }

        metric.fill(0.0);
        for eigen in 0..d {
            let mapped = soft_abs(eigenvalues[eigen], self.options.alpha);
            if !mapped.is_finite() || mapped <= 0.0 {
                return Err(SoftAbsError::NonFiniteMap);
            }
            for row in 0..d {
                for column in 0..d {
                    metric[row * d + column] +=
                        eigenvectors[row * d + eigen] * mapped * eigenvectors[column * d + eigen];
                }
            }
        }

        derivatives.fill(0.0);
        for coordinate in 0..d {
            let source = &hessian_derivatives[coordinate * d2..(coordinate + 1) * d2];
            // U^T (dH/dq_k), then apply U on the right. Keeping these as
            // matrix products makes each coordinate O(d^3), rather than an
            // element-wise O(d^4) contraction.
            for i in 0..d {
                for column in 0..d {
                    let mut value = 0.0;
                    for row in 0..d {
                        value += eigenvectors[row * d + i] * source[row * d + column];
                    }
                    hessian[i * d + column] = value;
                }
            }
            for i in 0..d {
                for j in 0..d {
                    let transformed = (0..d)
                        .map(|column| hessian[i * d + column] * eigenvectors[column * d + j])
                        .sum::<f64>();
                    temporary[i * d + j] = soft_abs_divided_difference(
                        eigenvalues[i],
                        eigenvalues[j],
                        self.options.alpha,
                    ) * transformed;
                }
            }
            // U B U^T, again as two matrix products.
            for row in 0..d {
                for j in 0..d {
                    hessian[row * d + j] = (0..d)
                        .map(|i| eigenvectors[row * d + i] * temporary[i * d + j])
                        .sum();
                }
            }
            let destination = &mut derivatives[coordinate * d2..(coordinate + 1) * d2];
            for row in 0..d {
                for column in 0..d {
                    destination[row * d + column] = (0..d)
                        .map(|j| hessian[row * d + j] * eigenvectors[column * d + j])
                        .sum();
                }
            }
        }
        if metric
            .iter()
            .chain(derivatives.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SoftAbsError::NonFiniteMap);
        }
        Ok(())
    }
}

fn symmetric_pair(left: f64, right: f64) -> bool {
    let scale = 1.0 + left.abs().max(right.abs());
    (left - right).abs() <= 64.0 * f64::EPSILON * scale
}

fn canonicalize_symmetric(matrix: &mut [f64], dimension: usize) -> Result<(), (usize, usize)> {
    for row in 0..dimension {
        for column in 0..row {
            let lower = matrix[row * dimension + column];
            let upper = matrix[column * dimension + row];
            if !symmetric_pair(lower, upper) {
                return Err((row, column));
            }
            let average = 0.5 * (lower + upper);
            matrix[row * dimension + column] = average;
            matrix[column * dimension + row] = average;
        }
    }
    Ok(())
}

fn canonicalize_third_derivative(
    tensor: &mut [f64],
    dimension: usize,
) -> Result<(), (usize, usize, usize)> {
    let index = |a: usize, b: usize, c: usize| a * dimension * dimension + b * dimension + c;
    for first in 0..dimension {
        for second in first..dimension {
            for third in second..dimension {
                let permutations = [
                    index(first, second, third),
                    index(first, third, second),
                    index(second, first, third),
                    index(second, third, first),
                    index(third, first, second),
                    index(third, second, first),
                ];
                let reference = tensor[permutations[0]];
                if permutations
                    .iter()
                    .any(|position| !symmetric_pair(reference, tensor[*position]))
                {
                    return Err((first, second, third));
                }
                let average = permutations
                    .iter()
                    .map(|position| tensor[*position])
                    .sum::<f64>()
                    / permutations.len() as f64;
                for position in permutations {
                    tensor[position] = average;
                }
            }
        }
    }
    Ok(())
}

fn soft_abs(value: f64, alpha: f64) -> f64 {
    let x = alpha * value;
    if x.abs() < 1.0e-4 {
        let x2 = x * x;
        (1.0 + x2 / 3.0 - x2 * x2 / 45.0) / alpha
    } else if x.abs() > 20.0 {
        value.abs()
    } else {
        value / x.tanh()
    }
}

fn soft_abs_derivative(value: f64, alpha: f64) -> f64 {
    let x = alpha * value;
    if x.abs() < 1.0e-4 {
        2.0 * x / 3.0 - 4.0 * x * x * x / 45.0
    } else if x.abs() > 20.0 {
        value.signum()
    } else {
        let sinh = x.sinh();
        1.0 / x.tanh() - x / (sinh * sinh)
    }
}

fn soft_abs_divided_difference(a: f64, b: f64, alpha: f64) -> f64 {
    let difference = a - b;
    let scale = 1.0 + a.abs().max(b.abs());
    if difference.abs() <= f64::EPSILON.sqrt() * scale {
        soft_abs_derivative(0.5 * (a + b), alpha)
    } else {
        (soft_abs(a, alpha) - soft_abs(b, alpha)) / difference
    }
}

fn jacobi_eigen(
    matrix: &mut [f64],
    eigenvectors: &mut [f64],
    eigenvalues: &mut [f64],
    dimension: usize,
    sweeps: usize,
    tolerance: f64,
) -> f64 {
    eigenvectors.fill(0.0);
    for index in 0..dimension {
        eigenvectors[index * dimension + index] = 1.0;
    }
    let mut residual = maximum_off_diagonal(matrix, dimension);
    for _ in 0..sweeps {
        if residual <= tolerance {
            break;
        }
        for p in 0..dimension {
            for q in (p + 1)..dimension {
                let apq = matrix[p * dimension + q];
                if apq.abs() <= tolerance {
                    continue;
                }
                let app = matrix[p * dimension + p];
                let aqq = matrix[q * dimension + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cosine = 1.0 / (1.0 + t * t).sqrt();
                let sine = t * cosine;
                for k in 0..dimension {
                    if k != p && k != q {
                        let akp = matrix[k * dimension + p];
                        let akq = matrix[k * dimension + q];
                        let next_kp = cosine * akp - sine * akq;
                        let next_kq = sine * akp + cosine * akq;
                        matrix[k * dimension + p] = next_kp;
                        matrix[p * dimension + k] = next_kp;
                        matrix[k * dimension + q] = next_kq;
                        matrix[q * dimension + k] = next_kq;
                    }
                    let vkp = eigenvectors[k * dimension + p];
                    let vkq = eigenvectors[k * dimension + q];
                    eigenvectors[k * dimension + p] = cosine * vkp - sine * vkq;
                    eigenvectors[k * dimension + q] = sine * vkp + cosine * vkq;
                }
                matrix[p * dimension + p] = app - t * apq;
                matrix[q * dimension + q] = aqq + t * apq;
                matrix[p * dimension + q] = 0.0;
                matrix[q * dimension + p] = 0.0;
            }
        }
        residual = maximum_off_diagonal(matrix, dimension);
    }
    for index in 0..dimension {
        eigenvalues[index] = matrix[index * dimension + index];
    }
    residual
}

fn maximum_off_diagonal(matrix: &[f64], dimension: usize) -> f64 {
    let scale = (0..dimension)
        .map(|index| matrix[index * dimension + index].abs())
        .fold(1.0, f64::max);
    let largest = (0..dimension)
        .flat_map(|row| ((row + 1)..dimension).map(move |column| (row, column)))
        .map(|(row, column)| matrix[row * dimension + column].abs())
        .fold(0.0, f64::max);
    largest / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_symmetry_is_checked_and_roundoff_is_canonicalized() {
        let mut hessian = [2.0, 0.25, 0.25 + f64::EPSILON, 3.0];
        canonicalize_symmetric(&mut hessian, 2).unwrap();
        assert_eq!(hessian[1].to_bits(), hessian[2].to_bits());

        let mut third = [1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0];
        third[4] += f64::EPSILON;
        canonicalize_third_derivative(&mut third, 2).unwrap();
        assert_eq!(third[1].to_bits(), third[2].to_bits());
        assert_eq!(third[1].to_bits(), third[4].to_bits());

        third[4] += 0.1;
        assert_eq!(canonicalize_third_derivative(&mut third, 2), Err((0, 0, 1)));
    }
}
