use super::{Kind, Transform, TransformError, size};

impl Transform {
    fn matrix(n: usize, correlation: bool, covariance: bool) -> Result<Self, TransformError> {
        if n == 0 {
            return Err(TransformError::InvalidDimension);
        }
        let output = size(n.checked_mul(n).ok_or(TransformError::InvalidDimension)?)?;
        let scratch = size(
            output
                .checked_mul(2)
                .ok_or(TransformError::InvalidDimension)?,
        )?;
        let input = if correlation {
            n * (n - 1) / 2
        } else {
            output / 2 + n.div_ceil(2)
        };
        Ok(Self {
            kind: Kind::Matrix {
                n,
                correlation,
                covariance,
            },
            input,
            output,
            scratch,
        })
    }
    /// Dense lower-triangular Cholesky factor, positive diagonal, packed inputs.
    pub fn cholesky_covariance(n: usize) -> Result<Self, TransformError> {
        Self::matrix(n, false, false)
    }
    /// Dense lower-triangular factor with unit row norms and positive diagonal.
    pub fn cholesky_correlation(n: usize) -> Result<Self, TransformError> {
        Self::matrix(n, true, false)
    }
    /// Symmetric positive-definite covariance matrix, formed as L Lᵀ.
    pub fn covariance(n: usize) -> Result<Self, TransformError> {
        Self::matrix(n, false, true)
    }
    /// Symmetric positive-definite correlation matrix, formed from a Cholesky chart.
    pub fn correlation(n: usize) -> Result<Self, TransformError> {
        Self::matrix(n, true, true)
    }

    fn matrix_shape(&self) -> (usize, bool, bool) {
        match self.kind {
            Kind::Matrix {
                n,
                correlation,
                covariance,
            } => (n, correlation, covariance),
            _ => unreachable!("matrix methods are dispatched only for matrix transforms"),
        }
    }

    fn factor(&self, q: &[f64], l: &mut [f64]) -> Result<f64, TransformError> {
        let (n, correlation, covariance) = self.matrix_shape();
        l.fill(0.0);
        let mut k = 0;
        let mut jac = if covariance && !correlation {
            n as f64 * std::f64::consts::LN_2
        } else {
            0.0
        };
        for i in 0..n {
            let mut remaining = 1.0;
            for j in 0..i {
                if correlation {
                    let z = q[k].tanh();
                    let s2 = (1.0 - z) * (1.0 + z);
                    if s2 <= 0.0 {
                        return Err(TransformError::PrecisionLoss { index: k });
                    }
                    l[i * n + j] = remaining * z;
                    remaining *= s2.sqrt();
                    let coefficient = if covariance { n - j } else { i - j + 1 };
                    jac += 0.5 * coefficient as f64 * s2.ln();
                } else {
                    l[i * n + j] = q[k];
                }
                k += 1;
            }
            if correlation {
                l[i * n + i] = remaining;
            } else {
                l[i * n + i] = q[k].exp();
                jac += if covariance {
                    (n + 1 - i) as f64 * q[k]
                } else {
                    q[k]
                };
                k += 1;
            }
            if l[i * n + i] <= 0.0 {
                return Err(TransformError::PrecisionLoss { index: i * n + i });
            }
        }
        super::output_finite(l)?;
        super::log_jacobian(jac)
    }

    pub(super) fn matrix_constrain(
        &self,
        q: &[f64],
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<f64, TransformError> {
        let (n, correlation, covariance) = self.matrix_shape();
        if !covariance {
            return self.factor(q, x);
        }
        let (l, rest) = scratch.split_at_mut(n * n);
        let jac = self.factor(q, l)?;
        for i in 0..n {
            for j in 0..=i {
                let v = if correlation && i == j {
                    1.0
                } else {
                    (0..=j).map(|k| l[i * n + k] * l[j * n + k]).sum()
                };
                x[i * n + j] = v;
                x[j * n + i] = v;
            }
        }
        super::output_finite(x)?;
        // A finite LLᵀ can round to a singular matrix. Do not advertise that
        // rounded result as positive definite. This check is deliberately O(n³).
        cholesky(x, &mut rest[..n * n], n)?;
        Ok(jac)
    }

    pub(super) fn matrix_unconstrain(
        &self,
        x: &[f64],
        q: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        let (n, correlation, covariance) = self.matrix_shape();
        let l = &mut scratch[..n * n];
        if covariance {
            for i in 0..n {
                if correlation && (x[i * n + i] - 1.0).abs() > 32.0 * f64::EPSILON * n as f64 {
                    return Err(TransformError::Domain { index: i * n + i });
                }
                for j in 0..i {
                    if x[i * n + j] != x[j * n + i] {
                        return Err(TransformError::Domain { index: i * n + j });
                    }
                }
            }
            cholesky(x, l, n)?;
        } else {
            l.copy_from_slice(x);
            for i in 0..n {
                for j in i + 1..n {
                    if l[i * n + j] != 0.0 {
                        return Err(TransformError::Domain { index: i * n + j });
                    }
                }
            }
        }
        let mut k = 0;
        for i in 0..n {
            if l[i * n + i] <= 0.0 {
                return Err(TransformError::Domain { index: i * n + i });
            }
            if correlation {
                let norm: f64 = (0..=i).map(|j| l[i * n + j].powi(2)).sum();
                if (norm - 1.0).abs() > 64.0 * f64::EPSILON * n as f64 {
                    return Err(TransformError::Domain { index: i * n + i });
                }
                let mut remaining = 1.0;
                for j in 0..i {
                    let z = l[i * n + j] / remaining;
                    if z.abs() >= 1.0 {
                        return Err(TransformError::Domain { index: i * n + j });
                    }
                    q[k] = 0.5 * (z.ln_1p() - (-z).ln_1p());
                    remaining *= ((1.0 - z) * (1.0 + z)).sqrt();
                    k += 1;
                }
            } else {
                for j in 0..i {
                    q[k] = l[i * n + j];
                    k += 1;
                }
                q[k] = l[i * n + i].ln();
                k += 1;
            }
        }
        Ok(())
    }

    pub(super) fn matrix_pullback(
        &self,
        q: &[f64],
        gx: &[f64],
        gq: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        let (n, correlation, covariance) = self.matrix_shape();
        let (l, rest) = scratch.split_at_mut(n * n);
        self.factor(q, l)?;
        let gl = &mut rest[..n * n];
        if covariance {
            for i in 0..n {
                for j in 0..=i {
                    gl[i * n + j] = (j..n)
                        .map(|k| {
                            if correlation && k == i {
                                0.0
                            } else {
                                (gx[i * n + k] + gx[k * n + i]) * l[k * n + j]
                            }
                        })
                        .sum();
                }
            }
        } else {
            gl.copy_from_slice(gx);
        }
        let mut k = 0;
        for i in 0..n {
            if correlation {
                let start = k;
                let mut remaining = 1.0;
                for j in 0..i {
                    gq[k] = remaining;
                    let z = q[k].tanh();
                    remaining *= ((1.0 - z) * (1.0 + z)).sqrt();
                    k += 1;
                    debug_assert!(j < n);
                }
                let mut adjoint = gl[i * n + i];
                for j in (0..i).rev() {
                    let index = start + j;
                    let z = q[index].tanh();
                    let s2 = (1.0 - z) * (1.0 + z);
                    let s = s2.sqrt();
                    let coefficient = if covariance { n - j } else { i - j + 1 };
                    gq[index] =
                        gq[index] * (gl[i * n + j] * s2 - adjoint * z * s) - coefficient as f64 * z;
                    adjoint = gl[i * n + j] * z + adjoint * s;
                }
            } else {
                for j in 0..i {
                    gq[k] = gl[i * n + j];
                    k += 1;
                }
                gq[k] = gl[i * n + i] * l[i * n + i]
                    + if covariance { (n + 1 - i) as f64 } else { 1.0 };
                k += 1;
            }
        }
        Ok(())
    }
}

fn cholesky(x: &[f64], l: &mut [f64], n: usize) -> Result<(), TransformError> {
    l.fill(0.0);
    for i in 0..n {
        for j in 0..=i {
            let residual = x[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            l[i * n + j] = if i == j {
                if residual <= 0.0 || !residual.is_finite() {
                    return Err(TransformError::Domain { index: i * n + i });
                }
                residual.sqrt()
            } else {
                residual / l[j * n + j]
            };
        }
    }
    super::output_finite(l)
}
