//! Allocation-free maps from unconstrained coordinates to constrained parameters.
//!
//! Matrices are **row-major**; inputs pack the lower triangle row by row.
//! Matrix Jacobians use independent lower-triangle coordinates, simplex Jacobians
//! use its first K-1 coordinates, and unit vectors use surface measure. The latter
//! use a stereographic chart (not Stan's redundant radial parameterization).
//! Floating-point boundary collapse is an error, never silently clamped.
//! Output contents after a numerical error are unspecified; shape errors precede writes.
#![forbid(unsafe_code)]

mod layout;
mod matrix;
pub use crate::error::TransformError;
pub use layout::{ParameterBlock, ParameterLayout};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Identity,
    Lower(f64),
    Upper(f64),
    Interval {
        lower: f64,
        upper: f64,
        width: f64,
    },
    Ordered {
        positive: bool,
    },
    Simplex,
    UnitVector,
    Matrix {
        n: usize,
        correlation: bool,
        covariance: bool,
    },
}

/// A validated transform shape. Construction is cheap and performs no allocation.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    kind: Kind,
    input: usize,
    output: usize,
    scratch: usize,
}

fn size(n: usize) -> Result<usize, TransformError> {
    if n > isize::MAX as usize / size_of::<f64>() {
        Err(TransformError::InvalidDimension)
    } else {
        Ok(n)
    }
}

pub(super) fn shape(expected: usize, actual: usize) -> Result<(), TransformError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TransformError::Dimension { expected, actual })
    }
}

pub(super) fn finite(x: &[f64]) -> Result<(), TransformError> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(TransformError::NonFiniteInput { index }),
        None => Ok(()),
    }
}

pub(super) fn output_finite(x: &[f64]) -> Result<(), TransformError> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(TransformError::NonFiniteOutput { index }),
        None => Ok(()),
    }
}

pub(super) fn log_jacobian(j: f64) -> Result<f64, TransformError> {
    if j.is_finite() {
        Ok(j)
    } else {
        Err(TransformError::NonFiniteJacobian)
    }
}

fn sigmoid(x: f64) -> (f64, f64, f64) {
    let t = (-x.abs()).exp();
    let small = t / (1.0 + t);
    let large = 1.0 / (1.0 + t);
    let log_product = -x.abs() - 2.0 * t.ln_1p();
    if x >= 0.0 {
        (large, small, log_product)
    } else {
        (small, large, log_product)
    }
}

impl Transform {
    fn vector(kind: Kind, n: usize) -> Result<Self, TransformError> {
        Ok(Self {
            kind,
            input: size(n)?,
            output: n,
            scratch: 0,
        })
    }

    /// Independent identity coordinates; an empty block is allowed.
    pub fn identity(n: usize) -> Result<Self, TransformError> {
        Self::vector(Kind::Identity, n)
    }
    /// Independent strictly positive coordinates, using exp.
    pub fn positive(n: usize) -> Result<Self, TransformError> {
        Self::lower(n, 0.0)
    }
    /// Independent coordinates strictly above a finite lower bound.
    pub fn lower(n: usize, bound: f64) -> Result<Self, TransformError> {
        if !bound.is_finite() {
            return Err(TransformError::InvalidBounds { index: 0 });
        }
        Self::vector(Kind::Lower(bound), n)
    }
    /// Independent coordinates strictly below a finite upper bound.
    pub fn upper(n: usize, bound: f64) -> Result<Self, TransformError> {
        if !bound.is_finite() {
            return Err(TransformError::InvalidBounds { index: 0 });
        }
        Self::vector(Kind::Upper(bound), n)
    }
    /// Logistic map to an open interval with a finite, representable width.
    pub fn interval(n: usize, lower: f64, upper: f64) -> Result<Self, TransformError> {
        let width = upper - lower;
        if !lower.is_finite()
            || !upper.is_finite()
            || !width.is_finite()
            || lower.next_up() >= upper
        {
            return Err(TransformError::InvalidBounds { index: 0 });
        }
        Self::vector(
            Kind::Interval {
                lower,
                upper,
                width,
            },
            n,
        )
    }
    /// Strictly increasing coordinates with an unconstrained first element.
    pub fn ordered(n: usize) -> Result<Self, TransformError> {
        Self::vector(Kind::Ordered { positive: false }, n)
    }
    /// Strictly positive, increasing coordinates.
    pub fn positive_ordered(n: usize) -> Result<Self, TransformError> {
        Self::vector(Kind::Ordered { positive: true }, n)
    }
    /// Centered stick breaking: K-1 inputs, K strictly positive outputs summing to one.
    pub fn simplex(n: usize) -> Result<Self, TransformError> {
        if n == 0 {
            return Err(TransformError::InvalidDimension);
        }
        Ok(Self {
            kind: Kind::Simplex,
            input: n - 1,
            output: size(n)?,
            scratch: n - 1,
        })
    }
    /// Stereographic R^(K-1) chart of the unit sphere, excluding its north pole.
    pub fn unit_vector(n: usize) -> Result<Self, TransformError> {
        if n < 2 {
            return Err(TransformError::InvalidDimension);
        }
        Ok(Self {
            kind: Kind::UnitVector,
            input: n - 1,
            output: size(n)?,
            scratch: 0,
        })
    }

    pub fn unconstrained_dimension(&self) -> usize {
        self.input
    }
    pub fn constrained_dimension(&self) -> usize {
        self.output
    }
    /// Minimum caller-owned scratch length, reused across calls.
    pub fn scratch_dimension(&self) -> usize {
        self.scratch
    }

    fn check(
        &self,
        input: &[f64],
        output: &[f64],
        scratch: &[f64],
        inverse: bool,
    ) -> Result<(), TransformError> {
        shape(if inverse { self.output } else { self.input }, input.len())?;
        shape(if inverse { self.input } else { self.output }, output.len())?;
        if scratch.len() < self.scratch {
            return Err(TransformError::Dimension {
                expected: self.scratch,
                actual: scratch.len(),
            });
        }
        finite(input)
    }

    /// Constrain and return log absolute Jacobian. No allocations.
    pub fn constrain(
        &self,
        q: &[f64],
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<f64, TransformError> {
        self.check(q, x, scratch, false)?;
        let mut jac = 0.0;
        match self.kind {
            Kind::Matrix { .. } => jac = self.matrix_constrain(q, x, scratch)?,
            Kind::Identity => x.copy_from_slice(q),
            Kind::Lower(bound) | Kind::Upper(bound) => {
                let lower = matches!(self.kind, Kind::Lower(_));
                for (i, (&u, value)) in q.iter().zip(x.iter_mut()).enumerate() {
                    *value = if lower {
                        bound + u.exp()
                    } else {
                        bound - u.exp()
                    };
                    if *value == bound {
                        return Err(TransformError::PrecisionLoss { index: i });
                    }
                    jac += u;
                }
            }
            Kind::Interval {
                lower,
                upper,
                width,
            } => {
                for (i, (&u, value)) in q.iter().zip(x.iter_mut()).enumerate() {
                    let (s, c, log_product) = sigmoid(u);
                    *value = if u >= 0.0 {
                        upper - width * c
                    } else {
                        lower + width * s
                    };
                    if *value <= lower || *value >= upper {
                        return Err(TransformError::PrecisionLoss { index: i });
                    }
                    jac += width.ln() + log_product;
                }
            }
            Kind::Ordered { positive } => {
                let mut previous = 0.0;
                for (i, (&u, value)) in q.iter().zip(x.iter_mut()).enumerate() {
                    *value = if i == 0 && !positive {
                        u
                    } else {
                        jac += u;
                        previous + u.exp()
                    };
                    if (i > 0 || positive) && *value <= previous {
                        return Err(TransformError::PrecisionLoss { index: i });
                    }
                    previous = *value;
                }
            }
            Kind::Simplex => {
                let mut remaining: f64 = 1.0;
                for (i, &u) in q.iter().enumerate() {
                    let (v, c, lp) = sigmoid(u - ((self.input - i) as f64).ln());
                    jac += remaining.ln() + lp;
                    x[i] = remaining * v;
                    remaining *= c;
                    if x[i] <= 0.0 || remaining <= 0.0 || v == 1.0 {
                        return Err(TransformError::PrecisionLoss { index: i });
                    }
                }
                x[self.input] = remaining;
            }
            Kind::UnitVector => {
                let r2: f64 = q.iter().map(|v| v * v).sum();
                let d = 1.0 + r2;
                for (value, &u) in x.iter_mut().zip(q) {
                    *value = (2.0 / d) * u;
                }
                x[self.input] = 1.0 - 2.0 / d;
                if !d.is_finite() || x[self.input] == 1.0 {
                    return Err(TransformError::PrecisionLoss { index: self.input });
                }
                jac = self.input as f64 * (std::f64::consts::LN_2 - d.ln());
            }
        }
        output_finite(x)?;
        log_jacobian(jac)
    }

    /// Inverse with explicit support validation. No allocations.
    pub fn unconstrain(
        &self,
        x: &[f64],
        q: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        self.check(x, q, scratch, true)?;
        match self.kind {
            Kind::Matrix { .. } => self.matrix_unconstrain(x, q, scratch)?,
            Kind::Identity => q.copy_from_slice(x),
            Kind::Lower(bound) | Kind::Upper(bound) => {
                for (i, (&v, u)) in x.iter().zip(q.iter_mut()).enumerate() {
                    let distance = if matches!(self.kind, Kind::Lower(_)) {
                        v - bound
                    } else {
                        bound - v
                    };
                    if distance <= 0.0 {
                        return Err(TransformError::Domain { index: i });
                    }
                    *u = distance.ln();
                }
            }
            Kind::Interval { lower, upper, .. } => {
                for (i, (&v, u)) in x.iter().zip(q.iter_mut()).enumerate() {
                    if v <= lower || v >= upper {
                        return Err(TransformError::Domain { index: i });
                    }
                    *u = (v - lower).ln() - (upper - v).ln();
                }
            }
            Kind::Ordered { positive } => {
                let mut previous = 0.0;
                for (i, (&v, u)) in x.iter().zip(q.iter_mut()).enumerate() {
                    if i == 0 && !positive {
                        *u = v;
                    } else {
                        if v <= previous {
                            return Err(TransformError::Domain { index: i });
                        }
                        *u = (v - previous).ln();
                    }
                    previous = v;
                }
            }
            Kind::Simplex => {
                if x.iter().any(|v| *v <= 0.0)
                    || (x.iter().sum::<f64>() - 1.0).abs()
                        > 32.0 * f64::EPSILON * self.output as f64
                {
                    return Err(TransformError::Domain { index: 0 });
                }
                let mut tail = x[self.input];
                for i in (0..self.input).rev() {
                    q[i] = x[i].ln() - tail.ln() + ((self.input - i) as f64).ln();
                    tail += x[i];
                }
            }
            Kind::UnitVector => {
                let norm: f64 = x.iter().map(|v| v * v).sum();
                if (norm - 1.0).abs() > 32.0 * f64::EPSILON * self.output as f64
                    || x[self.input] >= 1.0
                {
                    return Err(TransformError::Domain { index: self.input });
                }
                for (u, v) in q.iter_mut().zip(x) {
                    *u = v / (1.0 - x[self.input]);
                }
            }
        }
        output_finite(q)
    }

    /// Write Jᵀ times the constrained gradient **plus** the log-Jacobian gradient.
    /// Checks shape and finiteness. Call `constrain` first to check representable
    /// support. No constrained-value cache is accepted or trusted by this method.
    pub fn pullback(
        &self,
        q: &[f64],
        gx: &[f64],
        gq: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), TransformError> {
        shape(self.input, q.len())?;
        shape(self.output, gx.len())?;
        shape(self.input, gq.len())?;
        if scratch.len() < self.scratch {
            return Err(TransformError::Dimension {
                expected: self.scratch,
                actual: scratch.len(),
            });
        }
        finite(q)?;
        finite(gx)?;
        match self.kind {
            Kind::Matrix { .. } => self.matrix_pullback(q, gx, gq, scratch)?,
            Kind::Identity => gq.copy_from_slice(gx),
            Kind::Lower(_) | Kind::Upper(_) => {
                let sign = if matches!(self.kind, Kind::Lower(_)) {
                    1.0
                } else {
                    -1.0
                };
                for ((g, &u), &v) in gq.iter_mut().zip(q).zip(gx) {
                    *g = sign * u.exp() * v + 1.0;
                }
            }
            Kind::Interval { width, .. } => {
                for ((g, &u), &v) in gq.iter_mut().zip(q).zip(gx) {
                    let (s, c, _) = sigmoid(u);
                    *g = width * s * c * v + (c - s);
                }
            }
            Kind::Ordered { positive } => {
                let mut tail = 0.0;
                for i in (0..self.input).rev() {
                    tail += gx[i];
                    gq[i] = if i == 0 && !positive {
                        tail
                    } else {
                        q[i].exp() * tail + 1.0
                    };
                }
            }
            Kind::Simplex => {
                let mut remaining = 1.0;
                for (i, &u) in q.iter().enumerate() {
                    scratch[i] = remaining;
                    remaining *= sigmoid(u - ((self.input - i) as f64).ln()).1;
                }
                let mut adjoint = gx[self.input];
                for i in (0..self.input).rev() {
                    let (v, c, _) = sigmoid(q[i] - ((self.input - i) as f64).ln());
                    gq[i] =
                        scratch[i] * v * c * (gx[i] - adjoint) + 1.0 - (self.output - i) as f64 * v;
                    adjoint = gx[i] * v + adjoint * c;
                }
            }
            Kind::UnitVector => {
                let d = 1.0 + q.iter().map(|v| v * v).sum::<f64>();
                let dot: f64 = q.iter().zip(gx).map(|(u, g)| u * g).sum();
                for i in 0..self.input {
                    gq[i] = 2.0 / d * gx[i] + 4.0 * (q[i] / d) / d * (gx[self.input] - dot)
                        - 2.0 * self.input as f64 * (q[i] / d);
                }
            }
        }
        output_finite(gq)
    }
}
