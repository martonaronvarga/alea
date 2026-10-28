//! Small forward jets for analytic scalar scores. N=0 compiles to primal arithmetic.
use statrs::function::gamma::{digamma, ln_gamma};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub(super) struct Jet<const N: usize> {
    pub v: f64,
    pub d: [f64; N],
}
impl<const N: usize> Jet<N> {
    pub fn c(v: f64) -> Self {
        Self { v, d: [0.0; N] }
    }
    pub fn variable(v: f64, i: usize) -> Self {
        let mut x = Self::c(v);
        x.d[i] = 1.0;
        x
    }
    fn map(self, v: f64, slope: f64) -> Self {
        Self {
            v,
            d: self.d.map(|d| if d == 0.0 { 0.0 } else { d * slope }),
        }
    }
    pub fn ln(self) -> Self {
        self.map(self.v.ln(), self.v.recip())
    }
    pub fn ln_1p(self) -> Self {
        self.map(self.v.ln_1p(), (1.0 + self.v).recip())
    }
    pub fn exp(self) -> Self {
        let v = self.v.exp();
        self.map(v, v)
    }
    pub fn abs(self) -> Self {
        self.map(self.v.abs(), self.v.signum())
    }
    pub fn cos(self) -> Self {
        self.map(self.v.cos(), -self.v.sin())
    }
    pub fn lgamma(self) -> Self {
        self.map(ln_gamma(self.v), digamma(self.v))
    }
    pub fn softplus(self) -> Self {
        let v = self.v.max(0.0) + (-self.v.abs()).exp().ln_1p();
        let slope = if self.v >= 0.0 {
            1.0 / (1.0 + (-self.v).exp())
        } else {
            let e = self.v.exp();
            e / (1.0 + e)
        };
        self.map(v, slope)
    }
    /// log Phi(x), including its Mills-ratio derivative in the negative tail.
    pub fn log_phi(self) -> Self {
        if self.v < -10.0 {
            self.log_phi_scaled() - self * self * 0.5
        } else {
            let v = (0.5 * libm::erfc(-self.v / std::f64::consts::SQRT_2)).ln();
            self.map(v, (-0.5 * self.v * self.v - super::LN_TAU * 0.5 - v).exp())
        }
    }
    /// log Phi(x) + x²/2 for x < -10, without subtracting large squares.
    pub fn log_phi_scaled(self) -> Self {
        let r = -self.v;
        let mut fraction = 0.0;
        for k in (1..=80).rev() {
            fraction = k as f64 / (r + fraction);
        }
        let mills_inverse = r + fraction;
        self.map(-0.5 * super::LN_TAU - mills_inverse.ln(), fraction)
    }
}
impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: std::array::from_fn(|i| self.d[i] + b.d[i]),
        }
    }
}
impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: self.d.map(|v| -v),
        }
    }
}
// These are the product and quotient rules, not componentwise vector operations.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: std::array::from_fn(|i| (self.d[i] - v * b.d[i]) / b.v),
        }
    }
}
macro_rules! scalar_op {
    ($trait:ident, $method:ident) => {
        impl<const N: usize> $trait<f64> for Jet<N> {
            type Output = Self;
            fn $method(self, rhs: f64) -> Self {
                self.$method(Self::c(rhs))
            }
        }
        impl<const N: usize> $trait<Jet<N>> for f64 {
            type Output = Jet<N>;
            fn $method(self, rhs: Jet<N>) -> Jet<N> {
                Jet::c(self).$method(rhs)
            }
        }
    };
}
scalar_op!(Add, add);
scalar_op!(Sub, sub);
scalar_op!(Mul, mul);
scalar_op!(Div, div);
