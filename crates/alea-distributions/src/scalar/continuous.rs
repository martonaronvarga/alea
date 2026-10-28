//! Scalar families with shared, allocation-free primal and analytic-score kernels.
use super::jet::Jet;
use super::{LN_TAU, checked, finite, gamma_draw, positive, score, standard_normal, uniform};
use alea_core::distribution::{
    ContinuousDistribution, Distribution, DistributionError as E, EventShape, ParameterScore,
    Sample, Support,
};
use rand::Rng;
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Uniform,
    Cauchy,
    Laplace,
    Logistic,
    Gumbel,
    SkewNormal,
    ExpModNormal,
    NormalInverseGaussian,
    GeneralizedExtremeValue,
    BetaProportion,
    Weibull,
    Frechet,
    Pareto,
    ParetoType2,
    InverseGamma,
    ChiSquared,
    InverseChiSquared,
    ScaledInverseChiSquared,
    FisherF,
    Rayleigh,
    InverseGaussian,
    Levy,
    HalfNormal,
    HalfCauchy,
    LogLogistic,
    Triangular,
    Pert,
    VonMises,
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct Kernel {
    kind: Kind,
    p: [f64; 4],
}
impl Kernel {
    fn contains(self, x: f64, interior: bool) -> bool {
        let [a, b, c, _] = self.p;
        match self.kind {
            Kind::BetaProportion => {
                if interior {
                    x > 0.0 && x < 1.0
                } else {
                    (0.0..=1.0).contains(&x)
                }
            }
            Kind::GeneralizedExtremeValue => {
                let q = 1.0 + c * ((x - a) / b);
                c == 0.0 || if interior { q > 0.0 } else { q >= 0.0 }
            }
            Kind::Uniform => {
                if interior {
                    x > a && x < b
                } else {
                    x >= a && x <= b
                }
            }
            Kind::Triangular | Kind::Pert => {
                if interior {
                    x > a && x < b
                } else {
                    x >= a && x <= b
                }
            }
            Kind::ChiSquared
            | Kind::Weibull
            | Kind::Rayleigh
            | Kind::HalfNormal
            | Kind::HalfCauchy => {
                if interior {
                    x > 0.0
                } else {
                    x >= 0.0
                }
            }
            Kind::InverseGamma
            | Kind::InverseChiSquared
            | Kind::ScaledInverseChiSquared
            | Kind::FisherF
            | Kind::InverseGaussian
            | Kind::LogLogistic => x > 0.0,
            Kind::Frechet | Kind::Levy => x > a,
            Kind::Pareto => {
                if interior {
                    x > a
                } else {
                    x >= a
                }
            }
            Kind::ParetoType2 => {
                if interior {
                    x > a
                } else {
                    x >= a
                }
            }
            Kind::VonMises => (-PI..PI).contains(&x),
            _ => {
                let _ = c;
                true
            }
        }
    }
    fn eval<const N: usize>(self, x: Jet<N>, p: [Jet<N>; 4]) -> Jet<N> {
        let [a, b, c, d] = p;
        match self.kind {
            Kind::BetaProportion => {
                let alpha = a * b;
                let beta = (1.0 - a) * b;
                (alpha + beta).lgamma() - alpha.lgamma() - beta.lgamma()
                    + (alpha - 1.0) * x.ln()
                    + (beta - 1.0) * (-x).ln_1p()
            }
            Kind::GeneralizedExtremeValue => {
                let z = (x - a) / b;
                let u = c * z;
                let w = if u.v.abs() < 1e-5 {
                    z * (1.0 - u * 0.5 + u * u / 3.0 - u * u * u / 4.0 + u * u * u * u / 5.0)
                } else {
                    u.ln_1p() / c
                };
                -b.ln() - (1.0 + c) * w - (-w).exp()
            }
            Kind::Uniform => -(b - a).ln(),
            Kind::Cauchy | Kind::HalfCauchy => {
                let (z, scale, extra) = if self.kind == Kind::Cauchy {
                    ((x - a) / b, b, 0.0)
                } else {
                    (x / a, a, 2.0_f64.ln())
                };
                let l = if z.v.abs() > 1e150 {
                    let log_distance = if self.kind == Kind::Cauchy {
                        x.log_abs_difference(a)
                    } else {
                        x.ln()
                    };
                    (2.0 * (log_distance - scale.ln())).softplus()
                } else {
                    (z * z).ln_1p()
                };
                extra - PI.ln() - scale.ln() - l
            }
            Kind::Laplace => -(2.0 * b).ln() - ((x - a) / b).abs(),
            Kind::Logistic => {
                let z = (x - a) / b;
                -b.ln() - z - 2.0 * (-z).softplus()
            }
            Kind::Gumbel => {
                let z = (x - a) / b;
                -b.ln() - z - (-z).exp()
            }
            Kind::SkewNormal => {
                let z = (x - a) / b;
                2.0_f64.ln() - b.ln() - 0.5 * LN_TAU - 0.5 * z * z + (c * z).log_phi()
            }
            Kind::ExpModNormal => {
                let z = (x - a) / b;
                let v = z - c * b;
                if v.v < -10.0 {
                    c.ln() - 0.5 * z * z + v.log_phi_scaled()
                } else {
                    c.ln() - c * (x - a) + 0.5 * c * c * b * b + v.log_phi()
                }
            }
            Kind::NormalInverseGaussian => {
                let z = x - c;
                let gamma = (a - b.abs()).sqrt() * (a + b.abs()).sqrt();
                let r = d.hypot(z);
                let exponent = -d * b * (b / (a + gamma)) + b * z - a * z * (z / (r + d));
                a.ln() + d.ln() - PI.ln() + exponent + (a * r).log_k1_scaled() - r.ln()
            }
            Kind::Weibull => {
                let log_z = x.ln() - b.ln();
                let power = a - 1.0;
                let term = if x.v == 0.0 && power.v == 0.0 {
                    Jet::c(0.0)
                } else {
                    power * log_z
                };
                a.ln() - b.ln() + term - (a * log_z).exp()
            }
            Kind::Frechet => {
                let log_z = x.log_abs_difference(a) - b.ln();
                c.ln() - b.ln() - (c + 1.0) * log_z - (-c * log_z).exp()
            }
            Kind::Pareto => b.ln() - a.ln() - (b + 1.0) * (x.ln() - a.ln()),
            Kind::ParetoType2 => {
                let z = (x - a) / b;
                let log_z = if z.v.is_finite() {
                    z.ln_1p()
                } else {
                    (x.log_abs_difference(a) - b.ln()).softplus()
                };
                c.ln() - b.ln() - (c + 1.0) * log_z
            }
            Kind::InverseGamma => a * b.ln() - a.lgamma() - (a + 1.0) * x.ln() - b / x,
            Kind::ChiSquared => {
                let h = a * 0.5;
                -h * 2.0_f64.ln() - h.lgamma() + (h - 1.0) * x.ln() - x * 0.5
            }
            Kind::InverseChiSquared => {
                let h = a * 0.5;
                -h * 2.0_f64.ln() - h.lgamma() - (h + 1.0) * x.ln() - 0.5 / x
            }
            Kind::ScaledInverseChiSquared => {
                let h = a * 0.5;
                let r = h * b * b;
                h * r.ln() - h.lgamma() - (h + 1.0) * x.ln() - r / x
            }
            Kind::FisherF => {
                let h = a * 0.5;
                let j = b * 0.5;
                let log_ratio = a.ln() - b.ln();
                h * log_ratio + (h - 1.0) * x.ln()
                    - (h + j) * (log_ratio + x.ln()).softplus()
                    - (h.lgamma() + j.lgamma() - (h + j).lgamma())
            }
            Kind::Rayleigh => x.ln() - 2.0 * a.ln() - 0.5 * (x / a) * (x / a),
            Kind::InverseGaussian => {
                let z = x / a - 1.0;
                0.5 * (b.ln() - LN_TAU - 3.0 * x.ln()) - 0.5 * b * z * z / x
            }
            Kind::Levy => {
                let distance = x.log_abs_difference(a);
                0.5 * (b.ln() - LN_TAU) - 1.5 * distance - 0.5 * (b.ln() - distance).exp()
            }
            Kind::HalfNormal => 2.0_f64.ln() - 0.5 * LN_TAU - a.ln() - 0.5 * (x / a) * (x / a),
            Kind::LogLogistic => {
                let z = a * (x.ln() - b.ln());
                a.ln() - x.ln() + z - 2.0 * z.softplus()
            }
            Kind::Triangular => {
                if x.v < c.v {
                    2.0_f64.ln() + (x - a).ln() - (b - a).ln() - (c - a).ln()
                } else if x.v > c.v {
                    2.0_f64.ln() + (b - x).ln() - (b - a).ln() - (b - c).ln()
                } else {
                    2.0_f64.ln() - (b - a).ln()
                }
            }
            Kind::Pert => {
                let alpha = 1.0 + d * (c - a) / (b - a);
                let beta = 1.0 + d * (b - c) / (b - a);
                let z = (x - a) / (b - a);
                (alpha + beta).lgamma() - alpha.lgamma() - beta.lgamma()
                    + (alpha - 1.0) * z.ln()
                    + (beta - 1.0) * (-z).ln_1p()
                    - (b - a).ln()
            }
            Kind::VonMises => {
                let i0 = spec_math::cephes64::i0e(b.v);
                let scaled = i0.ln();
                let ratio = if N == 0 {
                    0.0
                } else {
                    spec_math::cephes64::i1e(b.v) / i0
                };
                let normalizer = Jet {
                    v: scaled,
                    d: b.d.map(|v| v * (ratio - 1.0)),
                };
                b * ((x - a).cos() - 1.0) - TAU.ln() - normalizer
            }
        }
    }
    fn logp(self, x: f64) -> Result<f64, E> {
        if x.is_nan() {
            return Err(E::Observation);
        }
        if !x.is_finite() || !self.contains(x, false) {
            return Ok(f64::NEG_INFINITY);
        }
        if self.kind == Kind::BetaProportion && (x == 0.0 || x == 1.0) {
            let alpha = self.p[0] * self.p[1];
            let beta = (1.0 - self.p[0]) * self.p[1];
            let (shape, other) = if x == 0.0 {
                (alpha, beta)
            } else {
                (beta, alpha)
            };
            return Ok(if shape == 1.0 {
                other.ln()
            } else if shape < 1.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            });
        }
        if self.kind == Kind::GeneralizedExtremeValue
            && self.p[2] != 0.0
            && 1.0 + self.p[2] * ((x - self.p[0]) / self.p[1]) == 0.0
        {
            return Ok(if self.p[2] == -1.0 {
                -self.p[1].ln()
            } else if self.p[2] < -1.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            });
        }
        // Gamma boundary limits avoid 0*log(0).
        if x == 0.0 && self.kind == Kind::ChiSquared {
            return Ok(if self.p[0] == 2.0 {
                -2.0_f64.ln()
            } else if self.p[0] < 2.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            });
        }
        if matches!(self.kind, Kind::Pert) && (x == self.p[0] || x == self.p[1]) {
            // A mode at an endpoint gives one beta shape exactly one.
            let [a, b, c, d] = self.p;
            let alpha = 1.0 + d * (c - a) / (b - a);
            let beta = 1.0 + d * (b - c) / (b - a);
            return Ok(if x == a && alpha == 1.0 {
                beta.ln() - (b - a).ln()
            } else if x == b && beta == 1.0 {
                alpha.ln() - (b - a).ln()
            } else {
                f64::NEG_INFINITY
            });
        }
        checked(self.eval(Jet::<0>::c(x), self.p.map(Jet::c)).v)
    }
    fn derivatives(self, x: f64) -> Result<[f64; 5], E> {
        if !x.is_finite() || !self.contains(x, true) {
            return Err(E::Observation);
        }
        if (self.kind == Kind::Laplace && x == self.p[0])
            || (self.kind == Kind::Triangular && x == self.p[2])
        {
            return Err(E::Observation);
        }
        let p = std::array::from_fn(|i| Jet::variable(self.p[i], i + 1));
        let result = self.eval(Jet::variable(x, 0), p);
        if result.d.iter().any(|d| !d.is_finite()) {
            Err(E::Numerical)
        } else {
            Ok(result.d)
        }
    }
    fn draw<R: Rng + ?Sized>(self, rng: &mut R) -> Result<f64, E> {
        let [a, b, c, d] = self.p;
        let u = uniform(rng);
        let x = match self.kind {
            Kind::BetaProportion => {
                let g = gamma_draw(a * b, rng)?;
                let h = gamma_draw((1.0 - a) * b, rng)?;
                g / (g + h)
            }
            Kind::GeneralizedExtremeValue => {
                let l = (-u.ln()).ln();
                a + b * if c == 0.0 { -l } else { (-c * l).exp_m1() / c }
            }
            Kind::Uniform => (1.0 - u) * a + u * b,
            Kind::Cauchy => a + b * (PI * (u - 0.5)).tan(),
            Kind::HalfCauchy => a * (PI * u * 0.5).tan(),
            Kind::HalfNormal => a * standard_normal(rng).abs(),
            Kind::Laplace => a - b * (u - 0.5).signum() * (1.0 - 2.0 * (u - 0.5).abs()).ln(),
            Kind::Logistic => a + b * (u.ln() - (-u).ln_1p()),
            Kind::Gumbel => a - b * (-u.ln()).ln(),
            Kind::SkewNormal => {
                let delta = c / c.hypot(1.0);
                a + b * (delta * standard_normal(rng).abs() + standard_normal(rng) / c.hypot(1.0))
            }
            Kind::ExpModNormal => a + b * standard_normal(rng) - u.ln() / c,
            Kind::NormalInverseGaussian => {
                let gamma = (a - b.abs()).sqrt() * (a + b.abs()).sqrt();
                let w = Kernel {
                    kind: Kind::InverseGaussian,
                    p: [d / gamma, d * d, 0.0, 0.0],
                }
                .draw(rng)?;
                c + b * w + w.sqrt() * standard_normal(rng)
            }
            Kind::Weibull => b * ((-u.ln()).ln() / a).exp(),
            Kind::Frechet => a + b * (-(-u.ln()).ln() / c).exp(),
            Kind::Pareto => a * (-u.ln() / b).exp(),
            Kind::ParetoType2 => a + b * (-u.ln() / c).exp_m1(),
            Kind::InverseGamma => b / gamma_draw(a, rng)?,
            Kind::ChiSquared => 2.0 * gamma_draw(a * 0.5, rng)?,
            Kind::InverseChiSquared => 0.5 / gamma_draw(a * 0.5, rng)?,
            Kind::ScaledInverseChiSquared => a * b * b / (2.0 * gamma_draw(a * 0.5, rng)?),
            Kind::FisherF => (gamma_draw(a * 0.5, rng)? / a) / (gamma_draw(b * 0.5, rng)? / b),
            Kind::Rayleigh => a * (-2.0 * u.ln()).sqrt(),
            Kind::Levy => {
                let z = standard_normal(rng);
                a + b / (z * z)
            }
            Kind::InverseGaussian => {
                let z = standard_normal(rng);
                let q = (a / b) * z * z * 0.5;
                let ratio = 1.0 / (1.0 + q + (q * (q + 2.0)).sqrt());
                if u <= 1.0 / (1.0 + ratio) {
                    a * ratio
                } else {
                    a / ratio
                }
            }
            Kind::LogLogistic => b * ((u.ln() - (-u).ln_1p()) / a).exp(),
            Kind::Triangular => {
                if u < (c - a) / (b - a) {
                    a + ((b - a) * (c - a) * u).sqrt()
                } else {
                    b - ((b - a) * (b - c) * (1.0 - u)).sqrt()
                }
            }
            Kind::Pert => {
                let alpha = 1.0 + d * (c - a) / (b - a);
                let beta = 1.0 + d * (b - c) / (b - a);
                let g = gamma_draw(alpha, rng)?;
                let h = gamma_draw(beta, rng)?;
                a + (b - a) * (g / (g + h))
            }
            Kind::VonMises => return von_mises_draw(a, b, rng),
        };
        if x.is_finite() && self.contains(x, true) {
            Ok(x)
        } else {
            Err(E::Numerical)
        }
    }
}
fn von_mises_draw<R: Rng + ?Sized>(mu: f64, k: f64, rng: &mut R) -> Result<f64, E> {
    let wrap = |x: f64| (x + PI).rem_euclid(TAU) - PI;
    if k < 1e-6 {
        for _ in 0..4096 {
            let angle = TAU * uniform(rng) - PI;
            if uniform(rng).ln() <= k * (angle.cos() - 1.0) {
                return Ok(wrap(mu + angle));
            }
        }
    } else {
        // Best--Fisher rejection; reciprocal form avoids cancellation at small k.
        let a = 1.0 + (1.0 + 4.0 * k * k).sqrt();
        let b = 2.0 * k / (a + (2.0 * a).sqrt());
        let r = (1.0 + b * b) / (2.0 * b);
        for _ in 0..4096 {
            let z = (PI * uniform(rng)).cos();
            let f = (1.0 + r * z) / (r + z);
            let c = k * (r - f);
            let u = uniform(rng);
            if u < c * (2.0 - c) || u.ln() <= c.ln() + 1.0 - c {
                return Ok(wrap(
                    mu + (uniform(rng) - 0.5).signum() * f.clamp(-1.0, 1.0).acos(),
                ));
            }
        }
    }
    Err(E::SamplingBudget)
}
macro_rules! family {
    ($name:ident,$kind:ident,$support:ident,$n:literal,$doc:literal) => {
        #[doc=$doc]
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use]
        pub struct $name(Kernel);
        impl Distribution for $name {
            type Event = f64;
            fn event_shape(&self) -> EventShape {
                EventShape::Scalar
            }
            fn support(&self) -> Support {
                Support::$support
            }
            fn log_prob(&self, x: &f64) -> Result<f64, E> {
                self.0.logp(*x)
            }
        }
        impl ParameterScore for $name {
            fn parameter_count(&self) -> usize {
                $n
            }
            fn parameter_score(&self, x: &f64, out: &mut [f64]) -> Result<(), E> {
                let s = self.0.derivatives(*x)?;
                score(out, &s[1..1 + $n])
            }
        }
        impl ContinuousDistribution for $name {
            fn event_score(&self, x: f64) -> Result<f64, E> {
                Ok(self.0.derivatives(x)?[0])
            }
        }
        impl Sample for $name {
            type Draw = f64;
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<f64, E> {
                self.0.draw(rng)
            }
        }
    };
}
family!(
    Uniform,
    Uniform,
    Interval,
    2,
    "Uniform on [lower, upper]. Score order: lower, upper; scores require interior events."
);
family!(
    Cauchy,
    Cauchy,
    Real,
    2,
    "Cauchy. Score order: location, scale."
);
family!(
    Laplace,
    Laplace,
    Real,
    2,
    "Laplace (double exponential). Score order: location, scale; no score at the cusp."
);
family!(
    Logistic,
    Logistic,
    Real,
    2,
    "Logistic. Score order: location, scale."
);
family!(
    Gumbel,
    Gumbel,
    Real,
    2,
    "Gumbel maximum distribution. Score order: location, scale."
);
family!(
    SkewNormal,
    SkewNormal,
    Real,
    3,
    "Skew normal. Score order: location, scale, skew."
);
family!(
    ExpModNormal,
    ExpModNormal,
    Real,
    3,
    "Normal plus independent exponential. Score order: location, scale, rate."
);
family!(
    Weibull,
    Weibull,
    Positive,
    2,
    "Weibull. Score order: shape, scale (not rate)."
);
family!(
    Frechet,
    Frechet,
    LowerBounded,
    3,
    "Frechet. Score order: location, scale, shape."
);
family!(
    Pareto,
    Pareto,
    LowerBounded,
    2,
    "Pareto type I. Score order: minimum, shape."
);
family!(
    ParetoType2,
    ParetoType2,
    LowerBounded,
    3,
    "Pareto type II (Lomax when location=0). Score order: location, scale, shape."
);
family!(
    InverseGamma,
    InverseGamma,
    Positive,
    2,
    "Inverse gamma. Score order: shape, rate of the reciprocal Gamma law; X=rate/Gamma(shape,1)."
);
family!(
    ChiSquared,
    ChiSquared,
    Positive,
    1,
    "Chi-squared. Score order: degrees of freedom."
);
family!(
    InverseChiSquared,
    InverseChiSquared,
    Positive,
    1,
    "Reciprocal chi-squared. Score order: degrees of freedom."
);
family!(
    ScaledInverseChiSquared,
    ScaledInverseChiSquared,
    Positive,
    2,
    "Scaled inverse chi-squared, X=df*scale²/chi²(df). Score order: df, scale."
);
family!(
    FisherF,
    FisherF,
    Positive,
    2,
    "F distribution. Score order: numerator df, denominator df."
);
family!(
    Rayleigh,
    Rayleigh,
    Positive,
    1,
    "Rayleigh. Score order: scale."
);
family!(
    InverseGaussian,
    InverseGaussian,
    Positive,
    2,
    "Inverse Gaussian (Wald). Score order: mean, shape."
);
family!(
    Levy,
    Levy,
    LowerBounded,
    2,
    "Levy. Score order: location, scale."
);
family!(
    HalfNormal,
    HalfNormal,
    Positive,
    1,
    "Half-normal with zero location. Score order: scale."
);
family!(
    HalfCauchy,
    HalfCauchy,
    Positive,
    1,
    "Half-Cauchy with zero location. Score order: scale."
);
family!(
    LogLogistic,
    LogLogistic,
    Positive,
    2,
    "Log-logistic. Score order: shape, scale."
);
family!(
    Triangular,
    Triangular,
    Interval,
    3,
    "Triangular on `[lower,upper]`. Score order: lower, upper, mode; no score at mode."
);
family!(
    Pert,
    Pert,
    Interval,
    4,
    "Beta-PERT. Score order: lower, upper, mode, weight; weight=4 is the common PERT law."
);
family!(
    VonMises,
    VonMises,
    Circle,
    2,
    "Von Mises on [-pi,pi), radians. Score order: mean angle, concentration; zero concentration is uniform."
);
macro_rules! constructor {
    ($name:ident,$kind:ident, $( $param:ident : $validate:ident ),+ ) => {
        impl $name {
            /// Constructs a normalized law in the parameter order documented on the type.
            /// # Errors
            /// Rejects non-finite parameters and non-positive shape, scale or rate.
            pub fn new($( $param:f64 ),+)->Result<Self,E>{
                $(let $param=$validate($param,stringify!($param))?;)+
                let mut p=[0.0;4];let values=[$($param),+];p[..values.len()].copy_from_slice(&values);
                Ok(Self(Kernel{kind:Kind::$kind,p}))
            }
        }
    };
}
constructor!(Cauchy,Cauchy,location:finite,scale:positive);
constructor!(Laplace,Laplace,location:finite,scale:positive);
constructor!(Logistic,Logistic,location:finite,scale:positive);
constructor!(Gumbel,Gumbel,location:finite,scale:positive);
constructor!(SkewNormal,SkewNormal,location:finite,scale:positive,skew:finite);
constructor!(ExpModNormal,ExpModNormal,location:finite,scale:positive,rate:positive);
constructor!(Weibull,Weibull,shape:positive,scale:positive);
constructor!(Frechet,Frechet,location:finite,scale:positive,shape:positive);
constructor!(Pareto,Pareto,minimum:positive,shape:positive);
constructor!(ParetoType2,ParetoType2,location:finite,scale:positive,shape:positive);
constructor!(InverseGamma,InverseGamma,shape:positive,rate:positive);
constructor!(ChiSquared,ChiSquared,df:positive);
constructor!(InverseChiSquared,InverseChiSquared,df:positive);
constructor!(ScaledInverseChiSquared,ScaledInverseChiSquared,df:positive,scale:positive);
constructor!(FisherF,FisherF,numerator_df:positive,denominator_df:positive);
constructor!(Rayleigh,Rayleigh,scale:positive);
constructor!(InverseGaussian,InverseGaussian,mean:positive,shape:positive);
constructor!(Levy,Levy,location:finite,scale:positive);
constructor!(HalfNormal,HalfNormal,scale:positive);
constructor!(HalfCauchy,HalfCauchy,scale:positive);
constructor!(LogLogistic,LogLogistic,shape:positive,scale:positive);
impl Uniform {
    /// # Errors
    /// Requires finite ordered bounds with representable positive width.
    pub fn new(lower: f64, upper: f64) -> Result<Self, E> {
        interval(lower, upper)?;
        Ok(Self(Kernel {
            kind: Kind::Uniform,
            p: [lower, upper, 0.0, 0.0],
        }))
    }
}
fn interval(a: f64, b: f64) -> Result<(), E> {
    finite(a, "lower")?;
    finite(b, "upper")?;
    positive(b - a, "interval width")?;
    Ok(())
}
impl Triangular {
    /// # Errors
    /// Requires ordered finite bounds and a mode in the closed interval.
    pub fn new(lower: f64, upper: f64, mode: f64) -> Result<Self, E> {
        interval(lower, upper)?;
        finite(mode, "mode")?;
        if mode < lower || mode > upper {
            return Err(E::Parameter("mode"));
        }
        Ok(Self(Kernel {
            kind: Kind::Triangular,
            p: [lower, upper, mode, 0.0],
        }))
    }
}
impl Pert {
    /// # Errors
    /// Requires finite ordered bounds, a mode in the interval and positive weight.
    pub fn new(lower: f64, upper: f64, mode: f64, weight: f64) -> Result<Self, E> {
        let _ = Triangular::new(lower, upper, mode)?;
        positive(weight, "weight")?;
        Ok(Self(Kernel {
            kind: Kind::Pert,
            p: [lower, upper, mode, weight],
        }))
    }
}
impl VonMises {
    /// # Errors
    /// Requires a finite mean and concentration in [0, 1e8]. Larger values cannot
    /// be sampled reliably by this binary64 rejection implementation.
    pub fn new(mean: f64, concentration: f64) -> Result<Self, E> {
        finite(mean, "mean")?;
        if !concentration.is_finite() || !(0.0..=1e8).contains(&concentration) {
            return Err(E::Parameter("concentration"));
        }
        Ok(Self(Kernel {
            kind: Kind::VonMises,
            p: [(mean + PI).rem_euclid(TAU) - PI, concentration, 0.0, 0.0],
        }))
    }
}

family!(
    NormalInverseGaussian,
    NormalInverseGaussian,
    Real,
    4,
    "Normal-inverse-Gaussian. Score order: alpha, beta, location, delta. Requires alpha>|beta| and delta>0. Uses exponentially scaled Bessel K functions."
);
impl NormalInverseGaussian {
    /// # Errors
    /// Requires finite alpha>|beta|, finite location and positive finite delta.
    pub fn new(alpha: f64, beta: f64, location: f64, delta: f64) -> Result<Self, E> {
        positive(alpha, "alpha")?;
        finite(beta, "beta")?;
        finite(location, "location")?;
        positive(delta, "delta")?;
        if alpha <= beta.abs() {
            return Err(E::Parameter("alpha must exceed abs(beta)"));
        }
        let gamma = (alpha - beta.abs()).sqrt() * (alpha + beta.abs()).sqrt();
        if !gamma.is_finite() {
            return Err(E::Numerical);
        }
        Ok(Self(Kernel {
            kind: Kind::NormalInverseGaussian,
            p: [alpha, beta, location, delta],
        }))
    }
}

family!(
    BetaProportion,
    BetaProportion,
    UnitInterval,
    2,
    "Beta with mean in (0,1) and precision>0. Score order: mean, precision; alpha=mean*precision, beta=(1-mean)*precision."
);
impl BetaProportion {
    /// # Errors
    /// Requires an interior mean, positive precision and representable beta shapes/normalization.
    pub fn new(mean: f64, precision: f64) -> Result<Self, E> {
        finite(mean, "mean")?;
        if !(0.0 < mean && mean < 1.0) {
            return Err(E::Parameter("mean"));
        }
        positive(precision, "precision")?;
        let _ = super::Beta::new(mean * precision, (1.0 - mean) * precision)?;
        Ok(Self(Kernel {
            kind: Kind::BetaProportion,
            p: [mean, precision, 0.0, 0.0],
        }))
    }
}
family!(
    GeneralizedExtremeValue,
    GeneralizedExtremeValue,
    ParameterDependent,
    3,
    "Generalized extreme value for maxima. Score order: location, scale, shape. Shape zero is Gumbel; support requires 1+shape*(x-location)/scale>=0."
);
constructor!(GeneralizedExtremeValue,GeneralizedExtremeValue,location:finite,scale:positive,shape:finite);
