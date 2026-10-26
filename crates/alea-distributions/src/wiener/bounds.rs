//! Real-arithmetic truncation bounds in the same scale as the finite kernels.
//!
//! For a decreasing positive sequence with decreasing consecutive ratio r<1,
//! the tail is bounded by its first omitted term/(1-r). This applies to
//! n^p exp(-c n²), p=1..4, and both image lattices (spacing two), p=0..3.
//! A second candidate is the integral test where its monotonicity condition
//! holds. Spectral sine tails additionally use |sin(nx)| <= n|sin(x)|.
//!
//! Bounds are evaluated in f64/log space here. These formulas bound truncation
//! in real arithmetic; they do not certify libm or accumulated floating roundoff.
use super::SeriesBranch;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub(super) struct Tails {
    /// Log absolute error in the scaled density.
    pub density: f64,
    /// Log absolute error in the scaled *full* time derivative (includes prefactor).
    pub time: f64,
    /// Log absolute error in the scaled starting-point derivative.
    pub start: f64,
}

fn add(x: f64, y: f64) -> f64 {
    let m = x.max(y);
    if m == f64::NEG_INFINITY {
        m
    } else {
        m + (-(x - y).abs()).exp().ln_1p()
    }
}

/// exp(c x²) integral_x^infinity z^p exp(-c z²) dz, p<=4.
/// Even moments may decline the integral candidate when erfc underflows;
/// the geometric candidate remains available. Never treat that underflow as zero.
fn integral_scaled(p: usize, c: f64, x: f64) -> Option<f64> {
    let mut moments = [0.; 5];
    if p.is_multiple_of(2) {
        let z = c.sqrt() * x;
        if z * z > 600. {
            return None;
        }
        let tail = statrs::function::erf::erfc(z);
        if tail == 0. {
            return None;
        }
        moments[0] = PI.sqrt() / (2. * c.sqrt()) * ((z * z).exp() * tail);
    }
    moments[1] = 0.5 / c;
    for j in 2..=p {
        moments[j] = (x.powi((j - 1) as i32) + (j - 1) as f64 * moments[j - 2]) / (2. * c);
    }
    let value = moments[p];
    (value.is_finite() && value > 0.).then_some(value)
}

/// Log spectral moment tail after K, scaled by exp(c), with first omitted n=K+1.
pub(super) fn spectral(p: usize, k: usize, c: f64) -> Option<f64> {
    let n = k as f64 + 1.;
    let ratio_log = p as f64 * (1. / n).ln_1p() - (2. * n + 1.) * c;
    let geometric = if ratio_log < 0. {
        p as f64 * n.ln() - (n - 1.) * (n + 1.) * c - (-ratio_log.exp_m1()).ln()
    } else {
        f64::INFINITY
    };
    let x = k as f64;
    let integral = if x * x >= p as f64 / (2. * c) {
        integral_scaled(p, c, x).map_or(f64::INFINITY, |j| (1. - x) * (1. + x) * c + j.ln())
    } else {
        f64::INFINITY
    };
    let best = geometric.min(integral);
    (best < f64::INFINITY && !best.is_nan()).then_some(best)
}

/// Log image moment tail at z,z+2,..., relative to exp(-q²/(2u)).
fn lattice(p: usize, z: f64, q: f64, u: f64) -> Option<f64> {
    let exponent = (q - z) * (q + z) / (2. * u);
    let ratio_log = p as f64 * (2. / z).ln_1p() - 2. * (z + 1.) / u;
    let geometric = if ratio_log < 0. {
        p as f64 * z.ln() + exponent - (-ratio_log.exp_m1()).ln()
    } else {
        f64::INFINITY
    };
    let integral = if z * z >= p as f64 * u {
        integral_scaled(p, 0.5 / u, z).map_or(f64::INFINITY, |j| {
            exponent + (z.powi(p as i32) + 0.5 * j).ln()
        })
    } else {
        f64::INFINITY
    };
    let best = geometric.min(integral);
    (best < f64::INFINITY && !best.is_nan()).then_some(best)
}

pub(super) fn tails(branch: SeriesBranch, u: f64, w: f64, q: f64, k: usize) -> Option<Tails> {
    if !(u > 0. && u.is_finite() && w > 0. && w <= 1. && q > 0. && q <= 1.) || k > 4096 {
        return None;
    }
    match branch {
        SeriesBranch::LargeTime => {
            let c = 0.5 * PI * PI * u;
            if !c.is_finite() {
                return None;
            }
            let t1 = spectral(1, k, c)?;
            let t2 = spectral(2, k, c)?;
            let t3 = spectral(3, k, c)?;
            let t4 = spectral(4, k, c).unwrap_or(f64::INFINITY);
            let sine = (PI * w.min(q)).sin().ln();
            Some(Tails {
                density: t1.min(sine + t2),
                time: (0.5 * PI * PI).ln() + t3.min(sine + t4),
                start: PI.ln() + t2,
            })
        }
        SeriesBranch::SmallTime => {
            let n = 2. * (k as f64 + 1.);
            let moments: [f64; 4] =
                std::array::from_fn(
                    |p| match (lattice(p, n + q, q, u), lattice(p, n - q, q, u)) {
                        (Some(a), Some(b)) => add(a, b),
                        _ => f64::INFINITY,
                    },
                );
            let value = Tails {
                density: moments[1],
                time: add(
                    moments[3] - (2_f64.ln() + 2. * u.ln()),
                    moments[1] + (1.5_f64.ln() - u.ln()),
                ),
                start: add(moments[0], moments[2] - u.ln()),
            };
            [value.density, value.time, value.start]
                .iter()
                .all(|x| *x < f64::INFINITY && !x.is_nan())
                .then_some(value)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RatioErrors {
    pub log: f64,
    pub time: f64,
    pub start: f64,
}

/// No natural-scale density is formed. Failure to separate the denominator is
/// distinct from a finite error estimate. Full score includes the series prefactor.
pub(super) fn ratio_errors(
    log_raw: f64,
    time_score: f64,
    start_score: f64,
    tails: Tails,
) -> Option<RatioErrors> {
    let relative = (tails.density - log_raw).exp();
    if !relative.is_finite() || relative >= 1. {
        return None;
    }
    let margin = 1. - relative;
    let log = -(-relative).ln_1p();
    let time = ((tails.time - log_raw).exp() + time_score.abs() * relative) / margin;
    let start = ((tails.start - log_raw).exp() + start_score.abs() * relative) / margin;
    [log, time, start]
        .iter()
        .all(|x| x.is_finite())
        .then_some(RatioErrors { log, time, start })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometric_example_is_valid_and_improves_the_integral_bound() {
        let u = 0.2;
        let c = 0.5 * PI * PI * u;
        let k = 2;
        let actual = PI * (-c).exp() * spectral(1, k, c).unwrap().exp();
        assert!((actual - 0.0013096847036836944).abs() < 1e-15);
        let nf = (-c * (k * k) as f64).exp() / (PI * u);
        assert!(nf / actual > 23.4);
    }
    #[test]
    fn bounds_dominate_direct_density_and_derivative_tail_sums() {
        for u in [0.001, 0.01, 0.1, 0.2, 0.5, 1., 10.] {
            for q in [1e-8, 0.01, 0.2, 0.5, 0.9, 1. - 1e-8] {
                for k in 1..=12 {
                    if let Some(b) = tails(SeriesBranch::LargeTime, u, 1. - q, q, k) {
                        let c = 0.5 * PI * PI * u;
                        let (mut d, mut t, mut w) = (0., 0., 0.);
                        for j in k + 1..=2048 {
                            let n = j as f64;
                            let e = (-(n * n - 1.) * c).exp();
                            d += n * (n * PI * q).sin() * e;
                            t -= 0.5 * PI * PI * n * n * n * (n * PI * q).sin() * e;
                            w -= PI * n * n * (n * PI * q).cos() * e;
                        }
                        for (sum, bound) in [(d, b.density), (t, b.time), (w, b.start)] {
                            assert!(
                                sum.abs() <= bound.exp() * (1. + 1e-6) + 1e-300,
                                "u={u} q={q} k={k}: {sum} > exp({bound})"
                            );
                        }
                    }
                    if let Some(b) = tails(SeriesBranch::SmallTime, u, 1. - q, q, k) {
                        let (mut d, mut t, mut w) = (0., 0., 0.);
                        for j in k + 1..=2048 {
                            for x in [q + 2. * j as f64, q - 2. * j as f64] {
                                let e = (-((x - q) * (x + q)) / (2. * u)).exp();
                                d += x * e;
                                t += x * (x * x / (2. * u * u) - 1.5 / u) * e;
                                w += (x * x / u - 1.) * e;
                            }
                        }
                        for (sum, bound) in [(d, b.density), (t, b.time), (w, b.start)] {
                            assert!(
                                sum.abs() <= bound.exp() * (1. + 1e-6) + 1e-300,
                                "u={u} q={q} k={k}: {sum} > exp({bound})"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn ratio_requires_a_separated_positive_denominator() {
        assert!(
            ratio_errors(
                0.,
                1.,
                1.,
                Tails {
                    density: 0.,
                    time: -2.,
                    start: -2.
                }
            )
            .is_none()
        );
        let e = ratio_errors(
            0.,
            2.,
            3.,
            Tails {
                density: 0.1_f64.ln(),
                time: 0.2_f64.ln(),
                start: 0.3_f64.ln(),
            },
        )
        .unwrap();
        assert!((e.log + 0.9_f64.ln()).abs() < 1e-15);
        assert!((e.time - 0.4 / 0.9).abs() < 1e-15);
        assert!((e.start - 0.6 / 0.9).abs() < 1e-15);
    }
}
