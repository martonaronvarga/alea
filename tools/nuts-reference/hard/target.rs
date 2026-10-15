//! Shared analytic benchmark cost for the two Rust samplers, not production code.
use std::cell::Cell;

pub const WARMUP: usize = 1024;
pub const DRAWS: usize = 2048;
pub const DEPTH: usize = 9;

#[derive(Clone, Copy)]
enum Kind {
    Gaussian(f64),
    Banana,
    Logistic,
    Funnel,
}

pub struct Target {
    kind: Kind,
    pub dimension: usize,
    pub step: f64,
    pub calls: Cell<usize>,
    x: Vec<f64>,
}
impl Target {
    pub fn new(case: usize) -> Option<Self> {
        let (kind, dimension, step) = match case {
            0 => (Kind::Gaussian(0.95), 32, 0.08),
            1 => (Kind::Gaussian(0.9), 128, 0.10),
            2 => (Kind::Banana, 32, 0.12),
            3 => (Kind::Logistic, 32, 0.02),
            4 => (Kind::Funnel, 10, 0.04),
            _ => return None,
        };
        let x = if matches!(kind, Kind::Logistic) {
            (0..512)
                .flat_map(|i| {
                    (0..dimension)
                        .map(move |j| ((i * 17 + j * 13 + i * j * 7) % 101) as f64 / 50.0 - 1.0)
                })
                .collect()
        } else {
            Vec::new()
        };
        Some(Self {
            kind,
            dimension,
            step,
            x,
            calls: Cell::new(0),
        })
    }
    pub fn evaluate(&self, q: &[f64], g: &mut [f64]) -> f64 {
        assert_eq!(q.len(), self.dimension);
        assert_eq!(g.len(), self.dimension);
        self.calls.set(self.calls.get() + 1);
        match self.kind {
            Kind::Gaussian(rho) => {
                g.fill(0.0);
                g[0] = -q[0];
                let mut lp = -0.5 * q[0] * q[0];
                for i in 1..self.dimension {
                    let z = q[i] - rho * q[i - 1];
                    let score = z / (1.0 - rho * rho);
                    lp -= 0.5 * z * score;
                    g[i] -= score;
                    g[i - 1] += rho * score;
                }
                lp
            }
            Kind::Banana => {
                let mut lp = 0.0;
                for (q, g) in q.chunks_exact(2).zip(g.chunks_exact_mut(2)) {
                    let z = q[1] - 0.8 * (q[0] * q[0] - 1.0);
                    lp -= 0.5 * (q[0] * q[0] + z * z);
                    g[0] = -q[0] + 1.6 * q[0] * z;
                    g[1] = -z;
                }
                lp
            }
            Kind::Logistic => {
                let mut lp = 0.0;
                for (q, g) in q.iter().zip(g.iter_mut()) {
                    lp -= 0.5 * q * q;
                    *g = -q;
                }
                for (i, x) in self.x.chunks_exact(self.dimension).enumerate() {
                    let y = f64::from((i * 37 % 101) < 50);
                    let eta: f64 = x.iter().zip(q).map(|(x, q)| x * q).sum();
                    let e = (-eta.abs()).exp();
                    let p = if eta >= 0.0 {
                        1.0 / (1.0 + e)
                    } else {
                        e / (1.0 + e)
                    };
                    lp += y * eta - eta.max(0.0) - e.ln_1p();
                    for (g, x) in g.iter_mut().zip(x) {
                        *g += x * (y - p);
                    }
                }
                lp
            }
            Kind::Funnel => {
                let precision = (-q[0]).exp();
                let squared: f64 = q[1..].iter().map(|x| x * x).sum();
                let n = (self.dimension - 1) as f64;
                g[0] = -q[0] / 9.0 - 0.5 * n + 0.5 * squared * precision;
                for (q, g) in q[1..].iter().zip(g[1..].iter_mut()) {
                    *g = -q * precision;
                }
                -0.5 * q[0] * q[0] / 9.0 - 0.5 * n * q[0] - 0.5 * squared * precision
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_scores_agree_with_central_differences() {
        for case in 0..5 {
            let target = Target::new(case).unwrap();
            let mut q: Vec<_> = (0..target.dimension)
                .map(|i| (i % 7) as f64 * 0.13 - 0.4)
                .collect();
            let mut g = vec![0.; q.len()];
            let mut scratch = g.clone();
            target.evaluate(&q, &mut g);
            for i in 0..q.len() {
                let old = q[i];
                let h = 1e-5;
                q[i] = old + h;
                let plus = target.evaluate(&q, &mut scratch);
                q[i] = old - h;
                let minus = target.evaluate(&q, &mut scratch);
                q[i] = old;
                let fd = (plus - minus) / (2. * h);
                assert!(
                    (fd - g[i]).abs() < 2e-6 * (1. + g[i].abs()),
                    "case={case} coordinate={i}"
                );
            }
        }
    }
}
