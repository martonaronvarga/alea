//! Benchmark-only unconstrained densities, shared by two Rust engine adapters.
//! Independent Python/JAX and Stan definitions are checked separately.
use std::io::Read;
use std::{error::Error, fmt, fs, path::Path};

#[derive(Debug)]
pub(super) struct Failure(pub String);
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for Failure {}
#[derive(Clone)]
pub(super) struct Task {
    pub kind: usize,
    pub dimension: usize,
    pub warmup: usize,
    pub draws: usize,
    pub depth: usize,
    pub step: f64,
    pub seeds: Vec<u64>,
    pub initial: Vec<f64>,
    pub data: Vec<f64>,
}
impl Task {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(1_048_577)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1_048_576 {
            return Err("task exceeds byte limit".into());
        }
        let text = String::from_utf8(bytes)?;
        let mut tokens = text.split_whitespace();
        fn next<T: std::str::FromStr>(
            t: &mut std::str::SplitWhitespace<'_>,
        ) -> Result<T, Box<dyn Error>> {
            t.next()
                .ok_or("missing task token")?
                .parse()
                .map_err(|_| "invalid task token".into())
        }
        if next::<usize>(&mut tokens)? != 1 {
            return Err("unsupported worker protocol".into());
        }
        let kind = next(&mut tokens)?;
        let dimension: usize = next(&mut tokens)?;
        let rows: usize = next(&mut tokens)?;
        let warmup = next(&mut tokens)?;
        let draws = next(&mut tokens)?;
        let depth = next(&mut tokens)?;
        let step = next(&mut tokens)?;
        let chains: usize = next(&mut tokens)?;
        let count: usize = next(&mut tokens)?;
        if dimension.checked_mul(draws).is_none_or(|v| v > 16_777_216)
            || dimension == 0
            || dimension > 128
            || rows > 4096
            || chains == 0
            || chains > 16
            || count > 100_000
            || warmup > 1_000_000
            || draws == 0
            || draws > 1_000_000
            || depth == 0
            || depth > 12
            || !f64::is_finite(step)
            || step <= 0.0
        {
            return Err("worker resource/configuration limit".into());
        }
        let seeds = (0..chains)
            .map(|_| next(&mut tokens))
            .collect::<Result<Vec<_>, _>>()?;
        let initial = (0..dimension)
            .map(|_| next(&mut tokens))
            .collect::<Result<Vec<f64>, _>>()?;
        let data = (0..count)
            .map(|_| next(&mut tokens))
            .collect::<Result<Vec<f64>, _>>()?;
        if tokens.next().is_some() || initial.iter().chain(&data).any(|v| !v.is_finite()) {
            return Err("invalid/trailing task values".into());
        }
        let task = Self {
            kind,
            dimension,
            warmup,
            draws,
            depth,
            step,
            seeds,
            initial,
            data,
        };
        let valid = match kind {
            0 => task.data.len() == dimension && task.data.iter().all(|s| *s > 0.0),
            1 => task.data.len() == 1 && task.data[0].abs() < 1.0,
            2 => dimension.is_multiple_of(2) && task.data.len() == 1,
            3 => dimension >= 2 && task.data.len() == 1 && task.data[0] > 0.0,
            4 => task.data.len() == rows * (dimension + 1),
            5 => dimension == 4 && task.data.len() == rows * 2,
            6 => task.data.len() == dimension + 2 && task.data[0].abs() < 1.0 && task.data[1] > 0.0,
            7 => task.data.len() == dimension,
            8 => {
                dimension == 1
                    && task.data.len() == rows * 3
                    && task
                        .data
                        .chunks_exact(3)
                        .all(|r| r[0] > 0.35 && r[1].abs() == 1.0)
            }
            9 => dimension == 4 && task.data.len() == rows * 2,
            10 | 11 => {
                dimension == 10
                    && rows == 8
                    && task.data.len() == 16
                    && task.data.chunks_exact(2).all(|r| r[1] > 0.0)
            }
            _ => false,
        };
        if !valid {
            return Err("invalid model data/shape".into());
        }
        if matches!(kind, 5 | 9)
            && task
                .data
                .chunks_exact(2)
                .any(|r| r[0] != 0.0 && r[0] != 1.0)
        {
            return Err("invalid group index".into());
        }
        if kind == 9
            && task
                .data
                .chunks_exact(2)
                .any(|r| r[1] != 0.0 && r[1] != 1.0)
        {
            return Err("invalid discrete observation".into());
        }
        if kind == 4
            && task
                .data
                .chunks_exact(dimension + 1)
                .any(|r| r[dimension] != 0.0 && r[dimension] != 1.0)
        {
            return Err("invalid logistic observation".into());
        }
        Ok(task)
    }
    pub fn evaluate(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Failure> {
        let d = self.dimension;
        let x = &self.data;
        if q.len() != d || g.len() != d {
            return Err(Failure("shape".into()));
        }
        g.fill(0.0);
        let mut lp = 0.0;
        match self.kind {
            0 => {
                for i in 0..d {
                    let z = q[i] / x[i];
                    lp -= 0.5 * z * z;
                    g[i] = -z / x[i];
                }
            }
            1 => {
                let rho = x[0];
                let variance = 1.0 - rho * rho;
                lp = -0.5 * q[0] * q[0];
                g[0] = -q[0];
                for i in 1..d {
                    let z = q[i] - rho * q[i - 1];
                    lp -= 0.5 * z * z / variance;
                    g[i] -= z / variance;
                    g[i - 1] += rho * z / variance;
                }
            }
            2 => {
                for i in (0..d).step_by(2) {
                    let z = q[i + 1] - x[0] * (q[i] * q[i] - 1.0);
                    lp -= 0.5 * (q[i] * q[i] + z * z);
                    g[i] = -q[i] + 2.0 * x[0] * q[i] * z;
                    g[i + 1] = -z;
                }
            }
            3 => {
                let precision = (-q[0]).exp();
                let square = q[1..].iter().map(|v| v * v).sum::<f64>();
                lp = -0.5 * (q[0] / x[0]).powi(2)
                    - 0.5 * (d - 1) as f64 * q[0]
                    - 0.5 * square * precision;
                g[0] = -q[0] / x[0].powi(2) - 0.5 * (d - 1) as f64 + 0.5 * square * precision;
                for i in 1..d {
                    g[i] = -q[i] * precision;
                }
            }
            4 => {
                for i in 0..d {
                    lp -= 0.5 * q[i] * q[i];
                    g[i] = -q[i];
                }
                for row in x.chunks_exact(d + 1) {
                    let eta = q.iter().zip(row).map(|(q, x)| q * x).sum::<f64>();
                    let y = row[d];
                    lp += y * eta - softplus(eta);
                    for i in 0..d {
                        g[i] += row[i] * (y - sigmoid(eta));
                    }
                }
            }
            5 | 9 => {
                let tau = q[1].exp();
                lp = -0.5 * (q[0] / 0.7).powi(2)
                    - 0.5 * ((q[1] + 0.7) / 0.3).powi(2)
                    - 0.5 * (q[2] * q[2] + q[3] * q[3]);
                g[0] = -q[0] / 0.49;
                g[1] = -(q[1] + 0.7) / 0.09;
                g[2] = -q[2];
                g[3] = -q[3];
                for row in x.chunks_exact(2) {
                    let group = if row[0] == 0.0 { 2 } else { 3 };
                    let eta = q[0] + tau * q[group];
                    let score = if self.kind == 5 {
                        let residual = row[1] - eta;
                        lp -= 0.5 * residual * residual / 0.49;
                        residual / 0.49
                    } else {
                        let logit = 1.2 * eta;
                        lp += row[1] * logit - softplus(logit);
                        1.2 * (row[1] - sigmoid(logit))
                    };
                    g[0] += score;
                    g[1] += score * tau * q[group];
                    g[group] += score * tau;
                }
            }
            6 => {
                let phi = x[0];
                let variance = x[1] * x[1];
                lp = -0.5 * q[0] * q[0] * (1.0 - phi * phi) / variance;
                g[0] = -q[0] * (1.0 - phi * phi) / variance;
                for i in 1..d {
                    let z = q[i] - phi * q[i - 1];
                    lp -= 0.5 * z * z / variance;
                    g[i] -= z / variance;
                    g[i - 1] += phi * z / variance;
                }
                for i in 0..d {
                    let e = x[i + 2] * x[i + 2] * (-q[i]).exp();
                    lp -= 0.5 * (q[i] + e);
                    g[i] += -0.5 + 0.5 * e;
                }
            }
            7 => {
                for i in 0..d {
                    let residual = x[i] - q[i];
                    lp -= 0.5 * q[i] * q[i] + 2.0 * residual * residual;
                    g[i] = -q[i] + 4.0 * residual;
                }
            }
            8 => {
                lp = -0.5 * q[0] * q[0];
                g[0] = -q[0];
                for row in x.chunks_exact(3) {
                    let t = row[0] - 0.15;
                    lp += row[2] + 0.6 * row[1] * q[0] - 0.5 * q[0] * q[0] * t;
                    g[0] += 0.6 * row[1] - q[0] * t;
                }
            }
            10 | 11 => {
                let tau = q[1].exp();
                let a = tau / 5.0;
                lp = -0.5 * (q[0] / 5.0).powi(2) + q[1] - (1.0 + a * a).ln();
                g[0] = -q[0] / 25.0;
                g[1] = 1.0 - 2.0 * a * a / (1.0 + a * a);
                for i in 0..8 {
                    let y = x[2 * i];
                    let sigma = x[2 * i + 1];
                    if self.kind == 10 {
                        let delta = q[i + 2] - q[0];
                        let z = delta / tau;
                        let r = (y - q[i + 2]) / sigma;
                        lp -= q[1] + 0.5 * z * z + 0.5 * r * r;
                        g[0] += delta / (tau * tau);
                        g[1] += -1.0 + z * z;
                        g[i + 2] = -delta / (tau * tau) + r / sigma;
                    } else {
                        let theta = q[0] + tau * q[i + 2];
                        let r = (y - theta) / sigma;
                        lp -= 0.5 * q[i + 2] * q[i + 2] + 0.5 * r * r;
                        g[0] += r / sigma;
                        g[1] += r / sigma * tau * q[i + 2];
                        g[i + 2] = -q[i + 2] + r / sigma * tau;
                    }
                }
            }
            _ => return Err(Failure("unknown model".into())),
        }
        if !lp.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err(Failure("nonfinite model evaluation".into()));
        }
        Ok(lp)
    }
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
