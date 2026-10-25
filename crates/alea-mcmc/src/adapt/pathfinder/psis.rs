//! Cold PSIS weights, using Zhang--Stephens GPD fitting and Stan2.38's tail policy.
//! No sampling state or retained transition depends on this initialization helper.
use super::PsisStatus;

pub(super) struct Weights {
    pub logs: Vec<f64>,
    pub shape: Option<f64>,
    pub tail: usize,
    pub status: PsisStatus,
    pub ess: f64,
}

pub(super) fn smooth(logs: &[f64]) -> Result<Weights, ()> {
    if logs.is_empty() || logs.iter().any(|x| x.is_nan() || *x == f64::INFINITY) {
        return Err(());
    }
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(());
    }
    let mut output: Vec<f64> = logs.iter().map(|x| x - maximum).collect();
    // Zero-mass draws remain in the output but cannot determine a GPD tail fit.
    let mut order: Vec<usize> = (0..logs.len()).filter(|i| logs[*i].is_finite()).collect();
    order.sort_by(|a, b| logs[*a].total_cmp(&logs[*b]).then(a.cmp(b)));
    let count = order.len();
    let tail = ((count as f64 * 0.2).min(3. * (count as f64).sqrt()).floor() as usize)
        .min(count.saturating_sub(1));
    let mut shape = None;
    let mut status = PsisStatus::InsufficientTail;
    if tail >= 5 {
        let start = count - tail;
        let cutoff = output[order[start - 1]];
        let x: Vec<f64> = order[start..]
            .iter()
            .map(|i| {
                let value = output[*i];
                value.exp() * (-(cutoff - value).exp_m1())
            })
            .collect();
        let quartile = ((tail as f64 / 4. + 0.5).floor() as usize).saturating_sub(1);
        if output[order[count - 1]] - output[order[start]] <= f64::MIN_POSITIVE * 10.
            || x[quartile] <= 0.
            || !x.iter().all(|v| v.is_finite())
            || x[tail - 1] <= 0.
        {
            status = PsisStatus::DegenerateTail;
        } else {
            let (sigma, k) = gpd_fit(&x)?;
            let threshold = cutoff.exp();
            for (j, index) in order[start..].iter().enumerate() {
                let p = (j as f64 + 0.5) / tail as f64;
                let t = -(-p).ln_1p();
                let z = k * t;
                let quantile = if z.abs() < 1e-6 {
                    sigma * t * (1. + z / 2. + z * z / 6.)
                } else {
                    sigma * z.exp_m1() / k
                };
                let value = (threshold + quantile).ln().min(0.);
                if !value.is_finite() {
                    return Err(());
                }
                output[*index] = value;
            }
            shape = Some(k);
            status = PsisStatus::Smoothed;
        }
    }
    let sum = output.iter().map(|x| x.exp()).sum::<f64>();
    let squares = output.iter().map(|x| (2. * x).exp()).sum::<f64>();
    let ess = sum * sum / squares;
    if !ess.is_finite() || sum <= 0. {
        return Err(());
    }
    Ok(Weights {
        logs: output,
        shape,
        tail,
        status,
        ess,
    })
}

fn gpd_fit(x: &[f64]) -> Result<(f64, f64), ()> {
    let n = x.len();
    let m = 30 + (n as f64).sqrt().floor() as usize;
    let quartile = ((n as f64 / 4. + 0.5).floor() as usize).saturating_sub(1);
    let mut grid = Vec::with_capacity(m);
    let mut posterior = Vec::with_capacity(m);
    for i in 0..m {
        let theta =
            x[n - 1].recip() + (1. - (m as f64 / (i as f64 + 0.5)).sqrt()) / (3. * x[quartile]);
        let k = x.iter().map(|value| (-theta * value).ln_1p()).sum::<f64>() / n as f64;
        let likelihood = n as f64 * ((-theta / k).ln() - k - 1.);
        grid.push(theta);
        posterior.push(if likelihood.is_finite() {
            likelihood
        } else {
            f64::NEG_INFINITY
        });
    }
    let max = posterior.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(());
    }
    let sum = posterior.iter().map(|x| (x - max).exp()).sum::<f64>();
    let theta = grid
        .iter()
        .zip(&posterior)
        .map(|(x, p)| x * (p - max).exp() / sum)
        .sum::<f64>();
    let raw = x.iter().map(|value| (-theta * value).ln_1p()).sum::<f64>() / n as f64;
    let sigma = if theta == 0. {
        x.iter().sum::<f64>() / n as f64
    } else {
        -raw / theta
    };
    let k = (n as f64 * raw + 5.) / (n as f64 + 10.);
    if !sigma.is_finite() || sigma <= 0. || !k.is_finite() {
        return Err(());
    }
    Ok((sigma, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn executed_stan_238_weights_and_pareto_shape() {
        let mut lines =
            include_str!("../../../tests/fixtures/pathfinder/stan-2.38-psis.txt").lines();
        while let Some(header) = lines.next() {
            let fields: Vec<_> = header.split_whitespace().collect();
            assert_eq!(fields[0], "case");
            let count: usize = fields[2].parse().unwrap();
            let tail: usize = fields[3].parse().unwrap();
            let shape: f64 = fields[4].parse().unwrap();
            let rows: Vec<(f64, f64)> = lines
                .by_ref()
                .take(count)
                .map(|s| {
                    let mut v = s.split_whitespace().map(|x| x.parse::<f64>().unwrap());
                    (v.next().unwrap(), v.next().unwrap())
                })
                .collect();
            let result = smooth(&rows.iter().map(|r| r.0).collect::<Vec<_>>()).unwrap();
            assert_eq!(result.tail, tail);
            assert_eq!(result.status, PsisStatus::Smoothed);
            assert!((result.shape.unwrap() - shape).abs() < 2e-12);
            for (actual, (_, expected)) in result.logs.iter().zip(&rows) {
                assert!((actual.exp() - expected).abs() < 2e-12);
            }
        }
    }
    #[test]
    fn small_tied_zero_mass_and_nonfinite_weights_are_explicit() {
        assert_eq!(
            smooth(&[0., 1., 2.]).unwrap().status,
            PsisStatus::InsufficientTail
        );
        assert_eq!(
            smooth(&[0.; 64]).unwrap().status,
            PsisStatus::DegenerateTail
        );
        let mut logs: Vec<_> = (0..128).map(|i| (i as f64 + 1.).ln()).collect();
        logs[0] = f64::NEG_INFINITY;
        let result = smooth(&logs).unwrap();
        assert_eq!(result.logs[0], f64::NEG_INFINITY);
        assert!(result.ess.is_finite());
        for invalid in [
            vec![],
            vec![f64::NEG_INFINITY; 10],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(smooth(&invalid).is_err());
        }
    }
}
