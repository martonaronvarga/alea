//! Statistical gates use independent RNG streams, not identical-chain assertions.
#![cfg(not(miri))]
use alea_core::target::LogDensityGradient;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric},
};
use alea_mcmc::{
    Hmc, HmcOptions,
    adapt::{HmcWarmup, MetricKind, WarmupOptions},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::convert::Infallible;

struct Rotated;
impl LogDensityGradient for Rotated {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        let x = (0.8 * q[0] + 0.6 * q[1]) / 0.1;
        let y = -0.6 * q[0] + 0.8 * q[1];
        g[0] = -8.0 * x + 0.6 * y;
        g[1] = -6.0 * x - 0.8 * y;
        Ok(-0.5 * (x * x + y * y))
    }
}
fn collect<M: EuclideanMetric>(
    chain: &mut Hmc<'_, Rotated, M>,
    rng: &mut SmallRng,
    out: &mut Vec<[f64; 5]>,
) {
    for _ in 0..8192 {
        let info = chain.step(rng).unwrap();
        assert!(info.divergence.is_none());
        assert_eq!(info.integration_steps, 5);
        let q = chain.point().position();
        let x = (0.8 * q[0] + 0.6 * q[1]) / 0.1;
        let y = -0.6 * q[0] + 0.8 * q[1];
        out.push([x, y, x * x, y * y, x * y]);
    }
}
fn statistics(samples: &[[f64; 5]]) -> ([f64; 5], [f64; 5], [f64; 5]) {
    let mean =
        std::array::from_fn(|i| samples.iter().map(|x| x[i]).sum::<f64>() / samples.len() as f64);
    let variance: [f64; 5] = std::array::from_fn(|i| {
        samples
            .iter()
            .map(|x| (x[i] - mean[i]).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64
    });
    let mut mcse = [0.0_f64; 5];
    for batch in [128, 256] {
        let means: Vec<[f64; 5]> = samples
            .chunks_exact(batch)
            .map(|xs| std::array::from_fn(|i| xs.iter().map(|x| x[i]).sum::<f64>() / batch as f64))
            .collect();
        for i in 0..5 {
            let variance = means.iter().map(|x| (x[i] - mean[i]).powi(2)).sum::<f64>()
                / (means.len() - 1) as f64;
            mcse[i] = mcse[i].max((variance / means.len() as f64).sqrt());
        }
    }
    let efficiency =
        std::array::from_fn(|i| variance[i] / mcse[i].powi(2) / (samples.len() * 5) as f64);
    (mean, mcse, efficiency)
}

#[test]
fn adapted_sampling_matches_blackjax_and_improves_ess_per_gradient_without_detected_bias() {
    let rows: Vec<_> = include_str!("fixtures/blackjax-warmup-sampling.csv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .skip(1)
        .collect();
    assert_eq!(rows.len(), 10);
    let mut efficiencies = Vec::new();
    for kind in [None, Some(MetricKind::Diagonal), Some(MetricKind::Dense)] {
        // Warmup trajectories legitimately separate after backend rounding.
        // Sixteen consecutive seeds reduce dependence of the efficiency gate on
        // a few near-resonant fixed-length chains.
        let mut samples = Vec::with_capacity(16 * 8192);
        for seed in 701..717 {
            let mut rng = SmallRng::seed_from_u64(seed);
            if let Some(kind) = kind {
                let (mut chain, report) = HmcWarmup::new(
                    &Rotated,
                    OwnedBuffer::new(2),
                    HmcOptions::new(0.1, 5).unwrap(),
                    WarmupOptions::new(1000).with_metric(kind),
                )
                .unwrap()
                .run(&mut rng)
                .unwrap();
                assert_eq!(report.metric_updates, 5);
                collect(&mut chain, &mut rng, &mut samples);
            } else {
                let mut chain = Hmc::new(
                    &Rotated,
                    OwnedBuffer::new(2),
                    IdentityMetric::new(2),
                    HmcOptions::new(0.08, 5).unwrap(),
                )
                .unwrap();
                for _ in 0..1000 {
                    let _ = chain.step(&mut rng).unwrap();
                }
                collect(&mut chain, &mut rng, &mut samples);
            }
        }
        let (mean, mcse, efficiency) = statistics(&samples);
        for i in 0..5 {
            let expected = [0.0, 0.0, 1.0, 1.0, 0.0][i];
            assert!(mcse[i] > 0.0 && mcse[i] < 0.12, "{kind:?} {mcse:?}");
            assert!(
                (mean[i] - expected).abs() < 6.0 * mcse[i] + 0.01,
                "{kind:?} observable {i}: {} +/- {}",
                mean[i],
                mcse[i]
            );
            if let Some(kind) = kind {
                let f: Vec<_> = rows[if kind == MetricKind::Diagonal {
                    i
                } else {
                    5 + i
                }]
                .split(',')
                .collect();
                assert_eq!(&f[1..5], &["4", "1000", "8192", "5"]);
                assert_eq!(f[5].parse::<usize>().unwrap(), i);
                let reference: f64 = f[6].parse().unwrap();
                let error: f64 = f[7].parse().unwrap();
                assert!((mean[i] - reference).abs() < 6.0 * mcse[i].hypot(error) + 0.01);
            }
        }
        eprintln!("{kind:?}: mean={mean:?}, mcse={mcse:?}, ESS/gradient={efficiency:?}");
        efficiencies.push(efficiency);
    }
    // Test both the slow-axis location and scale, avoiding a mean-only mixing claim.
    for i in [1, 3] {
        assert!(efficiencies[2][i] > 2.0 * efficiencies[0][i]);
        // The milestone requires improvement, not a universal 2x improvement
        // over an already adapted diagonal baseline. Faer gives ~1.81x for the
        // squared observable on this ensemble (reported in docs/m4-warmup.md).
        assert!(efficiencies[2][i] > efficiencies[1][i]);
    }
}
