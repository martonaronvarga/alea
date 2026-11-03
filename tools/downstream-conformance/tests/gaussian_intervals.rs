//! Fixed, predeclared marginal-coverage smoke cohort. No seed searching,
//! dropped fits, simultaneous-interval claim, or optimizer/sampler replacement.
#![cfg(feature = "faer")]
use alea::{
    distribution::Sample,
    distributions::{
        modeling::regression::{RegressionData, diagnostics::GaussianIntervalOptions},
        scalar::Normal,
    },
};
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn gaussian_marginal_intervals_cover_in_a_fixed_twelve_check_family() {
    const REPETITIONS: usize = 4096;
    const FAMILY: usize = 12; // Two levels times (1+2+3) coefficient margins.
    const FAMILY_ERROR: f64 = 0.001;
    let allowance = ((REPETITIONS as f64 / 2.) * (2. * FAMILY as f64 / FAMILY_ERROR).ln()).sqrt();
    let normal = Normal::new(0., 1.).unwrap();
    let mut total_checks = 0;
    for (case, (n, p, scale)) in [(6, 1, 1.), (12, 2, 4.), (32, 3, 0.25)]
        .into_iter()
        .enumerate()
    {
        let seed = 0x4753_4956_0000_0000_u64 + case as u64;
        let mut rng = SmallRng::seed_from_u64(seed);
        let beta: Vec<_> = (0..p).map(|j| [1.5, -0.75, 0.25][j] * scale).collect();
        let design: Vec<_> = (0..n)
            .flat_map(|i| {
                let x = (i as f64 - (n - 1) as f64 / 2.) / n as f64;
                (0..p).map(move |j| [1., x, x * x][j])
            })
            .collect();
        let offsets: Vec<_> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    0.25 * scale
                } else {
                    -0.5 * scale
                }
            })
            .collect();
        let mut covered = vec![vec![0; p]; 2];
        let mut failed = 0;
        for _ in 0..REPETITIONS {
            let response: Vec<_> = (0..n)
                .map(|i| {
                    let mean =
                        (0..p).map(|j| design[i * p + j] * beta[j]).sum::<f64>() + offsets[i];
                    mean + scale * normal.sample(&mut rng).unwrap()
                })
                .collect();
            let data = RegressionData::new(p, design.clone(), response)
                .unwrap()
                .with_offsets(offsets.clone())
                .unwrap();
            for (k, level) in [0.90, 0.95].into_iter().enumerate() {
                let options = GaussianIntervalOptions {
                    confidence_level: level,
                    relative_rank_tolerance: 1e-12,
                    relative_residual_tolerance: 1e-12,
                    quantile_relative_tolerance: 1e-8,
                };
                match data.gaussian_coefficient_intervals(options) {
                    Ok(result) => {
                        assert_eq!(
                            result.assessment.least_squares.residual_degrees_of_freedom,
                            n - p
                        );
                        for (j, &truth) in beta.iter().enumerate() {
                            let [lo, hi] = result.bounds[j];
                            covered[k][j] += usize::from(lo <= truth && truth <= hi);
                        }
                    }
                    Err(error) => {
                        failed += 1;
                        eprintln!("case={case}, level={level}, failure={error}");
                    }
                }
            }
        }
        assert_eq!(failed, 0, "case={case}: failed intervals are not dropped");
        for (k, level) in [0.90, 0.95].into_iter().enumerate() {
            for (j, &count) in covered[k].iter().enumerate() {
                eprintln!(
                    "seed={seed}, n={n}, p={p}, level={level}, coefficient={j}, covered={count}/{REPETITIONS}, family={FAMILY}, allowance={allowance}"
                );
                // Hoeffding + union bound, fixed before observing results. This
                // checks each MARGINAL coverage, not joint coverage of intervals.
                assert!((count as f64 - REPETITIONS as f64 * level).abs() <= allowance);
                total_checks += 1;
            }
        }
    }
    assert_eq!(total_checks, FAMILY);
}
