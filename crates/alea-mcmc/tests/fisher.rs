use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use alea_mcmc::{
    HmcOptions,
    adapt::{FisherHmcWarmup, FisherMetricAdapter, FisherOptions},
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[test]
fn initialization_is_separate_and_identity_is_tail_robust() {
    use alea_mcmc::adapt::MetricInitialization;
    let score = [0.0, 1e100, -1e100];
    let mut scales = [7.0; 3];
    MetricInitialization::default()
        .scales_into(&score, &mut scales)
        .unwrap();
    assert_eq!(scales, [1.0; 3]);
    MetricInitialization::ClippedScore
        .scales_into(&score, &mut scales)
        .unwrap();
    assert_eq!(scales, [1.0, 1e-10, 1e-10]);
    let old = scales;
    assert!(
        MetricInitialization::Identity
            .scales_into(&[f64::NAN; 3], &mut scales)
            .is_err()
    );
    assert_eq!(scales, old);
}

#[test]
fn diagonal_updates_match_independent_decimal_batch_fixtures() {
    let mut adapter = FisherMetricAdapter::new(2, 3, 6).unwrap();
    let mut cases = 0;
    for line in include_str!("fixtures/fisher-diagonal.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
    {
        let row: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
        assert_eq!(row.len(), 8);
        adapter.observe(&row[1..3], &row[3..5]).unwrap();
        assert_eq!(adapter.count(), row[5] as usize);
        let mut out = [0.0; 2];
        adapter.scales_into(&[1.0; 2], 1e-5, &mut out).unwrap();
        for (actual, expected) in out.into_iter().zip(&row[6..8]) {
            assert!((actual - expected).abs() < 2e-12 * expected.abs());
        }
        cases += 1;
    }
    assert_eq!(cases, 13);
}

#[test]
fn diagonal_estimation_handles_extreme_standardization_without_intermediate_overflow() {
    for magnitude in [1e-150, 1.0, 1e150] {
        for fallback in [1e-10, 1.0, 1e10] {
            for ridge in [1e-305, 1e-5] {
                let mut adapter = FisherMetricAdapter::new(1, 3, 0).unwrap();
                adapter.observe(&[-magnitude], &[magnitude]).unwrap();
                adapter.observe(&[magnitude], &[-magnitude]).unwrap();
                let mut actual = [0.0];
                adapter
                    .scales_into(&[fallback], ridge, &mut actual)
                    .unwrap();
                // Equivalent unstandardized formula; these additions/products
                // remain representable for the predeclared grid above.
                let scatter = 2.0 * magnitude * magnitude;
                let c = scatter + ridge * fallback * fallback;
                let f = scatter + ridge / fallback / fallback;
                let expected = ((c.ln() - f.ln()) * 0.25).exp().clamp(1e-10, 1e10);
                assert!(
                    (actual[0] - expected).abs() < 2e-12 * expected,
                    "magnitude={magnitude}, fallback={fallback}, ridge={ridge}: {} != {expected}",
                    actual[0]
                );
            }
        }
    }
}
#[test]
fn paired_diagonal_matches_fisher_formula_and_discards_transients() {
    let mut a = FisherMetricAdapter::new(2, 3, 6).unwrap();
    let mut scales = [0.0; 2];
    for q in [[-2.0, -4.0], [0.0, 0.0], [2.0, 4.0]] {
        a.observe(&q, &[-q[0] / 4.0, -q[1] / 16.0]).unwrap();
    }
    a.scales_into(&[2.0, 4.0], 1e-5, &mut scales).unwrap();
    for (actual, expected) in scales.into_iter().zip([2.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(a.observe(&[1.0, 2.0], &[f64::NAN, 0.0]).is_err());
    assert_eq!(a.count(), 3);
    assert!(a.observe(&[f64::MAX; 2], &[f64::MAX; 2]).is_err());
    assert_eq!(a.count(), 3);
    for _ in 0..4 {
        a.observe(&[0.0; 2], &[0.0; 2]).unwrap();
    }
    assert_eq!(a.count(), 4); // The first three observations are no longer foreground.
    a.scales_into(&[2.0, 4.0], 1e-5, &mut scales).unwrap();
    assert!((scales[0] - 2.0).abs() < 1e-12);
    a.reset(80).unwrap();
    assert_eq!(a.count(), 0);
}

#[test]
fn foreground_windows_match_centered_batch_after_failed_updates() {
    let mut adapter = FisherMetricAdapter::new(2, 3, 6).unwrap();
    let positions: Vec<[f64; 2]> = (0..13)
        .map(|i| [i as f64 - 4.0, (i * i % 11) as f64])
        .collect();
    let scores: Vec<[f64; 2]> = positions
        .iter()
        .map(|q| [-0.5 * q[0] + q[1], q[0] - 2.0 * q[1]])
        .collect();
    for n in 1..=positions.len() {
        // Exercise both the first background reset and subsequent swaps. The
        // first coordinate is valid; the second overflows while preparing moments.
        if n == 4 || n == 7 || n == 10 {
            let count = adapter.count();
            assert!(adapter.observe(&[0.0; 2], &[0.0, f64::MAX]).is_err());
            assert_eq!(adapter.count(), count);
        }
        adapter.observe(&positions[n - 1], &scores[n - 1]).unwrap();
        let start = if n <= 6 { 0 } else { ((n - 1) / 3 - 1) * 3 };
        assert_eq!(adapter.count(), n - start);
        let mut actual = [0.0; 2];
        adapter.scales_into(&[1.0; 2], 1e-5, &mut actual).unwrap();
        for axis in 0..2 {
            let scatter = |data: &[[f64; 2]]| {
                let mean = data[start..n].iter().map(|x| x[axis]).sum::<f64>() / (n - start) as f64;
                data[start..n]
                    .iter()
                    .map(|x| (x[axis] - mean).powi(2))
                    .sum::<f64>()
            };
            let expected = ((scatter(&positions) + 1e-5) / (scatter(&scores) + 1e-5)).powf(0.25);
            assert!((actual[axis] - expected).abs() < 1e-11 * (1.0 + expected));
        }
    }
}

#[cfg(feature = "faer")]
#[test]
fn low_rank_history_is_bounded_and_cannot_be_reinterpreted_at_another_dimension() {
    use alea_mcmc::adapt::FisherError;
    let mut adapter = FisherMetricAdapter::new(2, 3, 6).unwrap();
    let mut q = Vec::new();
    let mut s = Vec::new();
    for i in 0..13 {
        let point = [i as f64 - 4.0, (i * i % 11) as f64];
        let score = [-2.0 * point[0] + 0.5 * point[1], 0.5 * point[0] - point[1]];
        adapter.observe(&point, &score).unwrap();
        q.extend_from_slice(&point);
        s.extend_from_slice(&score);
    }
    assert!(matches!(
        adapter.fit_low_rank(&[1.0], 1e-5, 1.0, 1),
        Err(FisherError::Configuration)
    ));
    let fitted = adapter.fit_low_rank(&[1.0; 2], 1e-5, 1.0, 2).unwrap();
    let expected = alea_math::fisher::fit_low_rank(
        &q[q.len() - 12..],
        &s[s.len() - 12..],
        &[1.0; 2],
        1e-5,
        1.0,
        2,
    )
    .unwrap();
    for p in [[1.0, 0.0], [0.0, 1.0]] {
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        fitted.velocity(&p, &mut a).unwrap();
        expected.velocity(&p, &mut b).unwrap();
        for (a, b) in a.into_iter().zip(b) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()));
        }
    }
}
#[test]
fn fisher_warmup_freezes_and_zero_warmup_preserves_rng() {
    let target = Gaussian::new(2);
    let options = HmcOptions::new(0.1, 3).unwrap();
    let mut rng = SmallRng::seed_from_u64(517);
    let mut untouched = rng.clone();
    let (zero, report) =
        FisherHmcWarmup::new(&target, OwnedBuffer::new(2), options, FisherOptions::new(0))
            .unwrap()
            .run(&mut rng)
            .unwrap();
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    assert_eq!(report.metric_updates, 0);
    assert_eq!(zero.metric().scales(), &[1.0; 2]);
    let (mut chain, report) = FisherHmcWarmup::new(
        &target,
        OwnedBuffer::new(2),
        options,
        FisherOptions::new(100),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(report.metric_updates, 85);
    let scales = chain.metric().scales().to_vec();
    let step = chain.options().step_size();
    for _ in 0..10 {
        let _ = chain.step(&mut rng).unwrap();
    }
    assert_eq!(chain.metric().scales(), &scales);
    assert_eq!(chain.options().step_size(), step);
    let mut v = [0.0; 2];
    chain.metric().velocity(&[1.0; 2], &mut v).unwrap();
    assert!(v.iter().all(|x| x.is_finite() && *x > 0.0));
}
#[cfg(not(miri))]
#[test]
fn paired_observation_and_diagonal_estimation_do_not_allocate() {
    let mut a = FisherMetricAdapter::new(2, 10, 20).unwrap();
    let mut scales = [1.0; 2];
    let info = allocation_counter::measure(|| {
        for _ in 0..100 {
            a.observe(&[0.2, -0.3], &[-0.2, 0.3]).unwrap();
            a.scales_into(&[1.0; 2], 1e-5, &mut scales).unwrap();
        }
    });
    assert_eq!(info.count_total, 0);
}
#[cfg(feature = "faer")]
#[test]
fn low_rank_warmup_runs_with_bounded_history() {
    let target = Gaussian::new(3);
    let mut rng = SmallRng::seed_from_u64(982);
    let (mut chain, report) = FisherHmcWarmup::new(
        &target,
        OwnedBuffer::new(3),
        HmcOptions::new(0.1, 3).unwrap(),
        FisherOptions::new(300).with_max_rank(2),
    )
    .unwrap()
    .run(&mut rng)
    .unwrap();
    assert_eq!(report.metric_updates, 255);
    assert!(chain.metric().rank() <= 2);
    for _ in 0..20 {
        assert!(chain.step(&mut rng).unwrap().divergence.is_none());
    }
}

#[cfg(all(feature = "faer", not(miri)))]
#[test]
fn fisher_low_rank_and_omf_recover_correlated_gaussian_moments() {
    use alea_core::target::LogDensityGradient;
    use alea_mcmc::integrator::TwoStage;
    struct Correlated;
    impl LogDensityGradient for Correlated {
        type Error = std::convert::Infallible;
        fn dimension(&self) -> usize {
            2
        }
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
            g[0] = -50.5 * q[0] + 49.5 * q[1];
            g[1] = 49.5 * q[0] - 50.5 * q[1];
            Ok(0.5 * (q[0] * g[0] + q[1] * g[1]))
        }
    }
    let mut sums = [0.0; 4];
    let mut divergent = 0;
    for seed in 201..205 {
        let mut rng = SmallRng::seed_from_u64(seed);
        // One macrostep isolates metric learning. Four near-periodic OMF steps
        // can resonate after whitening; fixed length is not a mixing guarantee.
        let (mut chain, _) = FisherHmcWarmup::new_with_integrator(
            &Correlated,
            OwnedBuffer::new(2),
            HmcOptions::new(0.1, 1).unwrap(),
            FisherOptions::new(1000).with_max_rank(2),
            TwoStage::OMF,
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        assert_eq!(chain.metric().rank(), 2);
        let mut velocity = [0.0; 2];
        chain.metric().velocity(&[1.0, 0.0], &mut velocity).unwrap();
        assert!((velocity[0] - 0.505).abs() < 0.002);
        assert!((velocity[1] - 0.495).abs() < 0.002);
        for _ in 0..4096 {
            let info = chain.step(&mut rng).unwrap();
            divergent += usize::from(info.divergence.is_some());
            let q = chain.point().position();
            let a = (q[0] + q[1]) / 2.0_f64.sqrt();
            let b = (q[0] - q[1]) * 10.0 / 2.0_f64.sqrt();
            for (sum, value) in sums.iter_mut().zip([a, b, a * a, b * b]) {
                *sum += value;
            }
        }
    }
    assert_eq!(divergent, 0);
    for (sum, expected) in sums.into_iter().zip([0.0, 0.0, 1.0, 1.0]) {
        assert!(
            (sum / 16384.0 - expected).abs() < 0.10,
            "moment {}",
            sum / 16384.0
        );
    }
}
