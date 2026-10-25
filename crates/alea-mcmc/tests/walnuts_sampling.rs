use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::walnuts::{
    Walnuts, WalnutsMicrostepDistribution, WalnutsOptions, WalnutsTermination,
};
use rand::{SeedableRng, rngs::SmallRng};

#[path = "support/targets.rs"]
mod targets;

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |index| values[index])
}

#[test]
#[cfg_attr(miri, ignore = "long stationary WALNUTS sampling gate")]
fn four_targets_have_calibrated_moments_without_discarding_failed_extensions() {
    for target in [
        targets::Target::Correlated,
        targets::Target::Logistic,
        targets::Target::Banana,
        targets::Target::Funnel,
    ] {
        let (macro_step, tolerance) = if target == targets::Target::Funnel {
            (0.36, 0.21)
        } else {
            (0.3, 0.3)
        };
        let options = WalnutsOptions::new(macro_step, 8)
            .unwrap()
            .with_local_refinement(10, 1, tolerance)
            .unwrap();
        let mut batches = Vec::new();
        let mut numerical_stops = 0;
        let mut reverse_stops = 0;
        for seed in 701..705 {
            let sign = if seed % 2 == 0 { -1.0 } else { 1.0 };
            let mut chain = Walnuts::new(
                &target,
                buffer(&[sign, -sign]),
                IdentityMetric::new(2),
                options,
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut batch = [0.0; 5];
            for draw in 0..(1024 + 8192) {
                let info = chain.step(&mut rng).unwrap();
                numerical_stops += usize::from(matches!(
                    info.termination,
                    WalnutsTermination::Divergence(_) | WalnutsTermination::RefinementLimit
                ));
                reverse_stops +=
                    usize::from(info.termination == WalnutsTermination::ReverseSelection);
                let position = chain.point().position();
                assert!(position.iter().all(|value| value.is_finite()));
                assert!(
                    chain
                        .point()
                        .gradient()
                        .iter()
                        .all(|value| value.is_finite())
                );
                if draw >= 1024 {
                    for (sum, value) in batch.iter_mut().zip(target.observables(position)) {
                        *sum += value / 256.0;
                    }
                    if (draw - 1024 + 1) % 256 == 0 {
                        batches.push(batch);
                        batch = [0.0; 5];
                    }
                }
            }
        }
        let references = if target == targets::Target::Logistic {
            // Independently executed BlackJAX fixture used by the NUTS suite.
            [
                (-0.064_938_517_236_270_75, 0.013_312_099_811_553_645),
                (0.590_576_608_785_205_1, 0.005_674_950_972_920_785),
                (0.573_053_090_850_195, 0.007_239_998_752_135_707),
                (0.836_001_389_074_071_8, 0.015_449_974_270_321_05),
                (-0.085_315_783_890_813_37, 0.007_303_196_442_619_615),
            ]
        } else {
            [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
        };
        for observable in 0..5 {
            let count = batches.len() as f64;
            let mean = batches.iter().map(|batch| batch[observable]).sum::<f64>() / count;
            let se = (batches
                .iter()
                .map(|batch| (batch[observable] - mean).powi(2))
                .sum::<f64>()
                / (count * (count - 1.0)))
                .sqrt();
            let (reference, reference_se) = references[observable];
            assert!(
                (mean - reference).abs() <= 6.0 * se.hypot(reference_se) + 0.03,
                "{target:?} observable {observable}: mean={mean}, se={se}, reference={reference}, numerical_stops={numerical_stops}, reverse_stops={reverse_stops}"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "long stationary WALNUTS-R2P sampling gate")]
fn randomized_two_point_has_calibrated_correlated_gaussian_moments() {
    let target = targets::Target::Correlated;
    let options = WalnutsOptions::new(0.3, 8)
        .unwrap()
        .with_local_refinement(10, 1, 0.3)
        .unwrap()
        .with_microstep_distribution(WalnutsMicrostepDistribution::RandomizedTwoPoint)
        .unwrap()
        .with_macro_step_jitter(0.2)
        .unwrap();
    let mut batches = Vec::new();
    for seed in 801..805 {
        let sign = if seed % 2 == 0 { -1.0 } else { 1.0 };
        let mut chain = Walnuts::new(
            &target,
            buffer(&[sign, -sign]),
            IdentityMetric::new(2),
            options,
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut batch = [0.0; 5];
        for draw in 0..(512 + 4096) {
            let _ = chain.step(&mut rng).unwrap();
            if draw >= 512 {
                for (sum, value) in batch
                    .iter_mut()
                    .zip(target.observables(chain.point().position()))
                {
                    *sum += value / 128.0;
                }
                if (draw - 512 + 1) % 128 == 0 {
                    batches.push(batch);
                    batch = [0.0; 5];
                }
            }
        }
    }
    let references = [0.0, 0.0, 1.0, 1.0, 0.0];
    for observable in 0..5 {
        let count = batches.len() as f64;
        let mean = batches.iter().map(|batch| batch[observable]).sum::<f64>() / count;
        let se = (batches
            .iter()
            .map(|batch| (batch[observable] - mean).powi(2))
            .sum::<f64>()
            / (count * (count - 1.0)))
            .sqrt();
        assert!(
            (mean - references[observable]).abs() <= 6.0 * se + 0.03,
            "observable {observable}: mean={mean}, se={se}"
        );
    }
}
