use alea_core::{density::LogDensity, target::EvaluationError};
use alea_distributions::Gaussian;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric},
};
use alea_mcmc::{
    Rwmh, RwmhOptions,
    rwmh::{NumericalRejection, RwmhError},
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::cell::Cell;

#[derive(Debug, thiserror::Error)]
#[error("injected failure")]
struct Failure;
struct Target {
    mode: Cell<u8>,
    dimension: Cell<usize>,
}
impl LogDensity for Target {
    type Error = Failure;
    fn dimension(&self) -> usize {
        self.dimension.get()
    }
    fn logp(&self, q: &[f64]) -> Result<f64, Failure> {
        match self.mode.get() {
            1 => {
                return Err(Failure);
            }
            2 => {
                panic!("injected panic");
            }
            3 => return Ok(f64::NEG_INFINITY),
            4 => return Ok(f64::NAN),
            _ => {}
        }
        Ok(Gaussian::new(q.len()).logp(q).unwrap())
    }
}

#[test]
fn validated_configuration_and_dimensions() {
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(RwmhOptions::new(invalid).is_err());
    }
    assert!(matches!(
        Rwmh::new(
            &Gaussian::new(0),
            OwnedBuffer::new(0),
            IdentityMetric::new(0),
            RwmhOptions::new(1.0).unwrap()
        ),
        Err(RwmhError::EmptyTarget)
    ));
    assert!(matches!(
        Rwmh::new(
            &Gaussian::new(2),
            OwnedBuffer::new(2),
            IdentityMetric::new(1),
            RwmhOptions::new(1.0).unwrap()
        ),
        Err(RwmhError::MetricDimension { .. })
    ));
}

#[test]
fn errors_panics_and_numerical_rejections_preserve_cache_and_recover() {
    let target = Target {
        mode: Cell::new(0),
        dimension: Cell::new(2),
    };
    let mut chain = Rwmh::new(
        &target,
        OwnedBuffer::from_fn(2, |_| 1.0),
        IdentityMetric::new(2),
        RwmhOptions::new(0.3).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(17);
    for mode in 1..5 {
        target.mode.set(mode);
        if mode == 2 {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| chain.step(&mut rng)))
                    .is_err()
            );
        } else if mode == 1 {
            assert!(matches!(
                chain.step(&mut rng),
                Err(RwmhError::Evaluation(EvaluationError::Model(Failure)))
            ));
        } else {
            let info = chain.step(&mut rng).unwrap();
            assert!(!info.accepted);
            assert_eq!(info.acceptance_probability, 0.0);
            assert_eq!(
                info.numerical_rejection,
                Some(NumericalRejection::LogDensity)
            );
        }
        assert_eq!(chain.point().position(), &[1.0, 1.0]);
        assert_eq!(chain.point().log_density(), -1.0);
    }
    target.mode.set(0);
    target.dimension.set(3);
    let mut unchanged = rng.clone();
    assert!(matches!(
        chain.step(&mut rng),
        Err(RwmhError::Evaluation(
            EvaluationError::TargetDimensionChanged { .. }
        ))
    ));
    assert_eq!(rng.random::<u64>(), unchanged.random::<u64>());
    target.dimension.set(2);
    chain.set_position(&[0.0, 0.0]).unwrap();
    assert!(chain.step(&mut rng).unwrap().numerical_rejection.is_none());
}

#[test]
#[cfg_attr(miri, ignore = "long statistical gate; native CI covers it")]
fn gaussian_moments_for_all_proposal_covariances() {
    fn check<M: EuclideanMetric>(metric: M) {
        let target = Gaussian::new(2);
        let mut chain = Rwmh::new(
            &target,
            OwnedBuffer::new(2),
            metric,
            RwmhOptions::new(1.0).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(873);
        let (mut mean, mut square, mut cross) = ([0.0; 2], [0.0; 2], 0.0);
        for i in 0..62000 {
            let info = chain.step(&mut rng).unwrap();
            assert!(info.numerical_rejection.is_none());
            if i >= 2000 {
                let q = chain.point().position();
                for j in 0..2 {
                    mean[j] += q[j];
                    square[j] += q[j] * q[j];
                }
                cross += q[0] * q[1];
            }
        }
        for j in 0..2 {
            assert!((mean[j] / 60000.0).abs() < 0.07);
            assert!((square[j] / 60000.0 - 1.0).abs() < 0.1);
        }
        assert!((cross / 60000.0).abs() < 0.07);
    }
    check(IdentityMetric::new(2));
    check(DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| [2.0, 0.5][i])).unwrap());
    check(DenseMetric::new(
        CholeskyFactor::new_lower(2, OwnedBuffer::from_fn(4, |i| [1.5, 0.3, 0.0, 0.8][i])).unwrap(),
    ));
}

#[test]
#[cfg(not(miri))]
fn first_and_repeated_transitions_allocate_nothing() {
    let target = Gaussian::new(33);
    let mut chain = Rwmh::new(
        &target,
        OwnedBuffer::new(33),
        IdentityMetric::new(33),
        RwmhOptions::new(0.3).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(99);
    let counts = allocation_counter::measure(|| {
        for _ in 0..100 {
            let _ = std::hint::black_box(chain.step(&mut rng).unwrap());
        }
    });
    assert_eq!(counts.count_total, 0);
    assert_eq!(counts.bytes_total, 0);
}

#[test]
fn density_only_transitions_match_original_polar_and_metropolis_recurrence() {
    let target = Gaussian::new(1);
    let mut chain = Rwmh::new(
        &target,
        OwnedBuffer::from_fn(1, |_| 0.7),
        IdentityMetric::new(1),
        RwmhOptions::new(0.3).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(915);
    let mut reference_rng = rng.clone();
    let mut spare = None;
    let mut position = 0.7_f64;
    for _ in 0..20 {
        let normal = if let Some(value) = spare.take() {
            value
        } else {
            loop {
                let u = 2.0 * reference_rng.random::<f64>() - 1.0;
                let v = 2.0 * reference_rng.random::<f64>() - 1.0;
                let radius = u * u + v * v;
                if radius > 0.0 && radius < 1.0 {
                    let scale = (-2.0 * radius.ln() / radius).sqrt();
                    spare = Some(v * scale);
                    break u * scale;
                }
            }
        };
        let proposal = position + 0.3 * normal;
        let probability = (-0.5 * proposal * proposal + 0.5 * position * position)
            .min(0.0)
            .exp();
        let accepted = reference_rng.random::<f64>() < probability;
        if accepted {
            position = proposal;
        }
        let transition = chain.step(&mut rng).unwrap();
        assert_eq!(transition.accepted, accepted);
        assert!((transition.acceptance_probability - probability).abs() < 1e-12);
        assert!((chain.point().position()[0] - position).abs() < 1e-12);
        assert_eq!(rng, reference_rng);
    }
}
