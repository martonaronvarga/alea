//! Reusable downstream contract checks for Alea's open extension traits.
//!
//! This package is intentionally outside Alea's production workspace. Its tests
//! compile as an ordinary dependent package and cannot access crate-private APIs.
#![forbid(unsafe_code)]

use alea::math::prelude::EuclideanMetric;
use alea::output::DiagnosticTransition;
use alea::prelude::{LogDensityGradient, evaluate};
use alea::sampling::prelude::MarkovChain;
use rand::Rng;

/// One independently known target value and gradient.
#[derive(Debug, Clone, Copy)]
pub struct TargetCase<'a> {
    /// Unconstrained coordinates.
    pub position: &'a [f64],
    /// Expected finite log density.
    pub log_density: f64,
    /// Expected full gradient in coordinate order.
    pub gradient: &'a [f64],
}

fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance * (1.0 + expected.abs())
}

/// Checks the fixed-dimension, pure, complete-output target contract.
///
/// The cases should be independent mathematical fixtures rather than values
/// produced by the implementation under test.
///
/// # Panics
///
/// Panics when inputs are empty or malformed, the tolerance is invalid, or the
/// implementation violates a target law or disagrees with a fixture.
pub fn assert_target_conformance<T>(target: &T, cases: &[TargetCase<'_>], tolerance: f64)
where
    T: LogDensityGradient + ?Sized,
{
    assert!(!cases.is_empty(), "target conformance requires fixtures");
    assert!(
        tolerance.is_finite() && tolerance >= 0.0,
        "target tolerance must be finite and nonnegative"
    );
    let dimension = target.dimension();
    for case in cases {
        assert_eq!(case.position.len(), dimension, "fixture position shape");
        assert_eq!(case.gradient.len(), dimension, "fixture gradient shape");
        let mut gradient = vec![f64::NAN; dimension];
        let value = evaluate(target, case.position, &mut gradient)
            .expect("conforming target must evaluate a declared fixture");
        assert!(
            close(value, case.log_density, tolerance),
            "target log density differs: actual={value}, expected={}",
            case.log_density
        );
        for (index, (&actual, &expected)) in gradient.iter().zip(case.gradient).enumerate() {
            assert!(
                close(actual, expected, tolerance),
                "target gradient {index} differs: actual={actual}, expected={expected}"
            );
        }

        let mut repeated = vec![f64::NAN; dimension];
        let repeated_value = evaluate(target, case.position, &mut repeated)
            .expect("conforming target must be repeatable");
        assert!(close(repeated_value, value, tolerance));
        assert!(
            repeated
                .iter()
                .zip(&gradient)
                .all(|(&left, &right)| close(left, right, tolerance)),
            "target evaluation is not semantically pure"
        );
    }

    let bad_position = vec![0.0; dimension.saturating_add(1)];
    let mut untouched = vec![41.0; dimension];
    assert!(evaluate(target, &bad_position, &mut untouched).is_err());
    assert!(untouched.iter().all(|&value| value == 41.0));

    let position = vec![0.0; dimension];
    let mut bad_gradient = vec![42.0; dimension.saturating_add(1)];
    assert!(evaluate(target, &position, &mut bad_gradient).is_err());
    assert!(bad_gradient.iter().all(|&value| value == 42.0));
}

/// Checks shape atomicity and Gaussian mass/velocity/energy consistency.
///
/// For every standard-normal probe `z`, a conforming constant metric must map
/// `z` to momentum with kinetic energy `dot(z, z) / 2`.
///
/// # Panics
///
/// Panics when probes are absent or malformed, the tolerance is invalid, or a
/// metric method violates its declared algebra or mutates output on shape error.
pub fn assert_metric_conformance<M>(metric: &M, probes: &[&[f64]], tolerance: f64)
where
    M: EuclideanMetric + ?Sized,
{
    assert!(!probes.is_empty(), "metric conformance requires probes");
    assert!(
        tolerance.is_finite() && tolerance >= 0.0,
        "metric tolerance must be finite and nonnegative"
    );
    let dimension = metric.dimension();
    assert!(metric.log_det().is_finite(), "metric log determinant");
    for probe in probes {
        assert_eq!(probe.len(), dimension, "metric probe shape");
        let mut momentum = vec![0.0; dimension];
        metric
            .sample_momentum(probe, &mut momentum)
            .expect("metric must sample a declared finite probe");
        assert!(momentum.iter().all(|value| value.is_finite()));
        let mut velocity = vec![0.0; dimension];
        metric
            .velocity(&momentum, &mut velocity)
            .expect("metric must apply inverse mass to sampled momentum");
        assert!(velocity.iter().all(|value| value.is_finite()));
        let mut scratch = vec![0.0; dimension];
        let kinetic = metric
            .kinetic_energy(&momentum, &mut scratch)
            .expect("metric kinetic energy must accept matching storage");
        let expected = 0.5 * probe.iter().map(|value| value * value).sum::<f64>();
        assert!(
            kinetic.is_finite() && close(kinetic, expected, tolerance),
            "metric kinetic mismatch: actual={kinetic}, expected={expected}"
        );
    }

    let bad_source = vec![0.0; dimension.saturating_add(1)];
    let mut untouched = vec![43.0; dimension];
    assert!(metric.sample_momentum(&bad_source, &mut untouched).is_err());
    assert!(untouched.iter().all(|&value| value == 43.0));
    assert!(metric.velocity(&bad_source, &mut untouched).is_err());
    assert!(untouched.iter().all(|&value| value == 43.0));

    let source = vec![0.0; dimension];
    let mut bad_destination = vec![45.0; dimension.saturating_add(1)];
    assert!(
        metric
            .sample_momentum(&source, &mut bad_destination)
            .is_err()
    );
    assert!(bad_destination.iter().all(|&value| value == 45.0));
    assert!(metric.velocity(&source, &mut bad_destination).is_err());
    assert!(bad_destination.iter().all(|&value| value == 45.0));

    let mut bad_scratch = vec![47.0; dimension.saturating_add(1)];
    assert!(metric.kinetic_energy(&source, &mut bad_scratch).is_err());
    assert!(bad_scratch.iter().all(|&value| value == 47.0));

    let bad_momentum = vec![0.0; dimension.saturating_add(1)];
    let mut scratch = vec![49.0; dimension];
    assert!(metric.kinetic_energy(&bad_momentum, &mut scratch).is_err());
    assert!(scratch.iter().all(|&value| value == 49.0));
}

/// Exercises a chain together with its diagnostic adapter.
///
/// Successful positions must retain one finite dimension. Errors must preserve
/// the exact pre-step position, and diagnostic projection must be deterministic
/// with finite probabilities and optional energy.
///
/// # Panics
///
/// Panics if `steps` is zero, the initial position is empty/nonfinite, no step
/// succeeds, or the chain/diagnostic contracts are violated.
pub fn assert_chain_conformance<C, R>(chain: &mut C, rng: &mut R, steps: usize)
where
    C: MarkovChain,
    C::Transition: DiagnosticTransition,
    R: Rng + ?Sized,
{
    assert!(steps > 0, "chain conformance requires attempted steps");
    let dimension = chain.position().len();
    assert!(dimension > 0, "runtime chains must have nonempty positions");
    assert!(chain.position().iter().all(|value| value.is_finite()));
    let mut successes = 0;
    for _ in 0..steps {
        let before = chain.position().to_vec();
        match chain.step(rng) {
            Ok(transition) => {
                successes += 1;
                let position = chain.position();
                assert_eq!(position.len(), dimension, "chain dimension changed");
                assert!(position.iter().all(|value| value.is_finite()));
                let first = transition.diagnostic_stats();
                let second = transition.diagnostic_stats();
                assert_eq!(first, second, "diagnostic projection is stateful");
                assert!(
                    first.acceptance_probability.is_finite()
                        && (0.0..=1.0).contains(&first.acceptance_probability),
                    "invalid diagnostic acceptance probability"
                );
                assert!(first.energy.is_none_or(f64::is_finite));
            }
            Err(_) => assert_eq!(
                chain.position(),
                before,
                "chain error changed live position"
            ),
        }
    }
    assert!(
        successes > 0,
        "chain conformance observed no successful step"
    );
}
pub mod modeling;

/// Checks repeatable normalized scalar values against independent measure fixtures.
///
/// Support zeros and boundary infinities must match exactly; finite values use
/// the supplied absolute/relative tolerance. This does not numerically integrate
/// an arbitrary distribution or certify its normalization beyond the fixtures.
///
/// # Panics
/// Panics on invalid fixtures/tolerance, errors, NaNs or disagreement.
pub fn assert_distribution_conformance<D>(distribution: &D, cases: &[(f64, f64)], tolerance: f64)
where
    D: alea::distribution::Distribution<Event = f64>,
{
    assert!(!cases.is_empty());
    assert!(tolerance.is_finite() && tolerance >= 0.0);
    let shape = distribution.event_shape();
    let support = distribution.support();
    for (event, expected) in cases {
        let actual = distribution
            .log_prob(event)
            .expect("distribution fixture must evaluate");
        let repeated = distribution
            .log_prob(event)
            .expect("distribution fixture must repeat");
        assert_eq!(actual, repeated);
        if expected.is_infinite() {
            assert_eq!(actual, *expected);
        } else {
            assert!(expected.is_finite() && actual.is_finite());
            assert!(close(actual, *expected, tolerance));
        }
        assert_eq!(distribution.event_shape(), shape);
        assert_eq!(distribution.support(), support);
    }
}

pub mod profiling;
