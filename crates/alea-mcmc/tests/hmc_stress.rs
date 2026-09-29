#[path = "support/targets.rs"]
mod targets;
use alea_core::target::LogDensityGradient;
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::IdentityMetric;
use alea_mcmc::{Hmc, HmcOptions};
use rand::{SeedableRng, rngs::SmallRng};
use targets::Target;

#[test]
fn difficult_geometries_reject_without_publishing_partial_cache() {
    for (target, q, eps) in [
        (Target::Logistic, [12.0, -12.0], 10.0),
        (Target::Funnel, [-50.0, 0.1], 0.5),
        (Target::Rotated, [0.7, -0.4], 0.2),
    ] {
        let mut chain = Hmc::new(
            &target,
            OwnedBuffer::from_fn(2, |i| q[i]),
            IdentityMetric::new(2),
            HmcOptions::new(eps, 8).unwrap(),
        )
        .unwrap();
        let old_gradient = chain
            .point()
            .gradient()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        let old_density = chain.point().log_density().to_bits();
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..3 {
            let result = chain.step(&mut rng).unwrap();
            assert!(result.divergence.is_some(), "{target:?}: {result:?}");
            assert!(!result.accepted);
            assert_eq!(result.acceptance_probability, 0.0);
            assert!(
                chain
                    .point()
                    .position()
                    .iter()
                    .zip(q)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(
                chain
                    .point()
                    .gradient()
                    .iter()
                    .zip(&old_gradient)
                    .all(|(a, &b)| a.to_bits() == b)
            );
            assert_eq!(chain.point().log_density().to_bits(), old_density);
        }
        chain.set_position(&[0.0, 0.0]).unwrap();
        let mut g = [0.0; 2];
        let lp = target.logp_grad(&[0.0, 0.0], &mut g).unwrap();
        // Transcendental functions have unspecified precision; Miri may vary
        // their low bits between evaluations. Recomputed values need numerical
        // agreement, unlike the unchanged cache above, which must be bit-exact.
        for (actual, expected) in chain.point().gradient().iter().zip(g) {
            assert!((actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()));
        }
        assert!((chain.point().log_density() - lp).abs() <= 1e-12 * (1.0 + lp.abs()));
    }
}

#[test]
fn reference_gradients_remain_finite_in_representable_tails() {
    for target in [Target::Logistic, Target::Funnel, Target::Rotated] {
        for q in [[30.0, -30.0], [-30.0, 1e-6], [0.0, 0.0], [1e-6, -1e-6]] {
            let mut g = [0.0; 2];
            let value = target.logp_grad(&q, &mut g).unwrap();
            assert!(value.is_finite() && g.iter().all(|x| x.is_finite()));
        }
    }
}
