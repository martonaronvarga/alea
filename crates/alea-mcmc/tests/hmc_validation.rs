//! Independent transition and stationary-distribution regressions, not ESS estimates.
use std::convert::Infallible;

use alea_core::target::LogDensityGradient;
use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
};
use alea_mcmc::hmc::Divergence;
use alea_mcmc::{Hmc, HmcOptions};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

// Share an immutable metric across independent test chains without cloning its
// buffers or broadening the production trait API for a test-only convenience.
struct BorrowedMetric<'a, M>(&'a M);

impl<M: EuclideanMetric> EuclideanMetric for BorrowedMetric<'_, M> {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn sample_momentum(
        &self,
        src: &[f64],
        dst: &mut [f64],
    ) -> Result<(), alea_math::metric::MetricError> {
        self.0.sample_momentum(src, dst)
    }
    fn velocity(&self, src: &[f64], dst: &mut [f64]) -> Result<(), alea_math::metric::MetricError> {
        self.0.velocity(src, dst)
    }
    fn log_det(&self) -> f64 {
        self.0.log_det()
    }
}

#[derive(Clone, Copy, Debug)]
enum Target {
    Correlated,
    Banana,
    Scaled,
}

impl Target {
    // Each target is an invertible transformation of two independent N(0,1)s.
    // Constant Jacobian terms can be omitted from the log density.
    fn whiten(self, q: &[f64]) -> [f64; 2] {
        match self {
            Self::Correlated => [q[0], (q[1] - 0.6 * q[0]) / 0.8],
            Self::Banana => [q[0], q[1] - 0.4 * (q[0] * q[0] - 1.0)],
            Self::Scaled => [q[0] / 0.001, q[1] / 10.0],
        }
    }

    fn log_density(self, q: &[f64]) -> f64 {
        let z = self.whiten(q);
        -0.5 * (z[0] * z[0] + z[1] * z[1])
    }

    fn initial(self) -> [f64; 2] {
        match self {
            Self::Scaled => [0.0007, -2.0],
            _ => [0.7, -0.2],
        }
    }
}

impl LogDensityGradient for Target {
    type Error = Infallible;

    fn dimension(&self) -> usize {
        2
    }

    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let z = self.whiten(q);
        match self {
            Self::Correlated => {
                g[0] = -z[0] + 0.75 * z[1];
                g[1] = -z[1] / 0.8;
            }
            Self::Banana => {
                g[0] = -z[0] + 0.8 * q[0] * z[1];
                g[1] = -z[1];
            }
            Self::Scaled => {
                g[0] = -z[0] / 0.001;
                g[1] = -z[1] / 10.0;
            }
        }
        Ok(self.log_density(q))
    }
}

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "actual {actual:e}, expected {expected:e}, tolerance {tolerance:e}"
    );
}

#[test]
fn analytic_fixture_gradients_agree_with_central_differences() {
    for target in [Target::Correlated, Target::Banana, Target::Scaled] {
        let scale = match target {
            Target::Scaled => [0.001, 10.0],
            _ => [1.0, 1.0],
        };
        for z in [[0.0, 0.0], [0.7, -1.2], [-2.0, 0.3], [3.0, -2.0]] {
            let q = [z[0] * scale[0], z[1] * scale[1]];
            let mut gradient = [0.0; 2];
            target.logp_grad(&q, &mut gradient).unwrap();
            for i in 0..2 {
                let h = 1e-5 * scale[i];
                let mut plus = q;
                let mut minus = q;
                plus[i] += h;
                minus[i] -= h;
                let finite_difference =
                    (target.log_density(&plus) - target.log_density(&minus)) / (2.0 * h);
                close(gradient[i], finite_difference, 1e-8);
            }
        }
    }
}

// This deliberately does not use EuclideanMetric, the runtime integrator, or OwnedBuffer.
// L = [[a, 0], [b, c]], so M^-1 = [[1/a²+b²/(a²c²), -b/(ac²)],
//                                 [-b/(ac²), 1/c²]].
fn inverse_action(p: [f64; 2], [a, b, c]: [f64; 3]) -> [f64; 2] {
    let off = -b / (a * c * c);
    [
        (1.0 / (a * a) + b * b / (a * a * c * c)) * p[0] + off * p[1],
        off * p[0] + p[1] / (c * c),
    ]
}

fn kinetic(p: [f64; 2], factor: [f64; 3]) -> f64 {
    let v = inverse_action(p, factor);
    0.5 * (p[0] * v[0] + p[1] * v[1])
}

fn reference_transition<M: EuclideanMetric>(
    target: Target,
    metric: M,
    factor: [f64; 3],
    eps: f64,
) -> usize {
    let mut rejected = 0;
    for count in [1, 5, 11] {
        let options = HmcOptions::new(eps, count)
            .unwrap()
            .with_max_energy_error(f64::MAX)
            .unwrap();
        let mut chain = Hmc::new(
            &target,
            buffer(&target.initial()),
            BorrowedMetric(&metric),
            options,
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(912);
        for _ in 0..4 {
            let mut replay = rng.clone();
            // Replay the current Marsaglia-polar RNG contract, not the runtime helper.
            let normal = |rng: &mut SmallRng| {
                loop {
                    let u = 2.0 * rng.random::<f64>() - 1.0;
                    let v = 2.0 * rng.random::<f64>() - 1.0;
                    let radius = u * u + v * v;
                    if radius > 0.0 && radius < 1.0 {
                        break u * (-2.0 * radius.ln() / radius).sqrt();
                    }
                }
            };
            let z = [normal(&mut replay), normal(&mut replay)];
            let [a, b, c] = factor;
            let mut p = [a * z[0], b * z[0] + c * z[1]];
            let mut q = [chain.point().position()[0], chain.point().position()[1]];
            let initial = -target.log_density(&q) + kinetic(p, factor);
            let mut g = [0.0; 2];
            target.logp_grad(&q, &mut g).unwrap();
            for _ in 0..count {
                p = [p[0] + eps * g[0] / 2.0, p[1] + eps * g[1] / 2.0];
                let v = inverse_action(p, factor);
                q = [q[0] + eps * v[0], q[1] + eps * v[1]];
                target.logp_grad(&q, &mut g).unwrap();
                p = [p[0] + eps * g[0] / 2.0, p[1] + eps * g[1] / 2.0];
            }
            let proposed = -target.log_density(&q) + kinetic(p, factor);
            let probability = (initial - proposed).min(0.0).exp();
            let accepted = replay.random::<f64>() < probability;
            let old_q = chain.point().position().to_vec();
            let old_g = chain.point().gradient().to_vec();
            let old_lp = chain.point().log_density();
            let info = chain.step(&mut rng).unwrap();
            assert!(info.divergence.is_none(), "{target:?}: {info:?}");
            assert_eq!(info.accepted, accepted);
            assert_eq!(info.leapfrog_steps, count);
            assert_eq!(rng, replay);
            close(info.initial_energy.unwrap(), initial, 1e-11);
            close(info.proposal_energy.unwrap(), proposed, 1e-11);
            close(info.energy_error.unwrap(), proposed - initial, 1e-11);
            close(info.acceptance_probability, probability, 1e-11);
            if accepted {
                for i in 0..2 {
                    close(chain.point().position()[i], q[i], 1e-11);
                    close(chain.point().gradient()[i], g[i], 1e-11);
                }
                close(chain.point().log_density(), target.log_density(&q), 1e-11);
            } else {
                rejected += 1;
                // Exact comparison is intentional: rejection must not write the cache.
                assert_eq!(chain.point().position(), old_q);
                assert_eq!(chain.point().gradient(), old_g);
                assert_eq!(chain.point().log_density(), old_lp);
            }
        }
    }
    rejected
}

#[test]
fn full_transitions_match_independent_scalar_reference() {
    for target in [Target::Correlated, Target::Banana] {
        reference_transition(target, IdentityMetric::new(2), [1.0, 0.0, 1.0], 0.25);
        reference_transition(
            target,
            DiagonalMetric::new(buffer(&[4.0, 0.25])).unwrap(),
            [2.0, 0.0, 0.5],
            0.15,
        );
        reference_transition(
            target,
            DenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 1.3])).unwrap()),
            [2.0, 0.5, 1.3],
            0.25,
        );
    }
    reference_transition(
        Target::Scaled,
        DiagonalMetric::new(buffer(&[1e6, 0.01])).unwrap(),
        [1000.0, 0.0, 0.1],
        0.15,
    );
    // Force substantial but finite energy errors: the oracle must exercise MH
    // rejection as well as acceptance, without the numerical-divergence cutoff.
    assert!(
        reference_transition(
            Target::Correlated,
            IdentityMetric::new(2),
            [1.0, 0.0, 1.0],
            1.3
        ) > 0
    );
}

// Whitened coordinates have known moments for every target. Rejected states
// MUST be counted. Fixed seeds and tolerances are regression gates, not a
// convergence diagnostic or proof of mixing on other targets.
fn stationary_moments<M: EuclideanMetric>(target: Target, metric: M, eps: f64) {
    for seed in [19, 6539] {
        let mut chain = Hmc::new(
            &target,
            buffer(&target.initial()),
            BorrowedMetric(&metric),
            HmcOptions::new(eps, 9).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut sum = [0.0; 7];
        let mut accepted = 0;
        for iteration in 0..13_000 {
            let info = chain.step(&mut rng).unwrap();
            assert!(
                info.divergence.is_none(),
                "{target:?}, seed {seed}: {info:?}"
            );
            if iteration >= 1000 {
                accepted += usize::from(info.accepted);
                let [x, y] = target.whiten(chain.point().position());
                let values = [x, y, x * x, y * y, x * y, x.powi(4), y.powi(4)];
                for (total, value) in sum.iter_mut().zip(values) {
                    *total += value;
                }
            }
        }
        let expected = [0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 3.0];
        let tolerances = [0.08, 0.08, 0.12, 0.12, 0.08, 0.65, 0.65];
        for (i, ((total, expected), tolerance)) in
            sum.into_iter().zip(expected).zip(tolerances).enumerate()
        {
            let observed = total / 12_000.0;
            assert!(
                (observed - expected).abs() < tolerance,
                "{target:?}, seed {seed}, observable {i}: {observed}, expected {expected} ± {tolerance}"
            );
        }
        assert!(
            accepted > 9600,
            "{target:?}: only {accepted}/12000 accepted"
        );
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "long statistical regression; deterministic oracle runs under Miri"
)]
fn correlated_gaussian_stationary_moments_for_every_mass_type() {
    stationary_moments(Target::Correlated, IdentityMetric::new(2), 0.15);
    stationary_moments(
        Target::Correlated,
        DiagonalMetric::new(buffer(&[2.0, 0.5])).unwrap(),
        0.15,
    );
    stationary_moments(
        Target::Correlated,
        DenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[1.5, 0.4, 0.0, 0.8])).unwrap()),
        0.15,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "long statistical regression; deterministic oracle runs under Miri"
)]
fn banana_stationary_moments_for_every_mass_type() {
    stationary_moments(Target::Banana, IdentityMetric::new(2), 0.15);
    stationary_moments(
        Target::Banana,
        DiagonalMetric::new(buffer(&[2.0, 0.5])).unwrap(),
        0.15,
    );
    stationary_moments(
        Target::Banana,
        DenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[1.5, 0.4, 0.0, 0.8])).unwrap()),
        0.15,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "long statistical regression; deterministic oracle runs under Miri"
)]
fn condition_number_1e8_gaussian_with_matched_mass() {
    // Mass is precision, not covariance: M^-1 * precision = I.
    stationary_moments(
        Target::Scaled,
        DiagonalMetric::new(buffer(&[1e6, 0.01])).unwrap(),
        0.15,
    );
}

#[test]
fn unstable_ill_conditioned_trajectory_rejects_without_corrupting_cache() {
    let target = Target::Scaled;
    let mut chain = Hmc::new(
        &target,
        buffer(&target.initial()),
        IdentityMetric::new(2),
        HmcOptions::new(0.1, 9).unwrap(),
    )
    .unwrap();
    let old_q = chain.point().position().to_vec();
    let old_g = chain.point().gradient().to_vec();
    let old_lp = chain.point().log_density();
    let info = chain.step(&mut SmallRng::seed_from_u64(19)).unwrap();
    assert!(info.divergence.is_some());
    assert!(!info.accepted);
    assert_eq!(chain.point().position(), old_q);
    assert_eq!(chain.point().gradient(), old_g);
    assert_eq!(chain.point().log_density(), old_lp);
}

#[test]
fn actual_floating_point_overflow_has_typed_rejection_and_preserves_cache() {
    // Valid finite starting states and configurations; no injected model errors.
    let target = Gaussian::new(1);
    for (position, mass, eps, expected) in [
        (1e154, 1.0, f64::MAX, Divergence::Momentum { index: 0 }),
        (0.0, 1e-308, f64::MAX, Divergence::Position { index: 0 }),
        (0.0, 1.0, 1e155, Divergence::LogDensity),
    ] {
        let mut chain = Hmc::new(
            &target,
            buffer(&[position]),
            DiagonalMetric::new(buffer(&[mass])).unwrap(),
            HmcOptions::new(eps, 2).unwrap(),
        )
        .unwrap();
        let old_gradient = chain.point().gradient()[0];
        let old_density = chain.point().log_density();
        let info = chain.step(&mut SmallRng::seed_from_u64(19)).unwrap();
        assert_eq!(info.divergence, Some(expected));
        assert_eq!(info.leapfrog_steps, 1);
        assert!(!info.accepted);
        assert_eq!(info.acceptance_probability, 0.0);
        assert_eq!(chain.point().position(), &[position]);
        assert_eq!(chain.point().gradient(), &[old_gradient]);
        assert_eq!(chain.point().log_density(), old_density);
    }
}
