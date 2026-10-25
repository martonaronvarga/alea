use std::convert::Infallible;

use alea_core::{
    capability::{
        HessianVector, PositionMetric, PotentialHessianDerivative, evaluate_position_metric,
    },
    target::LogDensityGradient,
};
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::riemannian::{
    ExactnessPendingRmhmc, HierarchicalNormalFisherMetric, LogisticFisherMetric, RmhmcOptions,
    RmhmcTermination, SoftAbsMetric, SoftAbsOptions,
};
use rand::{SeedableRng, rngs::SmallRng};

struct IdentityPositionMetric {
    dimension: usize,
}
impl PositionMetric for IdentityPositionMetric {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn metric_and_derivatives(
        &self,
        _: &[f64],
        metric: &mut [f64],
        derivatives: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), Self::Error> {
        metric.fill(0.0);
        derivatives.fill(0.0);
        for index in 0..self.dimension {
            metric[index * self.dimension + index] = 1.0;
        }
        Ok(())
    }
}

struct Gaussian {
    dimension: usize,
}
impl LogDensityGradient for Gaussian {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let mut squared = 0.0;
        for (gradient, value) in gradient.iter_mut().zip(q) {
            *gradient = -*value;
            squared += value * value;
        }
        Ok(-0.5 * squared)
    }
}

struct LogisticPosterior {
    successes: f64,
    trials: f64,
    prior_precision: f64,
}
impl LogDensityGradient for LogisticPosterior {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        1
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let probability = if q[0] >= 0.0 {
            1.0 / (1.0 + (-q[0]).exp())
        } else {
            let exponential = q[0].exp();
            exponential / (1.0 + exponential)
        };
        gradient[0] = self.successes - self.trials * probability - self.prior_precision * q[0];
        let log_normalizer = if q[0] > 0.0 {
            q[0] + (-q[0]).exp().ln_1p()
        } else {
            q[0].exp().ln_1p()
        };
        Ok(self.successes * q[0]
            - self.trials * log_normalizer
            - 0.5 * self.prior_precision * q[0] * q[0])
    }
}

struct Funnel;
impl LogDensityGradient for Funnel {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let exponential = q[0].exp();
        gradient[0] = -q[0] / 9.0 + 0.5 - 0.5 * q[1] * q[1] * exponential;
        gradient[1] = -q[1] * exponential;
        Ok(-q[0] * q[0] / 18.0 + 0.5 * q[0] - 0.5 * q[1] * q[1] * exponential)
    }
}
impl HessianVector for Funnel {
    fn potential_hvp(
        &self,
        q: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        let exponential = q[0].exp();
        let h00 = 1.0 / 9.0 + 0.5 * q[1] * q[1] * exponential;
        let h01 = q[1] * exponential;
        output[0] = h00 * vector[0] + h01 * vector[1];
        output[1] = h01 * vector[0] + exponential * vector[1];
        Ok(())
    }
}
impl PotentialHessianDerivative for Funnel {
    fn potential_hessian_and_derivatives(
        &self,
        q: &[f64],
        hessian: &mut [f64],
        derivatives: &mut [f64],
    ) -> Result<(), Self::Error> {
        let exponential = q[0].exp();
        let h00 = 1.0 / 9.0 + 0.5 * q[1] * q[1] * exponential;
        let h01 = q[1] * exponential;
        hessian.copy_from_slice(&[h00, h01, h01, exponential]);
        derivatives.copy_from_slice(&[
            0.5 * q[1] * q[1] * exponential,
            h01,
            h01,
            exponential,
            h01,
            exponential,
            exponential,
            0.0,
        ]);
        Ok(())
    }
}

fn moments(draws: &[f64]) -> (f64, f64) {
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let variance = draws
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (draws.len() - 1) as f64;
    (mean, variance)
}

#[test]
fn rmhmc_constant_metric_has_stationary_gaussian_moments_and_exact_costs() {
    let target = Gaussian { dimension: 2 };
    let metric = IdentityPositionMetric { dimension: 2 };
    let options = RmhmcOptions::new(0.18, 5, 2, 1.0e-14).unwrap();
    let mut draws = Vec::with_capacity(24_000);
    for chain_id in 0..4 {
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::new(2), options).unwrap();
        let mut rng = SmallRng::seed_from_u64(11_000 + chain_id);
        for iteration in 0..6_500 {
            let transition = chain.step(&mut rng).unwrap();
            assert!(
                transition.termination.is_none(),
                "termination={:?}",
                transition.termination
            );
            if iteration == 0 {
                assert_eq!(transition.costs.target_evaluations, 5);
                assert_eq!(transition.costs.metric_derivative_evaluations, 15);
                assert_eq!(transition.costs.factorizations, 15);
                assert_eq!(transition.costs.triangular_solves, 60);
                assert_eq!(transition.costs.dynamics_steps, 5);
            }
            if iteration >= 500 {
                draws.extend_from_slice(chain.point().position());
            }
        }
    }
    let (mean, variance) = moments(&draws);
    assert!(mean.abs() < 0.035, "mean={mean}");
    assert!((variance - 1.0).abs() < 0.06, "variance={variance}");
}

#[test]
fn logistic_fisher_derivative_and_posterior_moments_match_independent_grid() {
    let design = OwnedBuffer::from_fn(10, |_| 1.0);
    let prior = OwnedBuffer::from_fn(1, |_| 0.25);
    let metric = LogisticFisherMetric::new(10, 1, design, prior).unwrap();
    let mut matrix = [0.0];
    let mut derivative = [0.0];
    evaluate_position_metric(&metric, &[0.4], &mut matrix, &mut derivative, &mut []).unwrap();
    let epsilon = 1.0e-6;
    let mut plus = [0.0];
    let mut minus = [0.0];
    let mut unused = [0.0];
    evaluate_position_metric(&metric, &[0.4 + epsilon], &mut plus, &mut unused, &mut []).unwrap();
    evaluate_position_metric(&metric, &[0.4 - epsilon], &mut minus, &mut unused, &mut []).unwrap();
    let finite_difference = (plus[0] - minus[0]) / (2.0 * epsilon);
    assert!((derivative[0] - finite_difference).abs() < 1.0e-9);

    let target = LogisticPosterior {
        successes: 7.0,
        trials: 10.0,
        prior_precision: 0.25,
    };
    let spacing = 20.0 / 40_000.0;
    let mut normalizer = 0.0;
    let mut expected = 0.0;
    let mut expected_square = 0.0;
    for index in 0..=40_000 {
        let q = -10.0 + index as f64 * spacing;
        let mut gradient = [0.0];
        let weight = target.logp_grad(&[q], &mut gradient).unwrap().exp();
        normalizer += weight;
        expected += weight * q;
        expected_square += weight * q * q;
    }
    expected /= normalizer;
    let expected_variance = expected_square / normalizer - expected * expected;

    let options = RmhmcOptions::new(0.13, 5, 12, 1.0e-9).unwrap();
    let mut draws = Vec::with_capacity(24_000);
    for chain_id in 0..4 {
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::new(1), options).unwrap();
        let mut rng = SmallRng::seed_from_u64(11_100 + chain_id);
        for iteration in 0..7_000 {
            let transition = chain.step(&mut rng).unwrap();
            assert!(
                transition.termination.is_none(),
                "termination={:?}",
                transition.termination
            );
            if iteration >= 1_000 {
                draws.push(chain.point().position()[0]);
            }
        }
    }
    let (mean, variance) = moments(&draws);
    assert!(
        (mean - expected).abs() < 0.06,
        "mean={mean}, reference={expected}"
    );
    assert!(
        (variance - expected_variance).abs() < 0.08,
        "variance={variance}, reference={expected_variance}"
    );
}

#[test]
fn hierarchical_fisher_and_softabs_derivatives_match_finite_differences() {
    let hierarchical = HierarchicalNormalFisherMetric::new(2, 0.2, 0.3, 1.5).unwrap();
    let q = [0.1, -0.3, 0.4, -0.2];
    let mut matrix = [0.0; 16];
    let mut derivatives = [0.0; 64];
    evaluate_position_metric(&hierarchical, &q, &mut matrix, &mut derivatives, &mut []).unwrap();
    let epsilon = 1.0e-6;
    let mut plus_q = q;
    let mut minus_q = q;
    plus_q[1] += epsilon;
    minus_q[1] -= epsilon;
    let mut plus = [0.0; 16];
    let mut minus = [0.0; 16];
    let mut unused = [0.0; 64];
    evaluate_position_metric(&hierarchical, &plus_q, &mut plus, &mut unused, &mut []).unwrap();
    evaluate_position_metric(&hierarchical, &minus_q, &mut minus, &mut unused, &mut []).unwrap();
    for index in 0..16 {
        let finite_difference = (plus[index] - minus[index]) / (2.0 * epsilon);
        assert!((derivatives[16 + index] - finite_difference).abs() < 2.0e-9);
    }

    let target = Funnel;
    let softabs =
        SoftAbsMetric::new(&target, SoftAbsOptions::new(3.0, 32, 1.0e-13).unwrap()).unwrap();
    let workspace_len = PositionMetric::workspace_len(&softabs);
    let mut workspace = vec![0.0; workspace_len];
    let mut soft_matrix = [0.0; 4];
    let mut soft_derivative = [0.0; 8];
    evaluate_position_metric(
        &softabs,
        &[0.2, -0.4],
        &mut soft_matrix,
        &mut soft_derivative,
        &mut workspace,
    )
    .unwrap();
    let baseline_derivative = soft_derivative;
    for coordinate in 0..2 {
        let mut plus_position = [0.2, -0.4];
        let mut minus_position = plus_position;
        plus_position[coordinate] += epsilon;
        minus_position[coordinate] -= epsilon;
        let mut plus_metric = [0.0; 4];
        let mut minus_metric = [0.0; 4];
        evaluate_position_metric(
            &softabs,
            &plus_position,
            &mut plus_metric,
            &mut soft_derivative,
            &mut workspace,
        )
        .unwrap();
        evaluate_position_metric(
            &softabs,
            &minus_position,
            &mut minus_metric,
            &mut soft_derivative,
            &mut workspace,
        )
        .unwrap();
        for index in 0..4 {
            let finite_difference = (plus_metric[index] - minus_metric[index]) / (2.0 * epsilon);
            assert!(
                (baseline_derivative[coordinate * 4 + index] - finite_difference).abs() < 2.0e-6
            );
        }
    }
}

#[test]
fn softabs_rmhmc_has_stationary_funnel_moments() {
    let target = Funnel;
    let softabs =
        SoftAbsMetric::new(&target, SoftAbsOptions::new(1.0, 32, 1.0e-12).unwrap()).unwrap();
    let options = RmhmcOptions::new(0.16, 8, 16, 1.0e-8).unwrap();
    let mut hierarchy = Vec::with_capacity(24_000);
    let mut standardized_square = Vec::with_capacity(24_000);
    let mut failures = 0;
    let mut accepted = 0;
    let mut attempted = 0;
    for chain_id in 0..4 {
        let mut chain =
            ExactnessPendingRmhmc::new(&target, &softabs, OwnedBuffer::new(2), options).unwrap();
        let mut rng = SmallRng::seed_from_u64(11_150 + chain_id);
        for iteration in 0..8_000 {
            let transition = chain.step(&mut rng).unwrap();
            attempted += 1;
            accepted += usize::from(transition.accepted);
            failures += usize::from(transition.termination.is_some());
            if iteration >= 2_000 {
                let q = chain.point().position();
                hierarchy.push(q[0]);
                standardized_square.push(q[1] * q[1] * q[0].exp());
            }
        }
    }
    let (hierarchy_mean, hierarchy_variance) = moments(&hierarchy);
    let standardized_mean =
        standardized_square.iter().sum::<f64>() / standardized_square.len() as f64;
    assert!(
        hierarchy_mean.abs() < 0.35,
        "mean={hierarchy_mean}, variance={hierarchy_variance}, failures={failures}, acceptance={}",
        accepted as f64 / attempted as f64
    );
    assert!(
        (hierarchy_variance - 9.0).abs() < 1.2,
        "mean={hierarchy_mean}, variance={hierarchy_variance}, failures={failures}, acceptance={}",
        accepted as f64 / attempted as f64
    );
    assert!(
        (standardized_mean - 1.0).abs() < 0.12,
        "standardized second moment={standardized_mean}, failures={failures}, acceptance={}",
        accepted as f64 / attempted as f64
    );
    assert!(failures < attempted / 100);
}

#[test]
fn bounded_solve_failure_is_a_rejected_event() {
    struct ExponentialMetric;
    impl PositionMetric for ExponentialMetric {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            2
        }
        fn metric_and_derivatives(
            &self,
            q: &[f64],
            metric: &mut [f64],
            derivatives: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), Self::Error> {
            metric.fill(0.0);
            derivatives.fill(0.0);
            for coordinate in 0..2 {
                let value = q[coordinate].exp();
                metric[coordinate * 2 + coordinate] = value;
                derivatives[coordinate * 4 + coordinate * 2 + coordinate] = value;
            }
            Ok(())
        }
    }
    let target = Gaussian { dimension: 2 };
    let metric = ExponentialMetric;
    let options = RmhmcOptions::new(0.8, 2, 1, 1.0e-16).unwrap();
    let mut chain =
        ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::new(2), options).unwrap();
    let before = chain.point().position().to_vec();
    let mut rng = SmallRng::seed_from_u64(11_200);
    let transition = chain.step(&mut rng).unwrap();
    assert!(matches!(
        transition.termination,
        Some(RmhmcTermination::ImplicitSolve(_))
    ));
    assert_eq!(chain.point().position(), before);
}
