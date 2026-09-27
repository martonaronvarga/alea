use std::cell::Cell;

use kernels::{
    buffer::OwnedBuffer,
    density::{FusedLogDensity, GradLogDensity, LogDensity},
    dist::Gaussian,
    kernel::Kernel,
    metric::{CholeskyFactor, DenseMetric, DiagonalMetric, IdentityMetric, Metric},
    state::{ChainState, GradientBuffers},
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use runtime::{
    integrator::leapfrog_step,
    mcmc::hmc::{Hmc, HmcConfig},
};

fn state(position: Vec<f64>) -> ChainState<Vec<f64>, GradientBuffers<Vec<f64>>> {
    let dim = position.len();
    ChainState::with_aux(
        position,
        GradientBuffers {
            gradient: vec![0.0; dim],
            momentum: vec![0.0; dim],
        },
    )
}

fn buffer(values: &[f64]) -> OwnedBuffer {
    let mut out = OwnedBuffer::new(values.len());
    out.copy_from_slice(values);
    out
}

struct FusedOnly {
    calls: Cell<usize>,
    bad_after: usize,
    bad_value: f64,
    bad_gradient: bool,
}

impl LogDensity for FusedOnly {
    type Point = [f64];
    fn log_prob(&self, _: &[f64]) -> f64 {
        panic!("must use fused evaluation")
    }
}
impl GradLogDensity for FusedOnly {
    type Gradient = [f64];
    fn grad_log_prob(&self, _: &[f64], _: &mut [f64]) {
        panic!("must use fused evaluation")
    }
}
impl FusedLogDensity for FusedOnly {
    fn log_prob_and_grad(&self, q: &[f64], g: &mut [f64]) -> f64 {
        self.calls.set(self.calls.get() + 1);
        let lp = Gaussian.log_prob_and_grad(q, g);
        if self.calls.get() > self.bad_after {
            if self.bad_gradient {
                g.fill(f64::NAN);
            }
            self.bad_value
        } else {
            lp
        }
    }
}

#[test]
fn hmc_evaluates_once_per_position_and_keeps_accepted_caches_consistent() {
    let target = FusedOnly {
        calls: Cell::new(0),
        bad_after: usize::MAX,
        bad_value: f64::NAN,
        bad_gradient: false,
    };
    let mut chain = state(vec![0.5, -0.25]);
    let mut hmc = Hmc::new(
        HmcConfig {
            step_size: 0.01,
            n_leapfrog: 4,
        },
        IdentityMetric::new(2),
    );
    let mut rng = SmallRng::seed_from_u64(42);
    assert!(hmc.step(&mut chain, &target, &mut rng));
    assert_eq!(target.calls.get(), 5);
    assert_eq!(chain.log_prob, Gaussian.log_prob(&chain.position));
    for (q, g) in chain.position.iter().zip(&chain.aux.gradient) {
        assert_eq!(*g, -*q);
    }
    // Public state mutation cannot leave a stale log-density or gradient in use.
    chain.position[0] = 2.0;
    hmc.step(&mut chain, &target, &mut rng);
    assert_eq!(chain.log_prob, Gaussian.log_prob(&chain.position));
}

#[test]
fn nonfinite_proposals_are_rejected_without_changing_position_or_gradient() {
    for (bad_value, bad_gradient) in [
        (f64::NAN, false),
        (f64::INFINITY, false),
        (f64::NEG_INFINITY, false),
        (0.0, true),
    ] {
        let target = FusedOnly {
            calls: Cell::new(0),
            bad_after: 1,
            bad_value,
            bad_gradient,
        };
        let mut chain = state(vec![0.5]);
        let mut hmc = Hmc::new(HmcConfig::default(), IdentityMetric::new(1));
        assert!(!hmc.step(&mut chain, &target, &mut SmallRng::seed_from_u64(3)));
        assert_eq!(chain.position, [0.5]);
        assert_eq!(chain.aux.gradient, [-0.5]);
        assert_eq!(chain.log_prob, -0.125);
    }
}

#[test]
fn invalid_initial_evaluations_and_configs_do_not_move_the_chain() {
    for (bad_value, bad_gradient) in [(f64::NAN, false), (0.0, true)] {
        let target = FusedOnly {
            calls: Cell::new(0),
            bad_after: 0,
            bad_value,
            bad_gradient,
        };
        let mut chain = state(vec![0.5]);
        let mut hmc = Hmc::new(HmcConfig::default(), IdentityMetric::new(1));
        assert!(!hmc.step(&mut chain, &target, &mut SmallRng::seed_from_u64(9)));
        assert_eq!(chain.position, [0.5]);
        assert_eq!(target.calls.get(), 1);
    }
    for config in [
        HmcConfig {
            step_size: 0.0,
            n_leapfrog: 10,
        },
        HmcConfig {
            step_size: -0.1,
            n_leapfrog: 10,
        },
        HmcConfig {
            step_size: f64::NAN,
            n_leapfrog: 10,
        },
        HmcConfig {
            step_size: f64::INFINITY,
            n_leapfrog: 10,
        },
        HmcConfig {
            step_size: 0.1,
            n_leapfrog: 0,
        },
    ] {
        let mut chain = state(vec![0.5]);
        let mut hmc = Hmc::new(config, IdentityMetric::new(1));
        assert!(!hmc.step(&mut chain, &Gaussian, &mut SmallRng::seed_from_u64(9)));
        assert_eq!(chain.position, [0.5]);
    }
}

fn check_momentum<M: Metric>(metric: M) {
    let mut rng = SmallRng::seed_from_u64(21);
    let mut replay = rng.clone();
    let mut z = vec![0.0; metric.dim()];
    for value in &mut z {
        loop {
            let u = 2.0 * replay.random::<f64>() - 1.0;
            let v = 2.0 * replay.random::<f64>() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                *value = u * (-2.0 * s.ln() / s).sqrt();
                break;
            }
        }
    }
    let mut expected = vec![0.0; z.len()];
    metric.apply_sqrt(&z, &mut expected);
    let mut chain = state(vec![0.0; z.len()]);
    let mut hmc = Hmc::new(HmcConfig::default(), metric);
    hmc.step(&mut chain, &Gaussian, &mut rng);
    assert_eq!(chain.aux.momentum, expected);
}

#[test]
fn momentum_uses_diagonal_and_dense_mass_matrices() {
    check_momentum(DiagonalMetric::new(buffer(&[4.0, 9.0])).unwrap());
    check_momentum(DenseMetric::new(
        CholeskyFactor::new_lower(2, buffer(&[2.0, 0.5, 0.0, 3.0])).unwrap(),
    ));
}

#[test]
fn leapfrog_is_reversible_and_has_second_order_energy_error() {
    let metric = IdentityMetric::new(1);
    let integrate = |eps: f64, steps: usize| {
        let mut chain = state(vec![0.7]);
        chain.aux.momentum[0] = -0.3;
        chain.log_prob = Gaussian.log_prob_and_grad(&chain.position, &mut chain.aux.gradient);
        let mut velocity = [0.0];
        for _ in 0..steps {
            leapfrog_step(&metric, eps, &mut chain, &Gaussian, &mut velocity);
        }
        chain
    };
    let mut coarse = integrate(0.1, 10);
    let fine = integrate(0.05, 20);
    let energy_error = |chain: &ChainState<Vec<f64>, GradientBuffers<Vec<f64>>>| {
        (-chain.log_prob + 0.5 * chain.aux.momentum[0].powi(2) - 0.29).abs()
    };
    assert!(energy_error(&fine) < energy_error(&coarse) / 3.5);
    for _ in 0..10 {
        leapfrog_step(&metric, -0.1, &mut coarse, &Gaussian, &mut [0.0]);
    }
    assert!((coarse.position[0] - 0.7).abs() < 1e-12);
    assert!((coarse.aux.momentum[0] + 0.3).abs() < 1e-12);
}

#[test]
fn gaussian_stationary_moments_with_nonidentity_mass() {
    let mut chain = state(vec![0.0, 0.0]);
    let mut hmc = Hmc::new(
        HmcConfig {
            step_size: 0.2,
            n_leapfrog: 15,
        },
        DiagonalMetric::new(buffer(&[4.0, 0.5])).unwrap(),
    );
    let mut rng = SmallRng::seed_from_u64(734);
    let mut sums = [0.0; 2];
    let mut squares = [0.0; 2];
    for i in 0..21_000 {
        hmc.step(&mut chain, &Gaussian, &mut rng);
        if i >= 1000 {
            for j in 0..2 {
                sums[j] += chain.position[j];
                squares[j] += chain.position[j].powi(2);
            }
        }
    }
    for j in 0..2 {
        assert!((sums[j] / 20_000.0).abs() < 0.05);
        assert!((squares[j] / 20_000.0 - 1.0).abs() < 0.08);
    }
}
