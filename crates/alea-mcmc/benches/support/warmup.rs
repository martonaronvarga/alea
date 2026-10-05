//! Fixed cost workloads shared by Criterion, the CSV example and contract tests.
use alea_core::target::LogDensityGradient;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    HmcOptions,
    adapt::{FisherHmcWarmup, FisherOptions, HmcWarmup, HmcWarmupReport, WarmupOptions},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, convert::Infallible, hint::black_box};

pub const DIMENSIONS: [usize; 2] = [8, 64];
pub const SEEDS: [u64; 3] = [7, 19, 43];
pub const ITERATIONS: usize = 200;
pub const STEPS: usize = 5;

#[derive(Debug, Clone, Copy)]
pub enum Shape {
    Isotropic,
    Correlated,
}
pub const SHAPES: [Shape; 2] = [Shape::Isotropic, Shape::Correlated];
impl Shape {
    pub fn name(self) -> &'static str {
        match self {
            Self::Isotropic => "isotropic",
            Self::Correlated => "correlated",
        }
    }
}

/// Correlated precision is diag(1/4,1,4,...) + 9 uu^T, u_i=1/sqrt(d).
/// Both targets are proper centered Gaussians; normalization is irrelevant here.
pub struct Target {
    dimension: usize,
    shape: Shape,
}
impl Target {
    pub fn new(dimension: usize, shape: Shape) -> Self {
        assert!(dimension > 0);
        Self { dimension, shape }
    }
}
impl LogDensityGradient for Target {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        let coupling = match self.shape {
            Shape::Isotropic => 0.0,
            Shape::Correlated => 9.0 * q.iter().sum::<f64>() / self.dimension as f64,
        };
        let mut logp = 0.0;
        for (i, (&q, g)) in q.iter().zip(g).enumerate() {
            let precision = match self.shape {
                Shape::Isotropic => 1.0,
                Shape::Correlated => [0.25, 1.0, 4.0][i % 3],
            };
            *g = -precision * q - coupling;
            logp += 0.5 * q * *g;
        }
        Ok(logp)
    }
}

/// Count real backend entries, including initialization and search, outside timing.
pub struct Counted<T> {
    pub target: T,
    pub calls: Cell<usize>,
}
impl<T: LogDensityGradient> LogDensityGradient for Counted<T> {
    type Error = T::Error;
    fn dimension(&self) -> usize {
        self.target.dimension()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, T::Error> {
        self.calls.set(self.calls.get().checked_add(1).unwrap());
        self.target.logp_grad(q, g)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Method {
    Covariance,
    FisherDiagonal,
    #[cfg(feature = "faer")]
    FisherLowRank,
}
pub const METHODS: &[Method] = &[
    Method::Covariance,
    Method::FisherDiagonal,
    #[cfg(feature = "faer")]
    Method::FisherLowRank,
];
impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Self::Covariance => "covariance_diagonal",
            Self::FisherDiagonal => "fisher_diagonal",
            #[cfg(feature = "faer")]
            Self::FisherLowRank => "fisher_rank4",
        }
    }
}

#[derive(Debug)]
pub struct Outcome {
    pub report: HmcWarmupReport,
    pub rank: usize,
    pub logp: f64,
}

/// Includes controller construction, initial evaluation, search, fitting and drop.
/// RNG and q0 are reset for every run. Failures are fatal, never filtered out.
pub fn run<T: LogDensityGradient<Error = Infallible>>(
    target: &T,
    method: Method,
    seed: u64,
) -> Outcome {
    let position = OwnedBuffer::from_fn(target.dimension(), |_| 0.25);
    let options = HmcOptions::new(0.1, STEPS).unwrap();
    let mut rng = SmallRng::seed_from_u64(seed);
    match method {
        Method::Covariance => {
            let (chain, report) =
                HmcWarmup::new(target, position, options, WarmupOptions::new(ITERATIONS))
                    .unwrap()
                    .run(&mut rng)
                    .unwrap();
            black_box(chain.point().position());
            Outcome {
                report,
                rank: 0,
                logp: chain.point().log_density(),
            }
        }
        _ => {
            let fisher = FisherOptions::new(ITERATIONS);
            #[cfg(feature = "faer")]
            let fisher = if matches!(method, Method::FisherLowRank) {
                fisher.with_max_rank(4)
            } else {
                fisher
            };
            let (chain, report) = FisherHmcWarmup::new(target, position, options, fisher)
                .unwrap()
                .run(&mut rng)
                .unwrap();
            black_box(chain.point().position());
            Outcome {
                report,
                rank: chain.metric().rank(),
                logp: chain.point().log_density(),
            }
        }
    }
}
