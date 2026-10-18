use alea_core::reference::LogDensityGradient;
use alea_downstream_conformance::{
    TargetCase, assert_chain_conformance, assert_metric_conformance, assert_target_conformance,
};
use alea_math::reference::{EuclideanMetric, MetricError};
use alea_mcmc::reference::MarkovChain;
use alea_runtime::{
    ensemble::{self, Chain, Execution, RunOptions, Termination, chain_rng},
    output::{ChainId, DiagnosticTransition, Draw, Phase, TransitionStats},
};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};
use std::{fmt, num::NonZeroUsize, ops::ControlFlow};
use tokio_util::sync::CancellationToken;

struct ExternalTarget;

impl LogDensityGradient for ExternalTarget {
    type Error = std::convert::Infallible;

    fn dimension(&self) -> usize {
        2
    }

    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        for (output, &value) in gradient.iter_mut().zip(position) {
            *output = -value;
        }
        Ok(-0.5 * position.iter().map(|value| value * value).sum::<f64>())
    }
}

struct ExternalDiagonalMetric {
    mass: [f64; 2],
}

impl ExternalDiagonalMetric {
    fn check(&self, source: &[f64], destination: &[f64]) -> Result<(), MetricError> {
        if source.len() != self.mass.len() || destination.len() != self.mass.len() {
            return Err(MetricError::VectorLength {
                expected: self.mass.len(),
                source_len: source.len(),
                destination_len: destination.len(),
            });
        }
        Ok(())
    }
}

impl EuclideanMetric for ExternalDiagonalMetric {
    fn dimension(&self) -> usize {
        self.mass.len()
    }

    fn velocity(&self, momentum: &[f64], velocity: &mut [f64]) -> Result<(), MetricError> {
        self.check(momentum, velocity)?;
        for ((output, &value), &mass) in velocity.iter_mut().zip(momentum).zip(&self.mass) {
            *output = value / mass;
        }
        Ok(())
    }

    fn sample_momentum(
        &self,
        standard_normal: &[f64],
        momentum: &mut [f64],
    ) -> Result<(), MetricError> {
        self.check(standard_normal, momentum)?;
        for ((output, &value), &mass) in momentum.iter_mut().zip(standard_normal).zip(&self.mass) {
            *output = value * mass.sqrt();
        }
        Ok(())
    }

    fn log_det(&self) -> f64 {
        self.mass.iter().map(|mass| mass.ln()).sum()
    }
}

#[derive(Debug)]
struct ExternalChainError;

impl fmt::Display for ExternalChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("injected downstream chain failure")
    }
}

impl std::error::Error for ExternalChainError {}

#[derive(Clone, Copy)]
struct ExternalTransition {
    accepted: bool,
    probability: f64,
    energy: f64,
}

impl DiagnosticTransition for ExternalTransition {
    fn diagnostic_stats(&self) -> TransitionStats {
        TransitionStats {
            accepted: self.accepted,
            acceptance_probability: self.probability,
            energy: Some(self.energy),
            divergent: Some(false),
            integration_steps: Some(1),
            ..TransitionStats::default()
        }
    }
}

struct ExternalChain {
    position: [f64; 2],
    attempts: usize,
    fail_at: Option<usize>,
}

impl ExternalChain {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            position: [0.0; 2],
            attempts: 0,
            fail_at,
        }
    }
}

impl MarkovChain for ExternalChain {
    type Error = ExternalChainError;
    type Transition = ExternalTransition;

    fn position(&self) -> &[f64] {
        &self.position
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Self::Transition, Self::Error> {
        let attempt = self.attempts;
        self.attempts += 1;
        if self.fail_at == Some(attempt) {
            return Err(ExternalChainError);
        }
        for value in &mut self.position {
            *value = rng.random_range(-1.0..1.0);
        }
        Ok(ExternalTransition {
            accepted: true,
            probability: 1.0,
            energy: self.position.iter().map(|value| value * value).sum(),
        })
    }
}

#[test]
fn external_target_and_metric_pass_reusable_laws() {
    let cases = [
        TargetCase {
            position: &[0.0, 0.0],
            log_density: 0.0,
            gradient: &[0.0, 0.0],
        },
        TargetCase {
            position: &[1.5, -2.0],
            log_density: -3.125,
            gradient: &[-1.5, 2.0],
        },
    ];
    assert_target_conformance(&ExternalTarget, &cases, 1.0e-14);

    let metric = ExternalDiagonalMetric { mass: [4.0, 9.0] };
    assert_metric_conformance(&metric, &[&[0.0, 0.0], &[1.5, -2.0]], 1.0e-14);
    assert!((metric.log_det() - 36.0_f64.ln()).abs() < 1.0e-14);
}

#[test]
fn external_chain_diagnostic_and_borrowed_sink_compose() {
    let mut rng = SmallRng::seed_from_u64(91);
    let mut fallible = ExternalChain::new(Some(2));
    assert_chain_conformance(&mut fallible, &mut rng, 6);

    let mut records = Vec::new();
    let sink = |draw: Draw<'_, ExternalTransition>| {
        records.push((
            draw.chain,
            draw.phase,
            draw.index,
            draw.position.to_vec(),
            draw.transition.diagnostic_stats(),
        ));
        ControlFlow::Continue(())
    };
    let id = ChainId(17);
    let mut chains = [Chain {
        id,
        sampler: ExternalChain::new(None),
        rng: chain_rng([23; 32], id),
        sink,
    }];
    let options = RunOptions {
        warmup: 1,
        draws: 4,
        chunk: NonZeroUsize::new(2).expect("two is nonzero"),
        window: NonZeroUsize::new(4).expect("four is nonzero"),
        coordinates: vec![0, 1],
    };
    let report = ensemble::run(
        &mut chains,
        &options,
        Execution::Sequential,
        &CancellationToken::new(),
        None,
    )
    .expect("external chain and sink must run");
    assert_eq!(report.termination, Termination::Completed);
    drop(chains);

    assert_eq!(records.len(), 5);
    assert_eq!(records[0].1, Phase::Warmup);
    assert!(
        records[1..]
            .iter()
            .all(|record| record.1 == Phase::Sampling)
    );
    assert!(records.iter().all(|record| record.0 == id));
    assert_eq!(
        records[1..]
            .iter()
            .map(|record| record.2)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}
