//! M10 equal-force-budget benchmark; refuses to overwrite its CSV output.

use std::{
    convert::Infallible,
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use alea_core::{
    capability::{
        BatchLane, BatchLayout, BatchLogDensityGradient, BatchShape, HessianVector,
        copy_batch_layout, evaluate_hvp,
    },
    target::LogDensityGradient,
};
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::{
    HmcOptions,
    adapt::{
        CurvatureProbeOptions, HmcWarmup, JointHmcWarmup, JointWarmupOptions, MetricKind,
        OnlineCovariance, WarmupOptions,
    },
    hamiltonian::SignedStep,
    integrator::{ForceBudgetPlan, Integrator, Leapfrog, TwoStage},
};
use alea_runtime::diagnostics::modern;
use rand::{SeedableRng, rngs::SmallRng};

const DIMENSION: usize = 16;
const CHAINS: usize = 4;
const WARMUP: usize = 300;
const DRAWS: usize = 2_048;
const FORCE_BUDGET: usize = 32;
const REFERENCE_STEP: f64 = 0.025;

struct Target {
    calls: AtomicUsize,
    hvp_calls: AtomicUsize,
}

impl Target {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            hvp_calls: AtomicUsize::new(0),
        }
    }
    fn precision(index: usize) -> f64 {
        (0.5 + 3.5 * index as f64 / (DIMENSION - 1) as f64).powi(2)
    }
}

impl LogDensityGradient for Target {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        DIMENSION
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut potential = 0.0;
        for i in 0..DIMENSION {
            let precision = Self::precision(i);
            gradient[i] = -precision * q[i];
            potential += 0.5 * precision * q[i] * q[i];
        }
        Ok(-potential)
    }
}

impl HessianVector for Target {
    fn potential_hvp(
        &self,
        _: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        self.hvp_calls.fetch_add(1, Ordering::Relaxed);
        for i in 0..DIMENSION {
            output[i] = Self::precision(i) * vector[i];
        }
        Ok(())
    }
}

impl BatchLogDensityGradient for Target {
    fn batch_logp_grad(
        &self,
        shape: BatchShape,
        positions: &[f64],
        gradients: &mut [f64],
        results: &mut [BatchLane<Self::Error>],
    ) {
        for (chain, result) in results.iter_mut().enumerate() {
            let mut potential = 0.0;
            for coordinate in 0..DIMENSION {
                let index = shape.index(chain, coordinate).unwrap();
                let precision = Self::precision(coordinate);
                gradients[index] = -precision * positions[index];
                potential += 0.5 * precision * positions[index] * positions[index];
            }
            *result = BatchLane::Complete(Ok(-potential));
        }
    }
}

struct MethodResult {
    draws: Vec<Vec<Vec<f64>>>,
    launch_seconds: f64,
    warmup_seconds: f64,
    sampling_seconds: f64,
    calls: usize,
    hvp_calls: usize,
    metric_updates: usize,
    coefficient_updates: usize,
    step_size: f64,
}

fn initial(chain: usize) -> OwnedBuffer {
    let sign = if chain.is_multiple_of(2) { -1.0 } else { 1.0 };
    OwnedBuffer::from_fn(DIMENSION, |index| sign * (0.2 + index as f64 * 0.01))
}

fn run_fixed<I: Integrator + Copy + std::fmt::Debug>(
    integrator: I,
    plan: ForceBudgetPlan,
) -> Result<MethodResult, Box<dyn std::error::Error>> {
    let mut draws = (0..CHAINS)
        .map(|_| (0..4).map(|_| Vec::with_capacity(DRAWS)).collect())
        .collect::<Vec<Vec<Vec<f64>>>>();
    let mut launch_seconds = 0.0;
    let mut warmup_seconds = 0.0;
    let mut sampling_seconds = 0.0;
    let mut calls = 0;
    let mut metric_updates = 0;
    let mut step_size = 0.0;
    for (chain_index, chain_draws) in draws.iter_mut().enumerate() {
        let target = Target::new();
        let mut rng = SmallRng::seed_from_u64(20_000 + chain_index as u64);
        let launch = Instant::now();
        let warmup = HmcWarmup::new_with_integrator(
            &target,
            initial(chain_index),
            HmcOptions::new(plan.macrostep_size, plan.macrosteps)?,
            WarmupOptions::new(WARMUP).with_target(0.9.try_into()?),
            integrator,
        )?;
        launch_seconds += launch.elapsed().as_secs_f64();
        let started = Instant::now();
        let (mut chain, report) = warmup.run(&mut rng)?;
        warmup_seconds += started.elapsed().as_secs_f64();
        metric_updates += report.metric_updates;
        step_size += report.step_size.value();
        let started = Instant::now();
        for _ in 0..DRAWS {
            let transition = chain.step(&mut rng)?;
            if transition.divergence.is_some() {
                return Err("unexpected benchmark divergence".into());
            }
            for (coordinate, series) in chain_draws.iter_mut().enumerate() {
                series.push(chain.point().position()[coordinate]);
            }
        }
        sampling_seconds += started.elapsed().as_secs_f64();
        calls += target.calls.load(Ordering::Relaxed);
    }
    Ok(MethodResult {
        draws,
        launch_seconds,
        warmup_seconds,
        sampling_seconds,
        calls,
        hvp_calls: 0,
        metric_updates,
        coefficient_updates: 0,
        step_size: step_size / CHAINS as f64,
    })
}

fn run_adaptive(plan: ForceBudgetPlan) -> Result<MethodResult, Box<dyn std::error::Error>> {
    let mut draws = (0..CHAINS)
        .map(|_| (0..4).map(|_| Vec::with_capacity(DRAWS)).collect())
        .collect::<Vec<Vec<Vec<f64>>>>();
    let mut launch_seconds = 0.0;
    let mut warmup_seconds = 0.0;
    let mut sampling_seconds = 0.0;
    let mut calls = 0;
    let mut hvp_calls = 0;
    let mut metric_updates = 0;
    let mut coefficient_updates = 0;
    let mut step_size = 0.0;
    for (chain_index, chain_draws) in draws.iter_mut().enumerate() {
        let target = Target::new();
        let mut rng = SmallRng::seed_from_u64(20_000 + chain_index as u64);
        let options = JointWarmupOptions::new(
            WarmupOptions::new(WARMUP).with_target(0.9.try_into()?),
            CurvatureProbeOptions::new(2, 4)?,
            1.25,
        )?;
        let launch = Instant::now();
        let warmup = JointHmcWarmup::new(
            &target,
            initial(chain_index),
            HmcOptions::new(plan.macrostep_size, plan.macrosteps)?,
            options,
        )?;
        launch_seconds += launch.elapsed().as_secs_f64();
        let started = Instant::now();
        let (mut chain, report) = warmup.run(&mut rng)?;
        warmup_seconds += started.elapsed().as_secs_f64();
        metric_updates += report.metric_updates;
        coefficient_updates += report.coefficient_updates;
        step_size += report.step_size.value();
        let started = Instant::now();
        for _ in 0..DRAWS {
            let transition = chain.step(&mut rng)?;
            if transition.divergence.is_some() {
                return Err("unexpected benchmark divergence".into());
            }
            for (coordinate, series) in chain_draws.iter_mut().enumerate() {
                series.push(chain.point().position()[coordinate]);
            }
        }
        sampling_seconds += started.elapsed().as_secs_f64();
        calls += target.calls.load(Ordering::Relaxed);
        hvp_calls += target.hvp_calls.load(Ordering::Relaxed);
    }
    Ok(MethodResult {
        draws,
        launch_seconds,
        warmup_seconds,
        sampling_seconds,
        calls,
        hvp_calls,
        metric_updates,
        coefficient_updates,
        step_size: step_size / CHAINS as f64,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("provide a new output CSV")?;
    let mut output = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);

    let source_shape = BatchShape::new(DIMENSION, CHAINS, BatchLayout::ChainMajor)?;
    let destination_shape = BatchShape::new(DIMENSION, CHAINS, BatchLayout::CoordinateMajor)?;
    let source = OwnedBuffer::new(source_shape.storage_len());
    let mut destination = OwnedBuffer::new(destination_shape.storage_len());
    let started = Instant::now();
    for _ in 0..1_000 {
        copy_batch_layout(source_shape, &source, destination_shape, &mut destination)?;
    }
    let layout_conversion_seconds = started.elapsed().as_secs_f64() / 1_000.0;

    let mut covariance = OnlineCovariance::new(DIMENSION, MetricKind::Dense)?;
    for observation in 0..WARMUP {
        let point = OwnedBuffer::from_fn(DIMENSION, |index| {
            ((observation + 1) as f64 * (index + 1) as f64).sin()
        });
        covariance.update(&point)?;
    }
    let started = Instant::now();
    let _metric = covariance.metric()?;
    let factorization_seconds = started.elapsed().as_secs_f64();

    let derivative_target = Target::new();
    let q = OwnedBuffer::new(DIMENSION);
    let v = OwnedBuffer::from_fn(DIMENSION, |_| 1.0 / (DIMENSION as f64).sqrt());
    let mut hvp = OwnedBuffer::new(DIMENSION);
    let started = Instant::now();
    for _ in 0..1_000 {
        evaluate_hvp(&derivative_target, &q, &v, &mut hvp)?;
    }
    let derivative_seconds = started.elapsed().as_secs_f64() / 1_000.0;

    let batch_target = Target::new();
    let started = Instant::now();
    let mut batch = alea_mcmc::batch::BatchedLeapfrog::new(
        &batch_target,
        CHAINS,
        BatchLayout::CoordinateMajor,
        OwnedBuffer::new(DIMENSION * CHAINS),
        OwnedBuffer::new(DIMENSION * CHAINS),
    )?;
    let _ = batch.advance(SignedStep::new(REFERENCE_STEP).unwrap(), 1)?;
    let batch_launch_seconds = started.elapsed().as_secs_f64();

    writeln!(
        output,
        "target,method,observable,force_budget,macrosteps,stages,mean,mcse,rhat,bulk_ess,tail_ess,step_size,launch_seconds,layout_conversion_seconds,batch_launch_seconds,warmup_seconds,derivative_seconds,factorization_seconds,sampling_seconds,total_seconds,target_calls,hvp_calls,metric_updates,coefficient_updates,ess_per_sampling_second,ess_per_total_second,ess_per_call"
    )?;
    for (name, stages, result) in [
        (
            "lf",
            1,
            run_fixed(
                Leapfrog,
                ForceBudgetPlan::new(FORCE_BUDGET, REFERENCE_STEP, &Leapfrog)?,
            )?,
        ),
        (
            "omf2",
            2,
            run_fixed(
                TwoStage::OMF,
                ForceBudgetPlan::new(FORCE_BUDGET, REFERENCE_STEP, &TwoStage::OMF)?,
            )?,
        ),
        (
            "bcss2",
            2,
            run_fixed(
                TwoStage::BCSS,
                ForceBudgetPlan::new(FORCE_BUDGET, REFERENCE_STEP, &TwoStage::BCSS)?,
            )?,
        ),
        (
            "adaptive2",
            2,
            run_adaptive(ForceBudgetPlan::new(
                FORCE_BUDGET,
                REFERENCE_STEP,
                &TwoStage::OMF,
            )?)?,
        ),
    ] {
        let macrosteps = FORCE_BUDGET / stages;
        let total_seconds = result.launch_seconds
            + layout_conversion_seconds
            + batch_launch_seconds
            + result.warmup_seconds
            + result.sampling_seconds;
        for observable in 0..4 {
            let views = result
                .draws
                .iter()
                .map(|chain| chain[observable].as_slice())
                .collect::<Vec<_>>();
            let summary = modern::summarize(&views)?;
            writeln!(
                output,
                "heterogeneous_gaussian,{name},{observable},{FORCE_BUDGET},{macrosteps},{stages},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.9},{layout_conversion_seconds:.9},{batch_launch_seconds:.9},{:.9},{derivative_seconds:.9},{factorization_seconds:.9},{:.9},{total_seconds:.9},{},{},{},{},{:.17e},{:.17e},{:.17e}",
                summary.mean,
                summary.mcse_mean,
                summary.rhat,
                summary.bulk_ess,
                summary.tail_ess,
                result.step_size,
                result.launch_seconds,
                result.warmup_seconds,
                result.sampling_seconds,
                result.calls,
                result.hvp_calls,
                result.metric_updates,
                result.coefficient_updates,
                summary.bulk_ess / result.sampling_seconds,
                summary.bulk_ess / total_seconds,
                summary.bulk_ess / result.calls as f64,
            )?;
        }
    }
    output.flush()?;
    Ok(())
}
