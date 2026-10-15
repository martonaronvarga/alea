use std::convert::Infallible;

use alea_core::{
    capability::{
        BatchLane, BatchLayout, BatchLogDensityGradient, BatchShape, GaussianResidual,
        HessianVector, SplitLogDensityGradient, copy_batch_layout,
    },
    target::LogDensityGradient,
};
use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    AdvancedHmcOptions, BatchedLeapfrog, ForceGradientHmc, ForceGradientMode, GaussianSplitHmc,
    SplitForceHmc,
    adapt::{JointHmcWarmup, JointWarmupOptions, WarmupOptions},
    atlas::AtlasProbeOptions,
    hamiltonian::SignedStep,
    integrator::{
        CurvatureBound, ForceBudgetPlan, Leapfrog, TwoStage, select_saia_two_stage,
        two_stage_energy_bound,
    },
};
use rand::{SeedableRng, rngs::SmallRng};

#[derive(Debug)]
struct Quadratic {
    precision: [f64; 2],
    center: [f64; 2],
}

impl Quadratic {
    const fn new() -> Self {
        Self {
            precision: [1.0, 4.0],
            center: [0.0, 0.0],
        }
    }
}

impl LogDensityGradient for Quadratic {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        let mut potential = 0.0;
        for i in 0..2 {
            let delta = q[i] - self.center[i];
            gradient[i] = -self.precision[i] * delta;
            potential += 0.5 * self.precision[i] * delta * delta;
        }
        Ok(-potential)
    }
}

impl HessianVector for Quadratic {
    fn potential_hvp(
        &self,
        _: &[f64],
        vector: &[f64],
        output: &mut [f64],
    ) -> Result<(), Self::Error> {
        for i in 0..2 {
            output[i] = self.precision[i] * vector[i];
        }
        Ok(())
    }
}

impl SplitLogDensityGradient for Quadratic {
    fn cheap_score(&self, q: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        for i in 0..2 {
            output[i] = -0.75 * self.precision[i] * q[i];
        }
        Ok(())
    }
    fn expensive_score(&self, q: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        for i in 0..2 {
            output[i] = -0.25 * self.precision[i] * q[i];
        }
        Ok(())
    }
}

impl GaussianResidual for Quadratic {
    fn gaussian_center(&self) -> &[f64] {
        &self.center
    }
    fn gaussian_precision(&self) -> &[f64] {
        &self.precision
    }
    fn residual_score(&self, _: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        output.fill(0.0);
        Ok(())
    }
}

impl BatchLogDensityGradient for Quadratic {
    fn batch_logp_grad(
        &self,
        shape: BatchShape,
        positions: &[f64],
        gradients: &mut [f64],
        results: &mut [BatchLane<Self::Error>],
    ) {
        for (chain, result) in results.iter_mut().enumerate() {
            let mut potential = 0.0;
            for coordinate in 0..2 {
                let index = shape.index(chain, coordinate).unwrap();
                let q = positions[index];
                gradients[index] = -self.precision[coordinate] * q;
                potential += 0.5 * self.precision[coordinate] * q * q;
            }
            *result = BatchLane::Complete(Ok(-potential));
        }
    }
}

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |index| values[index])
}

#[test]
fn saia_bound_selection_and_equal_force_budget_follow_published_scaling() {
    let small = select_saia_two_stage(0.05).unwrap();
    assert!((small.integrator.lambda() - TwoStage::OMF.lambda()).abs() < 2e-4);
    let middle = select_saia_two_stage(2.0).unwrap();
    assert!((middle.integrator.lambda() - TwoStage::BCSS.lambda()).abs() < 2e-3);
    for fixed in [TwoStage::OMF, TwoStage::BCSS, TwoStage::new(0.25).unwrap()] {
        assert!(
            middle.worst_bound
                <= two_stage_energy_bound(middle.dimensionless_limit, fixed.lambda()) * 1.001
        );
    }
    let bound = CurvatureBound::new(4.0, 1.5).unwrap();
    assert_eq!(bound.upper(), 6.0);
    assert!((bound.dimensionless_step(0.2).unwrap() - 0.2 * 6.0_f64.sqrt()).abs() < 1e-15);
    let leapfrog = ForceBudgetPlan::new(64, 0.05, &Leapfrog).unwrap();
    let two_stage = ForceBudgetPlan::new(64, 0.05, &TwoStage::OMF).unwrap();
    assert_eq!((leapfrog.macrosteps, leapfrog.macrostep_size), (64, 0.05));
    assert_eq!((two_stage.macrosteps, two_stage.macrostep_size), (32, 0.1));
}

#[test]
fn exact_gaussian_flow_has_oracle_energy_and_stationary_moments() {
    let target = Quadratic::new();
    let mut chain = GaussianSplitHmc::new(
        &target,
        buffer(&[0.7, -0.4]),
        AdvancedHmcOptions::new(0.37, 3).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(10_001);
    let mut sum = [0.0; 2];
    let mut square = [0.0; 2];
    for iteration in 0..6_000 {
        let transition = chain.step(&mut rng).unwrap();
        assert!(transition.divergence.is_none());
        assert_eq!(transition.evaluations.residual, 4);
        assert_eq!(transition.evaluations.full, 1);
        assert!(transition.energy_error.unwrap().abs() < 2e-12);
        assert!(transition.acceptance_probability > 1.0 - 3e-12);
        if iteration >= 1_000 {
            for i in 0..2 {
                let q = chain.point().position()[i];
                sum[i] += q;
                square[i] += q * q;
            }
        }
    }
    for i in 0..2 {
        let count = 5_000.0;
        assert!((sum[i] / count).abs() < 0.06);
        assert!((square[i] / count - 1.0 / target.precision[i]).abs() < 0.07);
    }
}

#[test]
fn split_force_and_force_gradient_kernels_are_costed_and_stationary() {
    let target = Quadratic::new();
    let options = AdvancedHmcOptions::new(0.12, 5).unwrap();
    let mut split = SplitForceHmc::new(
        &target,
        buffer(&[0.2, -0.1]),
        IdentityMetric::new(2),
        options,
        3,
    )
    .unwrap();
    let mut analytic = ForceGradientHmc::new(
        &target,
        buffer(&[0.2, -0.1]),
        IdentityMetric::new(2),
        options,
        ForceGradientMode::AnalyticHvp,
    )
    .unwrap();
    let mut hessian_free = ForceGradientHmc::new(
        &target,
        buffer(&[0.2, -0.1]),
        IdentityMetric::new(2),
        options,
        ForceGradientMode::HessianFree,
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(10_002);
    let first = split.step(&mut rng).unwrap();
    assert_eq!(first.evaluations.cheap, 20);
    assert_eq!(first.evaluations.expensive, 6);
    assert_eq!(first.evaluations.full, 1);
    let exact = analytic.step(&mut rng).unwrap();
    assert_eq!(exact.evaluations.full, 10);
    assert_eq!(exact.evaluations.hvp, 5);
    let approximate = hessian_free.step(&mut rng).unwrap();
    assert_eq!(approximate.evaluations.full, 15);
    assert_eq!(approximate.evaluations.hvp, 0);

    let mut means = [[0.0; 2]; 3];
    let mut squares = [[0.0; 2]; 3];
    for iteration in 0..5_000 {
        let _ = split.step(&mut rng).unwrap();
        let _ = analytic.step(&mut rng).unwrap();
        let _ = hessian_free.step(&mut rng).unwrap();
        if iteration >= 500 {
            for (method, position) in [
                split.point().position(),
                analytic.point().position(),
                hessian_free.point().position(),
            ]
            .iter()
            .enumerate()
            {
                for i in 0..2 {
                    means[method][i] += position[i];
                    squares[method][i] += position[i] * position[i];
                }
            }
        }
    }
    for method in 0..3 {
        for i in 0..2 {
            assert!((means[method][i] / 4_500.0).abs() < 0.09);
            assert!((squares[method][i] / 4_500.0 - 1.0 / target.precision[i]).abs() < 0.1);
        }
    }
}

#[test]
fn force_gradient_energy_error_has_fourth_order_fixed_time_scaling() {
    let target = Quadratic::new();
    for mode in [
        ForceGradientMode::AnalyticHvp,
        ForceGradientMode::HessianFree,
    ] {
        let run = |step: f64, count: usize| {
            let mut chain = ForceGradientHmc::new(
                &target,
                buffer(&[0.7, -0.4]),
                IdentityMetric::new(2),
                AdvancedHmcOptions::new(step, count).unwrap(),
                mode,
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(10_004);
            chain.step(&mut rng).unwrap().energy_error.unwrap().abs()
        };
        let coarse = run(0.2, 5);
        let fine = run(0.1, 10);
        assert!(
            fine < coarse / 12.0,
            "{mode:?}: coarse={coarse}, fine={fine}"
        );
    }
}

#[test]
fn explicit_layout_adapter_and_batched_fsm_match_and_reverse() {
    let target = Quadratic::new();
    let chain_shape = BatchShape::new(2, 3, BatchLayout::ChainMajor).unwrap();
    let coordinate_shape = BatchShape::new(2, 3, BatchLayout::CoordinateMajor).unwrap();
    let q_chain = buffer(&[0.1, -0.4, 0.2, 0.5, -0.7, 0.3]);
    let p_chain = buffer(&[0.3, 0.8, -0.2, 0.4, 0.6, -0.5]);
    let mut q_coordinate = OwnedBuffer::new(6);
    let mut p_coordinate = OwnedBuffer::new(6);
    copy_batch_layout(chain_shape, &q_chain, coordinate_shape, &mut q_coordinate).unwrap();
    copy_batch_layout(chain_shape, &p_chain, coordinate_shape, &mut p_coordinate).unwrap();
    let mut chain_major = BatchedLeapfrog::new(
        &target,
        3,
        BatchLayout::ChainMajor,
        q_chain.clone(),
        p_chain.clone(),
    )
    .unwrap();
    let mut coordinate_major = BatchedLeapfrog::new(
        &target,
        3,
        BatchLayout::CoordinateMajor,
        q_coordinate,
        p_coordinate,
    )
    .unwrap();
    let forward = SignedStep::new(0.07).unwrap();
    let reverse = SignedStep::new(-0.07).unwrap();
    let report = chain_major.advance(forward, 11).unwrap();
    coordinate_major.advance(forward, 11).unwrap();
    assert_eq!(report.batch_launches, 11);
    assert_eq!(report.scalar_equivalent_evaluations, 33);
    let mut converted = OwnedBuffer::new(6);
    copy_batch_layout(
        coordinate_shape,
        coordinate_major.positions(),
        chain_shape,
        &mut converted,
    )
    .unwrap();
    for (a, b) in chain_major.positions().iter().zip(converted.iter()) {
        assert!((a - b).abs() < 1e-14);
    }
    chain_major.advance(reverse, 11).unwrap();
    for (actual, expected) in chain_major.positions().iter().zip(q_chain.iter()) {
        assert!((actual - expected).abs() < 2e-13);
    }
    for (actual, expected) in chain_major.momenta().iter().zip(p_chain.iter()) {
        assert!((actual - expected).abs() < 2e-13);
    }
}

#[test]
fn joint_warmup_changes_only_at_epochs_and_returns_frozen_chain() {
    let target = Gaussian::new(3);
    let options = JointWarmupOptions::new(
        WarmupOptions::new(80),
        AtlasProbeOptions::new(2, 3, 0.8, 1e-4, 2.0).unwrap(),
        1.25,
    )
    .unwrap();
    let warmup = JointHmcWarmup::new(
        &target,
        OwnedBuffer::new(3),
        alea_mcmc::HmcOptions::new(0.1, 4).unwrap(),
        options,
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(10_003);
    let (mut chain, report) = warmup.run(&mut rng).unwrap();
    assert_eq!(report.iterations, 80);
    assert!(report.metric_updates >= 1);
    assert_eq!(report.coefficient_updates, report.metric_updates + 2);
    assert!(report.hvp_calls > 0);
    assert!((TwoStage::OMF.lambda()..=0.25).contains(&report.two_stage_coefficient));
    assert_eq!(chain.options().step_size(), report.step_size);
    for _ in 0..20 {
        assert!(chain.step(&mut rng).unwrap().divergence.is_none());
    }
}
