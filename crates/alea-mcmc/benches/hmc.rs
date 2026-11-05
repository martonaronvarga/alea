//! Cheap-gradient steady-state baseline: no construction or allocation in timing.
use alea_distributions::StandardNormalTarget;
use alea_distributions::wiener::WienerPrimitive;
use alea_distributions::wiener::{Boundary, WienerObservation};
use alea_math::buffer::OwnedBuffer;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use alea_core::model::{PrimitiveModel, TransformedTarget};
use alea_math::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
};

use alea_core::target::LogDensityGradient;
use alea_core::transform::{ParameterLayout, Transform};
use alea_mcmc::{Hmc, HmcOptions, Nuts, NutsOptions};
use rand::{SeedableRng, rngs::SmallRng};
use std::{hint::black_box, time::Duration};

#[allow(dead_code)]
#[path = "support/warmup.rs"]
mod warmup_workload;

fn workload<T: LogDensityGradient, M: EuclideanMetric>(
    c: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    target: &T,
    metric: M,
) {
    let mut chain = Hmc::new(
        target,
        OwnedBuffer::new(target.dimension()),
        metric,
        HmcOptions::new(0.02, 8).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(912);
    c.bench_function(name, |b| {
        b.iter(|| {
            let info = chain.step(&mut rng).unwrap();
            assert_eq!(info.integration_steps, 8);
            assert!(info.divergence.is_none());
            let _ = black_box(info);
        })
    });
}

fn workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_workloads");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    let gaussian = StandardNormalTarget::new(64);
    workload(
        &mut group,
        "gaussian64_identity",
        &gaussian,
        IdentityMetric::new(64),
    );
    workload(
        &mut group,
        "gaussian64_diagonal",
        &gaussian,
        DiagonalMetric::new(OwnedBuffer::from_fn(64, |i| 1.0 + (i % 3) as f64)).unwrap(),
    );
    let factor = OwnedBuffer::from_fn(64 * 64, |i| {
        let (row, col) = (i % 64, i / 64);
        if row == col {
            1.5
        } else if row == col + 1 {
            0.2
        } else {
            0.0
        }
    });
    workload(
        &mut group,
        "gaussian64_dense",
        &gaussian,
        DenseMetric::new(CholeskyFactor::new_lower(64, factor).unwrap()),
    );

    // Production public native GLM, never a hand-written benchmark likelihood.
    let data = (0..1024 * 16)
        .map(|i| ((i * 17 % 101) as f64 - 50.0) / 50.0)
        .collect::<Vec<_>>();
    let mut builder = alea_core::model::assembly::ModelBuilder::new();
    let beta: Vec<_> = (0..16)
        .map(|i| builder.real(&format!("b{i}")).unwrap().expr())
        .collect();
    for &b in &beta {
        let f = alea_distributions::modeling::normal(&mut builder, b, 0., 1.).unwrap();
        builder.factor(f).unwrap();
    }
    let data = alea_distributions::modeling::regression::RegressionData::new(
        16,
        data,
        (0..1024).map(|i| i % 2 != 0).collect(),
    )
    .unwrap();
    let f = alea_distributions::modeling::regression::bernoulli_logit_regression(
        &mut builder,
        &beta,
        data,
    )
    .unwrap();
    builder.likelihood("data", f).unwrap();
    let logistic = builder.compile().unwrap().into_target().unwrap();
    workload(
        &mut group,
        "logistic1024x16",
        &logistic,
        IdentityMetric::new(16),
    );

    let wiener = PrimitiveModel(
        WienerPrimitive::new(
            WienerObservation {
                rt: 0.8,
                boundary: Boundary::Upper,
            },
            1e-12,
        )
        .unwrap(),
    );
    // Bound otherwise improper single-observation tails so this benchmark is a
    // fixed valid target throughout Criterion's unbounded number of iterations.
    let layout = ParameterLayout::new([
        Transform::interval(1, 0.5, 2.0).unwrap(),
        Transform::interval(1, 0.0, 0.4).unwrap(),
        Transform::interval(1, 0.1, 0.9).unwrap(),
        Transform::interval(1, -3.0, 3.0).unwrap(),
    ])
    .unwrap();
    let wiener = TransformedTarget::new(wiener, layout).unwrap();
    workload(
        &mut group,
        "wiener4_transformed",
        &wiener,
        IdentityMetric::new(4),
    );
    group.finish();
}

fn hmc(c: &mut Criterion) {
    let mut group = c.benchmark_group("hmc_protocol");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    for dim in [8, 64, 1024] {
        let target = StandardNormalTarget::new(dim);
        let mut chain = Hmc::new(
            &target,
            OwnedBuffer::new(dim),
            IdentityMetric::new(dim),
            HmcOptions::new(0.1, 8).unwrap(),
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(912);
        group.bench_with_input(BenchmarkId::new("hmc", dim), &dim, |b, _| {
            b.iter(|| {
                let _ = black_box(chain.step(&mut rng).unwrap());
                black_box((
                    chain.point().position(),
                    chain.point().gradient(),
                    chain.point().log_density(),
                ));
            });
        });
    }
    group.finish();
}

fn nuts(c: &mut Criterion) {
    use criterion::{BatchSize, Throughput};
    for dim in [4, 64] {
        let target = StandardNormalTarget::new(dim);
        let options = NutsOptions::new(0.2, 6).unwrap();
        // Separate instrumented admission pass; no counters/clocks in timed target.
        let counted = warmup_workload::Counted {
            target: StandardNormalTarget::new(dim),
            calls: std::cell::Cell::new(0),
        };
        let mut check = Nuts::new(
            &counted,
            OwnedBuffer::from_fn(dim, |_| 0.1),
            IdentityMetric::new(dim),
            options,
        )
        .unwrap();
        let initialize_calls = counted.calls.get();
        let receipt = check.step(&mut SmallRng::seed_from_u64(912)).unwrap();
        let calls = counted.calls.get() - initialize_calls;
        assert_eq!(calls, receipt.integration_steps);
        eprintln!(
            "ALEA_NUTS_FIXED_WORK {{\"dimension\":{dim},\"seed\":912,\"initialize_calls\":{initialize_calls},\"gradient_calls\":{calls},\"leapfrog_steps\":{},\"tree_depth\":{}}}",
            receipt.integration_steps, receipt.tree_depth
        );
        let create = || {
            Nuts::new(
                &target,
                OwnedBuffer::from_fn(dim, |_| 0.1),
                IdentityMetric::new(dim),
                options,
            )
            .unwrap()
        };
        let mut group = c.benchmark_group(format!("nuts/{dim}"));
        group.bench_function("initialize_and_drop", |b| b.iter(|| black_box(create())));
        group.throughput(Throughput::Elements(1)); // Transition, NOT leapfrog/gradient.
        group.bench_function("fixed_initial_state_transition", |b| {
            b.iter_batched_ref(
                || (create(), SmallRng::seed_from_u64(912)),
                |(chain, rng)| black_box(chain.step(rng).unwrap()),
                BatchSize::SmallInput,
            )
        });
        let mut chain = create();
        let mut rng = SmallRng::seed_from_u64(912);
        group.bench_function("evolving_transitions", |b| {
            b.iter(|| black_box(chain.step(&mut rng).unwrap()))
        });
        group.finish();
    }
}
// Shared native chain harness: seed and construction outside fixed-transition
// timing, with no new target formulas or sampler implementations.
fn engine<C: alea_mcmc::MarkovChain>(
    c: &mut Criterion,
    name: &str,
    make: impl Fn(&mut SmallRng) -> C,
) where
    C::Transition: std::fmt::Debug,
{
    use criterion::BatchSize;
    let prepare = || {
        let mut rng = SmallRng::seed_from_u64(912);
        let chain = make(&mut rng);
        (chain, rng)
    };
    let (mut check, mut rng) = prepare();
    let receipt = check.step(&mut rng).unwrap();
    assert!(check.position().iter().all(|x| x.is_finite()));
    eprintln!("ALEA_ENGINE_FIXED_WORK {name}: {receipt:?}");
    let mut group = c.benchmark_group(format!("experimental_engines/{name}"));
    group.bench_function("initialize_and_drop", |b| b.iter(|| black_box(prepare())));
    group.bench_function("fixed_initial_state_transition", |b| {
        b.iter_batched_ref(
            prepare,
            |(chain, rng)| black_box(chain.step(rng).unwrap()),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn experimental_engines(c: &mut Criterion) {
    use alea_mcmc::{
        adapt::{CheesCriterion, CheesEnsembleAdapter},
        advanced::{AdvancedHmcOptions, ForceGradientHmc, ForceGradientMode},
        atlas_sampler::{Atlas, AtlasOptions},
        chees::{CheesHmc, CheesHmcOptions},
        ghmc::{DrGhmc, DrGhmcOptions, Ghmc, GhmcOptions},
        mams::{ApproximateMclmc, ApproximateMclmcOptions, ApproximateTarget, Mams, MamsOptions},
        walnuts::{Walnuts, WalnutsOptions},
    };
    for dim in [4, 64] {
        let target = StandardNormalTarget::new(dim);
        let q = || OwnedBuffer::from_fn(dim, |_| 0.1);
        let metric = || IdentityMetric::new(dim);
        engine(c, &format!("ghmc/{dim}"), |rng| {
            Ghmc::new(
                &target,
                q(),
                metric(),
                GhmcOptions::new(0.1, 4, 0.8).unwrap(),
                rng,
            )
            .unwrap()
        });
        engine(c, &format!("drghmc/{dim}"), |rng| {
            DrGhmc::new(
                &target,
                q(),
                metric(),
                DrGhmcOptions::new(0.2, 2., 0.8)
                    .unwrap()
                    .with_integration_steps(2)
                    .unwrap(),
                rng,
            )
            .unwrap()
        });
        engine(c, &format!("mams/{dim}"), |_| {
            Mams::new(&target, q(), MamsOptions::new(0.1, 4).unwrap()).unwrap()
        });
        // This method is explicitly approximate. A timing is not posterior or
        // exactness qualification, nor an efficiency comparison with NUTS.
        engine(c, &format!("approximate_mclmc/{dim}"), |rng| {
            ApproximateMclmc::new(
                ApproximateTarget::acknowledge_bias(),
                &target,
                q(),
                ApproximateMclmcOptions::new(0.1, 4, Some(1.)).unwrap(),
                rng,
            )
            .unwrap()
        });
        engine(c, &format!("walnuts/{dim}"), |_| {
            Walnuts::new(&target, q(), metric(), WalnutsOptions::new(0.1, 4).unwrap()).unwrap()
        });
        engine(c, &format!("atlas/{dim}"), |_| {
            Atlas::from_hmc(
                Hmc::new(&target, q(), metric(), HmcOptions::new(0.1, 4).unwrap()).unwrap(),
                AtlasOptions::new(0.1, 64).unwrap(),
                &[4, 6, 8],
            )
            .unwrap()
        });
        let frozen = CheesEnsembleAdapter::new(
            dim,
            0.1.try_into().unwrap(),
            0.8,
            0.1,
            2.,
            CheesCriterion::Global,
        )
        .unwrap()
        .finish()
        .unwrap();
        engine(c, &format!("chees/{dim}"), |_| {
            CheesHmc::new(&target, q(), metric(), CheesHmcOptions::new(frozen)).unwrap()
        });
        for mode in [
            ForceGradientMode::AnalyticHvp,
            ForceGradientMode::HessianFree,
        ] {
            engine(c, &format!("force_gradient_{mode:?}/{dim}"), |_| {
                ForceGradientHmc::new(
                    &target,
                    q(),
                    metric(),
                    AdvancedHmcOptions::new(0.1, 4).unwrap(),
                    mode,
                )
                .unwrap()
            });
        }
    }
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=hmc, workloads, nuts, experimental_engines}
#[path = "../../benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
